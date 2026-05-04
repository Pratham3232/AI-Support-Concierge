"""
ADK runner abstraction.

This is the test seam — `run_root_agent` is what `app.srop.pipeline` calls,
and it's what tests monkeypatch to avoid real LLM calls. The mock should
behave like the real ADK boundary: take a user message + state, return an
`AgentRunResult` with the final text, the routing decision, observed tool
calls, and any chunk IDs surfaced by `search_docs`.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog
from google.adk.runners import InMemoryRunner
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.agents.orchestrator import build_root_agent
from app.agents.tools.search_docs import get_last_hits, reset_last_hits
from app.api.errors import UpstreamTimeoutError
from app.settings import settings
from app.srop.state import SessionState

log = structlog.get_logger()


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()

APP_NAME = "helix_srop"


@dataclass
class AgentRunResult:
    text: str
    routed_to: str  # knowledge | account | smalltalk | escalation | root
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)


_AGENT_NAME_TO_ROUTE = {
    "knowledge_agent": "knowledge",
    "account_agent": "account",
    "escalation_agent": "escalation",
    "srop_root": "smalltalk",
}


def _classify_route(events_authors: list[str], tool_calls: list[dict[str, Any]]) -> str:
    """Map the observed event stream to a single 'routed_to' label."""
    # The first sub-agent invoked tells us where the turn was routed.
    for tc in tool_calls:
        name = tc.get("tool_name", "")
        if name in _AGENT_NAME_TO_ROUTE:
            return _AGENT_NAME_TO_ROUTE[name]
    # Fallback: if only the root produced a final response, this was smalltalk.
    for author in reversed(events_authors):
        if author in _AGENT_NAME_TO_ROUTE:
            return _AGENT_NAME_TO_ROUTE[author]
    return "smalltalk"


def _final_text_from_event(event: Any) -> str:
    content = getattr(event, "content", None)
    if not content:
        return ""
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts).strip()


@retry(
    retry=retry_if_exception(_is_rate_limit),
    wait=wait_exponential(multiplier=1, min=10, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _run_adk(user_message: str, state: SessionState) -> AgentRunResult:
    reset_last_hits()
    agent = build_root_agent(state)
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)

    session = await runner.session_service.create_session(
        app_name=APP_NAME, user_id=state.user_id
    )
    new_message = types.Content(role="user", parts=[types.Part(text=user_message)])

    tool_calls: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    authors: list[str] = []
    final_text = ""

    try:
        async for event in runner.run_async(
            user_id=state.user_id,
            session_id=session.id,
            new_message=new_message,
        ):
            authors.append(getattr(event, "author", "") or "")

            for fc in event.get_function_calls() or []:
                call_id = getattr(fc, "id", None) or f"{fc.name}:{len(tool_calls)}"
                rec = {"tool_name": fc.name, "args": dict(fc.args or {}), "result": None}
                tool_calls.append(rec)
                pending_calls[call_id] = rec

            for fr in event.get_function_responses() or []:
                call_id = getattr(fr, "id", None) or ""
                rec = pending_calls.get(call_id)
                if rec is None and tool_calls:
                    # Best-effort match by name when ADK doesn't echo the id.
                    for candidate in reversed(tool_calls):
                        if candidate["tool_name"] == fr.name and candidate["result"] is None:
                            rec = candidate
                            break
                if rec is not None:
                    rec["result"] = fr.response

            if event.is_final_response():
                final_text = _final_text_from_event(event) or final_text

    finally:
        await runner.close()

    routed_to = _classify_route(authors, tool_calls)
    chunk_ids = [h["chunk_id"] for h in get_last_hits()]

    return AgentRunResult(
        text=final_text,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=chunk_ids,
    )


async def run_root_agent(user_message: str, state: SessionState) -> AgentRunResult:
    """Run one turn through the ADK orchestrator with a hard timeout.

    Tests patch THIS function to mock the LLM at the ADK boundary.
    """
    from app.api.errors import RateLimitedError

    try:
        return await asyncio.wait_for(
            _run_adk(user_message, state), timeout=settings.llm_timeout_seconds
        )
    except TimeoutError as exc:
        raise UpstreamTimeoutError(
            f"LLM did not respond within {settings.llm_timeout_seconds}s"
        ) from exc
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            raise RateLimitedError(
                "Gemini API quota exceeded. Enable billing or wait for quota reset."
            ) from exc
        raise
