"""
SROP pipeline — runs once per chat turn.

Order of operations (matters):

  1. Load SessionState from DB.            ← survives uvicorn restart
  2. Persist the user message.             ← idempotency keys live here
  3. Run the ADK orchestrator with state injected as system context.
  4. Capture routing + tool calls + retrieved chunk IDs from ADK events.
  5. Write `agent_traces` row.
  6. Update SessionState (turn_count, last_agent, last_chunk_ids) and persist.
  7. Persist the assistant message linked to the trace.

If step 3 raises (timeout, etc.) we propagate — no partial state is committed.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import runner as agent_runner
from app.api.errors import SessionNotFoundError
from app.db.models import AgentTrace, Message
from app.srop import repo

log = structlog.get_logger()


@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str


async def run(
    session_id: str,
    user_message: str,
    db: AsyncSession,
    idempotency_key: str | None = None,
) -> PipelineResult:
    state = await repo.load_state(db, session_id)
    if state is None:
        raise SessionNotFoundError(f"Session {session_id} does not exist")

    structlog.contextvars.bind_contextvars(
        session_id=session_id, user_id=state.user_id, turn=state.turn_count + 1
    )

    if idempotency_key:
        cached = await repo.find_replay(db, session_id, idempotency_key)
        if cached is not None:
            log.info("idempotency_hit", trace_id=cached.trace_id)
            return PipelineResult(
                content=cached.content,
                routed_to=cached.routed_to or "smalltalk",
                trace_id=cached.trace_id or "",
            )

    trace_id = str(uuid.uuid4())
    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
    ))

    started = time.perf_counter()
    log.info("pipeline_started", message_chars=len(user_message))
    result = await agent_runner.run_root_agent(user_message, state)
    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "pipeline_completed",
        routed_to=result.routed_to,
        latency_ms=latency_ms,
        tool_calls=len(result.tool_calls),
        chunks=len(result.retrieved_chunk_ids),
    )

    db.add(AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=result.routed_to,
        tool_calls=result.tool_calls,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        latency_ms=latency_ms,
    ))

    state.turn_count += 1
    state.last_agent = result.routed_to if result.routed_to in {
        "knowledge", "account", "escalation", "smalltalk"
    } else state.last_agent
    state.last_chunk_ids = result.retrieved_chunk_ids
    await repo.save_state(db, session_id, state)

    db.add(Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=result.text,
        routed_to=result.routed_to,
        trace_id=trace_id,
        idempotency_key=idempotency_key,
    ))

    try:
        await db.commit()
    except IntegrityError:
        # Race: a concurrent request committed the same idempotency key first.
        await db.rollback()
        cached = await repo.find_replay(db, session_id, idempotency_key or "")
        if cached is not None:
            return PipelineResult(
                content=cached.content,
                routed_to=cached.routed_to or "smalltalk",
                trace_id=cached.trace_id or "",
            )
        raise

    return PipelineResult(
        content=result.text, routed_to=result.routed_to, trace_id=trace_id
    )
