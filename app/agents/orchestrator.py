"""
Root orchestrator. Routing happens via ADK's AgentTool — the LLM picks which
specialist to invoke as a function call. We do NOT string-parse routing
decisions (see assignment penalty list).

State persistence: Pattern 3 from `docs/google-adk-guide.md`. SessionState is
loaded from the DB by the pipeline and rendered into the system instruction
at runtime via `build_root_agent(state)`. Nothing about the agent objects is
mutated between turns; we just reconstruct the root with a fresh prompt.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.account import account_agent
from app.agents.knowledge import knowledge_agent
from app.settings import settings
from app.srop.state import SessionState

ROOT_INSTRUCTION_TEMPLATE = """\
You are the Helix Support Concierge — a routing agent.

Decide which specialist handles the user's message and call it as a tool.

Routing rules:
  • How-to / what-is / docs / feature questions   →  call `knowledge_agent`
  • The user's account, builds, status, usage     →  call `account_agent`
  • Greetings, thanks, off-topic                  →  reply directly. No tool call.

Hard rules:
  • Never answer knowledge or account questions yourself — always defer.
  • Never ask the user for `user_id` or `plan_tier`. They are below.
  • For follow-up turns, prefer the `last_agent` again unless the topic
    clearly switches.

Current user context:
  user_id:        {user_id}
  plan_tier:      {plan_tier}
  turn_count:     {turn_count}
  last_agent:     {last_agent}
  open_tickets:   {open_tickets}
"""


def _render_instruction(state: SessionState) -> str:
    return ROOT_INSTRUCTION_TEMPLATE.format(
        user_id=state.user_id,
        plan_tier=state.plan_tier,
        turn_count=state.turn_count,
        last_agent=state.last_agent or "none",
        open_tickets=", ".join(state.open_ticket_ids) if state.open_ticket_ids else "none",
    )


def build_root_agent(state: SessionState) -> LlmAgent:
    """Build the root orchestrator with state injected as system context."""
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        description="Helix Support Concierge — routes user turns to specialist sub-agents.",
        instruction=_render_instruction(state),
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
        ],
    )
