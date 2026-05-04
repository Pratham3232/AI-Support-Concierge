"""AccountAgent — handles per-user account / build queries."""
from google.adk.agents import LlmAgent

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.settings import settings

ACCOUNT_INSTRUCTION = """\
You are the Helix account agent. Use the provided tools to look up account
data. Never ask the user for their `user_id` or `plan_tier` — both are
provided in the system context above. Pass them through to tools verbatim.

Tool selection:
  - Builds, pipeline runs, failures, recent activity → `get_recent_builds`
  - Plan tier, usage limits, storage, concurrency → `get_account_status`

Format build lists as a short bulleted summary. Surface failures explicitly.
"""

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    description=(
        "Answers questions about the user's Helix account: builds, usage, plan tier."
    ),
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)
