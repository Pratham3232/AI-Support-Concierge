"""
Session state — persisted as JSON in sessions.state.

Pattern 3 from google-adk-guide.md: we keep this small (only fields that the
agent cannot re-derive from message history) and inject it into the root
agent's instruction every turn.
"""
from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal["knowledge", "account", "escalation", "smalltalk", "root"]


class SessionState(BaseModel):
    user_id: str
    plan_tier: Literal["free", "pro", "enterprise"] = "free"
    last_agent: AgentName | None = None
    turn_count: int = 0
    open_ticket_ids: list[str] = Field(default_factory=list)
    last_chunk_ids: list[str] = Field(default_factory=list)

    def to_db_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_db_dict(cls, data: dict) -> "SessionState":
        return cls.model_validate(data or {})
