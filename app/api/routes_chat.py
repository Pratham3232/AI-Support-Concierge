"""POST /v1/chat/{session_id} — run one turn of the SROP pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.srop import pipeline

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    content: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    reply: str
    routed_to: str  # knowledge | account | escalation | smalltalk
    trace_id: str


@router.post("/chat/{session_id}", response_model=ChatResponse)
async def chat(
    session_id: str,
    body: ChatRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Run one turn. Errors surface as RFC 7807 problem-details:
        404 SESSION_NOT_FOUND
        504 UPSTREAM_TIMEOUT
    """
    result = await pipeline.run(session_id, body.content, db, idempotency_key)
    return ChatResponse(
        reply=result.content, routed_to=result.routed_to, trace_id=result.trace_id
    )
