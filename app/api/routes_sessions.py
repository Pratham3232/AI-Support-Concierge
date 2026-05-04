"""POST /v1/sessions — create a session."""
from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Session as SessionModel
from app.db.session import get_db
from app.srop import repo
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str
    plan_tier: str


@router.post("/sessions", response_model=CreateSessionResponse, status_code=200)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    await repo.get_or_create_user(db, body.user_id, body.plan_tier)

    session_id = str(uuid.uuid4())
    initial_state = SessionState(user_id=body.user_id, plan_tier=body.plan_tier)
    db.add(SessionModel(
        session_id=session_id,
        user_id=body.user_id,
        state=initial_state.to_db_dict(),
    ))
    await db.commit()

    return CreateSessionResponse(
        session_id=session_id, user_id=body.user_id, plan_tier=body.plan_tier
    )
