"""DB access helpers used by routes and the pipeline. All async."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentTrace, Message, User
from app.db.models import Session as SessionModel
from app.srop.state import SessionState


async def get_or_create_user(db: AsyncSession, user_id: str, plan_tier: str) -> User:
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        user = User(user_id=user_id, plan_tier=plan_tier)
        db.add(user)
    elif plan_tier and user.plan_tier != plan_tier:
        user.plan_tier = plan_tier
    return user


async def get_session(db: AsyncSession, session_id: str) -> SessionModel | None:
    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == session_id)
    )
    return result.scalar_one_or_none()


async def load_state(db: AsyncSession, session_id: str) -> SessionState | None:
    sess = await get_session(db, session_id)
    if sess is None:
        return None
    return SessionState.from_db_dict(sess.state or {})


async def save_state(db: AsyncSession, session_id: str, state: SessionState) -> None:
    sess = await get_session(db, session_id)
    if sess is None:
        return
    sess.state = state.to_db_dict()


async def find_replay(
    db: AsyncSession, session_id: str, idempotency_key: str
) -> Message | None:
    result = await db.execute(
        select(Message).where(
            Message.session_id == session_id,
            Message.idempotency_key == idempotency_key,
            Message.role == "assistant",
        )
    )
    return result.scalar_one_or_none()


async def get_trace(db: AsyncSession, trace_id: str) -> AgentTrace | None:
    result = await db.execute(select(AgentTrace).where(AgentTrace.trace_id == trace_id))
    return result.scalar_one_or_none()
