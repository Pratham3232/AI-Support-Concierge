"""
Shared test fixtures.

Tests use an in-memory SQLite engine (one engine per test module run) and
a Chroma store under a tmp dir so they never touch the dev DB. The
`mock_adk` fixture patches the ADK boundary in `app.agents.runner` — tests
do not hit a real LLM.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Force chroma to a tmpdir BEFORE any app code imports settings.
_CHROMA_TMP = Path(__file__).parent / "_chroma_test"
os.environ.setdefault("CHROMA_PERSIST_DIR", str(_CHROMA_TMP))
os.environ.setdefault("CHROMA_COLLECTION", "helix_docs_test")

from app.agents import runner as agent_runner  # noqa: E402
from app.agents.runner import AgentRunResult  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.rag import store as vstore  # noqa: E402

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db():
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client():
    """Async test client. Each request gets its own DB session bound to the
    same in-memory engine so transactions across calls share state."""

    async def _override_get_db():
        async with TestSessionLocal() as s:
            yield s

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    """
    Replace `app.agents.runner.run_root_agent` with a deterministic stub that
    behaves like the ADK boundary:

      • "rotate"/"deploy key"/"how"/"what is"  → routes to knowledge
      • "build"/"plan"/"account"/"usage"       → routes to account
      • everything else                        → smalltalk

    The knowledge route surfaces fake chunk IDs so the trace assertions in
    `test_api.py` see retrieved_chunk_ids > 0. The account route reads the
    injected SessionState so we can verify state persistence end-to-end.
    """
    from app.srop.state import SessionState

    knowledge_tokens = ("rotate", "deploy key", "how do", "how to", "what is", "docs")
    account_tokens = ("build", "plan", "account", "usage", "tier", "concurrency")

    async def fake_run(user_message: str, state: SessionState) -> AgentRunResult:
        msg = user_message.lower()
        if any(t in msg for t in account_tokens):
            return AgentRunResult(
                text=(
                    f"Your plan tier is {state.plan_tier}. You're on the "
                    f"{state.plan_tier} tier with the standard limits."
                ),
                routed_to="account",
                tool_calls=[{
                    "tool_name": "get_account_status",
                    "args": {"user_id": state.user_id, "plan_tier": state.plan_tier},
                    "result": {"plan_tier": state.plan_tier},
                }],
                retrieved_chunk_ids=[],
            )
        if any(t in msg for t in knowledge_tokens):
            return AgentRunResult(
                text=(
                    "According to [chunk_test_001] and [chunk_test_002], "
                    "rotate your deploy key by generating a new ed25519 keypair, "
                    "adding the new public key in the Helix UI, updating CI "
                    "secrets, then deleting the old key."
                ),
                routed_to="knowledge",
                tool_calls=[{
                    "tool_name": "search_docs",
                    "args": {"query": user_message, "k": 5},
                    "result": {"chunks": [{"chunk_id": "chunk_test_001"}]},
                }],
                retrieved_chunk_ids=["chunk_test_001", "chunk_test_002"],
            )
        return AgentRunResult(
            text="Hi! I'm the Helix Support Concierge. Ask me anything.",
            routed_to="smalltalk",
            tool_calls=[],
            retrieved_chunk_ids=[],
        )

    monkeypatch.setattr(agent_runner, "run_root_agent", fake_run)
    return fake_run


@pytest_asyncio.fixture(scope="session")
async def seeded_vector_store():
    """One-time ingest into the test Chroma path so retriever tests have data.

    Forces the local hashing embedder so tests are network-independent.
    The same embedder is used at query time via reset_embedder() below.
    """
    from app.rag.embeddings import LocalHashingEmbedder, reset_embedder
    from app.rag.ingest import ingest_directory

    reset_embedder()
    # Pin to local embedder for the whole test session.
    import app.rag.embeddings as _emb_mod
    _emb_mod._embedder = LocalHashingEmbedder()

    vstore.reset_for_tests(persist_dir=str(_CHROMA_TMP), collection="helix_docs_test")
    docs_path = Path(__file__).parent.parent / "docs"
    await ingest_directory(docs_path, chunk_size=800, chunk_overlap=1)
    yield
    # Restore so other code picks up real embedder again.
    reset_embedder()


# pytest-asyncio: ensure the event loop policy is consistent
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
