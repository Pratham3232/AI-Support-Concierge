# Helix SROP — AI Support Concierge

A stateful FastAPI service that answers Helix support questions across two
workflows in one ongoing conversation:

1. **Knowledge** — RAG over the product docs in `docs/`.
2. **Account** — per-user lookups (recent builds, plan tier, usage).

Routing happens through Google ADK's `AgentTool` — the LLM picks the
specialist; we never string-parse its output. Session state is persisted to
SQLite so every turn survives a process restart.

## Setup

```bash
git clone <repo-url> helix-srop && cd helix-srop
uv sync --extra dev
cp .env.example .env                       # fill in GOOGLE_API_KEY (optional)
uv run python -m app.rag.ingest --path docs/
uv run uvicorn app.main:app --reload
```

`GOOGLE_API_KEY` is **optional for ingest and tests** — without it the
embedder falls back to a deterministic local hashing model so retrieval
still works offline. The chat endpoint, however, calls Gemini through ADK
and does need the key.

## Quick test

```bash
SESSION=$(curl -s -X POST localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_demo","plan_tier":"pro"}' | jq -r .session_id)

curl -s -X POST localhost:8000/v1/chat/$SESSION \
  -H "Content-Type: application/json" \
  -d '{"content":"How do I rotate a deploy key?"}' | jq .

curl -s localhost:8000/v1/traces/<trace_id> | jq .
```

## Architecture

```
                    POST /v1/chat/{session_id}
                                │
                                ▼
                    ┌────────────────────────┐
                    │      pipeline.run      │
                    │  1. load SessionState  │  ← sqlite (sessions.state)
                    │  2. run ADK root       │
                    │  3. capture trace      │
                    │  4. update state       │
                    │  5. commit messages    │
                    └────────────┬───────────┘
                                 │
                  AgentTool routing (LLM-decided)
                       │                │
                       ▼                ▼
                KnowledgeAgent     AccountAgent
                ─────────────      ─────────────
                search_docs()      get_recent_builds()
                                   get_account_status()
                       │
                       ▼
                  Chroma (cosine, persisted)
                  embeddings: gemini-embedding-001 (3072-dim)
                              ↳ local hashing fallback (512-dim)
```

State of the world per turn:

```
sqlite              chroma             ADK
──────              ──────             ───
sessions ─load──▶ search_docs() ◀─tool─ KnowledgeAgent ◀─AgentTool─ root
sessions ◀─save── pipeline      ◀─text──────────────────────────────┘
agent_traces ◀──── pipeline (one row per turn)
messages ◀──────── pipeline (user + assistant)
```

## Design decisions

### State persistence — Pattern 3 from the ADK guide

I store a small `SessionState` Pydantic model as JSON in `sessions.state` and
inject it into the root agent's system prompt at the start of every turn
(`app/agents/orchestrator.py::build_root_agent`). The full message log lives
separately in `messages` for audit, but the agent's per-turn context comes
from the rendered prompt.

Why this over Patterns 1/2 (custom `BaseSessionService` or rehydrating
turns into ADK):

- Restart-safe by construction. There's no in-memory ADK session to lose;
  the source of truth is one SQLite row per session.
- Trivial to reason about. State is a typed Pydantic model, not opaque ADK
  internals; one schema change is a one-line edit in `app/srop/state.py`.
- Cheap on the context window. We only push 5 fields (`user_id`,
  `plan_tier`, `turn_count`, `last_agent`, `open_ticket_ids`) instead of
  full prior turns. For a routing+lookup workload that's enough.

The trade-off is that the agent doesn't see verbatim prior turns. For this
workload (route → specialist → return) that hasn't been a problem; for a
truly conversational flow I'd switch to Pattern 2.

### Chunking — heading-aware with sentence sub-splitting

`app/rag/chunker.py` splits markdown on `##`/`###` headings (each section
becomes one or more chunks), then sub-splits long sections on sentence
boundaries with one sentence of overlap. Short trailing sections are merged
into the previous chunk.

Reasoning: the corpus in `docs/` is well-structured product reference; an
H2 section maps cleanly onto a single user-intent ("rotate a key", "view
audit logs"). Fixed-size chunking would split mid-procedure and ruin
citations; sentence-only chunking would lose the heading context that makes
chunks self-explanatory when shown to the agent.

### Vector store — Chroma

Persistent, cosine space, zero infra. The whole `docs/` folder fits in
~90 chunks so anything heavier (LanceDB, Pinecone) would be overkill.

### Embeddings — Google `gemini-embedding-001` with a local fallback

`app/rag/embeddings.py` uses Google's 3072-dim `gemini-embedding-001` model
when `GOOGLE_API_KEY` is configured and a deterministic 512-dim hashing-trick
embedder otherwise. Same backend at ingest and query time (mixing models
silently corrupts retrieval). The fallback exists so ingest, tests, and
`pytest -q` all run on a clean clone with no API keys.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/sessions` | `{user_id, plan_tier}` → `{session_id, user_id, plan_tier}` |
| `POST` | `/v1/chat/{session_id}` | `{content}` → `{reply, routed_to, trace_id}`; supports `Idempotency-Key` |
| `GET`  | `/v1/traces/{trace_id}` | full structured trace |
| `GET`  | `/healthz` | liveness |

Errors are RFC 7807 problem details. Codes: `SESSION_NOT_FOUND` (404),
`TRACE_NOT_FOUND` (404), `UPSTREAM_TIMEOUT` (504).

## Tests

```bash
uv run pytest -q
```

Nine tests, all passing on a clean clone. Coverage:

- create-session round-trip
- two-turn integration (knowledge → account) with ADK mocked at the
  `app.agents.runner.run_root_agent` boundary, plus an assertion that the
  follow-up turn sees `plan_tier` from injected state without re-asking
- 404 problem details for missing session and missing trace
- idempotency replay returns the same `trace_id`
- `/healthz`
- `search_docs` over a freshly ingested test Chroma store — non-empty
  chunk IDs, scores in [0, 1], top hit comes from `deploy-keys.md`
- chunker non-empty + empty-input handling

## Known limitations

- The mock ADK fixture in `tests/conftest.py` is keyword-based, not a real
  routing test. Genuine routing accuracy needs a live LLM and an eval
  harness (E7).
- The local hashing embedder is a fallback for offline testing — semantic
  quality is BoW-grade. Configure `GOOGLE_API_KEY` for production use.
- No streaming SSE, no reranker, no Docker (extensions E3/E4/E6 not done).
- `score_threshold` is 0 by default; the prompt enforces "I don't have
  documentation on that" when the agent can't find an answer instead.

## What I'd do with more time

- Pull two-stage retrieval (cheap top-20 → LLM-as-judge top-5).
- Move ADK runner construction out of the per-turn hot path — currently I
  rebuild the root agent each turn so the state-injected instruction
  changes; cache the sub-agents and only re-init root.
- Eval harness scoring routing accuracy + answer faithfulness against a
  small labelled set, wired into CI.
- Postgres + Alembic for production; the schema is already
  Postgres-compatible.

## Time spent

| Phase | Time |
|---|---|
| Setup + DB + FastAPI scaffolding | 25 min |
| RAG ingest + chunker + search_docs | 35 min |
| Agents (knowledge, account, root) + AgentTool wiring | 30 min |
| pipeline.py + state injection + ADK runner abstraction | 45 min |
| Routes + RFC 7807 errors + idempotency | 25 min |
| Tests (mock at ADK boundary + retriever) | 25 min |
| README + verification | 20 min |
| **Total** | **~3h 25min** |

## Extensions completed

- [x] **E1: Idempotency** — `Idempotency-Key` header, `UNIQUE(session_id,
      idempotency_key)` on assistant messages, replay returns the original
      `trace_id`. Test in `tests/test_api.py::test_idempotency_replays_same_response`.
- [ ] E2: Escalation agent (`tickets` table is in the schema, not yet wired)
- [ ] E3: Streaming SSE
- [ ] E4: Reranking
- [ ] E5: Guardrails
- [ ] E6: Docker
- [ ] E7: Eval harness
# AI-Support-Concierge
