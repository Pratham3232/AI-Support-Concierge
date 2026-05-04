"""Integration tests for the REST surface — LLM mocked at the ADK boundary."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/v1/sessions", json={"user_id": "u_test_001"})
    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert body["user_id"] == "u_test_001"
    assert body["plan_tier"] == "free"


@pytest.mark.asyncio
async def test_knowledge_query_routes_correctly(client, mock_adk):
    sess = await client.post(
        "/v1/sessions", json={"user_id": "u_test_002", "plan_tier": "pro"}
    )
    assert sess.status_code == 200
    session_id = sess.json()["session_id"]

    r1 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"}
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["routed_to"] == "knowledge"
    assert "[chunk_" in body1["reply"]
    trace_id = body1["trace_id"]

    trace = await client.get(f"/v1/traces/{trace_id}")
    assert trace.status_code == 200
    tbody = trace.json()
    assert tbody["routed_to"] == "knowledge"
    assert len(tbody["retrieved_chunk_ids"]) > 0
    assert tbody["latency_ms"] >= 0
    assert any(tc["tool_name"] == "search_docs" for tc in tbody["tool_calls"])

    # Turn 2 — agent must know plan_tier from injected state, no re-ask.
    r2 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"}
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["routed_to"] == "account"
    assert "pro" in r2.json()["reply"].lower()


@pytest.mark.asyncio
async def test_session_not_found_returns_404(client, mock_adk):
    resp = await client.post("/v1/chat/nonexistent-id", json={"content": "hello"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["title"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_trace_not_found_returns_404(client):
    resp = await client.get("/v1/traces/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["title"] == "TRACE_NOT_FOUND"


@pytest.mark.asyncio
async def test_idempotency_replays_same_response(client, mock_adk):
    sess = await client.post("/v1/sessions", json={"user_id": "u_idem"})
    session_id = sess.json()["session_id"]

    headers = {"Idempotency-Key": "abc-123"}
    r1 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    r2 = await client.post(
        f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"},
        headers=headers,
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["trace_id"] == r2.json()["trace_id"]


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
