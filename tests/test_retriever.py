"""Unit tests for the RAG layer."""
from __future__ import annotations

import pytest

from app.agents.tools.search_docs import search_docs
from app.rag.chunker import chunk_markdown


@pytest.mark.asyncio
async def test_search_docs_returns_results_with_chunk_ids(seeded_vector_store):
    result = await search_docs("how to rotate a deploy key", k=3)
    chunks = result["chunks"]

    assert len(chunks) > 0, "expected at least one chunk for this query"
    assert all(c["chunk_id"].startswith("chunk_") for c in chunks)
    assert all(0.0 <= c["score"] <= 1.0 for c in chunks)
    assert all(c["snippet"] for c in chunks)
    # Top hit should come from deploy-keys.md.
    sources = {c["source"] for c in chunks}
    assert "deploy-keys.md" in sources


def test_chunker_produces_non_empty_chunks():
    text = (
        "## Section A\n\nFirst paragraph with a sentence. Another sentence here.\n\n"
        "## Section B\n\nMore content.\n\n### Subsection\n\nEven more content."
    )
    chunks = chunk_markdown(text, chunk_size=120, overlap=1)
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


def test_chunker_handles_empty_input():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []
