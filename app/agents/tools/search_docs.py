"""
search_docs — vector search tool exposed to KnowledgeAgent.

Returns chunk IDs, similarity scores in [0, 1] and snippet text. The agent is
instructed to cite chunk_id in its answer (e.g. "[chunk_abc123]"), so the IDs
must round-trip through the tool result verbatim.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

from app.rag import store
from app.rag.embeddings import get_embedder
from app.settings import settings

# Module-level capture of the most recent retrieval. The pipeline reads this
# after the agent run completes so it can populate `retrieved_chunk_ids` in
# the trace without parsing tool output. Reset by the pipeline at the start of
# each turn.
_LAST_HITS: list[dict[str, Any]] = []


def get_last_hits() -> list[dict[str, Any]]:
    return list(_LAST_HITS)


def reset_last_hits() -> None:
    _LAST_HITS.clear()


@dataclass
class DocChunk:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


def _do_search(query: str, k: int, product_area: str | None) -> list[DocChunk]:
    emb = get_embedder().embed_query(query)
    where = {"product_area": product_area} if product_area else None
    hits = store.query(emb, k=k, where=where)
    threshold = settings.score_threshold
    if threshold > 0:
        hits = [h for h in hits if h.score >= threshold]
    return [
        DocChunk(chunk_id=h.chunk_id, score=h.score, content=h.content, metadata=h.metadata)
        for h in hits
    ]


async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> dict[str, Any]:
    """Search Helix product documentation by semantic similarity.

    Args:
        query: natural-language query from the user.
        k: number of chunks to return (default 5).
        product_area: optional metadata filter — one of `security`, `ci-cd`,
            `billing`, `observability`, etc.

    Returns:
        dict with key `chunks`: list of `{chunk_id, score, snippet, source,
        title, product_area}`. Always cite `chunk_id` when answering.
    """
    chunks = await asyncio.to_thread(_do_search, query, k, product_area)

    _LAST_HITS.clear()
    _LAST_HITS.extend(asdict(c) for c in chunks)

    return {
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "score": c.score,
                "snippet": c.content[:600],
                "source": c.metadata.get("source", ""),
                "title": c.metadata.get("title", ""),
                "product_area": c.metadata.get("product_area", ""),
            }
            for c in chunks
        ],
    }
