"""
Chroma wrapper. We persist to `settings.chroma_persist_dir` so the index
survives restarts. Cosine space matches our L2-normalised embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.settings import settings


@dataclass
class Hit:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


_client: chromadb.api.ClientAPI | None = None
_collection_name: str | None = None
_persist_dir: str | None = None


def _get_client() -> chromadb.api.ClientAPI:
    global _client, _persist_dir
    if _client is None or _persist_dir != settings.chroma_persist_dir:
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _persist_dir = settings.chroma_persist_dir
    return _client


def _collection():
    return _get_client().get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def reset_for_tests(persist_dir: str | None = None, collection: str | None = None) -> None:
    """Force a fresh client. Used by tests pointing at a tmpdir."""
    global _client, _persist_dir, _collection_name
    _client = None
    _persist_dir = None
    _collection_name = collection
    if persist_dir is not None:
        settings.chroma_persist_dir = persist_dir
    if collection is not None:
        settings.chroma_collection = collection


def upsert(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    if not ids:
        return
    _collection().upsert(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )


def count() -> int:
    return _collection().count()


def query(
    embedding: list[float],
    k: int = 5,
    where: dict[str, Any] | None = None,
) -> list[Hit]:
    res = _collection().query(query_embeddings=[embedding], n_results=k, where=where or None)
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    hits: list[Hit] = []
    for cid, dist, doc, meta in zip(ids, dists, docs, metas):
        # Chroma cosine distance is in [0, 2]; clamp similarity to [0, 1].
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        hits.append(
            Hit(chunk_id=cid, score=round(score, 4), content=doc or "", metadata=meta or {})
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits
