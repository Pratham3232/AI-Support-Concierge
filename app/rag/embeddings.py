"""
Embedding backends.

Defaults to Google text-embedding-004 when GOOGLE_API_KEY is set.
Falls back to a local hashing embedder so ingest and tests run offline.
The same backend is used at ingest time and query time — mixing models
silently corrupts retrieval.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

from app.settings import settings


class Embedder(ABC):
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class LocalHashingEmbedder(Embedder):
    """Deterministic bag-of-words hashing trick.

    Fast, offline, no API key needed. Good enough for the 90-chunk corpus in
    `docs/` when used consistently at both ingest and query time.
    """

    dimension = 512

    def __init__(self, dim: int | None = None) -> None:
        if dim is not None:
            self.dimension = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return v
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class GoogleEmbedder(Embedder):
    """Google gemini-embedding-001 via the google-genai SDK (3072-dim)."""

    dimension = 3072

    def __init__(self) -> None:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        from google import genai

        self._client = genai.Client(api_key=settings.google_api_key)

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google import genai

        response = self._client.models.embed_content(
            model=settings.embedding_model,
            contents=texts,
            config=genai.types.EmbedContentConfig(task_type=task_type),
        )
        return [list(e.values) for e in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch in groups of 20 to stay within API limits.
        results: list[list[float]] = []
        for i in range(0, len(texts), 20):
            results.extend(self._embed_batch(texts[i : i + 20], "RETRIEVAL_DOCUMENT"))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], "RETRIEVAL_QUERY")[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Lazy singleton — picks GoogleEmbedder when key is set and valid, else local.

    Validates the key with a tiny test call so a bad/placeholder key doesn't
    crash mid-ingest; instead we fall back gracefully and print a warning.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    if settings.google_api_key and settings.google_api_key != "your-google-api-key-here":
        try:
            candidate = GoogleEmbedder()
            # Verify the key works before committing to this backend.
            candidate.embed_query("ping")
            _embedder = candidate
            return _embedder
        except Exception as exc:
            print(
                f"[embeddings] GoogleEmbedder unavailable ({exc.__class__.__name__}: {exc})\n"
                "  Falling back to local hashing embedder."
            )
    _embedder = LocalHashingEmbedder()
    return _embedder


def reset_embedder() -> None:
    """Test hook — clears singleton so a test can force a specific backend."""
    global _embedder
    _embedder = None
