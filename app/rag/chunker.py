"""
Heading-aware markdown chunker.

We split on H2/H3 headings so each chunk corresponds to a coherent section.
Sections longer than `max_chars` are sub-split on sentence boundaries with a
small sentence overlap to preserve context. Short tail-sections are merged
into the previous chunk to avoid one-line chunks.
"""
from __future__ import annotations

import re

_HEADING_SPLIT = re.compile(r"\n(?=#{2,3}\s)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentence_chunks(text: str, max_chars: int, overlap_sentences: int) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if cur_len + len(s) + 1 > max_chars and cur:
            out.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences > 0 else []
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_markdown(
    text: str,
    chunk_size: int = 800,
    overlap: int = 1,
    min_chars: int = 80,
) -> list[str]:
    """Split markdown body text into retrieval-friendly chunks.

    Args:
        text: markdown body (frontmatter already stripped).
        chunk_size: target maximum characters per chunk.
        overlap: sentence overlap between sub-chunks of long sections.
        min_chars: shorter trailing sections are merged into the previous chunk.
    """
    text = text.strip()
    if not text:
        return []

    sections = [s.strip() for s in _HEADING_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(_sentence_chunks(section, chunk_size, overlap))

    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]
