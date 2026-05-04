"""
RAG ingest CLI.

    uv run python -m app.rag.ingest --path docs/

Re-ingest is idempotent — chunk IDs are derived from `(file, chunk_index)` so
upserts overwrite rather than duplicate. Reference guides are skipped (we only
index Helix product docs, not the developer-facing assignment guides).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from app.rag import store
from app.rag.chunker import chunk_markdown
from app.rag.embeddings import get_embedder

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_SKIP_BASENAMES = {
    "rag-guide.md",
    "google-adk-guide.md",
    "fastapi-async-guide.md",
}


def extract_metadata(file_path: Path, text: str) -> tuple[dict[str, Any], str]:
    """Returns (frontmatter_dict, body_without_frontmatter)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {"source": file_path.name}, text
    raw = yaml.safe_load(m.group(1)) or {}
    if not isinstance(raw, dict):
        raw = {}
    meta: dict[str, Any] = {
        "source": file_path.name,
        "title": str(raw.get("title", file_path.stem.replace("-", " ").title())),
        "product_area": str(raw.get("product_area", "general")),
    }
    tags = raw.get("tags") or []
    if isinstance(tags, list):
        meta["tags"] = ", ".join(str(t) for t in tags)
    return meta, text[m.end():]


def make_chunk_id(file_name: str, chunk_index: int) -> str:
    raw = f"{file_name}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


async def ingest_directory(
    docs_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    skip_guides: bool = True,
) -> dict[str, int]:
    md_files = sorted(p for p in docs_path.rglob("*.md") if p.is_file())
    if skip_guides:
        md_files = [p for p in md_files if p.name not in _SKIP_BASENAMES]

    embedder = get_embedder()
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    total_chunks = 0
    ids: list[str] = []
    embs: list[list[float]] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []

    for fp in md_files:
        text = fp.read_text(encoding="utf-8")
        meta_base, body = extract_metadata(fp, text)
        chunks = chunk_markdown(body, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            print(f"  {fp.name}: skipped (empty after chunking)")
            continue

        file_ids = [make_chunk_id(fp.name, i) for i in range(len(chunks))]
        file_metas = [
            {**meta_base, "chunk_index": i, "char_count": len(c)}
            for i, c in enumerate(chunks)
        ]
        file_embs = embedder.embed_documents(chunks)

        ids.extend(file_ids)
        embs.extend(file_embs)
        docs.extend(chunks)
        metas.extend(file_metas)
        total_chunks += len(chunks)
        print(f"  {fp.name}: {len(chunks)} chunks")

    if ids:
        store.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

    print(f"Ingest complete. {total_chunks} chunks across {len(md_files)} files.")
    return {"files": len(md_files), "chunks": total_chunks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest markdown docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument("--include-guides", action="store_true", help="Include *-guide.md files")
    args = parser.parse_args()

    asyncio.run(
        ingest_directory(
            args.path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            skip_guides=not args.include_guides,
        )
    )


if __name__ == "__main__":
    main()
