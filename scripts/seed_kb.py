"""
Task 3 — Chunk, embed, and index the knowledge base into TWO ChromaDB collections.

Collections:
  policy_fixed    — fixed-size chunks (200 chars, 40-char overlap)
  policy_sentence — sentence-based chunks (2 sentences per chunk)

Embedding backend:
  OFFLINE_MODE=true  (default) → TF-IDF + cosine (no network required)
  OFFLINE_MODE=false            → SentenceTransformers all-MiniLM-L6-v2

Run:  python scripts/seed_kb.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from app.config import (
    CHROMA_DIR,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    EMBEDDING_MODEL,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    KB_DIR,
    OFFLINE_MODE,
)
from app.embeddings import get_embedding_function, reset_embedding_function


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(kb_dir: Path) -> list[dict]:
    """Load all .txt files from the knowledge-base directory."""
    docs: list[dict] = []
    for path in sorted(kb_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        source = lines[0].replace("source:", "").strip() if lines[0].startswith("source:") else path.stem
        body = "\n".join(lines[1:]).strip()
        docs.append({"source": source, "text": body, "filename": path.name})
    return docs


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def fixed_size_chunks(text: str, size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def sentence_chunks(text: str, group_size: int = 2) -> list[str]:
    """Split text into sentence-based chunks, grouping `group_size` sentences."""
    raw_sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    chunks: list[str] = []
    for i in range(0, len(sentences), group_size):
        group = sentences[i: i + group_size]
        chunks.append(" ".join(group))
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB indexing
# ---------------------------------------------------------------------------

def _collect_chunks(documents: list[dict], chunker_fn) -> tuple[list[str], list[str], list[dict]]:
    """Build parallel lists of ids, texts, and metadatas from all documents."""
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metas: list[dict] = []
    for doc in documents:
        chunks = chunker_fn(doc["text"])
        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}__chunk{j:03d}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append(
                {
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                }
            )
    return all_ids, all_texts, all_metas


def index_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    all_ids: list[str],
    all_texts: list[str],
    all_metas: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Create (or recreate) a collection and add pre-computed embeddings."""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # Use cosine space so that dot product of normalised vecs == cosine sim
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metas,
        embeddings=embeddings,
    )
    return len(all_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Practo Knowledge Base — Seeding ChromaDB Collections")
    print(f"Mode: {'OFFLINE (TF-IDF)' if OFFLINE_MODE else f'ONLINE (SentenceTransformers: {EMBEDDING_MODEL})'}")
    print("=" * 60)

    docs = load_documents(KB_DIR)
    print(f"\nLoaded {len(docs)} documents from {KB_DIR}")
    for d in docs:
        print(f"  {d['filename']}  ({len(d['text'])} chars)  source={d['source']}")

    # Collect chunks for both strategies
    fixed_ids, fixed_texts, fixed_metas = _collect_chunks(
        docs, lambda t: fixed_size_chunks(t, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP)
    )
    sent_ids, sent_texts, sent_metas = _collect_chunks(
        docs, lambda t: sentence_chunks(t, group_size=2)
    )

    all_texts_combined = fixed_texts + sent_texts
    print(f"\nTotal chunks — fixed: {len(fixed_texts)}, sentence: {len(sent_texts)}")

    # Build embedding function — fit TF-IDF on combined corpus so vocabulary
    # covers chunks from both strategies
    reset_embedding_function()
    ef = get_embedding_function(corpus=all_texts_combined)

    print("\nEmbedding all chunks...")
    fixed_embeddings = ef(fixed_texts)
    sent_embeddings = ef(sent_texts)
    print(f"  Fixed-size   : {len(fixed_embeddings)} vectors, dim={len(fixed_embeddings[0])}")
    print(f"  Sentence-based: {len(sent_embeddings)} vectors, dim={len(sent_embeddings[0])}")

    # Index into ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    n_fixed = index_collection(client, COLLECTION_FIXED, fixed_ids, fixed_texts, fixed_metas, fixed_embeddings)
    print(f"\nCollection '{COLLECTION_FIXED}'  : {n_fixed} chunks indexed")
    print(f"  chunk size={FIXED_CHUNK_SIZE} chars, overlap={FIXED_CHUNK_OVERLAP} chars")

    n_sent = index_collection(client, COLLECTION_SENTENCE, sent_ids, sent_texts, sent_metas, sent_embeddings)
    print(f"Collection '{COLLECTION_SENTENCE}': {n_sent} chunks indexed")
    print(f"  strategy=sentence-based, group_size=2 sentences")

    # Spot-check
    print("\nSpot-check — 'cancellation fee policy'")
    query_vec = ef(["cancellation fee policy"])[0]
    for coll_name in [COLLECTION_FIXED, COLLECTION_SENTENCE]:
        coll = client.get_collection(name=coll_name)
        res = coll.query(query_embeddings=[query_vec], n_results=1,
                         include=["documents", "distances"])
        dist = res["distances"][0][0]
        # cosine collection: distance = 1 - cosine_sim
        sim = max(0.0, 1.0 - dist)
        snippet = res["documents"][0][0][:80].replace("\n", " ")
        print(f"  [{coll_name}]  sim={sim:.4f}  snippet: {snippet!r}")

    print(f"\nSeeding complete. ChromaDB written to: {CHROMA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
