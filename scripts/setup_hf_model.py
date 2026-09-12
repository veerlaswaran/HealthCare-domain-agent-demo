"""
One-time HuggingFace model setup script.

Downloads and caches 'all-MiniLM-L6-v2' from HuggingFace, then re-seeds
both ChromaDB collections using SentenceTransformers embeddings (replacing
the TF-IDF offline index).

Run once on any machine with internet access:
    python scripts/setup_hf_model.py

After this completes:
  - Model is cached in ~/.cache/huggingface/hub/ (no further downloads needed)
  - Both ChromaDB collections are rebuilt with dense semantic embeddings
  - OFFLINE_MODE=false is written to .env at the project root
  - All subsequent scripts auto-detect the cached model and use it
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, title: str) -> None:
    print(f"\n[Step {n}] {title}")
    print("-" * 60)


def _ok(msg: str) -> None:
    print(f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 — Connectivity check
# ---------------------------------------------------------------------------

def check_connectivity() -> None:
    _step(1, "Checking HuggingFace connectivity")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        _ok("HuggingFace is reachable.")
    except Exception as exc:
        _fail(f"Cannot reach HuggingFace: {exc}\n"
              "  Ensure internet access is available and HuggingFace is not blocked.")


# ---------------------------------------------------------------------------
# Step 2 — Download and cache the embedding model
# ---------------------------------------------------------------------------

def download_model(model_name: str) -> None:
    _step(2, f"Downloading model: {model_name}")
    print("  This may take 1-2 minutes on first run (~90 MB).")

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)

    # Smoke-test
    vecs = model.encode(["appointment booking", "cricket stadium"], normalize_embeddings=True)
    if vecs.shape[1] != 384:
        _fail(f"Unexpected embedding dimension: {vecs.shape[1]} (expected 384)")

    sim = float(np.dot(vecs[0], vecs[1]))
    _ok(f"Model loaded. dim=384. Smoke-test cosine sim: {sim:.4f}")
    _ok("Model cached at: ~/.cache/huggingface/hub/")


# ---------------------------------------------------------------------------
# Step 3 — Write OFFLINE_MODE=false to .env
# ---------------------------------------------------------------------------

def write_env() -> None:
    _step(3, "Writing OFFLINE_MODE=false to .env")
    env_path = ROOT / ".env"

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = [
            line for line in env_path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("OFFLINE_MODE")
        ]

    existing_lines.append("OFFLINE_MODE=false")
    env_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
    _ok(f"Written to {env_path}")
    _ok("Load with:  export $(grep -v '^#' .env | xargs)")


# ---------------------------------------------------------------------------
# Step 4 — Re-seed ChromaDB with SentenceTransformer embeddings
# ---------------------------------------------------------------------------

def reseed_chromadb(model_name: str) -> None:
    _step(4, "Re-seeding ChromaDB with SentenceTransformer embeddings")

    # Set env before any app imports so config.OFFLINE_MODE resolves correctly
    os.environ["OFFLINE_MODE"] = "false"

    from app.config import (
        CHROMA_DIR, COLLECTION_FIXED, COLLECTION_SENTENCE,
        FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP, KB_DIR,
    )
    from scripts.seed_kb import (
        load_documents, fixed_size_chunks, sentence_chunks,
        _collect_chunks, index_collection,
    )
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Load KB documents
    docs = load_documents(KB_DIR)
    print(f"  Loaded {len(docs)} documents.")

    # Collect chunks for both strategies
    fixed_ids, fixed_texts, fixed_metas = _collect_chunks(
        docs, lambda t: fixed_size_chunks(t, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP)
    )
    sent_ids, sent_texts, sent_metas = _collect_chunks(
        docs, lambda t: sentence_chunks(t, group_size=2)
    )
    print(f"  Chunks — fixed: {len(fixed_texts)}, sentence: {len(sent_texts)}")

    # Embed with SentenceTransformers
    model = SentenceTransformer(model_name)

    print("  Embedding fixed-size chunks...")
    fixed_vecs = model.encode(fixed_texts, normalize_embeddings=True, show_progress_bar=False)
    fixed_embeddings = fixed_vecs.tolist()

    print("  Embedding sentence-based chunks...")
    sent_vecs = model.encode(sent_texts, normalize_embeddings=True, show_progress_bar=False)
    sent_embeddings = sent_vecs.tolist()

    # Index
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    n_fixed = index_collection(client, COLLECTION_FIXED,
                               fixed_ids, fixed_texts, fixed_metas, fixed_embeddings)
    _ok(f"Collection '{COLLECTION_FIXED}' : {n_fixed} chunks")

    n_sent = index_collection(client, COLLECTION_SENTENCE,
                              sent_ids, sent_texts, sent_metas, sent_embeddings)
    _ok(f"Collection '{COLLECTION_SENTENCE}': {n_sent} chunks")


# ---------------------------------------------------------------------------
# Step 5 — Calibrate and verify
# ---------------------------------------------------------------------------

def verify(model_name: str) -> None:
    _step(5, "Calibration & verification with SentenceTransformer scores")

    os.environ["OFFLINE_MODE"] = "false"

    from app.config import CHROMA_DIR, COLLECTION_FIXED
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import chromadb

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_FIXED)

    in_scope = [
        "How do I cancel or reschedule my appointment?",
        "What are the consultation fees for Cardiology?",
        "How long do blood test results take?",
    ]
    out_of_scope = [
        "What is the best cricket stadium in India?",
        "How do I apply for a bank loan?",
    ]

    def top1_sim(query: str) -> float:
        vec = model.encode([query], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=vec, n_results=1, include=["distances"])
        dist = res["distances"][0][0]
        return max(0.0, 1.0 - dist)   # cosine collection: distance = 1 - sim

    print("\n  In-scope queries (expect high similarity):")
    in_scores: list[float] = []
    for q in in_scope:
        s = top1_sim(q)
        in_scores.append(s)
        print(f"    {s:.4f}  |  {q}")

    print("\n  Out-of-scope queries (expect low similarity):")
    out_scores: list[float] = []
    for q in out_of_scope:
        s = top1_sim(q)
        out_scores.append(s)
        print(f"    {s:.4f}  |  {q}")

    min_in = min(in_scores)
    max_out = max(out_scores)
    midpoint = (min_in + max_out) / 2.0
    suggested = round((min_in + max_out) / 2.0, 2)

    print(f"\n  Min in-scope    : {min_in:.4f}")
    print(f"  Max out-of-scope: {max_out:.4f}")
    print(f"  Midpoint gap    : {midpoint:.4f}")
    print(f"  Suggested threshold for OFFLINE_MODE=false: {suggested}")
    print()

    if min_in > max_out:
        _ok("Clean separation — in-scope scores are all above out-of-scope scores.")
        _ok(f"Update SIMILARITY_THRESHOLD to {suggested} in app/config.py "
            "if you want optimal performance with SentenceTransformers.")
    else:
        print("  WARNING: Scores overlap — consider adjusting threshold after "
              "running scripts/calibrate_threshold.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from app.config import EMBEDDING_MODEL

    print("=" * 60)
    print("Practo — HuggingFace Model Setup")
    print(f"Model : {EMBEDDING_MODEL}")
    print(f"Root  : {ROOT}")
    print("=" * 60)

    check_connectivity()
    download_model(EMBEDDING_MODEL)
    write_env()
    reseed_chromadb(EMBEDDING_MODEL)
    verify(EMBEDDING_MODEL)

    print("\n" + "=" * 60)
    print("Setup complete.")
    print()
    print("Next steps:")
    print("  1. Load env:  export $(grep -v '^#' .env | xargs)")
    print("  2. Re-run calibration:  python scripts/calibrate_threshold.py")
    print("  3. Update SIMILARITY_THRESHOLD in app/config.py with the")
    print("     suggested value printed above (Step 5).")
    print("  4. Run demo:  python -m app.rag")
    print("=" * 60)


if __name__ == "__main__":
    main()
