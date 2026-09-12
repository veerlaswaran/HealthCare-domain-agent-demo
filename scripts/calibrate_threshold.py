"""
Threshold calibration script (supporting Task 4).

Measures top-1 cosine similarity for:
  - 3 in-scope queries (with domain keywords)
  - 2 deliberately out-of-scope queries (no domain keywords)

Also demonstrates the two-layer groundedness check:
  Layer 1: domain keyword gate
  Layer 2: cosine similarity >= SIMILARITY_THRESHOLD

Prints observed values and the chosen threshold (from config.py).

Run:  python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import COLLECTION_FIXED, OFFLINE_MODE, SIMILARITY_THRESHOLD
from app.rag import retrieve, _passes_domain_gate

IN_SCOPE = [
    "How do I cancel or reschedule my appointment?",
    "What are the consultation fees for Cardiology?",
    "How long do blood test results take?",
]

OUT_OF_SCOPE = [
    "What is the best cricket stadium in India?",
    "How do I apply for a bank loan?",
]


def measure(queries: list[str], label: str) -> list[float]:
    scores: list[float] = []
    print(f"\n{label}")
    print("-" * 70)
    for q in queries:
        hits = retrieve(q, collection_name=COLLECTION_FIXED, top_k=1)
        score = hits[0]["score"] if hits else 0.0
        scores.append(score)
        kw_pass = _passes_domain_gate(q)
        print(f"  sim={score:.4f}  kw_gate={'PASS' if kw_pass else 'FAIL'}  |  {q}")
    return scores


def main() -> None:
    print("=" * 70)
    print("Similarity Threshold Calibration — Two-Layer Groundedness Check")
    print(f"Embedding : {'TF-IDF (offline)' if OFFLINE_MODE else 'SentenceTransformers'}")
    print(f"Collection: {COLLECTION_FIXED}")
    print("=" * 70)
    print("\nLayer 1: Domain keyword gate (kw_gate)")
    print("Layer 2: Cosine similarity >= threshold")

    in_scores = measure(IN_SCOPE, "In-scope queries (expect kw_gate=PASS, high sim)")
    out_scores = measure(OUT_OF_SCOPE, "Out-of-scope queries (expect kw_gate=FAIL, low sim)")

    min_in = min(in_scores)
    max_out = max(out_scores)
    midpoint = (min_in + max_out) / 2

    print(f"\nIn-scope    scores : {[round(s, 4) for s in in_scores]}")
    print(f"Out-of-scope scores: {[round(s, 4) for s in out_scores]}")
    print(f"Min in-scope       : {min_in:.4f}")
    print(f"Max out-of-scope   : {max_out:.4f}")
    print(f"Midpoint gap       : {midpoint:.4f}")
    print(f"Chosen threshold   : {SIMILARITY_THRESHOLD:.4f}  (from app/config.py)")

    print("\nTwo-layer verdict for each query:")
    all_queries = [(q, True) for q in IN_SCOPE] + [(q, False) for q in OUT_OF_SCOPE]
    all_scores = in_scores + out_scores
    for (q, expected_in_scope), score in zip(all_queries, all_scores):
        kw = _passes_domain_gate(q)
        sim = score >= SIMILARITY_THRESHOLD
        grounded = kw and sim
        correct = grounded == expected_in_scope
        flag = "OK" if correct else "WRONG"
        print(f"  [{flag}] grounded={grounded} (kw={kw}, sim_ok={sim}) | {q[:55]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
