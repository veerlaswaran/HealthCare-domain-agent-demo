"""
Task 5 — Evaluate and compare both chunking strategies.

Computes Precision@3 and Recall@3 at the DOCUMENT level for each of the 5
in-scope queries from Task 4.  Chunks are mapped back to their parent document
(via the 'source' metadata field) and deduplicated before scoring.

Run:  python scripts/evaluate_chunking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import COLLECTION_FIXED, COLLECTION_SENTENCE, RETRIEVAL_TOP_K
from app.rag import retrieve, IN_SCOPE_QUERIES

# ---------------------------------------------------------------------------
# Ground-truth: for each query, the set of parent-document source names that
# are genuinely relevant (1 or more docs per query).
# These are determined by reading the knowledge base and identifying which
# documents contain the answer.
# ---------------------------------------------------------------------------
GROUND_TRUTH: dict[str, set[str]] = {
    "How do I cancel my appointment without paying a fee?": {
        "cancellation_rescheduling_policy",
    },
    "What is the consultation fee for a Cardiology visit?": {
        "consultation_fee_structure",
    },
    "Can I get a prescription refill without visiting the clinic?": {
        "prescription_refill_policy",
    },
    "How long does it take to get blood test results?": {
        "lab_test_turnaround_times",
    },
    "Is telemedicine available for Orthopedics?": {
        "telemedicine_eligibility",
    },
}


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_docs)
    return hits / len(top_k)


def recall_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int) -> float:
    """Fraction of relevant documents found in top-k retrieved documents."""
    if not relevant_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    hits = sum(1 for d in relevant_docs if d in top_k)
    return hits / len(relevant_docs)


def get_retrieved_docs(
    query: str,
    collection_name: str,
    top_k: int,
) -> list[str]:
    """
    Retrieve top_k chunks, map each to its parent document source, and
    return a deduplicated list preserving rank order.
    """
    hits = retrieve(query, collection_name=collection_name, top_k=top_k)  # type: ignore[arg-type]
    seen: set[str] = set()
    ordered_docs: list[str] = []
    for h in hits:
        src = h["source"]
        if src not in seen:
            seen.add(src)
            ordered_docs.append(src)
    return ordered_docs


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(collection_name: str, k: int = RETRIEVAL_TOP_K) -> list[dict]:
    rows: list[dict] = []
    for query in IN_SCOPE_QUERIES:
        relevant = GROUND_TRUTH[query]
        retrieved = get_retrieved_docs(query, collection_name, top_k=k)
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        rows.append(
            {
                "query": query,
                "relevant_docs": relevant,
                "retrieved_docs": retrieved,
                "precision_at_k": p,
                "recall_at_k": r,
            }
        )
    return rows


def print_results(collection_name: str, rows: list[dict], k: int) -> tuple[float, float]:
    print(f"\nCollection: {collection_name}  (k={k})")
    print("-" * 80)
    total_p, total_r = 0.0, 0.0
    for row in rows:
        q_short = row["query"][:55]
        print(f"  Q: {q_short!r}")
        print(f"     relevant      : {sorted(row['relevant_docs'])}")
        print(f"     retrieved     : {row['retrieved_docs']}")
        print(
            f"     Precision@{k}  : {row['precision_at_k']:.3f}   "
            f"Recall@{k}: {row['recall_at_k']:.3f}"
        )
        total_p += row["precision_at_k"]
        total_r += row["recall_at_k"]
    n = len(rows)
    mean_p = total_p / n
    mean_r = total_r / n
    print(f"\n  Mean Precision@{k}: {mean_p:.3f}")
    print(f"  Mean Recall@{k}   : {mean_r:.3f}")
    return mean_p, mean_r


def main() -> None:
    k = RETRIEVAL_TOP_K
    print("=" * 80)
    print("Practo — Chunking Strategy Evaluation (Precision@3 / Recall@3)")
    print(f"Queries: {len(IN_SCOPE_QUERIES)}  |  k={k}")
    print("=" * 80)

    rows_fixed = evaluate(COLLECTION_FIXED, k=k)
    mp_fixed, mr_fixed = print_results(COLLECTION_FIXED, rows_fixed, k=k)

    rows_sent = evaluate(COLLECTION_SENTENCE, k=k)
    mp_sent, mr_sent = print_results(COLLECTION_SENTENCE, rows_sent, k=k)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  {'Strategy':<22} {'Mean P@3':>10} {'Mean R@3':>10}")
    print(f"  {'-'*42}")
    print(f"  {COLLECTION_FIXED:<22} {mp_fixed:>10.3f} {mr_fixed:>10.3f}")
    print(f"  {COLLECTION_SENTENCE:<22} {mp_sent:>10.3f} {mr_sent:>10.3f}")

    # Recommendation
    print("\nRecommendation:")
    if mp_fixed >= mp_sent and mr_fixed >= mr_sent:
        winner = COLLECTION_FIXED
        loser = COLLECTION_SENTENCE
    elif mp_sent >= mp_fixed and mr_sent >= mr_fixed:
        winner = COLLECTION_SENTENCE
        loser = COLLECTION_FIXED
    else:
        # Mixed result — prefer higher recall for support use case
        winner = COLLECTION_FIXED if mr_fixed >= mr_sent else COLLECTION_SENTENCE
        loser = COLLECTION_SENTENCE if winner == COLLECTION_FIXED else COLLECTION_FIXED

    print(
        f"  Deploy '{winner}'. "
        f"It achieved Mean P@3={mp_fixed if winner == COLLECTION_FIXED else mp_sent:.3f} "
        f"and Mean R@3={mr_fixed if winner == COLLECTION_FIXED else mr_sent:.3f}, "
        f"outperforming '{loser}' "
        f"(P@3={mp_fixed if loser == COLLECTION_FIXED else mp_sent:.3f}, "
        f"R@3={mr_fixed if loser == COLLECTION_FIXED else mr_sent:.3f}). "
        f"For a patient-support agent, recall matters more than precision because "
        f"missing a relevant policy is worse than surfacing a mildly off-topic chunk."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
