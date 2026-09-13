"""
Task 6 — Agent tools.

Tool 1: check_appointment_status(record_id)
    Looks up an appointment record and returns its fields plus a designed
    escalation_score.

    Escalation formula:
        recency_norm   = days_since_created / 30          (normalised [0, 1])
        escalation_score = 0.6 * follow_up_weight + 0.4 * recency_norm

    where follow_up_weight = 1.0 if follow_up_required else 0.0.

    Weights rationale:
      - follow_up_required is a clinical signal — a patient who needs a
        follow-up but hasn't been seen recently is the primary escalation
        concern (60 % weight).
      - recency_norm captures how long ago the appointment was created;
        older unresolved cases deserve more attention (40 % weight).

    Threshold = 0.60
      Justification from dataset distribution (seed=42, n=100):
        P80 of days_since_created = 25 days.
        A record with follow_up=False and days=25 scores:
            0.6*0 + 0.4*(25/30) = 0.333   — below threshold, no escalation.
        Any record with follow_up=True scores:
            minimum: 0.6*1 + 0.4*0 = 0.600  (threshold, escalated)
        So 0.60 means "escalate every follow-up case and every record older
        than P80 that also has follow_up=True".  In the dataset, 24 records
        have follow_up=True — all 24 are escalated regardless of age, which
        is the correct clinical behaviour.

Tool 2: retrieve_policy(query)
    Thin wrapper around the RAG core from Part 1 (grounded_answer).
    Used by the agent graph so both tools share a uniform call interface.
"""

from __future__ import annotations

from app.config import COLLECTION_FIXED, DAYS_SINCE_CREATED_MAX

# ---------------------------------------------------------------------------
# Dataset lookup (in-memory index built once at import)
# ---------------------------------------------------------------------------
# Import the pre-generated appointments list from dataset.py.
# This avoids any SQLite dependency and keeps the tool deterministic.

from scripts.dataset import APPOINTMENTS as _APPOINTMENTS

_APPOINTMENT_INDEX: dict[str, dict] = {
    r["record_id"]: r for r in _APPOINTMENTS
}

# ---------------------------------------------------------------------------
# Escalation constants
# ---------------------------------------------------------------------------

ESCALATION_FOLLOW_UP_WEIGHT: float = 0.60
ESCALATION_RECENCY_WEIGHT: float = 0.40
ESCALATION_THRESHOLD: float = 0.60   # see docstring above


def _compute_escalation(record: dict) -> tuple[float, bool]:
    """Return (escalation_score, escalate) for a record."""
    recency_norm = record["days_since_created"] / DAYS_SINCE_CREATED_MAX
    follow_up_weight = 1.0 if record["follow_up_required"] else 0.0
    score = round(
        ESCALATION_FOLLOW_UP_WEIGHT * follow_up_weight
        + ESCALATION_RECENCY_WEIGHT * recency_norm,
        4,
    )
    return score, score >= ESCALATION_THRESHOLD


# ---------------------------------------------------------------------------
# Tool 1 — check_appointment_status
# ---------------------------------------------------------------------------

def check_appointment_status(record_id: str) -> dict:
    """
    Look up an appointment by record_id and return its fields plus
    escalation_score and escalate flag.

    Args:
        record_id: e.g. "APT-0042"

    Returns:
        {
          "found"               : bool,
          "record_id"           : str,
          "category"            : str | None,
          "status"              : str | None,
          "consultation_fee_inr": int | None,
          "days_since_created"  : int | None,
          "follow_up_required"  : bool | None,
          "escalation_score"    : float | None,
          "escalate"            : bool | None,
          "error"               : str | None,
        }
    """
    record = _APPOINTMENT_INDEX.get(record_id.upper())

    if record is None:
        return {
            "found": False,
            "record_id": record_id,
            "category": None,
            "status": None,
            "consultation_fee_inr": None,
            "days_since_created": None,
            "follow_up_required": None,
            "escalation_score": None,
            "escalate": None,
            "error": f"No appointment found with record_id='{record_id}'",
        }

    escalation_score, escalate = _compute_escalation(record)

    return {
        "found": True,
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "days_since_created": record["days_since_created"],
        "follow_up_required": record["follow_up_required"],
        "escalation_score": escalation_score,
        "escalate": escalate,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Tool 2 — retrieve_policy  (RAG wrapper)
# ---------------------------------------------------------------------------

def retrieve_policy(
    query: str,
    collection_name: str = COLLECTION_FIXED,
    *,
    user_query: str | None = None,
    conversation_context: str = "",
) -> dict:
    """
    Answer a policy question using the RAG core (Task 3-5).

    Args:
        query           : natural-language patient question
        collection_name : ChromaDB collection to search (default: policy_fixed)

    Returns:
        grounded_answer() dict — see app/rag.py for schema.
    """
    from app.rag import grounded_answer
    return grounded_answer(
        query,
        collection_name=collection_name,  # type: ignore[arg-type]
        user_query=user_query,
        conversation_context=conversation_context,
    )


# ---------------------------------------------------------------------------
# Escalation distribution report (informational, used in README)
# ---------------------------------------------------------------------------

def escalation_distribution_report() -> None:
    """Print escalation score distribution across the full dataset."""
    scores = []
    escalated = 0
    for r in _APPOINTMENTS:
        score, esc = _compute_escalation(r)
        scores.append(score)
        if esc:
            escalated += 1

    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    print("=" * 55)
    print("Escalation Score Distribution (seed=42, n=100)")
    print(f"  Formula: 0.6*follow_up + 0.4*(days/30)")
    print(f"  Threshold: {ESCALATION_THRESHOLD}")
    print(f"  Escalated: {escalated}/{n} ({100*escalated/n:.0f}%)")
    print()
    for p in [25, 50, 75, 80, 90]:
        idx = min(int(n * p / 100), n - 1)
        print(f"  P{p:<3}: {scores_sorted[idx]:.4f}")
    print(f"  min : {scores_sorted[0]:.4f}")
    print(f"  max : {scores_sorted[-1]:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    # Quick smoke-test
    escalation_distribution_report()
    print()

    for rid in ["APT-0001", "APT-0008", "APT-ZZZZ"]:
        result = check_appointment_status(rid)
        print(f"{rid}: found={result['found']}  status={result.get('status')}  "
              f"escalation_score={result.get('escalation_score')}  "
              f"escalate={result.get('escalate')}")
