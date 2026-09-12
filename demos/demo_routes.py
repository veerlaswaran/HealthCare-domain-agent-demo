"""
Task 7 — Demonstrate both routing branches of the LangGraph agent.

Shows:
  - RAG route firing on a policy question
  - Appointment route firing on an appointment-status query (with APT-ID)
  - Appointment route firing on a natural-language status query (no explicit ID,
    triggers keyword-based classification, then asks for ID gracefully)

Run:  python demos/demo_routes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph import run_agent
from app.memory import new_session_id


def _print_response(resp, label: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Session : {resp.session_id}")
    print(f"  Turn    : {resp.turn}")
    print(f"  Route   : {resp.route.value}")
    print(f"  Query   : {resp.query_original!r}")
    if resp.query_sanitised != resp.query_original:
        print(f"  Masked  : {resp.query_sanitised!r}")

    print(f"\n  Answer:\n")
    for line in resp.answer.split("\n"):
        print(f"    {line}")

    if resp.rag_hits:
        print(f"\n  RAG hits (top {len(resp.rag_hits)}):")
        for h in resp.rag_hits:
            print(f"    [{h.score:.4f}] {h.source} — {h.snippet[:70]!r}")
        print(f"  Top similarity : {resp.top_similarity_score:.4f}")
        print(f"  Grounded       : {resp.grounded}")

    if resp.appointment:
        a = resp.appointment
        print(f"\n  Appointment detail:")
        print(f"    record_id         : {a.record_id}")
        print(f"    status            : {a.status}")
        print(f"    category          : {a.category}")
        print(f"    fee (INR)         : {a.consultation_fee_inr}")
        print(f"    days_since_created: {a.days_since_created}")
        print(f"    follow_up_required: {a.follow_up_required}")
        print(f"    escalation_score  : {a.escalation_score:.4f}")
        print(f"    escalate          : {a.escalate}")

    if resp.guardrail_events:
        print(f"\n  Guardrail events:")
        for ev in resp.guardrail_events:
            print(f"    [{ev.event.value}] {ev.detail}")

    print()


def main() -> None:
    print("\nPracto Agent — Route Demonstration (Task 7)")
    print("Both RAG and Appointment branches are exercised below.\n")

    # ------------------------------------------------------------------
    # Route 1: RAG — policy question, no appointment ID
    # ------------------------------------------------------------------
    sid_rag = new_session_id()
    resp1 = run_agent(
        query="How do I cancel my appointment without paying a cancellation fee?",
        session_id=sid_rag,
        turn=1,
    )
    _print_response(resp1, "ROUTE: RAG  |  Policy query — cancellation policy")

    # Second RAG query in same session (different policy topic)
    resp2 = run_agent(
        query="Is telemedicine available for Cardiology follow-ups?",
        session_id=sid_rag,
        turn=2,
    )
    _print_response(resp2, "ROUTE: RAG  |  Policy query — telemedicine eligibility")

    # ------------------------------------------------------------------
    # Route 2: Appointment — explicit APT-ID in query
    # ------------------------------------------------------------------
    sid_apt = new_session_id()
    resp3 = run_agent(
        query="What is the status of appointment APT-0008?",
        session_id=sid_apt,
        turn=1,
    )
    _print_response(resp3, "ROUTE: APPOINTMENT  |  Explicit APT-ID — APT-0008 (follow_up=True)")

    # Appointment that does NOT require escalation
    resp4 = run_agent(
        query="Can you check appointment APT-0001 for me?",
        session_id=sid_apt,
        turn=2,
    )
    _print_response(resp4, "ROUTE: APPOINTMENT  |  Explicit APT-ID — APT-0001 (follow_up=False)")

    # ------------------------------------------------------------------
    # Route 2b: Appointment — keyword-based classification, no explicit ID
    # ------------------------------------------------------------------
    sid_kw = new_session_id()
    resp5 = run_agent(
        query="I want to check the status of my cancelled booking",
        session_id=sid_kw,
        turn=1,
    )
    _print_response(resp5, "ROUTE: APPOINTMENT  |  Keyword intent, no ID provided (graceful ask)")

    print("Demo complete. Both route branches demonstrated.")
    print("Log written to logs/agent.jsonl")


if __name__ == "__main__":
    main()
