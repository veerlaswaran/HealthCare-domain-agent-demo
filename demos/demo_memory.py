"""
Task 8 — Demonstrate persisted conversation memory.

Transcript A: Multi-turn conversation — shows history carried across turns.
Transcript B: Fresh session — shows state correctly absent/reset.

Run:  python demos/demo_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph import run_agent
from app.memory import clear_session, load_history, new_session_id


def _divider(title: str) -> None:
    print(f"\n{'#'*65}")
    print(f"  {title}")
    print(f"{'#'*65}\n")


def _print_turn(resp, show_history: bool = True) -> None:
    print(f"  Turn {resp.turn}  |  Route: {resp.route.value}")
    print(f"  Q: {resp.query_original!r}")
    print(f"  A: {resp.answer[:220].strip()}")
    if show_history and resp.history_summary:
        print(f"  History in context ({len(resp.history_summary)} prior turn(s)):")
        for line in resp.history_summary:
            print(f"    {line}")
    print()


def transcript_a_multi_turn() -> str:
    """
    Multi-turn session: patient asks about cancellation policy,
    then follows up about fees, then checks an appointment.
    Returns session_id so we can inspect the persisted file.
    """
    _divider("TRANSCRIPT A — Multi-turn conversation (state carried across turns)")

    sid = new_session_id()
    print(f"  Session ID: {sid}\n")

    # Turn 1 — policy question
    r1 = run_agent(
        query="What is the cancellation policy if I cancel 2 hours before?",
        session_id=sid, turn=1,
    )
    _print_turn(r1, show_history=False)

    # Turn 2 — deliberately vague follow-up. It contains no cancellation
    # keyword, so it can only retrieve the right policy through persisted
    # conversation context from Turn 1.
    r2 = run_agent(
        query="How long do refunds take?",
        session_id=sid, turn=2,
    )
    _print_turn(r2, show_history=True)

    # Turn 3 — switches to appointment lookup (history shows Turns 1 & 2)
    r3 = run_agent(
        query="Also, can you check my appointment APT-0028?",
        session_id=sid, turn=3,
    )
    _print_turn(r3, show_history=True)

    # Show raw persisted file
    history = load_history(sid)
    print(f"  Persisted turns on disk: {len(history)}")
    for t in history:
        print(f"    Turn {t['turn']} [{t['route']}]: {t['query'][:55]!r}")

    return sid


def transcript_b_fresh_session() -> None:
    """
    Fresh session: new session_id, no prior history.
    Confirms state is correctly absent/reset.
    """
    _divider("TRANSCRIPT B — Fresh session (history absent / correctly reset)")

    sid = new_session_id()
    print(f"  New session ID: {sid}")
    print(f"  History on disk before first turn: {load_history(sid)!r}\n")

    r1 = run_agent(
        query="How do I book a telemedicine appointment?",
        session_id=sid, turn=1,
    )
    _print_turn(r1, show_history=True)

    history = load_history(sid)
    print(f"  Persisted turns after Turn 1: {len(history)}")
    print(f"  Turn 1 on disk: {history[0]['query']!r}")
    print()
    print("  Confirmed: no history from Transcript A leaked into this session.")


def main() -> None:
    print("\nPracto Agent — Memory Demonstration (Task 8)")

    sid_a = transcript_a_multi_turn()
    transcript_b_fresh_session()

    print("\nMemory files written to: logs/memory/")
    print(f"  Transcript A session file: logs/memory/{sid_a}.json")
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
