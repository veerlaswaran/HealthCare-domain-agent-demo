"""
Task 16 — Retry, per-node timeout, and global timeout demonstration.

(a) Retry policy recovering within configured attempts:
    A simulated flaky node fails the first 2 calls via an attempt counter,
    then succeeds on attempt 3.  Exponential backoff with jitter is applied.

    Config: max_attempts=3, initial_delay=0.05s, max_delay=0.5s,
            backoff_factor=2.0, jitter=True

(b) Per-node timeout firing a clean error:
    A node simulates a call that sleeps for 2s; the per-node timeout is
    set to 0.3s.  NodeTimeoutError is raised cleanly — no hang.

(c) Global timeout cancelling the whole run:
    A graph run is launched where one node sleeps for 3s; the global
    timeout is set to 0.5s.  GlobalTimeoutError is raised.

Run:
  python demos/demo_resilience.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.resilience import (
    GlobalTimeoutError,
    NodeTimeoutError,
    RetryExhaustedError,
    node_timeout,
    retry_with_backoff,
    run_with_global_timeout,
)
from app.graph import run_agent
from app.memory import new_session_id


def _section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# ---------------------------------------------------------------------------
# (a) Retry recovering within max_attempts
# ---------------------------------------------------------------------------

def demo_retry_recovery() -> None:
    _section("(a) Retry — recovers after 2 transient failures")

    attempt_counter = {"n": 0}

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=0.05,
        max_delay=0.5,
        backoff_factor=2.0,
        jitter=True,
    )
    def flaky_appointment_lookup(record_id: str) -> dict:
        """
        Simulates a flaky tool call that fails the first 2 attempts
        (e.g. transient DB connection error) then succeeds on attempt 3.
        """
        attempt_counter["n"] += 1
        if attempt_counter["n"] < 3:
            raise ConnectionError(
                f"Simulated transient DB error (attempt {attempt_counter['n']})"
            )
        from app.tools import check_appointment_status
        return check_appointment_status(record_id)

    print(f"  Config: max_attempts=3, initial_delay=0.05s, backoff_factor=2.0, jitter=True")
    print(f"  Simulated failure on attempts 1 and 2; success on attempt 3.")
    print()

    t0 = time.perf_counter()
    result = flaky_appointment_lookup("APT-0008")
    elapsed = time.perf_counter() - t0

    print(f"\n  SUCCESS on attempt {attempt_counter['n']}")
    print(f"  Total elapsed : {elapsed:.3f}s")
    print(f"  Result        : found={result['found']}  status={result['status']}  "
          f"escalate={result['escalate']}")

    assert attempt_counter["n"] == 3, "Expected exactly 3 attempts"
    assert result["found"] is True
    print("  PASS — retry recovered within 3 attempts.")


def demo_retry_exhausted() -> None:
    print()
    print("  Bonus: retry exhausted (max_attempts=2, always fails):")

    @retry_with_backoff(max_attempts=2, initial_delay=0.02, max_delay=0.1)
    def always_fails():
        raise ValueError("Permanent failure")

    try:
        always_fails()
        print("  FAIL — should have raised RetryExhaustedError")
    except RetryExhaustedError as exc:
        print(f"  RetryExhaustedError caught: {exc}")
        print("  PASS — exhaustion raised correctly.")


# ---------------------------------------------------------------------------
# (b) Per-node timeout firing a clean error
# ---------------------------------------------------------------------------

def demo_node_timeout() -> None:
    _section("(b) Per-node timeout — fires clean error, no hang")

    print(f"  Simulated node polls a slow operation for 2.0s; timeout set to 0.3s.")
    print(f"  The node uses node_timeout.check() in its polling loop to detect expiry.")
    print()

    def slow_node_with_check():
        """
        Simulates a node that periodically checks the timeout.
        In production, any node wrapping an external call would poll
        node_timeout.check() between retries or streaming chunks.
        """
        with node_timeout("rag_tool", seconds=0.3) as nt:
            print("  [rag_tool] Starting slow operation (polling every 50ms)...")
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                time.sleep(0.05)      # simulate incremental work
                nt.check()            # raises NodeTimeoutError if timer fired
            print("  [rag_tool] Completed (should not reach here)")

    t0 = time.perf_counter()
    try:
        slow_node_with_check()
    except NodeTimeoutError as exc:
        elapsed = time.perf_counter() - t0
        print(f"  NodeTimeoutError: {exc}")
        print(f"  Elapsed         : {elapsed:.3f}s  (timeout was 0.3s)")
        assert elapsed < 1.0, "Timeout did not fire quickly enough"
        print("  PASS — per-node timeout fired cleanly, no hang.")


# ---------------------------------------------------------------------------
# (c) Global timeout cancelling the whole run
# ---------------------------------------------------------------------------

def demo_global_timeout() -> None:
    _section("(c) Global timeout — cancels entire graph run")

    def slow_agent_run():
        """Simulates a full agent run that takes too long."""
        sid = new_session_id()
        print("  [agent] Starting slow policy query...")
        time.sleep(3.0)   # simulates a hung LLM call
        return run_agent("What is the cancellation policy?", sid, turn=1)

    print(f"  Simulated run sleeps 3.0s; global timeout set to 0.5s.")
    print()

    t0 = time.perf_counter()
    try:
        run_with_global_timeout(slow_agent_run, timeout_seconds=0.5)
        print("  FAIL — should have raised GlobalTimeoutError")

    except GlobalTimeoutError as exc:
        elapsed = time.perf_counter() - t0
        print(f"  GlobalTimeoutError: {exc}")
        print(f"  Elapsed           : {elapsed:.3f}s  (timeout was 0.5s)")
        assert elapsed < 2.0, "Global timeout did not cancel run quickly"
        print("  PASS — global timeout cancelled the run cleanly.")


# ---------------------------------------------------------------------------
# Bonus: apply retry + per-node timeout to a real agent node
# ---------------------------------------------------------------------------

def demo_combined_real_agent() -> None:
    _section("Bonus: real agent run wrapped with global timeout (should complete)")

    sid = new_session_id()
    print("  Running real agent query with 10s global timeout...")

    t0 = time.perf_counter()
    resp = run_with_global_timeout(
        run_agent,
        args=("What is the cancellation policy?", sid),
        kwargs={"turn": 1},
        timeout_seconds=10.0,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.3f}s")
    print(f"  Route  : {resp.route.value}")
    print(f"  Grounded: {resp.grounded}")
    print("  PASS — agent completed well within 10s timeout.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nPracto Agent — Resilience Demonstration (Task 16)")

    demo_retry_recovery()
    demo_retry_exhausted()
    demo_node_timeout()
    demo_global_timeout()
    demo_combined_real_agent()

    print("\n" + "=" * 65)
    print("All resilience demonstrations complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
