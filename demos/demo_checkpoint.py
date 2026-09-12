"""
Task 15 — SQLite-based LangGraph checkpointing.

Demonstrates:
  (a) A run executing guardrail_input + classify_intent (2 of 4 nodes).
  (b) Execution deliberately interrupted before the remaining 2 nodes run
      (rag_tool / appointment_tool + format_response).
  (c) Resuming the SAME thread_id and completing the run, with explicit
      printed proof that the already-completed nodes' results were loaded
      from the checkpoint and NOT re-executed (via execution counters).

Checkpoint DB: data/checkpoints.sqlite

Run:
  python demos/demo_checkpoint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import DATA_DIR
from app.guardrails import run_input_guardrails, InputGuardrailResult
from app.graph import _classify, _route_after_classify
from app.tools import check_appointment_status, retrieve_policy
from app.config import COLLECTION_FIXED

CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"

# ---------------------------------------------------------------------------
# Execution counters — prove nodes not re-run after resume
# ---------------------------------------------------------------------------

_NODE_EXEC_COUNTS: dict[str, int] = {
    "guardrail_input": 0,
    "classify_intent": 0,
    "rag_tool": 0,
    "appointment_tool": 0,
    "format_response": 0,
}


def _tick(node: str) -> None:
    _NODE_EXEC_COUNTS[node] += 1
    print(f"  [EXEC] Node '{node}' executed (total calls: {_NODE_EXEC_COUNTS[node]})")


# ---------------------------------------------------------------------------
# Minimal graph state for checkpoint demo
# ---------------------------------------------------------------------------

class CheckpointState(TypedDict, total=False):
    query: str
    session_id: str
    guardrail_result: Optional[InputGuardrailResult]
    blocked: bool
    intent: str
    record_id: Optional[str]
    rag_result: Optional[dict]
    appointment_result: Optional[dict]
    answer: str


# ---------------------------------------------------------------------------
# Node definitions (instrumented with counters)
# ---------------------------------------------------------------------------

def cp_guardrail_input(state: CheckpointState) -> CheckpointState:
    _tick("guardrail_input")
    result = run_input_guardrails(state["query"])
    return {
        **state,
        "guardrail_result": result,
        "blocked": result.injection_blocked,
    }


def cp_classify_intent(state: CheckpointState) -> CheckpointState:
    _tick("classify_intent")
    if state.get("blocked"):
        return {**state, "intent": "blocked", "record_id": None}
    query = state["guardrail_result"].sanitised_query  # type: ignore[union-attr]
    intent, record_id = _classify(query)
    return {**state, "intent": intent, "record_id": record_id}


def cp_rag_tool(state: CheckpointState) -> CheckpointState:
    _tick("rag_tool")
    query = state["guardrail_result"].sanitised_query  # type: ignore[union-attr]
    result = retrieve_policy(query, collection_name=COLLECTION_FIXED)
    answer = result.get("answer", "No answer found.")
    return {**state, "rag_result": result, "answer": answer}


def cp_appointment_tool(state: CheckpointState) -> CheckpointState:
    _tick("appointment_tool")
    record_id = state.get("record_id")
    if not record_id:
        return {**state, "appointment_result": {"found": False}, "answer": "No record ID."}
    result = check_appointment_status(record_id)
    answer = (
        f"APT {result['record_id']}: {result['status']}, "
        f"fee={result['consultation_fee_inr']}, "
        f"escalate={result['escalate']}"
        if result["found"] else result.get("error", "Not found.")
    )
    return {**state, "appointment_result": result, "answer": answer}


def cp_format_response(state: CheckpointState) -> CheckpointState:
    _tick("format_response")
    answer = state.get("answer", "No answer.")
    print(f"  [format_response] Final answer: {answer[:120]}")
    return {**state, "answer": answer}


def _cp_route(state: CheckpointState) -> str:
    if state.get("blocked"):
        return "format_response"
    return "appointment_tool" if state.get("intent") == "appointment" else "rag_tool"


# ---------------------------------------------------------------------------
# Build checkpointed graph
# ---------------------------------------------------------------------------

def build_checkpointed_graph(checkpointer) -> Any:
    g = StateGraph(CheckpointState)
    g.add_node("guardrail_input", cp_guardrail_input)
    g.add_node("classify_intent", cp_classify_intent)
    g.add_node("rag_tool", cp_rag_tool)
    g.add_node("appointment_tool", cp_appointment_tool)
    g.add_node("format_response", cp_format_response)
    g.set_entry_point("guardrail_input")
    g.add_edge("guardrail_input", "classify_intent")
    g.add_conditional_edges(
        "classify_intent", _cp_route,
        {"rag_tool": "rag_tool", "appointment_tool": "appointment_tool",
         "format_response": "format_response"},
    )
    g.add_edge("rag_tool", "format_response")
    g.add_edge("appointment_tool", "format_response")
    g.add_edge("format_response", END)
    return g.compile(checkpointer=checkpointer, interrupt_after=["classify_intent"])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Task 15 — LangGraph SQLite Checkpointing Demo")
    print(f"Checkpoint DB: {CHECKPOINT_DB}")
    print("=" * 65)

    thread_id = "demo-checkpoint-thread-001"
    config = {"configurable": {"thread_id": thread_id}}
    query = "What is the status of appointment APT-0008?"

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = build_checkpointed_graph(checkpointer)

        # Reset execution counters
        for k in _NODE_EXEC_COUNTS:
            _NODE_EXEC_COUNTS[k] = 0

        # ---------------------------------------------------------------
        # (a) + (b) First run — executes guardrail_input + classify_intent,
        #           then STOPS at interrupt_after=["classify_intent"]
        # ---------------------------------------------------------------
        print(f"\n[Phase A+B] First invoke — thread_id='{thread_id}'")
        print(f"  Query: {query!r}")
        print(f"  interrupt_after=['classify_intent']")
        print()

        state_after_interrupt = graph.invoke(
            {"query": query, "session_id": "demo-session"},
            config=config,
        )

        print()
        print("  Run INTERRUPTED after classify_intent.")
        print(f"  Nodes executed so far: "
              f"{[k for k,v in _NODE_EXEC_COUNTS.items() if v > 0]}")
        print(f"  intent resolved to: {state_after_interrupt.get('intent')!r}")
        print(f"  record_id resolved to: {state_after_interrupt.get('record_id')!r}")
        print(f"  Checkpoint persisted to: {CHECKPOINT_DB}")

        # Snapshot counts after phase 1 — these should NOT increase in phase 2
        counts_after_phase1 = dict(_NODE_EXEC_COUNTS)

        # ---------------------------------------------------------------
        # (c) Resume — same thread_id, no input; loads checkpoint state
        #     and continues from classify_intent onwards
        # ---------------------------------------------------------------
        print(f"\n[Phase C] Resume same thread_id='{thread_id}'")
        print("  Passing None as input — LangGraph loads state from checkpoint.")
        print()

        final_state = graph.invoke(None, config=config)

        print()
        print("  Run COMPLETED.")
        print()
        print("  Execution counts — PROOF that phases 1 nodes NOT re-run:")
        print(f"  {'Node':<22} {'Phase 1 calls':>14} {'Phase 2 calls':>14} {'Re-executed?':>13}")
        print("  " + "-" * 65)
        for node, phase1_count in counts_after_phase1.items():
            phase2_delta = _NODE_EXEC_COUNTS[node] - phase1_count
            re_run = "YES (unexpected)" if phase1_count > 0 and phase2_delta > 0 else (
                     "no" if phase1_count > 0 else "-")
            print(f"  {node:<22} {phase1_count:>14} {phase2_delta:>14} {re_run:>13}")

        print()
        print(f"  Final answer: {final_state.get('answer', 'n/a')[:120]}")
        print()

        # Verify checkpoint was used
        assert counts_after_phase1["guardrail_input"] == 1, "guardrail_input ran in phase 1"
        assert counts_after_phase1["classify_intent"] == 1, "classify_intent ran in phase 1"
        assert _NODE_EXEC_COUNTS["guardrail_input"] == 1, "guardrail_input NOT re-run in phase 2"
        assert _NODE_EXEC_COUNTS["classify_intent"] == 1, "classify_intent NOT re-run in phase 2"
        assert _NODE_EXEC_COUNTS["format_response"] == 1, "format_response ran in phase 2"
        print("  All assertions PASSED — checkpointing works correctly.")

    print("=" * 65)


if __name__ == "__main__":
    main()
