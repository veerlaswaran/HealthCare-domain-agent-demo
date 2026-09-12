"""
Task 7 — LangGraph agent graph.

Graph topology (4 nodes, 1 conditional edge):

    [guardrail_input]
          |
          v
    [classify_intent]  --conditional--> [appointment_tool]
          |                                     |
          v (default)                           |
      [rag_tool]                                |
          |                                     |
          +---------------+---------------------+
                          |
                          v
                  [format_response]

Nodes:
  guardrail_input   — run input guardrails (PII mask + injection detect)
  classify_intent   — decide route: "rag" or "appointment"
  rag_tool          — call retrieve_policy() and ground the answer
  appointment_tool  — call check_appointment_status()
  format_response   — build + validate AgentResponse, persist memory, log

Conditional edge:
  classify_intent → appointment_tool  if intent == "appointment"
  classify_intent → rag_tool          otherwise

State (TypedDict):
  All fields are optional so each node only writes what it produces.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from app.config import COLLECTION_FIXED
from app.guardrails import (
    InputGuardrailResult,
    check_groundedness,
    run_input_guardrails,
)
from app.models import GuardrailEvent
from app.log import new_trace_id, now_iso, write_turn
from app.memory import (
    get_recent_history,
    history_summary_lines,
    save_turn,
)
from app.models import (
    AgentResponse,
    AppointmentDetail,
    GuardrailTrace,
    RetrievedChunk,
    RouteType,
    validate_response,
)
from app.tools import (
    ESCALATION_THRESHOLD,
    check_appointment_status,
    retrieve_policy,
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Input
    session_id: str
    turn: int
    query_original: str
    trace_id: str           # per-request UUID for log correlation

    # After guardrail node
    guardrail_result: InputGuardrailResult
    guardrail_events: list[GuardrailTrace]
    blocked: bool               # True → skip tools, go straight to format

    # After classify node
    intent: str                 # "rag" | "appointment"
    record_id: Optional[str]    # populated when intent == "appointment"

    # After tool nodes
    rag_result: Optional[dict]
    appointment_result: Optional[dict]

    # After format node
    response: AgentResponse


# ---------------------------------------------------------------------------
# Intent classification helpers
# ---------------------------------------------------------------------------

_APT_ID_RE = re.compile(r"\bAPT-\d{4}\b", re.I)

# Keywords that signal lookup/status intent (NOT policy questions)
# These must co-occur to trigger appointment routing without an explicit ID.
_LOOKUP_ACTION_KEYWORDS = frozenset([
    "status", "check", "look up", "lookup", "find", "show me", "what is",
    "escalat", "track", "where is", "update on",
])
_APPOINTMENT_OBJECT_KEYWORDS = frozenset([
    "booking", "booked", "scheduled", "no-show", "completed",
    "my appointment", "my booking",
])


def _classify(query: str) -> tuple[str, Optional[str]]:
    """
    Return (intent, record_id).

    Rules (applied in order):
      1. If the query contains an APT-XXXX pattern → "appointment"
      2. If the query contains ≥1 lookup-action keyword AND ≥1 appointment-
         object keyword → "appointment" (e.g. "check my booking status")
      3. Otherwise → "rag"  (policy questions fall here even if they mention
         'appointment' in passing, e.g. "cancel appointment policy")
    """
    q_lower = query.lower()
    apt_match = _APT_ID_RE.search(query)
    if apt_match:
        return "appointment", apt_match.group(0).upper()

    has_action = any(kw in q_lower for kw in _LOOKUP_ACTION_KEYWORDS)
    has_object = any(kw in q_lower for kw in _APPOINTMENT_OBJECT_KEYWORDS)
    if has_action and has_object:
        return "appointment", None

    return "rag", None


# ---------------------------------------------------------------------------
# Node 1 — guardrail_input
# ---------------------------------------------------------------------------

def guardrail_input(state: AgentState) -> AgentState:
    query = state["query_original"]
    result = run_input_guardrails(query)

    events: list[GuardrailTrace] = []
    if result.pii_masked:
        events.append(GuardrailTrace(
            event=GuardrailEvent.PII_MASKED,
            detail=f"Phone number(s) masked in query",
        ))
    if result.injection_blocked:
        events.append(GuardrailTrace(
            event=GuardrailEvent.INJECTION_BLOCKED,
            detail=f"Score={result.injection_score:.2f}; triggers={result.injection_triggers}",
        ))

    return {
        **state,
        "guardrail_result": result,
        "guardrail_events": events,
        "blocked": result.injection_blocked,
    }


# ---------------------------------------------------------------------------
# Node 2 — classify_intent
# ---------------------------------------------------------------------------

def classify_intent(state: AgentState) -> AgentState:
    if state.get("blocked"):
        return {**state, "intent": "blocked", "record_id": None}

    query = state["guardrail_result"].sanitised_query
    intent, record_id = _classify(query)
    return {**state, "intent": intent, "record_id": record_id}


# ---------------------------------------------------------------------------
# Node 3a — rag_tool
# ---------------------------------------------------------------------------

def rag_tool(state: AgentState) -> AgentState:
    query = state["guardrail_result"].sanitised_query
    result = retrieve_policy(query, collection_name=COLLECTION_FIXED)

    events: list[GuardrailTrace] = list(state.get("guardrail_events", []))
    is_grounded, detail = check_groundedness(
        top_score=result.get("top_score", 0.0),
        domain_gate_passed=result.get("grounded", False),
    )
    if not is_grounded:
        events.append(GuardrailTrace(
            event=GuardrailEvent.UNGROUNDED_BLOCKED,
            detail=detail,
        ))

    return {
        **state,
        "rag_result": result,
        "guardrail_events": events,
    }


# ---------------------------------------------------------------------------
# Node 3b — appointment_tool
# ---------------------------------------------------------------------------

def appointment_tool(state: AgentState) -> AgentState:
    record_id = state.get("record_id")

    if record_id is None:
        # No APT-XXXX found — ask user to supply one
        return {
            **state,
            "appointment_result": {
                "found": False,
                "record_id": None,
                "error": "Please provide an appointment ID (e.g. APT-0042) to check status.",
            },
        }

    result = check_appointment_status(record_id)
    return {**state, "appointment_result": result}


# ---------------------------------------------------------------------------
# Node 4 — format_response
# ---------------------------------------------------------------------------

def _build_answer(state: AgentState) -> str:
    """Compose the natural-language answer from tool outputs."""
    intent = state.get("intent", "rag")

    # Blocked by injection guardrail
    if state.get("blocked"):
        return (
            "Your request has been blocked. It appears to contain content "
            "that attempts to override or manipulate the assistant's behaviour. "
            "Please rephrase your question."
        )

    # Appointment route
    if intent == "appointment":
        apt = state.get("appointment_result") or {}
        if not apt.get("found"):
            return apt.get("error") or "Appointment not found."

        esc_flag = " Escalation is recommended." if apt.get("escalate") else ""
        follow_note = (
            " A follow-up visit is required." if apt.get("follow_up_required") else ""
        )
        return (
            f"Appointment {apt['record_id']}: "
            f"Status is {apt['status']}, "
            f"Category: {apt['category']}, "
            f"Consultation fee: INR {apt['consultation_fee_inr']}, "
            f"Created {apt['days_since_created']} day(s) ago."
            f"{follow_note}"
            f" Escalation score: {apt['escalation_score']:.2f}."
            f"{esc_flag}"
        )

    # RAG route
    rag = state.get("rag_result") or {}
    return rag.get("answer") or "I could not find an answer in the knowledge base."


def format_response(state: AgentState) -> AgentState:
    session_id = state["session_id"]
    turn = state.get("turn", 1)
    ts = now_iso()
    intent = state.get("intent", "rag")
    guardrail_result = state.get("guardrail_result")

    # Determine route enum
    if state.get("blocked"):
        route = RouteType.GUARDRAIL_BLOCKED
    elif intent == "appointment":
        route = RouteType.APPOINTMENT
    else:
        rag = state.get("rag_result") or {}
        route = RouteType.RAG if rag.get("grounded") else RouteType.UNGROUNDED

    answer = _build_answer(state)

    # RAG fields
    rag_hits: list[RetrievedChunk] = []
    top_sim: Optional[float] = None
    grounded: Optional[bool] = None
    if intent == "rag" and not state.get("blocked"):
        rag = state.get("rag_result") or {}
        top_sim = rag.get("top_score")
        grounded = rag.get("grounded")
        for h in (rag.get("hits") or []):
            rag_hits.append(RetrievedChunk(
                source=h["source"],
                score=h["score"],
                snippet=h["text"][:120],
            ))

    # Appointment field
    apt_detail: Optional[AppointmentDetail] = None
    if intent == "appointment" and not state.get("blocked"):
        apt = state.get("appointment_result") or {}
        if apt.get("found"):
            apt_detail = AppointmentDetail(
                record_id=apt["record_id"],
                category=apt["category"],
                status=apt["status"],
                consultation_fee_inr=apt["consultation_fee_inr"],
                days_since_created=apt["days_since_created"],
                follow_up_required=apt["follow_up_required"],
                escalation_score=apt["escalation_score"],
                escalate=apt["escalate"],
            )

    # Guardrail events — add CLEAN if none fired
    events = list(state.get("guardrail_events") or [])
    if not events:
        events.append(GuardrailTrace(event=GuardrailEvent.CLEAN, detail="All checks passed"))

    # History
    history_lines = history_summary_lines(session_id)

    # Build raw dict and validate
    raw: dict[str, Any] = {
        "session_id": session_id,
        "turn": turn,
        "timestamp": ts,
        "route": route.value,
        "query_original": state.get("query_original", ""),
        "query_sanitised": guardrail_result.sanitised_query if guardrail_result else state.get("query_original", ""),
        "answer": answer,
        "rag_hits": [h.model_dump() for h in rag_hits],
        "top_similarity_score": top_sim,
        "grounded": grounded,
        "appointment": apt_detail.model_dump() if apt_detail else None,
        "guardrail_events": [e.model_dump() for e in events],
        "history_summary": history_lines,
    }

    try:
        validated: AgentResponse = validate_response(raw)
    except ValidationError as exc:
        # Schema violation — surface as a blocked response rather than crash
        validated = AgentResponse(
            session_id=session_id,
            turn=turn,
            timestamp=ts,
            route=RouteType.GUARDRAIL_BLOCKED,
            query_original=state.get("query_original", ""),
            query_sanitised=state.get("query_original", ""),
            answer=f"Internal schema validation error: {exc}",
            guardrail_events=[GuardrailTrace(
                event=GuardrailEvent.INJECTION_BLOCKED,
                detail=f"ValidationError: {exc}",
            )],
        )

    # Persist turn to memory
    save_turn(session_id, {
        "turn": turn,
        "query": validated.query_sanitised,
        "answer": validated.answer[:200],
        "route": validated.route.value,
        "timestamp": ts,
    })

    # Write JSONL log (pass trace_id for HTTP correlation)
    write_turn(validated.model_dump(), trace_id=state.get("trace_id"))

    return {**state, "response": validated}


# ---------------------------------------------------------------------------
# Conditional edge router
# ---------------------------------------------------------------------------

def _route_after_classify(state: AgentState) -> str:
    """Return the name of the next node after classify_intent."""
    if state.get("blocked"):
        return "format_response"
    if state.get("intent") == "appointment":
        return "appointment_tool"
    return "rag_tool"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("guardrail_input", guardrail_input)
    g.add_node("classify_intent", classify_intent)
    g.add_node("rag_tool", rag_tool)
    g.add_node("appointment_tool", appointment_tool)
    g.add_node("format_response", format_response)

    g.set_entry_point("guardrail_input")
    g.add_edge("guardrail_input", "classify_intent")

    # Genuine conditional edge: routes to rag_tool OR appointment_tool
    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "rag_tool": "rag_tool",
            "appointment_tool": "appointment_tool",
            "format_response": "format_response",  # injection blocked
        },
    )

    g.add_edge("rag_tool", "format_response")
    g.add_edge("appointment_tool", "format_response")
    g.add_edge("format_response", END)

    return g


# Compiled graph — import and call .invoke() from demos and API
agent_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# Public invoke helper
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    session_id: str,
    turn: int = 1,
    trace_id: Optional[str] = None,
) -> AgentResponse:
    """
    Invoke the agent graph for one turn.

    Args:
        query      : raw user query
        session_id : conversation session (use memory.new_session_id())
        turn       : 1-indexed turn number within session
        trace_id   : optional per-request UUID (generated if not supplied)

    Returns:
        Validated AgentResponse
    """
    initial_state: AgentState = {
        "query_original": query,
        "session_id": session_id,
        "turn": turn,
        "trace_id": trace_id or new_trace_id(),
    }
    final_state = agent_graph.invoke(initial_state)
    return final_state["response"]
