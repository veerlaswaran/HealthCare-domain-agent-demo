"""
Task 11 — FastAPI deployment.
Task 12 — Structured logging middleware.

Endpoints:
  POST /ask                      — invoke the agent (session-aware)
  POST /add-document             — add a new KB doc and re-index both collections
  GET  /appointments/{record_id} — direct appointment lookup
  GET  /health                   — liveness + component status

Logging:
  Every request writes one JSON-Lines entry to logs/requests.jsonl via the
  LoggingMiddleware.  The entry contains a trace_id (UUID) for correlation
  with the agent-turn entry in logs/agent.jsonl.
  Only query_sanitised (PII-masked) is logged; query_original never reaches disk.

Run:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import re
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from api.schemas import (
    AddDocumentRequest,
    AddDocumentResponse,
    AppointmentResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrievedChunkOut,
    AppointmentOut,
    GuardrailEventOut,
)
from app.config import (
    CHROMA_DIR,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    KB_DIR,
    MOCK_LLM_MODE,
    OFFLINE_MODE,
)
from app.graph import run_agent
from app.log import new_trace_id, now_iso, write_request
from app.memory import load_history, new_session_id
from app.tools import check_appointment_status


# ---------------------------------------------------------------------------
# Lifespan — warm up embedding function and ChromaDB client on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load embedding function so first request is not slow."""
    from app.rag import _get_ef, _get_client
    _get_ef()
    _get_client()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Practo Domain Support Agent",
    description=(
        "LangGraph-based patient-support agent. "
        "Answers clinic-policy questions via RAG and checks appointment status."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Task 12 — Logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    """
    Log every HTTP request as one JSONL entry.

    The trace_id is injected into request.state so endpoint handlers can
    pass it to run_agent() for cross-log correlation.

    Security: The middleware logs path and method only — no request body.
    The sanitised query is extracted from the response payload by the
    /ask endpoint handler after guardrails have run, then written via
    write_request() which accepts only query_sanitised.
    """
    trace_id = new_trace_id()
    request.state.trace_id = trace_id
    request.state.start_time = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - request.state.start_time) * 1000

    # /ask endpoint writes its own enriched log entry (with query_sanitised).
    # For all other endpoints we write a basic entry here.
    if not request.url.path.startswith("/ask"):
        write_request(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

    return response


# ---------------------------------------------------------------------------
# POST /ask  (Task 11 — main agent endpoint)
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, tags=["Agent"])
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """
    Submit a query to the Practo support agent.

    - Starts a new session if session_id is omitted.
    - Auto-increments turn from persisted history if turn is omitted.
    - Applies PII masking and injection detection before any processing.
    - Routes to RAG (policy questions) or appointment lookup based on intent.
    """
    trace_id: str = getattr(request.state, "trace_id", new_trace_id())
    start = time.perf_counter()

    # Resolve session and turn
    session_id = body.session_id or new_session_id()
    if body.turn is not None:
        turn = body.turn
    else:
        history = load_history(session_id)
        turn = len(history) + 1

    # Invoke agent
    resp = run_agent(
        query=body.query,
        session_id=session_id,
        turn=turn,
        trace_id=trace_id,
    )

    latency_ms = (time.perf_counter() - start) * 1000

    # Write enriched HTTP log entry (sanitised query, route, latency)
    write_request(
        trace_id=trace_id,
        method="POST",
        path="/ask",
        status_code=200,
        latency_ms=latency_ms,
        session_id=session_id,
        route=resp.route.value,
        query_sanitised=resp.query_sanitised,  # PII-masked — safe to log
    )

    return AskResponse(
        trace_id=trace_id,
        session_id=resp.session_id,
        turn=resp.turn,
        route=resp.route.value,
        answer=resp.answer,
        grounded=resp.grounded,
        top_similarity_score=resp.top_similarity_score,
        rag_hits=[
            RetrievedChunkOut(
                source=h.source, score=h.score, snippet=h.snippet
            )
            for h in resp.rag_hits
        ],
        appointment=(
            AppointmentOut(**resp.appointment.model_dump())
            if resp.appointment else None
        ),
        guardrail_events=[
            GuardrailEventOut(event=e.event.value, detail=e.detail)
            for e in resp.guardrail_events
        ],
        latency_ms=round(latency_ms, 2),
    )


# ---------------------------------------------------------------------------
# POST /add-document  (Task 11 — knowledge-base extension endpoint)
# ---------------------------------------------------------------------------

@app.post("/add-document", response_model=AddDocumentResponse, tags=["Knowledge Base"])
async def add_document(body: AddDocumentRequest) -> AddDocumentResponse:
    """
    Add a new policy document to the knowledge base and re-index both
    ChromaDB collections from scratch.

    Re-seeding the full corpus on each add ensures the TF-IDF vocabulary
    (offline mode) stays consistent across all stored embeddings.
    The document is written to data/knowledge_base/ before re-seeding.
    """
    from scripts.seed_kb import (
        fixed_size_chunks, sentence_chunks,
        load_documents, _collect_chunks, index_collection,
    )
    from app.rag import _get_client
    from app.embeddings import reset_embedding_function, get_embedding_function
    import chromadb

    source = body.source_name
    content = body.content.strip()

    # Persist to KB directory first
    filename = f"{source}.txt"
    kb_path = KB_DIR / filename
    kb_path.write_text(f"source: {source}\n\n{content}", encoding="utf-8")

    # Reload all documents (including the new one) and re-seed
    docs = load_documents(KB_DIR)

    fixed_ids, fixed_texts, fixed_metas = _collect_chunks(
        docs, lambda t: fixed_size_chunks(t, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP)
    )
    sent_ids, sent_texts, sent_metas = _collect_chunks(
        docs, lambda t: sentence_chunks(t, group_size=2)
    )
    all_texts = fixed_texts + sent_texts

    # Re-fit embedding function on the expanded corpus
    reset_embedding_function()
    ef = get_embedding_function(corpus=all_texts)

    fixed_embeddings = ef(fixed_texts)
    sent_embeddings = ef(sent_texts)

    client = _get_client()
    n_fixed = index_collection(client, COLLECTION_FIXED,
                               fixed_ids, fixed_texts, fixed_metas, fixed_embeddings)
    n_sent = index_collection(client, COLLECTION_SENTENCE,
                              sent_ids, sent_texts, sent_metas, sent_embeddings)

    # Reset ChromaDB client singleton so next query sees the new collections
    from app import rag as _rag_mod
    _rag_mod._client = None

    return AddDocumentResponse(
        source_name=source,
        chunks_fixed=n_fixed,
        chunks_sentence=n_sent,
        message=(
            f"Document '{source}' added. Full re-index complete: "
            f"{n_fixed} fixed chunks, {n_sent} sentence chunks across {len(docs)} documents."
        ),
    )


# ---------------------------------------------------------------------------
# GET /appointments/{record_id}  (Task 11 — direct lookup endpoint)
# ---------------------------------------------------------------------------

@app.get(
    "/appointments/{record_id}",
    response_model=AppointmentResponse,
    tags=["Appointments"],
)
async def get_appointment(record_id: str) -> AppointmentResponse:
    """
    Direct appointment status lookup by record ID (e.g. APT-0042).
    Returns status, fee, escalation score, and escalate flag.
    No agent graph or LLM call is made — pure dataset lookup.
    """
    result = check_appointment_status(record_id.upper())
    return AppointmentResponse(**result)


# ---------------------------------------------------------------------------
# GET /health  (Task 11 — liveness check)
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    """
    Liveness and component status check.
    Returns ChromaDB collection sizes, KB document count, and mode flags.
    """
    from app.rag import _get_client

    details: dict[str, Any] = {}
    status = "ok"

    # ChromaDB
    fixed_count = 0
    sent_count = 0
    try:
        client = _get_client()
        fixed_count = client.get_collection(COLLECTION_FIXED).count()
        sent_count = client.get_collection(COLLECTION_SENTENCE).count()
    except Exception as exc:
        status = "degraded"
        details["chroma_error"] = str(exc)

    # KB documents
    kb_docs = len(list(KB_DIR.glob("*.txt"))) if KB_DIR.exists() else 0

    return HealthResponse(
        status=status,
        chroma_fixed_count=fixed_count,
        chroma_sentence_count=sent_count,
        kb_documents=kb_docs,
        mock_llm=MOCK_LLM_MODE,
        offline_mode=OFFLINE_MODE,
        details=details,
    )
