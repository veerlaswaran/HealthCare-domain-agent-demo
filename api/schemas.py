"""
FastAPI request / response Pydantic models — kept separate from app/models.py
so the API surface can evolve independently of the agent's internal schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(
        description="Patient's natural-language question or appointment ID query.",
        min_length=1,
        max_length=2000,
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Existing session UUID for conversation continuity. "
            "Omit or pass null to start a fresh session."
        ),
    )
    turn: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Turn number within the session. If omitted the API auto-derives "
            "it from the session's persisted history length + 1."
        ),
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "query": "What is the cancellation policy for appointments?",
                "session_id": None,
            },
            {
                "query": "What is the status of appointment APT-0042?",
                "session_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "turn": 2,
            },
        ]
    }}


class RetrievedChunkOut(BaseModel):
    source: str
    score: float
    snippet: str


class AppointmentOut(BaseModel):
    record_id: str
    category: str
    status: str
    consultation_fee_inr: int
    days_since_created: int
    follow_up_required: bool
    escalation_score: float
    escalate: bool


class GuardrailEventOut(BaseModel):
    event: str
    detail: str


class AskResponse(BaseModel):
    trace_id: str = Field(description="Per-request UUID for log correlation.")
    session_id: str
    turn: int
    route: str
    answer: str
    grounded: Optional[bool] = None
    top_similarity_score: Optional[float] = None
    rag_hits: List[RetrievedChunkOut] = Field(default_factory=list)
    appointment: Optional[AppointmentOut] = None
    guardrail_events: List[GuardrailEventOut] = Field(default_factory=list)
    latency_ms: float = Field(description="End-to-end request latency in milliseconds.")


# ---------------------------------------------------------------------------
# POST /add-document
# ---------------------------------------------------------------------------

class AddDocumentRequest(BaseModel):
    source_name: str = Field(
        description="Unique snake_case identifier for the document (e.g. 'new_referral_policy').",
        min_length=1,
        max_length=120,
        pattern=r"^[a-z0-9_]+$",
    )
    content: str = Field(
        description="Full document text (≥2 sentences).",
        min_length=20,
        max_length=10000,
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "source_name": "referral_policy",
            "content": (
                "Patients requiring specialist referrals must first consult a "
                "General Medicine doctor at a Practo clinic. The referral letter "
                "is valid for 60 days from the date of issue."
            ),
        }]
    }}


class AddDocumentResponse(BaseModel):
    source_name: str
    chunks_fixed: int = Field(description="Chunks added to policy_fixed collection.")
    chunks_sentence: int = Field(description="Chunks added to policy_sentence collection.")
    message: str


# ---------------------------------------------------------------------------
# GET /appointments/{record_id}
# ---------------------------------------------------------------------------

class AppointmentResponse(BaseModel):
    found: bool
    record_id: str
    category: Optional[str] = None
    status: Optional[str] = None
    consultation_fee_inr: Optional[int] = None
    days_since_created: Optional[int] = None
    follow_up_required: Optional[bool] = None
    escalation_score: Optional[float] = None
    escalate: Optional[bool] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str                          # "ok" | "degraded"
    chroma_fixed_count: int
    chroma_sentence_count: int
    kb_documents: int
    mock_llm: bool
    offline_mode: bool
    details: Dict[str, Any] = Field(default_factory=dict)
