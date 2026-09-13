"""
Task 9 — Structured output schema.

Every agent response is validated against AgentResponse before being
returned to the caller.  Validation is enforced in graph.py's
format_response node.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class RouteType(str, Enum):
    CONVERSATIONAL = "conversational"
    RAG = "rag"
    APPOINTMENT = "appointment"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    UNGROUNDED = "ungrounded"


class GuardrailEvent(str, Enum):
    PII_MASKED = "pii_masked"
    INJECTION_BLOCKED = "injection_blocked"
    UNGROUNDED_BLOCKED = "ungrounded_blocked"
    CLEAN = "clean"


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    source: str
    score: float = Field(ge=0.0, le=1.0)
    snippet: str  # first 120 chars of chunk text


class AppointmentDetail(BaseModel):
    record_id: str
    category: str
    status: str
    consultation_fee_inr: int = Field(gt=0)
    days_since_created: int = Field(ge=0, le=30)
    follow_up_required: bool
    escalation_score: float = Field(ge=0.0, le=1.0)
    escalate: bool


class GuardrailTrace(BaseModel):
    event: GuardrailEvent
    detail: str = ""


# ---------------------------------------------------------------------------
# Top-level response schema
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """
    Every response produced by the Practo agent must conform to this schema.
    Validated in format_response node before being returned to the caller.
    """

    session_id: str = Field(description="Conversation session identifier")
    turn: int = Field(ge=1, description="Turn number within session (1-indexed)")
    timestamp: str = Field(description="ISO-8601 UTC timestamp")
    route: RouteType = Field(description="Which tool branch handled the query")
    query_original: str = Field(description="Raw user query before any masking")
    query_sanitised: str = Field(description="Query after PII masking (may equal original)")
    answer: str = Field(description="Agent's natural-language answer to the user")

    # Optional — populated on RAG route
    rag_hits: List[RetrievedChunk] = Field(default_factory=list)
    top_similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    grounded: Optional[bool] = None

    # Optional — populated on appointment route
    appointment: Optional[AppointmentDetail] = None

    # Guardrail trace — always present
    guardrail_events: List[GuardrailTrace] = Field(default_factory=list)

    # Conversation context carried forward (last N turns)
    history_summary: List[str] = Field(
        default_factory=list,
        description="Brief summary of prior turns in this session",
    )

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @model_validator(mode="after")
    def _route_consistency(self) -> "AgentResponse":
        if self.route == RouteType.RAG and self.appointment is not None:
            raise ValueError("RAG route must not populate appointment field")
        if self.route == RouteType.APPOINTMENT and self.rag_hits:
            raise ValueError("APPOINTMENT route must not populate rag_hits")
        return self

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResponse":
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Validation helper used by graph nodes
# ---------------------------------------------------------------------------

def validate_response(data: dict[str, Any]) -> AgentResponse:
    """
    Validate a raw dict against AgentResponse.
    Raises pydantic.ValidationError on schema violation (never silently passes).
    """
    return AgentResponse.model_validate(data)
