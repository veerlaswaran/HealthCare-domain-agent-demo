"""
Task 12 — Structured JSONL logging.

Every HTTP request and every agent turn is logged as one JSON-Lines entry.

Security guarantee:
  - query_sanitised (PII-masked) is logged, never query_original.
  - Fixed-format PII (phone numbers) is stripped by app/guardrails.py before
    the text reaches this module. The masking is applied in the guardrail_input
    node, which runs before any tool call or log write, so the raw phone number
    never touches disk.

Log files:
  logs/agent.jsonl    — one line per agent turn (from graph.py)
  logs/requests.jsonl — one line per HTTP request (from FastAPI middleware)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import LOGS_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AGENT_LOG: Path = LOGS_DIR / "agent.jsonl"
REQUEST_LOG: Path = LOGS_DIR / "requests.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    """Generate a fresh UUID4 trace identifier for log correlation."""
    return str(uuid.uuid4())


def _append(path: Path, entry: dict[str, Any]) -> None:
    """Atomically append one JSON line to a log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Agent turn logger (called from graph.py format_response node)
# ---------------------------------------------------------------------------

def write_turn(response_dict: dict[str, Any], trace_id: Optional[str] = None) -> None:
    """
    Append one JSONL entry per agent turn.

    Only logs query_sanitised — the PII-masked version — never query_original.
    This ensures fixed-format PII (phone numbers) never reaches disk in clear.
    """
    entry: dict[str, Any] = {
        "log_type": "agent_turn",
        "trace_id": trace_id or new_trace_id(),
        "timestamp": response_dict.get("timestamp"),
        "session_id": response_dict.get("session_id"),
        "turn": response_dict.get("turn"),
        "route": response_dict.get("route"),
        # SECURITY: log sanitised query only — PII already masked by guardrail_input
        "query_sanitised": response_dict.get("query_sanitised", "")[:200],
        "top_similarity_score": response_dict.get("top_similarity_score"),
        "grounded": response_dict.get("grounded"),
        "escalation_score": (
            (response_dict.get("appointment") or {}).get("escalation_score")
        ),
        "escalate": (
            (response_dict.get("appointment") or {}).get("escalate")
        ),
        "guardrail_events": [
            e.get("event") if isinstance(e, dict) else str(e)
            for e in response_dict.get("guardrail_events", [])
        ],
    }
    _append(AGENT_LOG, entry)


# ---------------------------------------------------------------------------
# HTTP request logger (called from FastAPI middleware)
# ---------------------------------------------------------------------------

def write_request(
    trace_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    session_id: Optional[str] = None,
    route: Optional[str] = None,
    # SECURITY: only the sanitised query is accepted here
    query_sanitised: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Append one JSONL entry per HTTP request.

    The caller is responsible for passing query_sanitised (not the raw query).
    The FastAPI endpoint extracts the sanitised form from the AgentResponse
    after the guardrail node has run.
    """
    entry: dict[str, Any] = {
        "log_type": "http_request",
        "trace_id": trace_id,
        "timestamp": now_iso(),
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "session_id": session_id,
        "route": route,
        # SECURITY: PII-masked query only
        "query_sanitised": (query_sanitised or "")[:200],
    }
    if extra:
        entry.update(extra)
    _append(REQUEST_LOG, entry)
