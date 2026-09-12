"""
Task 8 — Persisted conversation memory.

Conversation history is stored as a JSON file under:
    logs/memory/<session_id>.json

Each session file contains an ordered list of turn records.  On every agent
call the history is loaded, the new turn is appended, and the file is
atomically rewritten.

Design decisions:
  - One file per session — keeps sessions isolated and makes replay trivial.
  - Atomic write (write to .tmp then rename) — prevents partial writes from
    corrupting history on crash.
  - No in-process cache — each call reads from disk so the store is correct
    even when multiple processes share a session (e.g. FastAPI workers).
  - history_window controls how many prior turns are included in the prompt
    context; older turns are retained on disk but not surfaced to the agent.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from app.config import LOGS_DIR

MEMORY_DIR: Path = LOGS_DIR / "memory"
HISTORY_WINDOW: int = 10  # turns surfaced to agent context


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    """Generate a fresh UUID4 session identifier."""
    return str(uuid.uuid4())


def _session_path(session_id: str) -> Path:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return MEMORY_DIR / f"{session_id}.json"


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load_history(session_id: str) -> list[dict[str, Any]]:
    """
    Load the full turn history for a session.
    Returns an empty list if the session does not exist yet.
    """
    path = _session_path(session_id)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_turn(session_id: str, turn_record: dict[str, Any]) -> None:
    """
    Append a turn record to the session history and persist atomically.

    turn_record should contain at minimum:
        turn      : int
        query     : str (sanitised)
        answer    : str
        route     : str
        timestamp : str
    """
    path = _session_path(session_id)
    history = load_history(session_id)
    history.append(turn_record)

    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)  # atomic on POSIX and Windows


def get_recent_history(session_id: str, window: int = HISTORY_WINDOW) -> list[dict[str, Any]]:
    """Return the most recent `window` turns for context injection."""
    return load_history(session_id)[-window:]


def history_summary_lines(session_id: str, window: int = HISTORY_WINDOW) -> list[str]:
    """
    Return a compact list of strings summarising recent turns.
    Included in AgentResponse.history_summary for schema compliance.

    Format:  "T<n> [<route>]: <query_snippet> -> <answer_snippet>"
    """
    turns = get_recent_history(session_id, window=window)
    lines: list[str] = []
    for t in turns:
        q_snip = t.get("query", "")[:60]
        a_snip = t.get("answer", "")[:80].replace("\n", " ")
        route = t.get("route", "?")
        turn_n = t.get("turn", "?")
        lines.append(f"T{turn_n} [{route}]: {q_snip!r} -> {a_snip!r}")
    return lines


def clear_session(session_id: str) -> None:
    """Delete a session file (used in tests / fresh-conversation demo)."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()


def list_sessions() -> list[str]:
    """Return all known session IDs."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return [p.stem for p in MEMORY_DIR.glob("*.json")]
