"""
Task 10 — Input and output guardrails.

Input-side:
  1. PII masking  — masks Indian mobile phone numbers (10-digit, optionally
                    prefixed with +91 or 0).  This is the fixed-format PII
                    field identified in the brief.  Demonstrated to fire on
                    a deliberate test case in demos/demo_guardrails.py.

  2. Prompt-injection detection — blocks queries that attempt to override
                    the system prompt, extract instructions, or role-play as
                    the model.  Uses a pattern set + scoring heuristic.

Output-side:
  3. Groundedness check — refuses the answer and returns a fallback when
                    the RAG retrieval similarity is below threshold (already
                    enforced inside grounded_answer(); this module provides
                    the structured event trace used by format_response).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# PII masking — Indian mobile phone numbers
# ---------------------------------------------------------------------------
# Format: optional country code (+91 / 0), then 10 digits starting with
# 6-9 (all valid Indian mobile prefixes).  Allows optional spaces or hyphens
# between digit groups.

_PHONE_PATTERN = re.compile(
    r"""
    (?:                        # optional country-code prefix
        (?:\+91|0091|0)        # +91 | 0091 | leading 0
        [\s\-]?               # optional separator after code
    )?
    (?<![\d])                 # not immediately preceded by digit
    [6-9]\d{9}               # 10-digit mobile: first digit 6-9 then 9 more
    (?![\d])                  # not immediately followed by digit
    |                         # OR spaced/hyphenated variant
    (?:                       # optional country-code prefix
        (?:\+91|0091|0)
        [\s\-]?
    )?
    (?<![\d])
    [6-9]\d{2}               # first 3 digits
    [\s\-]
    \d{3}[\s\-]?\d{4}       # next 3 then last 4
    (?![\d])
    |                         # OR 5+5 split
    (?:
        (?:\+91|0091|0)
        [\s\-]?
    )?
    (?<![\d])
    [6-9]\d{4}               # first 5 digits
    [\s\-]
    \d{5}                    # last 5 digits
    (?![\d])
    |                         # OR 0XX-XXX-XXXX (leading 0 attached to first group)
    0[6-9]\d                 # 0 + first 2 mobile digits = 3 chars
    [\s\-]
    \d{3}                    # next 3
    [\s\-]
    \d{5}                    # last 5
    (?![\d])
    """,
    re.VERBOSE,
)

PHONE_MASK = "[PHONE_REDACTED]"


def mask_pii(text: str) -> tuple[str, bool]:
    """
    Mask Indian mobile phone numbers in `text`.

    Returns:
        (masked_text, was_masked)  — was_masked=True if at least one number
        was found and replaced.
    """
    masked, count = _PHONE_PATTERN.subn(PHONE_MASK, text)
    return masked, count > 0


# ---------------------------------------------------------------------------
# Prompt-injection detection
# ---------------------------------------------------------------------------
# Patterns that indicate attempts to override, leak, or hijack the system
# prompt.  Each pattern is scored; total score >= INJECTION_THRESHOLD => block.

@dataclass
class _InjectionPattern:
    pattern: re.Pattern
    score: float
    description: str


_INJECTION_PATTERNS: list[_InjectionPattern] = [
    _InjectionPattern(
        re.compile(r"\bignore\b.{0,60}\b(instruction|prompt|rule|context|above|previous|prior)\b", re.I | re.S),
        score=1.0, description="ignore-instructions override",
    ),
    _InjectionPattern(
        re.compile(r"\bforget\b.{0,60}\b(instruction|prompt|rule|context|everything)\b", re.I | re.S),
        score=1.0, description="forget-instructions override",
    ),
    _InjectionPattern(
        re.compile(r"\byou are now\b|\bact as\b|\bpretend (you are|to be)\b|\brole.?play\b", re.I),
        score=0.8, description="role-play / persona hijack",
    ),
    _InjectionPattern(
        re.compile(r"\brepeat after me\b|\bsay exactly\b|\bprint your (system|original) prompt\b", re.I),
        score=1.0, description="output-echo / leak attempt",
    ),
    _InjectionPattern(
        re.compile(r"\bdisregard\b.{0,20}\b(safety|guideline|restriction|filter)\b", re.I),
        score=1.0, description="safety-filter bypass",
    ),
    _InjectionPattern(
        re.compile(r"\bDAN\b|do anything now|jailbreak|bypass (filter|restriction|moderation)", re.I),
        score=1.0, description="DAN / jailbreak pattern",
    ),
    _InjectionPattern(
        re.compile(r"system\s*:\s*you|<\s*system\s*>|\[\s*system\s*\]", re.I),
        score=0.9, description="fake system-role injection",
    ),
    _InjectionPattern(
        re.compile(r"\bwhat (are|were) your instructions\b|\bshow me your prompt\b", re.I),
        score=0.8, description="prompt-extraction attempt",
    ),
]

INJECTION_THRESHOLD: float = 0.8  # score >= this => blocked


@dataclass
class InjectionResult:
    blocked: bool
    score: float
    triggers: list[str] = field(default_factory=list)


def detect_injection(text: str) -> InjectionResult:
    """
    Score the query for prompt-injection signals.

    Returns InjectionResult with blocked=True if total score >= threshold.
    Score is capped at 1.0 (multiple triggers don't stack beyond 1).
    """
    total_score = 0.0
    triggers: list[str] = []

    for ip in _INJECTION_PATTERNS:
        if ip.pattern.search(text):
            total_score += ip.score
            triggers.append(ip.description)

    total_score = min(total_score, 1.0)
    return InjectionResult(
        blocked=total_score >= INJECTION_THRESHOLD,
        score=round(total_score, 3),
        triggers=triggers,
    )


# ---------------------------------------------------------------------------
# Output-side groundedness check
# ---------------------------------------------------------------------------

def check_groundedness(top_score: float, domain_gate_passed: bool) -> tuple[bool, str]:
    """
    Determine whether the RAG output is grounded enough to return to the user.

    This is a post-retrieval check used in format_response to add the
    groundedness event to the trace.  The actual answer/fallback decision
    is already made inside grounded_answer(); this function provides the
    structured audit event.

    Args:
        top_score         : best cosine similarity from retrieval
        domain_gate_passed: whether the domain keyword gate passed

    Returns:
        (is_grounded, detail_message)
    """
    from app.config import SIMILARITY_THRESHOLD

    if not domain_gate_passed:
        return False, f"Domain gate failed — no medical/scheduling keywords in query"
    if top_score < SIMILARITY_THRESHOLD:
        return False, (
            f"Similarity score {top_score:.4f} is below threshold "
            f"{SIMILARITY_THRESHOLD} — answer withheld"
        )
    return True, f"Grounded (top_score={top_score:.4f} >= {SIMILARITY_THRESHOLD})"


# ---------------------------------------------------------------------------
# Composite input guardrail — single entry point used by the graph
# ---------------------------------------------------------------------------

@dataclass
class InputGuardrailResult:
    original_query: str
    sanitised_query: str
    pii_masked: bool
    injection_blocked: bool
    injection_score: float
    injection_triggers: list[str]
    passed: bool          # True → proceed to agent; False → block immediately


def run_input_guardrails(query: str) -> InputGuardrailResult:
    """
    Apply all input-side guardrails in sequence:
      1. PII masking (always applied; does not block)
      2. Injection detection (blocks if score >= threshold)

    Returns InputGuardrailResult; caller checks .passed before continuing.
    """
    # Step 1 — mask PII
    sanitised, was_masked = mask_pii(query)

    # Step 2 — injection detection (run on sanitised text)
    inj = detect_injection(sanitised)

    return InputGuardrailResult(
        original_query=query,
        sanitised_query=sanitised,
        pii_masked=was_masked,
        injection_blocked=inj.blocked,
        injection_score=inj.score,
        injection_triggers=inj.triggers,
        passed=not inj.blocked,
    )
