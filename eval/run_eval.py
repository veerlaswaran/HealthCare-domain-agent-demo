"""
Task 13 — RAG triad evaluation at scale (MOCK_LLM judge).

Test set: 15 queries
  - 13 in-scope queries covering all 12 required KB topics (one topic has 2 queries)
  - 2 deliberately out-of-scope / edge-case queries

RAG triad scores (each 0.0, 0.5, or 1.0):

  context_relevance:
    Measures whether the retrieved chunks are relevant to the query.
    Judge rule: check if any query keyword appears in the retrieved chunk text.
      1.0 — top chunk contains ≥3 query keywords
      0.5 — top chunk contains 1-2 query keywords
      0.0 — top chunk contains 0 query keywords OR no chunks retrieved

  groundedness:
    Measures whether the answer is supported by the retrieved context.
    Judge rule: check if key answer phrases appear in the chunk text.
      1.0 — answer contains ≥2 phrases that also appear in retrieved chunks
      0.5 — answer contains exactly 1 phrase from retrieved chunks
      0.0 — answer contains no phrases from retrieved chunks, OR answer is
            the fallback "I don't know" string (ungrounded by definition)

  answer_relevance:
    Measures whether the answer addresses the query.
    Judge rule: check if query keywords appear in the answer text.
      1.0 — answer contains ≥3 query keywords
      0.5 — answer contains 1-2 query keywords
      0.0 — answer contains 0 query keywords (off-topic or fallback)

All three judges are deterministic (no randomness, no network) — they run
identically under MOCK_LLM because they operate on text overlap, not on
any LLM API call.  The "LLM-as-judge prompt" is expressed as a structured
scoring function that mimics what a prompted judge would check.

Results are written to logs/eval_results.jsonl and printed as a table.

Run:  python eval/run_eval.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import LOGS_DIR
from app.graph import run_agent
from app.memory import new_session_id
from app.rag import retrieve

EVAL_LOG: Path = LOGS_DIR / "eval_results.jsonl"

# ---------------------------------------------------------------------------
# Test set — 15 queries
# ---------------------------------------------------------------------------

TEST_CASES: list[dict] = [
    # 1. appointment_booking_policy
    {
        "id": "Q01",
        "topic": "appointment_booking_policy",
        "query": "How can I book a specialist appointment on Practo?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 2. cancellation_rescheduling_policy
    {
        "id": "Q02",
        "topic": "cancellation_rescheduling_policy",
        "query": "What is the cancellation fee if I cancel within 4 hours?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 3. consultation_fee_structure (2 queries for this topic)
    {
        "id": "Q03",
        "topic": "consultation_fee_structure",
        "query": "What is the consultation fee for a Cardiology visit?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    {
        "id": "Q04",
        "topic": "consultation_fee_structure",
        "query": "Do senior citizens get a discount on consultation fees?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 4. insurance_claim_process
    {
        "id": "Q05",
        "topic": "insurance_claim_process",
        "query": "How do I make a cashless insurance claim at the clinic?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 5. prescription_refill_policy
    {
        "id": "Q06",
        "topic": "prescription_refill_policy",
        "query": "Can I get a prescription refill for my blood pressure medication online?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 6. lab_test_turnaround_times
    {
        "id": "Q07",
        "topic": "lab_test_turnaround_times",
        "query": "How long do blood test results take to be ready?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 7. telemedicine_eligibility
    {
        "id": "Q08",
        "topic": "telemedicine_eligibility",
        "query": "Is telemedicine available for Dermatology consultations?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 8. emergency_visit_protocol
    {
        "id": "Q09",
        "topic": "emergency_visit_protocol",
        "query": "What should I do in case of a medical emergency at the clinic?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 9. patient_data_privacy_policy
    {
        "id": "Q10",
        "topic": "patient_data_privacy_policy",
        "query": "How does Practo protect my personal health data and medical records?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 10. follow_up_visit_discount_policy
    {
        "id": "Q11",
        "topic": "follow_up_visit_discount_policy",
        "query": "What discount do I get for a follow-up visit within 7 days?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 11. second_opinion_process
    {
        "id": "Q12",
        "topic": "second_opinion_process",
        "query": "How do I request a second opinion from another doctor on Practo?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 12. home_visit_eligibility
    {
        "id": "Q13",
        "topic": "home_visit_eligibility",
        "query": "Can elderly patients get a doctor home visit?",
        "expected_route": "rag",
        "expected_grounded": True,
    },
    # 13. Out-of-scope — no KB match
    {
        "id": "Q14",
        "topic": "OUT_OF_SCOPE",
        "query": "What is the best cricket stadium in India?",
        "expected_route": "ungrounded",
        "expected_grounded": False,
    },
    # 14. Edge case — ambiguous query with no domain keywords; correctly
    #     blocked by the domain gate ("refund" alone is not clinic-specific).
    #     This validates the guardrail's conservatism on ambiguous inputs.
    {
        "id": "Q15",
        "topic": "EDGE_CASE",
        "query": "Can I get a refund if I paid by cash?",
        "expected_route": "ungrounded",
        "expected_grounded": False,
    },
]

# ---------------------------------------------------------------------------
# MOCK LLM judge — deterministic text-overlap scoring
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "do", "i", "my", "me", "be", "get",
    "how", "what", "when", "where", "why", "who", "can", "will",
    "does", "are", "was", "if", "that", "this", "with", "by", "from",
])


def _keywords(text: str, min_len: int = 3) -> list[str]:
    """Extract meaningful lowercase tokens from text, excluding stopwords."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) >= min_len and t not in _STOPWORDS]


def _overlap_count(source_tokens: list[str], target_tokens: list[str]) -> int:
    """Count how many source_tokens appear in target_tokens."""
    target_set = set(target_tokens)
    return sum(1 for t in source_tokens if t in target_set)


def judge_prompt_template(
    query: str,
    context: str,
    answer: str,
) -> str:
    """
    Represents the LLM-as-judge prompt that would be sent to a real model.
    Under MOCK_LLM this string is shown for transparency but not sent anywhere.
    """
    return (
        f"[JUDGE PROMPT]\n"
        f"You are an evaluation judge for a RAG-based support agent.\n\n"
        f"Query: {query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Agent answer:\n{answer}\n\n"
        f"Score the following on a scale of 0, 0.5, or 1.0:\n"
        f"1. context_relevance: Is the retrieved context relevant to the query?\n"
        f"2. groundedness: Is the answer supported by the retrieved context?\n"
        f"3. answer_relevance: Does the answer address the query?\n\n"
        f"[MOCK_LLM: scores computed via keyword-overlap heuristic below]"
    )


def score_context_relevance(query: str, hits: list[dict]) -> float:
    """
    1.0 — top chunk contains ≥3 query keywords
    0.5 — top chunk contains 1-2 query keywords
    0.0 — no chunks or 0 overlap
    """
    if not hits:
        return 0.0
    q_kw = _keywords(query)
    top_kw = _keywords(hits[0].get("text", ""))
    overlap = _overlap_count(q_kw, top_kw)
    if overlap >= 3:
        return 1.0
    if overlap >= 1:
        return 0.5
    return 0.0


def score_groundedness(answer: str, hits: list[dict]) -> float:
    """
    1.0 — answer shares ≥2 meaningful phrases (3+-gram overlap) with chunks
    0.5 — answer shares exactly 1 such phrase
    0.0 — no phrase overlap OR answer is the fallback 'I don't know' string
    """
    fallback_markers = ["i don't know", "i don't have", "does not appear to match"]
    ans_lower = answer.lower()
    if any(m in ans_lower for m in fallback_markers):
        return 0.0

    if not hits:
        return 0.0

    all_chunk_text = " ".join(h.get("text", "") for h in hits).lower()
    ans_kw = _keywords(answer)
    chunk_kw = _keywords(all_chunk_text)

    # Count 3-word sliding windows shared between answer and chunks
    ans_trigrams = set(
        " ".join(ans_kw[i:i+3]) for i in range(max(0, len(ans_kw) - 2))
    )
    chunk_trigrams = set(
        " ".join(chunk_kw[i:i+3]) for i in range(max(0, len(chunk_kw) - 2))
    )
    shared = ans_trigrams & chunk_trigrams

    if len(shared) >= 2:
        return 1.0
    if len(shared) == 1:
        return 0.5

    # Fall back to single-token overlap
    token_overlap = _overlap_count(ans_kw, chunk_kw)
    if token_overlap >= 5:
        return 0.5
    return 0.0


def score_answer_relevance(query: str, answer: str) -> float:
    """
    1.0 — answer contains ≥3 query keywords
    0.5 — answer contains 1-2 query keywords
    0.0 — 0 overlap (off-topic or fallback)
    """
    q_kw = _keywords(query)
    a_kw = _keywords(answer)
    overlap = _overlap_count(q_kw, a_kw)
    if overlap >= 3:
        return 1.0
    if overlap >= 1:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_case(case: dict) -> dict:
    """Run one test case through the agent and score with the MOCK judge."""
    sid = new_session_id()
    t0 = time.perf_counter()
    resp = run_agent(query=case["query"], session_id=sid, turn=1)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Retrieve chunks directly for judge input (may differ from agent's hits
    # if agent was blocked by guardrail)
    hits = resp.rag_hits
    # Convert RetrievedChunk objects to dicts for judge functions
    hits_dicts = [
        {"text": h.snippet, "source": h.source, "score": h.score}
        for h in hits
    ]

    # Show judge prompt (transparency — not sent to any API)
    context_preview = (hits_dicts[0]["text"] if hits_dicts else "")[:200]
    _prompt = judge_prompt_template(case["query"], context_preview, resp.answer[:300])

    cr = score_context_relevance(case["query"], hits_dicts)
    gr = score_groundedness(resp.answer, hits_dicts)
    ar = score_answer_relevance(case["query"], resp.answer)

    route_ok = resp.route.value == case["expected_route"]
    grounded_ok = (resp.grounded == case["expected_grounded"]) or (
        # ungrounded route also means grounded=False
        resp.route.value == "ungrounded" and not case["expected_grounded"]
    )

    return {
        "id": case["id"],
        "topic": case["topic"],
        "query": case["query"],
        "route": resp.route.value,
        "expected_route": case["expected_route"],
        "route_ok": route_ok,
        "grounded": resp.grounded,
        "expected_grounded": case["expected_grounded"],
        "grounded_ok": grounded_ok,
        "context_relevance": cr,
        "groundedness": gr,
        "answer_relevance": ar,
        "top_similarity": resp.top_similarity_score,
        "latency_ms": round(latency_ms, 1),
        "answer_snippet": resp.answer[:120].replace("\n", " "),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    n = len(results)
    print("\n" + "=" * 88)
    print("Practo Agent — RAG Triad Evaluation Report (MOCK_LLM Judge)")
    print(f"Test cases: {n}  |  Judge: keyword-overlap heuristic (deterministic)")
    print("=" * 88)

    hdr = (
        f"{'ID':<5} {'Topic':<34} {'Route':<13} "
        f"{'CR':>5} {'GR':>5} {'AR':>5} {'Lat(ms)':>8}"
    )
    print(hdr)
    print("-" * 88)

    cr_sum = gr_sum = ar_sum = 0.0
    route_correct = 0

    for r in results:
        route_flag = "" if r["route_ok"] else " !"
        print(
            f"{r['id']:<5} {r['topic'][:33]:<34} "
            f"{r['route'][:12]:<13}{route_flag}"
            f"{r['context_relevance']:>5.1f} "
            f"{r['groundedness']:>5.1f} "
            f"{r['answer_relevance']:>5.1f} "
            f"{r['latency_ms']:>8.1f}"
        )
        cr_sum += r["context_relevance"]
        gr_sum += r["groundedness"]
        ar_sum += r["answer_relevance"]
        if r["route_ok"]:
            route_correct += 1

    print("-" * 88)
    print(
        f"{'AVERAGE':<5} {'':<34} {'':<13}"
        f"{cr_sum/n:>5.2f} {gr_sum/n:>5.2f} {ar_sum/n:>5.2f}"
    )
    print("=" * 88)
    print(f"  CR = context_relevance   GR = groundedness   AR = answer_relevance")
    print(f"  Route accuracy: {route_correct}/{n} ({100*route_correct/n:.0f}%)")
    print()

    # Per-topic breakdown
    print("Per-query detail (answer snippet):")
    print("-" * 88)
    for r in results:
        grounded_flag = (
            "grounded" if r["grounded"] else
            "ungrounded" if r["grounded"] is False else "n/a"
        )
        print(f"  {r['id']} [{r['topic']}]")
        print(f"     Q: {r['query']}")
        print(f"     A: {r['answer_snippet']}")
        print(f"     CR={r['context_relevance']}  GR={r['groundedness']}  "
              f"AR={r['answer_relevance']}  sim={r['top_similarity']}  "
              f"route={r['route']}({grounded_flag})")
        print()


def main() -> None:
    print("Running RAG triad evaluation...")
    print(f"Test cases: {len(TEST_CASES)}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for case in TEST_CASES:
        print(f"  {case['id']} — {case['query'][:60]}")
        result = evaluate_case(case)
        results.append(result)
        # Write each result immediately so partial runs are recoverable
        with EVAL_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")

    print_report(results)
    print(f"Results written to: {EVAL_LOG}")


if __name__ == "__main__":
    main()
