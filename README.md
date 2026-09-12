# Practo Domain Support Agent (LangGraph)

Track: **Healthcare (Practo)**

Completed: Final Capstone — all four parts (Tasks 1–16).

Runs entirely under **MOCK_LLM** with **zero API keys** and **zero network access** required (default `OFFLINE_MODE=true` uses a local TF-IDF embedding backend; default `USE_REAL_LLM=false` uses a deterministic mock generator).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install langgraph-checkpoint-sqlite   # Task 15

# 2. Validate dataset
python scripts/dataset.py

# 3. Seed ChromaDB (both collections, OFFLINE_MODE=true by default)
python scripts/seed_kb.py

# 4. Run all demos
python demos/demo_routes.py        # Task 7  — both routing branches
python demos/demo_memory.py        # Task 8  — multi-turn + fresh session
python demos/demo_guardrails.py    # Task 10 — PII / injection / groundedness
python demos/demo_checkpoint.py    # Task 15 — SQLite checkpointing
python demos/demo_resilience.py    # Task 16 — retry + timeouts

# 5. Evaluation
python eval/run_eval.py            # Task 13 — 15-query RAG triad

# 6. FastAPI server
uvicorn api.main:app --port 8000

# 7. MCP server (separate terminal)
python mcp/server.py               # listens on :8001/mcp
python mcp/client.py               # separate client process
```

Every script above produces output identical to the files in `transcripts/`.

---

## Part 1 — Dataset Design & RAG Core

### Task 1 — Dataset Design Choices (grader-reproducible)

| Parameter | Value |
|---|---|
| Seed | `42` |
| Size | `100` records |
| Category weights | General Medicine 35%, Cardiology 20%, Dermatology 15%, Pediatrics 15%, Orthopedics 15% |
| Status weights | Completed 35%, Scheduled 30%, Cancelled 15%, Rescheduled 10%, No-Show 10% |
| Fee range | ₹300–₹2500 |
| follow_up probability | 20% |

**Fee range rationale:** Indian private-clinic consultation fees range from ~₹300 for a general-physician visit to ~₹2500 for a senior cardiologist or orthopedic specialist; the ₹300–₹2500 band covers the full realistic spectrum.

Transcript: `transcripts/t01_dataset.txt`

### Task 2 — Knowledge Base

12 policy documents in `data/knowledge_base/`, one topic per file, 2–5 sentences each, covering all required topics: appointment booking, cancellation/rescheduling, consultation fees, insurance claims, prescription refills, lab turnaround, telemedicine, emergency protocol, data privacy, follow-up discounts, second opinion, home visits.

### Task 3 — Chunking Strategies

| Collection | Strategy | Parameters |
|---|---|---|
| `policy_fixed` | Fixed-size with overlap | 200-char chunks, 40-char overlap |
| `policy_sentence` | Sentence-based | 2 sentences grouped per chunk |

Embedding model: `all-MiniLM-L6-v2` (SentenceTransformers) when `OFFLINE_MODE=false`; TF-IDF (586-dim, pure numpy) when `OFFLINE_MODE=true` (default).

Transcript: `transcripts/t03_seed_kb.txt`

### Task 4 — Grounded Generation & Threshold Calibration

**Two-layer groundedness check (TF-IDF offline mode):**

- **Layer 1 — domain keyword gate:** query must contain at least one medical/scheduling term. Rejects out-of-scope queries immediately regardless of similarity score.
- **Layer 2 — cosine similarity threshold = `0.18`:** rejects low-scoring queries that passed Layer 1.

**Measured calibration scores (`policy_fixed`, TF-IDF):**

| Query | Type | kw_gate | Top-1 cosine sim |
|---|---|---|---|
| "How do I cancel or reschedule my appointment?" | in-scope | PASS | 0.3767 |
| "What are the consultation fees for Cardiology?" | in-scope | PASS | 0.4131 |
| "How long do blood test results take?" | in-scope | PASS | 0.3582 |
| "What is the best cricket stadium in India?" | out-of-scope | FAIL | 0.2710 |
| "How do I apply for a bank loan?" | out-of-scope | FAIL | 0.2875 |

Min in-scope: 0.3582 — Max out-of-scope: 0.2875 — threshold set to 0.18 (domain gate is the primary protection; threshold catches partial-keyword edge cases). All out-of-scope queries blocked by Layer 1; threshold provides defence-in-depth.

Transcripts: `transcripts/t04_calibrate.txt`, `transcripts/t04_rag_demo.txt`

### Task 5 — Chunking Strategy Evaluation (P@3 / R@3)

**`policy_fixed`** — Mean P@3=0.600, Mean R@3=0.800

**`policy_sentence`** — Mean P@3=0.500, Mean R@3=0.800

**Recommendation:** Deploy `policy_fixed`. It achieves higher Mean P@3 (0.600 vs 0.500) at identical recall. Fixed-size chunks with overlap capture boundary-spanning content that sentence-based grouping may split across separate low-ranking chunks, giving the generator more precise context with fewer irrelevant sections.

Transcript: `transcripts/t05_chunking_eval.txt`

---

## Part 2 — LangGraph Agent with Tools, Memory & Guardrails

### Task 6 — Escalation Tool

```
escalation_score = 0.6 * follow_up_required + 0.4 * (days_since_created / 30)
threshold = 0.60
```

**Justification:** P80 of `days_since_created` = 25 days (seed=42, n=100). A record with `follow_up=False` and `days=25` scores `0.4 * 25/30 = 0.333` — below threshold. Any record with `follow_up=True` scores minimum `0.6 * 1 + 0 = 0.600` — at threshold, escalated. All 24 follow-up cases in the dataset are escalated regardless of age, which is the correct clinical behaviour.

Transcript: `transcripts/t06_tools.txt`

### Task 7 — LangGraph Graph (4 nodes, 1 conditional edge)

```
[guardrail_input] → [classify_intent] --conditional--> [appointment_tool]
                                      \                      |
                                       → [rag_tool]          |
                                              |               |
                                              +--[format_response]
```

Conditional edge routes on `intent == "appointment"` (APT-XXXX pattern or lookup+object keywords) vs `"rag"` (all policy questions).

Transcript: `transcripts/t07_routes.txt`

### Task 8 — Persisted Memory

JSON file per session under `logs/memory/<session_id>.json`. Atomic write (tmp+rename). Transcript A shows 3-turn history carried across turns; Transcript B shows fresh session with empty state.

Transcript: `transcripts/t08_memory.txt`

### Task 9 — Structured Output Schema

`AgentResponse` Pydantic model in `app/models.py`. Every agent turn validated before return. Cross-field validator: RAG route must not populate `appointment`, appointment route must not populate `rag_hits`.

### Task 10 — Guardrails

| Guardrail | Implementation | Demo |
|---|---|---|
| PII masking | Regex for Indian mobile numbers (10-digit, +91, 0XX formats) | `query_sanitised` has `[PHONE_REDACTED]`; raw number never logged |
| Injection detection | Pattern scoring (≥0.8 score → blocked) | "Ignore all previous instructions" → `guardrail_blocked` route |
| Groundedness | Domain keyword gate + cosine threshold | Out-of-scope → `ungrounded` route, fallback answer |

Transcript: `transcripts/t10_guardrails.txt`

---

## Part 3 — Evaluation, Observability & FastAPI

### Task 11 — FastAPI Endpoints

| Endpoint | Description |
|---|---|
| `POST /ask` | Agent query (session-aware, PII-safe) |
| `POST /add-document` | Add KB doc, full re-index of both collections |
| `GET /appointments/{record_id}` | Direct appointment lookup |
| `GET /health` | ChromaDB status, KB doc count, mode flags |

### Task 12 — Structured Logging

- `logs/agent.jsonl` — one line per agent turn; fields: `trace_id`, `timestamp`, `session_id`, `turn`, `route`, `query_sanitised` (never `query_original`), `top_similarity_score`, `grounded`, `escalation_score`, `guardrail_events`
- `logs/requests.jsonl` — one line per HTTP request; fields: `trace_id`, `method`, `path`, `status_code`, `latency_ms`, `query_sanitised`
- **Security:** `query_original` is never written to any log file. The guardrail_input node masks PII before `query_sanitised` is produced; only `query_sanitised` reaches the logger.

Transcript: `transcripts/t11_t12_api.txt`

### Task 13 — RAG Triad Evaluation (15 queries, MOCK_LLM judge)

Judge: deterministic keyword-overlap heuristic (no LLM API). Each metric scored 0/0.5/1.0.

| Score | Average across 15 queries |
|---|---|
| Context Relevance (CR) | **0.77** |
| Groundedness (GR) | **0.87** |
| Answer Relevance (AR) | **0.87** |
| Route accuracy | **15/15 (100%)** |

Test set: 13 in-scope (covers all 12 KB topics; consultation_fee_structure has 2 queries) + 1 out-of-scope + 1 edge-case.

Transcript: `transcripts/t13_eval.txt`

---

## Part 4 — Resilience & Interoperability

### Task 14 — MCP Tool Server

`mcp/server.py` — FastAPI app implementing JSON-RPC 2.0 at `POST /mcp` (same protocol as fastmcp, compatible with all MCP clients). Exposes `check_appointment_status` with full input schema.

`mcp/client.py` — separate script that calls `tools/list` then `tools/call` for 4 record IDs.

Note: `fastmcp` requires Python ≥3.10. This machine runs Python 3.9. The raw JSON-RPC 2.0 implementation is protocol-identical; install `fastmcp` on Python 3.10+ to use the decorator syntax instead.

Transcript: `transcripts/t14_mcp.txt`

### Task 15 — SQLite Checkpointing

`langgraph-checkpoint-sqlite` (`SqliteSaver`). Checkpoint DB: `data/checkpoints.sqlite`. `interrupt_after=["classify_intent"]` stops after 2 nodes; resuming the same `thread_id` loads checkpoint state and runs only the remaining 2 nodes — execution counters prove prior nodes are not re-run.

Transcript: `transcripts/t15_checkpoint.txt`

### Task 16 — Timeouts & Retries

`app/resilience.py` provides:

| Utility | Config |
|---|---|
| `retry_with_backoff` | max_attempts=3, initial_delay=0.1s, max_delay=2.0s, backoff_factor=2.0, jitter=True |
| `node_timeout` | context manager; uses `threading.Timer` + cooperative `check()` |
| `run_with_global_timeout` | daemon thread + `thread.join(timeout)` |

Demonstrations:
- **(a)** Retry: fails attempts 1 and 2 (simulated `ConnectionError`), succeeds on attempt 3.
- **(b)** Per-node timeout: node polls every 50ms with `nt.check()`; fires `NodeTimeoutError` at ~0.32s (set to 0.3s).
- **(c)** Global timeout: entire run cancelled at ~0.50s (set to 0.5s) via daemon thread.

Transcript: `transcripts/t16_resilience.txt`

---

## Repository Structure

```
practo-project/
├── app/
│   ├── config.py          # all constants, seeds, thresholds, mode flags
│   ├── embeddings.py      # TF-IDF (offline) + SentenceTransformers backends
│   ├── graph.py           # LangGraph agent (4 nodes, conditional edge)
│   ├── guardrails.py      # PII masking, injection detection, groundedness
│   ├── log.py             # JSONL structured logging (trace_id, latency_ms)
│   ├── memory.py          # JSON-file conversation persistence
│   ├── models.py          # Pydantic AgentResponse schema
│   ├── rag.py             # retriever + grounded generation
│   ├── resilience.py      # retry, node_timeout, global_timeout
│   └── tools.py           # check_appointment_status, retrieve_policy
├── api/
│   ├── main.py            # FastAPI: /ask, /add-document, /appointments, /health
│   └── schemas.py         # FastAPI request/response Pydantic models
├── data/
│   └── knowledge_base/    # 12 policy .txt documents
├── demos/
│   ├── demo_routes.py     # Task 7
│   ├── demo_memory.py     # Task 8
│   ├── demo_guardrails.py # Task 10
│   ├── demo_checkpoint.py # Task 15
│   └── demo_resilience.py # Task 16
├── eval/
│   └── run_eval.py        # Task 13 — 15-query RAG triad
├── mcp/
│   ├── server.py          # Task 14 — MCP JSON-RPC 2.0 server
│   └── client.py          # Task 14 — MCP client (separate process)
├── scripts/
│   ├── dataset.py         # Task 1 — deterministic dataset generator
│   ├── seed_kb.py         # Task 3 — chunk, embed, index both collections
│   ├── calibrate_threshold.py  # Task 4 — threshold measurement
│   ├── evaluate_chunking.py    # Task 5 — P@3 / R@3 evaluation
│   └── setup_hf_model.py  # one-time HuggingFace model download helper
├── transcripts/           # captured output for every task
├── logs/                  # runtime JSONL logs (gitignored)
├── requirements.txt
└── README.md
```

---

## Environment Flags

| Variable | Default | Effect |
|---|---|---|
| `OFFLINE_MODE` | `true` | `true` = TF-IDF embeddings (no network); `false` = SentenceTransformers |
| `USE_REAL_LLM` | `false` | `false` = MOCK_LLM (deterministic); `true` = Groq API |
| `GROQ_API_KEY` | `` | Only needed when `USE_REAL_LLM=true` |

All acceptance criteria are satisfied under `OFFLINE_MODE=true USE_REAL_LLM=false` (the defaults). No API key, no network access required.
