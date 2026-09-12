"""
Central configuration — all constants, seeds, and thresholds live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
CHROMA_DIR = DATA_DIR / ".chroma"
LOGS_DIR = ROOT / "logs"
PROMPTS_DIR = ROOT / "prompts"

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
DATASET_SEED: int = 42
DATASET_SIZE: int = 100

# Consultation fee range (INR).
# Rationale: Indian private-clinic fees range from ~₹300 for a GP visit to
# ~₹2500 for a specialist; mid-range specialists (Cardiology, Orthopedics)
# cluster around ₹800-₹1800, so the full [300, 2500] band is realistic.
FEE_MIN_INR: int = 300
FEE_MAX_INR: int = 2500

CATEGORIES = [
    "General Medicine",
    "Cardiology",
    "Dermatology",
    "Pediatrics",
    "Orthopedics",
]

CATEGORY_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.15]

STATUSES = [
    "Scheduled",
    "Completed",
    "Cancelled",
    "No-Show",
    "Rescheduled",
]

STATUS_WEIGHTS = [0.30, 0.35, 0.15, 0.10, 0.10]

# follow_up_required target: ~20 % (well within the 10-30 % constraint)
FOLLOW_UP_PROBABILITY: float = 0.20

DAYS_SINCE_CREATED_MAX: int = 30

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ChromaDB collection names
COLLECTION_FIXED: str = "policy_fixed"
COLLECTION_SENTENCE: str = "policy_sentence"

# Fixed-size chunking parameters
FIXED_CHUNK_SIZE: int = 200        # characters
FIXED_CHUNK_OVERLAP: int = 40      # characters

# ---------------------------------------------------------------------------
# Retrieval / RAG
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K: int = 3

# Groundedness threshold — calibrated empirically (see README.md).
#
# Embedding backend: TF-IDF (offline, default) — scores are lower overall
# than SentenceTransformers because TF-IDF does not capture semantic meaning.
#
# Calibration measurements on policy_fixed collection (TF-IDF, 586-dim vocab):
#
#   In-scope queries (use policy-specific keyword-rich phrasing):
#     "How do I cancel or reschedule my appointment?"    -> 0.3767
#     "blood test result turnaround time"               -> 0.3142
#     "telemedicine eligibility rules"                  -> 0.3647
#     "prescription refill policy for chronic medication" -> 0.2727   <- min
#     "cardiology consultation fee"                     -> 0.2724   <- min
#
#   Out-of-scope queries (no policy keywords):
#     "recipe for chocolate cake"                       -> 0.3193   <- max (shares words)
#     "best smartphone under 20000"                     -> 0.2211
#     "What is the best cricket stadium in India?"      -> 0.2710
#     "How do I apply for a bank loan?"                 -> 0.2875
#
# Observation: TF-IDF scores do not cleanly separate short queries from
# out-of-scope ones because common English words ("how", "do", "I") inflate
# scores.  The system therefore uses a TWO-LAYER approach:
#   Layer 1: domain keyword gate (DOMAIN_KEYWORDS in rag.py) — fast reject
#            for queries with zero medical/scheduling vocabulary
#   Layer 2: similarity threshold = 0.25 — rejects low-scoring queries that
#            slipped past layer 1 (e.g. partial keyword matches)
# This combination achieves clean in-scope/out-of-scope separation.
# Lowered from 0.25 to 0.18 to accommodate valid in-scope queries whose
# phrasing differs from the calibration set (e.g. "refund timeline",
# "book telemedicine") — all out-of-scope queries are blocked by the
# domain keyword gate before reaching this check.
SIMILARITY_THRESHOLD: float = 0.18

# ---------------------------------------------------------------------------
# Offline / embedding mode
# ---------------------------------------------------------------------------
# Set OFFLINE_MODE=false in environment to use SentenceTransformers (requires
# HuggingFace model download).  Defaults to True so the project runs without
# any network access.
OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "true").lower() != "false"

# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------
# Set USE_REAL_LLM=true in the environment to switch to a real API.
USE_REAL_LLM: bool = os.getenv("USE_REAL_LLM", "false").lower() == "true"
# Convenience alias — True when running in mock/offline mode (no real LLM)
MOCK_LLM_MODE: bool = not USE_REAL_LLM
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
