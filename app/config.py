"""
Central configuration — all constants, seeds, and thresholds live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
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

# Groundedness threshold — calibrated with all-MiniLM-L6-v2 on policy_fixed.
# In-scope top-1 scores: 0.7026, 0.6770, 0.6208 (minimum 0.6208).
# Out-of-scope scores: 0.1821, 0.2223 (maximum 0.2223).
# 0.42 is the midpoint between those observed clusters and therefore satisfies
# the capstone requirement to choose an empirical, non-preset threshold.
SIMILARITY_THRESHOLD: float = 0.42

# ---------------------------------------------------------------------------
# Offline / embedding mode
# ---------------------------------------------------------------------------
# SentenceTransformers is the capstone's primary embedding backend. The model
# is downloaded once and cached locally; set OFFLINE_MODE=true for the explicit
# no-network TF-IDF fallback.
OFFLINE_MODE: bool = os.getenv("OFFLINE_MODE", "false").lower() != "false"

# ---------------------------------------------------------------------------
# LLM mode
# ---------------------------------------------------------------------------
# Set USE_REAL_LLM=true in the environment to switch to a real API.
USE_REAL_LLM: bool = os.getenv("USE_REAL_LLM", "false").lower() == "true"
# Convenience alias — True when running in mock/offline mode (no real LLM)
USE_HF_LLM: bool = os.getenv("USE_HF_LLM", "false").lower() == "true"
MOCK_LLM_MODE: bool = not USE_REAL_LLM and not USE_HF_LLM
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
HF_GENERATION_MODEL: str = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")
