"""Download and cache the optional local Hugging Face generation model once."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import HF_GENERATION_MODEL


def main() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Downloading and caching {HF_GENERATION_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_GENERATION_MODEL)
    model.eval()
    print("Model cached successfully. Set USE_HF_LLM=true to enable local generation.")


if __name__ == "__main__":
    main()
