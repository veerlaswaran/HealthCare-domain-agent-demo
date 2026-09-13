"""
Embedding layer — two interchangeable backends:

  1. SentenceTransformers (preferred, requires HuggingFace model download)
  2. TF-IDF + cosine similarity (offline fallback, pure numpy/stdlib)

The active backend is selected at import time via OFFLINE_MODE in config.py
(or by setting OFFLINE_MODE=true in the environment).  All downstream code
calls embed() and is backend-agnostic.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Protocol

import numpy as np

from app.config import EMBEDDING_MODEL, OFFLINE_MODE

# ---------------------------------------------------------------------------
# Protocol — any backend must satisfy this
# ---------------------------------------------------------------------------

class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 L2-normalised embedding matrix."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Return (D,) float32 L2-normalised embedding vector."""
        ...


# ---------------------------------------------------------------------------
# Backend 1 — SentenceTransformers
# ---------------------------------------------------------------------------

class SentenceTransformerBackend:
    """SentenceTransformers wrapper that uses the locally cached model only."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer
        try:
            # The setup script downloads this model once.  local_files_only
            # prevents request-time metadata checks and keeps normal agent runs
            # deterministic and network-independent afterwards.
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except OSError as exc:
            raise RuntimeError(
                f"Embedding model '{model_name}' is not cached locally. "
                "Run `python scripts/setup_hf_model.py` once, or set "
                "OFFLINE_MODE=true to use the TF-IDF fallback."
            ) from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Backend 2 — TF-IDF offline fallback
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFBackend:
    """
    Offline TF-IDF embedding backend.

    fit() must be called with the corpus before embed().  After fitting,
    embed() produces L2-normalised TF-IDF vectors in vocab space.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: list[str]) -> "TFIDFBackend":
        tokens_per_doc = [_tokenize(doc) for doc in corpus]
        # Build vocabulary from all terms
        all_terms: set[str] = set()
        for tokens in tokens_per_doc:
            all_terms.update(tokens)
        self._vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}
        V = len(self._vocab)
        N = len(corpus)
        # Document frequency
        df = np.zeros(V, dtype=np.float32)
        for tokens in tokens_per_doc:
            unique = set(tokens)
            for t in unique:
                if t in self._vocab:
                    df[self._vocab[t]] += 1.0
        # IDF: log((N+1)/(df+1)) + 1  (scikit-learn smooth variant)
        self._idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _tfidf_vector(self, text: str) -> np.ndarray:
        if not self._fitted or self._idf is None:
            raise RuntimeError("TFIDFBackend must be fit() before encoding.")
        V = len(self._vocab)
        tokens = _tokenize(text)
        tf_counts = Counter(tokens)
        total = max(len(tokens), 1)
        vec = np.zeros(V, dtype=np.float32)
        for term, count in tf_counts.items():
            if term in self._vocab:
                idx = self._vocab[term]
                tf = count / total
                vec[idx] = tf * self._idf[idx]
        # L2 normalise
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._tfidf_vector(t) for t in texts], axis=0)

    def embed_one(self, text: str) -> np.ndarray:
        return self._tfidf_vector(text)


# ---------------------------------------------------------------------------
# ChromaDB-compatible embedding function wrapper (TF-IDF)
# ---------------------------------------------------------------------------

class TFIDFEmbeddingFunction:
    """
    Wraps TFIDFBackend to satisfy ChromaDB's EmbeddingFunction interface:
        __call__(input: list[str]) -> list[list[float]]
    Also exposes embed() and embed_one() for direct use.
    """

    def __init__(self, backend: TFIDFBackend) -> None:
        self._backend = backend

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        vecs = self._backend.embed(input)
        return vecs.tolist()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._backend.embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self._backend.embed_one(text)


# ---------------------------------------------------------------------------
# ChromaDB-compatible wrapper for SentenceTransformers
# ---------------------------------------------------------------------------

class SentenceTransformerEmbeddingFunctionWrapper:
    """
    Wraps SentenceTransformerBackend to satisfy both ChromaDB EmbeddingFunction
    interface and direct embed()/embed_one() calls.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._backend = SentenceTransformerBackend(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._backend.embed(input).tolist()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._backend.embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self._backend.embed_one(text)


# ---------------------------------------------------------------------------
# Factory — returns the correct embedding function based on OFFLINE_MODE
# ---------------------------------------------------------------------------

# Module-level singleton (lazy)
_ef_instance: TFIDFEmbeddingFunction | SentenceTransformerEmbeddingFunctionWrapper | None = None
_tfidf_backend: TFIDFBackend | None = None


def get_embedding_function(
    corpus: list[str] | None = None,
) -> TFIDFEmbeddingFunction | SentenceTransformerEmbeddingFunctionWrapper:
    """
    Return a module-level singleton embedding function.

    If OFFLINE_MODE is True, returns a TFIDFEmbeddingFunction.
      - corpus must be provided on the FIRST call so the TF-IDF model can be fit.
      - Subsequent calls with corpus=None reuse the fitted instance.
    If OFFLINE_MODE is False, returns a SentenceTransformerEmbeddingFunctionWrapper.
    """
    global _ef_instance, _tfidf_backend

    if _ef_instance is not None:
        if OFFLINE_MODE and corpus is not None and _tfidf_backend is not None:
            # Re-fit with updated corpus if explicitly provided
            _tfidf_backend.fit(corpus)
        return _ef_instance

    if OFFLINE_MODE:
        _tfidf_backend = TFIDFBackend()
        if corpus:
            _tfidf_backend.fit(corpus)
        _ef_instance = TFIDFEmbeddingFunction(_tfidf_backend)
    else:
        _ef_instance = SentenceTransformerEmbeddingFunctionWrapper(EMBEDDING_MODEL)

    return _ef_instance


def reset_embedding_function() -> None:
    """Reset singleton — used in tests and re-seeding."""
    global _ef_instance, _tfidf_backend
    _ef_instance = None
    _tfidf_backend = None
