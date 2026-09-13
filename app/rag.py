"""
Task 4 — Grounded generation with ChromaDB retrieval.

Supports both chunking collections (policy_fixed, policy_sentence).
Uses MOCK_LLM by default; switches to Groq when USE_REAL_LLM=true.

Embedding backend is determined by OFFLINE_MODE (see app/config.py).
IMPORTANT: seed_kb.py must be run before this module is used, so that
           the TF-IDF vocabulary is built from the same corpus.
           At query time we reload the fitted EF from the module singleton.
"""

from __future__ import annotations

import textwrap
import re
from typing import Literal

import chromadb

from app.config import (
    CHROMA_DIR,
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
    USE_REAL_LLM,
    GROQ_API_KEY,
    GROQ_MODEL,
    HF_GENERATION_MODEL,
    OFFLINE_MODE,
    USE_HF_LLM,
)
from app.config import KB_DIR
from app.embeddings import get_embedding_function, reset_embedding_function

# ---------------------------------------------------------------------------
# ChromaDB client (shared, lazy)
# ---------------------------------------------------------------------------
_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _load_kb_corpus() -> list[str]:
    """Load all KB document bodies to fit the TF-IDF model from disk."""
    corpus: list[str] = []
    for path in sorted(KB_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        body = "\n".join(lines[1:]).strip() if lines and lines[0].startswith("source:") else text
        corpus.append(body)
    return corpus


def _get_ef():
    """Return the embedding function, fitting TF-IDF from KB corpus if needed."""
    import re

    def _sentence_chunks(text: str) -> list[str]:
        raw = re.split(r"(?<=[.?!])\s+", text.strip())
        sents = [s.strip() for s in raw if s.strip()]
        return [" ".join(sents[i:i+2]) for i in range(0, len(sents), 2)]

    def _fixed_chunks(text: str, size: int = 200, overlap: int = 40) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = text[start:start + size].strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(text):
                break
            start = start + size - overlap
        return chunks

    from app.embeddings import _ef_instance
    if _ef_instance is not None:
        return _ef_instance
    # Re-fit from disk
    corpus = _load_kb_corpus()
    all_chunks: list[str] = []
    for doc in corpus:
        all_chunks.extend(_fixed_chunks(doc))
        all_chunks.extend(_sentence_chunks(doc))
    return get_embedding_function(corpus=all_chunks)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

CollectionName = Literal["policy_fixed", "policy_sentence"]


def retrieve(
    query: str,
    collection_name: CollectionName = COLLECTION_FIXED,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Return the top-k chunks from the named collection with their cosine
    similarity scores.

    Each result dict has:
        text       : chunk text
        source     : originating document name (metadata field)
        score      : cosine similarity (0-1, higher = more similar)
        chunk_id   : ChromaDB document id
    """
    ef = _get_ef()
    query_embedding = ef([query])[0]

    client = _get_client()
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for doc, meta, dist, cid in zip(documents, metadatas, distances, ids):
        # cosine collection: distance = 1 - cosine_similarity
        score = max(0.0, 1.0 - dist)
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(score, 4),
                "chunk_id": cid,
            }
        )

    return hits


# ---------------------------------------------------------------------------
# Domain keyword gate (Layer 1 of groundedness check)
# ---------------------------------------------------------------------------
# These are terms that appear in the knowledge base but NOT in typical
# out-of-scope queries.  A query must contain at least one of these terms
# (case-insensitive substring match) to pass layer 1.
DOMAIN_KEYWORDS: frozenset[str] = frozenset([
    "appointment", "cancel", "reschedule", "book", "booking", "slot",
    "consultation", "fee", "specialist", "clinic", "practo",
    "cardiology", "dermatology", "pediatrics", "orthopedics", "medicine",
    "prescription", "refill", "medication", "drug", "dosage",
    "lab", "test", "result", "blood", "biopsy", "report",
    "telemedicine", "video", "teleconsult", "online",
    "emergency", "triage", "urgent",
    "insurance", "claim", "cashless", "reimbursement",
    "follow", "follow-up", "second opinion",
    "home visit", "home-visit", "data", "privacy", "record",
    "doctor", "patient", "hospital", "discharge",
])


def _passes_domain_gate(query: str) -> bool:
    """Return True if the query contains at least one domain keyword."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in DOMAIN_KEYWORDS)


# ---------------------------------------------------------------------------
# MOCK LLM
# ---------------------------------------------------------------------------

def _mock_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Deterministic mock answer: assembles retrieved chunks into a structured
    response without any real LLM call.  Used for MOCK_LLM mode.
    """
    if not context_chunks:
        return "I don't have enough information in the knowledge base to answer that question."

    lines = [f"Based on Practo clinic policy (top {len(context_chunks)} relevant section(s)):", ""]
    for i, chunk in enumerate(context_chunks, 1):
        wrapped = textwrap.fill(chunk["text"].strip(), width=90)
        lines.append(f"[{i}] (source: {chunk['source']}, similarity: {chunk['score']})")
        lines.append(wrapped)
        lines.append("")
    lines.append(
        "Note: This response is assembled from the knowledge base under MOCK_LLM mode. "
        "No language model inference was performed."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real LLM (Groq) — optional
# ---------------------------------------------------------------------------

def _groq_answer(query: str, context_chunks: list[dict]) -> str:
    """Call Groq API with retrieved context.  Only used when USE_REAL_LLM=true."""
    import httpx

    context_text = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(context_chunks, 1)
    )
    system_prompt = (
        "You are a helpful Practo patient-support assistant. "
        "Answer the patient's question using ONLY the context provided below. "
        "If the context does not contain enough information, say 'I don't know'. "
        "Do not invent facts.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0,
    }
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Local Hugging Face generation — optional
# ---------------------------------------------------------------------------

_hf_generator: tuple[object, object] | None = None


def _get_hf_generator() -> tuple[object, object]:
    """Load the cached FLAN-T5 model once without making network requests."""
    global _hf_generator
    if _hf_generator is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                HF_GENERATION_MODEL, local_files_only=True
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                HF_GENERATION_MODEL, local_files_only=True
            )
        except OSError as exc:
            raise RuntimeError(
                f"Generation model '{HF_GENERATION_MODEL}' is not cached locally. "
                "Run `python scripts/setup_hf_generation_model.py` once, or set "
                "USE_HF_LLM=false to return to deterministic MOCK_LLM mode."
            ) from exc
        model.eval()
        _hf_generator = (tokenizer, model)
    return _hf_generator


def _hf_answer(
    query: str, context_chunks: list[dict], conversation_context: str = ""
) -> str:
    """Generate a concise answer from retrieved context using local FLAN-T5."""
    tokenizer, model = _get_hf_generator()
    context = "\n".join(chunk["text"] for chunk in context_chunks)
    prompt = (
        "Answer the healthcare support question using only the supplied context. "
        "If the context does not answer it, say I don't know. Respond in one complete "
        "sentence and include any relevant currency, time period, or eligibility condition.\n\n"
        f"Knowledge-base context: {context}\n\n"
        f"Earlier conversation (may be empty): {conversation_context}\n\n"
        f"Current question: {query}\nAnswer:"
    )
    encoded = tokenizer(  # type: ignore[operator]
        prompt, return_tensors="pt", truncation=True, max_length=512
    )
    generated = model.generate(  # type: ignore[operator]
        **encoded, max_new_tokens=160, do_sample=False, num_beams=1
    )
    answer = tokenizer.decode(  # type: ignore[operator]
        generated[0], skip_special_tokens=True
    ).strip()
    # FLAN-T5-small sometimes returns an amount alone (for example "100")
    # despite the complete-sentence instruction. Preserve the generated fact
    # while presenting a clear answer for the user-facing fee questions.
    numeric_answer = answer.removeprefix("₹").strip().rstrip(".")
    if "fee" in query.lower() and re.fullmatch(r"\d+(?:\.\d+)?", numeric_answer):
        answer = f"The applicable fee is ₹{numeric_answer}."
    # A small local model occasionally emits a single keyword or simply echoes
    # the question. In that case, return a grounded sentence from the highest
    # ranked retrieved chunk instead of exposing an unusable fragment.
    answer_words = re.findall(r"\w+", answer.lower())
    if len(answer_words) < 3 or answer.strip().lower() == query.strip().lower():
        for chunk in context_chunks:
            sentences = re.split(r"(?<=[.?!])\s+", chunk["text"].strip())
            for sentence in sentences:
                # Fixed-size chunks can begin in the middle of a sentence.
                # Prefer a complete, capitalised sentence for user-facing text.
                if len(sentence.split()) >= 6 and sentence[:1].isupper():
                    return f"Based on clinic policy: {sentence.strip()}"
    return answer or "I don't know."


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def grounded_answer(
    query: str,
    collection_name: CollectionName = COLLECTION_FIXED,
    top_k: int = RETRIEVAL_TOP_K,
    threshold: float = SIMILARITY_THRESHOLD,
    *,
    user_query: str | None = None,
    conversation_context: str = "",
) -> dict:
    """
    Retrieve top-k chunks and generate a grounded answer.

    Returns:
        {
          "query"       : original query,
          "answer"      : generated text,
          "hits"        : list of retrieved chunk dicts,
          "top_score"   : best cosine similarity seen,
          "grounded"    : True if top_score >= threshold,
          "collection"  : collection used,
        }
    """
    current_question = user_query or query
    hits = retrieve(query, collection_name=collection_name, top_k=top_k)
    top_score = hits[0]["score"] if hits else 0.0
    # Two-layer groundedness: domain keyword gate AND similarity threshold
    grounded = _passes_domain_gate(query) and top_score >= threshold

    if not grounded:
        answer = (
            "I don't know. Your question does not appear to match any topic "
            "covered in the Practo clinic knowledge base."
        )
    elif USE_REAL_LLM:
        answer = _groq_answer(current_question, hits)
    elif USE_HF_LLM:
        answer = _hf_answer(current_question, hits, conversation_context)
    else:
        answer = _mock_answer(query, hits)

    return {
        "query": query,
        "user_query": current_question,
        "answer": answer,
        "hits": hits,
        "top_score": top_score,
        "grounded": grounded,
        "domain_gate_passed": _passes_domain_gate(query),
        "collection": collection_name,
    }


# ---------------------------------------------------------------------------
# CLI demo (Task 4 demonstration)
# ---------------------------------------------------------------------------

IN_SCOPE_QUERIES: list[str] = [
    "How do I cancel my appointment without paying a fee?",
    "What is the consultation fee for a Cardiology visit?",
    "Can I get a prescription refill without visiting the clinic?",
    "How long does it take to get blood test results?",
    "Is telemedicine available for Orthopedics?",
]

OUT_OF_SCOPE_QUERY: str = "What is the best cricket stadium in India?"


def run_demo(collection_name: CollectionName = COLLECTION_FIXED) -> None:
    all_queries = IN_SCOPE_QUERIES + [OUT_OF_SCOPE_QUERY]
    print("=" * 70)
    print(f"Grounded Generation Demo  |  collection: {collection_name}")
    print(f"Embedding: {'TF-IDF (offline)' if OFFLINE_MODE else 'SentenceTransformers'}")
    print(f"Threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 70)

    for q in all_queries:
        result = grounded_answer(q, collection_name=collection_name)
        label = "IN-SCOPE" if q != OUT_OF_SCOPE_QUERY else "OUT-OF-SCOPE"
        print(f"\n[{label}] Q: {q}")
        print(f"  top_score={result['top_score']:.4f}  grounded={result['grounded']}")
        # Print first retrieved source
        if result["hits"]:
            print(f"  top_source={result['hits'][0]['source']}")
        print(f"  A: {result['answer'][:200].strip()}")
        print()


if __name__ == "__main__":
    import sys
    coll = sys.argv[1] if len(sys.argv) > 1 else COLLECTION_FIXED
    run_demo(collection_name=coll)  # type: ignore[arg-type]
