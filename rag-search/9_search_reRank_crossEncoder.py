"""
9_search_reRank_crossEncoder.py
--------------------------------
Two-stage retrieval demo (retrieve → re-rank):

  Stage A (fast, broad): get a shortlist of candidate documents
  Stage B (slower, precise): score each candidate against the query and
                            reorder so the best match is first

Why not local CrossEncoder?
  sentence_transformers.CrossEncoder needs PyTorch. On this Python 3.13 /
  macOS setup PyTorch is not installable, so Stage B uses Hugging Face
  Inference API `sentence_similarity` instead.
"""

import importlib
import os

from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Demo inputs (in a real RAG pipeline, candidates come from LanceDB search)
# ---------------------------------------------------------------------------
DEFAULT_CANDIDATES = [
    "LanceDB stores vectors in local binary files.",
    "Python is a popular programming language.",
    "Hugging Face models run locally using sentence-transformers.",
]

DEFAULT_QUERY = "Where are vectors saved in LanceDB?"
# Embedding model used by HF sentence_similarity to score query vs candidates
RERANK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def rerank_candidates(
    query: str = DEFAULT_QUERY,
    candidates: list[str] | None = None,
    model: str = RERANK_MODEL,
) -> list[tuple[float, str]]:
    """Run the full retrieve → score → re-rank pipeline.

    What this function does, step by step:
      1) Take (or default) a candidate list from first-stage retrieval
      2) Authenticate to Hugging Face using HF_TOKEN from .env
      3) Ask the API for a similarity score for each (query, candidate) pair
      4) Sort pairs by score descending so rank-1 is the best context
      5) Print an explanation of each step and the final answer + score

    Args:
        query: Natural-language question (what the user wants answered).
        candidates: Shortlist of docs from a fast retriever (vector/BM25/etc.).
        model: HF model id used to compute similarity scores.

    Returns:
        List of (score, document) tuples, best match first.
    """
    # ------------------------------------------------------------------
    # STEP 1 — Prepare the candidate set (first-stage retrieval result)
    # In production you would replace DEFAULT_CANDIDATES with something like:
    #   table.search(query).limit(20).to_pandas()["text"].tolist()
    # Re-ranking then picks the best few from those ~20 for the LLM.
    # ------------------------------------------------------------------
    if candidates is None:
        candidates = DEFAULT_CANDIDATES

    print("=" * 60)
    print("STEP 1 — Fast initial retrieval (candidate set)")
    print("=" * 60)
    print(f"Query: {query}")
    print(f"Retrieved {len(candidates)} candidates (before re-ranking):")
    for i, doc in enumerate(candidates, start=1):
        print(f"  Candidate {i}: {doc}")

    # ------------------------------------------------------------------
    # STEP 2a — Load credentials
    # Import the shared helper from 0_lanceDB_create.py so we do not
    # hardcode secrets. HF_TOKEN must exist in feature-arun/.env (or env).
    # ------------------------------------------------------------------
    create_db = importlib.import_module("0_lanceDB_create")
    create_db._load_dotenv()  # Reads KEY=VALUE from nearby .env into os.environ

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set. Add it to feature-arun/.env")

    # Authenticated client for Hugging Face Inference API calls
    client = InferenceClient(api_key=token)

    # ------------------------------------------------------------------
    # STEP 2b — Score every candidate against the query
    # sentence_similarity(query, [doc1, doc2, ...]) returns one float per
    # doc (typically ~0 to 1). Higher = more semantically related.
    # This is the "re-ranker" stage (API stand-in for a CrossEncoder).
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("STEP 2 — Re-rank with Hugging Face sentence similarity")
    print("=" * 60)
    print(f"Model: {model}")
    print("Scoring each (query, candidate) pair...")

    scores = client.sentence_similarity(
        query,  # The user question
        other_sentences=candidates,  # Docs to compare against the question
        model=model,  # Which embedding/similarity model to use remotely
    )

    # Show raw scores in the original candidate order (before sorting)
    print("Raw matching scores (same order as candidates):")
    for i, (score, doc) in enumerate(zip(scores, candidates), start=1):
        print(f"  Candidate {i}: score={score:.4f} | {doc}")

    # ------------------------------------------------------------------
    # STEP 3 — Sort by score (re-ordering = the actual "re-rank")
    # zip(scores, candidates) pairs each score with its document, then
    # sorted(..., reverse=True) puts the highest score at index 0.
    # ------------------------------------------------------------------
    ranked_results = sorted(zip(scores, candidates), reverse=True)

    print()
    print("=" * 60)
    print("STEP 3 — Sort by score (highest first) → final ranking")
    print("=" * 60)
    for rank, (score, doc) in enumerate(ranked_results, start=1):
        # Mark rank-1 clearly — this is what you would pass to the LLM
        marker = " ← FINAL ANSWER (best match)" if rank == 1 else ""
        print(f"  Rank {rank}: score={score:.4f} | {doc}{marker}")

    # ------------------------------------------------------------------
    # FINAL ANSWER — best document + its matching score
    # ranked_results[0] is always the top hit after descending sort.
    # ------------------------------------------------------------------
    top_score, top_doc = ranked_results[0]
    print()
    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(f"Best matching document: {top_doc}")
    print(f"Matching score:         {top_score:.4f}")
    print("(Higher score = more similar to the query)")

    return ranked_results


if __name__ == "__main__":
    # Running: python 9_search_reRank_crossEncoder.py
    # Executes the full pipeline with default query + candidates
    rerank_candidates()
