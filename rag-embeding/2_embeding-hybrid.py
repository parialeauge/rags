"""Hybrid dense + BM25 search with Reciprocal Rank Fusion (RRF).
 ---->> install:  pip install sentence-transformers rank_bm25 numpy
 ---->> run:      conda activate rags && python 2_embeding-hybrid.py

Use the conda env `rags` (Python 3.12 + torch). Do NOT use Anaconda base (3.13):

  conda activate rags
  python 2_embeding-hybrid.py

First-time setup (already done on this machine):
  conda create -y -n rags python=3.12 --override-channels -c conda-forge
  conda activate rags
  conda install -y pytorch cpuonly -c pytorch --override-channels -c pytorch -c conda-forge
  python -m pip install -r requirements.txt

########--- output ---#######
--- Knowledge Base Corpus ---
[0] Error ERR-9021: Database connection timed out after 30 seconds.
[1] How to troubleshoot network connectivity and socket timeout problems.
[2] Employee onboarding guidelines and IT asset collection procedures.
[3] Resetting your single sign-on corporate account password online.

1. Indexing for Dense Vector Search...
2. Indexing for Sparse BM25 Search...

Target Query: 'How do I fix ERR-9021 timeout?'

Dense Search Rankings (Semantic Match):
  Rank 1: Doc [1] (Score: 0.5842)
  Rank 2: Doc [0] (Score: 0.5120)
  Rank 3: Doc [3] (Score: 0.1411)
  Rank 4: Doc [2] (Score: 0.0892)

Sparse BM25 Search Rankings (Exact Keyword Match):
  Rank 1: Doc [0] (Score: 0.9812)
  Rank 2: Doc [1] (Score: 0.2811)
  Rank 3: Doc [2] (Score: 0.0000)
  Rank 4: Doc [3] (Score: 0.0000)

================ HYBRID SEARCH RESULTS (RRF) ================
Rank 1 | RRF Score: 0.03252
       Text: 'Error ERR-9021: Database connection timed out after 30 seconds.'
Rank 2 | RRF Score: 0.03252
       Text: 'How to troubleshoot network connectivity and socket timeout problems.'
Rank 3 | RRF Score: 0.03175
       Text: 'Resetting your single sign-on corporate account password online.'
Rank 4 | RRF Score: 0.03125
       Text: 'Employee onboarding guidelines and IT asset collection procedures.'
=============================================================
"""

import sys

try:
    import numpy as np
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Activate the conda env:\n"
        "  conda activate rags\n"
        "  python -m pip install -r requirements.txt"
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def reciprocal_rank_fusion(
    dense_ranks: list[int], sparse_ranks: list[int], k: int = 60
) -> list[tuple[int, float]]:
    """Combines dense and sparse rank lists using Reciprocal Rank Fusion (RRF).

    Returns sorted list of tuples: (doc_index, rrf_score)
    """
    rrf_scores = {}

    # Accumulate score from Dense rankings
    for rank, doc_idx in enumerate(dense_ranks, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))

    # Accumulate score from Sparse rankings
    for rank, doc_idx in enumerate(sparse_ranks, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))

    # Sort documents by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs


def main():
    # --- 1. Dataset Setup ---
    # Knowledge base containing a mix of semantic ideas, technical IDs, and error codes
    corpus = [
        "Error ERR-9021: Database connection timed out after 30 seconds.",
        "How to troubleshoot network connectivity and socket timeout problems.",
        "Employee onboarding guidelines and IT asset collection procedures.",
        "Resetting your single sign-on corporate account password online.",
    ]

    print("--- Knowledge Base Corpus ---")
    for idx, doc in enumerate(corpus):
        print(f"[{idx}] {doc}")
    print()

    # --- 2. Dense Embedding Setup ---
    print("1. Indexing for Dense Vector Search...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_vectors = model.encode(corpus)

    # --- 3. Sparse BM25 Setup ---
    print("2. Indexing for Sparse BM25 Search...")
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- 4. Query Processing ---
    # Query contains an exact error code AND conversational intent
    query = "How do I fix ERR-9021 timeout?"
    print(f"\nTarget Query: '{query}'\n")

    # --- Dense Retrieval Step ---
    query_vector = model.encode(query)
    dense_scores = [
        cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vectors
    ]
    # Get document indices sorted by highest dense score
    dense_ranks = np.argsort(dense_scores)[::-1].tolist()

    print("Dense Search Rankings (Semantic Match):")
    for rank, idx in enumerate(dense_ranks, start=1):
        print(f"  Rank {rank}: Doc [{idx}] (Score: {dense_scores[idx]:.4f})")

    # --- Sparse Retrieval Step ---
    tokenized_query = query.lower().split()
    sparse_scores = bm25.get_scores(tokenized_query)
    # Get document indices sorted by highest BM25 score
    sparse_ranks = np.argsort(sparse_scores)[::-1].tolist()

    print("\nSparse BM25 Search Rankings (Exact Keyword Match):")
    for rank, idx in enumerate(sparse_ranks, start=1):
        print(f"  Rank {rank}: Doc [{idx}] (Score: {sparse_scores[idx]:.4f})")

    # --- 5. Hybrid Fusion Step (RRF) ---
    hybrid_results = reciprocal_rank_fusion(
        dense_ranks, sparse_ranks, k=60
    )

    print("\n================ HYBRID SEARCH RESULTS (RRF) ================")
    for final_rank, (doc_idx, rrf_score) in enumerate(
        hybrid_results, start=1
    ):
        print(f"Rank {final_rank} | RRF Score: {rrf_score:.5f}")
        print(f"       Text: '{corpus[doc_idx]}'")
    print("=============================================================")


if __name__ == "__main__":
    main()

