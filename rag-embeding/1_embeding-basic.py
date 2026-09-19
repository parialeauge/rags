#!/usr/bin/env python3
"""Basic local embedding + cosine similarity RAG demo.
 ---->> install:  pip install sentence-transformers numpy
 ---->> run:      conda activate rags && python 1_embeding-basic.py

Use conda env `rags` (not Anaconda base / Python 3.13):
  conda activate rags
  python 1_embeding-basic.py

Also works with the project venv:
  cd .. && source .venv-rag/bin/activate && cd rag-embeding
  python 1_embeding-basic.py

Or without activating:
  ../.venv-rag/bin/python 1_embeding-basic.py

#######--- output ---#######
--- 1. Loading Embedding Model ---
Model loaded successfully!

--- 3. Generating Document Embeddings (Ingestion Stage) ---
Number of documents indexed: 4
Dimensions per embedding vector: 384
Sample raw values from Document 0 (first 5 dimensions):
  [-0.03842183  0.08129034 -0.01239102  0.04510982 -0.09102831]

--- 4. Process User Query (Retrieval Stage) ---
User Query: 'I locked myself out of my account and need a new password'

--- 5. Computing Vector Similarities ---
Doc 0 Similarity: 0.6841 -> 'To reset your corporate password, visit https://id...'
Doc 1 Similarity: 0.0812 -> 'The company annual holiday party takes place on Dec...'
Doc 2 Similarity: 0.2104 -> 'Remote work policies require employees to connect v...'
Doc 3 Similarity: 0.1193 -> 'To submit travel expense reports, use the Concur p...'

================ RAG RETRIEVAL RESULT ================
Top Matching Doc Index : 0
Similarity Confidence  : 68.41%
Retrieved Context Text : 'To reset your corporate password, visit https://identity.company.com and click 'Forgot Credentials'.'
======================================================

--- Final Prompt Prepared for LLM ---
System: Answer the question using ONLY the provided context.
Context: To reset your corporate password, visit https://identity.company.com and click 'Forgot Credentials'.
Question: I locked myself out of my account and need a new password
Answer:

"""

import sys

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    # Common cause: Anaconda Python 3.13 (no torch wheels). Use the 3.12 env instead.
    sys.exit(
        f"Missing dependency: {exc}\n"
        "You are not in the `rags` conda env. Run:\n"
        "  conda activate rags\n"
        "  python 1_embeding-basic.py"
    )

def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two 1D vectors.
    Formula: (A . B) / (||A|| * ||B||)
    Returns a score between -1.0 (opposite) and 1.0 (identical direction).
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return float(dot_product / (norm_a * norm_b))

def main():
    print("--- 1. Loading Embedding Model ---")
    # Load a small, fast local embedding model (384 dimensions)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Model loaded successfully!\n")

    # --- 2. Define Knowledge Base (Documents) ---
    knowledge_base = [
        "To reset your corporate password, visit https://identity.company.com and click 'Forgot Credentials'.",
        "The company annual holiday party takes place on December 18th in the main cafeteria.",
        "Remote work policies require employees to connect via the VPN when accessing internal servers.",
        "To submit travel expense reports, use the Concur portal before the 25th of each month."
    ]

    print("--- 3. Generating Document Embeddings (Ingestion Stage) ---")
    # Convert all documents into 384-dimensional vector embeddings
    doc_embeddings = model.encode(knowledge_base)
    
    print(f"Number of documents indexed: {len(doc_embeddings)}")
    print(f"Dimensions per embedding vector: {doc_embeddings[0].shape[0]}")
    print(f"Sample raw values from Document 0 (first 5 dimensions):\n  {doc_embeddings[0][:5]}\n")

    # --- 4. Process User Query (Retrieval Stage) ---
    user_query = "I locked myself out of my account and need a new password"
    print(f"User Query: '{user_query}'")

    # Embed the query using the EXACT SAME model
    query_embedding = model.encode(user_query)

    print("\n--- 5. Computing Vector Similarities ---")
    scores = []
    for idx, doc_emb in enumerate(doc_embeddings):
        # Measure distance/similarity between query vector and each doc vector
        similarity = calculate_cosine_similarity(query_embedding, doc_emb)
        scores.append((similarity, idx))
        print(f"Doc {idx} Similarity: {similarity:.4f} -> '{knowledge_base[idx][:50]}...'")

    # Sort documents by similarity score in descending order
    scores.sort(key=lambda x: x[0], reverse=True)

    # --- 6. Retrieve Top Match for RAG Context ---
    top_score, top_doc_index = scores[0]
    retrieved_context = knowledge_base[top_doc_index]

    print("\n================ RAG RETRIEVAL RESULT ================")
    print(f"Top Matching Doc Index : {top_doc_index}")
    print(f"Similarity Confidence  : {top_score * 100:.2f}%")
    print(f"Retrieved Context Text : '{retrieved_context}'")
    print("======================================================\n")

    # Simulated LLM Prompt Preparation
    llm_prompt = f"""
System: Answer the question using ONLY the provided context.
Context: {retrieved_context}
Question: {user_query}
Answer:
"""
    print("--- Final Prompt Prepared for LLM ---")
    print(llm_prompt.strip())

if __name__ == "__main__":
    main()

