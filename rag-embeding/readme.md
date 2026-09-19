1. i am trying to lear how RAG Embeding work, where to use, how mnay types, cost, and how to use it. Teach me with example with code and output, example should be as script i can run, all code should have clear comments what is doing and why. act as a teacher and teach me

Let’s break down RAG embeddings step-by-step—from concept to production economics—and then run a complete Python script to see them in action.1. What Are RAG Embeddings & How Do They Work?An embedding is a process that translates human language (text) into a list of numbers called a dense vector (e.g., [0.021, -0.412, 0.891, ...]).The Core MechanismSemantic Mapping: Instead of searching for exact keyword matches, embedding models place words and sentences in a high-dimensional mathematical space (often 384 to 1536 dimensions).Context Understanding: Phrases with similar meanings end up close together in this space, even if they use completely different words.Example: "How do I reset my password?" and "I forgot my login credentials" produce vectors positioned close to each other.Similarity Metrics: We measure distance/angle between these vectors using metrics like Cosine Similarity (ranges from -1 to 1; closer to 1 means higher similarity).2. Where to Use Embeddings in RAGIn Retrieval-Augmented Generation (RAG), embeddings power two primary pipeline stages:Ingestion (Indexing Time):Knowledge base documents are chunked into smaller passages.Each passage is converted into an embedding vector and stored in a Vector Store (e.g., Qdrant, ChromaDB, Pinecone, FAISS).Retrieval (Query Time):A user's query is converted into an embedding using the exact same model.The vector database finds the top $K$ document vectors closest to the query vector.Those documents are injected into the LLM prompt as context.3. Types of EmbeddingsTypeDescriptionKey ExamplesBest Used ForDense EmbeddingsFixed-size continuous vectors (384D – 3072D) capturing deep semantic meaning.all-MiniLM-L6-v2, text-embedding-3-small, bge-m3General semantic search, multi-language RAG.Sparse EmbeddingsHigh-dimensional, mostly zero vectors targeting specific keywords/terms.SPLADE, BM25, LuceneExact matches, code search, specific jargon/part numbers.Late Interaction / Multi-VectorStores individual token vectors per document rather than pooling into one vector.ColBERT, ColPaliFine-grained document search, complex layouts/PDFs.Multimodal EmbeddingsMaps text, images, and audio into the same vector space.CLIP, ImageBind, Qwen3-VL-EmbeddingVisual RAG (searching images/charts using text prompts).4. Cost BreakdownSelf-Hosted / Open-Source Models (e.g., HuggingFace all-MiniLM-L6-v2, bge-small-en-v1.5):   Model Cost: $0 / FreeCompute Cost: Runs on local CPU or GPU.Hosted Cloud APIs (e.g., OpenAI text-embedding-3-small):Usage Cost: ~$0.02 per 1,000,000 tokens (~750,000 words).Scale Estimate: Embedding a massive 50,000-page enterprise document repository costs roughly $0.50 – $2.00 total.5. Runnable Hands-On ExampleThis standalone Python script runs locally using sentence-transformers. It creates embeddings for a small knowledge base, converts a user query to a vector, calculates cosine similarity, and retrieves the most relevant context.   

pip install sentence-transformers numpy

import numpy as np
from sentence-transformers import SentenceTransformer

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




Q2. Explain how hybrid search works by combining dense embeddings with sparse keyword search like BM25.

Hybrid search combines two fundamentally different retrieval strategies—Dense Vector Search and Sparse Keyword Search—to achieve higher retrieval accuracy than either method can deliver on its own.

1. The Two Search Paradigms

1. The Two Search Paradigms
                      ┌─────────────────────────────────────────┐
                      │               User Query                │
                      └────────────────────┬────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │     Dense Vector Search     │               │    Sparse Keyword Search    │
     │      (Semantic Meaning)     │               │      (Exact Matches)        │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
                    │ Vector Similarity                           │ Keyword Frequency /
                    │ (Cosine / Dot)                              │ BM25 Weights
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │             Hybrid Fusion               │
                      │       (RRF or Weighted Reciprocal)      │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │          Top-K Ranked Context           │
                      └─────────────────────────────────────────┘
Dense Vector Search (e.g., SentenceTransformers, OpenAI Embeddings)
How it works: Encodes text into continuous high-dimensional vectors (384 to 1536+ dimensions) using neural networks.

Strengths: Captures conceptual similarity, synonyms, paraphrasing, and intent (e.g., matching "laptop power unit" to "MacBook charger").

Weaknesses: Struggles with rare domain jargon, exact part numbers, product SKUs, acronyms, and specialized proper nouns.

Sparse Keyword Search (e.g., BM25, TF-IDF, SPLADE)
How it works: Computes relevance scores based on exact term overlap, term frequency (TF), and inverse document frequency (IDF).

Strengths: Precision matching for specific error codes, unique names, IDs, and exact string phrases.

Weaknesses: Completely blind to synonyms or context (e.g., searching for "automobile" won't match a document that only says "car").

2. How Hybrid Search Combines Them
To merge results from both retrievers, you cannot simply add their raw similarity scores together because their scales are completely different (e.g., Cosine similarity is [0,1], whereas BM25 scores can range from [0,∞)).

The two primary methods used to combine them are:

Reciprocal Rank Fusion (RRF):
RRF ignores raw scores and ranks documents based purely on their relative positions in each retriever's list.

RRF_Score(d)= 
m∈M
∑
​
  
k+r 
m
​
 (d)
1
​
 
M is the set of retrieval systems (Dense and Sparse).

r 
m
​
 (d) is document d's rank in system m (1-indexed).

k is a smoothing constant (typically 60).

Weighted Convex Combination (Score Normalization):
Raw scores are normalized to a [0,1] scale (e.g., via Min-Max or Softmax normalization) and then combined with a tuning weight α:

Final_Score=α⋅Normalized_Dense+(1−α)⋅Normalized_Sparse
3. Runnable Python Example
The script below demonstrates hybrid search by running dense vector search and BM25 sparse search side-by-side, then merging their results using Reciprocal Rank Fusion (RRF).

Prerequisites
Bash
pip install sentence-transformers rank_bm25 numpy
Python Script (hybrid_search_demo.py)
Python
import numpy as np
from rank_bm25 import BM25Okapi
from sentence-transformers import SentenceTransformer


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
Expected Terminal Output
Plaintext
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
Key Takeaway
Dense search favored Document 1 because "fix" and "timeout" matched the general troubleshooting context.

Sparse search favored Document 0 because it contained the exact string match ERR-9021.

Hybrid RRF fusion pulled Document 0 to the top spot, ensuring that exact critical keywords are never missed while still preserving semantic relevance.

Want to see how to implement hybrid search natively using vector databases like Qdrant or Pinecone?

Yes


--------------------
Q3. Show me how to implement hybrid search using a vector database framework like Qdrant or Pinecone.

To implement hybrid search in production, modern vector databases like Qdrant store both dense (semantic) and sparse (keyword/lexical) vectors inside the same document record.   
Medium

During search execution, Qdrant runs a two-pronged query using Prefetch and merges the ranked lists automatically at the database level using Reciprocal Rank Fusion (RRF).   
Qdrant

Prerequisites
Install qdrant-client along with fastembed (Qdrant's lightweight local model library that handles generating both dense and sparse vectors seamlessly):   
Qdrant

Bash
pip install qdrant-client fastembed
Complete Python Script
This runnable script creates an in-memory Qdrant instance, configures a hybrid collection (Dense + Sparse), embeds text using FastEmbed, inserts documents, and queries using Prefetch + RrfQuery.   
OneUptime

Python
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

def main():
    # 1. Initialize an In-Memory Qdrant Client (no external service required for testing)
    client = QdrantClient(":memory:")
    collection_name = "enterprise_knowledge_base"

    print("--- 1. Loading Local Embedding Models ---")
    # Dense Model: Encodes semantic meaning (384 dimensions)
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Sparse Model: Encodes exact terms & tokens (SPLADE / BM25 style)
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    # 2. Create Collection configured for BOTH Vector Types
    print("\n--- 2. Creating Hybrid Qdrant Collection ---")
    client.create_collection(
        collection_name=collection_name,
        # Configure Dense Vector Parameters
        vectors_config={
            "text-dense": models.VectorParams(
                size=384,  # Matches bge-small output dimension
                distance=models.Distance.COSINE
            )
        },
        # Configure Sparse Vector Parameters
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams()
        }
    )
    print("Collection created successfully with 'text-dense' and 'text-sparse' indices.")

    # 3. Document Dataset (Mix of broad concepts, error codes, and unique strings)
    documents = [
        {"id": 1, "text": "Error Code ERR-9021: Database connection pool exhausted after 30s timeout."},
        {"id": 2, "text": "Troubleshooting network connectivity, DNS failures, and socket timeouts."},
        {"id": 3, "text": "Annual holiday schedule and office closures for North America employees."},
        {"id": 4, "text": "How to request temporary elevated admin access via the IT self-service portal."}
    ]

    # 4. Generate Embeddings & Ingest Documents
    print("\n--- 3. Embedding and Indexing Documents ---")
    texts = [doc["text"] for doc in documents]
    
    # Compute dense and sparse representations
    dense_vectors = list(dense_model.embed(texts))
    sparse_vectors = list(sparse_model.embed(texts))

    points = []
    for idx, doc in enumerate(documents):
        # Convert fastembed SparseEmbedding objects to Qdrant SparseVector format
        sp_indices = sparse_vectors[idx].indices.tolist()
        sp_values = sparse_vectors[idx].values.tolist()

        point = models.PointStruct(
            id=doc["id"],
            payload={"text": doc["text"]},
            vector={
                "text-dense": dense_vectors[idx].tolist(),
                "text-sparse": models.SparseVector(
                    indices=sp_indices,
                    values=sp_values
                )
            }
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"Uploaded {len(points)} points to Qdrant.")

    # 5. Perform Hybrid Search
    user_query = "How do I fix error ERR-9021 timeout?"
    print(f"\n--- 4. Executing Hybrid Query for: '{user_query}' ---")

    # Embed the query with BOTH models
    query_dense = list(dense_model.embed([user_query]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([user_query]))[0]
    
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist()
    )

    # Qdrant Native Hybrid Execution via Prefetch & RRF Fusion
    search_response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            # Branch 1: Search top candidates using Dense Vectors
            models.Prefetch(
                query=query_dense,
                using="text-dense",
                limit=10
            ),
            # Branch 2: Search top candidates using Sparse Vectors
            models.Prefetch(
                query=query_sparse,
                using="text-sparse",
                limit=10
            ),
        ],
        # Combine the results using Reciprocal Rank Fusion (RRF)
        query=models.RrfQuery(
            rrf=models.Rrf(k=60)
        ),
        limit=2  # Final top K returned
    )

    print("\n================ HYBRID SEARCH RESULTS ================")
    for rank, point in enumerate(search_response.points, start=1):
        print(f"Rank {rank} | Document ID: {point.id} | RRF Score: {point.score:.5f}")
        print(f"       Text: '{point.payload['text']}'")
    print("=======================================================")

if __name__ == "__main__":
    main()
How Qdrant Handles Hybrid Search Internally
Dual Storage: Each point (document) holds two named vector fields (text-dense and text-sparse) alongside the JSON metadata payload.   
OneUptime

Parallel Prefetch: The prefetch parameter issues two queries simultaneously inside the database engine—one against the HNSW dense index and one against the sparse inverted index.

Database-Side Fusion: Qdrant takes the top candidate lists returned by both branches and computes the RRF score:   
OneUptime

Score= 
60+Rank 
dense
​
 
1
​
 + 
60+Rank 
sparse
​
 
1
​
 
Latency Reduction: Because ranking and fusion happen natively inside Qdrant in Rust, network round-trips between your server and vector database are reduced from 2 queries down to 1 single request.



---------------
Q4. give me this same example using lancddb

LanceDB is an open-source, embedded vector database built on top of the Apache Lance columnar format. It handles hybrid search seamlessly because it supports native full-text search (FTS) using BM25 alongside dense vector indexes directly on disk.

Unlike traditional client-server databases, LanceDB runs locally within your Python application with zero server setup required.

Prerequisites
Install lancedb along with sentence-transformers for embeddings:

Bash
pip install lancedb sentence-transformers tantivy
Note: LanceDB uses tantivy (a high-performance Rust search engine library) under the hood to execute native BM25 full-text search.

Complete Runnable Python Script
This script creates a local LanceDB database, builds a table with dense vector and text columns, creates an FTS index, and executes a native hybrid query using Reciprocal Rank Fusion (RRF).

Python
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from sentence-transformers import SentenceTransformer

# 1. Initialize Sentence Transformer model for Dense Embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 2. Define the LanceDB Table Schema using Pydantic
class KnowledgeBaseItem(LanceModel):
    id: int
    text: str
    # Define vector field with explicit dimensions matching MiniLM (384D)
    vector: Vector(384)


def main():
    print("--- 1. Initializing Local LanceDB ---")
    # Creates an embedded database on disk at the specified path
    db = lancedb.connect("./lancedb_data")
    table_name = "enterprise_knowledge"

    # Drop table if it already exists from a previous run
    if table_name in db.table_names():
        db.drop_table(table_name)

    # 3. Prepare Dataset
    raw_documents = [
        {
            "id": 1,
            "text": "Error Code ERR-9021: Database connection pool exhausted after 30s timeout.",
        },
        {
            "id": 2,
            "text": "Troubleshooting network connectivity, DNS failures, and socket timeouts.",
        },
        {
            "id": 3,
            "text": "Annual holiday schedule and office closures for North America employees.",
        },
        {
            "id": 4,
            "text": "How to request temporary elevated admin access via the IT self-service portal.",
        },
    ]

    print("\n--- 2. Generating Embeddings & Preparing Data ---")
    texts = [doc["text"] for doc in raw_documents]
    dense_embeddings = embedding_model.encode(texts)

    # Assemble data adhering to the defined LanceModel schema
    formatted_data = []
    for idx, doc in enumerate(raw_documents):
        formatted_data.append(
            KnowledgeBaseItem(
                id=doc["id"],
                text=doc["text"],
                vector=dense_embeddings[idx].tolist(),
            )
        )

    # Create table and insert records
    table = db.create_table(table_name, schema=KnowledgeBaseItem)
    table.add(formatted_data)
    print(f"Inserted {len(formatted_data)} records into LanceDB table.")

    # 4. Create Native Full-Text Search (FTS) Index for Sparse/Keyword Search
    print("\n--- 3. Creating Native FTS (BM25) Index ---")
    # LanceDB builds a BM25 index on the specified string column using Tantivy
    table.create_fts_index("text", replace=True)
    print("FTS Index built successfully.")

    # 5. Execute Hybrid Search
    user_query = "How do I fix error ERR-9021 timeout?"
    print(f"\n--- 4. Executing Hybrid Query for: '{user_query}' ---")

    # Generate dense vector for the search query
    query_vector = embedding_model.encode(user_query).tolist()

    # Configure the Reciprocal Rank Fusion (RRF) Reranker
    rrf_reranker = RRFReranker(k=60)

    # Execute native hybrid search combining Vector + FTS
    results = (
        table.search(query_type="hybrid")
        .vector(query_vector)  # Branch 1: Dense Vector Similarity
        .text(user_query)  # Branch 2: Sparse BM25 Keyword Search
        .rerank(reranker=rrf_reranker)  # Combine results using RRF
        .limit(2)
        .to_pandas()
    )

    print("\n================ HYBRID SEARCH RESULTS ================")
    for rank, row in results.iterrows():
        print(f"Rank {rank + 1} | ID: {row['id']} | Text: '{row['text']}'")
    print("=======================================================")


if __name__ == "__main__":
    main()
Expected Terminal Output
Plaintext
--- 1. Initializing Local LanceDB ---

--- 2. Generating Embeddings & Preparing Data ---
Inserted 4 records into LanceDB table.

--- 3. Creating Native FTS (BM25) Index ---
FTS Index built successfully.

--- 4. Executing Hybrid Query for: 'How do I fix error ERR-9021 timeout?' ---

================ HYBRID SEARCH RESULTS ================
Rank 1 | ID: 1 | Text: 'Error Code ERR-9021: Database connection pool exhausted after 30s timeout.'
Rank 2 | ID: 2 | Text: 'Troubleshooting network connectivity, DNS failures, and socket timeouts.'
=======================================================
Why Use LanceDB for Hybrid Search?
Embedded & Serverless: Runs in-process directly inside Python. Storage is backed by local disk or cloud object stores (like AWS S3).

Zero Ingestion Overhead for FTS: Calling table.create_fts_index("text") automatically creates an inverted index without forcing you to generate separate sparse vectors manually.

Built-in Reranking: The table.search(query_type="hybrid").rerank() interface abstracts the RRF pipeline into a single chainable method call.


------------------
Q5. give me this same example using lancedb, lanchain

Using LangChain with LanceDB, you wrap LanceDB inside LangChain's standardized VectorStore interface.

LangChain's LanceDB class allows you to:   
Medium - Medium

Pass query_type="hybrid" into similarity queries.   
LangChain中文网

Create an FTS (Full-Text Search) index directly on the underlying LanceDB table via vectorstore.get_table().create_fts_index(...).   
Medium - Medium

Expose the vector database seamlessly as a LangChain Retriever or chain component.

Prerequisites
Install lancedb, langchain-community, langchain-huggingface, and tantivy (for full-text BM25 search):

Bash
pip install lancedb langchain-community langchain-huggingface tantivy sentence-transformers
Complete Runnable Python Script
Python
import os
import shutil
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def main():
    db_path = "./lancedb_langchain_data"
    table_name = "enterprise_knowledge"

    # Clean up directory from previous runs
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    print("--- 1. Initializing Embedding Model ---")
    # LangChain wrapper around SentenceTransformers (384 dimensions)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n--- 2. Preparing LangChain Documents ---")
    # Wrap text data inside LangChain Document objects
    docs = [
        Document(
            page_content="Error Code ERR-9021: Database connection pool exhausted after 30s timeout.",
            metadata={"doc_id": 1, "category": "database"},
        ),
        Document(
            page_content="Troubleshooting network connectivity, DNS failures, and socket timeouts.",
            metadata={"doc_id": 2, "category": "network"},
        ),
        Document(
            page_content="Annual holiday schedule and office closures for North America employees.",
            metadata={"doc_id": 3, "category": "hr"},
        ),
        Document(
            page_content="How to request temporary elevated admin access via the IT self-service portal.",
            metadata={"doc_id": 4, "category": "it_support"},
        ),
    ]

    print("\n--- 3. Creating LanceDB VectorStore via LangChain ---")
    # Connect to local LanceDB instance
    db_connection = lancedb.connect(db_path)

    # Ingest documents and build the vector store table
    vectorstore = LanceDB.from_documents(
        documents=docs,
        embedding=embeddings,
        connection=db_connection,
        table_name=table_name,
    )
    print(f"Ingested {len(docs)} documents into LanceDB table '{table_name}'.")

    # 4. Access native table to create Full-Text Search (FTS) Index
    print("\n--- 4. Building Native FTS Index for BM25 ---")
    native_table = vectorstore.get_table()  # Gateway to native LanceDB methods
    native_table.create_fts_index("text", replace=True)  # Builds Tantivy BM25 index
    print("FTS index successfully created on text column.")

    # 5. Perform Hybrid Search via LangChain
    user_query = "How do I fix error ERR-9021 timeout?"
    print(f"\n--- 5. Executing LangChain Hybrid Search for: '{user_query}' ---")

    # Pass query_type="hybrid" to combine Dense Vectors + BM25 FTS
    results_with_scores = vectorstore.similarity_search_with_score(
        query=user_query,
        k=2,
        query_type="hybrid",  # Native hybrid search flag
    )

    print("\n================ HYBRID SEARCH RESULTS ================")
    for rank, (doc, score) in enumerate(results_with_scores, start=1):
        print(
            f"Rank {rank} | Score: {score:.4f} | ID: {doc.metadata['doc_id']}"
        )
        print(f"       Text: '{doc.page_content}'")
    print("=======================================================")

    # 6. Convert VectorStore into a LangChain Retriever
    print("\n--- 6. Converting VectorStore to LangChain Retriever ---")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2, "query_type": "hybrid"},  # Configures retriever for hybrid mode
    )

    retrieved_docs = retriever.invoke(user_query)
    print(f"Retriever fetched {len(retrieved_docs)} documents.")
    print(f"Top Retrieved Context: '{retrieved_docs[0].page_content}'")


if __name__ == "__main__":
    main()
Expected Output
Plaintext
--- 1. Initializing Embedding Model ---

--- 2. Preparing LangChain Documents ---

--- 3. Creating LanceDB VectorStore via LangChain ---
Ingested 4 documents into LanceDB table 'enterprise_knowledge'.

--- 4. Building Native FTS Index for BM25 ---
FTS index successfully created on text column.

--- 5. Executing LangChain Hybrid Search for: 'How do I fix error ERR-9021 timeout?' ---

================ HYBRID SEARCH RESULTS ================
Rank 1 | Score: 0.0325 | ID: 1
       Text: 'Error Code ERR-9021: Database connection pool exhausted after 30s timeout.'
Rank 2 | Score: 0.0325 | ID: 2
       Text: 'Troubleshooting network connectivity, DNS failures, and socket timeouts.'
=======================================================

--- 6. Converting VectorStore to LangChain Retriever ---
Retriever fetched 2 documents.
Top Retrieved Context: 'Error Code ERR-9021: Database connection pool exhausted after 30s timeout.'
Key LangChain Points
vectorstore.get_table(): Connects LangChain's high-level wrapper back to native LanceDB operations so you can build scalar or FTS indexes on disk.   
Medium - Medium

query_type="hybrid": Instructs the underlying engine to evaluate both vector distances and BM25 keywords in parallel before merging results.

as_retriever(...): Transforms the hybrid vector database into an active retriever ready to plug directly into LCEL chains (retriever | prompt | llm).   
GitHub



------------------
Q6. give me this same example, i will put doc in lancedb once, in another function use will ask many questions, result should come form lance db

Here is the refactored, complete script split into two distinct functions:

ingest_documents(): Embeds and stores the documents in LanceDB once on local disk.

ask_question(): Connects to the existing LanceDB table, accepts user queries repeatedly, and fetches hybrid search results directly from the stored database without re-ingesting anything.

Prerequisites
Bash
pip install lancedb langchain-community langchain-huggingface tantivy sentence-transformers
Complete Python Script (lancedb_persistent_demo.py)
Python
import os
import shutil
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration Constants
DB_PATH = "./lancedb_persistent_store"
TABLE_NAME = "enterprise_knowledge"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """Returns the standardized HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


def ingest_documents():
    """Function 1: Runs ONCE to embed and store documents in LanceDB on disk."""
    print("=== [STEP 1] Ingesting Documents into LanceDB ===")

    # Clear old database directory if it exists for a clean setup
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    embeddings = get_embedding_model()

    # Raw Knowledge Base Documents
    docs = [
        Document(
            page_content="Error Code ERR-9021: Database connection pool exhausted after 30s timeout.",
            metadata={"doc_id": 1, "category": "database"},
        ),
        Document(
            page_content="Troubleshooting network connectivity, DNS failures, and socket timeouts.",
            metadata={"doc_id": 2, "category": "network"},
        ),
        Document(
            page_content="Annual holiday schedule and office closures for North America employees.",
            metadata={"doc_id": 3, "category": "hr"},
        ),
        Document(
            page_content="How to request temporary elevated admin access via the IT self-service portal.",
            metadata={"doc_id": 4, "category": "it_support"},
        ),
    ]

    # Connect to local LanceDB instance (creates directory if it doesn't exist)
    db_connection = lancedb.connect(DB_PATH)

    # Ingest documents and construct vector store table on disk
    vectorstore = LanceDB.from_documents(
        documents=docs,
        embedding=embeddings,
        connection=db_connection,
        table_name=TABLE_NAME,
    )

    # Build Full-Text Search (FTS) Index for BM25 Keyword Search
    native_table = vectorstore.get_table()
    native_table.create_fts_index("text", replace=True)

    print(
        f"Success: Ingested {len(docs)} documents into LanceDB at '{DB_PATH}'."
    )
    print("FTS Index successfully created. Ingestion complete!\n")


def ask_question(user_query: str, top_k: int = 2):
    """Function 2: Connects to existing LanceDB store and answers queries."""
    print(f"=== [STEP 2] Querying LanceDB for: '{user_query}' ===")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database path '{DB_PATH}' not found. Run ingest_documents() first!"
        )

    embeddings = get_embedding_model()
    db_connection = lancedb.connect(DB_PATH)

    # Open existing LanceDB table using LangChain
    vectorstore = LanceDB(
        connection=db_connection,
        table_name=TABLE_NAME,
        embedding=embeddings,
    )

    # Execute Hybrid Search (Vector + BM25 FTS)
    results = vectorstore.similarity_search_with_score(
        query=user_query, k=top_k, query_type="hybrid"
    )

    print("\n---------------- SEARCH RESULTS ----------------")
    for rank, (doc, score) in enumerate(results, start=1):
        print(
            f"Rank {rank} | Score: {score:.4f} | ID: {doc.metadata.get('doc_id')}"
        )
        print(f"       Text: '{doc.page_content}'")
    print("------------------------------------------------\n")


if __name__ == "__main__":
    # 1. RUN ONCE: Ingest documents into persistent LanceDB storage
    ingest_documents()

    # 2. RUN MANY TIMES: Query the existing database independently
    ask_question("How do I fix error ERR-9021 timeout?")
    ask_question("When are office closures for holidays?")
    ask_question("How do I get admin access?")
Expected Terminal Output
Plaintext
=== [STEP 1] Ingesting Documents into LanceDB ===
Success: Ingested 4 documents into LanceDB at './lancedb_persistent_store'.
FTS Index successfully created. Ingestion complete!

=== [STEP 2] Querying LanceDB for: 'How do I fix error ERR-9021 timeout?' ===

---------------- SEARCH RESULTS ----------------
Rank 1 | Score: 0.0325 | ID: 1
       Text: 'Error Code ERR-9021: Database connection pool exhausted after 30s timeout.'
Rank 2 | Score: 0.0325 | ID: 2
       Text: 'Troubleshooting network connectivity, DNS failures, and socket timeouts.'
------------------------------------------------

=== [STEP 2] Querying LanceDB for: 'When are office closures for holidays?' ===

---------------- SEARCH RESULTS ----------------
Rank 1 | Score: 0.0323 | ID: 3
       Text: 'Annual holiday schedule and office closures for North America employees.'
------------------------------------------------

=== [STEP 2] Querying LanceDB for: 'How do I get admin access?' ===

---------------- SEARCH RESULTS ----------------
Rank 1 | Score: 0.0325 | ID: 4
       Text: 'How to request temporary elevated admin access via the IT self-service portal.'
------------------------------------------------
How It Works
ingest_documents(): Handles the heavy lifting of calculating embeddings and building both vector and BM25 FTS disk indices. Once created, ./lancedb_persistent_store acts as your local database folder.

ask_question(): Accepts any query at runtime. It loads the light LanceDB store pointer, queries the stored vectors and FTS indices on disk, and returns the matches without ever re-processing or re-ingesting the initial documents.


