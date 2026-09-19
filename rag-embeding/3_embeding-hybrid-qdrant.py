"""
To implement hybrid search in production, modern vector databases 
like Qdrant store both dense (semantic) and sparse (keyword/lexical) 
vectors inside the same document record.   During search execution, 
Qdrant runs a two-pronged query using Prefetch and merges the ranked 
lists automatically at the database level using Reciprocal Rank Fusion (RRF).

 ---->> install:  pip install qdrant-client fastembed
 ---->> run:      conda activate rags && python 3_embeding-hybrid-qdrant.py

#######--- output ---#######
--- 1. Loading Local Embedding Models ---
Collection created successfully with 'text-dense' and 'text-sparse' indices.

--- 3. Embedding and Indexing Documents ---
Uploaded 4 points to Qdrant.

--- 4. Executing Hybrid Query for: 'How do I fix error ERR-9021 timeout?' ---

================ HYBRID SEARCH RESULTS ===============
Rank 1 | Document ID: 1 | RRF Score: 0.03252
       Text: 'Error Code ERR-9021: Database connection pool exhausted after 30s timeout.'
Rank 2 | Document ID: 2 | RRF Score: 0.03252
       Text: 'Troubleshooting network connectivity, DNS failures, and socket timeouts.'
======================================================


"""

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

