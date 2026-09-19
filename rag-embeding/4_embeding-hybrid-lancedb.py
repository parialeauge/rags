"""
LanceDB is an open-source, embedded vector database built on top of the 
Apache Lance columnar format. It handles hybrid search seamlessly because 
it supports native full-text search (FTS) using BM25 alongside 
dense vector indexes directly on disk.

 ---->> install:  pip install lancedb sentence-transformers tantivy pandas
 ---->> run:      conda activate rags && python 4_embeding-hybrid-lancedb.py

#######--- output ---#######
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
"""


import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from sentence_transformers import SentenceTransformer

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
    rrf_reranker = RRFReranker(K=60)

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

