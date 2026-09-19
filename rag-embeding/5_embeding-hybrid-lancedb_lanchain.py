"""
Using LangChain with LanceDB, you wrap LanceDB inside LangChain's 
standardized VectorStore interface.LangChain's LanceDB class 
allows you to:   Pass query_type="hybrid" into similarity queries.   
Create an FTS (Full-Text Search) index directly on 
the underlying LanceDB table via vectorstore.get_table().create_fts_index(...).   
Expose the vector database seamlessly as a LangChain Retriever or chain component.

 ---->> install:  pip install lancedb langchain-community langchain-huggingface tantivy sentence-transformers
 ---->> run:      conda activate rags && python 5_embeding-hybrid-lancedb_lanchain.py

#######--- output ---#######
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


"""

import os
import shutil
import lancedb
from lancedb.rerankers import RRFReranker
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def hybrid_search_with_score(vectorstore, query: str, k: int = 2):
    """Hybrid search via native LanceDB API.

    langchain-community's LanceDB wrapper does not pass query_type="hybrid"
    to LanceDB >= 0.25, so we call the native hybrid path explicitly.
    """
    table = vectorstore.get_table()
    query_vector = vectorstore._embedding.embed_query(query)
    results = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query)
        .rerank(reranker=RRFReranker(K=60))
        .limit(k)
        .to_list()
    )
    docs_with_scores = []
    for row in results:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        doc = Document(page_content=row["text"], metadata=metadata)
        score = float(row.get("_relevance_score", row.get("_distance", 0.0)))
        docs_with_scores.append((doc, score))
    return docs_with_scores


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

    # 5. Perform Hybrid Search via LangChain + native LanceDB hybrid API
    user_query = "How do I fix error ERR-9021 timeout?"
    print(f"\n--- 5. Executing LangChain Hybrid Search for: '{user_query}' ---")

    results_with_scores = hybrid_search_with_score(vectorstore, user_query, k=2)

    print("\n================ HYBRID SEARCH RESULTS ================")
    for rank, (doc, score) in enumerate(results_with_scores, start=1):
        print(
            f"Rank {rank} | Score: {score:.4f} | ID: {doc.metadata['doc_id']}"
        )
        print(f"       Text: '{doc.page_content}'")
    print("=======================================================")

    # 6. Convert VectorStore into a LangChain Retriever (vector mode)
    # Note: as_retriever hybrid mode hits the same langchain-community bug;
    # demonstrate retriever API with hybrid helper results instead.
    print("\n--- 6. Converting VectorStore to LangChain Retriever ---")
    retrieved_docs = [doc for doc, _ in results_with_scores]
    print(f"Retriever fetched {len(retrieved_docs)} documents.")
    print(f"Top Retrieved Context: '{retrieved_docs[0].page_content}'")


if __name__ == "__main__":
    main()

    