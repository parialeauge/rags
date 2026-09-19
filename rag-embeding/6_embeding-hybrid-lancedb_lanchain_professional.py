"""
Using LangChain with LanceDB, you wrap LanceDB inside LangChain's 
standardized VectorStore interface.LangChain's LanceDB class 
allows you to:   Pass query_type="hybrid" into similarity queries.   
Create an FTS (Full-Text Search) index directly on 
the underlying LanceDB table via vectorstore.get_table().create_fts_index(...).   
Expose the vector database seamlessly as a LangChain Retriever or chain component.

efactored, complete script split into two distinct functions:
ingest_documents(): Embeds and stores the documents in LanceDB once on local disk.
ask_question(): Connects to the existing LanceDB table, accepts user queries repeatedly, 
and fetches hybrid search results directly from the stored database 
without re-ingesting anything.


 ---->> install:  pip install lancedb langchain-community langchain-huggingface tantivy sentence-transformers
 ---->> run:      conda activate rags && python 6_embeding-hybrid-lancedb_lanchain_professional.py

#######--- output ---#######
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


"""


import os
import shutil
import lancedb
from lancedb.rerankers import RRFReranker
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


def hybrid_search_with_score(vectorstore, query: str, k: int = 2):
    """Hybrid search via native LanceDB API (LanceDB >= 0.25 compatible)."""
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

    # Execute Hybrid Search (Vector + BM25 FTS) via native LanceDB API
    results = hybrid_search_with_score(vectorstore, user_query, k=top_k)

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

    