"""
the complete implementation combining Native Hybrid Search 
(Dense Vectors + BM25 Full-Text Search) with Role-Based Access Control (RBAC) 
metadata filtering using LangChain and LanceDB.

- SQL Pre-Filtering: LanceDB applies the SQL WHERE predicate first to eliminate unauthorized rows.
- Parallel Dual Execution: The search executes Dense Vector similarity and BM25 keyword search exclusively on the authorized subset of documents.
- Reciprocal Rank Fusion (RRF): The dense and sparse candidate lists are merged and re-ranked before context is returned.

 ---->> install:  pip install -r requirements.txt
 ---->> run:      conda activate rags && python 7_embeding-hybrid-lancedb_lanchainLCEL_metadata.py
 ---->> env:      put HF_TOKEN=hf_... in feature-arun/.env (loaded automatically)

#######--- output ---#######
=== [STEP 1] Ingesting Classified Documents into LanceDB ===
Success: Ingested 4 documents into LanceDB at './langchain_hybrid_rbac_store'.
FTS Index successfully created. Ingestion complete!

================ SCENARIO 1: INTERN ROLE ================
Executing Search | Query: 'How do I fix error ERR-9021 timeout?' | User Role: 'intern'
  RESULT: Info not found or not accessible for your role.

================ SCENARIO 2: ENGINEER ROLE ================
Executing Search | Query: 'How do I fix error ERR-9021 timeout?' | User Role: 'engineer'
  RESULT: Matches Found:
    [1] Engineering Troubleshooting Guide (RRF Score: 0.0325 | Category: Engineering)
        Content: 'Critical Fix for ERR-9021: Database connection pool exhausted. Increase max connections to 100.'
        Allowed Roles: [admin,manager,engineer]

================ SCENARIO 3: FINANCE ROLE ================
Executing Search | Query: 'What was our EBITDA and financial margin?' | User Role: 'finance'
  RESULT: Matches Found:
    [1] Q3 Financial Performance Report (RRF Score: 0.0325 | Category: Finance)
        Content: 'Net operational margin increased by 14% with EBITDA exceeding internal projections.'
        Allowed Roles: [admin,executive,finance]



"""


import os
import shutil
import sys
from pathlib import Path

# Fail fast if run under Anaconda base (Python 3.13) without sentence-transformers/torch.
# Correct env example:
#   conda activate rags
#   python 7_embeding-hybrid-lancedb_lanchainLCEL_metadata.py
try:
    import lancedb
    from lancedb.rerankers import RRFReranker
    from langchain_community.vectorstores import LanceDB
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_huggingface import HuggingFaceEmbeddings
    import sentence_transformers  # noqa: F401  # required by HuggingFaceEmbeddings
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}\n\n"
        "You are not in the `rags` conda env (Anaconda base cannot install torch).\n"
        "Run:\n"
        "  conda activate rags\n"
        "  python 7_embeding-hybrid-lancedb_lanchainLCEL_metadata.py\n"
    )

# ---------------------------------------------------------------------------
# CONFIGURATION
# Example layout after ingestion:
#   ./lcel_rbac_store/
#     enterprise_knowledge.lance/   <-- vectors + text + metadata
# ---------------------------------------------------------------------------
DB_PATH = "./lcel_rbac_store"  # local folder where LanceDB stores tables on disk
TABLE_NAME = "enterprise_knowledge"  # table name inside that folder
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim dense vectors
# Instruct LLM called over Hugging Face Inference API (needs HF_TOKEN)
HF_LLM_REPO_ID = "microsoft/Phi-3-mini-4k-instruct"


def _load_dotenv() -> None:
    """Load HF_TOKEN (and other keys) from a nearby .env without overwriting exports.

    Example feature-arun/.env line:
        HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    """
    # Search common locations so the script works from different cwd's
    candidates = [
        Path.cwd() / ".env",  # e.g. .../rag-embeding/.env
        Path.cwd().parent / ".env",  # e.g. .../rags/.env
        Path(__file__).resolve().parents[2] / ".env",  # .../feature-arun/.env
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            # Skip blanks, comments, and lines without KEY=VALUE
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)  # split only on first '='
            key, value = key.strip(), value.strip().strip('"').strip("'")
            # setdefault: do not overwrite a token already exported in the shell
            os.environ.setdefault(key, value)


def get_embedding_model():
    """Returns standardized HuggingFace embedding model.

    Example:
        emb = get_embedding_model()
        vec = emb.embed_query("reset password")  # -> list[float] length 384
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def hybrid_search_with_score(
    vectorstore,
    query: str,
    k: int = 2,
    sql_filter: str | None = None,
):
    """Hybrid search via native LanceDB API with optional SQL pre-filter.

    Flow:
      1) Optional SQL WHERE (RBAC) removes unauthorized rows first (prefilter)
      2) Dense vector search (semantic meaning)
      3) Sparse BM25 / FTS search (exact keywords like ERR-9021)
      4) RRF merges both ranked lists into one score

    Example call:
        hits = hybrid_search_with_score(
            vectorstore,
            query="How do I fix ERR-9021?",
            k=2,
            sql_filter="metadata.allowed_roles LIKE '%engineer%'",
        )
        # hits -> [(Document(...), 0.0328), ...]

    Note: langchain-community does not pass query_type="hybrid" to LanceDB >= 0.25,
    so we call the native hybrid API explicitly.
    """
    # Native LanceDB table object (not the LangChain wrapper)
    table = vectorstore.get_table()

    # Convert the user question into a dense vector with the SAME model used at ingest
    # Example: "fix ERR-9021" -> [0.01, -0.22, ...] (384 floats)
    query_vector = vectorstore._embedding.embed_query(query)

    # Build hybrid query: dense branch (.vector) + sparse branch (.text)
    search = (
        table.search(query_type="hybrid")
        .vector(query_vector)  # semantic similarity branch
        .text(query)  # BM25 keyword branch (uses FTS index on "text")
        .rerank(reranker=RRFReranker(K=60))  # Reciprocal Rank Fusion merge
        .limit(k)  # return top-k fused results
    )

    if sql_filter:
        # Example filter for role="engineer":
        #   metadata.allowed_roles LIKE '%engineer%'
        # prefilter=True => apply WHERE BEFORE scoring (security boundary)
        # Metadata is a nested struct from LangChain's LanceDB wrapper
        search = search.where(sql_filter, prefilter=True)

    docs_with_scores = []
    for row in search.to_list():
        # row example:
        # {
        #   "text": "Critical Fix for ERR-9021...",
        #   "metadata": {"doc_id": "DOC-202", "allowed_roles": "admin,manager,engineer", ...},
        #   "_relevance_score": 0.0328,
        #   "vector": [...],
        # }
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = dict(metadata)

        # Wrap LanceDB row as a LangChain Document for the LCEL pipeline
        doc = Document(page_content=row["text"], metadata=metadata)
        score = float(row.get("_relevance_score", row.get("_distance", 0.0)))

        # Drop weak RRF matches so RBAC-filtered queries don't return unrelated docs
        # Example: intern asking about ERR-9021 may only see HR docs with ~0.016 score
        #          -> filtered out; engineer sees ERR-9021 doc with ~0.032 -> kept
        if score < 0.02:
            continue
        docs_with_scores.append((doc, score))
    return docs_with_scores


# ----------------------------------------------------------------------
# FUNCTION 1: INGESTION & FTS INDEXING (RUN ONCE)
# ----------------------------------------------------------------------
def ingest_classified_documents():
    """Ingests documents with role metadata and constructs vector + BM25 indices.

    Example after this runs:
      - 4 rows stored with dense vectors
      - each row has metadata.allowed_roles used later for RBAC filtering
      - FTS index on column "text" enables BM25 keyword search
    """
    print("=== [STEP 1] Ingesting Classified Documents into LanceDB ===")

    # Clean previous demo DB so each run starts from a known state
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    embeddings = get_embedding_model()

    # Each Document = page_content (searchable text) + metadata (RBAC + labels)
    # Example RBAC:
    #   intern  can see DOC-101 only
    #   engineer can see DOC-202 (and not finance/executive)
    docs = [
        Document(
            page_content="Standard company working hours are 9 AM to 5 PM with flexible remote work policies.",
            metadata={
                "doc_id": "DOC-101",
                "title": "Public Employee Handbook",
                "category": "HR",
                "allowed_roles": "admin,manager,intern,employee",  # CSV of allowed roles
            },
        ),
        Document(
            page_content="Critical Fix for ERR-9021: Database connection pool exhausted. Increase max connections to 100 in config.yaml.",
            metadata={
                "doc_id": "DOC-202",
                "title": "Engineering Troubleshooting Guide",
                "category": "Engineering",
                "allowed_roles": "admin,manager,engineer",
            },
        ),
        Document(
            page_content="Net operational margin increased by 14% with EBITDA exceeding internal projections.",
            metadata={
                "doc_id": "DOC-303",
                "title": "Q3 Financial Performance Report",
                "category": "Finance",
                "allowed_roles": "admin,executive,finance",
            },
        ),
        Document(
            page_content="Restricted Stock Units (RSUs) vesting schedule for C-suite personnel starts in Q4.",
            metadata={
                "doc_id": "DOC-404",
                "title": "Executive Compensation Strategy",
                "category": "Executive",
                "allowed_roles": "admin,executive",
            },
        ),
    ]

    # Open / create the on-disk LanceDB database folder
    db_connection = lancedb.connect(DB_PATH)

    # Embed each Document.page_content and write rows into the named table
    vectorstore = LanceDB.from_documents(
        documents=docs,
        embedding=embeddings,
        connection=db_connection,
        table_name=TABLE_NAME,
    )

    # Build BM25 Full-Text Search index on the "text" column (Tantivy under the hood)
    # Required for the sparse/keyword half of hybrid search
    native_table = vectorstore.get_table()
    native_table.create_fts_index("text", replace=True)

    print(
        f"Success: Ingested {len(docs)} documents into LanceDB at '{DB_PATH}'."
    )
    print("FTS Index successfully created. Ingestion complete!\n")


# ----------------------------------------------------------------------
# HELPER: CONTEXT FORMATTER
# ----------------------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    """Formats retrieved documents into one context string for the LLM prompt.

    Example input docs:
        [Document(page_content="Critical Fix for ERR-9021...", metadata={"title": "Engineering Troubleshooting Guide", ...})]

    Example output string:
        [Source: Engineering Troubleshooting Guide | Category: Engineering]
        Critical Fix for ERR-9021...

    If no docs passed RBAC + relevance filters -> sentinel string below.
    """
    if not docs:
        # Sentinel the prompt tells the LLM to treat as "deny / not found"
        return "NO_ACCESSIBLE_CONTEXT_FOUND"

    formatted = []
    for doc in docs:
        source_info = (
            f"[Source: {doc.metadata.get('title', 'Unknown')} | "
            f"Category: {doc.metadata.get('category', 'General')}]"
        )
        formatted.append(f"{source_info}\n{doc.page_content}")

    return "\n\n".join(formatted)


def get_llm():
    """Hugging Face Inference chat model when HF_TOKEN is set; else a local stub.

    Example with token:
        llm = get_llm()  # ChatHuggingFace(Phi-3-mini...)
        answer = llm.invoke("Say hello")

    Example without token:
        llm = get_llm()  # RunnableLambda stub that echoes context / denial
    """
    _load_dotenv()
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        print(
            "(No HF_TOKEN set — using local stub LLM that answers from context only.)\n"
            "Add HF_TOKEN to feature-arun/.env to use Hugging Face Inference.\n"
        )

        def stub_llm(prompt_value):
            # prompt_value is usually a ChatPromptValue from ChatPromptTemplate
            text = (
                prompt_value.to_string()
                if hasattr(prompt_value, "to_string")
                else str(prompt_value)
            )
            # Only inspect the runtime Context block, not the instruction text
            # (instructions also mention NO_ACCESSIBLE_CONTEXT_FOUND as a rule)
            context = ""
            if "Context:\n" in text and "\n\nQuestion:" in text:
                context = text.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0].strip()
            if not context or context == "NO_ACCESSIBLE_CONTEXT_FOUND":
                return (
                    "I cannot answer this request because the information does not "
                    "exist or you do not have permission to access it."
                )
            # Demo fallback: return retrieved context as the "answer"
            return context

        # Wrap plain function so it can sit in an LCEL chain like a real chat model
        return RunnableLambda(stub_llm)

    # LangChain / huggingface_hub conventionally read this env var name
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    print(f"(Using Hugging Face Inference LLM: {HF_LLM_REPO_ID})\n")
    # HuggingFaceEndpoint = remote text-generation API for a model repo
    endpoint = HuggingFaceEndpoint(
        repo_id=HF_LLM_REPO_ID,  # e.g. microsoft/Phi-3-mini-4k-instruct
        task="text-generation",
        max_new_tokens=256,  # cap answer length
        temperature=0.1,  # low = more deterministic / less creative
        do_sample=False,
        huggingfacehub_api_token=hf_token,
    )
    # ChatHuggingFace adapts the endpoint to LangChain's chat interface
    return ChatHuggingFace(llm=endpoint)


# ----------------------------------------------------------------------
# FUNCTION 2: SECURE RAG LCEL PIPELINE (RUN REPEATEDLY)
# ----------------------------------------------------------------------
def generate_secure_answer(user_query: str, user_role: str):
    """Constructs a role-gated LCEL pipeline and generates an LLM answer.

    Example:
        generate_secure_answer(
            user_query="How do I fix error ERR-9021 timeout?",
            user_role="engineer",
        )
        # -> LLM answers from Engineering Troubleshooting Guide

        generate_secure_answer(
            user_query="How do I fix error ERR-9021 timeout?",
            user_role="intern",
        )
        # -> denial (engineering doc blocked by RBAC filter)
    """
    print(
        f"=== Generating RAG Answer | Query: '{user_query}' | User Role: '{user_role}' ==="
    )

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database path '{DB_PATH}' not found. Run ingest_classified_documents() first!"
        )

    # 1) Re-open the existing on-disk LanceDB table (no re-ingest)
    embeddings = get_embedding_model()
    db_connection = lancedb.connect(DB_PATH)
    vectorstore = LanceDB(
        connection=db_connection,
        table_name=TABLE_NAME,
        embedding=embeddings,  # needed to embed the query at search time
    )

    # 2) Role-specific SQL pre-filter for RBAC
    # Example for user_role="finance":
    #   metadata.allowed_roles LIKE '%finance%'
    # matches "admin,executive,finance" but not "admin,manager,engineer"
    sql_filter = f"metadata.allowed_roles LIKE '%{user_role}%'"

    def retrieve(query: str) -> list[Document]:
        # Closure captures vectorstore + sql_filter for this user_role
        hits = hybrid_search_with_score(
            vectorstore, query, k=2, sql_filter=sql_filter
        )
        # Drop scores; LCEL formatter only needs Document objects
        return [doc for doc, _ in hits]

    # RunnableLambda lets a plain Python function act as a LangChain "retriever"
    retriever = RunnableLambda(retrieve)

    # 3) Prompt template with placeholders filled by LCEL:
    #    {context}  <- retrieved docs (or NO_ACCESSIBLE_CONTEXT_FOUND)
    #    {question} <- original user_query
    prompt_template = """You are a secure corporate AI assistant. Answer the user's question using ONLY the provided context below.

CRITICAL INSTRUCTIONS:
- If the context is exactly 'NO_ACCESSIBLE_CONTEXT_FOUND' (or empty), reply with ONLY this sentence and nothing else:
  I cannot answer this request because the information does not exist or you do not have permission to access it.
- Do NOT invent fixes, numbers, or details that are not in the context.
- Do NOT use outside knowledge.
- Keep the answer concise and direct. Quote only facts present in the context.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 4) Hugging Face Inference LLM (or local stub if no HF_TOKEN)
    llm = get_llm()

    # 5) LCEL chain (LangChain Expression Language)
    # Pipeline shape:
    #   user_query
    #     -> {"context": retrieve|format_docs, "question": passthrough}
    #     -> prompt template
    #     -> LLM
    #     -> plain string
    #
    # Example for engineer + ERR-9021:
    #   context  = "[Source: Engineering Troubleshooting Guide] Critical Fix..."
    #   question = "How do I fix error ERR-9021 timeout?"
    #   answer   = "Increase max connections to 100 in config.yaml."
    rag_chain = (
        {
            "context": retriever | format_docs,  # retrieve docs then stringify
            "question": RunnablePassthrough(),  # forward the raw user_query string
        }
        | prompt
        | llm
        | StrOutputParser()  # AIMessage -> str
    )

    # 6) Run the chain end-to-end for this query
    response = rag_chain.invoke(user_query)

    print("\n---------------- FINAL LLM RESPONSE ----------------")
    print(response)
    print("----------------------------------------------------\n")
    return response


# ----------------------------------------------------------------------
# EXECUTION DEMONSTRATION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Build the classified knowledge base once
    ingest_classified_documents()

    # Scenario A: intern is NOT in DOC-202.allowed_roles -> retrieval empty -> deny
    print("================ SCENARIO 1: INTERN ATTEMPTS SECURE QUERY ================")
    generate_secure_answer(
        user_query="How do I fix error ERR-9021 timeout?",
        user_role="intern",
    )

    # Scenario B: engineer IS allowed on DOC-202 -> answer from engineering guide
    print("================ SCENARIO 2: ENGINEER AUTHORIZED QUERY ================")
    generate_secure_answer(
        user_query="How do I fix error ERR-9021 timeout?",
        user_role="engineer",
    )

    # Scenario C: finance IS allowed on DOC-303 -> answer from finance report
    print("================ SCENARIO 3: FINANCE AUTHORIZED QUERY ================")
    generate_secure_answer(
        user_query="What was our EBITDA and financial margin for Q3?",
        user_role="finance",
    )
