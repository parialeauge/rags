"""
the complete implementation combining Native Hybrid Search 
(Dense Vectors + BM25 Full-Text Search) with Role-Based Access Control (RBAC) 
metadata filtering using LangChain and LanceDB.

Organizational Hierarchy: HR Lead, Finance Lead, Business Lead, Engineer Lead, Contractor Lead, Intern Lead, Executive Lead.

Allowed Document Categories & Tiers:
- HR Policy, HR Confidential, HR Restricted, Public
- Finance, Business Requirement, Business Product
- Engineering, Restricted, Public
- Contractor-Tagged Only, Public
- HR Policy, Business Requirement, Restricted Tier Docs, Finance, HR Confidential
- Business Product (Full), Business Requirement (Full), HR Policy, Business Restricted, Public
- Finance, HR Confidential
- Business Product (Full), Business Requirement (Full), HR Policy, Engineering Restricted, Public
- Finance, HR Confidential, Executive Confidential
- Business Product (Full), HR Policy, Public
- Business Requirement, Restricted Tier Docs, Finance, HR Confidential

Core Access Rules Summary
1. Universal Base Layer: Every department has Public and Restricted documents. 
Roles like Intern and Contractor are blocked from Restricted tier docs across all categories.
2. HR Policy Cross-Access: HR Lead, Finance Lead, Business Lead, Engineer, 
and Intern all have access to general HR Policy documents. 
Contractor remains blocked from HR policies.
3. Business & Requirements Split: Business Lead and Engineer have access to 
both Product and Requirement docs. Intern has access to Product docs only (no Requirements). 
Contractor has access only to Product docs specifically flagged for contractor visibility.

                 ┌───────────────────────────────────────────────┐
                 │    User Question + User Role (e.g., 'intern') │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │    Dynamic RBAC LanceDB Hybrid Retriever      │
                 │  (Dense + BM25 filtered by role SQL predicate)│
                 └───────────────────────┬───────────────────────┘
                                         │
                              LangChain LCEL Chain
                 ┌───────────────────────┴───────────────────────┐
                 │ 1. Retrieve Context (Role-Filtered)          │
                 │ 2. Format Documents (or inject Access Denied) │
                 │ 3. Pass to Prompt Template                    │
                 │ 4. Execute LLM (ChatOpenAI / Local LLM)       │
                 │ 5. Parse Output String                        │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │     Secure Grounded Response to End User      │
                 └───────────────────────────────────────────────┘

- SQL Pre-Filtering: LanceDB applies the SQL WHERE predicate first to eliminate unauthorized rows.
- Parallel Dual Execution: The search executes Dense Vector similarity and BM25 keyword search exclusively on the authorized subset of documents.
- Reciprocal Rank Fusion (RRF): The dense and sparse candidate lists are merged and re-ranked before context is returned.

 ---->> install:  pip install -r requirements.txt 
                  pip install lancedb langchain-community langchain-huggingface langchain-openai langchain-core pyarrow sentence-transformers tantivy
 ---->> run:      conda activate rags && python 7_embeding-hybrid-lancedb_lanchainLCEL_metadata.py
 ---->> env:      put HF_TOKEN=hf_... in feature-arun/.env (loaded automatically)

#######--- output ---#######
=== [STEP 1] Ingesting Org Documents into LanceDB ===
Success: Ingested 6 documents into LanceDB at './lcel_org_hierarchy_store'.

================ SCENARIO 1: CONTRACTOR ATTEMPTS HR POLICY QUERY ================
=== Running RAG Pipeline | Query: 'What is the company vacation leave policy?' | Role: 'CONTRACTOR' ===

---------------- LLM RESPONSE ----------------
Access Denied: You do not have permission to view the requested information.
----------------------------------------------

================ SCENARIO 2: INTERN ATTEMPTS BUSINESS REQUIREMENTS QUERY ================
=== Running RAG Pipeline | Query: 'What are the real-time stream processing SLA requirements?' | Role: 'INTERN' ===

---------------- LLM RESPONSE ----------------
Access Denied: You do not have permission to view the requested information.
----------------------------------------------

================ SCENARIO 3: ENGINEER AUTHORIZED REQUIREMENTS QUERY ================
=== Running RAG Pipeline | Query: 'What are the real-time stream processing SLA requirements?' | Role: 'ENGINEER' ===

---------------- LLM RESPONSE ----------------
The business requirement spec for real-time stream processing architecture specifies an SLA under 50ms.
----------------------------------------------

================ SCENARIO 4: FINANCE LEAD AUTHORIZED FINANCE QUERY ================
=== Running RAG Pipeline | Query: 'What was our Q3 operational margin performance?' | Role: 'FINANCE_LEAD' ===

---------------- LLM RESPONSE ----------------
In Q3, operational margins grew by 14% with EBITDA exceeding projections.
----------------------------------------------


"""


import os
import shutil
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Configuration Constants
DB_PATH = "./lcel_org_hierarchy_store"
TABLE_NAME = "classified_org_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """Returns standardized HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ----------------------------------------------------------------------
# FUNCTION 1: INGESTION & INDEXING
# ----------------------------------------------------------------------
def ingest_classified_documents():
    """Ingests multi-department documents with role & security tier metadata."""
    print("=== [STEP 1] Ingesting Org Documents into LanceDB ===")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    embeddings = get_embedding_model()

    docs = [
        # --- HR DEPARTMENT ---
        Document(
            page_content="Standard company vacation policy allows 20 days paid leave annually.",
            metadata={
                "doc_id": "HR-101",
                "category": "hr_policy",
                "tier": "public",
                "contractor_accessible": False,
            },
        ),
        Document(
            page_content="Restricted HR payroll review process and executive compensation benchmark data.",
            metadata={
                "doc_id": "HR-202",
                "category": "hr_confidential",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
        # --- FINANCE DEPARTMENT ---
        Document(
            page_content="Q3 financial report shows operational margins grew by 14% with EBITDA exceeding projections.",
            metadata={
                "doc_id": "FIN-301",
                "category": "finance",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
        # --- BUSINESS DEPARTMENT: PRODUCT ---
        Document(
            page_content="Product Roadmap 2026: Mobile app UI redesign and new payment gateway integration.",
            metadata={
                "doc_id": "BIZ-401",
                "category": "biz_product",
                "tier": "public",
                "contractor_accessible": True,
            },
        ),
        # --- BUSINESS DEPARTMENT: REQUIREMENTS ---
        Document(
            page_content="Business requirement spec: Real-time stream processing architecture with SLA under 50ms.",
            metadata={
                "doc_id": "BIZ-501",
                "category": "biz_requirement",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
        # --- ENGINEERING DEPARTMENT ---
        Document(
            page_content="Database maintenance procedure: Execute blue-green failover script during scheduled window.",
            metadata={
                "doc_id": "ENG-601",
                "category": "engineering",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
    ]

    db_connection = lancedb.connect(DB_PATH)

    vectorstore = LanceDB.from_documents(
        documents=docs,
        embedding=embeddings,
        connection=db_connection,
        table_name=TABLE_NAME,
    )

    # Build Full-Text Search (BM25) Index
    native_table = vectorstore.get_table()
    native_table.create_fts_index("text", replace=True)

    print(f"Success: Ingested {len(docs)} documents into LanceDB at '{DB_PATH}'.\n")


# ----------------------------------------------------------------------
# HELPER 1: DYNAMIC SQL PRE-FILTER GENERATOR
# ----------------------------------------------------------------------
def build_rbac_sql_filter(user_role: str) -> str:
    """Translates organizational role into exact SQL pre-filter condition."""
    role = user_role.lower()

    if role == "executive":
        return "1=1"
    elif role == "hr_lead":
        return "category IN ('hr_policy', 'hr_confidential')"
    elif role == "finance_lead":
        return "category IN ('finance', 'hr_policy')"
    elif role == "business_lead":
        return "category IN ('biz_product', 'biz_requirement', 'hr_policy')"
    elif role == "engineer":
        return "category IN ('biz_product', 'biz_requirement', 'hr_policy', 'engineering')"
    elif role == "intern":
        return "category IN ('biz_product', 'hr_policy') AND tier = 'public'"
    elif role == "contractor":
        return "category = 'biz_product' AND contractor_accessible = true AND tier = 'public'"
    else:
        return "1=0"


# ----------------------------------------------------------------------
# HELPER 2: CONTEXT FORMATTER
# ----------------------------------------------------------------------
def format_docs(docs: list[Document]) -> str:
    """Formats retrieved context or injects security fallback string if empty."""
    if not docs:
        return "NO_ACCESSIBLE_CONTEXT_FOUND"

    formatted = []
    for doc in docs:
        meta = doc.metadata
        source = f"[ID: {meta.get('doc_id')} | Category: {meta.get('category')} | Tier: {meta.get('tier')}]"
        formatted.append(f"{source}\n{doc.page_content}")

    return "\n\n".join(formatted)


# ----------------------------------------------------------------------
# FUNCTION 2: SECURE LCEL RAG PIPELINE
# ----------------------------------------------------------------------
def generate_secure_answer(query: str, user_role: str):
    """Executes a secure LCEL pipeline gated by hierarchical RBAC filters."""
    print(f"=== Running RAG Pipeline | Query: '{query}' | Role: '{user_role.upper()}' ===")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database path '{DB_PATH}' not found. Run ingest_classified_documents() first!")

    # 1. Connect Vector Store
    embeddings = get_embedding_model()
    db_connection = lancedb.connect(DB_PATH)
    vectorstore = LanceDB(
        connection=db_connection,
        table_name=TABLE_NAME,
        embedding=embeddings,
    )

    # 2. Build Role-Filtered Native Hybrid Retriever
    sql_filter = build_rbac_sql_filter(user_role)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 2,
            "query_type": "hybrid",
            "filter": sql_filter,
        },
    )

    # 3. Prompt Template with System Guardrails
    prompt_template = """You are a secure enterprise AI assistant. Answer the user's question strictly using the provided context below.

CRITICAL SECURITY RULES:
- If context is 'NO_ACCESSIBLE_CONTEXT_FOUND', reply EXACTLY: "Access Denied: You do not have permission to view the requested information."
- Do NOT use external knowledge or fabricate details for restricted topics.
- Keep answers professional, concise, and clear.

Context:
{context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 4. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5. Build LCEL Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Run Chain
    response = rag_chain.invoke(query)

    print("\n---------------- LLM RESPONSE ----------------")
    print(response)
    print("----------------------------------------------\n")
    return response


# ----------------------------------------------------------------------
# DEMONSTRATION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Ingest Documents
    ingest_classified_documents()

    print("================ SCENARIO 1: CONTRACTOR ATTEMPTS HR POLICY QUERY ================")
    # Contractor asks for vacation policy (Blocked: Contractors have no HR policy access)
    generate_secure_answer(
        query="What is the company vacation leave policy?",
        user_role="contractor",
    )

    print("================ SCENARIO 2: INTERN ATTEMPTS BUSINESS REQUIREMENTS QUERY ================")
    # Intern asks for real-time SLA specs (Blocked: Interns have no requirement access)
    generate_secure_answer(
        query="What are the real-time stream processing SLA requirements?",
        user_role="intern",
    )

    print("================ SCENARIO 3: ENGINEER AUTHORIZED REQUIREMENTS QUERY ================")
    # Engineer asks for real-time SLA specs (Allowed)
    generate_secure_answer(
        query="What are the real-time stream processing SLA requirements?",
        user_role="engineer",
    )

    print("================ SCENARIO 4: FINANCE LEAD AUTHORIZED FINANCE QUERY ================")
    # Finance Lead asks about Q3 margins (Allowed)
    generate_secure_answer(
        query="What was our Q3 operational margin performance?",
        user_role="finance_lead",
    )

