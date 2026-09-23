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

                                  ┌───────────────────────────┐
                                  │         Executive         │
                                  │   (Full Org-Wide Access)  │
                                  └─────────────┬─────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐                    ┌─────────────────┐
│     HR Lead     │                    │  Finance Lead   │                    │  Business Lead  │
│  (HR Docs +     │                    │(Finance Docs +  │                    │(Prod, Req +     │
│   Public/Restr) │                    │  HR Policy)     │                    │  HR Policy)     │
└────────┬────────┘                    └─────────────────┘                    └────────┬────────┘
         │                                                                             │
         │ (HR Policy Access)                                                          │ (Prod / Req Access)
         ▼                                                                             ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                          Engineer                                             │
│               (Full Product + Full Requirements + HR Policy + Public/Restricted)              │
└──────────────────────────────────────────────┬────────────────────────────────────────────────┘
                                               │
                            ┌──────────────────┴──────────────────┐
                            ▼                                     ▼
                 ┌────────────────────┐                ┌────────────────────┐
                 │       Intern       │                │     Contractor     │
                 │(Full Prod + HR Pol │                │(Contractor Prod    │
                 │  + Public Docs)    │                │  + Public Docs)    │
                 └────────────────────┘                └────────────────────┘

- SQL Pre-Filtering: LanceDB applies the SQL WHERE predicate first to eliminate unauthorized rows.
- Parallel Dual Execution: The search executes Dense Vector similarity and BM25 keyword search exclusively on the authorized subset of documents.
- Reciprocal Rank Fusion (RRF): The dense and sparse candidate lists are merged and re-ranked before context is returned.

 ---->> install:  pip install -r requirements.txt
 ---->> run:      conda activate rags && python 7_embeding-hybrid-lancedb_lanchainLCEL_metadata.py
 ---->> env:      put HF_TOKEN=hf_... in feature-arun/.env (loaded automatically)

#######--- output ---#######
=== [STEP 1] Ingesting Org Documents into LanceDB ===
Success: Ingested 7 documents into LanceDB at './org_hierarchy_lancedb_store'.
FTS Index successfully created. Ingestion complete!

================ SCENARIO 1: HR POLICY ACCESS ================
Executing Search | Query: 'What is the vacation leave policy?' | Role: 'CONTRACTOR'
  RESULT: [ACCESS DENIED / NO MATCH] No accessible documents found for this role.

Executing Search | Query: 'What is the vacation leave policy?' | Role: 'INTERN'
  RESULT: Matches Found:
    [1] ID: HR-101 | Category: hr_policy | Tier: public | Score: 0.0325
        Content: 'Standard company vacation policy allows 20 days paid leave annually.'

================ SCENARIO 2: BUSINESS REQUIREMENTS ACCESS ================
Executing Search | Query: 'What are the real-time stream processing SLA requirements?' | Role: 'INTERN'
  RESULT: [ACCESS DENIED / NO MATCH] No accessible documents found for this role.

Executing Search | Query: 'What are the real-time stream processing SLA requirements?' | Role: 'ENGINEER'
  RESULT: Matches Found:
    [1] ID: BIZ-501 | Category: biz_requirement | Tier: restricted | Score: 0.0325
        Content: 'Business requirement spec: Real-time stream processing with SLA under 50ms.'

================ SCENARIO 3: FINANCE LEAD ACCESS ================
Executing Search | Query: 'What is our operational margin and revenue growth?' | Role: 'FINANCE_LEAD'
  RESULT: Matches Found:
    [1] ID: FIN-301 | Category: finance | Tier: restricted | Score: 0.0325
        Content: 'Q3 financial report shows operational margins grew by 14% with strong revenue.'

================ SCENARIO 4: ENGINEER ACCESS TO ENGINEERING ================




"""


import os
import shutil
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration Constants
DB_PATH = "./org_hierarchy_lancedb_store"
TABLE_NAME = "org_classified_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """Returns standardized HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


# ----------------------------------------------------------------------
# FUNCTION 1: DOCUMENT INGESTION & INDEXING (RUN ONCE)
# ----------------------------------------------------------------------
def ingest_knowledge_base():
    """Ingests multi-department documents with role & security tier metadata."""
    print("=== [STEP 1] Ingesting Org Documents into LanceDB ===")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    embeddings = get_embedding_model()

    # Documents covering all categories & security tiers
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
            page_content="Restricted HR payroll review process and compensation benchmark data for 2026.",
            metadata={
                "doc_id": "HR-202",
                "category": "hr_confidential",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
        # --- FINANCE DEPARTMENT ---
        Document(
            page_content="Q3 financial report shows operational margins grew by 14% with strong revenue.",
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
                "contractor_accessible": True,  # Contractor-tagged
            },
        ),
        Document(
            page_content="Internal product release specs for enterprise multi-tenancy dashboard.",
            metadata={
                "doc_id": "BIZ-402",
                "category": "biz_product",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
        # --- BUSINESS DEPARTMENT: REQUIREMENTS ---
        Document(
            page_content="Business requirement spec: Real-time stream processing with SLA under 50ms.",
            metadata={
                "doc_id": "BIZ-501",
                "category": "biz_requirement",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
        # --- ENGINEERING DEPARTMENT ---
        Document(
            page_content="Database maintenance procedure: Execute blue-green failover script during window.",
            metadata={
                "doc_id": "ENG-601",
                "category": "engineering",
                "tier": "restricted",
                "contractor_accessible": False,
            },
        ),
    ]

    db_connection = lancedb.connect(DB_PATH)

    # Store vector embeddings and metadata
    vectorstore = LanceDB.from_documents(
        documents=docs,
        embedding=embeddings,
        connection=db_connection,
        table_name=TABLE_NAME,
    )

    # Build Full-Text Search (BM25) index on text column
    native_table = vectorstore.get_table()
    native_table.create_fts_index("text", replace=True)

    print(
        f"Success: Ingested {len(docs)} documents into LanceDB at '{DB_PATH}'."
    )
    print("FTS Index successfully created. Ingestion complete!\n")


# ----------------------------------------------------------------------
# HELPER: CONSTRUCT SQL FILTER BASED ON HIERARCHY MATRIX
# ----------------------------------------------------------------------
def build_rbac_sql_filter(user_role: str) -> str:
    """Translates organizational role into exact SQL pre-filter condition."""
    role = user_role.lower()

    if role == "executive":
        # Full access across all categories and tiers
        return "1=1"

    elif role == "hr_lead":
        # HR Policy, HR Confidential, Public/Restricted HR
        return "category IN ('hr_policy', 'hr_confidential')"

    elif role == "finance_lead":
        # Finance docs + HR Policy
        return "category IN ('finance', 'hr_policy')"

    elif role == "business_lead":
        # Business Product, Business Requirement, HR Policy
        return "category IN ('biz_product', 'biz_requirement', 'hr_policy')"

    elif role == "engineer":
        # Business Product, Business Requirement, HR Policy, Engineering
        return "category IN ('biz_product', 'biz_requirement', 'hr_policy', 'engineering')"

    elif role == "intern":
        # Business Product (Full) + HR Policy (No Requirements, No Restricted)
        return "category IN ('biz_product', 'hr_policy') AND tier = 'public'"

    elif role == "contractor":
        # Business Product (Contractor-tagged only) + Public Tier Only (No HR Policy)
        return "category = 'biz_product' AND contractor_accessible = true AND tier = 'public'"

    else:
        # Default security fallback (Deny All)
        return "1=0"


# ----------------------------------------------------------------------
# FUNCTION 2: HYBRID SEARCH WITH HIERARCHICAL RBAC (RUN REPEATEDLY)
# ----------------------------------------------------------------------
def search_with_rbac(query: str, user_role: str, top_k: int = 3):
    """Executes native hybrid search (BM25 + Dense) with hierarchical SQL pre-filtering."""
    print(f"Executing Search | Query: '{query}' | Role: '{user_role.upper()}'")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database path '{DB_PATH}' not found. Run ingest_knowledge_base() first!"
        )

    embeddings = get_embedding_model()
    db_connection = lancedb.connect(DB_PATH)

    vectorstore = LanceDB(
        connection=db_connection,
        table_name=TABLE_NAME,
        embedding=embeddings,
    )

    # 1. Build role-specific SQL filter rule
    sql_filter = build_rbac_sql_filter(user_role)

    # 2. Execute Native Hybrid Search with SQL Pre-filtering
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k,
        query_type="hybrid",
        filter=sql_filter,
    )

    if not results:
        print("  RESULT: [ACCESS DENIED / NO MATCH] No accessible documents found for this role.\n")
        return []

    print("  RESULT: Matches Found:")
    for rank, (doc, score) in enumerate(results, start=1):
        print(
            f"    [{rank}] ID: {doc.metadata.get('doc_id')} | Category: {doc.metadata.get('category')} | Tier: {doc.metadata.get('tier')} | Score: {score:.4f}"
        )
        print(f"        Content: '{doc.page_content}'")
    print()
    return results


# ----------------------------------------------------------------------
# EXECUTION DEMONSTRATION
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Run Ingestion Once
    ingest_knowledge_base()

    # --- TEST CASE 1: CONTRACTOR vs. INTERN ON HR POLICY ---
    print("================ SCENARIO 1: HR POLICY ACCESS ================")
    search_with_rbac(
        query="What is the vacation leave policy?",
        user_role="contractor",  # Blocked from HR Policy
    )
    search_with_rbac(
        query="What is the vacation leave policy?",
        user_role="intern",  # Allowed HR Policy
    )

    # --- TEST CASE 2: INTERN vs. ENGINEER ON BUSINESS REQUIREMENTS ---
    print("================ SCENARIO 2: BUSINESS REQUIREMENTS ACCESS ================")
    search_with_rbac(
        query="What are the real-time stream processing SLA requirements?",
        user_role="intern",  # Blocked from Requirements
    )
    search_with_rbac(
        query="What are the real-time stream processing SLA requirements?",
        user_role="engineer",  # Allowed Requirements
    )

    # --- TEST CASE 3: FINANCE LEAD ACCESS TO FINANCE + HR POLICY ---
    print("================ SCENARIO 3: FINANCE LEAD ACCESS ================")
    search_with_rbac(
        query="What is our operational margin and revenue growth?",
        user_role="finance_lead",  # Allowed Finance Docs
    )

