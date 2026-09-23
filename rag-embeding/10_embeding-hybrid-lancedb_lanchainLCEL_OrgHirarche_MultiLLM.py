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
                  pip install lancedb langchain-community langchain-huggingface langchain-openai langchain-core pyarrow sentence-transformers tantivy langchain-experimental
                  # For OpenRouter
                export OPENROUTER_API_KEY="your-openrouter-key"

                # For Hugging Face
                export HUGGINGFACEHUB_API_TOKEN="your-hf-token"

 ---->> run:      conda activate rags && python 7_embeding-hybrid-lancedb_lanchainLCEL_metadata.py
 ---->> env:      put HF_TOKEN=hf_... in feature-arun/.env (loaded automatically)

#######--- output ---#######
=== [STEP 1] Ingesting Org Documents into LanceDB ===
Success: Ingested 6 documents into LanceDB at './lcel_org_hierarchy_store'.

=== Running RAG Pipeline | Role: 'CONTRACTOR' | Provider: 'openrouter' ===
[LLM Factory] Initializing OpenRouter model...

---------------- LLM RESPONSE ----------------
Access Denied: You do not have permission to view the requested information.
----------------------------------------------

=== Running RAG Pipeline | Role: 'ENGINEER' | Provider: 'openrouter' ===
[LLM Factory] Initializing OpenRouter model...

---------------- LLM RESPONSE ----------------
The business requirement specification for real-time stream processing architecture mandates an SLA under 50ms.
----------------------------------------------



"""


import os
import shutil
import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

# Define local filesystem database path
DB_PATH = "./lcel_org_hierarchy_store"

# Define table name within LanceDB
TABLE_NAME = "classified_org_docs"

# Define HuggingFace embedding model for vector encoding
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Instantiates and returns the HuggingFace sentence transformer embedding model.

    Returns:
        HuggingFaceEmbeddings: The initialized embedding model wrapper.
    """
    # Create and return HuggingFace embeddings instance using MiniLM model
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# ----------------------------------------------------------------------
# PROVIDER CONFIGURATION FUNCTIONS
# ----------------------------------------------------------------------
def configure_openrouter_llm(model_name: str | None = None) -> BaseChatModel:
    """Configures and returns an OpenRouter Chat model instance using the OpenAI-compatible API.

    Args:
        model_name (str, optional): OpenRouter model path. Defaults to Llama-3.3-70B.

    Returns:
        BaseChatModel: Configured ChatOpenAI instance pointing to OpenRouter API.

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set.
    """
    # Read OpenRouter API key from system environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")

    # Validate that API key exists before proceeding
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    # Fallback to default high-performing open model if none specified
    selected_model = model_name or "meta-llama/llama-3.3-70b-instruct"

    # Return ChatOpenAI configured to point to OpenRouter base URL
    return ChatOpenAI(
        model=selected_model,                                # Model path on OpenRouter
        openai_api_key=api_key,                              # API key authentication
        openai_api_base="https://openrouter.ai/api/v1",      # Custom API base endpoint
        temperature=0,                                       # Zero temperature for deterministic responses
        default_headers={                                    # Headers required/recommended by OpenRouter
            "HTTP-Referer": "https://localhost",             # Optional application identification URL
            "X-Title": "Hierarchical RBAC LCEL Chain",        # Optional application title for tracking
        },
    )


def configure_huggingface_llm(model_name: str | None = None) -> BaseChatModel:
    """Configures and returns a Hugging Face Inference Endpoint Chat LLM instance.

    Args:
        model_name (str, optional): Hugging Face repo ID. Defaults to Meta-Llama-3-8B-Instruct.

    Returns:
        BaseChatModel: Configured ChatHuggingFace instance.

    Raises:
        ValueError: If HUGGINGFACEHUB_API_TOKEN environment variable is not set.
    """
    # Read Hugging Face user access token from environment
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Raise explicit error if user token is absent
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")

    # Fallback to default Hugging Face repository ID
    selected_repo = model_name or "meta-llama/Meta-Llama-3-8B-Instruct"

    # Initialize low-level HuggingFace inference endpoint connection
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=selected_repo,                               # Target HF repository model ID
        huggingfacehub_api_token=hf_token,                   # Token authentication
        temperature=0.01,                                    # Low temperature for precise factual retrieval
        max_new_tokens=512,                                  # Maximum output tokens generated per call
    )

    # Wrap raw endpoint inside LangChain's ChatHuggingFace chat abstraction layer
    return ChatHuggingFace(llm=llm_endpoint)


# ----------------------------------------------------------------------
# FACTORY FUNCTION
# ----------------------------------------------------------------------
def get_llm_factory(provider: str = "openrouter", model_name: str | None = None) -> BaseChatModel:
    """Factory function that selects and instantiates the requested LLM provider.

    Args:
        provider (str): 'openrouter' or 'huggingface'.
        model_name (str, optional): Override model identifier.

    Returns:
        BaseChatModel: The configured Chat LLM object.

    Raises:
        ValueError: If an unsupported provider string is passed.
    """
    # Normalize input string to lowercase and strip whitespace
    provider = provider.lower().strip()

    # Route request to OpenRouter configuration handler
    if provider == "openrouter":
        print("[LLM Factory] Initializing OpenRouter model...")
        return configure_openrouter_llm(model_name=model_name)

    # Route request to Hugging Face configuration handler
    elif provider == "huggingface":
        print("[LLM Factory] Initializing Hugging Face model...")
        return configure_huggingface_llm(model_name=model_name)

    # Throw exception for invalid provider choices
    else:
        raise ValueError(
            f"Invalid provider '{provider}'. Must be either 'openrouter' or 'huggingface'."
        )


# ----------------------------------------------------------------------
# FUNCTION 1: INGESTION & INDEXING
# ----------------------------------------------------------------------
def ingest_classified_documents():
    """Ingests multi-department corporate documents with security metadata into LanceDB

    Processes raw documents, calculates dense embeddings, stores metadata, and generates
    a BM25 Full-Text Search (FTS) index.
    """
    print("=== [STEP 1] Ingesting Org Documents into LanceDB ===")

    # Check if a database folder already exists on disk
    if os.path.exists(DB_PATH):
        # Remove old database directory to create a clean fresh setup for testing
        shutil.rmtree(DB_PATH)

    # Instantiate the embedding model function
    embeddings = get_embedding_model()

    # Construct corpus documents with detailed category, security tier, and role flags
    docs = [
        # --- HR DEPARTMENT ---
        Document(
            page_content="Standard company vacation policy allows 20 days paid leave annually.",
            metadata={
                "doc_id": "HR-101",                         # Unique identifier
                "category": "hr_policy",                    # HR category classification
                "tier": "public",                           # Security tier level
                "contractor_accessible": False,             # Flag explicitly restricting contractors
            },
        ),
        Document(
            page_content="Restricted HR payroll review process and executive compensation benchmark data.",
            metadata={
                "doc_id": "HR-202",                         # Unique identifier
                "category": "hr_confidential",              # Confidential HR category
                "tier": "restricted",                       # Restricted tier level
                "contractor_accessible": False,             # Blocked for contractors
            },
        ),
        # --- FINANCE DEPARTMENT ---
        Document(
            page_content="Q3 financial report shows operational margins grew by 14% with EBITDA exceeding projections.",
            metadata={
                "doc_id": "FIN-301",                         # Unique identifier
                "category": "finance",                      # Finance category classification
                "tier": "restricted",                       # Restricted tier level
                "contractor_accessible": False,             # Blocked for contractors
            },
        ),
        # --- BUSINESS DEPARTMENT: PRODUCT ---
        Document(
            page_content="Product Roadmap 2026: Mobile app UI redesign and new payment gateway integration.",
            metadata={
                "doc_id": "BIZ-401",                         # Unique identifier
                "category": "biz_product",                  # Business Product category
                "tier": "public",                           # Public tier level
                "contractor_accessible": True,              # Explicitly accessible to contractors
            },
        ),
        # --- BUSINESS DEPARTMENT: REQUIREMENTS ---
        Document(
            page_content="Business requirement spec: Real-time stream processing architecture with SLA under 50ms.",
            metadata={
                "doc_id": "BIZ-501",                         # Unique identifier
                "category": "biz_requirement",              # Business Requirement category
                "tier": "restricted",                       # Restricted tier level
                "contractor_accessible": False,             # Blocked for contractors
            },
        ),
        # --- ENGINEERING DEPARTMENT ---
        Document(
            page_content="Database maintenance procedure: Execute blue-green failover script during scheduled window.",
            metadata={
                "doc_id": "ENG-601",                         # Unique identifier
                "category": "engineering",                  # Engineering Ops category
                "tier": "restricted",                       # Restricted tier level
                "contractor_accessible": False,             # Blocked for contractors
            },
        ),
    ]

    # Open local LanceDB connection handle
    db_connection = lancedb.connect(DB_PATH)

    # Embed documents and write PyArrow columnar dataset to LanceDB table
    vectorstore = LanceDB.from_documents(
        documents=docs,                                     # Input document payload
        embedding=embeddings,                               # Sentence transformer encoder
        connection=db_connection,                          # Database connection handle
        table_name=TABLE_NAME,                              # Target database table name
    )

    # Access raw LanceDB table handle to generate search index
    native_table = vectorstore.get_table()

    # Build BM25 Tantivy Full-Text Search index on the text body column
    native_table.create_fts_index("text", replace=True)

    print(f"Success: Ingested {len(docs)} documents into LanceDB at '{DB_PATH}'.\n")


# ----------------------------------------------------------------------
# HELPER FUNCTIONS FOR RAG PIPELINE
# ----------------------------------------------------------------------
def build_rbac_sql_filter(user_role: str) -> str:
    """Generates an SQL WHERE clause matching user role against document metadata matrix.

    Args:
        user_role (str): The role string of the requesting user (e.g., 'intern', 'engineer').

    Returns:
        str: SQL filter string to pass into LanceDB.
    """
    # Convert input role string to lowercase for safe matching
    role = user_role.lower()

    # Executive: Unrestricted access across all categories and tiers
    if role == "executive":
        return "1=1"

    # HR Lead: Restricted to HR Policies and HR Confidential documents
    elif role == "hr_lead":
        return "category IN ('hr_policy', 'hr_confidential')"

    # Finance Lead: Access to Finance docs plus general HR Policy
    elif role == "finance_lead":
        return "category IN ('finance', 'hr_policy')"

    # Business Lead: Access to Product, Requirements, and HR Policy
    elif role == "business_lead":
        return "category IN ('biz_product', 'biz_requirement', 'hr_policy')"

    # Engineer: Access to Product, Requirements, HR Policy, and Engineering docs
    elif role == "engineer":
        return "category IN ('biz_product', 'biz_requirement', 'hr_policy', 'engineering')"

    # Intern: Access to Product and HR Policy; strictly blocked from Requirements and Restricted tier
    elif role == "intern":
        return "category IN ('biz_product', 'hr_policy') AND tier = 'public'"

    # Contractor: Access ONLY to Public Product docs explicitly flagged contractor_accessible=True
    elif role == "contractor":
        return "category = 'biz_product' AND contractor_accessible = true AND tier = 'public'"

    # Fallback default: Hard security denial condition for unrecognized roles
    else:
        return "1=0"


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved Document objects into a clean text context block for the prompt.

    Args:
        docs (list[Document]): List of retrieved documents from LanceDB.

    Returns:
        str: Combined document string or fallback security signal if empty.
    """
    # Check if the search returned no authorized results
    if not docs:
        # Return fallback text token signaling authorization denial to prompt
        return "NO_ACCESSIBLE_CONTEXT_FOUND"

    # Initialize accumulator list for formatted output strings
    formatted = []

    # Iterate through each authorized document retrieved
    for doc in docs:
        # Extract metadata dictionary from document object
        meta = doc.metadata

        # Format header with document metadata tags for model provenance
        source = f"[ID: {meta.get('doc_id')} | Category: {meta.get('category')} | Tier: {meta.get('tier')}]"

        # Combine header tag and document text content
        formatted.append(f"{source}\n{doc.page_content}")

    # Join formatted list items with double newlines
    return "\n\n".join(formatted)


# ----------------------------------------------------------------------
# SECURE LCEL RAG PIPELINE
# ----------------------------------------------------------------------
def generate_secure_answer(query: str, user_role: str, provider: str = "openrouter"):
    """Executes a complete LCEL RAG pipeline with role-based security filters and custom LLM provider.

    Args:
        query (str): The natural language query string from the user.
        user_role (str): Role identifier used to enforce security permissions.
        provider (str): 'openrouter' or 'huggingface' LLM selector.

    Returns:
        str: Final response string generated by the LLM.
    """
    print(f"=== Running RAG Pipeline | Role: '{user_role.upper()}' | Provider: '{provider}' ===")

    # Verify vector store database path exists before attempting read operations
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database path '{DB_PATH}' not found. Run ingest_classified_documents() first!")

    # Load embedding model instance
    embeddings = get_embedding_model()

    # Open persistent database connection handle
    db_connection = lancedb.connect(DB_PATH)

    # Re-connect vector store wrapper to database table
    vectorstore = LanceDB(
        connection=db_connection,                          # Database connection handle
        table_name=TABLE_NAME,                              # Target table name
        embedding=embeddings,                               # Embedding model wrapper
    )

    # Build SQL security predicate for current requesting user role
    sql_filter = build_rbac_sql_filter(user_role)

    # Convert LanceDB vectorstore into a retriever object with security pre-filter
    retriever = vectorstore.as_retriever(
        search_type="similarity",                          # Similarity search interface
        search_kwargs={
            "k": 2,                                        # Return top 2 candidate matches
            "query_type": "hybrid",                        # Use Native Hybrid Search (BM25 + Vector)
            "filter": sql_filter,                          # Pass SQL pre-filter for database RBAC enforcement
        },
    )

    # Define system prompt template containing security guardrail rules
    prompt_template = """You are a secure enterprise AI assistant. Answer the user's question strictly using the provided context below.

CRITICAL SECURITY RULES:
- If context is 'NO_ACCESSIBLE_CONTEXT_FOUND', reply EXACTLY: "Access Denied: You do not have permission to view the requested information."
- Do NOT use external knowledge or fabricate details for restricted topics.
- Keep answers professional, concise, and clear.

Context:
{context}

Question: {question}

Answer:"""

    # Parse template string into a LangChain ChatPromptTemplate object
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Call factory function to instantiate requested LLM provider (OpenRouter or HuggingFace)
    llm = get_llm_factory(provider=provider)

    # Construct declarative LangChain Expression Language (LCEL) chain sequence
    rag_chain = (
        {
            "context": retriever | format_docs,             # Step 1a: Retrieve and format authorized docs
            "question": RunnablePassthrough(),             # Step 1b: Pass user query through unchanged
        }
        | prompt                                            # Step 2: Format prompt with context and question
        | llm                                               # Step 3: Run target Chat LLM model
        | StrOutputParser()                                 # Step 4: Extract string output from response object
    )

    # Execute chain synchronously with input user query
    response = rag_chain.invoke(query)

    print("\n---------------- LLM RESPONSE ----------------")
    print(response)                                        # Print generated output string to terminal
    print("----------------------------------------------\n")

    # Return response payload
    return response


# ----------------------------------------------------------------------
# MAIN EXECUTION FUNCTION (CHOOSE PROVIDER HERE)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # USER SELECTION: Change provider to either 'openrouter' or 'huggingface'
    # ------------------------------------------------------------------
    SELECTED_PROVIDER = "openrouter"

    # Step 1: Run Document Ingestion (Builds database and FTS index)
    ingest_classified_documents()

    # --- TEST SCENARIO A: BLOCKED QUERY (CONTRACTOR HAS NO HR ACCESS) ---
    generate_secure_answer(
        query="What is the company vacation leave policy?", # Query targeting restricted HR document
        user_role="contractor",                             # Role that is blocked from HR policy
        provider=SELECTED_PROVIDER,                        # Active LLM provider selection
    )

    # --- TEST SCENARIO B: AUTHORIZED QUERY (ENGINEER HAS REQUIREMENTS ACCESS) ---
    generate_secure_answer(
        query="What are the real-time stream processing SLA requirements?", # Query targeting requirements doc
        user_role="engineer",                               # Role that has access to requirements
        provider=SELECTED_PROVIDER,                        # Active LLM provider selection
    )

