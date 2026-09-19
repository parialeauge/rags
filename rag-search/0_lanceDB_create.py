"""
0_lanceDB_create.py
-------------------
Creates a local LanceDB database + "documents" table for RAG search demos.

Why Hugging Face Inference API?
  Local sentence-transformers needs PyTorch. On this machine (Python 3.13 /
  macOS), PyTorch is not installable, so we embed text via Hugging Face's
  cloud Inference API using HF_TOKEN from .env instead.
"""

import os
from pathlib import Path

import lancedb
import numpy as np
from huggingface_hub import InferenceClient
from lancedb.embeddings import TextEmbeddingFunction, register
from lancedb.pydantic import LanceModel, Vector

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./.lancedb"  # Folder where LanceDB stores tables on disk
TABLE_NAME = "documents"  # Logical table name used by search scripts
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HF model id
EMBEDDING_DIMS = 384  # Output vector size for all-MiniLM-L6-v2

# Sample rows inserted when a brand-new table is created
# Metadata fields used by filtered / self-query search demos
DEFAULT_DOCUMENTS = [
    {
        "text": "Error 404: Page not found on server",
        "source": "logs/2026_system.txt",
        "vector_index": 1,
        "category": "ops",
    },
    {
        "text": "Quarterly financial revenue report for 2026",
        "source": "reports/finance_q1.txt",
        "vector_index": 2,
        "category": "finance",
    },
    {
        "text": "System restart required after updating driver",
        "source": "logs/2026_system.txt",
        "vector_index": 6,
        "category": "ops",
    },
    {
        "text": "Cloud deployment failed with error code 503",
        "source": "logs/2026_system.txt",
        "vector_index": 8,
        "category": "ops",
    },
    {
        "text": "Root cause analysis: server outage due to memory leak",
        "source": "logs/2026_system.txt",
        "vector_index": 10,
        "category": "ops",
    },
    {
        "text": "Q3 2026 financial performance metrics and revenue growth",
        "source": "reports/finance_q3.txt",
        "vector_index": 3,
        "category": "finance",
    },
]


def _load_dotenv() -> None:
    """Load secrets/config from a nearby .env file into process environment.

    Looks for .env in:
      1) current working directory
      2) parent directory
      3) feature-arun/.env (two levels above this file)

    Only sets keys that are not already present in os.environ
    (setdefault), so real shell exports always win.
    """
    #---------------------------------------------------------------------------
    # candidates = List of possible paths to the .env file
    # icandidatest will return all three paths if the .env 
    # candidates will return as absoolute path if the .env file 
    #---------------------------------------------------------------------------
    candidates = [
        Path.cwd() / ".env", # Current working directory/.env
        Path.cwd().parent / ".env", # Parent directory/.env
        Path(__file__).resolve().parents[2] / ".env",  # .../feature-arun/.env
    ]
    #---------------------------------------------------------------------------
    # Travers the candidates list and read the .env file
    # if file found set the environment variables as key-value pairs
    #---------------------------------------------------------------------------
    for env_path in candidates: # Iterate over the candidates all 3 paths
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            # Skip blanks, comments, and malformed lines
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1) # Split only on first '='
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value) # Do not overwrite existing env

@register("hf-inference")
class HFInferenceEmbeddings(TextEmbeddingFunction):
    """LanceDB embedding function backed by Hugging Face Inference API.

    @register("hf-inference") stores this class in LanceDB's global registry
    under the name "hf-inference". That name is also written into the table
    metadata when the table is created. Later search scripts MUST import this
    module so the same name is registered again — otherwise you get:
        KeyError: 'hf-inference'
    """

    name: str = EMBEDDING_MODEL  # Model id sent to Hugging Face
    _ndims: int = EMBEDDING_DIMS  # Fixed vector length for schema + search

    def ndims(self) -> int:
        """Return embedding dimensionality (required by LanceDB schema).

        LanceDB calls this to size the `vector` column, e.g. Vector(384).
        """
        return self._ndims

    def generate_embeddings(self, texts):
        """Convert one or more text strings into dense float vectors.

        Called automatically by LanceDB when:
          - inserting rows that only have a `text` field
          - searching with a string query (vector half of hybrid search)

        Steps:
          1) Read HF_TOKEN from environment
          2) Call Hugging Face feature_extraction for each text
          3) Normalize shape to a single 1-D vector per text
          4) Return list of Python lists (LanceDB-friendly format)
        """
        # Accept either common env var name for the Hugging Face token
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError(
                "HF_TOKEN is not set. Add it to your environment or feature-arun/.env"
            )

        #---------------------------------------------------------------------------
        # client = InferenceClient(api_key=token)  # Authenticated HF API client
        #---------------------------------------------------------------------------
        client = InferenceClient(api_key=token)  # Authenticated HF API client
        texts = self.sanitize_input(texts)  # Normalize input to a list of strings
        vectors = []
        for text in texts:
            # Remote embedding call (network required)
            emb = client.feature_extraction(text, model=self.name)
            arr = np.asarray(emb, dtype=np.float32)
            # Some APIs return (tokens, dims); average-pool to one sentence vector
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            vectors.append(arr.tolist())  # LanceDB expects plain Python lists
        return vectors


def create_lancedb(
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    documents: list | None = None,
    overwrite: bool = False,
):
    """Create or open a LanceDB database and documents table.

    What this function does:
      1) Loads HF_TOKEN from .env
      2) Builds an embedding-aware Document schema
      3) Connects to (or creates) the local DB folder
      4) Opens an existing table, OR creates a new one with sample docs
      5) Creates a Full-Text Search (FTS) index on `text` for hybrid search

    Args:
        db_path: Filesystem path for the LanceDB store (default ./.lancedb).
        table_name: Name of the table to open/create.
        documents: Optional list of {"text": "..."} dicts to insert on create.
        overwrite: If True and table exists, rebuild it from `documents`.

    Returns:
        Tuple (db, table): LanceDB connection and table handle.
    """
    _load_dotenv()  # Ensure HF_TOKEN is available before embedding -- private

    if documents is None:
        documents = DEFAULT_DOCUMENTS

    # Instantiate the registered embedding function for this table schema
    embed_fn = HFInferenceEmbeddings.create()

    class Document(LanceModel):
        # SourceField: raw text LanceDB should embed automatically
        text: str = embed_fn.SourceField()
        # VectorField: dense vector column; filled by embed_fn on insert/search
        vector: Vector(embed_fn.ndims()) = embed_fn.VectorField()
        # Metadata fields used by filtered / metadata / self-query search demos
        source: str
        vector_index: int
        category: str

    # connect() creates the folder if it does not exist yet
    db = lancedb.connect(db_path)

    table_exists = table_name in db.table_names()
    if table_exists and not overwrite:
        # Reuse existing data (idempotent / safe to re-run)
        table = db.open_table(table_name)
        print(f"Opened existing table '{table_name}' in {db_path}")
        return db, table

    # mode="overwrite" replaces an existing table; otherwise create fresh
    mode = "overwrite" if table_exists and overwrite else None
    if mode:
        # Inserts docs and auto-embeds `text` -> `vector` via HF API
        table = db.create_table(table_name, schema=Document, data=documents, mode=mode)
    else:
        table = db.create_table(table_name, schema=Document, data=documents)

    # FTS (keyword) index on text — required for query_type="hybrid"
    # Hybrid = vector similarity + keyword search, fused with RRF
    table.create_fts_index("text")
    print(f"Created table '{table_name}' + FTS index in {db_path}")
    return db, table


if __name__ == "__main__":
    # Running: python 0_lanceDB_create.py
    # Creates ./.lancedb/documents (or opens it if already present)
    create_lancedb()
