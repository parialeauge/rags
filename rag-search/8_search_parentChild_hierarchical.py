"""
8_search_parentChild_hierarchical.py
-------------------------------------
Parent-Child (hierarchical) retrieval:

  1) Embed + search small "child" chunks (precise matching)
  2) Return the larger "parent" section as LLM context (richer answer context)

Uses a dedicated LanceDB table: parent_child
"""

import importlib

import lancedb
from lancedb.pydantic import LanceModel, Vector

# Register hf-inference embedder + load HF_TOKEN
create_db = importlib.import_module("0_lanceDB_create")
create_db._load_dotenv()
embed_fn = create_db.HFInferenceEmbeddings.create()

DB_PATH = "./.lancedb"
TABLE_NAME = "parent_child"

# Demo hierarchy: small child phrases map to larger parent sections
PARENT_CHILD_DOCS = [
    {
        "child_id": "c1_1",
        "child_text": "trust account setup",
        "parent_id": "p1",
        "parent_text": (
            "Detailed section on trust account setup and tax implications. "
            "Open a dedicated trust account at an approved bank, keep client "
            "funds segregated, and track every deposit for audit compliance."
        ),
    },
    {
        "child_id": "c1_2",
        "child_text": "tax implications for trusts",
        "parent_id": "p1",
        "parent_text": (
            "Detailed section on trust account setup and tax implications. "
            "Open a dedicated trust account at an approved bank, keep client "
            "funds segregated, and track every deposit for audit compliance."
        ),
    },
    {
        "child_id": "c2_1",
        "child_text": "server outage root cause",
        "parent_id": "p2",
        "parent_text": (
            "Incident report: the production outage was caused by a memory leak "
            "in the caching service. Mitigation included a rolling restart and "
            "a patch to release unused buffers under load."
        ),
    },
]


class ParentChildDoc(LanceModel):
    # Search happens on the small child text
    child_text: str = embed_fn.SourceField()
    vector: Vector(embed_fn.ndims()) = embed_fn.VectorField()
    child_id: str
    parent_id: str
    # Larger parent text is returned to the LLM (not what we embed/search)
    parent_text: str


def get_parent_child_table():
    """Create or open the parent_child table."""
    db = lancedb.connect(DB_PATH)
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, schema=ParentChildDoc, data=PARENT_CHILD_DOCS)
    print(f"Created table '{TABLE_NAME}' with {len(PARENT_CHILD_DOCS)} child chunks")
    return table


table = get_parent_child_table()
query = "How to set up trust accounts?"

# 1) Vector search against small child embeddings
results = table.search(query).limit(1).to_pandas()
matched = results.iloc[0]

print(f"Query: {query}")
print(f"Matched child ({matched['child_id']}): {matched['child_text']}")

# 2) Use larger parent text as LLM prompt context
context_for_llm = matched["parent_text"]
print(f"Context passed to LLM (parent {matched['parent_id']}): {context_for_llm}")
