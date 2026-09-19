"""
6_search_multiVector_selfQuery.py
---------------------------------
Self-query pattern:
  1) Parse a natural-language question into structured fields
     (semantic search text + metadata filter) — here simulated as LLM output
  2) Run vector search with that filter against LanceDB
"""

import importlib

import lancedb
from pydantic import BaseModel


class StructuredQuery(BaseModel):
    """LLM-parsed form of a natural language question."""

    semantic_query: str  # Unstructured text used for vector similarity
    filter_category: str  # Structured metadata filter value


# Register hf-inference embedder + load HF_TOKEN
create_db = importlib.import_module("0_lanceDB_create")
create_db._load_dotenv()

db = lancedb.connect("./.lancedb")
table = db.open_table("documents")

# Simulated LLM parse of: "Show me Q3 finance reports from 2026"
parsed_query = StructuredQuery(
    semantic_query="financial performance metrics",
    filter_category="finance",
)

# Vector search + metadata filter derived from the structured query
results = (
    table.search(parsed_query.semantic_query)
    .where(f"category = '{parsed_query.filter_category}'")
    .limit(2)
    .to_pandas()
)

print(results[["text", "category", "source"]])
