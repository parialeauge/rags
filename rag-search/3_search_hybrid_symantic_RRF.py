"""
3_search_hybrid_symantic_RRF.py
--------------------------------
Runs hybrid search on the LanceDB table created by 0_lanceDB_create.py.

Hybrid search = dense vector search + keyword (FTS) search.
LanceDB fuses both ranked lists with Reciprocal Rank Fusion (RRF).
"""

import importlib

import lancedb

# Import create module by path-style name (starts with "0_") so that:
#   1) @register("hf-inference") runs again in THIS process
#   2) LanceDB can embed the text query during search
# Without this import you get: KeyError: 'hf-inference'
create_db = importlib.import_module("0_lanceDB_create")

# Load HF_TOKEN from .env (needed to embed the query string via HF API)
create_db._load_dotenv()

# Open the same local DB/table created earlier
db = lancedb.connect("./.lancedb")
table = db.open_table("documents")

# query_type="hybrid" runs vector + FTS, then RRF-ranks the combined results
results = (
    table.search(query="error code in cloud deployment", query_type="hybrid")
    .limit(3)  # Keep top-3 matches
    .to_pandas()  # Convert Arrow result to a pandas DataFrame
)

# RRF usually exposes "_relevance_score"; fall back to "_score" if needed
score_col = "_relevance_score" if "_relevance_score" in results.columns else "_score"
print(results[["text", score_col]])
