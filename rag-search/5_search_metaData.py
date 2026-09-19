"""
5_search_metaData.py
--------------------
Vector search with a SQL-like metadata filter (.where(...)).

Requires the LanceDB table from 0_lanceDB_create.py, including metadata
columns: source, vector_index.
"""

import importlib

import lancedb

# Register hf-inference embedder + load HF_TOKEN
create_db = importlib.import_module("0_lanceDB_create")
create_db._load_dotenv()

db = lancedb.connect("./.lancedb")
table = db.open_table("documents")

# Vector search, then keep only rows matching metadata filter
results = (
    table.search("What were the server outage root causes?")
    .where("source = 'logs/2026_system.txt' AND vector_index > 5")
    .limit(3)
    .to_pandas()
)

print(results[["text", "source", "vector_index"]])
