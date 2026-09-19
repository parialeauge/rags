"""
1_search_denseVector_symantic.py
--------------------------------
Dense vector (semantic) search against the LanceDB "documents" table.

Requires importing 0_lanceDB_create so the "hf-inference" embedding
function is registered (needed to embed the text query).
"""

import importlib

import lancedb

# Register hf-inference embedder + load HF_TOKEN
create_db = importlib.import_module("0_lanceDB_create")
create_db._load_dotenv()

# Connect to database and open table
db = lancedb.connect("./.lancedb")
table = db.open_table("documents")

query = "What are the financial results for this year?"

# Standard vector similarity search (text query is embedded via HF API)
results = table.search(query).limit(3).to_pandas()
print(results[["text", "_distance"]])
