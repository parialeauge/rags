import lancedb

# Connect to database and open table
db = lancedb.connect("./.lancedb")
table = db.open_table("documents")

query = "What are the financial results for this year?"

# Standard vector similarity search
results = table.search(query).limit(3).to_pandas()
print(results[["text", "_distance"]])

