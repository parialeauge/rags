from rank_bm25 import BM25Okapi

# Corpus of documents
documents = [
    "Error 404: Page not found on server",
    "Quarterly financial revenue report for 2026",
    "System restart required after updating driver"
]

# Tokenize corpus and initialize BM25 index
tokenized_corpus = [doc.split(" ") for doc in documents] # Tokenize the corpus - split the documents into words
bm25 = BM25Okapi(tokenized_corpus) # Initialize the BM25 index

query = "Error 404"
tokenized_query = query.split(" ") # Tokenize the query - split the query into words

# Get scores for each document
scores = bm25.get_scores(tokenized_query) # Get the scores for the query
top_doc = documents[scores.argmax()] # Get the document with the highest score  

print(f"Top Matched Document: {top_doc}") # Print the document with the highest score

