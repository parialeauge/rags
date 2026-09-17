Core Search Types
------------------
1. Dense Vector Search (Semantic Search): Converts queries and documents into high-dimensional vector embeddings (e.g., via Hugging Face sentence-transformers or OpenAI models) and uses similarity metrics like Cosine Similarity, Euclidean Distance, or Dot Product to retrieve semantically related text, even if keywords do not match directly.

2. Sparse Vector / Keyword Search (Lexical Search): Uses traditional term-frequency algorithms like BM25, TF-IDF, or exact match indexing to locate documents containing exact keyword matches. This is effective for searching specific identifiers, proper nouns, error codes, or technical jargon.

3. Hybrid Search: Combines Dense Vector Search and Sparse Keyword Search. Scores from both search types are normalized and combined using algorithms like Reciprocal Rank Fusion (RRF) to provide high semantic understanding alongside exact keyword accuracy.

4. Full-Text / Keyword Search: Traditional database search relying on inverted indexes to match exact words or phrases within text documents without vector representation.

Advanced & Structured Search Extensions
------------------------------------------
5. Metadata / Filtered Search: Restricts or pre-filters vector searches based on structured metadata fields (such as document date, author, category, or access permissions) before running similarity operations.

6.Parent-Child / Hierarchical Search: Searches small text chunks (child chunks) for high semantic retrieval precision, but retrieves the larger surrounding section or full document (parent chunk) to feed richer context to the LLM.

7.Multi-Vector / Self-Query Search: Uses an LLM to parse a natural language query into both a structured metadata filter and a unstructured search prompt, executing a refined query against the database.

8. Graph-Based Search (GraphRAG): Extracts entities and relationships from documents to construct a Knowledge Graph, enabling graph traversal and relational reasoning across connected facts.

9. Re-Ranking Search (Cross-Encoder Search): A two-stage retrieval strategy where a fast vector/keyword search retrieves a candidate list (e.g., top 20 chunks), and a heavy Cross-Encoder model re-ranks the candidates to output the most contextually relevant results (e.g., top 3 chunks) for the final LLM prompt.





