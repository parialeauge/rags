import json
from config_ai import TABLE_NAME, get_db_connection


def search_vectors(query: str, target_dir: str, limit: int = 3) -> str:
    """Performs semantic vector search in LanceDB and returns results formatted as a JSON string.

    Args:
        query (str): Natural language search prompt.
        target_dir (str): Directory path containing the .lancedb database folder.
        limit (int): Number of top search results to return. Default is 3.

    Returns:
        str: JSON string containing query, distance scores, text, source, and vector_index.

    Example Input:
        query = "How to configure Google Drive?"
        target_dir = "/projects/my_lancedb_project"
        limit = 1

    Example Output:
        {
          "query": "How to configure Google Drive?",
          "results": [
            {
              "vector_index": 0,
              "score": 0.2451,
              "text": "Place credentials.json in root directory and run authentication...",
              "source": "config_gdrive.py",
              "id": "e8123456-89ab-cdef-0123-456789abcdef"
            }
          ]
        }
    """
    db = get_db_connection(target_dir)
    table = db.open_table(TABLE_NAME)

    results = table.search(query).limit(limit).to_pandas()

    output = []
    for idx, row in results.iterrows():
        output.append({
            "vector_index": int(row["vector_index"]),
            "score": float(row.get("_distance", 0.0)),
            "text": row["text"],
            "source": row["source"],
            "id": row["id"]
        })

    return json.dumps({"query": query, "results": output}, indent=2)