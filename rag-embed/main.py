import os
from reader_local import read_local_file
from reader_gdrive import read_gdrive_file
from indexer import embed_and_store
from search import search_vectors


def process_file(source: str, is_gdrive: bool = False, target_dir: str = ".") -> int:
    """Orchestrates reading, chunking, embedding, and indexing a local or Google Drive file.

    Args:
        source (str): Local file path OR Google Drive File ID.
        is_gdrive (bool): Set True if reading from Google Drive, False for local files.
        target_dir (str): Root path where LanceDB directory is located.

    Returns:
        int: Count of ingested text chunks.

    Example Input (Local):
        process_file("data/sample.pdf", is_gdrive=False, target_dir="/project")

    Example Output (Local):
        Reading Local File: data/sample.pdf
        Embedding chunks with Hugging Face model and writing to LanceDB...
        5
    """
    if is_gdrive:
        print(f"Downloading & Reading Google Drive File ID: {source}")
        text = read_gdrive_file(source)
    else:
        print(f"Reading Local File: {source}")
        text = read_local_file(source)

    print("Embedding chunks with Hugging Face model and writing to LanceDB...")
    num_chunks = embed_and_store(text, source_path=source, target_dir=target_dir)
    print(f"Ingestion complete. {num_chunks} chunk(s) stored.")
    return num_chunks


if __name__ == "__main__":
    # Base directory where LanceDB will reside
    WORK_DIR = os.path.dirname(os.path.abspath(__file__))

    # Example 1: Ingest Local File
    # process_file("sample.pdf", is_gdrive=False, target_dir=WORK_DIR)

    # Example 2: Ingest Google Drive File
    # process_file("YOUR_GDRIVE_FILE_ID", is_gdrive=True, target_dir=WORK_DIR)

    # Example 3: Perform Vector Search and output JSON with vector index
    query_string = "Find main details about project setup"
    json_results = search_vectors(query_string, target_dir=WORK_DIR, limit=3)

    print("\n--- Search Results (JSON) ---")
    print(json_results)

    