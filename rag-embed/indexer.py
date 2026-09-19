import uuid
from typing import List
from lancedb.pydantic import LanceModel, Vector
from config_ai import func, TABLE_NAME, get_db_connection


class DocumentSchema(LanceModel):
    """Pydantic schema for LanceDB document table."""
    id: str
    vector_index: int
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()
    source: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits a long string into overlapping text chunks by word count.

    Args:
        text (str): Input text string.
        chunk_size (int): Max number of words per chunk. Default is 500.
        overlap (int): Word overlap between consecutive chunks. Default is 50.

    Returns:
        List[str]: List of text chunks.

    Example Input:
        text = "word1 word2 word3 ... word1200", chunk_size = 500, overlap = 50

    Example Output:
        ["word1 word2 ... word500", "word451 word452 ... word950", "word901 ... word1200"]
    """
    words = text.split() 
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def embed_and_store(text: str, source_path: str, target_dir: str) -> int:
    """Chunks text, computes embeddings using Hugging Face, and stores vectors in LanceDB.

    Args:
        text (str): Extracted file content string.
        source_path (str): File path or Google Drive ID source identifier.
        target_dir (str): Directory path where LanceDB is stored.

    Returns:
        int: Total number of chunks successfully embedded and inserted.

    Example Input:
        text = "Sample project document details..."
        source_path = "docs/readme.txt"
        target_dir = "/projects/my_lancedb_project"

    Example Output:
        3
    """
    db = get_db_connection(target_dir)
    chunks = chunk_text(text)

    # Determine starting vector_index offset based on existing records
    try:
        table = db.open_table(TABLE_NAME)
        start_idx = len(table.to_pandas())
    except Exception:
        start_idx = 0
        table = None

    data = [
        DocumentSchema(
            id=str(uuid.uuid4()),
            vector_index=start_idx + idx,
            text=chunk,
            source=source_path
        )
        for idx, chunk in enumerate(chunks)
    ]
    # alternative way to create the data
    # data = []
    # for idx, chunk in enumerate(chunks):
    # item = DocumentSchema(
    #     id=str(uuid.uuid4()),
    #     vector_index=start_idx + idx,
    #     text=chunk,
    #     source=source_path
    # )
    # data.append(item)

    if table is None:
        db.create_table(TABLE_NAME, schema=DocumentSchema, data=data)
    else:
        table.add(data)

    return len(data)
