import os
import lancedb
from lancedb.embeddings import get_registry

# Read Hugging Face token from environment; leave unset for public models
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")

# Configure Hugging Face token in environment if provided
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Free Hugging Face embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "documents"

# Initialize LanceDB embedding function registration
func = get_registry().get("sentence-transformers").create(name=EMBEDDING_MODEL_NAME)


def get_db_connection(db_dir: str) -> lancedb.DBConnection:
    """Initializes and returns a LanceDB connection stored in the specified directory.

    Args:
        db_dir (str): Directory path where the database folder will be created/accessed.

    Returns:
        lancedb.DBConnection: Active LanceDB connection instance.

    Example Input:
        db_dir = "/path/to/my_project"

    Example Output:
        <lancedb.remote.db.LanceDBConnection object at 0x7f8a12345670>
    """
    os.makedirs(db_dir, exist_ok=True) # create the directory if it doesn't exist
    db_path = os.path.join(db_dir, ".lancedb") # create the database path
    return lancedb.connect(db_path) # connect to the database