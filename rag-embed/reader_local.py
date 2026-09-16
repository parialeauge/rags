import os
from pypdf import PdfReader
from docx import Document


def read_local_file(file_path: str) -> str:
    """Reads and extracts raw text from a local file (.txt, .md, .csv, .pdf, .docx).

    Args:
        file_path (str): The relative or absolute file path.

    Returns:
        str: Extracted text content from the file.

    Example Input:
        file_path = "data/report.pdf"

    Example Output:
        "Quarterly Financial Report\nSummary of revenue and expenses for Q1..."
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.txt', '.md', '.csv']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    elif ext == '.pdf':
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    elif ext == '.docx':
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

