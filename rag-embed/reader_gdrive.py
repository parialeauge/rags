import os
import io
from googleapiclient.http import MediaIoBaseDownload
from config_gdrive import get_gdrive_service
from reader_local import read_local_file


def read_gdrive_file(file_id: str) -> str:
    """Reads text content from Google Drive by file ID (supports Google Docs and binary files).

    Args:
        file_id (str): The unique ID of the Google Drive file.

    Returns:
        str: Extracted text content of the file.

    Example Input:
        file_id = "1A2b3C4d5E6f7G8h9I0j"

    Example Output:
        "Meeting Minutes\nDate: 2026-09-15\nAttendees: Alice, Bob, Charlie..."
    """
    service = get_gdrive_service()

    # Fetch metadata to determine mimeType
    file_metadata = service.files().get(fileId=file_id, fields='mimeType, name').execute()
    mime_type = file_metadata.get('mimeType')

    # Handle native Google Docs
    if mime_type == 'application/vnd.google-apps.document':
        request = service.files().export_media(fileId=file_id, mimeType='text/plain')
        response = request.execute()
        return response.decode('utf-8')

    # Handle binary files (PDFs, DOCX, TXT, CSV) by temporary download
    request = service.files().get_media(fileId=file_id)
    temp_filename = f"temp_{file_metadata['name']}"

    with open(temp_filename, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    try:
        content = read_local_file(temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return content
    