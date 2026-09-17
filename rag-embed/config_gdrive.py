import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json' 
TOKEN_FILE = 'token.json' 


def get_gdrive_service():
    """Authenticates and builds the Google Drive API client service using OAuth 2.0.

    Returns:
        Resource: Authenticated Google Drive API service client.

    Example Input:
        None (Requires credentials.json present in root folder)

    Example Output:
        <googleapiclient.discovery.Resource object at 0x7f8a12345890>
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0) 
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    from googleapiclient.discovery import build
    return build('drive', 'v3', credentials=creds)