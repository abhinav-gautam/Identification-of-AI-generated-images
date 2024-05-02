from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, BatchHttpRequest
import os
import mimetypes

SCOPES = ["https://www.googleapis.com/auth/drive"]


def authenticate():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


def callback(request_id, response, exception):
    if exception:
        print(exception)
    else:
        print(f"Request ID: {request_id}, File ID: {response.get('id')}")


def batch_upload_images(creds, directory_path):
    service = build("drive", "v3", credentials=creds)
    batch = service.new_batch_http_request(callback=callback)

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith("image"):
                file_metadata = {"name": filename}
                media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
                request = service.files().create(
                    body=file_metadata, media_body=media, fields="id"
                )
                batch.add(request)

    batch.execute()


# Authenticate and upload files
creds = authenticate()
batch_upload_images(
    creds,
    "D:\Creations\Deep Learning\Projects\Identification of AI-generated images\scripts\docs",
)
