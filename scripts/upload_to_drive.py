from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob, os

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

folderName = "FAKE"
input_path = "D:\\Creations\\Deep Learning\\Projects\\Identification of AI-generated images\\datasets\\CIFAKE\\test\\FAKE splitted\\subfolder_1"

folders = drive.ListFile(
    {
        "q": "title='"
        + folderName
        + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }
).GetList()

os.chdir(input_path)

counter = 0
for file in glob.glob("*.jpg"):
    for folder in folders:
        if folder["title"] == folderName:
            with open(file, "r") as f:
                filename = os.path.basename(f.name)
                file_drive = drive.CreateFile(
                    {
                        "title": filename,
                        "mimeType": "image/jpeg",
                        "parents": [{"id": folder["id"]}],
                    }
                )

                file_drive.SetContentFile(filename)
                file_drive.Upload()

                counter += 1
                print("[+] " + filename + " uploaded")
                print("[+] Total " + str(counter) + " files uploaded")

print(f"Uploaded {counter} files")
