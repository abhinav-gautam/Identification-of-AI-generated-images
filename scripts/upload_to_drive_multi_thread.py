import threading
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob, os
import time

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


def upload_to_drive(
    folders,
    destination_folder_name,
    thread_index,
    start_index,
    files_per_thread,
    files,
):
    try:
        # To track the number of files uploaded
        counter = 0

        # Loop over the files
        for i in range(start_index, files_per_thread + start_index):

            # Loop over the folders present in GDrive
            for folder in folders:
                if folder["title"] == destination_folder_name:

                    # Open the file that needs to be written
                    with open(files[i], "r") as f:
                        filename = os.path.basename(f.name)

                        # Get the list of files with same name
                        file_list = drive.ListFile(
                            {
                                "q": f"title='{filename}' and '{folder['id']}' in parents and trashed=false"
                            }
                        ).GetList()

                        # If file not already exists
                        if not file_list:
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
                            print(f"[Thread: {thread_index+1}] {filename} uploaded")
                            print(
                                f"[Thread: {thread_index+1}] Total {str(counter)} files uploaded"
                            )
                        else:
                            print(
                                f"[Thread: {thread_index+1}] {filename} already exists"
                            )

        print(f"[Thread: {thread_index+1}] Uploaded {counter} files.")

    except Exception as e:
        print(f"[Thread: {thread_index+1}] Exception occurred.")
        print(f"[Thread: {thread_index+1}] Waiting to restart.")
        print(e)
        return e


def handle_thread(
    folders,
    destination_folder_name,
    thread_index,
    start_index,
    files_per_thread,
    files,
):
    # Attempt to run the thread and handle exceptions
    result = upload_to_drive(
        folders,
        destination_folder_name,
        thread_index,
        start_index,
        files_per_thread,
        files,
    )

    if isinstance(result, Exception):
        time.sleep(120)

        print(f"[Thread: {thread_index+1}] Restarting.")
        handle_thread(
            folders,
            destination_folder_name,
            thread_index,
            start_index,
            files_per_thread,
            files,
        )  # Recursive restart


def start_uploading(
    source_folder_path,
    destination_folder_name,
    num_threads,
    files_per_thread,
    start_offset=0,
):
    # Get the folder from the drive
    folders = drive.ListFile(
        {
            "q": "title='"
            + destination_folder_name
            + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        }
    ).GetList()

    threads = []
    os.chdir(source_folder_path)
    files = glob.glob("*.jpg")

    for index in range(num_threads):
        start_index = files_per_thread * index + start_offset
        # Create a Thread object that targets the thread_function
        x = threading.Thread(
            target=handle_thread,
            args=(
                folders,
                destination_folder_name,
                index,
                start_index,
                files_per_thread,
                files,
            ),
        )
        threads.append(x)
        print(f"[Thread: {index+1}] Started")
        x.start()  # Start the thread execution

    # Wait for all threads to complete
    for index, thread in enumerate(threads):
        thread.join()
        print(f"[Thread: {index+1}] Finished")


if __name__ == "__main__":
    source_folder_path = "D:\Creations\Deep Learning\Projects\Identification of AI-generated images\datasets\CIFAKE\\train\REAL"
    destination_folder_name = "REAL"
    start_uploading(
        source_folder_path,
        destination_folder_name,
        num_threads=100,
        files_per_thread=100,
        start_offset=10000,
    )
