import os
import shutil
from math import ceil


def distribute_files(source_folder, destination_folder, num_subfolders):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get all files from the source folder
    files = [
        file
        for file in os.listdir(source_folder)
        if os.path.isfile(os.path.join(source_folder, file))
    ]

    # Determine the number of files per folder
    files_per_folder = ceil(len(files) / num_subfolders)

    # Create subfolders and distribute files
    current_folder = 0
    for index, file in enumerate(files):
        # Determine the current subfolder
        if index % files_per_folder == 0 and current_folder < num_subfolders:
            subfolder_path = os.path.join(
                destination_folder, f"subfolder_{current_folder + 1}"
            )
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            current_folder += 1

        # Copy file to the current subfolder
        src_file_path = os.path.join(source_folder, file)
        dst_file_path = os.path.join(subfolder_path, file)
        shutil.copy(src_file_path, dst_file_path)

    print(f"Files have been distributed into {current_folder} subfolders.")


# Usage
source_folder = "D:\Creations\Deep Learning\Projects\Identification of AI-generated images\datasets\CIFAKE\\test\FAKE"
destination_folder = "D:\Creations\Deep Learning\Projects\Identification of AI-generated images\datasets\CIFAKE\\test\FAKE splitted"
num_subfolders = 1000

distribute_files(source_folder, destination_folder, num_subfolders)
