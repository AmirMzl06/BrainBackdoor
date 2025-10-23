import os
import tarfile

base_folder = "models_all"

tar_files = sorted([f for f in os.listdir(base_folder) if f.endswith(".tar.gz")])

if not tar_files:
    print("File not found!")
else:
    for idx, tar_file in enumerate(tar_files, start=1):
        tar_path = os.path.join(base_folder, tar_file)

        extracted_folder_name = f"{idx:02d}"
        extracted_folder = os.path.join(base_folder, extracted_folder_name)

        os.makedirs(extracted_folder, exist_ok=True)

        print(f"Extracting {tar_file} -> {extracted_folder}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extracted_folder)

print("done")
