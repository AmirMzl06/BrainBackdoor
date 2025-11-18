import os
import shutil
import tarfile
import zipfile
import subprocess
import time

DEST_FOLDER = 'round4'
TEMP_FOLDER = 'temp_extraction'
FILE_LINKS = [
    {'id': '1nvMCYgZhr7Xh05kbC38txvOKe7IFwhuN', 'name': 'archive_1.tar.gz'},
    {'id': '1XaAsEupjstjNcLaUWLYg_t9SRwLzWWz3', 'name': 'archive_2.tar.gz'},
    {'id': '1T67-LczZQrvYGM-e5Qm_d0jf5TCz-V_e', 'name': 'archive_3.tar.gz'}
]


def extract_file(file_path, extract_to):
    print(f"Attempting to extract: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found locally: {file_path}")
        return False
        
    try:
        if file_path.endswith("tar.gz") or file_path.endswith("tar"):
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=extract_to)
        elif file_path.endswith("zip"):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(f"Warning: Unknown format for {file_path}. Skipping extraction.")
            return False
        return True
    except Exception as e:
        print(f"FATAL ERROR extracting {file_path}: {e}")
        return False

print("--- Setup ---")
if os.path.exists(DEST_FOLDER):
    print(f"WARNING: '{DEST_FOLDER}' already exists. Files will be appended.")
else:
    os.makedirs(DEST_FOLDER)
    print(f"Folder '{DEST_FOLDER}' created.")

global_counter = 0

for file_info in FILE_LINKS:
    print(f"\n--- Processing {file_info['name']} ---")
    
    download_cmd = f"gdown --id {file_info['id']} -O {file_info['name']}"
    print(f"Running command: {download_cmd}")
    result = subprocess.run(download_cmd, shell=True, check=False)
    
    if result.returncode != 0:
        print(f"ERROR: gdown failed for {file_info['name']}. Check gdown installation.")
        continue

    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
    os.makedirs(TEMP_FOLDER)
    
    if not extract_file(file_info['name'], TEMP_FOLDER):
        continue

    found_files = []
    for root, dirs, files in os.walk(TEMP_FOLDER):
        for file in files:
            if file.startswith("id-"):
                found_files.append(os.path.join(root, file))
    
    found_files.sort()
    
    print(f"Found {len(found_files)} files matching 'id-*'. Moving and renaming...")
    
    for src_path in found_files:
        filename = os.path.basename(src_path)
        name, ext = os.path.splitext(filename)
        
        new_name = f"{global_counter:03d}{ext}"
        dst_path = os.path.join(DEST_FOLDER, new_name)
        
        try:
            shutil.move(src_path, dst_path)
            global_counter += 1
        except Exception as e:
            print(f"FATAL ERROR moving {src_path} to {dst_path}: {e}")
            
    os.remove(file_info['name'])
    shutil.rmtree(TEMP_FOLDER)

print("\n" + "="*50)
print("PROCESS COMPLETE!")
print(f"Total files collected in '{DEST_FOLDER}': {global_counter}")
print("="*50)

print("\n--- Final Check (First 5 files in round4) ---")
try:
    print('\n'.join(os.listdir(DEST_FOLDER)[:5]))
except:
    print("Could not list files.")
