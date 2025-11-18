import os
import shutil
import tarfile
import zipfile
import glob
import gdown 

DEST_FOLDER = 'round4'
TEMP_FOLDER = 'temp_extraction'
FILE_LINKS = [
    {'id': '1nvMCYgZhr7Xh05kbC38txvOKe7IFwhuN', 'name': 'archive_1.tar.gz'},
    {'id': '1XaAsEupjstjNcLaUWLYg_t9SRwLzWWz3', 'name': 'archive_2.tar.gz'}, # فرض بر این است که فایل‌ها tar.gz هستند
    {'id': '1T67-LczZQrvYGM-e5Qm_d0jf5TCz-V_e', 'name': 'archive_3.tar.gz'}
]

if os.path.exists(DEST_FOLDER):
    shutil.rmtree(DEST_FOLDER)
os.makedirs(DEST_FOLDER)
print(f"Folder '{DEST_FOLDER}' created.")

global_counter = 0

def extract_file(file_path, extract_to):
    if file_path.endswith("tar.gz") or file_path.endswith("tar"):
        with tarfile.open(file_path, "r:*") as tar:
            tar.extractall(path=extract_to)
    elif file_path.endswith("zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"Unknown format for {file_path}")

for file_info in FILE_LINKS:
    print(f"\n--- Processing {file_info['name']} ---")
   
    download_cmd = f"gdown --id {file_info['id']} -O {file_info['name']}"
    os.system(download_cmd)
    
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
    os.makedirs(TEMP_FOLDER)
    
    print(f"Extracting {file_info['name']}...")
    try:
        extract_file(file_info['name'], TEMP_FOLDER)
    except Exception as e:
        print(f"Error extracting {file_info['name']}: {e}")
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
        
        shutil.move(src_path, dst_path)
        global_counter += 1

    os.remove(file_info['name'])
    shutil.rmtree(TEMP_FOLDER)

print("\n" + "="*40)
print(f"Process Complete!")
print(f"Total files in '{DEST_FOLDER}': {global_counter}")
print(f"Files range from 000 to {global_counter-1:03d}")
print("="*40)

