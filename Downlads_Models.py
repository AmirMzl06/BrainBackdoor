import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
import requests

def gdrive_download(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"[DOWNLOAD] {file_id}")

    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)

    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    CHUNK = 32768
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(CHUNK):
            if chunk:
                f.write(chunk)

    print(f"[OK] Saved as: {dest_path}")
    return dest_path


def extract_all(files, dest="temp_extract"):
    os.makedirs(dest, exist_ok=True)
    extracted_dirs = []

    for f in files:
        name = os.path.basename(f)
        out = os.path.join(dest, name)
        os.makedirs(out, exist_ok=True)

        print(f"[EXTRACT] {f} -> {out}")

        try:
            if tarfile.is_tarfile(f):
                with tarfile.open(f) as tar:
                    tar.extractall(path=out)
            elif zipfile.is_zipfile(f):
                with zipfile.ZipFile(f) as z:
                    z.extractall(out)
            else:
                print(f"[WARN] Unknown format: {f}")
        except Exception as e:
            print(f"[ERROR] Failed to extract {f}: {e}")

        extracted_dirs.append(out)

    return extracted_dirs

def collect_id_folders(root="temp_extract"):
    id_dirs = []
    for r, dirs, files in os.walk(root):
        for d in dirs:
            if d.startswith("id-"):
                id_dirs.append(os.path.join(r, d))

    id_dirs = sorted(set(id_dirs))
    print(f"[INFO] Found {len(id_dirs)} id-* folders.")
    return id_dirs


def move_in_order(id_dirs, dest="round4"):
    os.makedirs(dest, exist_ok=True)
    pad = max(3, len(str(len(id_dirs) - 1)))

    for idx, src in enumerate(id_dirs):
        newname = str(idx).zfill(pad)
        dst = os.path.join(dest, newname)

        print(f"[MOVE] {src} -> {dst}")

        try:
            shutil.copytree(src, dst)
        except:
            shutil.move(src, dst)

    print("[DONE] All folders moved.")

if __name__ == "__main__":

    FILE_IDS = [
        "1rC6UpkRHCB1qegueU-vnoGPf_hSjXmbO",
        "1ArpFG5VaHgVzDLM2nlgbR6XnHz6XbIOp"
    ]

    print("\n=== DOWNLOADING ===")
    downloaded = []
    for fid in FILE_IDS:
        fname = fid + ".download"
        downloaded.append(gdrive_download(fid, fname))

    print("\n=== EXTRACTING ===")
    extract_all(downloaded)

    print("\n=== SCANNING FOR id-* ===")
    id_dirs = collect_id_folders()

    print("\n=== MOVING INTO round4 ===")
    move_in_order(id_dirs, dest="round4")

    print("\nAll operations finished successfully.")
