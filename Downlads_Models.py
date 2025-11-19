salam in code ro bbin:
import os
import shutil
import subprocess
import sys
# import tarfilea
import zipfile


def ensure_gdown():
    try:
        import gdown
    except ImportError:
        print("[INFO] gdown not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
    finally:
        globals()["gdown"] = __import__("gdown")

def download_files(ids, outdir="downloads"):
    os.makedirs(outdir, exist_ok=True)
    downloaded = []

    for file_id in ids:
        outfile = os.path.join(outdir, file_id)
        print(f"[DOWNLOAD] {file_id}")

        try:
            gdown.download(id=file_id, output=outfile, quiet=False)
        except:
            print("[WARN] direct download failed, trying fuzzy...")
            gdown.download(
                url=f"https://drive.google.com/file/d/{file_id}/view",
                output=outfile,
                quiet=False,
                fuzzy=True
            )

        detected = outfile
        if "." not in outfile:
            try:
                ftype = subprocess.check_output(["file", "-b", "--mime-type", outfile]).decode().strip()
            except:
                ftype = "application/octet-stream"

            if "gzip" in ftype:
                detected = outfile + ".tar.gz"
            elif "zip" in ftype:
                detected = outfile + ".zip"
            else:
                detected = outfile + ".bin"

            os.rename(outfile, detected)

        downloaded.append(detected)
        print(f"[OK] Saved as: {detected}")

    return downloaded


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
    ensure_gdown()

    FILE_IDS = [
        # "1rC6UpkRHCB1qegueU-vnoGPf_hSjXmbO",
        "1T67-LczZQrvYGM-e5Qm_d0jf5TCz-V_e",
        "1ArpFG5VaHgVzDLM2nlgbR6XnHz6XbIOp"
    ]

    
    print("\n=== DOWNLOADING ===")
    downloaded = download_files(FILE_IDS)

    print("\n=== EXTRACTING ===")
    extract_all(downloaded)

    print("\n=== SCANNING FOR id-* ===")
    id_dirs = collect_id_folders()

    print("\n=== MOVING INTO round4 ===")
    move_in_order(id_dirs, dest="round4")

    print("\nAll operations finished successfully.")


