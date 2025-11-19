# salam in code ro bbin:
# import os
# import shutil
# import subprocess
# import sys
# # import tarfilea
# import zipfile


# def ensure_gdown():
#     try:
#         import gdown
#     except ImportError:
#         print("[INFO] gdown not found. Installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"])
#     finally:
#         globals()["gdown"] = __import__("gdown")

# def download_files(ids, outdir="downloads"):
#     os.makedirs(outdir, exist_ok=True)
#     downloaded = []

#     for file_id in ids:
#         outfile = os.path.join(outdir, file_id)
#         print(f"[DOWNLOAD] {file_id}")

#         try:
#             gdown.download(id=file_id, output=outfile, quiet=False)
#         except:
#             print("[WARN] direct download failed, trying fuzzy...")
#             gdown.download(
#                 url=f"https://drive.google.com/file/d/{file_id}/view",
#                 output=outfile,
#                 quiet=False,
#                 fuzzy=True
#             )

#         detected = outfile
#         if "." not in outfile:
#             try:
#                 ftype = subprocess.check_output(["file", "-b", "--mime-type", outfile]).decode().strip()
#             except:
#                 ftype = "application/octet-stream"

#             if "gzip" in ftype:
#                 detected = outfile + ".tar.gz"
#             elif "zip" in ftype:
#                 detected = outfile + ".zip"
#             else:
#                 detected = outfile + ".bin"

#             os.rename(outfile, detected)

#         downloaded.append(detected)
#         print(f"[OK] Saved as: {detected}")

#     return downloaded


# def extract_all(files, dest="temp_extract"):
#     os.makedirs(dest, exist_ok=True)
#     extracted_dirs = []

#     for f in files:
#         name = os.path.basename(f)
#         out = os.path.join(dest, name)
#         os.makedirs(out, exist_ok=True)

#         print(f"[EXTRACT] {f} -> {out}")

#         try:
#             if tarfile.is_tarfile(f):
#                 with tarfile.open(f) as tar:
#                     tar.extractall(path=out)
#             elif zipfile.is_zipfile(f):
#                 with zipfile.ZipFile(f) as z:
#                     z.extractall(out)
#             else:
#                 print(f"[WARN] Unknown format: {f}")
#         except Exception as e:
#             print(f"[ERROR] Failed to extract {f}: {e}")

#         extracted_dirs.append(out)

#     return extracted_dirs


# def collect_id_folders(root="temp_extract"):
#     id_dirs = []
#     for r, dirs, files in os.walk(root):
#         for d in dirs:
#             if d.startswith("id-"):
#                 id_dirs.append(os.path.join(r, d))

#     id_dirs = sorted(set(id_dirs))
#     print(f"[INFO] Found {len(id_dirs)} id-* folders.")
#     return id_dirs


# def move_in_order(id_dirs, dest="round4"):
#     os.makedirs(dest, exist_ok=True)
#     pad = max(3, len(str(len(id_dirs) - 1)))

#     for idx, src in enumerate(id_dirs):
#         newname = str(idx).zfill(pad)
#         dst = os.path.join(dest, newname)

#         print(f"[MOVE] {src} -> {dst}")

#         try:
#             shutil.copytree(src, dst)
#         except:
#             shutil.move(src, dst)

#     print("[DONE] All folders moved.")


# if __name__ == "__main__":
#     ensure_gdown()

#     FILE_IDS = [
#         # "1rC6UpkRHCB1qegueU-vnoGPf_hSjXmbO",
#         "1T67-LczZQrvYGM-e5Qm_d0jf5TCz-V_e",
#         "1ArpFG5VaHgVzDLM2nlgbR6XnHz6XbIOp"
#     ]

    
#     print("\n=== DOWNLOADING ===")
#     downloaded = download_files(FILE_IDS)

#     print("\n=== EXTRACTING ===")
#     extract_all(downloaded)

#     print("\n=== SCANNING FOR id-* ===")
#     id_dirs = collect_id_folders()

#     print("\n=== MOVING INTO round4 ===")
#     move_in_order(id_dirs, dest="round4")

#     print("\nAll operations finished successfully.")



import os
import shutil
import tarfile
import gdown
import sys

# --- تنظیمات ---
FILE_IDS = [
    "17eE_TiatDf0iKb6O9UjZL1Slr12Zi_jE",  # آیدی فایل اول
    "1fH1v8o_0szzKtU8qCcvtONcJUWYj4Dtk"   # آیدی فایل دوم
]
DOWNLOAD_EXTRACT_DIR = "files"  # پوشه‌ای که فایل‌ها در آن دانلود و استخراج می‌شوند
FINAL_DIR = "round3"            # پوشه نهایی برای فایل‌های تغییر نام یافته

def main():
    print("--- 🏁 شروع اسکریپت ---")

    # --- 1. ساخت پوشه‌های مورد نیاز ---
    try:
        print(f"ساخت پوشه‌ها: {DOWNLOAD_EXTRACT_DIR}, {FINAL_DIR}")
        os.makedirs(DOWNLOAD_EXTRACT_DIR, exist_ok=True)
        os.makedirs(FINAL_DIR, exist_ok=True)
    except Exception as e:
        print(f"خطا در ساخت پوشه: {e}")
        return

    downloaded_tars = []  # لیستی برای نگهداری مسیر فایل‌های tar دانلود شده

    # --- 2. دانلود و استخراج فایل‌ها ---
    for i, file_id in enumerate(FILE_IDS):
        download_path = os.path.join(DOWNLOAD_EXTRACT_DIR, f"archive_{i}.tar.gz")
        print(f"\n⬇️ درحال دانلود فایل {i+1} (ID: {file_id}) به {download_path}...")
        
        try:
            # دانلود فایل با gdown
            gdown.download(id=file_id, output=download_path, quiet=False)
            print("✅ دانلود کامل شد.")
            downloaded_tars.append(download_path)
        except Exception as e:
            print(f"❌ خطا در دانلود فایل {file_id}: {e}")
            continue  # اگر دانلود نشد، برو سراغ فایل بعدی

        print(f"🗜️ درحال استخراج {download_path} به {DOWNLOAD_EXTRACT_DIR}...")
        try:
            # باز کردن و استخراج فایل tar.gz
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path=DOWNLOAD_EXTRACT_DIR)
            print("✅ استخراج کامل شد.")
        except tarfile.TarError as e:
            print(f"❌ خطا در استخراج فایل {download_path}: {e}")
        except Exception as e:
            print(f"❌ یک خطای غیرمنتظره در استخراج رخ داد: {e}")

    # --- 3. پیدا کردن پوشه‌های 'id-' ---
    print(f"\n🔍 درحال جستجو برای پوشه‌های 'id-' در {DOWNLOAD_EXTRACT_DIR}...")
    try:
        all_items = os.listdir(DOWNLOAD_EXTRACT_DIR)
        # فیلتر کردن فقط پوشه‌هایی که با 'id-' شروع می‌شوند
        id_dirs = [
            item for item in all_items 
            if item.startswith("id-") and os.path.isdir(os.path.join(DOWNLOAD_EXTRACT_DIR, item))
        ]
        
        if not id_dirs:
            print(f"⚠️ هشدار: هیچ پوشه‌ای با پیشوند 'id-' در {DOWNLOAD_EXTRACT_DIR} پیدا نشد.")
            return

        # مرتب‌سازی پوشه‌ها برای تضمین ترتیب 000, 001, ...
        id_dirs.sort()
        print(f"✅ پیدا شد: {len(id_dirs)} پوشه 'id-'.")

    except Exception as e:
        print(f"❌ خطا در خواندن پوشه {DOWNLOAD_EXTRACT_DIR}: {e}")
        return

    # --- 4. انتقال و تغییر نام ---
    print(f"\n🚚 درحال انتقال و تغییر نام پوشه‌ها به {FINAL_DIR}...")
    counter = 0
    for dir_name in id_dirs:
        old_path = os.path.join(DOWNLOAD_EXTRACT_DIR, dir_name)
        
        # فرمت‌بندی نام جدید به صورت سه رقمی (e.g., 000, 001, ..., 010, ..., 100)
        new_name = f"{counter:03d}" 
        new_path = os.path.join(FINAL_DIR, new_name)
        
        try:
            # انتقال و تغییر نام در یک حرکت
            shutil.move(old_path, new_path)
            print(f"  {dir_name}  ->  {new_path}")
            counter += 1
        except shutil.Error as e:
             print(f"❌ خطا در انتقال {old_path}: {e} (ممکن است مقصد از قبل وجود داشته باشد)")
        except Exception as e:
            print(f"❌ خطای ناشناخته در انتقال {old_path}: {e}")

    # --- 5. پاکسازی فایل‌های .tar.gz ---
    print(f"\n🧹 پاکسازی فایل‌های .tar.gz دانلود شده از پوشه {DOWNLOAD_EXTRACT_DIR}...")
    for tar_path in downloaded_tars:
        try:
            os.remove(tar_path)
            print(f"  - حذف شد: {tar_path}")
        except Exception as e:
            print(f"❌ خطا در حذف {tar_path}: {e}")

    print(f"\n--- 🎉 عملیات تمام شد ---")
    print(f"{counter} پوشه به {FINAL_DIR} منتقل و تغییر نام داده شد.")

if __name__ == "__main__":
    main()


