import subprocess
import sys
import os

try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

folder_url = "https://drive.google.com/drive/folders/1I59inpSNP0Fr6oSfkOL8Gigo0WIXZCBL?usp=drive_link"
download_dir = "models_all"

if os.path.exists(download_dir) and os.listdir(download_dir):
    print("Downloaded already)
else:
    os.makedirs(download_dir, exist_ok=True)
    gdown.download_folder(url=folder_url, output=download_dir, quiet=False, use_cookies=False)
