
import os
import subprocess
from config import INSWAPPER_MODEL, GFPGAN_MODEL_PATH

def download_file(url, path, min_size_mb=0):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if min_size_mb > 0 and size_mb < min_size_mb:
            print(f"File exists but is too small ({size_mb:.1f}MB < {min_size_mb}MB). Identifying as corrupted.")
            os.remove(path)
            print(f"Deleted corrupted file: {path}")
        else:
            print(f"File exists and size is valid ({size_mb:.1f}MB): {path}")
            return
    
    print(f"Downloading {url} to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Use wget or curl for robustness
    try:
        # Try wget first
        subprocess.run(["wget", "-O", path, url], check=True)
        print("Download complete (wget).")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("wget failed or not found. Trying curl...")
        try:
            # Try curl
            subprocess.run(["curl", "-L", "-o", path, url], check=True)
            print("Download complete (curl).")
        except Exception as e:
             print(f"Download failed: {e}")
             # Don't crash, let the app try to run (will fail later if model really needed)

def download_models():
    # 1. Inswapper (Face Fusion)
    # The ONNX file is ~529MB. If it's < 200MB, it's definitely corrupted (e.g. 404 HTML)
    download_file(
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        INSWAPPER_MODEL,
        min_size_mb=200
    )
    
    # 2. GFPGAN (Face Enhancer)
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        GFPGAN_MODEL_PATH
    )

if __name__ == "__main__":
    download_models()
