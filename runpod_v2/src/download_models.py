
import os
import subprocess
from config import INSWAPPER_MODEL, GFPGAN_MODEL_PATH

def download_file(url, path):
    if os.path.exists(path):
        print(f"File exists: {path}")
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
    download_file(
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        INSWAPPER_MODEL
    )
    
    # 2. GFPGAN (Face Enhancer)
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        GFPGAN_MODEL_PATH
    )

if __name__ == "__main__":
    download_models()

    # 3. InsightFace Buffalo Model (usually downloaded automatically by insightface, but can pre-download if needed)
    # kept simple for now, insightface will fetch if missing.

if __name__ == "__main__":
    download_models()
