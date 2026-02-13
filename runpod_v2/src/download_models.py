
import os
import urllib.request
from config import MODELS_DIR, INSWAPPER_MODEL, GFPGAN_MODEL_PATH

def download_file(url, path):
    if os.path.exists(path):
        print(f"File exists: {path}")
        return
    
    print(f"Downloading {url} to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    print("Download complete.")

def download_models():
    # 1. Inswapper (Face Fusion)
    download_file(
        "https://huggingface.co/eziorry/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        INSWAPPER_MODEL
    )
    
    # 2. GFPGAN (Face Enhancer) - Optional but good to have
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        GFPGAN_MODEL_PATH
    )

    # 3. InsightFace Buffalo Model (usually downloaded automatically by insightface, but can pre-download if needed)
    # kept simple for now, insightface will fetch if missing.

if __name__ == "__main__":
    download_models()
