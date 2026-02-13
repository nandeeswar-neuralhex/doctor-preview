
import os
import urllib.request
from config import MODELS_DIR, INSWAPPER_MODEL, GFPGAN_MODEL_PATH

def download_file(url, path):
    if os.path.exists(path):
        print(f"File exists: {path}")
        return
    
    print(f"Downloading {url} to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Add User-Agent to avoid 403/401 errors from some servers
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
        
    print("Download complete.")

def download_models():
    # 1. Inswapper (Face Fusion) - Using reliable GitHub release mirror
    download_file(
        "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        INSWAPPER_MODEL
    )
    
    # 2. GFPGAN (Face Enhancer)
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        GFPGAN_MODEL_PATH
    )

    # 3. InsightFace Buffalo Model (usually downloaded automatically by insightface, but can pre-download if needed)
    # kept simple for now, insightface will fetch if missing.

if __name__ == "__main__":
    download_models()
