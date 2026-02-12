"""
Model download script - Downloads required AI models during Docker build
"""
import os
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path("/app/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model URLs
MODELS = {
    "inswapper_128.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
    "buffalo_l.zip": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "wav2lip.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_gan_96.onnx",
}

def download_file(url: str, destination: Path):
    """Download a file with progress"""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"  -> Saved to {destination}")
    except Exception as e:
        print(f"  -> Failed: {e}")
        # Try alternative download method
        import requests
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  -> Saved to {destination} (via requests)")

def download_models():
    """Download all required models"""
    print("=" * 50)
    print("Downloading FaceFusion models...")
    print("=" * 50)
    
    for filename, url in MODELS.items():
        dest_path = MODELS_DIR / filename
        if dest_path.exists():
            print(f"Skipping {filename} (already exists)")
            continue
        
        download_file(url, dest_path)
        
        # Extract zip files
        if filename.endswith(".zip"):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                zip_ref.extractall(MODELS_DIR)
            dest_path.unlink()  # Remove zip after extraction
    
    print("=" * 50)
    print("Model download complete!")
    print("=" * 50)

if __name__ == "__main__":
    download_models()
