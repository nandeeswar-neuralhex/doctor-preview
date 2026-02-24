
import os
import subprocess
from config import INSWAPPER_MODEL, GFPGAN_MODEL_PATH

import hashlib

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url, path, min_size_mb=0, expected_sha256=None):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if min_size_mb > 0 and size_mb < min_size_mb:
            print(f"File exists but is too small ({size_mb:.1f}MB < {min_size_mb}MB). Identifying as corrupted.")
            os.remove(path)
            print(f"Deleted corrupted file: {path}")
        elif expected_sha256:
            print(f"Verifying hash for {path}...")
            current_hash = calculate_sha256(path)
            if current_hash != expected_sha256:
                print(f"Hash mismatch! Expected {expected_sha256}, got {current_hash}")
                os.remove(path)
                print(f"Deleted corrupted file (hash mismatch): {path}")
            else:
                print(f"File integrity verified (SHA256 match).")
                return
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
    # 529MB, SHA256: e4a3f08c753cb72d04e10aa0f7dbe3deebbf399cb48dc476af41638f6c84ceaa
    download_file(
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        INSWAPPER_MODEL,
        min_size_mb=200,
        expected_sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf399cb48dc476af41638f6c84ceaa"
    )
    
    # 2. GFPGAN (Face Enhancer)
    download_file(
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        GFPGAN_MODEL_PATH
    )

    # 3. Wav2Lip GAN (Lip Sync) — 96x96 ONNX from FaceFusion assets
    download_file(
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/wav2lip_gan_96.onnx",
        WAV2LIP_MODEL_PATH,
        min_size_mb=1
    )

if __name__ == "__main__":
    download_models()
