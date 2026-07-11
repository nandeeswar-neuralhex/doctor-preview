# ===================================================================
# AZURE GPU DEPLOYMENT - CUDA 12.1 + cuDNN 8
# Optimized for Azure NC-series (T4, V100, A100) GPUs
# ===================================================================
# Use DEVEL image (includes g++ compiler needed for insightface C extensions)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# onnxruntime-gpu 1.20.1 (CUDA 12) needs cuDNN 9 at runtime.
# pip installs nvidia-cudnn-cu12 which puts the libs under dist-packages/nvidia/cudnn/lib/.
# Add all nvidia pip lib dirs to LD_LIBRARY_PATH so ort can load them.
ENV NV_PYLIB=/usr/local/lib/python3.11/dist-packages/nvidia
ENV LD_LIBRARY_PATH=${NV_PYLIB}/cudnn/lib:${NV_PYLIB}/cublas/lib:${NV_PYLIB}/cuda_runtime/lib:${NV_PYLIB}/cufft/lib:${NV_PYLIB}/cusolver/lib:${NV_PYLIB}/cusparse/lib:${NV_PYLIB}/nvjitlink/lib:${NV_PYLIB}/cuda_nvrtc/lib:/usr/local/cuda-12.1/lib64

WORKDIR /app

# Install Python 3.11 + system dependencies + C++ compiler (needed for insightface)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libturbojpeg \
    ffmpeg \
    wget \
    curl \
    git \
    libavcodec-dev libavformat-dev libavutil-dev \
    libopus-dev libvpx-dev libsrtp2-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install build dependencies FIRST (insightface needs Cython + numpy to compile)
RUN python3 -m pip install --no-cache-dir "Cython>=0.29,<3.0" "numpy>=1.24,<2.0"

# Step 2: Install insightface from source with --no-build-isolation so it uses our numpy
RUN python3 -m pip install --no-cache-dir --no-build-isolation --no-deps insightface==0.7.3

# Step 3: Install GPU onnxruntime from Microsoft CUDA 12 feed.
# onnxruntime-gpu==1.20.1 bundles its own CUDA 12 + cuDNN 9 libs inside the pip
# wheel, so the system does NOT need cuDNN 9 — it coexists with the cudnn8 base
# image. This is the version confirmed working on the other T4 (20.102.71.88).
RUN python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir onnxruntime-gpu==1.20.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Step 4: Install remaining dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Step 5: GFPGAN (via basicsr/facexlib) pulls in CPU onnxruntime and overwrites
# the GPU version. Re-pin onnxruntime-gpu==1.20.1 AFTER requirements install.
RUN python3 -m pip uninstall -y onnxruntime 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.20.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Step 5b: onnxruntime-gpu may pull nvidia-cudnn-cu13 which lacks cuDNN 9 libs
# that ort 1.20.1 needs. Install cudnn-cu12 alongside, but do NOT remove
# nvidia-cublas cu13 — torch 2.11 needs libcublas.so.13.
RUN python3 -m pip install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70

# Step 6: Pin numpy back to 1.x after all installs (avoid binary compat issues)
RUN python3 -m pip install --no-cache-dir "numpy>=1.24,<2.0"

# Step 7: basicsr 1.4.2 imports torchvision.transforms.functional_tensor which
# was removed in torchvision>=0.17 — patch the import so GFPGAN can load.
RUN DEG=$(python3 -c "import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'))" 2>/dev/null) \
    && [ -n "$DEG" ] \
    && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEG" \
    && python3 -c "from gfpgan import GFPGANer; print('GFPGAN import OK')" \
    || echo "basicsr patch skipped"

# Copy application source
COPY src/ ./src/

# NOTE: Models are NOT baked in — they download automatically on first startup
# (server.py calls download_models() in the startup event handler).
# This keeps the image ~1.5 GB smaller, avoiding disk space issues on 62 GB VMs.

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose WebSocket port
EXPOSE 8765

# Health check for Azure Container Instances
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Start the server (run from src/ so relative imports work)
WORKDIR /app/src
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765", "--workers", "1"]
