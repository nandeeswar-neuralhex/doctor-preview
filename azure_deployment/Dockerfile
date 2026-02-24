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
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m ensurepip --upgrade \
    && pip3 install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install build dependencies FIRST (insightface needs Cython + numpy to compile)
RUN pip3 install --no-cache-dir "Cython>=0.29,<3.0" "numpy>=1.24,<2.0"

# Step 2: Install insightface from source with --no-build-isolation so it uses our numpy
RUN pip3 install --no-cache-dir --no-build-isolation --no-deps insightface==0.7.3

# Step 3: Install GPU onnxruntime (overwrites any CPU version)
RUN pip3 install --no-cache-dir onnxruntime-gpu==1.20.1 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Step 4: Install remaining dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 5: Pin numpy back to 1.x after all installs (avoid binary compat issues)
RUN pip3 install --no-cache-dir "numpy>=1.24,<2.0"

# Copy application source
COPY src/ ./src/

# Download AI models at build time (reduces startup time)
RUN python3 src/download_models.py

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
