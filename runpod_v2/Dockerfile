# ── CUDA 12.1 + cuDNN 8 base image for GPU inference ──
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.11 + system dependencies for OpenCV / image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libturbojpeg \
    wget \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m ensurepip --upgrade \
    && pip3 install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Order matters: insightface installs CPU onnxruntime, then onnxruntime-gpu overwrites it
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y onnxruntime 2>/dev/null || true \
    && pip install --no-cache-dir onnxruntime-gpu==1.20.1

# Verify GPU provider is available at build time
RUN python3 -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('Available providers:', providers)"

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 8765

# Command to run the application
CMD ["python", "src/server.py"]
