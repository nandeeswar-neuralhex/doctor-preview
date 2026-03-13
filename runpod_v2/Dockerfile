# ── CUDA 12.1 + cuDNN 8 DEVEL image (has build tools for insightface/cv2) ──
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies (devel image has build-essential already)
# Added WebRTC codec libs: libavcodec, libopus, libvpx, libsrtp2
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libturbojpeg \
    libavcodec-dev libavformat-dev libavutil-dev \
    libopus-dev libvpx-dev libsrtp2-dev \
    wget curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip setuptools wheel Cython \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages — CRITICAL: use python3 -m pip, never bare pip
COPY requirements.txt .

# Step 1: Core packages
RUN python3 -m pip install --no-cache-dir \
    fastapi==0.109.2 uvicorn[standard]==0.27.1 python-multipart==0.0.9 \
    opencv-python-headless==4.9.0.80 numpy==1.26.4 Pillow>=10.0.0 \
    PyTurboJPEG>=1.7.1 scikit-image scikit-learn prettytable tqdm \
    onnx>=1.14.0 albumentations>=1.0 librosa>=0.10.0 soundfile>=0.12.0

# Step 2: PyTorch + torchvision (PINNED for GFPGAN compatibility)
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Step 3: insightface
RUN python3 -m pip install --no-cache-dir insightface==0.7.3

# Step 4: onnxruntime-gpu — MUST be 1.18.1 for cuDNN 8 compatibility
# Install here first, then reinstall after GFPGAN (which pulls CPU onnxruntime)
RUN python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir onnxruntime-gpu==1.18.1

# Step 5: Face enhancement
RUN python3 -m pip install --no-cache-dir gfpgan>=1.3.8 basicsr>=1.4.2 realesrgan>=0.3.0

# Fix GFPGAN: torchvision >= 0.19 removed functional_tensor, create shim
RUN python3 -c "import torchvision, os; p=os.path.join(os.path.dirname(torchvision.__file__), 'transforms', 'functional_tensor.py'); open(p,'w').write('from torchvision.transforms.functional import *\n')"

# Step 6: WebRTC (aiortc + av)
RUN python3 -m pip install --no-cache-dir aiortc>=1.7.0 av>=11.0.0

# Step 7: CRITICAL — reinstall onnxruntime-gpu AFTER gfpgan (which pulls CPU onnxruntime)
RUN python3 -m pip uninstall -y onnxruntime 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.18.1

# Verify critical imports
RUN python3 -c "\
import cv2; print('cv2', cv2.__version__); \
import onnxruntime as ort; print('ort', ort.__version__); \
print('providers:', ort.get_available_providers()); \
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); \
import insightface; print('insightface OK'); \
import aiortc; print('aiortc OK'); \
"

# Copy source code
COPY src/ ./src/

# Expose port (HTTP signaling + WebRTC UDP)
EXPOSE 8765
EXPOSE 40000-60000/udp

# Command to run the application
CMD ["python3", "src/server.py"]
