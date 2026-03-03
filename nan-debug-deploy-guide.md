# Deployment Guide — Doctor Preview (runpod_v2) on Azure T4 GPU

> **Purpose:** Step-by-step guide to deploy the face-swap service on an Azure GPU VM without repeating the mistakes from previous attempts. Written after a painful multi-hour debugging session.

---

## Table of Contents

1. [What Went Wrong (Lessons Learned)](#1-what-went-wrong-lessons-learned)
2. [Infrastructure Requirements](#2-infrastructure-requirements)
3. [The Correct Dockerfile](#3-the-correct-dockerfile)
4. [Package Version Compatibility Matrix](#4-package-version-compatibility-matrix)
5. [Step-by-Step Deployment](#5-step-by-step-deployment)
6. [Performance Tuning](#6-performance-tuning)
7. [Verification Checklist](#7-verification-checklist)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What Went Wrong (Lessons Learned)

### Problem 1: `cv2` ImportError — missing build tools

**Symptom:** `ImportError: libGL.so.1: cannot open shared object file`  
**Root Cause:** The original `runpod_v2/Dockerfile` uses `nvidia/cuda:...-runtime-...` base image which does NOT include `build-essential`, `g++`, or `libGL`. Many Python packages (insightface, opencv, gfpgan) need C compilation or shared libraries.  
**Fix:** Use the `devel` base image (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`) OR install `build-essential`, `g++`, `python3.11-dev`, `libgl1-mesa-glx`, `libglib2.0-0` explicitly.

### Problem 2: Python version mismatch (pip installs to 3.10, CMD runs 3.11)

**Symptom:** `ModuleNotFoundError` at runtime even though `pip install` succeeded during build.  
**Root Cause:** Using bare `pip install` defaults to the system Python (3.10 on Ubuntu 22.04). But `CMD ["python3", ...]` runs Python 3.11 from the deadsnakes PPA.  
**Fix:** Always use `python3 -m pip install` (never bare `pip`) to ensure packages go into the correct Python's site-packages.

### Problem 3: onnxruntime-gpu 1.20.1 requires cuDNN 9, container has cuDNN 8

**Symptom:** All ONNX models fall back to `CPUExecutionProvider`. Logs show:
```
EP Error: ...: cuDNN 9 handle creation failed
```
**Root Cause:** `onnxruntime-gpu==1.20.1` was built against cuDNN 9. The `cudnn8` Docker images only have cuDNN 8.x.  
**Fix:** Use `onnxruntime-gpu==1.18.1` which is compatible with cuDNN 8. Or switch to a `cudnn9` base image.

> **THIS WAS THE #1 PERFORMANCE KILLER.** It caused the entire face-swap pipeline to run on CPU (~200ms/frame) instead of GPU (~25ms/frame). The server appeared healthy (HTTP 200) but was secretly running everything on CPU.

### Problem 4: GFPGAN broke — `torchvision.transforms.functional_tensor` removed

**Symptom:** `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`  
**Root Cause:** `torchvision >= 0.20` removed the deprecated `functional_tensor` module. GFPGAN (via `basicsr`/`facexlib`) imports it directly.  
**Fix:** Pin `torch==2.4.1+cu121` and `torchvision==0.19.1+cu121`. OR create a shim file:
```python
# /path/to/torchvision/transforms/functional_tensor.py
from torchvision.transforms.functional import *
```

### Problem 5: Docker build OOM (killed during pip install)

**Symptom:** Build fails with `Killed` during `pip install torch` or `pip install insightface`.  
**Root Cause:** The `NC4as_T4_v3` VM has 28GB RAM but only 4 CPUs. Building large packages (torch, ONNX) in parallel can exceed memory.  
**Fix:** Add 4GB swap before building:
```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

### Problem 6: Docker disk full during image export

**Symptom:** `no space left on device` during `docker build` or `docker commit`.  
**Root Cause:** The VM has a 62GB disk. The Docker image is ~14GB. Multiple failed builds accumulate dangling images.  
**Fix:** Clean before building:
```bash
sudo docker system prune -af    # WARNING: removes ALL unused images
sudo docker builder prune -af
```
Always check disk: `df -h /` — need at least 20GB free for a clean build.

### Problem 7: LipSync fails every frame — dimension mismatch

**Symptom:** Logs flood with `LipSync inference failed: INVALID_ARGUMENT: Got invalid dimensions for input: target`  
**Root Cause:** The Wav2Lip ONNX model (`wav2lip_gan_96.onnx`) expects specific input shapes that don't match what the code produces. This was silently caught per-frame, adding exception overhead.  
**Fix:** Disable LipSync until the model/code is fixed: `ENABLE_LIPSYNC=false`.

### Problem 8: Latency regression (300ms → 600-2000ms)

**Symptom:** User sees 600-2000ms latency; server logs show only 85ms processing.  
**Root Cause:** Multiple factors stacking:

| Factor | Cost |
|--------|------|
| `ENABLE_SEAMLESS_CLONE=true` → runs LAB color matching every frame | +20-25ms |
| `ENABLE_LIPSYNC=true` → fails every frame (exception overhead) | +2-3ms |
| 128x128 mask recreated every frame (not cached) | +3ms |
| `frame.copy()` on full 1080p frame per swap | +2-5ms |
| `JPEG_QUALITY=90` at 1080p = ~200KB/frame → network congestion | +100-300ms queuing |
| India → US East Azure network RTT | ~300ms baseline |

**Fix:** All of the above were addressed — see [Performance Tuning](#6-performance-tuning).

---

## 2. Infrastructure Requirements

### Current Working Setup

| Component | Value |
|-----------|-------|
| VM SKU | `Standard_NC4as_T4_v3` |
| GPU | Tesla T4 (16GB VRAM) |
| CPU / RAM | 4 vCPUs / 28GB |
| OS | Ubuntu 22.04 LTS |
| Disk | 62GB (need 20GB+ free) |
| Region | `eastus` |
| Resource Group | `doctor-preview-rg` |
| VM Name | `doctor-preview-vm` |
| Public IP | `20.115.36.199` |
| NSG Ports | 22 (SSH), 8765 (service) |
| NVIDIA Driver | 590.48.01 |
| Swap | 4GB (must be enabled manually) |

### SSH Access
```bash
ssh azureuser@20.115.36.199
```

---

## 3. The Correct Dockerfile

> **DO NOT** use the `runpod_v2/Dockerfile` directly. It has the wrong base image and package versions.  
> Use this proven Dockerfile instead:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies (devel image has build-essential already)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libturbojpeg wget curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip setuptools wheel Cython \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages — CRITICAL: use python3 -m pip, never bare pip
COPY requirements.txt .

# Step 1: Core packages (no heavy torch yet)
RUN python3 -m pip install --no-cache-dir \
    fastapi==0.109.2 uvicorn[standard]==0.27.1 python-multipart==0.0.9 \
    opencv-python-headless==4.9.0.80 numpy==1.26.4 Pillow>=10.0.0 \
    PyTurboJPEG>=1.7.1 scikit-image scikit-learn prettytable tqdm \
    onnx>=1.14.0 albumentations>=1.0 librosa>=0.10.0 soundfile>=0.12.0

# Step 2: PyTorch + torchvision (PINNED for GFPGAN compatibility)
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Step 3: insightface (skip its onnxruntime dep, we install GPU version separately)
RUN python3 -m pip install --no-cache-dir insightface==0.7.3

# Step 4: onnxruntime-gpu — MUST be 1.18.1 for cuDNN 8 compatibility
RUN python3 -m pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir onnxruntime-gpu==1.18.1

# Step 5: Face enhancement
RUN python3 -m pip install --no-cache-dir gfpgan>=1.3.8 basicsr>=1.4.2 realesrgan>=0.3.0

# Verify critical imports
RUN python3 -c "\
import cv2; print('cv2', cv2.__version__); \
import onnxruntime as ort; print('ort', ort.__version__); \
print('providers:', ort.get_available_providers()); \
import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available()); \
import insightface; print('insightface OK'); \
"

COPY src/ ./src/

EXPOSE 8765

CMD ["python3", "src/server.py"]
```

### Key Differences from Original

| Original `runpod_v2/Dockerfile` | Correct Dockerfile |
|---|---|
| `cuda:...-runtime-...` (no compiler) | `cuda:...-devel-...` (has g++, build tools) |
| Bare `pip install` | `python3 -m pip install` |
| `onnxruntime-gpu==1.20.1` (needs cuDNN 9) | `onnxruntime-gpu==1.18.1` (works with cuDNN 8) |
| Latest torch/torchvision (breaks GFPGAN) | Pinned `torch==2.4.1+cu121`, `torchvision==0.19.1+cu121` |
| Single `pip install -r requirements.txt` | Split installs (better error isolation, avoids OOM) |
| No import verification | `RUN python3 -c "import ..."` build-time check |

---

## 4. Package Version Compatibility Matrix

These are the **proven working** versions inside the current container:

| Package | Version | Notes |
|---------|---------|-------|
| `onnxruntime-gpu` | **1.18.1** | cuDNN 8 compatible. DO NOT use 1.20.x (needs cuDNN 9) |
| `torch` | **2.4.1+cu121** | Must match CUDA 12.1 in base image |
| `torchvision` | **0.19.1+cu121** | Last version with `functional_tensor`. DO NOT use 0.20+ |
| `insightface` | **0.7.3** | Pulls onnxruntime CPU — override with GPU version after |
| `opencv-python-headless` | **4.9.0.80** | Headless = no GUI deps needed |
| `numpy` | **1.26.4** | Compatible with both torch and insightface |
| `gfpgan` | **1.3.8** | Needs basicsr, facexlib, realesrgan |
| `fastapi` | **0.109.2** | |
| `uvicorn` | **0.27.1** | |
| `librosa` | **0.11.0** | For LipSync audio processing |
| `PyTurboJPEG` | **2.2.0** | 3-5x faster JPEG encode/decode |

### Version Rules

- **cuDNN 8 base image → onnxruntime-gpu ≤ 1.18.x**
- **cuDNN 9 base image → onnxruntime-gpu ≥ 1.19.x**
- **torchvision < 0.20 → GFPGAN works**
- **torchvision ≥ 0.20 → GFPGAN breaks** (needs `functional_tensor` shim)

---

## 5. Step-by-Step Deployment

### Prerequisites
```bash
# SSH into the VM
ssh azureuser@20.115.36.199

# Ensure swap is enabled (prevents OOM during build)
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Check disk space (need 20GB+ free)
df -h /
# If less than 20GB free:
sudo docker system prune -af && sudo docker builder prune -af
```

### Step 1: Get the code
```bash
cd /opt
sudo rm -rf doctor-preview
sudo git clone -b nandeeswar-debug https://github.com/nandeeswar-neuralhex/doctor-preview.git
cd doctor-preview/runpod_v2
```

### Step 2: Copy the correct Dockerfile
Either use the one from [Section 3](#3-the-correct-dockerfile) or copy from `azure_deployment/`:
```bash
# The azure_deployment Dockerfile is the proven one
sudo cp ../azure_deployment/Dockerfile ./Dockerfile
```
> **IMPORTANT:** If using `azure_deployment/Dockerfile`, you may need to adjust the `COPY` paths and `CMD` to match `runpod_v2/src/` structure.

### Step 3: Build the image
```bash
# Build in background (takes ~15-20 minutes)
sudo nohup docker build --no-cache -t doctor-preview-v2:latest . > /tmp/docker-build.log 2>&1 &

# Monitor progress
tail -f /tmp/docker-build.log
# Or: watch -n5 'tail -20 /tmp/docker-build.log'
```

### Step 4: Verify the build
```bash
# Quick smoke test — should print versions without errors
sudo docker run --rm --gpus all doctor-preview-v2:latest python3 -c "
import cv2; print('cv2:', cv2.__version__)
import onnxruntime as ort; print('ort:', ort.__version__)
print('Providers:', ort.get_available_providers())
import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import insightface; print('insightface: OK')
from gfpgan import GFPGANer; print('GFPGAN: OK')
"
```

**Expected output:**
```
cv2: 4.9.0
ort: 1.18.1
Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
torch: 2.4.1+cu121 CUDA: True
insightface: OK
GFPGAN: OK
```

> **STOP HERE if `CUDAExecutionProvider` is NOT in the providers list.** That means the onnxruntime-gpu / cuDNN mismatch is back. See [Troubleshooting](#8-troubleshooting).

### Step 5: Start the container
```bash
# Remove old container if exists
sudo docker rm -f doctor-preview 2>/dev/null

# Start with optimized settings
sudo docker run -d \
  --name doctor-preview \
  --gpus all \
  --restart unless-stopped \
  -p 8765:8765 \
  -e EXECUTION_PROVIDER=CUDAExecutionProvider \
  -e PORT=8765 \
  -e ENABLE_LIPSYNC=false \
  -e JPEG_QUALITY=80 \
  -e ENABLE_GFPGAN=true \
  -e ENABLE_SEAMLESS_CLONE=false \
  -e ENABLE_TEMPORAL_SMOOTHING=true \
  -e FACE_MASK_BLUR=25 \
  -e FACE_MASK_SCALE=1.1 \
  -e SMOOTHING_ALPHA=0.4 \
  -e MAX_FACES=1 \
  -e TARGET_FPS=30 \
  doctor-preview-v2:latest
```

### Step 6: Wait and verify health
```bash
# Wait for model loading (~15-20 seconds)
for i in $(seq 1 20); do
  STATUS=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/health)
  echo "Health check $i: HTTP $STATUS"
  if [ "$STATUS" = "200" ]; then echo "HEALTHY"; break; fi
  sleep 3
done

# Check GPU status
curl -s http://127.0.0.1:8765/debug/gpu | python3 -m json.tool
```

**Must see:** `"swapper_provider": "CUDAExecutionProvider"` and `"gpu_active": true`

### Step 7: Check startup logs for red flags
```bash
sudo docker logs --tail 50 doctor-preview 2>&1 | grep -E 'Error|FAIL|CPU|CUDA|ready|enabled|disabled'
```

**Good signs:**
```
✅ GPU (CUDA) is available
INSwapper model → CUDAExecutionProvider
GFPGAN enhancer enabled
Cached 128x128 face oval mask
LipSyncer disabled by config.
✅ GPU warmup complete — CUDA context pre-heated
```

**Bad signs (act immediately):**
```
❌ GPU unavailable, falling back to CPU     → onnxruntime/cuDNN mismatch
❌ GFPGAN not available                     → torch/torchvision version issue
❌ LipSync inference failed                 → keep ENABLE_LIPSYNC=false
```

---

## 6. Performance Tuning

### Optimized Environment Variables

| Variable | Recommended | Default | Why |
|----------|-------------|---------|-----|
| `ENABLE_SEAMLESS_CLONE` | **false** | true | `_color_match()` costs ~25ms/frame. Alpha-blend is visually similar. |
| `ENABLE_LIPSYNC` | **false** | true | Wav2Lip model has dimension bug, fails every frame. |
| `JPEG_QUALITY` | **80** | 90 | ~40% smaller frames. Barely visible quality difference. |
| `TARGET_FPS` | **30** | 24 | Higher throughput. |
| `ENABLE_GFPGAN` | **true** | true | Quality enhancement. Only used for target upload, not per-frame. |
| `MAX_FACES` | **1** | 1 | Single face = fastest path. |

### Code-Level Optimizations (applied in-container)

These were patched in the running container. **If you rebuild from scratch, apply them to the source code:**

1. **Cache the 128x128 face mask** in `face_swapper.py` `__init__`:
   ```python
   # Add after self._last_result line in __init__:
   _mask = np.zeros((128, 128), dtype=np.float32)
   cv2.ellipse(_mask, (64, 64), (52, 58), 0, 0, 360, 1.0, -1)
   _mask = cv2.GaussianBlur(_mask, (31, 31), 0)
   self._cached_aimg_mask = (_mask * 255).astype(np.uint8)
   ```
   Then in `_swap_single_face`, replace the mask creation block with:
   ```python
   roi_mask = cv2.warpAffine(self._cached_aimg_mask, M_roi_inv, (roi_w, roi_h))
   ```

2. **Reduce output resolution** in `server.py`:
   ```python
   MAX_OUTPUT_WIDTH = 854  # was 1280
   ```

### Expected Performance After Tuning

| Metric | Before | After |
|--------|--------|-------|
| Server swap time | ~65ms | ~15-25ms |
| Server total per frame | ~85ms | ~30-40ms |
| Output frame size | ~200KB | ~60KB |
| Client-perceived latency | 600-2000ms | 350-500ms |

> **Note:** The ~300ms floor is the India → US East network round-trip. To get below 300ms, deploy in `centralindia` or `southeastasia` Azure region.

---

## 7. Verification Checklist

Run this after every deployment. **ALL must pass:**

```bash
echo "=== 1. Container running ==="
sudo docker ps --filter name=doctor-preview --format '{{.Status}}'
# Expected: Up X minutes

echo "=== 2. Health OK ==="
curl -s http://127.0.0.1:8765/health
# Expected: {"status": "ok", ...}

echo "=== 3. GPU active + CUDA provider ==="
curl -s http://127.0.0.1:8765/debug/gpu | python3 -c "
import sys, json
d = json.load(sys.stdin)
assert d['gpu_active'] == True, 'GPU NOT ACTIVE!'
assert d['swapper_provider'] == 'CUDAExecutionProvider', 'SWAPPER ON CPU!'
assert 'CUDAExecutionProvider' in d['available_providers'], 'NO CUDA PROVIDER!'
print('ALL GPU CHECKS PASSED')
"

echo "=== 4. No CPU fallback in logs ==="
sudo docker logs --tail 100 doctor-preview 2>&1 | grep -c 'CPUExecutionProvider' || echo "0 CPU fallback mentions"
# Expected: Only shows in "Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']" — that's normal.
# BAD: "Falling back to CPU" or "EP Error"

echo "=== 5. No LipSync errors ==="
sudo docker logs --tail 100 doctor-preview 2>&1 | grep -c 'LipSync inference failed' || echo "0 LipSync errors"
# Expected: 0 (because ENABLE_LIPSYNC=false)

echo "=== 6. nvidia-smi shows python process ==="
nvidia-smi | grep python
# Expected: python3 process using GPU memory
```

---

## 8. Troubleshooting

### "All models on CPUExecutionProvider"

```bash
# Check onnxruntime version
sudo docker exec doctor-preview python3 -c "import onnxruntime; print(onnxruntime.__version__)"
# If 1.20.x → WRONG. Downgrade:
sudo docker exec doctor-preview python3 -m pip install onnxruntime-gpu==1.18.1
sudo docker restart doctor-preview
```

### "GFPGAN not available: functional_tensor"

```bash
# Check torchvision version
sudo docker exec doctor-preview python3 -c "import torchvision; print(torchvision.__version__)"
# If 0.20+ → WRONG. Downgrade:
sudo docker exec doctor-preview python3 -m pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
sudo docker restart doctor-preview
```

### "No space left on device"

```bash
df -h /
sudo docker system prune -af
sudo docker builder prune -af
# If still not enough, remove the repo clone:
sudo rm -rf /opt/doctor-preview
# Re-clone after cleanup
```

### "Killed" during docker build (OOM)

```bash
# Enable swap
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
# Retry build
```

### Container starts but immediately exits

```bash
sudo docker logs doctor-preview 2>&1 | head -50
# Common cause: model files not found
# Models must be at /app/models/ inside the container
# Check: sudo docker exec doctor-preview ls -la /app/models/
```

### High swap time (>40ms) even with optimizations

```bash
# Check if ENABLE_SEAMLESS_CLONE crept back to true
sudo docker exec doctor-preview python3 -c "from config import ENABLE_SEAMLESS_CLONE; print('seamless:', ENABLE_SEAMLESS_CLONE)"
# Check GPU utilization
nvidia-smi
# If GPU util is 0% → models are on CPU. See "All models on CPUExecutionProvider" above.
```

### Persisting in-container fixes after rebuild

The current running container has fixes (onnxruntime downgrade, torch downgrade, code patches) that are **NOT in the Docker image**. If the container is recreated, these fixes are lost.

To persist:
```bash
# Option A: docker commit (needs ~15GB free disk)
sudo docker commit doctor-preview doctor-preview-v2:latest

# Option B: Fix the source code and Dockerfile, then rebuild
# (Recommended — update the repo with correct versions)
```

---

## Quick Deploy Command (Copy-Paste)

For a fresh deploy from scratch on the existing VM:

```bash
ssh azureuser@20.115.36.199 << 'DEPLOY'
set -e

# Swap
sudo swapon /swapfile 2>/dev/null || {
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
}

# Cleanup
sudo docker rm -f doctor-preview 2>/dev/null || true
sudo docker system prune -af

# Get code
cd /opt
sudo rm -rf doctor-preview
sudo git clone -b nandeeswar-debug https://github.com/nandeeswar-neuralhex/doctor-preview.git
cd doctor-preview/runpod_v2

# Use the correct Dockerfile (copy from azure_deployment and adjust)
# ... or use the Dockerfile from Section 3 of this guide

# Build
sudo docker build --no-cache -t doctor-preview-v2:latest .

# Run
sudo docker run -d --name doctor-preview --gpus all --restart unless-stopped \
  -p 8765:8765 \
  -e EXECUTION_PROVIDER=CUDAExecutionProvider \
  -e PORT=8765 \
  -e ENABLE_LIPSYNC=false \
  -e JPEG_QUALITY=80 \
  -e ENABLE_GFPGAN=true \
  -e ENABLE_SEAMLESS_CLONE=false \
  -e ENABLE_TEMPORAL_SMOOTHING=true \
  -e FACE_MASK_BLUR=25 \
  -e MAX_FACES=1 \
  -e TARGET_FPS=30 \
  doctor-preview-v2:latest

# Wait for ready
sleep 20
curl -s http://127.0.0.1:8765/debug/gpu | python3 -m json.tool
DEPLOY
```

---

*Last updated: March 2026 — after the deployment debugging session.*
