#!/bin/bash
# ===================================================================
# H100 VM (20.9.36.27) — WebRTC container restart script
# Domain: h100-faceiq.sparkiq.ai
# VM: doctor-preview-h100 — Standard_NC40ads_H100_v5
#       40 cores, 314 GB RAM, H100 NVL 94 GB VRAM
#
# Run this on the VM to (re)start the container with the
# production-grade config established on April 6, 2026:
#
#   ssh azureuser@20.9.36.27
#   bash deploy_h100_webrtc.sh
#
# ── Config rationale ────────────────────────────────────────────────
# ENABLE_GFPGAN=true         Post-swap GFPGAN super-resolution pass.
#                             Sharpens skin texture, removes ONNX
#                             model artifacts. +538 MB VRAM, +7% GPU
#                             at 30fps — zero FPS cost on H100.
# ENABLE_FACE_PARSING=true   BiSeNet segmentation mask instead of
#                             ellipse. Cleaner face boundary blending.
# DETECTION_SIZE=640         Full 640×640 analyzer (high-end GPU).
# ENABLE_TEMPORAL_SMOOTHING  Prevents face jitter between frames.
# ENABLE_LIPSYNC=false       Disabled — was throwing ONNX dim errors.
# OMP/MKL/OPENBLAS=4         Prevents thread-pool contention.
#                             Threads: 290 (was 437 uncapped).
# ===================================================================

set -e

IMAGE="doctor-preview-h100:latest"
CONTAINER="doctor-preview"
SERVER_PORT=8765

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()     { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo ""
echo "==============================================================="
echo "  H100 VM WebRTC Deploy — h100-faceiq.sparkiq.ai"
echo "==============================================================="
echo ""

# ── GPU persistence ─────────────────────────────────────────────────
info "Enabling GPU persistence mode..."
sudo nvidia-smi -pm 1 > /dev/null 2>&1 && success "GPU persistence enabled" || warn "Could not set persistence mode"

# ── Stop existing container ──────────────────────────────────────────
info "Stopping existing container (if any)..."
sudo docker rm -f "$CONTAINER" 2>/dev/null || true
success "Old container cleared"

# ── Start container ──────────────────────────────────────────────────
info "Starting $CONTAINER with GFPGAN + FaceParsing enabled..."

sudo docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --restart unless-stopped \
  -p "${SERVER_PORT}:${SERVER_PORT}" \
  -p 40000-40100:40000-40100/udp \
  -e EXECUTION_PROVIDER=CUDAExecutionProvider \
  -e PORT="${SERVER_PORT}" \
  -e HOST=0.0.0.0 \
  -e ENABLE_WEBRTC=true \
  -e ENABLE_LIPSYNC=false \
  -e JPEG_QUALITY=80 \
  -e ENABLE_GFPGAN=true \
  -e ENABLE_FACE_PARSING=true \
  -e ENABLE_SEAMLESS_CLONE=false \
  -e ENABLE_TEMPORAL_SMOOTHING=true \
  -e ENABLE_AV_SYNC_PIPELINE=false \
  -e DETECTION_SIZE=640 \
  -e FACE_MASK_BLUR=25 \
  -e FACE_MASK_SCALE=1.1 \
  -e SMOOTHING_ALPHA=0.3 \
  -e MAX_FACES=1 \
  -e TARGET_FPS=30 \
  -e SWAP_ENGINE=inswapper \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  -e OPENBLAS_NUM_THREADS=4 \
  -e NUMEXPR_NUM_THREADS=4 \
  "$IMAGE"

# ── Health check ────────────────────────────────────────────────────
info "Waiting for server health (model load takes ~20-30 s)..."
for i in $(seq 1 40); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${SERVER_PORT}/health 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
        echo ""
        RESP=$(curl -s http://127.0.0.1:${SERVER_PORT}/health)
        success "Container healthy after $((i*5))s"
        echo "  $RESP"
        break
    fi
    echo -ne "\r  waiting $((i*5))s — HTTP $CODE   "
    sleep 5
done

[ "$CODE" != "200" ] && err "Server did not come healthy — check: sudo docker logs $CONTAINER"

echo ""
success "H100 is live. Test at: https://h100-faceiq.sparkiq.ai"
echo ""
