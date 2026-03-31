#!/bin/bash

set -euo pipefail

IMAGE="doctor-preview-v2"
IMAGE_TAG="t4-debug-backup-2026-04-01"
CONTAINER="doctor-preview"
REPO="https://github.com/nandeeswar-neuralhex/doctor-preview.git"
CODE_DIR="/opt/doctor-preview"
BACKUP_BRANCH="t4-debug-backup-2026-04-01"
BACKUP_COMMIT="8396272884e2bf70079050665eda2e64634d76c3"
SERVER_PORT="8765"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()     { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo ""
echo "==============================================================="
echo "  T4 Debug Backup Restore"
echo "==============================================================="
echo ""

info "Step 1/5  Fetching backup branch and commit..."
if [ -d "$CODE_DIR/.git" ]; then
    cd "$CODE_DIR"
    sudo git fetch origin "$BACKUP_BRANCH"
else
    sudo rm -rf "$CODE_DIR"
    sudo git clone "$REPO" "$CODE_DIR"
    cd "$CODE_DIR"
fi

sudo git checkout "$BACKUP_COMMIT"
CURRENT_COMMIT=$(sudo git rev-parse HEAD)
if [ "$CURRENT_COMMIT" != "$BACKUP_COMMIT" ]; then
    err "Checked out $CURRENT_COMMIT instead of $BACKUP_COMMIT"
fi
success "Pinned source at $CURRENT_COMMIT"

info "Step 2/5  Building exact Debug backup image..."
cd "$CODE_DIR/azure_deployment"
sudo docker build --no-cache -t "${IMAGE}:${IMAGE_TAG}" .
sudo docker tag "${IMAGE}:${IMAGE_TAG}" "${IMAGE}:latest"
success "Image built as ${IMAGE}:${IMAGE_TAG}"

info "Step 3/5  Replacing container with Debug runtime settings..."
sudo docker rm -f "$CONTAINER" 2>/dev/null || true
sudo docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --restart unless-stopped \
  -p "${SERVER_PORT}:${SERVER_PORT}" \
  -e EXECUTION_PROVIDER=CUDAExecutionProvider \
  -e PORT="${SERVER_PORT}" \
  -e ENABLE_WEBRTC=true \
  -e ENABLE_LIPSYNC=true \
  -e JPEG_QUALITY=90 \
  -e ENABLE_GFPGAN=true \
  -e ENABLE_SEAMLESS_CLONE=true \
  -e ENABLE_TEMPORAL_SMOOTHING=true \
  -e FACE_MASK_BLUR=25 \
  -e FACE_MASK_SCALE=1.1 \
  -e SMOOTHING_ALPHA=0.4 \
  -e MAX_FACES=1 \
  -e TARGET_FPS=24 \
  "${IMAGE}:${IMAGE_TAG}"
success "Container started"

info "Step 4/5  Waiting for health endpoint..."
for i in $(seq 1 40); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${SERVER_PORT}/health" 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
        success "Container healthy after $((i * 5))s"
        break
    fi
    echo -ne "\r  waiting: $((i * 5))s — HTTP $CODE   "
    sleep 5
done

if [ "$CODE" != "200" ]; then
    err "Health check failed"
fi

info "Step 5/5  Printing deployed identity..."
sudo docker inspect "$CONTAINER" --format='IMAGE={{.Image}} ENV={{json .Config.Env}}'
success "Debug backup restore complete"
