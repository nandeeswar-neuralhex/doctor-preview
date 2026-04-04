#!/usr/bin/env bash
set -euo pipefail

LEGACY_HOST="azureuser@20.115.36.199"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)

LOCAL_SRC_DIR="/Users/nandeeswar/Desktop/FaceIQ/Doctor-preview-main/azure_deployment/src"
REMOTE_PATCH_DIR="/tmp/doctor-preview-audio-fix-$$"

echo "Uploading validated server files..."
ssh "${SSH_OPTS[@]}" "$LEGACY_HOST" "rm -rf '$REMOTE_PATCH_DIR' && mkdir -p '$REMOTE_PATCH_DIR'"
scp "${SSH_OPTS[@]}" \
  "$LOCAL_SRC_DIR/webrtc.py" \
  "$LOCAL_SRC_DIR/config.py" \
  "$LEGACY_HOST:$REMOTE_PATCH_DIR/"

ssh "${SSH_OPTS[@]}" "$LEGACY_HOST" "PATCH_DIR='$REMOTE_PATCH_DIR' bash -s" <<'REMOTE'
set -euo pipefail

CONTAINER="doctor-preview"
PATCH_CONTAINER="doctor-preview-patch"
NEW_IMAGE="doctor-preview-v2:audio-relay-fix-2026-04-03"
ROLLBACK_IMAGE="doctor-preview-v2:before-audio-relay-fix-2026-04-03"
LD_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/cuda-12.1/lib64"

current_image=""
if sudo docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  current_image=$(sudo docker inspect "$CONTAINER" --format '{{.Config.Image}}')
fi

if [ -z "$current_image" ]; then
  echo "Could not determine current running image" >&2
  exit 1
fi

echo "Current image: $current_image"

sudo docker rm -f "$PATCH_CONTAINER" >/dev/null 2>&1 || true
sudo docker create --name "$PATCH_CONTAINER" --entrypoint /usr/bin/python3 "$current_image" -c 'import time; time.sleep(3600)' >/dev/null
sudo docker start "$PATCH_CONTAINER" >/dev/null

sudo docker cp "$PATCH_DIR/webrtc.py" "$PATCH_CONTAINER:/app/src/webrtc.py"
sudo docker cp "$PATCH_DIR/config.py" "$PATCH_CONTAINER:/app/src/config.py"

echo "Validating patched files inside container..."
sudo docker exec "$PATCH_CONTAINER" sh -lc 'cd /app/src && python3 -m py_compile webrtc.py config.py'

sudo docker commit "$PATCH_CONTAINER" "$NEW_IMAGE" >/dev/null
sudo docker rm -f "$PATCH_CONTAINER" >/dev/null

sudo docker image tag "$current_image" "$ROLLBACK_IMAGE" >/dev/null 2>&1 || true
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

echo "Starting new container: $NEW_IMAGE"
sudo docker run -d \
  --name "$CONTAINER" \
  --gpus all \
  --restart unless-stopped \
  -p 8765:8765 \
  -w /app/src \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh \
  -e EXECUTION_PROVIDER=CUDAExecutionProvider \
  -e ENABLE_WEBRTC=true \
  -e ENABLE_LIPSYNC=true \
  -e WEBRTC_SYNC_MIN_DELAY_MS=150 \
  -e WEBRTC_SYNC_MAX_DELAY_MS=350 \
  -e WEBRTC_SYNC_SAFETY_MARGIN_MS=40 \
  -e LD_LIBRARY_PATH="$LD_PATH" \
  "$NEW_IMAGE" \
  python3 -m uvicorn server:app --host 0.0.0.0 --port 8765 --workers 1 >/dev/null

echo "Waiting for health check..."
for i in $(seq 1 48); do
  if curl -fsS http://127.0.0.1:8765/health >/dev/null 2>&1; then
    echo "healthy:$((i * 5))s"
    break
  fi
  sleep 5
  if [ "$i" -eq 48 ]; then
    echo "Container failed health check" >&2
    sudo docker logs "$CONTAINER" 2>&1 | tail -120 >&2
    exit 1
  fi
done

echo "--- image"
sudo docker inspect "$CONTAINER" --format 'IMAGE={{.Config.Image}}'
echo "--- audio relay fix verification"
sudo docker exec "$CONTAINER" grep -n 'relay.subscribe\|AudioRelayTrack\|on_track fired\|AudioRelay.*recv' /app/src/webrtc.py | head -15

rm -rf "$PATCH_DIR"
REMOTE
