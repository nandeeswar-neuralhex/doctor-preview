#!/usr/bin/env bash
set -euo pipefail

LEGACY_HOST="azureuser@20.115.36.199"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)

ssh "${SSH_OPTS[@]}" "$LEGACY_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

BASE_IMAGE="doctor-preview-v2:debug-parity-2026-04-01"
NEW_IMAGE="doctor-preview-v2:debug-parity-webrtc-latency-2026-04-01"
ROLLBACK_IMAGE="doctor-preview-v2:legacy-before-webrtc-latency-2026-04-01"
CONTAINER="doctor-preview"
LD_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/cuda-12.1/lib64"

if ! sudo docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Missing base image: $BASE_IMAGE" >&2
  exit 1
fi

current_image=""
if sudo docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  current_image=$(sudo docker inspect "$CONTAINER" --format '{{.Config.Image}}')
fi

sudo docker rm -f doctor-preview-patch >/dev/null 2>&1 || true
sudo docker create --name doctor-preview-patch --entrypoint /usr/bin/python3 "$BASE_IMAGE" -c 'import time; time.sleep(3600)' >/dev/null
sudo docker start doctor-preview-patch >/dev/null

sudo docker exec -i doctor-preview-patch python3 - <<'PY'
from pathlib import Path

path = Path('/app/src/webrtc.py')
text = path.read_text()

if 'import json\n' not in text:
    text = text.replace('import asyncio\n', 'import asyncio\nimport json\n', 1)

block = """
        @pc.on(\"datachannel\")
        def on_datachannel(channel):
            \"\"\"Echo ping messages back as pong for client-side latency measurement.\"\"\"
            print(f\"[WebRTC:{session_id}] Data channel '{channel.label}' opened\")

            @channel.on(\"message\")
            def on_message(message):
                try:
                    msg = json.loads(message)
                    if msg.get(\"type\") == \"ping\":
                        channel.send(json.dumps({\"type\": \"pong\", \"ts\": msg[\"ts\"]}))
                except Exception:
                    pass

"""
marker = '        @pc.on("connectionstatechange")'
if '@pc.on("datachannel")' not in text:
    text = text.replace(marker, block + marker, 1)

path.write_text(text)
PY

sudo docker commit doctor-preview-patch "$NEW_IMAGE" >/dev/null
sudo docker rm -f doctor-preview-patch >/dev/null

if [ -n "$current_image" ]; then
  sudo docker image tag "$current_image" "$ROLLBACK_IMAGE" >/dev/null 2>&1 || true
  sudo docker rm -f "$CONTAINER" >/dev/null
fi

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
  -e LD_LIBRARY_PATH="$LD_PATH" \
  "$NEW_IMAGE" \
  python3 -m uvicorn server:app --host 0.0.0.0 --port 8765 --workers 1 >/dev/null

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
sudo docker inspect "$CONTAINER" --format 'IMAGE={{.Config.Image}} PATH={{.Path}} ARGS={{json .Args}}'
echo "--- env"
sudo docker exec "$CONTAINER" env | grep -E '^(ENABLE_WEBRTC|ENABLE_LIPSYNC|EXECUTION_PROVIDER|LD_LIBRARY_PATH)=' | sort
echo "--- latency hook"
sudo docker exec "$CONTAINER" sh -lc 'grep -n -E "import json|@pc.on\(""datachannel""\)|pong" /app/src/webrtc.py'
REMOTE
