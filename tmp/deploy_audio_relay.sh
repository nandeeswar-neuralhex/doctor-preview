#!/usr/bin/env bash
set -euo pipefail

# Deploy AudioRelayTrack to T4 legacy server.
# This patches webrtc.py to relay incoming audio back to the client
# via WebRTC so the browser keeps audio and video in sync automatically.

LEGACY_HOST="azureuser@20.115.36.199"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)

ssh "${SSH_OPTS[@]}" "$LEGACY_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail

BASE_IMAGE="doctor-preview-v2:debug-parity-webrtc-latency-2026-04-01"
NEW_IMAGE="doctor-preview-v2:audio-relay-2026-04-03"
ROLLBACK_IMAGE="doctor-preview-v2:before-audio-relay-2026-04-03"
CONTAINER="doctor-preview"
LD_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/cuda-12.1/lib64"

if ! sudo docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Missing base image: $BASE_IMAGE" >&2
  echo "Available images:"
  sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep doctor-preview | sort
  exit 1
fi

current_image=""
if sudo docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  current_image=$(sudo docker inspect "$CONTAINER" --format '{{.Config.Image}}')
  echo "Current running image: $current_image"
fi

sudo docker rm -f doctor-preview-patch >/dev/null 2>&1 || true
sudo docker create --name doctor-preview-patch --entrypoint /usr/bin/python3 "$BASE_IMAGE" -c 'import time; time.sleep(3600)' >/dev/null
sudo docker start doctor-preview-patch >/dev/null

# --- Patch webrtc.py: Add AudioRelayTrack and update on_track handler ---
sudo docker exec -i doctor-preview-patch python3 - <<'PY'
from pathlib import Path

path = Path('/app/src/webrtc.py')
text = path.read_text()

# ── Step 1: Add AudioRelayTrack class after AudioBuffer ──
audio_relay_class = '''

class AudioRelayTrack(MediaStreamTrack):
    """Relay incoming audio back to the client unchanged.

    This keeps the audio in the same WebRTC session as the processed video,
    so the browser's built-in RTCP sync mechanism keeps them aligned.
    Also feeds each frame into the AudioBuffer for lip-sync processing.
    """
    kind = "audio"

    def __init__(self, track: MediaStreamTrack, audio_buffer: AudioBuffer):
        super().__init__()
        self.track = track
        self.audio_buffer = audio_buffer

    async def recv(self) -> AudioFrame:
        frame = await self.track.recv()
        # Feed into buffer for lip sync (non-blocking, same as before)
        self.audio_buffer.append(frame)
        # Return the exact same frame — zero processing, zero delay
        return frame

    def stop(self):
        super().stop()

'''

marker_class = 'class VideoTransformTrack(MediaStreamTrack):'
if 'class AudioRelayTrack' not in text:
    text = text.replace(marker_class, audio_relay_class + marker_class, 1)
    print("Added AudioRelayTrack class")
else:
    print("AudioRelayTrack class already exists")

# ── Step 2: Replace the on_track audio handler ──
# Check if on_track still uses the old recv_audio pattern.
# Use the exact handler code to find+replace.
if 'pc.addTrack(local_audio)' in text:
    print("on_track already uses AudioRelayTrack relay")
elif 'recv_audio' in text:
    # Replace the old fire-and-forget recv_audio handler
    old_handler = '''            if track.kind == "audio":
                async def recv_audio():
                    try:
                        while True:
                            frame = await track.recv()
                            audio_buffer.append(frame)
                    except Exception:
                        pass
                asyncio.ensure_future(recv_audio())'''

    new_handler = '''            if track.kind == "audio":
                # Relay audio back to the client via WebRTC so the browser
                # keeps it in sync with the processed video automatically.
                # The relay track also feeds the AudioBuffer for lip sync.
                local_audio = AudioRelayTrack(
                    self.relay.subscribe(track),
                    audio_buffer,
                )
                pc.addTrack(local_audio)'''

    if old_handler in text:
        text = text.replace(old_handler, new_handler, 1)
        print("Replaced on_track audio handler with AudioRelayTrack")
    else:
        # Try to find it with different whitespace
        import re
        pattern = r'(            if track\.kind == "audio":\n).*?(?=            elif track\.kind == "video":)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = text[:match.start()] + new_handler + '\n' + text[match.end():]
            print("Replaced on_track audio handler (regex fallback)")
        else:
            print("ERROR: Could not find on_track audio handler to replace")
            import sys; sys.exit(1)
else:
    print("ERROR: Neither recv_audio nor AudioRelayTrack found in on_track")
    import sys; sys.exit(1)

path.write_text(text)
print("webrtc.py patched successfully")

# Verify the patch
text2 = path.read_text()
assert 'class AudioRelayTrack' in text2, "AudioRelayTrack class missing after patch"
assert 'pc.addTrack(local_audio)' in text2, "pc.addTrack(local_audio) missing after patch"
assert 'recv_audio' not in text2, "Old recv_audio pattern still present"
print("All assertions passed")
PY

sudo docker commit doctor-preview-patch "$NEW_IMAGE" >/dev/null
sudo docker rm -f doctor-preview-patch >/dev/null
echo "Committed new image: $NEW_IMAGE"

if [ -n "$current_image" ]; then
  sudo docker image tag "$current_image" "$ROLLBACK_IMAGE" >/dev/null 2>&1 || true
  echo "Rollback image: $ROLLBACK_IMAGE"
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

echo "Container started, waiting for health check..."
for i in $(seq 1 48); do
  if curl -fsS http://127.0.0.1:8765/health >/dev/null 2>&1; then
    echo "HEALTHY after $((i * 5))s"
    break
  fi
  sleep 5
  if [ "$i" -eq 48 ]; then
    echo "Container failed health check after 240s" >&2
    sudo docker logs "$CONTAINER" 2>&1 | tail -120 >&2
    exit 1
  fi
done

echo ""
echo "=== Verification ==="
echo "--- image"
sudo docker inspect "$CONTAINER" --format 'IMAGE={{.Config.Image}} PATH={{.Path}} ARGS={{json .Args}}'
echo "--- env"
sudo docker exec "$CONTAINER" env | grep -E '^(ENABLE_WEBRTC|ENABLE_LIPSYNC|EXECUTION_PROVIDER)=' | sort
echo "--- AudioRelayTrack check"
sudo docker exec "$CONTAINER" grep -n 'class AudioRelayTrack\|pc.addTrack(local_audio)' /app/src/webrtc.py
echo "--- No old recv_audio pattern"
if sudo docker exec "$CONTAINER" grep -q 'recv_audio' /app/src/webrtc.py; then
  echo "WARNING: Old recv_audio pattern still found!"
  exit 1
else
  echo "OK: recv_audio removed"
fi
REMOTE
