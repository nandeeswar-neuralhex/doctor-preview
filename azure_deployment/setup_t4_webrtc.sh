#!/bin/bash
# ===================================================================
# T4 VM — Enable WebRTC + HTTPS (Cloudflare tunnel)
# Run this ON the VM:  ssh azureuser@20.115.36.199
# ===================================================================
# This script:
#   1. Cleans old container
#   2. Rebuilds Docker image with WebRTC enabled (onnxruntime 1.18.1 + aiortc)
#   3. Starts the container with ENABLE_WEBRTC=true
#   4. Opens UDP 49152–65535 in the Azure NSG (needed for WebRTC media)
#   5. Installs + configures cloudflared tunnel for HTTPS (same as H100 setup)
# ===================================================================

set -e

# ── Config ──────────────────────────────────────────────────────────
RESOURCE_GROUP="doctor-preview-rg"
VM_NAME="doctor-preview-vm"
NSG_NAME="doctor-preview-vm-nsg"     # Azure NIC-level NSG name
SUBSCRIPTION_ID="60fb43e3-960f-44d7-aad5-ec31a2c6d27c"
IMAGE_NAME="doctor-preview-v2"
CONTAINER_NAME="doctor-preview"
SERVER_PORT=8765
CODE_BRANCH="nandeeswar-webrtc"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "=================================================================="
echo "  T4 VM (20.115.36.199) — WebRTC + HTTPS setup"
echo "=================================================================="
echo ""

# ── Step 0: Ensure swap is on (prevents OOM during build) ───────────
info "Checking swap..."
if ! swapon --show | grep -q /swapfile 2>/dev/null; then
    warn "No swap found — creating 4GB swapfile to prevent OOM during build"
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
success "Swap: $(free -h | awk '/Swap/{print $2}')"

# ── Step 1: Check disk space ─────────────────────────────────────────
info "Disk space:"
df -h / | tail -1
AVAIL_GB=$(df / | awk 'NR==2{print int($4/1024/1024)}')
if [ "$AVAIL_GB" -lt 20 ]; then
    warn "Low disk space (${AVAIL_GB}GB free). Pruning Docker..."
    sudo docker system prune -af
    sudo docker builder prune -af
fi

# ── Step 2: Pull latest code (webrtc branch) ─────────────────────────
info "Pulling latest code (branch: $CODE_BRANCH)..."
if [ -d /opt/doctor-preview ]; then
    cd /opt/doctor-preview
    sudo git fetch origin
    sudo git checkout "$CODE_BRANCH"
    sudo git pull origin "$CODE_BRANCH"
else
    cd /opt
    sudo git clone -b "$CODE_BRANCH" https://github.com/nandeeswar-neuralhex/doctor-preview.git
fi
success "Code updated"

# ── Step 3: Build image with WebRTC ──────────────────────────────────
info "Building Docker image with WebRTC (this takes ~15-20 min)..."
cd /opt/doctor-preview/azure_deployment
sudo docker system prune -f   # remove dangling layers only
sudo nohup docker build --no-cache -t "${IMAGE_NAME}:latest" . \
    > /tmp/docker-build.log 2>&1 &
BUILD_PID=$!
info "Build running as PID $BUILD_PID — tailing logs..."
tail -f /tmp/docker-build.log &
TAIL_PID=$!
wait $BUILD_PID
kill $TAIL_PID 2>/dev/null || true
echo ""

if ! sudo docker image inspect "${IMAGE_NAME}:latest" &>/dev/null; then
    error "Build failed! Check /tmp/docker-build.log"
fi
success "Image built: ${IMAGE_NAME}:latest"

# ── Step 4: Smoke test the image (GPU + WebRTC imports) ──────────────
info "Verifying GPU + WebRTC imports inside image..."
sudo docker run --rm --gpus all "${IMAGE_NAME}:latest" python3 -c "
import cv2; print('cv2:', cv2.__version__)
import onnxruntime as ort; print('ort:', ort.__version__)
print('Providers:', ort.get_available_providers())
import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
import aiortc; print('aiortc:', aiortc.__version__)
from av import VideoFrame; print('av: OK')
" || error "Smoke test failed — check image build"
success "GPU + WebRTC imports OK"

# ── Step 5: Start container with ENABLE_WEBRTC=true ──────────────────
info "Starting container with WebRTC enabled..."
sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --restart unless-stopped \
  -p "${SERVER_PORT}:${SERVER_PORT}" \
  -e EXECUTION_PROVIDER=CUDAExecutionProvider \
  -e PORT="${SERVER_PORT}" \
  -e ENABLE_WEBRTC=true \
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
  "${IMAGE_NAME}:latest"

info "Waiting for server to be healthy (models load in ~20s)..."
for i in $(seq 1 30); do
    STATUS=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${SERVER_PORT}/health 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo ""
        success "Server healthy after ${i}0s"
        break
    fi
    echo -ne "\r  ⏳ ${i}0s — HTTP $STATUS   "
    sleep 10
done

# Verify WebRTC is enabled
WEBRTC_ENABLED=$(curl -s http://127.0.0.1:${SERVER_PORT}/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('webrtc_enabled','?'))" 2>/dev/null)
if [ "$WEBRTC_ENABLED" = "True" ]; then
    success "WebRTC confirmed enabled on server"
else
    warn "webrtc_enabled=$WEBRTC_ENABLED — check 'sudo docker logs $CONTAINER_NAME'"
fi

# ── Step 6: Open Azure NSG UDP ports for WebRTC media ────────────────
# WebRTC signaling goes through HTTP (proxied by cloudflared tunnel).
# WebRTC MEDIA (DTLS/SRTP) travels DIRECTLY between browser and VM
# over ephemeral UDP ports. Azure NSG must allow this inbound traffic.
info "Opening UDP 49152-65535 in Azure NSG for WebRTC media streams..."
if command -v az &>/dev/null && az account show &>/dev/null 2>&1; then
    az account set --subscription "$SUBSCRIPTION_ID" 2>/dev/null || true
    # Try NIC NSG first, then subnet NSG
    for NSG in "$NSG_NAME" "${VM_NAME}NSG" "${VM_NAME}-nsg"; do
        if az network nsg show -g "$RESOURCE_GROUP" -n "$NSG" &>/dev/null 2>&1; then
            az network nsg rule create \
                --resource-group "$RESOURCE_GROUP" \
                --nsg-name "$NSG" \
                --name "WebRTC-UDP-Media" \
                --priority 1100 \
                --protocol Udp \
                --direction Inbound \
                --source-address-prefixes '*' \
                --source-port-ranges '*' \
                --destination-port-ranges '49152-65535' \
                --access Allow \
                --description "WebRTC media streams (DTLS/SRTP over UDP)" 2>/dev/null \
            && success "NSG rule added to $NSG" && break \
            || warn "Rule may already exist in $NSG — continuing"
            break
        fi
    done
else
    warn "Azure CLI not available on this machine or not logged in."
    warn "Run this MANUALLY from your laptop to open UDP ports for WebRTC media:"
    echo ""
    echo "  az network nsg rule create \\"
    echo "    --resource-group $RESOURCE_GROUP \\"
    echo "    --nsg-name <YOUR_NSG_NAME> \\"
    echo "    --name WebRTC-UDP-Media \\"
    echo "    --priority 1100 \\"
    echo "    --protocol Udp \\"
    echo "    --direction Inbound \\"
    echo "    --destination-port-ranges 49152-65535 \\"
    echo "    --access Allow"
    echo ""
fi

# ── Step 7: Set up Cloudflare tunnel for HTTPS ───────────────────────
# The web app (https://faceiq.sparkiq.ai frontend) requires HTTPS on the
# backend — browsers block getUserMedia and mixed-content fetch on plain HTTP.
# Cloudflare tunnel is zero-config HTTPS, same as the H100 VM setup.
echo ""
echo "=================================================================="
echo "  STEP 7: Cloudflare Tunnel (HTTPS for web browser WebRTC)"
echo "=================================================================="
echo ""
echo "  The H100 VM uses cloudflared → https://faceiq.sparkiq.ai"
echo "  We need the same for this T4 VM."
echo ""

if ! command -v cloudflared &>/dev/null; then
    info "Installing cloudflared..."
    curl -L --output /tmp/cloudflared.deb \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i /tmp/cloudflared.deb
    rm -f /tmp/cloudflared.deb
    success "cloudflared installed: $(cloudflared --version)"
else
    success "cloudflared already installed: $(cloudflared --version)"
fi

# Check if tunnel is already configured as a service
if systemctl is-active --quiet cloudflared 2>/dev/null; then
    success "cloudflared service is already running"
    TUNNEL_URL=$(sudo journalctl -u cloudflared -n 50 --no-pager 2>/dev/null \
        | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || echo "")
    if [ -n "$TUNNEL_URL" ]; then
        success "Tunnel URL: $TUNNEL_URL"
    fi
else
    echo ""
    echo "  Choose one of the following HTTPS options:"
    echo ""
    echo "  ── Option A (Quick, temp URL, no account needed) ──────────"
    echo "  Run in a tmux session after this script:"
    echo ""
    echo "    tmux new-session -d -s tunnel 'cloudflared tunnel --url http://localhost:${SERVER_PORT}'"
    echo "    tmux attach -t tunnel   # to see the https://xxx.trycloudflare.com URL"
    echo ""
    echo "  ── Option B (Permanent subdomain, requires Cloudflare account) ──"
    echo "  If you have a Cloudflare account with sparkiq.ai:"
    echo ""
    echo "    1. cloudflared tunnel login"
    echo "    2. cloudflared tunnel create doctor-t4"
    echo "    3. Add CNAME in Cloudflare DNS:  t4.faceiq.sparkiq.ai → <tunnel-id>.cfargotunnel.com"
    echo "    4. cloudflared tunnel route dns doctor-t4 t4.faceiq.sparkiq.ai"
    echo "    5. Create /etc/cloudflared/config.yml:"
    echo "         tunnel: <tunnel-id>"
    echo "         credentials-file: /root/.cloudflared/<tunnel-id>.json"
    echo "         ingress:"
    echo "           - hostname: t4.faceiq.sparkiq.ai"
    echo "             service: http://localhost:${SERVER_PORT}"
    echo "           - service: http_status:404"
    echo "    6. sudo cloudflared service install"
    echo "    7. sudo systemctl start cloudflared"
    echo ""
    echo "  Then set VITE_SERVER_URL=https://t4.faceiq.sparkiq.ai in desktop_app/.env"
    echo "  and rebuild the web app: cd desktop_app && npm run build"
    echo ""

    read -r -p "Start a QUICK TEMP tunnel now? (y/N): " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        info "Starting quick tunnel in tmux..."
        command -v tmux &>/dev/null || sudo apt-get install -y tmux
        tmux new-session -d -s tunnel "cloudflared tunnel --url http://localhost:${SERVER_PORT} 2>&1 | tee /tmp/cloudflare-tunnel.log"
        sleep 5
        TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflare-tunnel.log 2>/dev/null | tail -1 || echo "")
        if [ -n "$TUNNEL_URL" ]; then
            success "Tunnel live at: $TUNNEL_URL"
            echo ""
            echo "  Update VITE_SERVER_URL to this URL in desktop_app/.env:"
            echo "    VITE_SERVER_URL=$TUNNEL_URL"
        else
            info "Tunnel starting... check in a few seconds:"
            echo "  cat /tmp/cloudflare-tunnel.log"
        fi
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  DONE"
echo "=================================================================="
echo ""
echo "  Container:   sudo docker ps -a"
echo "  Logs:        sudo docker logs -f $CONTAINER_NAME"
echo "  Health:      curl http://localhost:${SERVER_PORT}/health"
echo "  WebRTC test: curl http://localhost:${SERVER_PORT}/health | python3 -m json.tool"
echo ""
echo "  To check CUDA is actually running face-swap on GPU:"
echo "    curl http://localhost:${SERVER_PORT}/debug/gpu | python3 -m json.tool"
echo "    # Must show: swapper_provider: CUDAExecutionProvider"
echo ""
