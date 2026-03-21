#!/bin/bash
# ===================================================================
# ALL-IN-ONE: T4 VM (20.115.36.199) — WebRTC + HTTPS deploy
# Domain: t4lagcyfaceiq.sparkiq.ai
#
# Run this on the VM:
#   ssh azureuser@20.115.36.199
#   bash deploy_t4_webrtc.sh
#
# Script is safe to re-run. DNS must resolve before Step 6 (HTTPS).
# If DNS isn't live yet, everything else still completes fine —
# just re-run the nginx section separately once DNS propagates.
# ===================================================================

set -e

DOMAIN="t4lagcyfaceiq.sparkiq.ai"
SERVER_PORT=8765
IMAGE="doctor-preview-v2"
CONTAINER="doctor-preview"
BRANCH="nandeeswar-webrtc"
REPO="https://github.com/nandeeswar-neuralhex/doctor-preview.git"
CODE_DIR="/opt/doctor-preview"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()     { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

echo ""
echo "==============================================================="
echo "  T4 VM WebRTC Deploy — $DOMAIN"
echo "==============================================================="
echo ""

# ── STEP 1: Swap ────────────────────────────────────────────────────
info "Step 1/7  Ensuring 4 GB swap (prevents OOM during Docker build)..."
if ! swapon --show | grep -q /swapfile 2>/dev/null; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
    success "Swap created (4 GB)"
else
    success "Swap already active: $(free -h | awk '/Swap/{print $2}')"
fi

# ── STEP 2: Disk space ──────────────────────────────────────────────
info "Step 2/7  Checking disk space..."
AVAIL_GB=$(df / | awk 'NR==2{print int($4/1024/1024)}')
echo "  Available: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 20 ]; then
    warn "Low disk — pruning Docker to free space..."
    sudo docker system prune -af
    sudo docker builder prune -af
fi
success "Disk OK"

# ── STEP 3: Pull code ───────────────────────────────────────────────
info "Step 3/7  Pulling code (branch: $BRANCH)..."
if [ -d "$CODE_DIR/.git" ]; then
    cd "$CODE_DIR"
    sudo git fetch origin
    sudo git checkout "$BRANCH"
    sudo git pull origin "$BRANCH"
else
    sudo rm -rf "$CODE_DIR"
    sudo git clone -b "$BRANCH" "$REPO" "$CODE_DIR"
fi
success "Code up to date"

# ── STEP 4: Build Docker image ──────────────────────────────────────
info "Step 4/7  Building Docker image with WebRTC enabled..."
info "          This takes ~15-20 min. Tail: tail -f /tmp/docker-build.log"
cd "$CODE_DIR/azure_deployment"
sudo docker system prune -f > /dev/null 2>&1   # remove dangling layers only

sudo docker build --no-cache -t "${IMAGE}:latest" . 2>&1 | tee /tmp/docker-build.log
echo ""

# Smoke test: GPU + WebRTC imports must all succeed
info "Verifying GPU + WebRTC inside image..."
sudo docker run --rm --gpus all "${IMAGE}:latest" python3 -c "
import cv2; print('  cv2:      ', cv2.__version__)
import onnxruntime as ort; print('  ort:      ', ort.__version__)
providers = ort.get_available_providers()
print('  providers:', providers)
assert 'CUDAExecutionProvider' in providers, 'GPU NOT active — stop here!'
import torch; print('  torch:    ', torch.__version__, '  CUDA:', torch.cuda.is_available())
import aiortc; print('  aiortc:   ', aiortc.__version__)
print('  All imports OK')
" || err "Smoke test failed — check /tmp/docker-build.log"
success "Image built and verified"

# ── STEP 5: Start container ─────────────────────────────────────────
info "Step 5/7  Starting container (ENABLE_WEBRTC=true)..."
sudo docker rm -f "$CONTAINER" 2>/dev/null || true

sudo docker run -d \
  --name "$CONTAINER" \
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
  "${IMAGE}:latest"

info "Waiting for server health (model load takes ~20 s)..."
for i in $(seq 1 40); do
    CODE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${SERVER_PORT}/health 2>/dev/null || echo "000")
    if [ "$CODE" = "200" ]; then
        echo ""
        WEBRTC=$(curl -s http://127.0.0.1:${SERVER_PORT}/health | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('webrtc_enabled','?'))")
        success "Container healthy after $((i*5))s  (webrtc_enabled=$WEBRTC)"
        if [ "$WEBRTC" != "True" ]; then
            warn "webrtc_enabled is not True — check: sudo docker logs $CONTAINER"
        fi
        break
    fi
    echo -ne "\r  ⏳ $((i*5))s — HTTP $CODE   "
    sleep 5
done

# ── STEP 6: nginx + Let's Encrypt HTTPS ────────────────────────────
info "Step 6/7  Installing nginx + Let's Encrypt for $DOMAIN..."

# Quick DNS check first
DNS_IP=$(dig +short "$DOMAIN" 2>/dev/null | tail -1)
MY_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")

if [ "$DNS_IP" != "$MY_IP" ]; then
    warn "DNS not yet live:"
    warn "  $DOMAIN resolves to: '${DNS_IP}' (expected: $MY_IP)"
    warn ""
    warn "Skipping HTTPS setup for now — re-run Step 6 once DNS propagates:"
    warn ""
    warn "  bash $CODE_DIR/azure_deployment/setup_nginx_https.sh $DOMAIN"
    warn ""
else
    success "DNS resolves correctly: $DOMAIN → $MY_IP"

    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends nginx certbot python3-certbot-nginx

    # Allow ports in ufw if active
    if sudo ufw status 2>/dev/null | grep -q "Status: active"; then
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
    fi

    # Write nginx config
    sudo tee /etc/nginx/sites-available/doctor-preview > /dev/null <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    location / {
        proxy_pass         http://127.0.0.1:${SERVER_PORT};
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    \$http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_read_timeout    300s;
        proxy_send_timeout    300s;
        proxy_connect_timeout  10s;
        client_max_body_size 50M;
    }
}
NGINX

    sudo ln -sf /etc/nginx/sites-available/doctor-preview /etc/nginx/sites-enabled/doctor-preview
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t && sudo systemctl reload nginx

    # Get cert
    sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
        --email admin@sparkiq.ai --redirect

    sudo nginx -t && sudo systemctl reload nginx

    HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://${DOMAIN}/health" 2>/dev/null || echo "000")
    success "https://${DOMAIN}/health → HTTP $HTTP_CODE"
fi

# ── STEP 7: Open Azure NSG UDP ports for WebRTC media ───────────────
info "Step 7/7  NSG UDP ports for WebRTC media (49152-65535)..."
info "          Run this from your laptop (needs Azure CLI + login):"
echo ""
echo "  az network nsg rule create \\"
echo "    --resource-group doctor-preview-rg \\"
echo "    --nsg-name <YOUR-NSG-NAME> \\"
echo "    --name WebRTC-UDP-Media \\"
echo "    --priority 1100 \\"
echo "    --protocol Udp \\"
echo "    --direction Inbound \\"
echo "    --destination-port-ranges 49152-65535 \\"
echo "    --access Allow"
echo ""
echo "  (Also open TCP 80 and TCP 443 if not already open)"
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "==============================================================="
echo "  DONE"
echo "==============================================================="
echo ""
echo "  Local health:  curl http://localhost:${SERVER_PORT}/health"
echo "  HTTPS health:  curl https://${DOMAIN}/health"
echo "  GPU check:     curl https://${DOMAIN}/debug/gpu | python3 -m json.tool"
echo "  Logs:          sudo docker logs -f ${CONTAINER}"
echo ""
echo "  desktop_app/.env is already updated to:"
echo "    VITE_SERVER_URL=https://${DOMAIN}"
echo ""
