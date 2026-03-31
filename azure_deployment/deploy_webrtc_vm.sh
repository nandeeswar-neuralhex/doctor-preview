#!/bin/bash
# ===================================================================
# FRESH DEPLOY: doctor-preview-webrtc VM (20.102.71.88)
# T4 GPU — WebRTC + HTTPS
#
# SSH into the VM first, then run:
#   ssh azureuser@20.102.71.88
#   bash deploy_webrtc_vm.sh
#
# Or from your laptop:
#   scp deploy_webrtc_vm.sh azureuser@20.102.71.88:~/
#   ssh azureuser@20.102.71.88 "bash ~/deploy_webrtc_vm.sh"
#
# Safe to re-run. HTTPS step is skipped if DNS isn't live yet.
# ===================================================================

set -e

# ── CONFIG ──────────────────────────────────────────────────────────
# Change DOMAIN to whatever subdomain you point at 20.102.71.88
DOMAIN="${1:-}"
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
echo "  doctor-preview-webrtc VM (20.102.71.88) — Fresh Deploy"
if [ -n "$DOMAIN" ]; then
echo "  Domain: $DOMAIN"
fi
echo "==============================================================="
echo ""

# ── STEP 1: Cleanup — remove ALL old containers and images ──────────
info "Step 1/7  Cleaning up old containers and images..."

# Stop and remove ALL containers (debug leftovers)
CONTAINERS=$(sudo docker ps -aq 2>/dev/null)
if [ -n "$CONTAINERS" ]; then
    sudo docker stop $CONTAINERS 2>/dev/null || true
    sudo docker rm -f $CONTAINERS 2>/dev/null || true
    success "Removed old containers"
else
    success "No containers to remove"
fi

# Remove ALL old images
sudo docker system prune -af --volumes 2>/dev/null || true
sudo docker builder prune -af 2>/dev/null || true
success "Docker cleaned"

# ── STEP 2: Ensure swap (prevents OOM during build) ─────────────────
info "Step 2/7  Ensuring 4 GB swap..."
if ! swapon --show | grep -q /swapfile 2>/dev/null; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
    success "Swap created (4 GB)"
else
    success "Swap already active: $(free -h | awk '/Swap/{print $2}')"
fi

# Check disk space
AVAIL_GB=$(df / | awk 'NR==2{print int($4/1024/1024)}')
info "Disk available: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 20 ]; then
    err "Not enough disk space (need >= 20 GB free). Clean manually."
fi

# ── STEP 3: Pull latest code ────────────────────────────────────────
PINNED_COMMIT="ae0ee61"   # Stable version with lip sync of the real person
info "Step 3/7  Pulling code — latest Dockerfile + src/ from $PINNED_COMMIT..."
if [ -d "$CODE_DIR/.git" ]; then
    cd "$CODE_DIR"
    sudo git config --global --add safe.directory "$CODE_DIR"
    sudo git fetch origin
    sudo git checkout "$BRANCH"
    sudo git reset --hard "origin/$BRANCH"
    sudo git clean -fd
else
    sudo rm -rf "$CODE_DIR"
    sudo git clone -b "$BRANCH" "$REPO" "$CODE_DIR"
fi
cd "$CODE_DIR"
success "Dockerfile at: $(git log --oneline -1)"

# Now overlay only src/ from the stable commit
info "Checking out azure_deployment/src/ from $PINNED_COMMIT..."
sudo git checkout "$PINNED_COMMIT" -- azure_deployment/src/
success "src/ at: $PINNED_COMMIT (Stable version with lip sync of the real person)"

# ── STEP 4: Build Docker image ──────────────────────────────────────
info "Step 4/7  Building Docker image (this takes ~15-20 min)..."
info "          Tail logs: tail -f /tmp/docker-build.log"
cd "$CODE_DIR/azure_deployment"

sudo docker build --no-cache -t "${IMAGE}:latest" . 2>&1 | tee /tmp/docker-build.log

if ! sudo docker image inspect "${IMAGE}:latest" &>/dev/null; then
    err "Docker build failed — check /tmp/docker-build.log"
fi

# Smoke test
info "Verifying GPU + WebRTC inside image..."
sudo docker run --rm --gpus all "${IMAGE}:latest" python3 -c "
import cv2; print('  cv2:      ', cv2.__version__)
import onnxruntime as ort; print('  ort:      ', ort.__version__)
providers = ort.get_available_providers()
print('  providers:', providers)
assert 'CUDAExecutionProvider' in providers, 'GPU NOT active!'
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

info "Waiting for server health (model load takes ~20-30s)..."
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
    echo -ne "\r  waiting $((i*5))s — HTTP $CODE   "
    sleep 5
done

# ── STEP 6: HTTPS via nginx + Let's Encrypt ─────────────────────────
if [ -z "$DOMAIN" ]; then
    warn "Step 6/7  No domain provided — skipping HTTPS setup."
    warn "  To add HTTPS later, run:"
    warn "    bash $CODE_DIR/azure_deployment/setup_nginx_https.sh <your-domain>"
else
    info "Step 6/7  Setting up HTTPS for $DOMAIN..."

    # DNS check
    DNS_IP=$(dig +short "$DOMAIN" 2>/dev/null | tail -1)
    MY_IP=$(curl -s --max-time 5 https://api.ipify.org 2>/dev/null || echo "unknown")

    if [ "$DNS_IP" != "$MY_IP" ]; then
        warn "DNS not yet live:"
        warn "  $DOMAIN resolves to: '${DNS_IP}' (expected: $MY_IP)"
        warn ""
        warn "Skipping HTTPS — once DNS is live, run:"
        warn "    bash $CODE_DIR/azure_deployment/setup_nginx_https.sh $DOMAIN"
    else
        success "DNS resolves correctly: $DOMAIN -> $MY_IP"

        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends nginx certbot python3-certbot-nginx

        if sudo ufw status 2>/dev/null | grep -q "Status: active"; then
            sudo ufw allow 80/tcp
            sudo ufw allow 443/tcp
        fi

        # Remove old nginx configs
        sudo rm -f /etc/nginx/sites-enabled/default
        sudo rm -f /etc/nginx/sites-enabled/faceiq.sparkiq.ai 2>/dev/null || true

        # Write new nginx config
        sudo tee /etc/nginx/sites-available/doctor-preview > /dev/null <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    client_max_body_size 50M;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        if (\$request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin  "*" always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;
            add_header Content-Length 0;
            add_header Content-Type  text/plain;
            return 204;
        }

        proxy_hide_header Access-Control-Allow-Origin;
        proxy_hide_header Access-Control-Allow-Methods;
        proxy_hide_header Access-Control-Allow-Headers;
        proxy_hide_header Access-Control-Allow-Credentials;

        add_header Access-Control-Allow-Origin  "*" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;

        proxy_pass         http://127.0.0.1:${SERVER_PORT};
        proxy_http_version 1.1;
        proxy_set_header   Upgrade    \$http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;
        proxy_buffering    off;
        proxy_read_timeout    3600s;
        proxy_send_timeout    3600s;
        proxy_connect_timeout  10s;
    }
}
NGINX

        sudo ln -sf /etc/nginx/sites-available/doctor-preview /etc/nginx/sites-enabled/doctor-preview
        sudo nginx -t && sudo systemctl reload nginx

        # Get cert
        sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
            --email admin@sparkiq.ai --redirect

        sudo nginx -t && sudo systemctl reload nginx

        HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://${DOMAIN}/health" 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            WEBRTC=$(curl -s --max-time 5 "https://${DOMAIN}/health" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('webrtc_enabled','?'))" 2>/dev/null)
            success "https://${DOMAIN}/health -> HTTP 200  (webrtc_enabled=${WEBRTC})"
        else
            warn "https://${DOMAIN}/health -> HTTP $HTTP_CODE (may need a moment)"
        fi
    fi
fi

# ── STEP 7: NSG reminder ────────────────────────────────────────────
info "Step 7/7  Azure NSG — ensure these ports are open:"
echo ""
echo "  TCP 80, 443   — HTTP/HTTPS"
echo "  TCP 8765      — server"
echo "  UDP 49152-65535 — WebRTC media"
echo ""
echo "  Run from your laptop (if not already done):"
echo ""
echo "  az network nsg rule create \\"
echo "    --resource-group doctor-preview-rg \\"
echo "    --nsg-name doctor-preview-webrtcNSG \\"
echo "    --name WebRTC-UDP-Media \\"
echo "    --priority 1100 \\"
echo "    --protocol Udp \\"
echo "    --direction Inbound \\"
echo "    --destination-port-ranges 49152-65535 \\"
echo "    --access Allow"
echo ""

# ── Summary ──────────────────────────────────────────────────────────
echo "==============================================================="
echo "  DONE"
echo "==============================================================="
echo ""
echo "  VM:        doctor-preview-webrtc (20.102.71.88)"
echo "  Container: $CONTAINER"
echo "  Port:      $SERVER_PORT"
echo "  WebRTC:    enabled"
echo "  HTTP:      http://20.102.71.88:${SERVER_PORT}/health"
if [ -n "$DOMAIN" ]; then
echo "  HTTPS:     https://${DOMAIN}/health"
echo ""
echo "  Update .env:  VITE_SERVER_URL=https://${DOMAIN}"
fi
echo ""
echo "  Logs:      sudo docker logs -f $CONTAINER"
echo "  Rebuild:   bash ~/deploy_webrtc_vm.sh ${DOMAIN}"
echo "==============================================================="
