#!/bin/bash
# ===================================================================
# HTTPS + nginx setup for the T4 VM at 20.115.36.199
# Run this ON the VM AFTER the Docker container is running.
#
# Usage:
#   bash setup_nginx_https.sh <your-domain>
#   Example: bash setup_nginx_https.sh t4.faceiq.sparkiq.ai
#
# Pre-requisites:
#   1. Docker container is running on port 8765
#   2. DNS A record for <your-domain> is already pointing to 20.115.36.199
#   3. Port 80 and 443 are open in the Azure NSG
# ===================================================================

set -e

DOMAIN="${1:-}"
SERVER_PORT=8765

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

if [ -z "$DOMAIN" ]; then
    error "Usage: bash setup_nginx_https.sh <domain>
Example: bash setup_nginx_https.sh t4.faceiq.sparkiq.ai"
fi

echo "=================================================================="
echo "  HTTPS setup for: $DOMAIN → localhost:$SERVER_PORT"
echo "=================================================================="
echo ""

# ── Step 1: Install nginx + certbot ────────────────────────────────
info "Installing nginx and certbot..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends nginx certbot python3-certbot-nginx

# ── Step 2: Open ports 80 + 443 in firewall (ufw) ──────────────────
if command -v ufw &>/dev/null && sudo ufw status | grep -q "Status: active"; then
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
fi

# ── Step 3: Write nginx config ─────────────────────────────────────
info "Writing nginx config for $DOMAIN..."
sudo tee /etc/nginx/sites-available/doctor-preview > /dev/null <<NGINX
server {
    listen 80;
    server_name ${DOMAIN};

    # Certbot will inject SSL config here after we run it.
    # For now, proxy everything to the Docker container on port ${SERVER_PORT}.

    location / {
        proxy_pass         http://127.0.0.1:${SERVER_PORT};
        proxy_http_version 1.1;

        # WebSocket upgrade (for /ws/ endpoints)
        proxy_set_header   Upgrade    \$http_upgrade;
        proxy_set_header   Connection "upgrade";

        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_set_header   X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto \$scheme;

        # Long timeouts for WebSocket + WebRTC signaling
        proxy_read_timeout    300s;
        proxy_send_timeout    300s;
        proxy_connect_timeout  10s;

        # Large body for image uploads
        client_max_body_size 50M;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/doctor-preview /etc/nginx/sites-enabled/doctor-preview
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
success "nginx config OK"

# ── Step 4: Get Let's Encrypt cert ─────────────────────────────────
info "Obtaining Let's Encrypt certificate for $DOMAIN..."
info "(Certbot will verify ownership via port 80 — make sure DNS is live first)"
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
    --email admin@sparkiq.ai --redirect

sudo nginx -t
sudo systemctl reload nginx
success "SSL certificate installed for $DOMAIN"

# ── Step 5: Auto-renew cert ────────────────────────────────────────
# certbot installs a systemd timer automatically; verify it's active
if systemctl is-enabled certbot.timer &>/dev/null 2>&1; then
    success "Auto-renew timer: $(systemctl status certbot.timer | grep Active)"
else
    info "Setting up monthly cert renewal via cron..."
    echo "0 3 1 * * certbot renew --quiet --nginx" | sudo tee /etc/cron.d/certbot-renew
fi

# ── Step 6: Verify ─────────────────────────────────────────────────
echo ""
info "Testing HTTPS endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://${DOMAIN}/health" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    WEBRTC=$(curl -s --max-time 5 "https://${DOMAIN}/health" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('webrtc_enabled','?'))" 2>/dev/null)
    success "https://${DOMAIN}/health → HTTP 200  (webrtc_enabled=${WEBRTC})"
else
    warn "https://${DOMAIN}/health → HTTP $HTTP_CODE (may need a moment to propagate)"
fi

echo ""
echo "=================================================================="
echo "  DONE — HTTPS is live"
echo "=================================================================="
echo ""
echo "  Health:  curl https://${DOMAIN}/health"
echo "  WebRTC:  curl https://${DOMAIN}/health | python3 -m json.tool"
echo "           # must show  webrtc_enabled: true"
echo ""
echo "  Update desktop_app/.env:"
echo "    VITE_SERVER_URL=https://${DOMAIN}"
echo ""
