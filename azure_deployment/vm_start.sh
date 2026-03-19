#!/bin/bash
# ===================================================================
# Start Azure H100 GPU VM + Verify Services
# Starts the deallocated VM, waits for it to boot, and verifies
# the face-swap server + Cloudflare tunnel are running.
# ===================================================================

set -e

# ===================================================================
# CONFIGURATION — Update these to match your H100 VM
# ===================================================================
RESOURCE_GROUP="DOCTOR-PREVIEW-H100-RG"
VM_NAME="doctor-preview-h100"
SUBSCRIPTION_ID="60fb43e3-960f-44d7-aad5-ec31a2c6d27c"

# Service endpoint (via Cloudflare tunnel)
SERVICE_URL="https://faceiq.sparkiq.ai"
HEALTH_ENDPOINT="${SERVICE_URL}/health"

# Server port on the VM
SERVER_PORT=8765

# Timeouts (seconds)
VM_BOOT_TIMEOUT=300       # Max wait for VM to reach "Running" state
SERVICE_READY_TIMEOUT=180 # Max wait for health endpoint to respond

# ===================================================================
# COLOR HELPERS
# ===================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ===================================================================
# STEP 0: Pre-flight checks
# ===================================================================
echo "====================================================================="
echo "  🚀  Starting H100 GPU VM"
echo "====================================================================="
echo ""

# Check Azure CLI is installed
if ! command -v az &>/dev/null; then
    error "Azure CLI (az) is not installed."
    echo "  Install: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check login
if ! az account show &>/dev/null 2>&1; then
    warn "Not logged into Azure. Opening browser..."
    az login
fi

# Set subscription
az account set --subscription "$SUBSCRIPTION_ID" 2>/dev/null
info "Using subscription: $SUBSCRIPTION_ID"
echo ""

# ===================================================================
# STEP 1: Check current VM state
# ===================================================================
info "Checking current VM state..."
VM_STATE=$(az vm get-instance-view \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
    --output tsv 2>/dev/null || echo "Unknown")

echo "  Current state: $VM_STATE"

if [[ "$VM_STATE" == "VM running" ]]; then
    success "VM is already running!"
    echo ""
else
    # ===================================================================
    # STEP 2: Start the VM
    # ===================================================================
    info "Starting VM '$VM_NAME'..."
    az vm start \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --no-wait

    # Wait for VM to reach "Running" state
    info "Waiting for VM to boot (timeout: ${VM_BOOT_TIMEOUT}s)..."
    ELAPSED=0
    INTERVAL=10
    while [ $ELAPSED -lt $VM_BOOT_TIMEOUT ]; do
        VM_STATE=$(az vm get-instance-view \
            --resource-group "$RESOURCE_GROUP" \
            --name "$VM_NAME" \
            --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
            --output tsv 2>/dev/null || echo "Unknown")
        
        if [[ "$VM_STATE" == "VM running" ]]; then
            success "VM is running!"
            break
        fi
        
        echo -ne "\r  ⏳ ${ELAPSED}s elapsed — state: $VM_STATE   "
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    done
    echo ""

    if [[ "$VM_STATE" != "VM running" ]]; then
        error "VM did not start within ${VM_BOOT_TIMEOUT}s. Current state: $VM_STATE"
        exit 1
    fi
fi

# ===================================================================
# STEP 3: Get VM IP (for reference)
# ===================================================================
info "Getting VM details..."
VM_IP=$(az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --show-details \
    --query "publicIps" \
    --output tsv 2>/dev/null || echo "N/A")

echo "  Public IP: $VM_IP"
echo "  Cloudflare URL: $SERVICE_URL"
echo ""

# ===================================================================
# STEP 4: Wait for services to come up
# (Server + Cloudflare tunnel need time after VM boot)
# ===================================================================
info "Waiting for face-swap server to become ready..."
info "(The server needs to load AI models + GPU warmup after boot)"
echo ""

ELAPSED=0
INTERVAL=10
while [ $ELAPSED -lt $SERVICE_READY_TIMEOUT ]; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 --max-time 10 "$HEALTH_ENDPOINT" 2>/dev/null || echo "000")
    
    if [[ "$HTTP_CODE" == "200" ]]; then
        echo ""
        success "Server is healthy! (HTTP $HTTP_CODE)"
        break
    fi
    
    echo -ne "\r  ⏳ ${ELAPSED}s elapsed — HTTP status: $HTTP_CODE (waiting for 200)   "
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done
echo ""

if [[ "$HTTP_CODE" != "200" ]]; then
    warn "Server did not respond with 200 within ${SERVICE_READY_TIMEOUT}s"
    warn "The VM is running but the server may still be loading models."
    warn "You can check manually: curl $HEALTH_ENDPOINT"
    echo ""
    echo "  If the server needs to be started manually, SSH into the VM:"
    echo "    ssh <user>@$VM_IP"
    echo ""
    echo "  Then start the server:"
    echo "    cd /app && docker compose up -d"
    echo "    # or: cd /app/src && python3 -m uvicorn server:app --host 0.0.0.0 --port $SERVER_PORT"
    echo ""
    exit 1
fi

# ===================================================================
# DONE
# ===================================================================
echo ""
echo "====================================================================="
echo -e "  ${GREEN}✅  Everything is up and running!${NC}"
echo "====================================================================="
echo ""
echo "  VM Name:        $VM_NAME"
echo "  VM IP:          $VM_IP"
echo "  Server URL:     $SERVICE_URL"
echo "  Health Check:   $HEALTH_ENDPOINT"
echo "  WebSocket:      wss://faceiq.sparkiq.ai/ws"
echo ""
echo "  To stop the VM and save costs:"
echo "    ./vm_stop.sh"
echo ""
