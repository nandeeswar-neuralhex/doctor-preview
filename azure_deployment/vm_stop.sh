#!/bin/bash
# ===================================================================
# Stop (Deallocate) Azure H100 GPU VM
# Deallocates the VM to stop billing for compute.
# Disk and IP are preserved so it can be started again quickly.
# ===================================================================

set -e

# ===================================================================
# CONFIGURATION — Must match vm_start.sh
# ===================================================================
RESOURCE_GROUP="DOCTOR-PREVIEW-H100-RG"
VM_NAME="doctor-preview-h100"
SUBSCRIPTION_ID="60fb43e3-960f-44d7-aad5-ec31a2c6d27c"

# Service endpoint (for pre-stop health check)
SERVICE_URL="https://faceiq.sparkiq.ai"
HEALTH_ENDPOINT="${SERVICE_URL}/health"

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
echo "  🛑  Stopping H100 GPU VM (Deallocate)"
echo "====================================================================="
echo ""

# Check Azure CLI
if ! command -v az &>/dev/null; then
    error "Azure CLI (az) is not installed."
    exit 1
fi

# Check login
if ! az account show &>/dev/null 2>&1; then
    warn "Not logged into Azure. Opening browser..."
    az login
fi

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
echo ""

if [[ "$VM_STATE" == "VM deallocated" ]]; then
    success "VM is already deallocated (not billing)."
    exit 0
fi

if [[ "$VM_STATE" == "VM stopped" ]]; then
    warn "VM is stopped but NOT deallocated — you are still being billed!"
    warn "Proceeding to deallocate..."
    echo ""
fi

# ===================================================================
# STEP 2: Confirm with user
# ===================================================================
echo "====================================================================="
echo -e "  ${YELLOW}This will deallocate VM '$VM_NAME'${NC}"
echo "  - Compute billing will STOP"
echo "  - Disk + static IP are preserved"
echo "  - Cloudflare tunnel will go offline"
echo "  - $SERVICE_URL will become unreachable"
echo "====================================================================="
echo ""
read -p "Continue? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" && "$CONFIRM" != "yes" ]]; then
    info "Cancelled."
    exit 0
fi

echo ""

# ===================================================================
# STEP 3: Deallocate the VM
# ===================================================================
info "Deallocating VM '$VM_NAME'..."
info "(This stops the VM and releases the compute resources)"
echo ""

az vm deallocate \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME"

success "VM deallocated successfully!"
echo ""

# ===================================================================
# STEP 4: Verify state
# ===================================================================
info "Verifying final state..."
VM_STATE=$(az vm get-instance-view \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
    --output tsv 2>/dev/null || echo "Unknown")

echo "  Final state: $VM_STATE"
echo ""

# ===================================================================
# DONE
# ===================================================================
echo "====================================================================="
echo -e "  ${GREEN}✅  VM is deallocated — compute billing has stopped${NC}"
echo "====================================================================="
echo ""
echo "  What's preserved (still costs a little):"
echo "    - OS disk & data disks (storage costs only)"
echo "    - Static public IP (if configured)"
echo "    - Network resources"
echo ""
echo "  To start again:"
echo "    ./vm_start.sh"
echo ""
