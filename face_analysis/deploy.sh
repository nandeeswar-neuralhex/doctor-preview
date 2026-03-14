#!/bin/bash
# ===================================================================
# Deploy Face Analysis Service to Azure T4 GPU
# Uses existing ACR: doctorpreviewacr
# Creates a NEW container instance on port 8766
# ===================================================================

set -e

# ===================================================================
# CONFIGURATION
# ===================================================================
RESOURCE_GROUP="doctor-preview-rg"
LOCATION="eastus"
ACR_NAME="doctorpreviewacr"
IMAGE_NAME="face-analysis-gpu"
TAG="latest"
CONTAINER_NAME="face-analysis-gpu-instance"

# GPU SKU — T4 (same as your existing instances)
GPU_SKU="Standard_NC4as_T4_v3"   # 4 vCPU, 28 GB RAM, 1× T4 GPU
GPU_COUNT=1

# Network
PORT=8766

echo "====================================================================="
echo "🏥 Doctor Face Analysis — Azure T4 GPU Deployment"
echo "====================================================================="
echo ""
echo "Image:     $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG"
echo "Container: $CONTAINER_NAME"
echo "GPU:       $GPU_SKU (T4)"
echo "Port:      $PORT"
echo ""

# ===================================================================
# STEP 1: Verify Azure login
# ===================================================================
echo "Step 1/5: Verifying Azure login..."
if ! az account show &>/dev/null; then
    echo "Not logged in. Run: az login"
    exit 1
fi
az account show --query "[name, id]" -o tsv
echo ""

# ===================================================================
# STEP 2: Build and push Docker image via ACR Build
# ===================================================================
echo "====================================================================="
echo "Step 2/5: Building Docker image in ACR (cloud build)..."
echo "====================================================================="
echo "This takes 5-10 minutes on first build..."

az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_NAME:$TAG" \
    --file Dockerfile \
    --platform linux/amd64 \
    . \
    --resource-group "$RESOURCE_GROUP"

echo "✅ Image pushed: $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG"
echo ""

# ===================================================================
# STEP 3: Get ACR credentials
# ===================================================================
echo "Step 3/5: Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "loginServer" \
    --output tsv)

ACR_USERNAME=$(az acr credential show \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "username" \
    --output tsv)

ACR_PASSWORD=$(az acr credential show \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "passwords[0].value" \
    --output tsv)

echo "ACR: $ACR_LOGIN_SERVER"
echo ""

# ===================================================================
# STEP 4: Delete old container if exists (ignore errors)
# ===================================================================
echo "Step 4/5: Cleaning up old container (if exists)..."
az container delete \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --yes 2>/dev/null || true
echo ""

# ===================================================================
# STEP 5: Deploy container with T4 GPU
# ===================================================================
echo "====================================================================="
echo "Step 5/5: Deploying container with T4 GPU..."
echo "====================================================================="
echo "This takes 3-8 minutes..."

DNS_LABEL="face-analysis-$(date +%s | tail -c 6)"

az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG" \
    --registry-login-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu 4 \
    --memory 28 \
    --gpu-count $GPU_COUNT \
    --gpu-sku "$GPU_SKU" \
    --ports $PORT \
    --dns-name-label "$DNS_LABEL" \
    --environment-variables \
        PORT=$PORT \
    --ip-address Public \
    --protocol TCP \
    --location "$LOCATION" \
    --output table

# ===================================================================
# Get connection details
# ===================================================================
echo ""
echo "Waiting 15s for container to stabilize..."
sleep 15

FQDN=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query "ipAddress.fqdn" \
    --output tsv)

IP=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query "ipAddress.ip" \
    --output tsv)

STATE=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query "instanceView.state" \
    --output tsv 2>/dev/null || echo "Starting")

echo ""
echo "====================================================================="
echo "✅ FACE ANALYSIS SERVICE DEPLOYED!"
echo "====================================================================="
echo ""
echo "  Container: $CONTAINER_NAME"
echo "  State:     $STATE"
echo "  FQDN:      $FQDN"
echo "  IP:        $IP"
echo "  Port:      $PORT"
echo ""
echo "  API URL:        http://$IP:$PORT"
echo "  Health Check:   http://$IP:$PORT/health"
echo "  Analyze:        POST http://$IP:$PORT/analyze"
echo "  Validate:       POST http://$IP:$PORT/validate"
echo ""
echo "====================================================================="
echo "📱 Update your desktop app:"
echo "====================================================================="
echo "  In desktop_app/.env or vite.config.js, set:"
echo "    VITE_ANALYSIS_URL=http://$IP:$PORT"
echo ""
echo "  Or the FaceAnalysis.jsx will auto-derive from your server URL:"
echo "    serverUrl.replace(/:port/, ':$PORT')"
echo ""
echo "====================================================================="
echo "🔧 Management commands:"
echo "====================================================================="
echo "  Logs:    az container logs -g $RESOURCE_GROUP -n $CONTAINER_NAME"
echo "  Status:  az container show -g $RESOURCE_GROUP -n $CONTAINER_NAME -o table"
echo "  Stop:    az container stop -g $RESOURCE_GROUP -n $CONTAINER_NAME"
echo "  Delete:  az container delete -g $RESOURCE_GROUP -n $CONTAINER_NAME --yes"
echo "====================================================================="
