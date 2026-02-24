#!/bin/bash
# ===================================================================
# Azure GPU Deployment - Quick Deploy (Skip Login)
# Use this if you're already logged in with: az login
# ===================================================================

set -e

# ===================================================================
# CONFIGURATION
# ===================================================================
RESOURCE_GROUP="doctor-preview-rg"
LOCATION="eastus"
ACR_NAME="doctorpreviewacr"  # Must be globally unique
IMAGE_NAME="doctor-preview-gpu"
TAG="latest"
CONTAINER_NAME="doctor-preview-gpu-instance"
GPU_SKU="Standard_NC4as_T4_v3"  # T4 GPU
GPU_COUNT=1
PORT=8765

# ===================================================================
# Verify Login
# ===================================================================
echo "====================================================================="
echo "Doctor Preview - Azure GPU Deployment"
echo "====================================================================="
echo ""
echo "Using subscription:"
az account show --output table

echo ""
read -p "Continue with this subscription? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Deployment cancelled."
    exit 0
fi

# ===================================================================
# STEP 1: Create Resource Group
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 1/5: Creating Resource Group"
echo "====================================================================="
echo "Creating resource group: $RESOURCE_GROUP in $LOCATION"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# ===================================================================
# STEP 2: Create Azure Container Registry (ACR)
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 2/5: Creating Azure Container Registry"
echo "====================================================================="
echo "Creating ACR: $ACR_NAME"
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Premium \
    --admin-enabled true \
    --location "$LOCATION" \
    --output table

echo "Getting ACR login server..."
ACR_LOGIN_SERVER=$(az acr show \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "loginServer" \
    --output tsv)

echo "ACR Login Server: $ACR_LOGIN_SERVER"

# ===================================================================
# STEP 3: Build and Push Docker Image to ACR
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 3/5: Building and Pushing Docker Image"
echo "====================================================================="
echo "Building image: $ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG"
echo "⏱️  This will take 10-15 minutes (downloading AI models)..."
echo ""

# Build and push using ACR build (no local Docker needed!)
az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_NAME:$TAG" \
    --file Dockerfile \
    --platform linux/amd64 \
    . \
    --resource-group "$RESOURCE_GROUP"

echo "✅ Image pushed successfully!"

# ===================================================================
# STEP 4: Get ACR Credentials
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 4/5: Getting ACR Credentials"
echo "====================================================================="
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

echo "ACR Username: $ACR_USERNAME"
echo "ACR Password: [HIDDEN]"

# ===================================================================
# STEP 5: Deploy Container Instance with GPU
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 5/5: Deploying Container Instance with GPU"
echo "====================================================================="
echo "Deploying container: $CONTAINER_NAME"
echo "GPU SKU: $GPU_SKU (NVIDIA T4)"
echo "⏱️  This will take 5-10 minutes..."
echo ""

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
    --dns-name-label "doctor-preview-$RANDOM" \
    --environment-variables \
        EXECUTION_PROVIDER=CUDAExecutionProvider \
        PORT=$PORT \
        ENABLE_WEBRTC=true \
        ENABLE_LIPSYNC=true \
    --ip-address Public \
    --protocol TCP \
    --location "$LOCATION" \
    --output table

# ===================================================================
# STEP 6: Get Connection Details
# ===================================================================
echo ""
echo "====================================================================="
echo "Getting Connection Details"
echo "====================================================================="

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

echo ""
echo "====================================================================="
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "====================================================================="
echo ""
echo "Container FQDN: $FQDN"
echo "Container IP:   $IP"
echo "Port:           $PORT"
echo ""
echo "📡 WebSocket URL:  ws://$FQDN:$PORT/ws"
echo "🌐 API URL:        http://$FQDN:$PORT"
echo "❤️  Health Check:   http://$FQDN:$PORT/health"
echo ""
echo "====================================================================="
echo "Next Steps:"
echo "====================================================================="
echo "1. Test the service:"
echo "   curl http://$FQDN:$PORT/health"
echo ""
echo "2. Update your desktop app to use:"
echo "   ws://$FQDN:$PORT/ws"
echo ""
echo "3. View logs:"
echo "   ./monitor.sh logs"
echo ""
echo "4. Monitor container:"
echo "   ./monitor.sh"
echo ""
echo "5. Delete when done (to save costs):"
echo "   ./cleanup.sh"
echo ""
echo "💰 Current cost: ~\$0.53/hour = ~\$12.62/day (T4 GPU)"
echo "====================================================================="
