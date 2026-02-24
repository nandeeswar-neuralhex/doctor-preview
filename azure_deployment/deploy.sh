#!/bin/bash
# ===================================================================
# Azure Container Registry (ACR) + Container Instance Deployment
# Deploy Doctor Preview face swap service to Azure GPU
# ===================================================================

set -e

# ===================================================================
# CONFIGURATION - Update these values
# ===================================================================
RESOURCE_GROUP="doctor-preview-rg"
LOCATION="eastus"  # or southcentralus (NCasT4_v3), westus2, northeurope
ACR_NAME="doctorpreviewacr"  # Must be globally unique, alphanumeric only
IMAGE_NAME="doctor-preview-gpu"
TAG="latest"
CONTAINER_NAME="doctor-preview-gpu-instance"

# GPU SKU - Choose based on your needs and budget
# NCasT4_v3: T4 GPU (cheapest, good for inference)
# NC6s_v3: V100 GPU (more powerful)
# Standard_NC24ads_A100_v4: A100 GPU (most powerful, expensive)
GPU_SKU="Standard_NC4as_T4_v3"  # 4 vCPU, 28 GB RAM, 1x T4 GPU
GPU_COUNT=1

# Network settings
PORT=8765

# ===================================================================
# STEP 1: Check Azure Login (skip if already authenticated)
# ===================================================================
echo "====================================================================="
echo "Step 1: Checking Azure Login"
echo "====================================================================="
if az account show &>/dev/null; then
    echo "Already logged in!"
    az account show --output table
else
    echo "Not logged in. Opening browser for authentication..."
    az login
fi

echo ""
echo "Using subscription:"
az account show --output table

# ===================================================================
# STEP 2: Create Resource Group
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 2: Creating Resource Group"
echo "====================================================================="
echo "Creating resource group: $RESOURCE_GROUP in $LOCATION"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# ===================================================================
# STEP 3: Create Azure Container Registry (ACR)
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 3: Creating Azure Container Registry"
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
# STEP 4: Build and Push Docker Image to ACR
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 4: Building and Pushing Docker Image"
echo "====================================================================="
echo "Building image: $ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG"
echo "This may take 10-15 minutes (downloading models)..."

# Build and push using ACR build (no local Docker needed!)
az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_NAME:$TAG" \
    --file Dockerfile \
    --platform linux/amd64 \
    . \
    --resource-group "$RESOURCE_GROUP"

echo "Image pushed successfully!"

# ===================================================================
# STEP 5: Get ACR Credentials
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 5: Getting ACR Credentials"
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
# STEP 6: Deploy Container Instance with GPU
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 6: Deploying Container Instance with GPU"
echo "====================================================================="
echo "Deploying container: $CONTAINER_NAME"
echo "GPU SKU: $GPU_SKU"
echo "This may take 5-10 minutes..."

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
# STEP 7: Get Connection Details
# ===================================================================
echo ""
echo "====================================================================="
echo "Step 7: Getting Connection Details"
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
echo "WebSocket URL:  ws://$FQDN:$PORT/ws"
echo "API URL:        http://$FQDN:$PORT"
echo "Health Check:   http://$FQDN:$PORT/health"
echo ""
echo "====================================================================="
echo "Next Steps:"
echo "====================================================================="
echo "1. Test the service:"
echo "   curl http://$FQDN:$PORT/health"
echo ""
echo "2. Update your desktop app to use: ws://$FQDN:$PORT/ws"
echo ""
echo "3. View logs:"
echo "   az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "4. Monitor container:"
echo "   az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --output table"
echo ""
echo "5. Delete when done (to save costs):"
echo "   az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
echo "====================================================================="
