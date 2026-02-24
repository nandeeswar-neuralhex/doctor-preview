#!/bin/bash
# ===================================================================
# Update Container Instance (Pull new image and restart)
# ===================================================================

set -e

RESOURCE_GROUP="doctor-preview-rg"
CONTAINER_NAME="doctor-preview-gpu-instance"
ACR_NAME="doctorpreviewacr"
IMAGE_NAME="doctor-preview-gpu"
TAG="latest"

echo "====================================================================="
echo "Updating Container Instance"
echo "====================================================================="

# Get ACR details
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

# Rebuild image
echo "Rebuilding image..."
az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_NAME:$TAG" \
    --file Dockerfile \
    --platform linux/amd64 \
    . \
    --resource-group "$RESOURCE_GROUP"

# Delete old container
echo "Deleting old container..."
az container delete \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --yes

# Recreate container
echo "Creating new container..."
az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$TAG" \
    --registry-login-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu 4 \
    --memory 28 \
    --gpu-count 1 \
    --gpu-sku "Standard_NC4as_T4_v3" \
    --ports 8765 \
    --dns-name-label "doctor-preview-$RANDOM" \
    --environment-variables \
        EXECUTION_PROVIDER=CUDAExecutionProvider \
        PORT=8765 \
        ENABLE_WEBRTC=true \
        ENABLE_LIPSYNC=true \
    --ip-address Public \
    --protocol TCP \
    --output table

echo "Update complete!"
