#!/bin/bash
# Quick deployment script for RunPod

echo "=========================================="
echo "Doctor Preview - RunPod Deployment"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Get Docker Hub username
read -p "Enter your Docker Hub username: " DOCKER_USERNAME

if [ -z "$DOCKER_USERNAME" ]; then
    echo "❌ Docker Hub username is required"
    exit 1
fi

IMAGE_NAME="$DOCKER_USERNAME/doctor-preview:latest"

echo ""
echo "Building image: $IMAGE_NAME"
echo ""

# Build image
docker build -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""

# Login to Docker Hub
echo "Logging in to Docker Hub..."
docker login

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed"
    exit 1
fi

echo ""
echo "Pushing image to Docker Hub..."
docker push "$IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Push failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ SUCCESS! Image pushed to Docker Hub"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Go to https://www.runpod.io/console/pods"
echo "2. Click 'Deploy' or '+ GPU Pod'"
echo "3. Select RTX 5090 GPU"
echo "4. Use this image: $IMAGE_NAME"
echo "5. Container Disk: 25 GB"
echo "6. Expose Port: 8765"
echo ""
echo "Environment Variables:"
echo "  HOST=0.0.0.0"
echo "  PORT=8765"
echo "  JPEG_QUALITY=85"
echo "  MAX_SESSIONS=10"
echo ""
