#!/bin/bash
# Quick local test script for MacBook (CPU-only)

set -e

echo "=========================================="
echo "Local CPU Test (MacBook M3)"
echo "=========================================="
echo ""
echo "⚠️  This will be SLOW (~1-2 FPS) on CPU"
echo "   Purpose: Verify code works before RunPod"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Build image
echo "Building Docker image (CPU version)..."
docker build -t doctor-preview:cpu-test .

echo ""
echo "Starting server on http://localhost:8765"
echo "Press Ctrl+C to stop"
echo ""

# Run container (no --gpus flag for Mac)
docker run --rm -it \
    -p 8765:8765 \
    -e EXECUTION_PROVIDER=CPUExecutionProvider \
    -e JPEG_QUALITY=75 \
    doctor-preview:cpu-test
