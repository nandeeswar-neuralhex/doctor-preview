#!/bin/bash
# Local testing script (requires NVIDIA Docker)

set -e

echo "Building Docker image..."
docker build -t doctor-preview:test .

echo "Running container..."
docker run --rm -it \
    --gpus all \
    -p 8765:8765 \
    -e JPEG_QUALITY=85 \
    -e MAX_SESSIONS=5 \
    doctor-preview:test
