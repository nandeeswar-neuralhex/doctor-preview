#!/bin/bash
set -e

echo "Starting Doctor Preview deployment..."

# Update and install dependencies
apt-get update
apt-get install -y git python3.10 python3-pip curl

# Clone or update repository
if [ -d "/workspace/app" ]; then
    echo "Repository exists, pulling latest changes..."
    cd /workspace/app
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/nandeeswar-neuralhex/doctor-preview.git /workspace/app
    cd /workspace/app
fi

# Navigate to service directory
cd /workspace/app/runpod_service

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Download models
echo "Downloading AI models..."
python src/models/download_models.py

# Start server
echo "Starting server..."
python src/server.py
