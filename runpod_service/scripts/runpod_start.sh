#!/bin/bash
set -e

echo "Starting Doctor Preview deployment..."

# Update and install dependencies
apt-get update
apt-get install -y git software-properties-common curl
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
python -m ensurepip --upgrade
python -m pip install --upgrade pip

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
python -m pip install --no-cache-dir -r requirements.txt

# Download models
echo "Downloading AI models..."
python src/models/download_models.py

# Start server
echo "Starting server on port ${PORT:-8765}..."
exec python src/server.py
