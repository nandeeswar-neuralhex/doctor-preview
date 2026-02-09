#!/bin/bash
# Start script for the Face Swap server

echo "=========================================="
echo "Doctor Preview - Face Swap Server"
echo "=========================================="

# Set environment defaults
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8765}"
export JPEG_QUALITY="${JPEG_QUALITY:-85}"
export MAX_SESSIONS="${MAX_SESSIONS:-10}"

echo "Configuration:"
echo "  HOST: $HOST"
echo "  PORT: $PORT"
echo "  JPEG_QUALITY: $JPEG_QUALITY"
echo "  MAX_SESSIONS: $MAX_SESSIONS"
echo "=========================================="

# Start the server
cd /app
python src/server.py
