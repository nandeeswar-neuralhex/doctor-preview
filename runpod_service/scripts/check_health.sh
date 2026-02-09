#!/bin/bash
# Quick health check after starting the server

echo "Testing server health..."
echo ""

# Wait for server to start
sleep 2

# Test health endpoint
echo "1. Testing /health endpoint..."
curl -s http://localhost:8765/health | python3 -m json.tool

echo ""
echo ""
echo "2. Testing root endpoint..."
curl -s http://localhost:8765/ | python3 -m json.tool

echo ""
echo ""
echo "✅ If you see JSON responses above, the server is working!"
echo ""
echo "Next steps:"
echo "  1. Upload a target face image"
echo "  2. Connect via WebSocket"
echo "  3. See test_client.py for full example"
