# Local Testing on MacBook (CPU-only)

## ⚠️ Important: MacBook Air M3 Limitations

Your MacBook Air M3 **does not have NVIDIA GPU**, so:
- ✅ You CAN test the API structure and WebSocket connection
- ❌ You CANNOT test GPU performance (will be ~1-2 FPS on CPU)
- ✅ You CAN verify the code works before RunPod deployment

## Option 1: CPU-Only Docker Test (Recommended)

This will run the service on CPU to verify everything works:

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/runpod_service

# Build CPU version
docker build -t doctor-preview:cpu-test .

# Run without GPU flag
docker run --rm -it \
  -p 8765:8765 \
  -e EXECUTION_PROVIDER=CPUExecutionProvider \
  doctor-preview:cpu-test
```

**Expected**: Server starts, models load (takes 2-3 min), but FPS will be ~1-2 (very slow).

## Option 2: Python Virtual Environment (Faster Testing)

Test without Docker to iterate faster:

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/runpod_service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python src/models/download_models.py

# Run server
export EXECUTION_PROVIDER=CPUExecutionProvider
python src/server.py
```

Then test with:
```bash
# In another terminal
curl http://localhost:8765/health
```

## Option 3: Mock Test (No Face Swap, Just API)

I can create a mock version that skips face processing to test the WebSocket flow:

```bash
# Create a simple mock server
python scripts/mock_server.py
```

---

## What You Should Test Locally

| Test | Purpose | Expected Result |
|------|---------|-----------------|
| Server starts | Code has no syntax errors | Logs show "Server ready!" |
| `/health` endpoint | API works | `{"status": "healthy"}` |
| Model download | Models accessible | No errors in logs |
| WebSocket connect | Connection works | Client connects successfully |

## After Local Testing → RunPod

Once you verify the code structure works locally, deploy to RunPod where:
- ✅ NVIDIA GPU available → 24+ FPS
- ✅ Same Docker image works
- ✅ You already tested the API structure

---

**Which option do you want to try?**
1. CPU Docker test (slow but complete)
2. Python venv test (faster iteration)
3. Mock server (just test API structure)
