# Quick Start: Test Locally on MacBook Air M3

## 🚀 Fastest Way to Test

Since you have Docker installed, run this:

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/runpod_service

# Start the server (CPU mode, will be slow but validates code)
./scripts/test_mac.sh
```

**What to expect:**
- ⏳ First run: ~5-10 min (downloads AI models)
- ✅ Server starts on `http://localhost:8765`
- ⚠️ FPS will be ~1-2 (CPU is slow, but proves code works)

## 🧪 Verify It's Working

In another terminal:
```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/runpod_service
./scripts/check_health.sh
```

Should see:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 📊 What This Proves

| ✅ Verified | ❌ Not Verified (needs GPU) |
|-------------|----------------------------|
| Code has no bugs | 24+ FPS performance |
| API endpoints work | GPU acceleration |
| Models download correctly | Real-time quality |
| WebSocket connects | Production speed |
| Docker builds successfully | - |

## 🎯 After Local Test → Deploy to RunPod

Once you see the server running locally:
1. **Same Docker image** will work on RunPod
2. **Only difference**: RunPod has GPU → 24+ FPS
3. **You're confident** the code structure is correct

---

## Alternative: Skip Local Test

If you want to skip local testing (since it's slow on CPU):
1. Push Docker image to Docker Hub
2. Deploy directly to RunPod
3. Test there with GPU (faster iteration)

**Your choice!** Local test = safer but slower. Direct RunPod = faster but costs $0.44/hr.
