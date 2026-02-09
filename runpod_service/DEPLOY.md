# RunPod Deployment Guide - GitHub Method

## ✅ Code is on GitHub!

Repository: https://github.com/nandeeswar-neuralhex/doctor-preview

---

## 🚀 Deploy to RunPod (Step-by-Step)

### Step 1: Go to RunPod Console

Open: https://www.runpod.io/console/pods

### Step 2: Create New GPU Pod

1. Click **"+ GPU Pod"** or **"Deploy"**
2. Select GPU: **RTX 5090** (or RTX 4090 if 5090 unavailable)
   - RTX 5090: ~$0.50-0.60/hr, 30-40 FPS
   - RTX 4090: ~$0.44/hr, 24-30 FPS

### Step 3: Configure Template

Click **"Customize Deployment"** or **"Edit Template"**

#### Container Configuration

**Option A: Use Pre-built Base Image (Faster)**

```
Container Image: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
Docker Command: /bin/bash -c "cd /workspace && bash runpod_service/scripts/start.sh"
```

Then in **Volume** section:
- Click "Add Volume"
- Mount GitHub repo or use RunPod's GitHub integration

**Option B: Build from GitHub (Recommended)**

If RunPod supports GitHub builds:
1. Click **"Build from GitHub"**
2. Repository URL: `https://github.com/nandeeswar-neuralhex/doctor-preview`
3. Branch: `main`
4. Dockerfile path: `runpod_service/Dockerfile`
5. Build context: `runpod_service/`

**Option C: Manual Template**

```
Container Image: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```

In **Container Start Command**:
```bash
apt-get update && \
apt-get install -y git python3.10 python3-pip curl && \
git clone https://github.com/nandeeswar-neuralhex/doctor-preview.git /app && \
cd /app/runpod_service && \
pip install --no-cache-dir -r requirements.txt && \
python src/models/download_models.py && \
python src/server.py
```

### Step 4: Pod Settings

| Setting | Value |
|---------|-------|
| **Container Disk** | 25 GB |
| **Volume Disk** | 10 GB (optional) |
| **Expose HTTP Ports** | `8765` |
| **Expose TCP Ports** | Leave empty |

### Step 5: Environment Variables

Click **"+ Add Environment Variable"** for each:

```
HOST=0.0.0.0
PORT=8765
JPEG_QUALITY=85
MAX_SESSIONS=10
EXECUTION_PROVIDER=CUDAExecutionProvider
```

### Step 6: Deploy!

1. Review settings
2. Click **"Deploy"** or **"Continue"**
3. Wait for pod to start (2-3 minutes)
4. Wait for build/setup (10-15 minutes first time)

---

## 📊 Monitor Deployment

### Check Logs

1. Click on your pod
2. Click **"Logs"** tab
3. Look for:
   ```
   Downloading models...
   Models downloaded successfully
   Server ready!
   Uvicorn running on http://0.0.0.0:8765
   ```

### Get Your Endpoint URL

After deployment, you'll see:
```
https://YOUR_POD_ID-8765.proxy.runpod.net
```

**Copy this URL** - you'll need it for the desktop app!

---

## ✅ Test Your Deployment

### Test 1: Health Check

```bash
curl https://YOUR_POD_ID-8765.proxy.runpod.net/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "gpu_available": true
}
```

### Test 2: Desktop App

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/desktop_app

# Install dependencies (first time only)
npm install

# Start app
npm start
```

1. Click **Settings** (top right)
2. Enter your RunPod URL: `https://YOUR_POD_ID-8765.proxy.runpod.net`
3. Click **Save**
4. Upload a target image
5. Click **Start Preview**
6. Check FPS (should be 24-40)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Pod won't start | Check GPU availability, try different region |
| Build fails | Use Option C (manual install) instead |
| Port not accessible | Verify port 8765 is exposed in HTTP Ports |
| Models not downloading | Increase Container Disk to 30 GB |
| Low FPS | Check GPU utilization in logs |
| Connection timeout | Check firewall, verify URL is correct |

---

## 💰 Cost Management

- **Stop pod when not in use** (RunPod charges per hour)
- **Use Community Cloud** for development (cheaper, less reliable)
- **Use Secure Cloud** for production (more expensive, more reliable)
- **Expected cost**: $0.50-0.60/hr for RTX 5090

---

## 🎯 Next Steps

1. ✅ Code pushed to GitHub
2. ⏳ Deploy to RunPod (follow steps above)
3. ⏳ Get endpoint URL
4. ⏳ Test with desktop app
5. ⏳ Show patient real-time preview!

**Ready to deploy!** Follow the steps above and let me know when your pod is running.
