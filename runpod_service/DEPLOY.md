# RunPod Deployment - Step-by-Step Guide

## GPU Recommendation: RTX 5090

**Perfect choice!** RTX 5090 specs:
- **Expected FPS**: 30-40 FPS (better than RTX 4090)
- **VRAM**: 24GB (plenty for FaceFusion)
- **Cost**: ~$0.50-0.60/hr

### Recommended Settings
- **Container Disk**: 25 GB (for Docker image + models)
- **Volume Disk**: 10 GB (optional, for caching)
- **Expose Ports**: 8765

---

## Deployment Steps

### Option 1: Direct Docker Image (Recommended)

RunPod can pull directly from Docker Hub, so you need to:

#### Step 1: Build & Push Docker Image

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/runpod_service

# Login to Docker Hub (create account at hub.docker.com if needed)
docker login

# Build the image
docker build -t YOUR_DOCKERHUB_USERNAME/doctor-preview:latest .

# Push to Docker Hub
docker push YOUR_DOCKERHUB_USERNAME/doctor-preview:latest
```

**Replace `YOUR_DOCKERHUB_USERNAME`** with your actual Docker Hub username.

---

#### Step 2: Create RunPod Pod

1. **Go to RunPod Console**: https://www.runpod.io/console/pods
2. **Click "Deploy"** or "+ GPU Pod"
3. **Select GPU**: Choose **RTX 5090**
4. **Select Template**: Click "New Template" or use custom

**Template Configuration:**

| Setting | Value |
|---------|-------|
| **Template Name** | `doctor-preview-faceswap` |
| **Container Image** | `YOUR_DOCKERHUB_USERNAME/doctor-preview:latest` |
| **Docker Command** | Leave empty (uses CMD from Dockerfile) |
| **Container Disk** | `25 GB` |
| **Volume Disk** | `10 GB` (optional) |
| **Expose HTTP Ports** | `8765` |
| **Expose TCP Ports** | Leave empty |

**Environment Variables** (click "Add Environment Variable"):
```
HOST=0.0.0.0
PORT=8765
JPEG_QUALITY=85
MAX_SESSIONS=10
```

5. **Click "Deploy"**

---

#### Step 3: Wait for Deployment

- Pod will start (takes 2-5 minutes)
- Models will download automatically
- Check logs for "Server ready!"

---

#### Step 4: Get Your Endpoint URL

After deployment, you'll see:
```
https://YOUR_POD_ID-8765.proxy.runpod.net
```

**Test it:**
```bash
curl https://YOUR_POD_ID-8765.proxy.runpod.net/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### Option 2: RunPod Template (If You Don't Have Docker Hub)

If you don't want to use Docker Hub, you can use RunPod's built-in registry:

1. **Zip your project:**
```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main
tar -czf runpod_service.tar.gz runpod_service/
```

2. **Upload to RunPod** via their template builder
3. **Build on RunPod** (slower, but no Docker Hub needed)

---

## After Deployment

### Test with Desktop App

1. **Open Desktop App:**
```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/desktop_app
npm start
```

2. **Configure Server:**
   - Click "Settings"
   - Enter: `https://YOUR_POD_ID-8765.proxy.runpod.net`
   - Click "Save"

3. **Upload & Test:**
   - Upload a target image
   - Click "Start Preview"
   - Should see 24-40 FPS!

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Pod won't start | Check Docker image name is correct |
| Models not loading | Increase Container Disk to 30 GB |
| Port not accessible | Verify port 8765 is exposed |
| Low FPS | Check GPU utilization in RunPod logs |
| Connection timeout | Verify firewall/proxy settings |

---

## Cost Optimization

- **Stop pod when not in use** (RunPod charges per hour)
- **Use Community Cloud** for development (cheaper)
- **Use Secure Cloud** for production (more reliable)

---

## Next: Desktop App Configuration

Once your pod is running, copy the URL and use it in the desktop app settings!
