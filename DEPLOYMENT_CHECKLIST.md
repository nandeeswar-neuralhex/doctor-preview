# 🚀 Quick Deployment Checklist

## RTX 5090 Settings ✅

**GPU**: RTX 5090  
**Expected FPS**: 30-40 FPS  
**Container Disk**: 25 GB  
**Expose Port**: 8765

---

## Step 1: Build & Push Docker Image

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/runpod_service

# Run automated script
./scripts/deploy_to_dockerhub.sh
```

**OR manually:**
```bash
# Login to Docker Hub (create account at hub.docker.com)
docker login

# Build
docker build -t YOUR_USERNAME/doctor-preview:latest .

# Push
docker push YOUR_USERNAME/doctor-preview:latest
```

---

## Step 2: Create RunPod Pod

1. Go to: https://www.runpod.io/console/pods
2. Click **"+ GPU Pod"** or **"Deploy"**
3. Select **RTX 5090**

**Pod Configuration:**
- **Container Image**: `YOUR_USERNAME/doctor-preview:latest`
- **Container Disk**: `25 GB`
- **Volume Disk**: `10 GB` (optional)
- **Expose HTTP Ports**: `8765`

**Environment Variables** (click "+ Add"):
```
HOST=0.0.0.0
PORT=8765
JPEG_QUALITY=85
MAX_SESSIONS=10
```

4. Click **"Deploy"**

---

## Step 3: Wait & Test

**Wait**: 2-5 minutes for pod to start

**Check logs** for: `Server ready!`

**Get URL**: `https://YOUR_POD_ID-8765.proxy.runpod.net`

**Test:**
```bash
curl https://YOUR_POD_ID-8765.proxy.runpod.net/health
```

---

## Step 4: Configure Desktop App

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/desktop_app
npm install
npm start
```

1. Click **Settings**
2. Enter: `https://YOUR_POD_ID-8765.proxy.runpod.net`
3. Click **Save**
4. Upload target image
5. Click **Start Preview**

---

## Expected Performance

| Metric | Value |
|--------|-------|
| FPS | 30-40 |
| Latency | 50-100ms |
| Cost | ~$0.50-0.60/hr |

---

## Troubleshooting

**Pod won't start?**
- Check Docker image name
- Verify port 8765 is exposed

**Models not loading?**
- Increase Container Disk to 30 GB
- Check logs for download errors

**Can't connect?**
- Verify URL is correct
- Check firewall settings
