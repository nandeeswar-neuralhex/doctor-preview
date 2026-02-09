# Doctor Preview - Real-Time Face Swap System

Medical surgery preview system with real-time face swapping.

## 🚀 Quick Deploy to RunPod

### Step 1: Deploy GPU Pod

1. Go to https://www.runpod.io/console/pods
2. Click **"+ GPU Pod"**
3. Select **RTX 5090** (or RTX 4090)
4. Click **"Customize Deployment"**

### Step 2: Configure Container

**Container Image**: Use template or custom image
- Click **"Use Custom Template"**
- Select **"Build from GitHub"**
- Repository: `https://github.com/nandeeswar-neuralhex/doctor-preview`
- Dockerfile path: `runpod_service/Dockerfile`
- Build context: `runpod_service/`

**OR use direct image path**:
```
Container Image: nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
```
Then mount this repo as volume.

### Step 3: Pod Settings

| Setting | Value |
|---------|-------|
| **Container Disk** | 25 GB |
| **Volume Disk** | 10 GB (optional) |
| **Expose HTTP Ports** | 8765 |

**Environment Variables**:
```
HOST=0.0.0.0
PORT=8765
JPEG_QUALITY=85
MAX_SESSIONS=10
```

### Step 4: Deploy & Test

1. Click **"Deploy"**
2. Wait 10-15 minutes for build
3. Get endpoint: `https://YOUR_POD_ID-8765.proxy.runpod.net`
4. Test: `curl https://YOUR_POD_ID-8765.proxy.runpod.net/health`

---

## 📱 Desktop App

```bash
cd desktop_app
npm install
npm start
```

Configure RunPod URL in Settings → Start Preview!

---

## 📁 Project Structure

- `runpod_service/` - Cloud AI service (deploy to RunPod)
- `desktop_app/` - Desktop application (Windows/Mac)

See individual README files for details.
