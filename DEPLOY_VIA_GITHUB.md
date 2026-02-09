# GitHub + RunPod Deployment (Recommended for Mac)

## Why This Approach?

Building Docker on Mac M3 for RunPod (AMD64) is:
- ❌ Slow (cross-platform compilation)
- ❌ Uses lots of disk space
- ❌ Downloads models multiple times

**Better solution**: Push to GitHub → RunPod builds natively on AMD64

---

## Step 1: Push Code to GitHub

### Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `doctor-preview`
3. **Private** repository (recommended)
4. Click "Create repository"

### Push Your Code

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main

# Initialize git
git init
git add .
git commit -m "Initial commit - Doctor Preview face swap system"

# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/doctor-preview.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 2: Deploy on RunPod

### Option A: Use GitHub Container Registry (Recommended)

1. **Go to RunPod**: https://www.runpod.io/console/pods
2. **Click "Deploy"** or "+ GPU Pod"
3. **Select RTX 5090**
4. **Container Image**: Use RunPod's template builder
   - Click "Use Custom Template"
   - Select "Build from GitHub"
   - Connect your GitHub account
   - Select repository: `doctor-preview`
   - Dockerfile path: `runpod_service/Dockerfile`
   - Context path: `runpod_service/`

5. **Configure Pod**:
   - Container Disk: 25 GB
   - Expose Port: 8765
   - Environment Variables:
     ```
     HOST=0.0.0.0
     PORT=8765
     JPEG_QUALITY=85
     MAX_SESSIONS=10
     ```

6. **Click Deploy**

RunPod will build the image on their servers (faster, native AMD64).

---

### Option B: Use Pre-built Image (If Available)

If you want to skip building entirely, I can provide a public Docker image:

```
Container Image: facefusion/facefusion:latest
```

Then add your custom code via volume mount.

---

## Step 3: Clean Up Local Disk

Free up space on your Mac:

```bash
# Remove Docker build cache
docker system prune -a --volumes

# This will free up several GB
```

---

## Next Steps

1. **Create GitHub repo** (2 minutes)
2. **Push code** (1 minute)
3. **Deploy on RunPod** (5 minutes setup, 10 minutes build)
4. **Get endpoint URL** and test with desktop app

**Ready to proceed with GitHub?** Let me know and I'll guide you through each step!
