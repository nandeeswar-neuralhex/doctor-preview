# Desktop App - Quick Start Guide

## Installation

```bash
cd /Users/nandeeswar/Desktop/Doctor-preview-main/desktop_app

# Install dependencies
npm install

# Start development mode
npm start
```

The app will open automatically.

## First-Time Setup

1. **Click Settings** (top right)
2. **Enter RunPod URL**: `https://your-pod-id-8765.proxy.runpod.net`
3. **Click Save**

## How to Use

### Step 1: Upload Target Images
- Click **"Upload Images"** in the left sidebar
- Select 1-10 post-surgery preview images
- Click on an image to select it as active

### Step 2: Start Preview
- Click **"Start Preview"** button
- Allow webcam access when prompted
- You'll see:
  - **Left**: Original webcam feed
  - **Right**: AI-processed preview (with face swap)

### Step 3: Show Patient
- Position patient in front of webcam
- They'll see their face swapped with the post-surgery preview in real-time
- Monitor FPS and latency in the top-right corner

### Step 4: Stop
- Click **"Stop Preview"** when done
- Upload different target image to show different results

## Building for Distribution

### macOS
```bash
npm run build:mac
```
Output: `dist/Doctor Preview.dmg`

### Windows
```bash
npm run build:win
```
Output: `dist/Doctor Preview Setup.exe`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Server URL not set" | Go to Settings and enter your RunPod URL |
| "No target image" | Upload at least one image first |
| Webcam not working | Check browser permissions |
| Low FPS | Check RunPod GPU is running |
| Connection failed | Verify RunPod URL is correct |

## System Requirements

- **macOS**: 10.13 or later
- **Windows**: Windows 10 or later
- **Webcam**: Required
- **Internet**: Required (connects to RunPod)
