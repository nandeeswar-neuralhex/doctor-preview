# Performance Optimization Guide

## Current Bottleneck: CPU (JPEG operations)

The RTX 6000 GPU is only at 13-26% utilization because **CPU is bottlenecked at 100%** doing JPEG decode/encode.

## Quick Fix: Install TurboJPEG

TurboJPEG gives **3x faster** JPEG operations on CPU.

### On your RunPod server, run:

```bash
# Install TurboJPEG library
apt-get update && apt-get install -y libturbojpeg0-dev

# Install Python wrapper
pip install PyTurboJPEG

# Restart server
python3 src/server.py
```

### Verification

After restart, you should see:
```
✅ TurboJPEG available — using hardware-accelerated JPEG codec
```

Instead of:
```
ℹ️  TurboJPEG not available — using cv2 JPEG codec
```

## Expected Results After TurboJPEG

- ✅ CPU usage: 40-60% (down from 100%)
- ✅ GPU usage: 30-50% (up from 13-26%)
- ✅ FPS: 25-30 (stable, up from 13-22)
- ✅ Latency: 250-400ms (stable, down from 400-2000ms)

## Alternative: Reduce Resolution

If you can't install TurboJPEG, reduce to 640p instead:
- Edit `desktop_app/src/components/CameraView.jsx`
- Change `MAX_WIDTH = 720` to `MAX_WIDTH = 640`
