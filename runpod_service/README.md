# FaceFusion RunPod Service

Real-time face swap service using FaceFusion, deployed on RunPod with GPU.

## Quick Start

### Local Testing (requires NVIDIA GPU)
```bash
docker build -t doctor-preview:latest .
docker run --gpus all -p 8765:8765 doctor-preview:latest
```

### Deploy to RunPod
1. Push to Docker Hub: `docker push your-username/doctor-preview:latest`
2. Create RunPod template with the image
3. Start a GPU pod (RTX 4090 or A100 recommended)

## API

- `GET /health` - Health check
- `POST /upload-target?session_id=xxx` - Upload target face image
- `WS /ws/{session_id}` - Real-time face swap stream

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JPEG_QUALITY` | 85 | Output JPEG quality (1-100) |
| `MAX_SESSIONS` | 10 | Max concurrent sessions |
