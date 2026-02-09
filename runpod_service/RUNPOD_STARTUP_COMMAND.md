# RunPod Final Startup Command

Use this command in RunPod's **Container Start Command**:

```bash
bash -c "apt-get update && apt-get install -y git python3.10 python3-pip curl libgl1-mesa-glx libglib2.0-0 && rm -rf /workspace/app && git clone https://github.com/nandeeswar-neuralhex/doctor-preview.git /workspace/app && cd /workspace/app/runpod_service && pip install -r requirements.txt && python3 src/server.py"
```

## What This Adds

Added OpenCV dependencies:
- `libgl1-mesa-glx` - OpenGL library (fixes libGL.so.1 error)
- `libglib2.0-0` - GLib library (required by OpenCV)

## Steps

1. Stop the pod
2. Edit template
3. Replace Container Start Command with the command above
4. Deploy

## Expected Result

After ~15 minutes, you should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8765
```

Then test: `curl https://ncd79c8w7qd1rc-8765.proxy.runpod.net/health`
