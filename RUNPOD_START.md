# RunPod Quick Start

Copy-paste these commands in order every time you start a new pod.

---

## 1. First-Time Setup (only needed once per pod)

```bash
# Install git if missing
apt-get update && apt-get install -y git

# Clone the repo
git clone https://github.com/nandeeswar-neuralhex/doctor-preview.git
cd doctor-preview

# Install dependencies
pip install -r runpod_v2/requirements.txt
```

---

## 2. Every Time You Start (new or restarted pod)

```bash
cd /doctor-preview

# Pull latest fixes
git fetch
git checkout nandeeswar-debug
git pull origin nandeeswar-debug

# Start the face swap server (auto-downloads models on first run)
python runpod_v2/src/server.py
```

Wait for:
```
✅ GPU warmup complete — CUDA context pre-heated
FaceSwapper ready.
Uvicorn running on http://0.0.0.0:8765
```

---

## 3. Connect Desktop App

1. Copy your RunPod public URL: `https://<POD_ID>-8765.proxy.runpod.net`
2. Open **Doctor Preview** → Settings → paste URL
3. Upload a target face photo
4. Click **Start Preview**

---

## Baseline Network Test Only (no AI, pure speed test)

```bash
cd /doctor-preview
git checkout nandeeswar-debug
python runpod_v3/src/server.py
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `libcublasLt.so not found` | Re-run `pip install -r runpod_v2/requirements.txt` |
| `File doesn't exist` (model) | Just restart `python runpod_v2/src/server.py` — it re-downloads |
| Port not reachable | Confirm port `8765` is exposed in RunPod pod settings |
| Swap not working | Check logs for `INSwapper model → CUDAExecutionProvider` |
