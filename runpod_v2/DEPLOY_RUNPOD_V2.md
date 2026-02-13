
# Deploying Simple Flip Server (Git Clone Method)

This guide explains how to deploy the simplified `runpod_v2` service by cloning the public repository directly onto a RunPod instance.

## Step 1: Start a Standard Pod

1. **Go to RunPod Console**.
2. **Deploy Pod**.
3. Choose a **standard template** (e.g., PyTorch 2.1, Python 3.10).
   - *Any template with Python installed will work.*
   - If unsure, use the **RunPod PyTorch 2.1** template.
4. **Customize Deployment**:
   - Limit the **Container Disk** to 10GB (saves money, enough for this).
   - Ensure you add **Port 8765** (TCP) to the exposed ports list.
   - Ensure **HTTP Proxy** (or "Expose Public API") is enabled for this port.
5. **Start Pod**.

## Step 2: Clone & Run (GPU Integration)

Once the Pod is **Running**:

1. Click **Connect** > **Start Web Terminal**.
2. Run the following commands:

```bash
# 1. Clone the repository
git clone https://github.com/nandeeswar-neuralhex/doctor-preview.git

# 2. Enter the simple service directory
cd doctor-preview/runpod_v2

# 3. Install dependencies (now includes CUDA 12 support)
pip install -r requirements.txt

# 4. Start the server
# (This will automatically download the 500MB+ models on first run)
python src/server.py
```
*You should see "FaceSwapper ready" and "Uvicorn running on http://0.0.0.0:8765"*


## Step 3: Connect Desktop App

1. Copy the **Public URL** for port `8765` from the RunPod dashboard.
   - Format: `https://<POD_ID>-8765.proxy.runpod.net`
2. Open **Doctor Preview Desktop App**.
3. Go to **Settings**.
4. Paste the URL.
5. Upload dummy image.
6. Click **Start Preview**.

## Troubleshooting

- **`libcublasLt.so.11: cannot open shared object file`**: 
  - This means you have the wrong `onnxruntime-gpu` version. 
  - Run Step 4 in "Clone & Run" to install the CUDA 12 compatible version.

- **`INVALID_PROTOBUF` or "File doesn't exist"**:
  - The model file is corrupted (likely a partial download).
  - Restart the server (`python src/server.py`). The script is designed to auto-detect corrupted files, delete them, and re-download them.

- **Git not found**: Run `apt-get update && apt-get install -y git`.
- **Repo not found**: Ensure `https://github.com/nandeeswar-neuralhex/doctor-preview.git` is accessible.
- **Port not reachable**: Double check you added port `8765` in the Pod configuration.
