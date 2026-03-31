#!/usr/bin/env bash
set -euo pipefail

sudo docker rm -f debug-parity-tmp >/dev/null 2>&1 || true
sudo docker run -d --name debug-parity-tmp --entrypoint bash doctor-preview-v2:debug-pinned-2026-04-01 -lc "sleep infinity" >/dev/null
sudo docker exec debug-parity-tmp python3 -m pip install --no-cache-dir --no-deps \
  opencv-python==4.13.0.92 \
  triton==3.6.0 \
  cuda-bindings==13.2.0 \
  cuda-pathfinder==1.5.0 \
  cuda-toolkit==13.0.2 \
  nvidia-cusparselt-cu13==0.8.0 \
  nvidia-nccl-cu13==2.28.9 \
  nvidia-nvshmem-cu13==3.4.5 \
  >/tmp/debug-parity-pip.log 2>&1
sudo docker exec debug-parity-tmp python3 -m pip uninstall -y nvidia-cublas easydict >/tmp/debug-parity-uninstall.log 2>&1 || true
sudo docker commit debug-parity-tmp doctor-preview-v2:debug-parity-2026-04-01 >/dev/null
sudo docker rm -f debug-parity-tmp >/dev/null
sudo docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep 'doctor-preview-v2:debug-parity-2026-04-01'
