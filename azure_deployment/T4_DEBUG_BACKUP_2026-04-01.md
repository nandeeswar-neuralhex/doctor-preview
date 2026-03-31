# T4 Debug Backup 2026-04-01

This backup point is based on the live `doctor-preview` container running on `faceiq-debug.sparkiq.ai` (`20.127.128.253`) on 2026-04-01.

Verified source commit:

- Branch: `t4-debug-backup-2026-04-01`
- Commit: `8396272884e2bf70079050665eda2e64634d76c3`
- Original message: `add some changes`

Verified runtime facts from the live Debug container:

- Image id: `sha256:01221bf0ea93d8b291fcca9f81d12a32263b4dddb7426dca57008fd626c1d3ef`
- Entrypoint command: `python3 -m uvicorn server:app --host 0.0.0.0 --port 8765 --workers 1`
- App env: `EXECUTION_PROVIDER=CUDAExecutionProvider`
- App env: `ENABLE_WEBRTC=true`
- App env: `ENABLE_LIPSYNC=true`
- Effective config: `JPEG_QUALITY=90`
- Effective config: `ENABLE_SEAMLESS_CLONE=true`
- Effective config: `ENABLE_GFPGAN=true`
- Effective config: `ENABLE_TEMPORAL_SMOOTHING=true`
- Effective config: `FACE_MASK_BLUR=25`
- Effective config: `FACE_MASK_SCALE=1.1`
- Effective config: `SMOOTHING_ALPHA=0.4`
- Effective config: `MAX_FACES=1`
- Effective config: `TARGET_FPS=24`

The script `azure_deployment/deploy_t4_debug_backup.sh` restores this code and runtime configuration on another GPU VM.