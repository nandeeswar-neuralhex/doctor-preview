
import asyncio
import base64
import struct
import time
import uuid
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

import io
from PIL import Image, ImageOps
from face_swapper import FaceSwapper
from lip_syncer import LipSyncer
from config import JPEG_QUALITY, EXECUTION_PROVIDER, ENABLE_LIPSYNC
from download_models import download_models

# ── TurboJPEG: 3-5x faster JPEG encode/decode than cv2 ──
try:
    from turbojpeg import TurboJPEG, TJPF_BGR
    _tj = TurboJPEG()
    print("✅ TurboJPEG available — using hardware-accelerated JPEG codec")
except Exception:
    _tj = None
    print("ℹ️  TurboJPEG not available — using cv2 JPEG codec (install PyTurboJPEG for 3x speedup)")

# ── Thread pools: separate CPU and GPU work for pipelining ──
# GPU pool: 2 workers — 1 active on GPU, 1 preparing next frame
# More workers cause GPU contention (ONNX serializes GPU calls internally)
_gpu_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu")
# CPU pool: JPEG decode/encode — runs in parallel with GPU work
_cpu_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu")

# Constants
HOST = "0.0.0.0"
PORT = 8765

# Global swapper and lip syncer
swapper: FaceSwapper = None
lip_syncer: LipSyncer = None

app = FastAPI(title="Simple Flip Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global swapper, lip_syncer
    
    print("Checking models...")
    try:
        download_models()
    except Exception as e:
        print(f"Auto-download warning: {e}")

    print("Initializing FaceSwapper...")
    try:
        swapper = FaceSwapper()
        print("FaceSwapper ready.")
    except Exception as e:
        print(f"FaceSwapper initialization failed: {e}")
        swapper = None

    # Initialize lip syncer for real-time lip sync
    if ENABLE_LIPSYNC:
        try:
            providers = [EXECUTION_PROVIDER, "CPUExecutionProvider"]
            lip_syncer = LipSyncer(providers)
            if lip_syncer.is_ready():
                print("LipSyncer ready.")
            else:
                print("LipSyncer disabled (model not available or librosa missing).")
                lip_syncer = None
        except Exception as e:
            print(f"LipSyncer initialization failed: {e}")
            lip_syncer = None
    else:
        print("LipSyncer disabled by config.")

@app.get("/health")
async def health_check():
    gpu = swapper.gpu_status() if swapper else {}
    return {"status": "healthy", "mode": "simple-flip", "gpu_active": gpu.get("gpu_active", False)}

@app.get("/")
async def root():
    return {"service": "Simple Flip Server", "status": "running"}

@app.get("/debug/gpu")
async def debug_gpu():
    """Runtime GPU diagnostics — call this to verify GPU is being used."""
    if swapper is None:
        return {"error": "FaceSwapper not initialized"}
    return swapper.gpu_status()

@app.post("/create-session")
async def create_session():
    import uuid
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "websocket_url": f"ws://{HOST}:{PORT}/ws/{session_id}"
    }

@app.post("/upload-target")
async def upload_target(
    session_id: str = Query(...),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        image = None
        try:
            # Try PIL first for EXIF orientation
            pil_image = Image.open(io.BytesIO(contents))
            pil_image = ImageOps.exif_transpose(pil_image)
            pil_image = pil_image.convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            # Fallback to direct decode
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        if swapper is None:
             return JSONResponse(status_code=500, content={"error": "FaceSwapper not initialized"})

        # Clear previous faces and set new one
        if session_id in swapper.target_faces:
            del swapper.target_faces[session_id]
        success = swapper.set_target_face(session_id, image)
        
        if not success:
            return JSONResponse(status_code=400, content={"error": "No face detected in target image"})

        return {
            "status": "success",
            "session_id": session_id,
            "faces_stored": len(swapper.target_faces.get(session_id, [])),
            "message": "Target face set successfully"
        }
    except Exception as e:
        print(f"Error in upload_target: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload-targets")
async def upload_targets(
    session_id: str = Query(...),
    files: list[UploadFile] = File(...)
):
    """Upload multiple target images for expression matching.
    Each image should show a different expression (smile, neutral, open mouth, etc.).
    The server will match the webcam expression to the closest target.
    """
    try:
        if swapper is None:
            return JSONResponse(status_code=500, content={"error": "FaceSwapper not initialized"})

        if len(files) > 10:
            return JSONResponse(status_code=400, content={"error": "Maximum 10 images allowed"})

        images = []
        for f in files:
            contents = await f.read()
            image = None
            try:
                pil_image = Image.open(io.BytesIO(contents))
                pil_image = ImageOps.exif_transpose(pil_image)
                pil_image = pil_image.convert("RGB")
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception:
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)

        if not images:
            return JSONResponse(status_code=400, content={"error": "No valid images found"})

        result = swapper.set_target_faces(session_id, images)

        if not result["success"]:
            return JSONResponse(status_code=400, content={"error": result["message"]})

        return {
            "status": "success",
            "session_id": session_id,
            "faces_stored": result["count"],
            "total_uploaded": result["total"],
            "message": f"{result['count']}/{result['total']} faces extracted for expression matching"
        }
    except Exception as e:
        print(f"Error in upload_targets: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/session/settings")
async def update_settings(session_id: str = Query(...)):
    # Dummy endpoint to satisfy client
    return {"status": "success", "message": "Settings ignored"}

@app.post("/webrtc/offer")
async def webrtc_offer(session_id: str = Query(...)):
    # Force fallback to WebSocket by rejecting WebRTC
    return JSONResponse(
        status_code=400, 
        content={"error": "WebRTC disabled in simple mode. Use WebSocket."}
    )

# ── CODEC helpers (CPU-bound, run in _cpu_pool) ──

def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG to BGR numpy array — CPU only. Returns None on failure."""
    try:
        if _tj:
            return _tj.decode(jpeg_bytes, pixel_format=TJPF_BGR)
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode BGR numpy array to JPEG bytes — CPU only. Returns empty bytes on failure."""
    try:
        if frame is None or frame.size == 0:
            return b''
        # Ensure correct dtype — GPU can occasionally produce float32
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if _tj:
            return _tj.encode(frame, quality=JPEG_QUALITY)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes()
    except Exception:
        return b''


# ── GPU helper (runs in _gpu_pool) ──

def _gpu_swap(frame: np.ndarray, session_id: str, swapper_ref, lip_syncer_ref,
             audio_pcm: bytes, audio_sr: int):
    """GPU-bound: face detection + swap + optional lip sync. No JPEG work."""
    source_faces = []
    if swapper_ref and swapper_ref.has_target(session_id):
        result, source_faces = swapper_ref.swap_face_with_faces(session_id, frame)
    else:
        result = frame

    # Lip sync (GPU)
    if (lip_syncer_ref and lip_syncer_ref.is_ready()
            and audio_pcm and len(audio_pcm) > 0 and len(source_faces) > 0):
        try:
            mel = lip_syncer_ref.audio_to_mel(audio_pcm, audio_sr)
            if mel is not None:
                # Use the largest detected/cached face for bbox
                face = max(source_faces,
                           key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                # Clamp bbox to frame bounds to prevent out-of-bounds crop
                h_frame, w_frame = result.shape[:2]
                bbox = np.clip(face.bbox.astype(int), 0, [w_frame, h_frame, w_frame, h_frame])
                x1, y1 = max(0, bbox[0]), max(0, bbox[1])
                x2 = min(result.shape[1], bbox[2])
                y2 = min(result.shape[0], bbox[3])
                if x2 > x1 and y2 > y1:
                    face_crop = result[y1:y2, x1:x2]
                    synced = lip_syncer_ref.infer(face_crop, mel)
                    if synced is not None:
                        result = lip_syncer_ref.apply_mouth_only(
                            result, (x1, y1, x2, y2), synced)
        except Exception:
            pass

    return result


def _process_frame_binary(jpeg_bytes: bytes, audio_pcm: bytes, audio_sr: int,
                          session_id: str, swapper_ref, lip_syncer_ref):
    """
    Legacy: full pipeline in one call (kept for fallback).
    """
    t0 = time.time()
    frame = _decode_jpeg(jpeg_bytes)
    if frame is None:
        return None, None
    t1 = time.time()
    result = _gpu_swap(frame, session_id, swapper_ref, lip_syncer_ref, audio_pcm, audio_sr)
    t2 = time.time()
    out_bytes = _encode_jpeg(result)
    t3 = time.time()
    h, w = result.shape[:2]
    timing = {
        "decode_ms": (t1 - t0) * 1000,
        "swap_ms": (t2 - t1) * 1000,
        "lipsync_ms": 0,
        "encode_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
        "w": w, "h": h,
    }
    return out_bytes, timing


def _process_frame_text(frame_data: str, session_id: str, swapper_ref):
    """Legacy text-mode processor for backward compatibility."""
    t0 = time.time()
    frame_bytes = base64.b64decode(frame_data)
    frame = _decode_jpeg(frame_bytes)
    if frame is None:
        return None, None
    t1 = time.time()
    if swapper_ref and swapper_ref.has_target(session_id):
        result = swapper_ref.swap_face(session_id, frame)
    else:
        result = frame
    t2 = time.time()
    out_bytes = _encode_jpeg(result)
    result_b64 = base64.b64encode(out_bytes).decode('ascii')
    t3 = time.time()
    h, w = result.shape[:2]
    timing = {
        "decode_ms": (t1 - t0) * 1000,
        "swap_ms": (t2 - t1) * 1000,
        "encode_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
        "w": w, "h": h,
    }
    return result_b64, timing


@app.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"WebSocket connected: {session_id}")

    import json
    loop = asyncio.get_running_loop()
    frame_count = 0
    processed_count = 0
    dropped_count = 0
    last_log_time = time.time()
    last_timing = None

    # ── Bounded concurrency: 2 in-flight max ──
    # Frame N on GPU while Frame N+1 decodes on CPU = true pipeline
    # If both slots full → drop incoming frame (it's already stale)
    _sem = asyncio.Semaphore(2)
    _latest_frame_id = 0
    _send_lock = asyncio.Lock()

    async def process_binary_pipelined(raw_data: bytes):
        """3-stage pipeline: decode(CPU) → swap(GPU) → encode(CPU) with overlap."""
        nonlocal last_timing, _latest_frame_id, processed_count, dropped_count
        if len(raw_data) < 21:
            return

        frame_id_bytes = raw_data[:4]
        ts_bytes = raw_data[4:12]
        audio_len = struct.unpack('<I', raw_data[12:16])[0]
        audio_sr = struct.unpack('<I', raw_data[16:20])[0]
        audio_pcm = raw_data[20:20+audio_len] if audio_len > 0 else b''
        jpeg_bytes = raw_data[20+audio_len:]

        frame_id = struct.unpack('<I', frame_id_bytes)[0]

        # Drop stale frames
        if frame_id < _latest_frame_id:
            dropped_count += 1
            return
        _latest_frame_id = frame_id

        # Drop if pipeline is full (both slots taken)
        if _sem.locked():
            dropped_count += 1
            return

        async with _sem:
            t0 = time.time()

            # Stage 1: JPEG decode on CPU pool (overlaps with previous frame's GPU work)
            frame = await loop.run_in_executor(_cpu_pool, _decode_jpeg, jpeg_bytes)
            if frame is None:
                return
            t1 = time.time()

            # Stage 2: Face swap + lip sync on GPU pool
            result = await loop.run_in_executor(
                _gpu_pool, _gpu_swap,
                frame, session_id, swapper, lip_syncer, audio_pcm, audio_sr)
            t2 = time.time()

            # Stage 3: JPEG encode on CPU pool (overlaps with next frame's GPU work)
            out_bytes = await loop.run_in_executor(_cpu_pool, _encode_jpeg, result)
            t3 = time.time()

            if not out_bytes:
                return  # Encode failed — skip this frame

            h, w = result.shape[:2]
            last_timing = {
                "decode_ms": (t1 - t0) * 1000,
                "swap_ms": (t2 - t1) * 1000,
                "encode_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000,
                "w": w, "h": h,
            }
            processed_count += 1

            try:
                async with _send_lock:
                    await websocket.send_bytes(frame_id_bytes + ts_bytes + out_bytes)
            except Exception:
                return

    async def process_and_respond_text(data: str, client_ts):
        """Process a text/JSON frame and send result back."""
        nonlocal last_timing, processed_count
        frame_data = data
        if data[0] == '{':
            try:
                payload = json.loads(data)
                frame_data = payload.get("image", data)
                client_ts = payload.get("ts", client_ts)
            except (json.JSONDecodeError, TypeError):
                pass

        result_b64, timing = await loop.run_in_executor(
            _gpu_pool,
            _process_frame_text,
            frame_data,
            session_id,
            swapper
        )
        if result_b64 is None:
            return
        last_timing = timing
        processed_count += 1

        try:
            async with _send_lock:
                if client_ts is not None:
                    await websocket.send_text(
                        '{"image":"' + result_b64 + '","ts":' + str(client_ts) + '}'
                    )
                else:
                    await websocket.send_text(result_b64)
        except Exception:
            return

    try:
        while True:
            try:
                msg = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                print(f"WebSocket disconnected: {session_id} (Reason: {e})")
                break

            if "bytes" in msg and msg["bytes"]:
                asyncio.ensure_future(process_binary_pipelined(msg["bytes"]))
            elif "text" in msg and msg["text"]:
                asyncio.ensure_future(process_and_respond_text(msg["text"], None))
            else:
                continue

            frame_count += 1
            now = time.time()
            elapsed = now - last_log_time
            if elapsed >= 3.0:
                fps_in = frame_count / elapsed
                fps_out = processed_count / elapsed
                timing_str = (
                    f"decode={last_timing['decode_ms']:.1f}ms "
                    f"swap={last_timing['swap_ms']:.1f}ms "
                    f"encode={last_timing['encode_ms']:.1f}ms "
                    f"total={last_timing['total_ms']:.1f}ms | "
                    f"Res: {last_timing['w']}x{last_timing['h']}"
                ) if last_timing else "no timing yet"
                print(f"[{session_id}] IN={fps_in:.1f} OUT={fps_out:.1f} FPS | "
                      f"dropped={dropped_count} | {timing_str}")
                frame_count = 0
                processed_count = 0
                dropped_count = 0
                last_log_time = now

    except Exception as e:
        print(f"WebSocket error: {e}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if swapper:
        swapper.cleanup_session(session_id)
    print(f"Session {session_id} cleaned up")
    return {"status": "success", "message": "Session cleaned up"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
