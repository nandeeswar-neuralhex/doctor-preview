
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

# Thread pool for CPU-bound frame processing (decode/encode/swap)
# Size 8: RTX 6000 server has 16 vCPU, use more workers to distribute CPU load
_frame_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="frame")

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

def _process_frame_binary(jpeg_bytes: bytes, audio_pcm: bytes, audio_sr: int,
                          session_id: str, swapper_ref, lip_syncer_ref):
    """
    Process a raw JPEG frame with optional audio for lip sync.
    Input: raw JPEG bytes + PCM int16 audio
    Output: raw JPEG bytes
    Runs in thread pool for concurrency.
    """
    t0 = time.time()

    # Decode JPEG → BGR
    if _tj:
        frame = _tj.decode(jpeg_bytes, pixel_format=TJPF_BGR)
    else:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None

    t1 = time.time()

    # Face swap + get detected faces for lip sync
    source_faces = []
    if swapper_ref and swapper_ref.has_target(session_id):
        result, source_faces = swapper_ref.swap_face_with_faces(session_id, frame)
    else:
        result = frame

    t2 = time.time()

    # Lip sync: apply Wav2Lip mouth correction using audio
    t_lip = 0
    if (lip_syncer_ref and lip_syncer_ref.is_ready()
            and audio_pcm and len(audio_pcm) > 0 and len(source_faces) > 0):
        try:
            mel = lip_syncer_ref.audio_to_mel(audio_pcm, audio_sr)
            if mel is not None:
                # Use the largest detected face
                face = max(source_faces,
                           key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(result.shape[1], x2)
                y2 = min(result.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    face_crop = result[y1:y2, x1:x2]
                    synced = lip_syncer_ref.infer(face_crop, mel)
                    if synced is not None:
                        result = lip_syncer_ref.apply_mouth_only(
                            result, (x1, y1, x2, y2), synced
                        )
        except Exception as e:
            pass  # Don't break frame pipeline on lip sync error
        t_lip = (time.time() - t2) * 1000

    t3 = time.time()

    # Encode BGR → JPEG
    if _tj:
        out_bytes = _tj.encode(result, quality=JPEG_QUALITY)
    else:
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        out_bytes = buffer.tobytes()

    t4 = time.time()

    h, w = result.shape[:2]
    timing = {
        "decode_ms": (t1 - t0) * 1000,
        "swap_ms": (t2 - t1) * 1000,
        "lipsync_ms": t_lip,
        "encode_ms": (t4 - t3) * 1000,
        "total_ms": (t4 - t0) * 1000,
        "w": w, "h": h,
    }
    return out_bytes, timing


def _process_frame_text(frame_data: str, session_id: str, swapper_ref):
    """
    Legacy text-mode processor for backward compatibility.
    Handles: base64 decode → cv2 decode → face swap → cv2 encode → base64 encode
    """
    t0 = time.time()
    frame_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None

    t1 = time.time()
    if swapper_ref and swapper_ref.has_target(session_id):
        result = swapper_ref.swap_face(session_id, frame)
    else:
        result = frame

    t2 = time.time()
    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    result_b64 = base64.b64encode(buffer).decode('ascii')

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
    last_log_time = time.time()
    last_timing = None

    # Pipelined processing with bounded concurrency:
    # - We receive frames continuously (client sends at 20fps)
    # - Process up to 5 concurrently (matches thread pool + GPU)
    # - Drop frames if we fall behind (latest-frame-wins)
    # - This hides the RTT latency: output FPS ≈ server throughput
    _sem = asyncio.Semaphore(5)  # Max 5 frames in-flight (was 3)
    _sem = asyncio.Semaphore(5)  # Max 5 frames in-flight (was 3)
    _latest_frame_id = 0         # Track latest to drop stale frames
    _send_lock = asyncio.Lock()  # Prevent concurrent writes to WebSocket

    async def process_and_respond_binary(raw_data: bytes):
        """Process a binary frame (with optional audio) and send result back."""
        nonlocal last_timing, _latest_frame_id
        if len(raw_data) < 21:  # 20-byte header + at least 1 byte JPEG
            return
        # Header: 4B frameId + 8B timestamp + 4B audioLen + 4B sampleRate + audio + JPEG
        frame_id_bytes = raw_data[:4]
        ts_bytes = raw_data[4:12]
        audio_len = struct.unpack('<I', raw_data[12:16])[0]
        audio_sr = struct.unpack('<I', raw_data[16:20])[0]
        audio_pcm = raw_data[20:20+audio_len] if audio_len > 0 else b''
        jpeg_bytes = raw_data[20+audio_len:]

        frame_id = struct.unpack('<I', frame_id_bytes)[0]

        # Drop if a newer frame is already queued
        if frame_id < _latest_frame_id:
            return
        _latest_frame_id = frame_id

        # Bounded concurrency — if all slots full, skip this frame
        if _sem.locked():
            return

        async with _sem:
            out_bytes, timing = await loop.run_in_executor(
                _frame_pool,
                _process_frame_binary,
                jpeg_bytes,
                audio_pcm,
                audio_sr,
                session_id,
                swapper,
                lip_syncer
            )
            if out_bytes is None:
                return
            last_timing = timing

            # Response: [4 bytes frameId] + [8 bytes timestamp] + JPEG
            # Response: [4 bytes frameId] + [8 bytes timestamp] + JPEG
            try:
                async with _send_lock:
                    await websocket.send_bytes(frame_id_bytes + ts_bytes + out_bytes)
            except Exception:
                pass  # WebSocket may have closed

    async def process_and_respond_text(data: str, client_ts):
        """Process a text/JSON frame and send result back."""
        nonlocal last_timing
        # Parse frame data
        frame_data = data
        if data[0] == '{':
            try:
                payload = json.loads(data)
                frame_data = payload.get("image", data)
                client_ts = payload.get("ts", client_ts)
            except (json.JSONDecodeError, TypeError):
                pass

        result_b64, timing = await loop.run_in_executor(
            _frame_pool,
            _process_frame_text,
            frame_data,
            session_id,
            swapper
        )
        if result_b64 is None:
            return
        last_timing = timing

        if client_ts is not None:
            async with _send_lock:
                await websocket.send_text(
                    '{"image":"' + result_b64 + '","ts":' + str(client_ts) + '}'
                )
        else:
            async with _send_lock:
                await websocket.send_text(result_b64)

    try:
        while True:
            try:
                msg = await websocket.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                print(f"WebSocket disconnected: {session_id} (Reason: {e})")
                break

            if "bytes" in msg and msg["bytes"]:
                # Fire-and-forget: start processing, don't await before receiving next
                asyncio.ensure_future(process_and_respond_binary(msg["bytes"]))

            elif "text" in msg and msg["text"]:
                asyncio.ensure_future(process_and_respond_text(msg["text"], None))
            else:
                continue

            # Periodic logging
            frame_count += 1
            now = time.time()
            elapsed = now - last_log_time
            if elapsed >= 3.0:
                fps_val = frame_count / elapsed
                print(
                    f"[{session_id}] FPS: {fps_val:.1f} | "
                    + (f"decode={last_timing['decode_ms']:.1f}ms "
                       f"swap={last_timing['swap_ms']:.1f}ms "
                       f"lip={last_timing.get('lipsync_ms', 0):.1f}ms "
                       f"encode={last_timing['encode_ms']:.1f}ms "
                       f"total={last_timing['total_ms']:.1f}ms | "
                       f"Res: {last_timing['w']}x{last_timing['h']}"
                       if last_timing else "no timing yet")
                )
                frame_count = 0
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
