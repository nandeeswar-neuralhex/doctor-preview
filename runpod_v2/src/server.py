
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
from config import JPEG_QUALITY
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
# Size 2: one active + one preparing, prevents thread explosion
_frame_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame")

# Constants
HOST = "0.0.0.0"
PORT = 8765

# Global swapper
swapper: FaceSwapper = None

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
    global swapper
    
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

        success = swapper.set_target_face(session_id, image)
        
        if not success:
            return JSONResponse(status_code=400, content={"error": "No face detected in target image"})

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Target face set successfully"
        }
    except Exception as e:
        print(f"Error in upload_target: {e}")
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

def _process_frame_binary(jpeg_bytes: bytes, session_id: str, swapper_ref):
    """
    Process a raw JPEG frame — runs in thread pool.
    Input: raw JPEG bytes (no base64, no JSON)
    Output: raw JPEG bytes (no base64, no JSON)
    Eliminates ~10ms of base64+JSON overhead per frame.
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

    # Swap
    if swapper_ref and swapper_ref.has_target(session_id):
        result = swapper_ref.swap_face(session_id, frame)
    else:
        result = frame

    t2 = time.time()

    # Encode BGR → JPEG
    if _tj:
        out_bytes = _tj.encode(result, quality=JPEG_QUALITY)
    else:
        _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        out_bytes = buffer.tobytes()

    t3 = time.time()

    h, w = result.shape[:2]
    timing = {
        "decode_ms": (t1 - t0) * 1000,
        "swap_ms": (t2 - t1) * 1000,
        "encode_ms": (t3 - t2) * 1000,
        "total_ms": (t3 - t0) * 1000,
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

    # Detect mode from first message: binary (fast) or text (legacy)
    binary_mode = None  # Will be set on first message

    try:
        while True:
            # ── Receive next frame (accept both binary and text) ──
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {session_id}")
                break

            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                if binary_mode is None:
                    binary_mode = True
                    print(f"[{session_id}] Binary mode activated — low-latency path")
                # Binary protocol: first 8 bytes = float64 timestamp, rest = JPEG
                if len(raw) < 16:
                    continue
                client_ts = struct.unpack('<d', raw[:8])[0]
                jpeg_bytes = raw[8:]

                # Process in thread pool
                try:
                    out_bytes, timing = await loop.run_in_executor(
                        _frame_pool,
                        _process_frame_binary,
                        jpeg_bytes,
                        session_id,
                        swapper
                    )
                    if out_bytes is None:
                        continue

                    # Response: 8 bytes timestamp + JPEG
                    header = struct.pack('<d', client_ts)
                    await websocket.send_bytes(header + out_bytes)

                except Exception as e:
                    print(f"Binary frame error: {e}")
                    continue

            elif "text" in msg and msg["text"]:
                data = msg["text"]
                if binary_mode is None:
                    binary_mode = False
                    print(f"[{session_id}] Text/JSON mode (legacy) — consider upgrading client for lower latency")

                # Parse JSON wrapper
                client_ts = None
                if data[0] == '{':
                    try:
                        payload = json.loads(data)
                        frame_data = payload.get("image", data)
                        client_ts = payload.get("ts")
                    except (json.JSONDecodeError, TypeError):
                        frame_data = data
                else:
                    frame_data = data

                # Process in thread pool
                try:
                    result_b64, timing = await loop.run_in_executor(
                        _frame_pool,
                        _process_frame_text,
                        frame_data,
                        session_id,
                        swapper
                    )
                    if result_b64 is None:
                        continue

                    if client_ts is not None:
                        await websocket.send_text(
                            '{"image":"' + result_b64 + '","ts":' + str(client_ts) + '}'
                        )
                    else:
                        await websocket.send_text(result_b64)

                except Exception as e:
                    print(f"Text frame error: {e}")
                    continue
            else:
                continue

            # ── Periodic logging (every 3 seconds) ──
            frame_count += 1
            now = time.time()
            elapsed = now - last_log_time
            if elapsed >= 3.0 and timing:
                fps_val = frame_count / elapsed
                mode_str = "BIN" if binary_mode else "TXT"
                print(
                    f"[{session_id}] {mode_str} FPS: {fps_val:.1f} | "
                    f"decode={timing['decode_ms']:.1f}ms "
                    f"swap={timing['swap_ms']:.1f}ms "
                    f"encode={timing['encode_ms']:.1f}ms "
                    f"total={timing['total_ms']:.1f}ms | "
                    f"Res: {timing['w']}x{timing['h']}"
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
