
import asyncio
import base64
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

def _process_frame_sync(frame_data: str, session_id: str, swapper_ref):
    """
    CPU-bound frame processing — runs in thread pool to free the asyncio loop.
    Handles: base64 decode → cv2 decode → face swap → cv2 encode → base64 encode
    """
    t0 = time.time()

    # Decode: base64 → numpy → BGR
    frame_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(frame_bytes, np.uint8)
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

    # Encode: BGR → JPEG → base64
    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    result_b64 = base64.b64encode(buffer).decode('ascii')  # ascii is faster than utf-8 for b64

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

    # Latest incoming frame — older frames get dropped automatically
    latest_frame_data = None
    latest_client_ts = None
    processing = False  # True while a frame is being processed in the thread pool

    try:
        while True:
            # ── Receive next frame ──
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {session_id}")
                break

            # ── Parse (fast path: check for JSON wrapper) ──
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

            # ── Frame dropping: always keep only the latest frame ──
            # If we're still processing the previous frame, just overwrite the buffer.
            # This prevents queue buildup and latency spiral.
            latest_frame_data = frame_data
            latest_client_ts = client_ts

            if processing:
                # Drop this frame — the processor will pick up latest_frame_data when done
                continue

            # ── Process frames in a loop until buffer is drained ──
            while latest_frame_data is not None:
                current_data = latest_frame_data
                current_ts = latest_client_ts
                latest_frame_data = None  # Clear buffer
                latest_client_ts = None
                processing = True

                try:
                    # Offload ALL CPU work to thread pool
                    result_b64, timing = await loop.run_in_executor(
                        _frame_pool,
                        _process_frame_sync,
                        current_data,
                        session_id,
                        swapper
                    )

                    if result_b64 is None:
                        processing = False
                        continue

                    # Send response (async — doesn't block)
                    if current_ts is not None:
                        await websocket.send_text(
                            '{"image":"' + result_b64 + '","ts":' + str(current_ts) + '}'
                        )
                    else:
                        await websocket.send_text(result_b64)

                    # ── Periodic logging (every 3 seconds instead of every frame) ──
                    frame_count += 1
                    now = time.time()
                    elapsed = now - last_log_time
                    if elapsed >= 3.0:
                        fps = frame_count / elapsed
                        print(
                            f"[{session_id}] FPS: {fps:.1f} | "
                            f"Last: decode={timing['decode_ms']:.1f}ms "
                            f"swap={timing['swap_ms']:.1f}ms "
                            f"encode={timing['encode_ms']:.1f}ms "
                            f"total={timing['total_ms']:.1f}ms | "
                            f"Res: {timing['w']}x{timing['h']}"
                        )
                        frame_count = 0
                        last_log_time = now

                except Exception as e:
                    print(f"Frame processing error: {e}")
                finally:
                    processing = False

                # Check if a new frame arrived while we were processing
                # (latest_frame_data may have been set by a concurrent receive)
                # We need to drain any pending messages before looping
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    if data[0] == '{':
                        try:
                            payload = json.loads(data)
                            latest_frame_data = payload.get("image", data)
                            latest_client_ts = payload.get("ts")
                        except (json.JSONDecodeError, TypeError):
                            latest_frame_data = data
                    else:
                        latest_frame_data = data
                except asyncio.TimeoutError:
                    break  # No more pending frames, go back to blocking receive
                except WebSocketDisconnect:
                    print(f"WebSocket disconnected: {session_id}")
                    return

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
