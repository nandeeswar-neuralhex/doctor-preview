"""
FastAPI WebSocket Server for Real-time Face Swap
Optimized for 24+ FPS streaming on GPU
"""
import asyncio
import base64
import time
import uuid
from typing import Dict, Deque
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from pydantic import BaseModel

from face_swapper import FaceSwapper
from lip_syncer import LipSyncer
from webrtc import WebRTCManager
from config import HOST, PORT, JPEG_QUALITY, MAX_SESSIONS, EXECUTION_PROVIDER, ENABLE_WEBRTC

# Initialize FastAPI app
app = FastAPI(
    title="Doctor Preview - Face Swap API",
    description="Real-time face swap service for medical preview",
    version="1.0.0"
)

# Enable CORS for desktop app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global face swapper instance (singleton)
swapper: FaceSwapper = None
lip_syncer: LipSyncer = None
webrtc_manager: WebRTCManager = None
active_connections: Dict[str, WebSocket] = {}
diagnostic_events: Deque[dict] = deque(maxlen=200)


def log_event(event: str, session_id: str | None = None, detail: str | None = None):
    diagnostic_events.append({
        "ts": time.time(),
        "event": event,
        "session_id": session_id,
        "detail": detail
    })


@app.on_event("startup")
async def startup_event():
    """Initialize the face swapper on server startup"""
    global swapper, lip_syncer, webrtc_manager
    print("=" * 50)
    print("Starting Doctor Preview Face Swap Server...")
    print("=" * 50)
    swapper = FaceSwapper()
    providers = [EXECUTION_PROVIDER, "CPUExecutionProvider"]
    lip_syncer = LipSyncer(providers)
    webrtc_manager = WebRTCManager(swapper, lip_syncer)
    log_event("startup", detail=f"provider={EXECUTION_PROVIDER}, webrtc={ENABLE_WEBRTC}")
    print("Server ready!")


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Doctor Preview Face Swap",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for RunPod / load balancers"""
    return {
        "status": "healthy",
        "model_loaded": swapper.is_ready() if swapper else False,
        "active_sessions": swapper.get_session_count() if swapper else 0,
        "max_sessions": MAX_SESSIONS,
        "webrtc_sessions": len(webrtc_manager.pcs) if webrtc_manager else 0
    }


@app.post("/upload-target")
async def upload_target_face(
    session_id: str = Query(..., description="Unique session identifier"),
    file: UploadFile = File(..., description="Target face image (JPG/PNG)")
):
    """
    Upload the target face image (post-surgery preview).
    This image's face will be swapped onto the webcam feed.
    """
    if swapper.get_session_count() >= MAX_SESSIONS:
        log_event("upload_rejected", session_id=session_id, detail="max_sessions")
        return JSONResponse(
            status_code=503,
            content={"error": "Max sessions reached", "max": MAX_SESSIONS}
        )
    
    # Read and decode image (handle EXIF orientation if present)
    contents = await file.read()
    image = None
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        log_event("upload_failed", session_id=session_id, detail="invalid_image")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image format"}
        )
    
    # Extract and store target face
    success = swapper.set_target_face(session_id, image)
    
    if not success:
        log_event("upload_failed", session_id=session_id, detail="no_face_detected")
        return JSONResponse(
            status_code=400,
            content={"error": "No face detected in image"}
        )

    log_event("upload_success", session_id=session_id)
    
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Target face set successfully. Connect to WebSocket to start streaming."
    }


@app.post("/create-session")
async def create_session():
    """Create a new session and return session ID"""
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "websocket_url": f"ws://{HOST}:{PORT}/ws/{session_id}"
    }


class WebRTCOffer(BaseModel):
    sdp: str
    type: str


class SessionSettings(BaseModel):
    enable_lipsync: bool | None = None


@app.post("/webrtc/offer")
async def webrtc_offer(session_id: str = Query(...), offer: WebRTCOffer = None):
    if not ENABLE_WEBRTC:
        log_event("webrtc_offer_rejected", session_id=session_id, detail="disabled")
        return JSONResponse(status_code=400, content={"error": "WebRTC is disabled"})
    if not offer:
        log_event("webrtc_offer_failed", session_id=session_id, detail="missing_offer")
        return JSONResponse(status_code=400, content={"error": "Missing offer"})
    log_event("webrtc_offer_received", session_id=session_id)
    answer = await webrtc_manager.handle_offer(session_id, offer.sdp, offer.type)
    log_event("webrtc_offer_answered", session_id=session_id)
    return {"sdp": answer.sdp, "type": answer.type}


@app.post("/session/settings")
async def update_session_settings(session_id: str = Query(...), settings: SessionSettings = None):
    if not settings:
        log_event("settings_failed", session_id=session_id, detail="missing_settings")
        return JSONResponse(status_code=400, content={"error": "Missing settings"})
    webrtc_manager.set_session_settings(session_id, settings.dict(exclude_none=True))
    log_event("settings_updated", session_id=session_id, detail=str(settings.dict(exclude_none=True)))
    return {"status": "success", "session_id": session_id, "settings": settings.dict(exclude_none=True)}


@app.get("/mjpeg/{session_id}")
async def mjpeg_stream(session_id: str):
    if not ENABLE_WEBRTC:
        log_event("mjpeg_rejected", session_id=session_id, detail="disabled")
        return JSONResponse(status_code=400, content={"error": "WebRTC is disabled"})

    queue = webrtc_manager.get_latest_frame_queue(session_id)
    if queue is None:
        log_event("mjpeg_failed", session_id=session_id, detail="no_active_stream")
        return JSONResponse(status_code=404, content={"error": "No active stream for session"})

    async def generator():
        while True:
            frame = await queue.get()
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session and free resources"""
    swapper.cleanup_session(session_id)
    log_event("session_deleted", session_id=session_id)
    return {"status": "success", "message": f"Session {session_id} cleaned up"}


@app.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    Real-time face swap WebSocket endpoint.
    
    Protocol:
    1. Client sends base64-encoded JPEG frame
    2. Server processes and returns base64-encoded result
    
    Expected throughput: 24+ FPS on RTX 4090/A100
    """
    await websocket.accept()
    log_event("ws_connected", session_id=session_id)
    active_connections[session_id] = websocket
    
    frame_count = 0
    start_time = time.time()
    
    print(f"WebSocket connected: {session_id}")
    
    try:
        # Check if target face is set
        if not swapper.has_target(session_id):
            await websocket.send_json({
                "error": "No target face set. Upload target first via POST /upload-target"
            })
            # Keep connection open - target might be uploaded later
        
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            # Decode input frame
            frame_bytes = base64.b64decode(data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame data"})
                continue
            
            # Ensure target is set before processing
            if not swapper.has_target(session_id):
                await websocket.send_json({
                    "error": "No target face set. Upload target first via POST /upload-target"
                })
                continue

            # Process face swap
            result = swapper.swap_face(session_id, frame)
            
            # Encode result
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', result, encode_params)
            result_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send processed frame back
            await websocket.send_text(result_b64)
            
            # FPS tracking
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Session {session_id}: {fps:.1f} FPS")
                
    except WebSocketDisconnect:
        log_event("ws_disconnected", session_id=session_id)
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        log_event("ws_error", session_id=session_id, detail=str(e))
        print(f"WebSocket error for {session_id}: {e}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]
        # Cleanup session resources on disconnect
        swapper.cleanup_session(session_id)


@app.get("/debug/status")
async def debug_status():
    """Lightweight diagnostics endpoint for end-to-end verification."""
    return {
        "status": "ok",
        "model_loaded": swapper.is_ready() if swapper else False,
        "active_sessions": swapper.get_session_count() if swapper else 0,
        "webrtc_sessions": len(webrtc_manager.pcs) if webrtc_manager else 0,
        "events": list(diagnostic_events)
    }


if __name__ == "__main__":
    print("Starting Face Swap Server...")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=30
    )
