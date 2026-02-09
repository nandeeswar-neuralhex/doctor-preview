"""
FastAPI WebSocket Server for Real-time Face Swap
Optimized for 24+ FPS streaming on GPU
"""
import asyncio
import base64
import time
import uuid
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from face_swapper import FaceSwapper
from config import HOST, PORT, JPEG_QUALITY, MAX_SESSIONS

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
active_connections: Dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the face swapper on server startup"""
    global swapper
    print("=" * 50)
    print("Starting Doctor Preview Face Swap Server...")
    print("=" * 50)
    swapper = FaceSwapper()
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
        "max_sessions": MAX_SESSIONS
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
        return JSONResponse(
            status_code=503,
            content={"error": "Max sessions reached", "max": MAX_SESSIONS}
        )
    
    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image format"}
        )
    
    # Extract and store target face
    success = swapper.set_target_face(session_id, image)
    
    if not success:
        return JSONResponse(
            status_code=400,
            content={"error": "No face detected in image"}
        )
    
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


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session and free resources"""
    swapper.cleanup_session(session_id)
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
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"WebSocket error for {session_id}: {e}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]
        # Don't cleanup session automatically - user might reconnect


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
