"""
Mock Server for Testing Desktop App
Applies horizontal flip to webcam feed to show processing
No AI processing - for UI/UX testing only
"""
import asyncio
import base64
import time
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Doctor Preview - Mock Server",
    description="Mock server that echoes webcam feed for testing",
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

# Store uploaded images per session (just for mock purposes)
uploaded_images: Dict[str, bool] = {}
active_connections: Dict[str, WebSocket] = {}


@app.on_event("startup")
async def startup_event():
    """Startup message"""
    print("=" * 50)
    print("🎭 Mock Server Started - No AI Processing")
    print("Applying HORIZONTAL FLIP to webcam feed")
    print("=" * 50)


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Doctor Preview Mock Server",
        "version": "1.0.0",
        "status": "running",
        "mode": "mock",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,  # Fake it
        "active_sessions": len(active_connections),
        "max_sessions": 999,
        "mode": "mock"
    }


@app.post("/upload-target")
async def upload_target_face(
    session_id: str = Query(..., description="Unique session identifier"),
    file: UploadFile = File(..., description="Target face image (JPG/PNG)")
):
    """
    Mock upload - just stores that an image was uploaded
    """
    # Just mark that this session has an image
    uploaded_images[session_id] = True
    
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Mock: Target face received (not actually processing)",
        "mode": "mock"
    }


@app.post("/create-session")
async def create_session():
    """Create a new session and return session ID"""
    import uuid
    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "websocket_url": f"ws://localhost:8765/ws/{session_id}"
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session"""
    if session_id in uploaded_images:
        del uploaded_images[session_id]
    return {"status": "success", "message": f"Session {session_id} cleaned up"}


@app.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    Mock WebSocket - just echoes back the frame with 100ms delay
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    
    frame_count = 0
    start_time = time.time()
    
    print(f"🔌 WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            # Decode the base64 image
            frame_bytes = base64.b64decode(data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Apply horizontal flip (mirror effect)
                flipped_frame = cv2.flip(frame, 1)
                
                # Encode back to base64
                _, buffer = cv2.imencode('.jpg', flipped_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                result_b64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(result_b64)
            else:
                # If decode fails, just echo back
                await websocket.send_text(data)
            
            # FPS tracking
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"📊 Session {session_id}: {fps:.1f} FPS (mock mode)")
                
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error for {session_id}: {e}")
    finally:
        if session_id in active_connections:
            del active_connections[session_id]


if __name__ == "__main__":
    print("🚀 Starting Mock Face Swap Server...")
    print("⚠️  This is a MOCK server - no AI processing!")
    print("📹 It will just echo back your webcam feed")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=30
    )
