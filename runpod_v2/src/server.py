
import asyncio
import base64
import time
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Constants
HOST = "0.0.0.0"
PORT = 8765
JPEG_QUALITY = 85

app = FastAPI(title="Simple Flip Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "simple-flip"}

@app.get("/")
async def root():
    return {"service": "Simple Flip Server", "status": "running"}

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
    # We don't actually need the target for a simple flip, 
    # but we accept it to satisfy the client's protocol.
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Target received (ignored in simple mode)"
    }

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

@app.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive frame
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {session_id}")
                break
                
            # Decode
            try:
                frame_bytes = base64.b64decode(data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # TRANSFORM: Flip horizontally
                # 1 = Horizontal flip
                result = cv2.flip(frame, 1)
                
                # Encode
                _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                result_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send back
                await websocket.send_text(result_b64)
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
                
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    print(f"Session {session_id} cleaned up")
    return {"status": "success", "message": "Session cleaned up"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
