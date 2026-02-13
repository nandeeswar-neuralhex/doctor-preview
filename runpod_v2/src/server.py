
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
JPEG_QUALITY = 60

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
    
    import json
    try:
        while True:
            t0 = time.time()
            
            # Receive data
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {session_id}")
                break
            
            t1 = time.time() # Receive done

            # Parse input (JSON or raw base64)
            try:
                payload = json.loads(data)
                if isinstance(payload, dict) and "image" in payload:
                    frame_data = payload["image"]
                    client_ts = payload.get("ts")
                else:
                    frame_data = data
                    client_ts = None
            except json.JSONDecodeError:
                frame_data = data
                client_ts = None
            
            t2 = time.time() # Parse done

            # Decode image
            try:
                frame_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                t3 = time.time() # Decode done
                
                # TRANSFORM: Flip horizontally
                result = cv2.flip(frame, 1)

                t4 = time.time() # Process done
                
                # Encode response
                # Resize if HUGE (optional, let's log size first)
                h, w, _ = result.shape
                
                _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                result_b64 = base64.b64encode(buffer).decode('utf-8')
                
                t5 = time.time() # Encode done
                
                # Send back (JSON if timestamp exists, else raw)
                if client_ts:
                    response = json.dumps({
                        "image": result_b64,
                        "ts": client_ts
                    })
                    await websocket.send_text(response)
                else:
                    await websocket.send_text(result_b64)
                
                t6 = time.time() # Send done

                # Log performance
                total_server_time = (t6 - t1) * 1000
                decode_ms = (t3 - t2) * 1000
                proc_ms = (t4 - t3) * 1000
                encode_ms = (t5 - t4) * 1000
                recv_size_kb = len(data) / 1024
                send_size_kb = len(result_b64) / 1024
                
                print(f"[{session_id}] Server: {total_server_time:.1f}ms | Decode: {decode_ms:.1f}ms | Flip: {proc_ms:.1f}ms | Encode: {encode_ms:.1f}ms | In: {recv_size_kb:.1f}KB | Out: {send_size_kb:.1f}KB | Res: {w}x{h}")

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
