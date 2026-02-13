
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

import io
from PIL import Image, ImageOps
from face_swapper import FaceSwapper
from config import JPEG_QUALITY

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
    print("Initializing FaceSwapper...")
    try:
        swapper = FaceSwapper()
        print("FaceSwapper ready.")
    except Exception as e:
        print(f"FaceSwapper initialization failed: {e}")
        swapper = None

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

        success, message = swapper.set_target_face(session_id, image)
        
        if not success:
            return JSONResponse(status_code=400, content={"error": message})

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
                
                # TRANSFORM: Face Swap
                if swapper and swapper.has_target(session_id):
                    result = swapper.swap_face(session_id, frame)
                else:
                    # Fallback if no target set yet or swapper failed
                    result = frame

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
    if swapper:
        swapper.cleanup_session(session_id)
    print(f"Session {session_id} cleaned up")
    return {"status": "success", "message": "Session cleaned up"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
