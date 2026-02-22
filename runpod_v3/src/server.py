import asyncio
import base64
import time
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="RunPod V3 Baseline Network Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional TurboJPEG for faster encode/decode
try:
    from turbojpeg import TurboJPEG
    _tj = TurboJPEG()
    print("✅ TurboJPEG available for fast encoding/decoding")
except Exception:
    _tj = None
    print("ℹ️ TurboJPEG not available, using cv2")

JPEG_QUALITY = 75

@app.get("/")
def health_check():
    return {"status": "ok", "service": "runpod_v3_baseline"}

# Dummy endpoint to satisfy the desktop app's initialization flow
@app.post("/upload-target")
async def upload_target(session_id: str):
    print(f"[{session_id}] Received target face upload (ignored in baseline test)")
    return {"success": True, "message": "Dummy upload complete", "count": 1}

@app.post("/session/settings")
async def update_settings(session_id: str):
    print(f"[{session_id}] Received settings update (ignored in baseline test)")
    return {"success": True}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[{session_id}] Client connected (Baseline Test)")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Receive text (Base64) or binary data
            data = await websocket.receive()
            t0 = time.time()

            try:
                # 1. Decode payload
                # 1. Decode payload
                if "text" in data:
                    payload = data["text"]
                    if payload.startswith('{'):
                        import json
                        try:
                            msg = json.loads(payload)
                            if "image" in msg:
                                payload = msg["image"]
                            elif msg.get("type") == "ping":
                                await websocket.send_text('{"type":"pong"}')
                                continue
                            else:
                                continue
                        except:
                            pass
                            
                    if ',' in payload:
                        payload = payload.split(',')[1]
                    img_bytes = base64.b64decode(payload)
                elif "bytes" in data:
                    # Frontend sends binary struct:
                    # 4B frameId + 8B timestamp + 4B audioLen + 4B sampleRate = 20-byte header
                    # Then audio bytes, then JPEG bytes
                    raw_data = data["bytes"]
                    if len(raw_data) < 20: 
                        continue
                        
                    import struct
                    # Read the 4-byte audio length at offset 12
                    audio_len = struct.unpack_from('<I', raw_data, 12)[0]
                    
                    # JPEG starts after 20-byte header + audio_len
                    jpeg_offset = 20 + audio_len
                    if jpeg_offset >= len(raw_data):
                        continue
                        
                    img_bytes = raw_data[jpeg_offset:]
                else:
                    continue

                # 2. Decode JPEG to BGR
                t1 = time.time()
                if _tj:
                    frame = _tj.decode(img_bytes)
                else:
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue

                # 3. Process (Just a fast horizontal flip)
                t2 = time.time()
                result = cv2.flip(frame, 1)

                # 4. Encode BGR to JPEG
                t3 = time.time()
                if _tj:
                    out_bytes = _tj.encode(result, quality=JPEG_QUALITY)
                else:
                    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    out_bytes = buffer.tobytes()

                # 5. Send back
                t4 = time.time()
                b64_out = base64.b64encode(out_bytes).decode('utf-8')
                out_payload = f"data:image/jpeg;base64,{b64_out}"
                await websocket.send_text(out_payload)
                t5 = time.time()

                # Logging
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    decode = (t2 - t1) * 1000
                    process = (t3 - t2) * 1000
                    encode = (t4 - t3) * 1000
                    send = (t5 - t4) * 1000
                    total_server = (t5 - t0) * 1000
                    h, w = result.shape[:2]
                    
                    print(f"[{session_id}] FPS: {fps:.1f} | Res: {w}x{h} | "
                          f"dec={decode:.1f}ms flip={process:.1f}ms enc={encode:.1f}ms "
                          f"send={send:.1f}ms | TotalServer={total_server:.1f}ms")

            except Exception as e:
                print(f"Frame processing error: {e}")

    except WebSocketDisconnect:
        print(f"[{session_id}] Client disconnected")
    except Exception as e:
        print(f"[{session_id}] WebSocket error: {e}")

if __name__ == "__main__":
    print("Starting RunPod V3 Baseline Network Test Server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8765, workers=1)
