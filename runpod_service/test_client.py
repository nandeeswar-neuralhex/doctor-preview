"""
Test client for the Face Swap service
Run this on your local machine to test the RunPod deployment
"""
import asyncio
import base64
import time
import cv2
import websockets
import requests
import argparse

async def test_face_swap(server_url: str, target_image_path: str, session_id: str = "test-session"):
    """
    Test the face swap service with webcam
    
    Args:
        server_url: RunPod server URL (e.g., https://xxx-8765.proxy.runpod.net)
        target_image_path: Path to target face image
        session_id: Unique session ID
    """
    # Normalize URL
    http_url = server_url.replace("wss://", "https://").replace("ws://", "http://")
    if not http_url.startswith("http"):
        http_url = f"https://{http_url}"
    
    ws_url = http_url.replace("https://", "wss://").replace("http://", "ws://")
    
    print(f"Server: {http_url}")
    print(f"WebSocket: {ws_url}")
    
    # Step 1: Health check
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{http_url}/health", timeout=10)
        print(f"   Health: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Step 2: Upload target image
    print(f"\n2. Uploading target image: {target_image_path}")
    try:
        with open(target_image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(
                f"{http_url}/upload-target?session_id={session_id}",
                files=files,
                timeout=30
            )
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Step 3: Connect webcam and stream
    print("\n3. Opening webcam and connecting to WebSocket...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("   Error: Could not open webcam")
        return
    
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("   WebSocket connecting...")
    
    try:
        async with websockets.connect(f"{ws_url}/ws/{session_id}") as websocket:
            print("   Connected! Press 'q' to quit.\n")
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send to server
                send_time = time.time()
                await websocket.send(frame_b64)
                
                # Receive processed frame
                result_b64 = await websocket.recv()
                recv_time = time.time()
                
                # Decode result
                result_bytes = base64.b64decode(result_b64)
                result_arr = np.frombuffer(result_bytes, dtype=np.uint8)
                result_frame = cv2.imdecode(result_arr, cv2.IMREAD_COLOR)
                
                # Calculate latency & FPS
                latency_ms = (recv_time - send_time) * 1000
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                
                # Display
                cv2.putText(result_frame, f"FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show original and result side by side
                combined = cv2.hconcat([frame, result_frame])
                cv2.imshow("Doctor Preview - Face Swap (Original | Swapped)", combined)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"   WebSocket error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nTest complete!")


def main():
    import numpy as np  # Import here for webcam test
    
    parser = argparse.ArgumentParser(description="Test the Face Swap service")
    parser.add_argument("--server", required=True, help="Server URL (e.g., https://xxx-8765.proxy.runpod.net)")
    parser.add_argument("--target", required=True, help="Path to target face image")
    parser.add_argument("--session", default="test-session", help="Session ID")
    
    args = parser.parse_args()
    
    # Make numpy available in async function
    global np
    import numpy as np
    
    asyncio.run(test_face_swap(args.server, args.target, args.session))


if __name__ == "__main__":
    main()
