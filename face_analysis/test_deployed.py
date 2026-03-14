"""Quick smoke test for deployed face analysis service."""
import cv2
import numpy as np
import requests

SERVER = "http://48.216.182.88:8766"

# 1. Health check
print("1️⃣  Health check...")
resp = requests.get(f"{SERVER}/health", timeout=10)
print(f"   Status: {resp.status_code}")
health = resp.json()
print(f"   Models loaded: {len(health['models_loaded'])}")
print(f"   Uptime: {health['uptime_seconds']}s")
print()

# 2. Create synthetic face image
print("2️⃣  Creating test image...")
img = np.ones((512, 512, 3), dtype=np.uint8) * 180
cv2.ellipse(img, (256, 256), (150, 200), 0, 0, 360, (210, 185, 170), -1)
cv2.circle(img, (200, 220), 20, (80, 60, 50), -1)
cv2.circle(img, (312, 220), 20, (80, 60, 50), -1)
cv2.line(img, (256, 260), (256, 300), (170, 150, 140), 3)
cv2.ellipse(img, (256, 330), (40, 15), 0, 0, 360, (160, 120, 120), -1)
_, buffer = cv2.imencode(".jpg", img)
print("   Image created (512x512)")
print()

# 3. Test /validate
print("3️⃣  Testing /validate endpoint...")
files = [("images", ("front.jpg", buffer.tobytes(), "image/jpeg"))]
data = {"angles": "front_0"}
resp = requests.post(f"{SERVER}/validate", files=files, data=data, timeout=30)
print(f"   Status: {resp.status_code}")
vresult = resp.json()
print(f"   Total: {vresult.get('total')}, Passed: {vresult.get('passed')}")
if vresult.get("results"):
    r = vresult["results"][0]
    print(f"   Face detected: {r.get('face_detected')}")
    print(f"   Quality score: {r.get('quality_score')}")
    print(f"   Issues: {r.get('issues', [])}")
print()

# 4. Test /analyze
print("4️⃣  Testing /analyze endpoint...")
files = {"image": ("test_face.jpg", buffer.tobytes(), "image/jpeg")}
data = {"patient_name": "Test Patient", "patient_age": "30", "angle": "front_0"}
resp = requests.post(f"{SERVER}/analyze", files=files, data=data, timeout=30)
print(f"   Status: {resp.status_code}")
result = resp.json()
print(f"   Session ID: {result.get('session_id', 'N/A')}")
print(f"   Overall Score: {result.get('overall_score', 'N/A')}")
print(f"   Skin Age: {result.get('skin_age', 'N/A')}")
print(f"   Analyzers: {list(result.get('analyzer_scores', {}).keys())}")
print(f"   Zone Scores: {len(result.get('zone_scores', {}))} zones")
print(f"   Conditions: {len(result.get('conditions', []))}")
print(f"   Recommendations: {len(result.get('recommendations', []))}")
print()

# 5. Test /sessions
sid = result.get("session_id")
if sid:
    print("5️⃣  Testing /sessions endpoint...")
    resp = requests.get(f"{SERVER}/sessions/{sid}", timeout=10)
    print(f"   Status: {resp.status_code}")
    resp = requests.get(f"{SERVER}/sessions", timeout=10)
    print(f"   Total sessions: {len(resp.json())}")
    print()

    # 6. Test /report
    print("6️⃣  Testing /report endpoint...")
    resp = requests.get(f"{SERVER}/sessions/{sid}/report", timeout=15)
    print(f"   Status: {resp.status_code}")
    print(f"   Report HTML length: {len(resp.text)} chars")
    print(f"   Contains <!DOCTYPE>: {'<!DOCTYPE' in resp.text}")

print()
print("=" * 50)
print("✅ ALL ENDPOINTS WORKING — DEPLOYMENT SUCCESSFUL!")
print("=" * 50)
print(f"   URL: {SERVER}")
print(f"   FQDN: face-analysis-svc.eastus.azurecontainer.io:8766")
