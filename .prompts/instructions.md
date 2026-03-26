---
applyTo: '**'
---
# 📋 Global Project Instructions

## Project: Doctor Preview — Real-Time Face Swap

### What This Project Does
A real-time face replacement system for live video consultations.  
Users upload a target face → the system swaps their face in the live camera feed → output streamed via WebRTC or WebSocket.

---

## Architecture Overview

```
┌──────────────────────┐      WebRTC / WebSocket      ┌──────────────────────────┐
│   Desktop App        │ ◄──────────────────────────► │   GPU Server (RunPod)     │
│   (Electron + React) │                               │   (FastAPI + ONNX)        │
│                      │                               │                           │
│   - CameraView.jsx   │     Binary frames             │   - face_swapper.py       │
│   - useWebRTC.js     │     ◄──── swap ────►          │   - lip_syncer.py         │
│   - useWebSocket.js  │                               │   - webrtc.py             │
│   - Login.jsx (Clerk)│                               │   - server.py             │
│   - Settings.jsx     │                               │   - config.py             │
└──────────────────────┘                               └──────────────────────────┘
```

### Services in this Repo

| Service | Path | Tech | Purpose |
|---|---|---|---|
| Desktop App | `desktop_app/` | React 18, Vite, Electron, Tailwind, Clerk | Desktop client UI |
| RunPod v2 Server | `runpod_v2/` | FastAPI, ONNX, InsightFace, Wav2Lip, aiortc | GPU face swap server |
| RunPod v3 Server | `runpod_v3/` | FastAPI (lightweight) | Simplified server |
| Azure Deployment | `azure_deployment/` | Docker, nginx, Azure VM | Cloud GPU deployment |
| Face Analysis | `face_analysis/` | Python, pytest | Face analysis utilities |

---

## Tech Stack — Exact Versions

### Backend (Python 3.10+)
| Package | Version | Purpose |
|---|---|---|
| FastAPI | 0.109.2 | HTTP/WebSocket API framework |
| uvicorn | 0.27.1 | ASGI server |
| insightface | 0.7.3 | Face detection + ArcFace recognition |
| onnxruntime-gpu | 1.20.1 | GPU inference (CUDA 12 + cuDNN 9) |
| opencv-python-headless | 4.9.0.80 | Image processing |
| numpy | 1.26.4 | Array operations |
| aiortc | ≥1.7.0 | WebRTC server |
| PyTurboJPEG | ≥1.7.1 | Fast JPEG encode/decode |
| librosa | ≥0.10.0 | Audio processing for lip sync |

### Frontend (Node 18+)
| Package | Version | Purpose |
|---|---|---|
| React | 18.2.0 | UI framework |
| Vite | 5.0.12 | Build tool |
| Electron | 28.2.0 | Desktop wrapper |
| Tailwind CSS | 3.4.1 | Styling |
| Clerk | 5.61.3 | Authentication |

---

## Coding Rules — ALL Agents MUST Follow

### Python Rules
1. **Logger in every file**: `import logging; logger = logging.getLogger(__name__)`
2. **Type hints on all functions**: `def process(frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:`
3. **Docstrings on all public functions** (Google-style)
4. **Pydantic models** for all API request/response bodies
5. **try/except** on every external call (network, file I/O, GPU inference)
6. **No bare `except:`** — always catch specific exceptions
7. **Constants in config.py** — no magic numbers in logic code
8. **async def** for all FastAPI endpoints
9. **f-strings** for string formatting (not `.format()` or `%`)
10. **snake_case** for files, functions, variables; **PascalCase** for classes

### React/JS Rules
1. **Functional components only** (no class components)
2. **Custom hooks** for all data fetching and shared state
3. **Tailwind CSS** for all styling (no inline styles, no CSS modules)
4. **useCallback** for event handlers passed as props
5. **useRef** for mutable values that don't trigger re-render
6. **Loading + Error + Empty** states in every data component
7. **Environment variables** via `import.meta.env.VITE_*`
8. **Destructured props** with default values
9. **No console.log in production** — use conditional logging
10. **PascalCase** for components; **camelCase** for hooks/variables

### API Contract Rules
1. **REST conventions**: POST=create, GET=read, PUT=update, DELETE=delete
2. **Status codes**: 200=OK, 201=Created, 400=Bad Request, 404=Not Found, 422=Validation, 500=Server Error
3. **Error format**: `{"error": "ERROR_CODE", "message": "Human-readable message"}`
4. **Versioned URLs**: `/api/v1/resource`
5. **JSON request/response** for structured data
6. **Binary** for frame data (JPEG bytes via WebSocket/WebRTC)

### Git Rules
1. **Branch**: `nandeeswar-webrtc-uma-fix` (current working branch)
2. **Commit format**: `type: short description` (feat, fix, docs, refactor, test, chore)
3. **No secrets in commits** — use env vars
4. **Meaningful commit messages** — explain WHY, not just WHAT

---

## Environment Setup

### Backend
```bash
cd runpod_v2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 8765
```

### Frontend
```bash
cd desktop_app
npm install
npm run start:react    # Dev server on :3000
npm start              # Full Electron app
```

### Face Analysis
```bash
cd face_analysis/backend
source .venv/bin/activate
pytest
```

---

## Critical Files — Handle with Care

| File | Why Critical |
|---|---|
| `runpod_v2/src/face_swapper.py` | Core face swap pipeline — temporal smoothing, EMA, face mask |
| `runpod_v2/src/config.py` | All tunable constants — changing values affects quality |
| `runpod_v2/src/webrtc.py` | WebRTC pipeline — latency-sensitive, cross-fade logic |
| `runpod_v2/src/lip_syncer.py` | Lip sync pipeline — per-session state, temporal blending |
| `desktop_app/src/components/CameraView.jsx` | Canvas rendering — jitter buffer, paint loop |
| `desktop_app/src/hooks/useWebRTC.js` | WebRTC client — peer connection, media tracks |
| `desktop_app/src/hooks/useWebSocket.js` | WebSocket client — binary frame pipeline |

---

## Performance Constraints

| Metric | Target | Why |
|---|---|---|
| Frame processing | < 50ms per frame | Real-time at 20+ FPS |
| WebRTC latency | < 150ms end-to-end | Live video feel |
| WebSocket round-trip | < 200ms | Acceptable for preview |
| GPU memory | < 8GB | Fit on T4/A10 instances |
| Item size (if Cosmos DB) | < 2MB | Azure Cosmos DB limit |

---

## Do NOT

- ❌ Create new `CosmosClient` instances per request (reuse singleton)
- ❌ Block the event loop with synchronous GPU calls
- ❌ Allocate numpy arrays in hot loops (pre-allocate and reuse)
- ❌ Use `cv2.imshow()` on headless servers
- ❌ Commit `.env` files, API keys, or model weights
- ❌ Change SMOOTHING_ALPHA without testing on live video
- ❌ Use bare `except:` — always catch specific exceptions
