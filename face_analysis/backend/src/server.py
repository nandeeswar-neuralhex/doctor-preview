"""
Doctor Face Analysis — FastAPI Server
Enterprise-grade REST API for face skin analysis.
"""

import base64
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import get_config
from .models.schemas import (
    AnalysisResponse,
    CompareResponse,
    HealthResponse,
    ReportResponse,
)
from .pipeline.analysis_pipeline import AnalysisPipeline
from .reports.report_generator import ReportGenerator

# ── Globals ─────────────────────────────────────────────────────────────────

_pipeline: Optional[AnalysisPipeline] = None
_report_gen: Optional[ReportGenerator] = None
_sessions: dict = {}  # In-memory session store (use Cosmos DB in production)
_start_time: float = 0
_loaded_models: List[str] = []


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup."""
    global _pipeline, _report_gen, _start_time, _loaded_models

    print("[FaceAnalysis] Loading models...")
    _start_time = time.time()

    _pipeline = AnalysisPipeline()
    _loaded_models = _pipeline.load_models()
    _report_gen = ReportGenerator()

    print(f"[FaceAnalysis] Models loaded: {_loaded_models}")
    print(f"[FaceAnalysis] Server ready on port {get_config().server.port}")

    yield

    print("[FaceAnalysis] Shutting down...")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Doctor Face Analysis API",
    version="1.0.0",
    description="AURA-equivalent skin analysis system with AI-powered diagnostics",
    lifespan=lifespan,
)

config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _ensure_pipeline():
    """Lazily initialize pipeline if lifespan hasn't run (e.g. in tests)."""
    global _pipeline, _report_gen, _start_time, _loaded_models
    if _pipeline is None:
        _start_time = time.time()
        _pipeline = AnalysisPipeline()
        _loaded_models = _pipeline.load_models()
        _report_gen = ReportGenerator()
    return _pipeline

async def _decode_upload(upload: UploadFile) -> np.ndarray:
    """Decode uploaded image file to BGR numpy array."""
    contents = await upload.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image file")
    return image


def _store_session(session_id: str, data: dict):
    """Store session results (in-memory, replace with Cosmos DB)."""
    _sessions[session_id] = {
        **data,
        "stored_at": time.time(),
    }


def _get_session(session_id: str) -> dict:
    """Retrieve session by ID."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    return _sessions[session_id]


# ── Routes: Health ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check server health, GPU status, and loaded models."""
    gpu_available = False
    gpu_name = ""
    vram_used = 0
    vram_total = 0

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_used = torch.cuda.memory_allocated(0) / 1024 / 1024
            vram_total = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
    except ImportError:
        pass

    # Fallback: detect GPU via nvidia-smi if torch is not installed
    if not gpu_available:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                gpu_available = True
                gpu_name = parts[0] if len(parts) > 0 else "Unknown"
                vram_used = float(parts[1]) if len(parts) > 1 else 0
                vram_total = float(parts[2]) if len(parts) > 2 else 0
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_used_mb=round(vram_used, 1),
        vram_total_mb=round(vram_total, 1),
        models_loaded=_loaded_models,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ── Routes: Validation ──────────────────────────────────────────────────────

@app.post("/validate", tags=["Validation"])
async def validate_photos(
    images: List[UploadFile] = File(..., description="Face photos to validate"),
    angles: List[str] = Form(..., description="Expected angle for each photo"),
):
    """
    Validate captured photos for quality and face detection before analysis.

    Checks per photo:
      - Face detected (MediaPipe)
      - Sharpness (Laplacian variance)
      - Brightness (mean luminance)
      - Angle plausibility (nose-landmark deviation vs expected angle)

    Returns per-photo pass/fail with detailed issues list.
    """
    pipeline = _ensure_pipeline()
    results = []

    for i, upload in enumerate(images):
        image = await _decode_upload(upload)
        expected_angle = angles[i] if i < len(angles) else "front_0"

        issues = []
        face_detected = False
        angle_ok = True
        quality_score = 0.0

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Brightness
        brightness = float(np.mean(gray))
        if brightness < 50:
            issues.append("Image too dark — improve lighting")
        elif brightness > 230:
            issues.append("Image overexposed — reduce lighting")

        # 2. Sharpness (Laplacian variance on center crop)
        cy, cx = h // 4, w // 4
        center = gray[cy:cy + h // 2, cx:cx + w // 2]
        laplacian = cv2.Laplacian(center, cv2.CV_64F)
        sharpness = float(laplacian.var())
        if sharpness < 30:
            issues.append("Image is blurry — hold the camera steady")

        # 3. Contrast
        contrast = float(np.std(gray))
        if contrast < 20:
            issues.append("Low contrast — adjust lighting")

        # 4. Face detection via pipeline's face detector
        try:
            det = pipeline._face_detector
            if not det._is_loaded:
                det.load_model()
            det_result = det.analyze(image)

            if det_result.get("error"):
                issues.append("No face detected — reposition and try again")
            else:
                face_detected = True
                landmarks = det_result.get("landmarks_468")

                # 5. Angle verification using nose / eye landmarks
                if landmarks is not None and len(landmarks) >= 400:
                    nose_tip = landmarks[1]      # nose tip
                    left_ear = landmarks[234]     # left ear region
                    right_ear = landmarks[454]    # right ear region
                    left_eye = landmarks[33]
                    right_eye = landmarks[263]

                    face_width = float(np.linalg.norm(left_ear - right_ear))
                    nose_center = float(nose_tip[0])
                    face_center = float((left_ear[0] + right_ear[0]) / 2)
                    offset_ratio = (nose_center - face_center) / max(face_width, 1) if face_width > 0 else 0

                    # Expected offset ratios per angle
                    angle_offsets = {
                        "front_0": (-.12, .12),
                        "left_45": (-.35, -.05),
                        "right_45": (.05, .35),
                        "left_90": (-.6, -.2),
                        "right_90": (.2, .6),
                    }

                    expected = angle_offsets.get(expected_angle, (-.15, .15))
                    if not (expected[0] <= offset_ratio <= expected[1]):
                        angle_ok = False
                        issues.append(f"Head angle doesn't match '{expected_angle}' — reposition")

                    # 6. Eye openness check (both eyes visible)
                    eye_dist = float(np.linalg.norm(left_eye - right_eye))
                    if eye_dist < face_width * 0.15:
                        issues.append("Eyes not clearly visible — face the camera more")

        except Exception as e:
            issues.append(f"Face detection error: {str(e)[:60]}")

        # Quality score (0-100)
        score = 100.0
        if brightness < 50 or brightness > 230:
            score -= 20
        if sharpness < 30:
            score -= 25
        if contrast < 20:
            score -= 15
        if not face_detected:
            score -= 30
        if not angle_ok:
            score -= 15
        quality_score = max(0, min(100, score))

        passed = face_detected and len(issues) == 0

        results.append({
            "index": i,
            "angle": expected_angle,
            "face_detected": face_detected,
            "angle_ok": angle_ok,
            "brightness": round(brightness, 1),
            "sharpness": round(sharpness, 1),
            "contrast": round(contrast, 1),
            "quality_score": round(quality_score, 1),
            "passed": passed,
            "issues": issues,
        })

    overall_passed = all(r["passed"] for r in results)
    return JSONResponse(content={
        "status": "ok",
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "overall_passed": overall_passed,
        "results": results,
    })


# ── Routes: Analysis ────────────────────────────────────────────────────────

@app.post("/analyze", tags=["Analysis"])
async def analyze_single(
    image: UploadFile = File(..., description="Face photo (JPEG/PNG)"),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    patient_skin_type: Optional[str] = Form(None),
    angle: str = Form("front_0"),
    depth_map: Optional[UploadFile] = File(None, description="Optional depth map"),
):
    """
    Analyze a single face photo.
    Returns comprehensive skin analysis with scores, heatmaps, and recommendations.
    """
    face_image = await _decode_upload(image)

    # Decode optional depth map
    dm = None
    if depth_map:
        dm_bytes = await depth_map.read()
        dm = np.frombuffer(dm_bytes, dtype=np.float32)
        # Try to reshape — assume square-ish LiDAR resolution
        side = int(np.sqrt(len(dm)))
        if side > 0:
            dm = dm[:side * side].reshape(side, side)

    result = _ensure_pipeline().analyze_single(face_image, depth_map=dm)

    # Attach patient info
    if patient_name:
        result["patient"] = {
            "name": patient_name,
            "age": patient_age,
            "skin_type": patient_skin_type,
        }

    # Store session
    _store_session(result["session_id"], result)

    return JSONResponse(content=_make_serializable(result))


@app.post("/analyze/multi", tags=["Analysis"])
async def analyze_multi(
    front: UploadFile = File(..., description="Front view (0°)"),
    left_45: Optional[UploadFile] = File(None, description="Left 45° view"),
    right_45: Optional[UploadFile] = File(None, description="Right 45° view"),
    left_90: Optional[UploadFile] = File(None, description="Left 90° view"),
    right_90: Optional[UploadFile] = File(None, description="Right 90° view"),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
):
    """
    Multi-angle analysis (up to 5 views).
    Merges results for more comprehensive scoring.
    """
    images = {"front_0": await _decode_upload(front)}

    if left_45:
        images["left_45"] = await _decode_upload(left_45)
    if right_45:
        images["right_45"] = await _decode_upload(right_45)
    if left_90:
        images["left_90"] = await _decode_upload(left_90)
    if right_90:
        images["right_90"] = await _decode_upload(right_90)

    result = _ensure_pipeline().analyze_multi(images)

    if patient_name:
        result["patient"] = {"name": patient_name, "age": patient_age}

    _store_session(result["session_id"], result)

    return JSONResponse(content=_make_serializable(result))


@app.post("/analyze/depth-enhanced", tags=["Analysis"])
async def analyze_depth_enhanced(
    image: UploadFile = File(..., description="Face photo"),
    depth_frames: List[UploadFile] = File(..., description="Multiple depth frames for averaging"),
    patient_name: Optional[str] = Form(None),
):
    """
    Analysis with Technique 3 (Multi-Frame Averaging) + Technique 4 (AI Super-Resolution).
    Upload multiple LiDAR depth frames for enhanced accuracy.
    """
    face_image = await _decode_upload(image)

    frames = []
    for df in depth_frames:
        raw = await df.read()
        arr = np.frombuffer(raw, dtype=np.float32)
        side = int(np.sqrt(len(arr)))
        if side > 0:
            frames.append(arr[:side * side].reshape(side, side))

    if not frames:
        raise HTTPException(400, "At least one depth frame required")

    result = _ensure_pipeline().analyze_single(face_image, depth_frames=frames)

    if patient_name:
        result["patient"] = {"name": patient_name}

    _store_session(result["session_id"], result)

    return JSONResponse(content=_make_serializable(result))


# ── Routes: Sessions ────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str):
    """Retrieve analysis results by session ID."""
    session = _get_session(session_id)
    return JSONResponse(content=_make_serializable(session))


@app.get("/sessions", tags=["Sessions"])
async def list_sessions():
    """List all stored sessions."""
    sessions = []
    for sid, data in _sessions.items():
        sessions.append({
            "session_id": sid,
            "overall_score": data.get("overall_score"),
            "patient": data.get("patient"),
            "timestamp": data.get("stored_at"),
            "angles": data.get("angles_analyzed", []),
        })
    return sessions


# ── Routes: Reports ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/report", tags=["Reports"])
async def get_report_html(
    session_id: str,
    doctor_notes: str = "",
):
    """Generate HTML report for a session."""
    session = _get_session(session_id)
    patient = session.get("patient")

    html = _report_gen.generate_html(
        analysis=session,
        patient=patient,
        doctor_notes=doctor_notes,
    )

    return HTMLResponse(content=html)


@app.get("/sessions/{session_id}/report/pdf", tags=["Reports"])
async def get_report_pdf(session_id: str, doctor_notes: str = ""):
    """Generate PDF report for a session."""
    session = _get_session(session_id)
    patient = session.get("patient")

    html = _report_gen.generate_html(analysis=session, patient=patient, doctor_notes=doctor_notes)
    pdf_bytes = _report_gen.generate_pdf_bytes(html)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{session_id[:8]}.pdf"},
    )


# ── Routes: Comparison ──────────────────────────────────────────────────────

@app.get("/compare/{session_before}/{session_after}", tags=["Comparison"])
async def compare_sessions(session_before: str, session_after: str):
    """Compare two sessions for before/after analysis."""
    before = _get_session(session_before)
    after = _get_session(session_after)

    comparison = _ensure_pipeline().compare_sessions(before, after)

    return JSONResponse(content=_make_serializable(comparison))


# ── Routes: Annotations ────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/annotate", tags=["Annotations"])
async def save_annotations(
    session_id: str,
    annotations: str = Form(..., description="JSON annotations data"),
    notes: str = Form(""),
):
    """Save doctor's annotations and notes for a session."""
    session = _get_session(session_id)
    session["annotations"] = json.loads(annotations)
    session["doctor_notes"] = notes
    _store_session(session_id, session)

    return {"status": "saved", "session_id": session_id}


# ── Serialization Helper ───────────────────────────────────────────────────

def _make_serializable(obj):
    """Convert numpy types and other non-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=config.server.host,
        port=config.server.port,
        workers=1,  # Single worker for GPU
        reload=False,
    )
