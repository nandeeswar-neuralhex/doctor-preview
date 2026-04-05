"""
Configuration settings for the FaceFusion service
"""
import os

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))

# Processing settings
# Processing settings
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "90"))  # Premium quality for RTX 6000
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))

# Model paths
# Use local 'models' directory relative to this config file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.getenv("MODELS_DIR", os.path.join(BASE_DIR, "models"))
INSWAPPER_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")
BUFFALO_MODEL_DIR = os.path.join(MODELS_DIR, "buffalo_l")

# GPU settings
EXECUTION_PROVIDER = os.getenv("EXECUTION_PROVIDER", "CUDAExecutionProvider")

# Quality/Blending settings
ENABLE_SEAMLESS_CLONE = os.getenv("ENABLE_SEAMLESS_CLONE", "true").lower() == "true"
FACE_MASK_BLUR = int(os.getenv("FACE_MASK_BLUR", "35"))  # Wider feathering hides boundary jitter
FACE_MASK_SCALE = float(os.getenv("FACE_MASK_SCALE", "1.1"))

# Optional face enhancement
ENABLE_GFPGAN = os.getenv("ENABLE_GFPGAN", "true").lower() == "true"
_GFPGAN_PATH_ENV = os.getenv("GFPGAN_MODEL_PATH", "")
GFPGAN_MODEL_PATH = _GFPGAN_PATH_ENV or os.path.join(MODELS_DIR, "GFPGANv1.4.pth")

# Smoothing / tracking — alpha = how much old value is retained
# 0.30 = 30% old + 70% new → more responsive, catches subtle mouth/expression changes
# NOTE: bbox and kps are both smoothed, so do NOT also smooth the affine matrix
# (it derives from already-smoothed kps — double-smoothing causes extreme lag)
ENABLE_TEMPORAL_SMOOTHING = os.getenv("ENABLE_TEMPORAL_SMOOTHING", "true").lower() == "true"
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.30"))
MAX_FACES = int(os.getenv("MAX_FACES", "1"))

# WebRTC / Lip sync
ENABLE_WEBRTC = os.getenv("ENABLE_WEBRTC", "true").lower() == "true"
ENABLE_LIPSYNC = os.getenv("ENABLE_LIPSYNC", "true").lower() == "true"
WAV2LIP_MODEL_PATH = os.getenv("WAV2LIP_MODEL_PATH", os.path.join(MODELS_DIR, "wav2lip_gan_96.onnx"))
LIPSYNC_AUDIO_WINDOW_MS = int(os.getenv("LIPSYNC_AUDIO_WINDOW_MS", "300"))

# Frame processing
TARGET_FPS = int(os.getenv("TARGET_FPS", "24"))
FRAME_TIMEOUT_MS = 1000 // TARGET_FPS  # ~41ms for 24 FPS

# ── Phase 1: BiSeNet Face Parsing ──
ENABLE_FACE_PARSING = os.getenv("ENABLE_FACE_PARSING", "true").lower() == "true"
FACE_PARSING_MODEL = os.path.join(MODELS_DIR, "bisenet_face_parsing.onnx")
# Classes to include in parsing mask: face, hair, ears, neck (see face_parser.py)
PARSING_CLASSES = os.getenv("PARSING_CLASSES", "face,hair,ears,neck").split(",")

# ── Phase 1: Real-time GFPGAN Enhancement ──
ENABLE_REALTIME_ENHANCE = os.getenv("ENABLE_REALTIME_ENHANCE", "false").lower() == "true"
ENHANCE_EVERY_N_FRAMES = int(os.getenv("ENHANCE_EVERY_N_FRAMES", "1"))

# ── Phase 1: Detection Resolution (320 for T4, 640 for H100) ──
DETECTION_SIZE = int(os.getenv("DETECTION_SIZE", "320"))

# ── Phase 3: Swap Engine Selection ──
# "inswapper" = current INSwapper 128x128 pipeline
# "liveportrait" = LivePortrait motion-driven generation
SWAP_ENGINE = os.getenv("SWAP_ENGINE", "inswapper")

# ── Phase 3: LivePortrait Settings ──
LIVEPORTRAIT_MODEL_DIR = os.path.join(MODELS_DIR, "liveportrait")
LIVEPORTRAIT_RESOLUTION = int(os.getenv("LIVEPORTRAIT_RESOLUTION", "256"))
ENABLE_EYE_GAZE_CORRECTION = os.getenv("ENABLE_EYE_GAZE_CORRECTION", "true").lower() == "true"
MOTION_SMOOTHING_ALPHA = float(os.getenv("MOTION_SMOOTHING_ALPHA", "0.7"))

# ── Phase 4: A/V Sync Agent Pipeline (Approach 4: Server-Side Muxing) ──
# When enabled, replaces the legacy VideoTransformTrack with the 7-agent
# synchronized pipeline for broadcast-quality lip sync (target: AVO < 40ms).
# Falls back to legacy mode if the pipeline fails (circuit breaker).
ENABLE_AV_SYNC_PIPELINE = os.getenv("ENABLE_AV_SYNC_PIPELINE", "false").lower() == "true"
AV_SYNC_AUDIO_BUFFER_MS = int(os.getenv("AV_SYNC_AUDIO_BUFFER_MS", "500"))
AV_SYNC_MAX_AUDIO_WAIT_MS = int(os.getenv("AV_SYNC_MAX_AUDIO_WAIT_MS", "20"))
AV_SYNC_DRIFT_RECAL_FRAMES = int(os.getenv("AV_SYNC_DRIFT_RECAL_FRAMES", "300"))
AV_SYNC_AVO_WARNING_MS = float(os.getenv("AV_SYNC_AVO_WARNING_MS", "30"))
AV_SYNC_AVO_CRITICAL_MS = float(os.getenv("AV_SYNC_AVO_CRITICAL_MS", "60"))

