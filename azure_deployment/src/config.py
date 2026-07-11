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
FACE_MASK_BLUR = int(os.getenv("FACE_MASK_BLUR", "25"))
FACE_MASK_SCALE = float(os.getenv("FACE_MASK_SCALE", "1.1"))

# Optional face enhancement
ENABLE_GFPGAN = os.getenv("ENABLE_GFPGAN", "true").lower() == "true"
_GFPGAN_PATH_ENV = os.getenv("GFPGAN_MODEL_PATH", "")
GFPGAN_MODEL_PATH = _GFPGAN_PATH_ENV or os.path.join(MODELS_DIR, "GFPGANv1.4.pth")
# Blend weight of the GFPGAN-enhanced face vs raw swap (0=off, 1=full GFPGAN).
# Partial blend keeps identity/temporal stability while adding texture detail.
GFPGAN_BLEND = float(os.getenv("GFPGAN_BLEND", "0.5"))

# Smoothing / tracking
ENABLE_TEMPORAL_SMOOTHING = os.getenv("ENABLE_TEMPORAL_SMOOTHING", "true").lower() == "true"
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.4"))
MAX_FACES = int(os.getenv("MAX_FACES", "1"))

# WebRTC / Lip sync
ENABLE_WEBRTC = os.getenv("ENABLE_WEBRTC", "false").lower() == "true"
ENABLE_LIPSYNC = os.getenv("ENABLE_LIPSYNC", "true").lower() == "true"
WAV2LIP_MODEL_PATH = os.getenv("WAV2LIP_MODEL_PATH", os.path.join(MODELS_DIR, "wav2lip_gan_96.onnx"))
LIPSYNC_AUDIO_WINDOW_MS = int(os.getenv("LIPSYNC_AUDIO_WINDOW_MS", "500"))

# A/V sync: relay the client's mic audio back over the SAME WebRTC peer
# connection, delayed to match the measured video pipeline latency. The
# browser then aligns both tracks via RTCP sender reports, so voice and
# mouth arrive in sync (and OBS captures them already synced).
ENABLE_AUDIO_RELAY = os.getenv("ENABLE_AUDIO_RELAY", "true").lower() == "true"
# Extra fixed delay (ms) on top of the measured video latency — use to
# compensate for client-side render/display delay if needed.
AUDIO_RELAY_EXTRA_DELAY_MS = int(os.getenv("AUDIO_RELAY_EXTRA_DELAY_MS", "0"))

# Frame processing
TARGET_FPS = int(os.getenv("TARGET_FPS", "24"))
FRAME_TIMEOUT_MS = 1000 // TARGET_FPS  # ~41ms for 24 FPS
