"""
Configuration settings for the FaceFusion service
"""
import os

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8765"))

# Processing settings
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "10"))

# Model paths
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
INSWAPPER_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")
BUFFALO_MODEL_DIR = os.path.join(MODELS_DIR, "buffalo_l")

# GPU settings
EXECUTION_PROVIDER = os.getenv("EXECUTION_PROVIDER", "CUDAExecutionProvider")

# Frame processing
TARGET_FPS = int(os.getenv("TARGET_FPS", "24"))
FRAME_TIMEOUT_MS = 1000 // TARGET_FPS  # ~41ms for 24 FPS
