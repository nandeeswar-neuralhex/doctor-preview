"""
Doctor Face Analysis — Configuration Management
Enterprise-grade configuration with validation and environment variable support.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ServerConfig:
    """Server configuration — immutable after creation."""
    host: str = "0.0.0.0"
    port: int = 8766
    workers: int = 4
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class GPUConfig:
    """GPU/CUDA configuration for T4."""
    device: str = "cuda"
    provider: str = "CUDAExecutionProvider"
    vram_limit_gb: float = 14.0  # T4 = 16GB, reserve 2GB headroom
    cudnn_benchmark: bool = True
    half_precision: bool = True  # FP16 for T4 Tensor Cores


@dataclass(frozen=True)
class AnalysisConfig:
    """Skin analysis pipeline configuration."""
    # Input
    target_resolution: int = 1080  # 1080x1080 analysis resolution
    min_face_size: int = 100  # Minimum face size in pixels
    max_faces: int = 1  # Single face analysis

    # Multi-frame averaging (Technique 3)
    multi_frame_count: int = 30  # Frames to average for depth
    icp_iterations: int = 50  # ICP alignment iterations
    depth_noise_threshold: float = 0.005  # 5mm noise threshold

    # AI Depth Super-Resolution (Technique 4)
    depth_sr_scale: int = 4  # 256x192 → 1024x768
    depth_sr_model: str = "guided_filter"  # guided_filter | deep_sr

    # Scoring
    score_min: int = 0
    score_max: int = 100

    # Zones
    facial_zones: list = field(default_factory=lambda: [
        "forehead", "left_cheek", "right_cheek", "nose",
        "chin", "under_eye_left", "under_eye_right",
        "jawline_left", "jawline_right", "neck",
        "left_temple", "right_temple"
    ])


@dataclass(frozen=True)
class ModelPaths:
    """ML model file paths."""
    face_detection: str = "models/buffalo_l"
    wrinkle_model: str = "models/wrinkle_resnet50.onnx"
    pore_model: str = "models/pore_yolov8n.onnx"
    pigment_model: str = "models/pigment_sam_vit_b.onnx"
    texture_model: str = "models/texture_glcm.pkl"
    depth_sr_model: str = "models/depth_sr_net.onnx"
    face_parsing: str = "models/bisenet_face_parsing.onnx"


@dataclass(frozen=True)
class ReportConfig:
    """Report generation configuration."""
    template_dir: str = "templates"
    output_dir: str = "reports"
    logo_path: str = "assets/logo.png"
    clinic_name: str = "Doctor Preview"
    disclaimer: str = (
        "This analysis is AI-assisted and should be used as a "
        "consultation tool only. Clinical diagnosis should be made "
        "by a qualified professional."
    )


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration for sessions and images."""
    sessions_dir: str = "data/sessions"
    uploads_dir: str = "data/uploads"
    heatmaps_dir: str = "data/heatmaps"
    reports_dir: str = "data/reports"
    max_session_age_days: int = 365


class AppConfig:
    """
    Application-wide configuration singleton.
    Reads from environment variables with fallback to defaults.
    """
    _instance: Optional["AppConfig"] = None

    def __new__(cls) -> "AppConfig":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.server = ServerConfig(
            host=os.getenv("FA_HOST", "0.0.0.0"),
            port=int(os.getenv("FA_PORT", "8766")),
            workers=int(os.getenv("FA_WORKERS", "4")),
        )
        self.gpu = GPUConfig(
            device=os.getenv("FA_DEVICE", "cuda"),
            half_precision=os.getenv("FA_FP16", "1") == "1",
        )
        self.analysis = AnalysisConfig(
            target_resolution=int(os.getenv("FA_RESOLUTION", "1080")),
            multi_frame_count=int(os.getenv("FA_MULTI_FRAME", "30")),
            depth_sr_scale=int(os.getenv("FA_DEPTH_SR_SCALE", "4")),
        )
        self.models = ModelPaths()
        self.report = ReportConfig(
            clinic_name=os.getenv("FA_CLINIC_NAME", "Doctor Preview"),
        )
        self.storage = StorageConfig(
            sessions_dir=os.getenv("FA_SESSIONS_DIR", "data/sessions"),
        )

    @classmethod
    def reset(cls):
        """Reset singleton — for testing only."""
        cls._instance = None


def get_config() -> AppConfig:
    """Factory function to get configuration."""
    return AppConfig()
