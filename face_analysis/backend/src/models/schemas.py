"""
Pydantic schemas for the Doctor Face Analysis system.
All request/response models with strict validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ── Enumerations ────────────────────────────────────────────────────────────

class SkinConditionType(str, Enum):
    WRINKLES = "wrinkles"
    FINE_LINES = "fine_lines"
    PORES = "pores"
    PIGMENTATION = "pigmentation"
    BROWN_SPOTS = "brown_spots"
    REDNESS = "redness"
    ROSACEA = "rosacea"
    TEXTURE = "texture"
    ACNE = "acne"
    DARK_CIRCLES = "dark_circles"


class Severity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class CaptureAngle(str, Enum):
    FRONT = "front_0"
    LEFT_45 = "left_45"
    RIGHT_45 = "right_45"
    LEFT_90 = "left_90"
    RIGHT_90 = "right_90"


class FitzpatrickType(int, Enum):
    TYPE_I = 1
    TYPE_II = 2
    TYPE_III = 3
    TYPE_IV = 4
    TYPE_V = 5
    TYPE_VI = 6


# ── Sub-models ──────────────────────────────────────────────────────────────

class ZoneScore(BaseModel):
    """Score breakdown for a single facial zone."""
    wrinkles: float = Field(ge=0, le=100)
    pores: float = Field(ge=0, le=100)
    pigmentation: float = Field(ge=0, le=100)
    redness: float = Field(ge=0, le=100)
    texture: float = Field(ge=0, le=100)
    overall: float = Field(ge=0, le=100)

    @validator("overall", pre=True, always=True)
    def compute_overall(cls, v, values):
        if v is not None and v > 0:
            return v
        weights = {"wrinkles": 0.25, "pores": 0.20, "pigmentation": 0.20,
                    "redness": 0.15, "texture": 0.20}
        total = sum(values.get(k, 0) * w for k, w in weights.items())
        return round(total, 1)


class ConditionDetection(BaseModel):
    """A detected skin condition."""
    condition_type: SkinConditionType
    severity: Severity
    zones: List[str]
    count: int = 0
    confidence: float = Field(ge=0, le=1.0)
    description: str = ""


class SymmetryResult(BaseModel):
    """Facial symmetry analysis."""
    overall_score: float = Field(ge=0, le=100)
    eye_alignment: float = Field(ge=0, le=100)
    cheek_balance: float = Field(ge=0, le=100)
    jawline_symmetry: float = Field(ge=0, le=100)
    lip_symmetry: float = Field(ge=0, le=100)
    midline_deviation_mm: float = 0.0


class FacialMeasurement(BaseModel):
    """Facial measurement in millimeters / degrees."""
    name: str
    value: float
    unit: str = "mm"  # mm or degrees
    reference_range: Optional[str] = None
    percentile: Optional[float] = None


class FacialMeasurements(BaseModel):
    """All facial measurements."""
    interpupillary_distance: FacialMeasurement
    nasal_width: FacialMeasurement
    nasal_length: FacialMeasurement
    lip_width: FacialMeasurement
    lip_height: FacialMeasurement
    face_width: FacialMeasurement
    face_height: FacialMeasurement
    jawline_angle_left: FacialMeasurement
    jawline_angle_right: FacialMeasurement
    golden_ratio: FacialMeasurement
    facial_thirds: Dict[str, float]  # upper, middle, lower percentages
    nasofrontal_angle: FacialMeasurement
    nasolabial_angle: FacialMeasurement


class HeatmapData(BaseModel):
    """Heatmap visualization data."""
    wrinkle_map: Optional[str] = None   # Base64 PNG
    pore_map: Optional[str] = None
    pigmentation_map: Optional[str] = None
    redness_map: Optional[str] = None
    texture_map: Optional[str] = None
    combined_map: Optional[str] = None


class Recommendation(BaseModel):
    """Treatment recommendation."""
    priority: int = Field(ge=1, le=5)  # 1=highest
    area: str
    condition: str
    suggestion: str
    description: str = ""


# ── Request Models ──────────────────────────────────────────────────────────

class PatientInfo(BaseModel):
    """Patient demographic information."""
    name: str = Field(min_length=1, max_length=200)
    age: Optional[int] = Field(None, ge=1, le=120)
    gender: Optional[str] = None
    skin_type: Optional[str] = None
    fitzpatrick: Optional[FitzpatrickType] = None
    notes: Optional[str] = None


class AnalysisRequest(BaseModel):
    """Single-photo analysis request."""
    session_id: Optional[str] = None
    patient: Optional[PatientInfo] = None
    angle: CaptureAngle = CaptureAngle.FRONT
    # Image sent as multipart file, not in JSON body


class MultiAnalysisRequest(BaseModel):
    """Multi-angle analysis request (5 photos)."""
    session_id: Optional[str] = None
    patient: Optional[PatientInfo] = None
    angles: List[CaptureAngle] = [
        CaptureAngle.FRONT,
        CaptureAngle.LEFT_45,
        CaptureAngle.RIGHT_45,
        CaptureAngle.LEFT_90,
        CaptureAngle.RIGHT_90,
    ]


class AnnotationRequest(BaseModel):
    """Save annotation data for a session."""
    annotations: List[Dict]  # Canvas JSON objects
    notes: str = ""


class CompareRequest(BaseModel):
    """Before/After comparison request."""
    session_id_before: str
    session_id_after: str


# ── Response Models ─────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    patient: Optional[PatientInfo] = None

    # Scores
    overall_score: float = Field(ge=0, le=100)
    skin_age: Optional[int] = None
    zone_scores: Dict[str, ZoneScore] = {}

    # Detections
    conditions: List[ConditionDetection] = []

    # Measurements
    symmetry: Optional[SymmetryResult] = None
    measurements: Optional[FacialMeasurements] = None

    # Visualizations
    heatmaps: Optional[HeatmapData] = None
    annotated_image: Optional[str] = None  # Base64 PNG with overlays
    original_image: Optional[str] = None   # Base64 JPG of input face
    landmarks: Optional[List[Dict]] = None # 468-point face mesh [{x, y}, ...]
    landmarks_3d: Optional[List[Dict]] = None  # 468-point 3D face mesh [{x, y, z}, ...]

    # Recommendations
    recommendations: List[Recommendation] = []

    # Metadata
    processing_time_ms: float = 0
    angles_analyzed: List[CaptureAngle] = []
    resolution: str = "1080x1080"
    device_info: str = ""


class CompareResponse(BaseModel):
    """Before/After comparison response."""
    session_before: AnalysisResponse
    session_after: AnalysisResponse
    improvements: List[Dict]  # {zone, metric, before, after, delta}
    regressions: List[Dict]
    overall_delta: float  # Positive = improvement
    comparison_image: Optional[str] = None  # Base64 side-by-side


class ReportResponse(BaseModel):
    """Report generation response."""
    session_id: str
    report_url: str
    report_base64: Optional[str] = None  # PDF as base64
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    gpu_available: bool = False
    gpu_name: str = ""
    vram_used_mb: float = 0
    vram_total_mb: float = 0
    models_loaded: List[str] = []
    uptime_seconds: float = 0
