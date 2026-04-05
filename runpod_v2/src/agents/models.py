"""
Shared data models for the A/V Sync Agent Pipeline.

All agents communicate via these structures through asyncio.Queues.
Every frame/chunk carries timestamps for precise PTS alignment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any

import numpy as np


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────

class AlertLevel(Enum):
    """Quality alert severity."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


class AgentState(Enum):
    """Lifecycle state of an agent."""
    IDLE = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


# ─────────────────────────────────────────────────────────────
# Core Frame / Chunk Models
# ─────────────────────────────────────────────────────────────

@dataclass
class VideoFrameData:
    """Raw decoded video frame from Agent 1 (Ingest).

    Attributes:
        frame_id: Monotonically increasing sequence number.
        capture_ts: Original RTP/NTP timestamp from the WebRTC source (seconds).
        decode_ts: Wall-clock time when decode completed (seconds).
        image: BGR numpy array (H, W, 3), dtype uint8.
        width: Frame width in pixels.
        height: Frame height in pixels.
        keyframe: Whether this is an IDR/keyframe.
    """
    frame_id: int
    capture_ts: float
    decode_ts: float
    image: np.ndarray
    width: int
    height: int
    keyframe: bool = False


@dataclass
class AudioChunkData:
    """Raw decoded audio chunk from Agent 1 (Ingest).

    Attributes:
        chunk_id: Monotonically increasing sequence number.
        capture_ts: Original RTP/NTP timestamp (seconds).
        decode_ts: Wall-clock time when decode completed (seconds).
        samples: Raw PCM samples as numpy float32 array.
        sample_rate: Audio sample rate (e.g. 48000).
        channels: Number of audio channels (typically 1 for Opus mono).
        duration_ms: Duration of this chunk in milliseconds.
    """
    chunk_id: int
    capture_ts: float
    decode_ts: float
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_ms: float


@dataclass
class ProcessedVideoFrame:
    """Video frame after face-swap processing from Agent 2.

    Carries the original timestamps PLUS processing timing so
    Agent 4 (PTS Align) knows exactly how much delay the swap introduced.
    """
    frame_id: int
    capture_ts: float           # Original capture time
    decode_ts: float            # When decoded
    processing_start_ts: float  # When face swap began
    processing_end_ts: float    # When face swap finished
    processing_duration_ms: float
    image: np.ndarray           # Processed BGR array
    width: int
    height: int
    faces_detected: int         # Number of faces found
    lipsync_applied: bool       # Whether lip-sync was also applied


@dataclass
class SyncedPair:
    """Synchronized audio + video pair from Agent 4 (PTS Align).

    Both audio and video carry the SAME synced_pts value, guaranteeing
    they will be rendered at the same time by the receiver.
    """
    pair_id: int
    synced_pts: int             # Shared presentation timestamp (timebase units)
    synced_ts: float            # Wall-clock time of sync (seconds)

    # Video
    video_frame: ProcessedVideoFrame
    video_pts: int              # Video PTS in output timebase

    # Audio
    audio_samples: np.ndarray   # PCM float32 samples for this video frame duration
    audio_sample_rate: int
    audio_channels: int
    audio_pts: int              # Audio PTS in output timebase

    # Sync quality
    original_av_offset_ms: float  # How far apart audio/video were BEFORE alignment
    applied_delay_ms: float       # How much delay was applied to audio


# ─────────────────────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Configuration for the sync pipeline.

    Tunable parameters for buffer sizes, thresholds, and quality targets.
    """
    # Target output
    target_fps: int = 30
    target_audio_sample_rate: int = 48000

    # Agent 3: Audio buffer
    audio_buffer_capacity_ms: float = 500.0    # Max buffered audio
    audio_buffer_min_ms: float = 50.0          # Min buffer before output

    # Agent 4: PTS alignment
    max_audio_wait_ms: float = 20.0            # Max time to wait for audio
    drift_recalibrate_frames: int = 300        # Recalibrate every N frames
    max_tolerable_offset_ms: float = 40.0      # ITU-R BT.1359 broadcast standard

    # Agent 5: Encode
    video_codec: str = "libvpx"                # or "h264_nvenc" for GPU encode
    video_bitrate: int = 3_000_000             # 3 Mbps
    audio_codec: str = "opus"
    audio_bitrate: int = 64_000                # 64 kbps

    # Agent 6: Quality thresholds
    avo_warning_ms: float = 30.0               # Warn if AVO P95 > this
    avo_critical_ms: float = 60.0              # Critical if AVO P95 > this
    buffer_fill_warning_pct: float = 85.0
    min_lipsync_confidence: float = 85.0       # Percent
    max_frame_drop_pct: float = 2.0

    # Queue sizes (backpressure)
    video_queue_size: int = 10
    audio_queue_size: int = 50
    processed_queue_size: int = 10
    synced_queue_size: int = 10
    metrics_queue_size: int = 100

    # Circuit breaker
    max_consecutive_failures: int = 3
    fallback_to_adaptive_delay: bool = True

    # Video processing
    max_process_height: int = 720


# ─────────────────────────────────────────────────────────────
# Agent Health & Metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentHealth:
    """Health snapshot for a single agent."""
    agent_name: str
    state: AgentState
    last_heartbeat: float = 0.0
    consecutive_errors: int = 0
    total_processed: int = 0
    avg_latency_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Aggregated pipeline metrics from Agent 6."""
    timestamp: float = 0.0
    session_id: str = ""

    # Agent 1 - Ingest
    input_video_fps: float = 0.0
    input_audio_pps: float = 0.0     # Packets per second
    decode_latency_ms: float = 0.0

    # Agent 2 - Face Swap
    swap_fps: float = 0.0
    swap_latency_ms: float = 0.0
    swap_latency_stddev_ms: float = 0.0
    faces_detected_avg: float = 0.0

    # Agent 3 - Audio Buffer
    buffer_fill_pct: float = 0.0
    buffer_underflows: int = 0
    buffer_overflows: int = 0

    # Agent 4 - PTS Align
    avo_mean_ms: float = 0.0         # Audio-Video Offset mean
    avo_p50_ms: float = 0.0
    avo_p95_ms: float = 0.0
    avo_p99_ms: float = 0.0
    drift_correction_ms: float = 0.0
    silence_insertions: int = 0

    # Agent 5 - Encode
    encode_latency_ms: float = 0.0
    output_fps: float = 0.0
    output_bitrate_kbps: float = 0.0

    # Overall
    end_to_end_latency_ms: float = 0.0
    frame_drop_pct: float = 0.0
    lipsync_confidence: float = 0.0

    # Alerts
    active_warnings: int = 0
    active_criticals: int = 0


@dataclass
class MetricEvent:
    """Single metric data point sent from any agent to Agent 6."""
    agent_name: str
    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertEvent:
    """Alert from Agent 6 to Orchestrator."""
    level: AlertLevel
    agent_name: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
