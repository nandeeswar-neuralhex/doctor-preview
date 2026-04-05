"""
Agent-based A/V Sync Pipeline (Approach 4: Server-Side Muxing)

Architecture:
  Agent 0 (Orchestrator) — supervises all agents, circuit breaker
  Agent 1 (Ingest)       — receives WebRTC tracks, decodes, timestamps
  Agent 2 (Face Swap)    — GPU face swap with processing time tracking
  Agent 3 (Audio Buffer) — time-indexed ring buffer for audio frames
  Agent 4 (PTS Align)    — core sync: aligns audio to processed video
  Agent 5 (Mux & Encode) — encodes and creates synced output tracks
  Agent 6 (Quality)      — real-time AVO, lip-sync confidence, alerts
"""

# Models are pure-Python dataclasses — always importable (no GPU/WebRTC deps)
from agents.models import (
    VideoFrameData,
    AudioChunkData,
    ProcessedVideoFrame,
    SyncedPair,
    AgentHealth,
    PipelineMetrics,
    AlertLevel,
    PipelineConfig,
)
from agents.base_agent import BaseAgent

# Heavy imports (aiortc, av) — lazy-loaded to allow models to be used
# in tests/tools without requiring GPU packages on every machine.


def _lazy_import_orchestrator():
    from agents.orchestrator import OrchestratorAgent
    return OrchestratorAgent


def _lazy_import_pipeline():
    from agents.pipeline import SyncPipeline
    return SyncPipeline


def __getattr__(name):
    """Lazy import for heavy dependencies (aiortc/av)."""
    if name == "SyncPipeline":
        return _lazy_import_pipeline()
    if name == "OrchestratorAgent":
        return _lazy_import_orchestrator()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SyncPipeline",
    "OrchestratorAgent",
    "BaseAgent",
    "VideoFrameData",
    "AudioChunkData",
    "ProcessedVideoFrame",
    "SyncedPair",
    "AgentHealth",
    "PipelineMetrics",
    "AlertLevel",
    "PipelineConfig",
]
