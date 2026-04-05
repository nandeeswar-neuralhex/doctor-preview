"""
SyncPipeline — Wires all 7 agents into a single A/V sync pipeline.

Usage:
    pipeline = SyncPipeline(session_id, swapper, lip_syncer, config)
    pipeline.set_tracks(video_track, audio_track)
    await pipeline.start()

    # Get output tracks for aiortc
    pc.addTrack(pipeline.video_output_track)
    pc.addTrack(pipeline.audio_output_track)

    # On disconnect:
    await pipeline.stop()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from aiortc import MediaStreamTrack

from agents.models import PipelineConfig
from agents.orchestrator import OrchestratorAgent
from agents.ingest_agent import IngestAgent
from agents.face_swap_agent import FaceSwapAgent
from agents.audio_buffer_agent import AudioBufferAgent
from agents.pts_align_agent import PTSAlignAgent
from agents.mux_encode_agent import MuxEncodeAgent
from agents.quality_monitor import QualityMonitorAgent

logger = logging.getLogger("agents.pipeline")


class SyncPipeline:
    """
    Full A/V synchronization pipeline using 7 agents.

    Creates and wires:
      Agent 1 (Ingest)       → video queue → Agent 2 (FaceSwap)
      Agent 1 (Ingest)       → audio queue → Agent 3 (AudioBuffer)
      Agent 2 (FaceSwap)     → processed queue → Agent 4 (PTSAlign)
      Agent 3 (AudioBuffer)  ← queried by → Agent 4 (PTSAlign)
      Agent 4 (PTSAlign)     → synced queue → Agent 5 (MuxEncode)
      Agent 5 (MuxEncode)    → SyncedVideoTrack + SyncedAudioTrack
      Agent 6 (QualityMonitor) ← metrics from all agents
      Agent 0 (Orchestrator) → supervises all

    The output tracks are added to the RTCPeerConnection by webrtc.py.
    """

    def __init__(
        self,
        session_id: str,
        swapper,           # FaceSwapper instance
        lip_syncer,        # LipSyncer instance (or None)
        config: Optional[PipelineConfig] = None,
        target_bitrate: Optional[int] = None,
        enable_lipsync: bool = True,
        session_settings: Optional[dict] = None,
    ):
        self.session_id = session_id
        self.config = config or PipelineConfig()
        self._started = False

        # ── Create inter-agent queues ─────────────────────────
        self._video_queue = asyncio.Queue(maxsize=self.config.video_queue_size)
        self._audio_queue = asyncio.Queue(maxsize=self.config.audio_queue_size)
        self._processed_queue = asyncio.Queue(maxsize=self.config.processed_queue_size)
        self._synced_queue = asyncio.Queue(maxsize=self.config.synced_queue_size)
        self._metrics_queue = asyncio.Queue(maxsize=self.config.metrics_queue_size)
        self._alert_queue = asyncio.Queue(maxsize=20)

        # ── Create agents ─────────────────────────────────────

        # Agent 1: Ingest & Decode
        self.ingest = IngestAgent(
            config=self.config,
            video_out_queue=self._video_queue,
            audio_out_queue=self._audio_queue,
            metrics_queue=self._metrics_queue,
            session_id=session_id,
        )

        # Agent 3: Audio Buffer (created before Agent 2 because Agent 2 needs it)
        self.audio_buffer = AudioBufferAgent(
            config=self.config,
            audio_in_queue=self._audio_queue,
            metrics_queue=self._metrics_queue,
            session_id=session_id,
        )

        # Agent 2: Face Swap
        self.face_swap = FaceSwapAgent(
            swapper=swapper,
            lip_syncer=lip_syncer,
            config=self.config,
            video_in_queue=self._video_queue,
            audio_buffer_agent=self.audio_buffer,
            processed_out_queue=self._processed_queue,
            metrics_queue=self._metrics_queue,
            session_id=session_id,
            session_settings=session_settings,
            enable_lipsync=enable_lipsync,
        )

        # Agent 4: PTS Alignment
        self.pts_align = PTSAlignAgent(
            config=self.config,
            processed_in_queue=self._processed_queue,
            audio_buffer_agent=self.audio_buffer,
            synced_out_queue=self._synced_queue,
            face_swap_agent=self.face_swap,
            metrics_queue=self._metrics_queue,
            session_id=session_id,
        )

        # Agent 5: Mux & Encode
        self.mux_encode = MuxEncodeAgent(
            config=self.config,
            synced_in_queue=self._synced_queue,
            metrics_queue=self._metrics_queue,
            session_id=session_id,
            target_bitrate=target_bitrate,
        )

        # Agent 6: Quality Monitor
        self.quality_monitor = QualityMonitorAgent(
            config=self.config,
            metrics_queue=self._metrics_queue,
            alert_queue=self._alert_queue,
            session_id=session_id,
        )

        # Agent 0: Orchestrator
        self.orchestrator = OrchestratorAgent(
            session_id=session_id,
            config=self.config,
            metrics_queue=self._metrics_queue,
            alert_queue=self._alert_queue,
        )

        # Register all agents with orchestrator
        self.orchestrator.register_agents([
            self.ingest,
            self.face_swap,
            self.audio_buffer,
            self.pts_align,
            self.mux_encode,
            self.quality_monitor,
        ])

        logger.info(
            f"[{session_id}] SyncPipeline created with 7 agents, "
            f"queues: v={self.config.video_queue_size} "
            f"a={self.config.audio_queue_size} "
            f"p={self.config.processed_queue_size} "
            f"s={self.config.synced_queue_size}"
        )

    # ── Public API ────────────────────────────────────────────

    def set_tracks(
        self,
        video_track: Optional[MediaStreamTrack],
        audio_track: Optional[MediaStreamTrack],
    ) -> None:
        """Set the incoming WebRTC tracks before starting the pipeline."""
        self.ingest.set_tracks(video_track, audio_track)

    @property
    def video_output_track(self) -> MediaStreamTrack:
        """Get the synced video output track for RTCPeerConnection."""
        return self.mux_encode.video_track

    @property
    def audio_output_track(self) -> MediaStreamTrack:
        """Get the synced audio output track for RTCPeerConnection."""
        return self.mux_encode.audio_track

    @property
    def is_healthy(self) -> bool:
        """Whether the pipeline is operating within quality targets."""
        return self.orchestrator.is_pipeline_healthy

    @property
    def is_fallback_active(self) -> bool:
        """Whether the pipeline has fallen back to adaptive delay mode."""
        return self.orchestrator.is_fallback_active

    async def start(self) -> None:
        """Start the full pipeline (all 7 agents)."""
        if self._started:
            logger.warning(f"[{self.session_id}] Pipeline already started")
            return

        logger.info(f"[{self.session_id}] Starting A/V sync pipeline...")
        await self.orchestrator.start_pipeline()
        self._started = True
        logger.info(f"[{self.session_id}] ✅ A/V sync pipeline running")

    async def stop(self) -> None:
        """Stop the full pipeline gracefully."""
        if not self._started:
            return

        logger.info(f"[{self.session_id}] Stopping A/V sync pipeline...")
        await self.orchestrator.stop_pipeline()
        self._started = False
        logger.info(f"[{self.session_id}] A/V sync pipeline stopped")

    def get_metrics(self) -> Optional[dict]:
        """Get latest quality metrics snapshot."""
        report = self.quality_monitor.latest_report
        if not report:
            return None
        return {
            "input_video_fps": report.input_video_fps,
            "swap_latency_ms": report.swap_latency_ms,
            "avo_p95_ms": report.avo_p95_ms,
            "buffer_fill_pct": report.buffer_fill_pct,
            "output_fps": report.output_fps,
            "e2e_latency_ms": report.end_to_end_latency_ms,
            "frame_drop_pct": report.frame_drop_pct,
            "pipeline_healthy": self.is_healthy,
            "fallback_active": self.is_fallback_active,
        }
