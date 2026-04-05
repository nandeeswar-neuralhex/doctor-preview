"""
Agent 1 — Ingest & Decode Agent

Responsibilities:
- Receive incoming WebRTC audio + video tracks from aiortc
- Decode video frames (VP8/H.264 → raw BGR numpy)
- Decode audio frames (Opus → raw PCM float32)
- Stamp each frame with frame_id, capture_ts, decode_ts
- Push video frames → Agent 2 (Face Swap) input queue
- Push audio frames → Agent 3 (Audio Buffer) input queue
- Report input FPS, decode latency to Agent 6
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import cv2
import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame

from agents.base_agent import BaseAgent
from agents.models import (
    AudioChunkData,
    PipelineConfig,
    VideoFrameData,
)

logger = logging.getLogger("agents.ingest")


class IngestAgent(BaseAgent):
    """
    Receives raw WebRTC tracks, decodes, timestamps, and fans out
    to the video processing queue and audio buffer queue.
    """

    def __init__(
        self,
        config: PipelineConfig,
        video_out_queue: asyncio.Queue,
        audio_out_queue: asyncio.Queue,
        metrics_queue: Optional[asyncio.Queue] = None,
        session_id: str = "",
    ):
        super().__init__("ingest", metrics_queue)
        self.config = config
        self.session_id = session_id

        # Output queues
        self._video_out = video_out_queue
        self._audio_out = audio_out_queue

        # Input tracks (set externally before start)
        self._video_track: Optional[MediaStreamTrack] = None
        self._audio_track: Optional[MediaStreamTrack] = None

        # Sequence counters
        self._video_frame_id = 0
        self._audio_chunk_id = 0

        # Stats
        self._video_count = 0
        self._audio_count = 0
        self._last_stats_time = 0.0
        self._logged_resolution = False

    def set_tracks(
        self,
        video_track: Optional[MediaStreamTrack],
        audio_track: Optional[MediaStreamTrack],
    ) -> None:
        """Set the incoming WebRTC tracks to ingest."""
        self._video_track = video_track
        self._audio_track = audio_track
        logger.info(
            f"[{self.session_id}] Tracks set: "
            f"video={'yes' if video_track else 'no'}, "
            f"audio={'yes' if audio_track else 'no'}"
        )

    async def run(self) -> None:
        """Main loop — spin up reader coroutines for audio and video."""
        self._last_stats_time = time.time()

        tasks = []
        if self._video_track:
            tasks.append(asyncio.create_task(
                self._read_video(), name=f"ingest-video-{self.session_id}"
            ))
        if self._audio_track:
            tasks.append(asyncio.create_task(
                self._read_audio(), name=f"ingest-audio-{self.session_id}"
            ))

        if not tasks:
            logger.warning(f"[{self.session_id}] No tracks to ingest")
            return

        # Wait until all readers finish (they check self.should_stop internally)
        # or until this task is cancelled by the orchestrator
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            # Await cancellation to suppress warnings
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def _read_video(self) -> None:
        """Continuously read video frames from the WebRTC track."""
        logger.info(f"[{self.session_id}] Video reader started")

        while not self.should_stop:
            try:
                frame = await self._video_track.recv()
                decode_ts = time.time()

                # Convert av.VideoFrame → numpy BGR
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]

                if not self._logged_resolution:
                    self._logged_resolution = True
                    logger.info(f"[{self.session_id}] Input resolution: {w}×{h}")

                # Extract capture timestamp from RTP
                capture_ts = float(frame.pts) * float(frame.time_base) if frame.pts else decode_ts

                # Build video frame data
                vf = VideoFrameData(
                    frame_id=self._video_frame_id,
                    capture_ts=capture_ts,
                    decode_ts=decode_ts,
                    image=img,
                    width=w,
                    height=h,
                    keyframe=getattr(frame, 'key_frame', False),
                )
                self._video_frame_id += 1

                # Push to Agent 2 queue — drop oldest if full (backpressure)
                try:
                    self._video_out.put_nowait(vf)
                except asyncio.QueueFull:
                    # Drop the oldest frame to make room
                    try:
                        self._video_out.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self._video_out.put_nowait(vf)
                    except asyncio.QueueFull:
                        pass

                # Stats
                self._video_count += 1
                decode_latency_ms = (decode_ts - capture_ts) * 1000 if capture_ts != decode_ts else 0
                self.record_processed(decode_latency_ms)

                # Periodic stats
                await self._maybe_log_stats()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._consecutive_errors += 1
                if self._consecutive_errors <= 3:
                    logger.warning(f"[{self.session_id}] Video read error: {exc}")
                await asyncio.sleep(0.01)

    async def _read_audio(self) -> None:
        """Continuously read audio frames from the WebRTC track."""
        logger.info(f"[{self.session_id}] Audio reader started")

        while not self.should_stop:
            try:
                frame: AudioFrame = await self._audio_track.recv()
                decode_ts = time.time()

                # Convert av.AudioFrame → numpy float32 PCM
                # AudioFrame.to_ndarray() returns shape (channels, samples) as int16 or float
                raw = frame.to_ndarray()
                if raw.dtype == np.int16:
                    samples = raw.astype(np.float32) / 32768.0
                else:
                    samples = raw.astype(np.float32)

                # Flatten to mono if stereo
                if samples.ndim > 1:
                    samples = samples.mean(axis=0)

                sample_rate = frame.sample_rate
                channels = 1  # We flatten to mono
                num_samples = samples.shape[0]
                duration_ms = (num_samples / sample_rate) * 1000 if sample_rate > 0 else 0

                # Capture timestamp from RTP
                capture_ts = float(frame.pts) * float(frame.time_base) if frame.pts else decode_ts

                ac = AudioChunkData(
                    chunk_id=self._audio_chunk_id,
                    capture_ts=capture_ts,
                    decode_ts=decode_ts,
                    samples=samples,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration_ms=duration_ms,
                )
                self._audio_chunk_id += 1

                # Push to Agent 3 queue
                try:
                    self._audio_out.put_nowait(ac)
                except asyncio.QueueFull:
                    try:
                        self._audio_out.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self._audio_out.put_nowait(ac)
                    except asyncio.QueueFull:
                        pass

                self._audio_count += 1

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._consecutive_errors += 1
                if self._consecutive_errors <= 3:
                    logger.warning(f"[{self.session_id}] Audio read error: {exc}")
                await asyncio.sleep(0.01)

    async def _maybe_log_stats(self) -> None:
        """Log FPS stats every 3 seconds."""
        now = time.time()
        elapsed = now - self._last_stats_time
        if elapsed >= 3.0:
            v_fps = self._video_count / elapsed
            a_pps = self._audio_count / elapsed
            logger.info(
                f"[{self.session_id}] Ingest: video={v_fps:.1f}fps  audio={a_pps:.1f}pps"
            )
            await self.report_metric("input_video_fps", v_fps)
            await self.report_metric("input_audio_pps", a_pps)
            self._video_count = 0
            self._audio_count = 0
            self._last_stats_time = now

    async def cleanup(self) -> None:
        """Stop tracks on shutdown."""
        if self._video_track:
            try:
                self._video_track.stop()
            except Exception:
                pass
        if self._audio_track:
            try:
                self._audio_track.stop()
            except Exception:
                pass
        logger.info(f"[{self.session_id}] Ingest agent cleaned up")
