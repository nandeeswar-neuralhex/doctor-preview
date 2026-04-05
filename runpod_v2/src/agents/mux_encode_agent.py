"""
Agent 5 — Mux & Encode Agent

Responsibilities:
- Pull synchronized (video, audio) pairs from Agent 4
- Create custom aiortc MediaStreamTrack subclasses for video and audio
- Both tracks use PTS values that were pre-aligned by Agent 4
- When aiortc calls recv() on these tracks, return the next synced frame
- Handle encoding: video → VP8/H.264, audio → Opus (handled by aiortc codec layer)
- Report encode latency, output FPS, bitrate to Agent 6

Architecture:
  Agent 4 pushes SyncedPair → Agent 5 internal queue
  Agent 5 owns two tracks: SyncedVideoTrack + SyncedAudioTrack
  aiortc's RTCPeerConnection calls track.recv() at its own pace
  The tracks return the next available synced frame/chunk

  Since both tracks pull from the SAME SyncedPair, their PTS values
  are guaranteed to be aligned → receiver browser syncs them perfectly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction
from typing import Optional

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame, VideoFrame

from agents.base_agent import BaseAgent
from agents.models import PipelineConfig, SyncedPair

logger = logging.getLogger("agents.mux_encode")

# Standard timebases
VIDEO_TIMEBASE = Fraction(1, 90000)   # 90kHz — standard RTP video
AUDIO_TIMEBASE = Fraction(1, 48000)   # 48kHz — Opus native


class SyncedVideoTrack(MediaStreamTrack):
    """
    Custom video track that serves pre-aligned frames to aiortc.

    aiortc calls recv() to get the next frame to encode and send.
    We pull from the synced pair queue, which Agent 4 has already
    timestamp-aligned with the corresponding audio.
    """
    kind = "video"

    def __init__(self, session_id: str = "", target_fps: int = 30):
        super().__init__()
        self.session_id = session_id
        self._queue: asyncio.Queue[SyncedPair] = asyncio.Queue(maxsize=15)
        self._stopped = False
        self._frame_count = 0
        self._last_log_time = time.time()

        # Track last emitted PTS and resolution for monotonic timeout frames
        self._last_pts = 0
        self._pts_step = int((1.0 / target_fps) / float(VIDEO_TIMEBASE))
        self._last_width = 640
        self._last_height = 480

    async def recv(self) -> VideoFrame:
        """
        Called by aiortc's encoder to get next video frame.

        Returns the video portion of the next SyncedPair with
        the pre-aligned PTS from Agent 4.
        """
        if self._stopped:
            raise Exception("Track stopped")

        try:
            pair: SyncedPair = await asyncio.wait_for(
                self._queue.get(), timeout=0.1
            )
        except asyncio.TimeoutError:
            # No new frame — return a black frame with monotonically increasing PTS
            # and matching resolution to avoid codec re-init
            self._last_pts += self._pts_step
            frame = VideoFrame(width=self._last_width, height=self._last_height)
            frame.pts = self._last_pts
            frame.time_base = VIDEO_TIMEBASE
            return frame

        # Build av.VideoFrame from the processed numpy array
        img = pair.video_frame.image
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = pair.video_pts
        new_frame.time_base = VIDEO_TIMEBASE

        # Track for timeout fallback
        self._last_pts = pair.video_pts
        self._last_width = img.shape[1]
        self._last_height = img.shape[0]

        self._frame_count += 1

        # Periodic logging
        now = time.time()
        if now - self._last_log_time >= 5.0:
            elapsed = now - self._last_log_time
            fps = self._frame_count / elapsed
            logger.debug(
                f"[{self.session_id}] SyncedVideoTrack: output={fps:.1f}fps  "
                f"queue={self._queue.qsize()}"
            )
            self._frame_count = 0
            self._last_log_time = now

        return new_frame

    def feed(self, pair: SyncedPair) -> None:
        """Feed a synced pair into this track (called by MuxEncodeAgent)."""
        try:
            self._queue.put_nowait(pair)
        except asyncio.QueueFull:
            # Drop oldest to keep pipeline moving
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(pair)
            except asyncio.QueueFull:
                pass

    def stop(self):
        self._stopped = True
        super().stop()


class SyncedAudioTrack(MediaStreamTrack):
    """
    Custom audio track that serves pre-aligned audio chunks to aiortc.

    Each audio chunk corresponds to exactly one video frame's duration
    and carries the same PTS alignment from Agent 4.
    """
    kind = "audio"

    # Opus sends 20ms frames at 48kHz = 960 samples per frame
    OPUS_FRAME_SAMPLES = 960
    OPUS_FRAME_DURATION_S = 0.02  # 20ms

    def __init__(self, sample_rate: int = 48000, session_id: str = ""):
        super().__init__()
        self.session_id = session_id
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[SyncedPair] = asyncio.Queue(maxsize=15)
        self._stopped = False
        self._pts = 0

        # Buffer for splitting video-frame-duration audio into 20ms Opus chunks
        self._pending_samples: np.ndarray = np.array([], dtype=np.float32)
        # Track aligned PTS from Agent 4 to stay in sync with video track
        self._base_pts_set = False
        self._chunk_count = 0

    async def recv(self) -> AudioFrame:
        """
        Called by aiortc's Opus encoder to get next audio frame (20ms chunks).

        We split the per-video-frame audio (33ms at 30fps) into 20ms Opus
        chunks, using Agent 4's aligned PTS as the base for continuity.
        """
        if self._stopped:
            raise Exception("Track stopped")

        # If we have enough pending samples, emit an Opus frame
        while len(self._pending_samples) < self.OPUS_FRAME_SAMPLES:
            try:
                pair: SyncedPair = await asyncio.wait_for(
                    self._queue.get(), timeout=0.1
                )
                # Use Agent 4's aligned PTS as our base (first time or on resync)
                if not self._base_pts_set:
                    self._pts = pair.audio_pts
                    self._base_pts_set = True
                self._pending_samples = np.concatenate([
                    self._pending_samples, pair.audio_samples
                ])
            except asyncio.TimeoutError:
                # No audio available — generate silence
                silence = np.zeros(self.OPUS_FRAME_SAMPLES, dtype=np.float32)
                self._pending_samples = np.concatenate([
                    self._pending_samples, silence
                ])

        # Take exactly one Opus frame worth of samples
        chunk = self._pending_samples[:self.OPUS_FRAME_SAMPLES]
        self._pending_samples = self._pending_samples[self.OPUS_FRAME_SAMPLES:]

        # Convert float32 → int16 for AudioFrame
        pcm_int16 = (chunk * 32768.0).clip(-32768, 32767).astype(np.int16)

        # Build av.AudioFrame
        audio_frame = AudioFrame(
            format="s16",
            layout="mono",
            samples=self.OPUS_FRAME_SAMPLES,
        )
        audio_frame.sample_rate = self._sample_rate
        audio_frame.pts = self._pts
        audio_frame.time_base = AUDIO_TIMEBASE

        # Copy PCM data into the frame
        audio_frame.planes[0].update(pcm_int16.tobytes())

        self._pts += self.OPUS_FRAME_SAMPLES
        self._chunk_count += 1

        return audio_frame

    def feed(self, pair: SyncedPair) -> None:
        """Feed a synced pair into this track (called by MuxEncodeAgent)."""
        try:
            self._queue.put_nowait(pair)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(pair)
            except asyncio.QueueFull:
                pass

    def stop(self):
        self._stopped = True
        super().stop()


class MuxEncodeAgent(BaseAgent):
    """
    Takes synchronized A/V pairs from Agent 4 and feeds them
    to the synced video and audio tracks for WebRTC output.

    The actual encoding (VP8/H.264, Opus) is handled by aiortc's
    codec layer — we just provide properly-timed frames.
    """

    def __init__(
        self,
        config: PipelineConfig,
        synced_in_queue: asyncio.Queue,
        metrics_queue: Optional[asyncio.Queue] = None,
        session_id: str = "",
        target_bitrate: Optional[int] = None,
    ):
        super().__init__("mux_encode", metrics_queue)
        self.config = config
        self.session_id = session_id
        self._synced_in = synced_in_queue
        self._target_bitrate = target_bitrate or config.video_bitrate

        # Output tracks (created once, used by aiortc)
        self.video_track = SyncedVideoTrack(session_id=session_id, target_fps=config.target_fps)
        self.audio_track = SyncedAudioTrack(
            sample_rate=config.target_audio_sample_rate,
            session_id=session_id,
        )

        # Stats
        self._pairs_processed = 0
        self._last_stats_time = 0.0

    async def run(self) -> None:
        """Main loop — pull synced pairs and feed to output tracks."""
        self._last_stats_time = time.time()
        logger.info(
            f"[{self.session_id}] MuxEncode started: "
            f"bitrate={self._target_bitrate // 1000}kbps"
        )

        while not self.should_stop:
            try:
                # Wait for next synced pair from Agent 4
                try:
                    pair: SyncedPair = await asyncio.wait_for(
                        self._synced_in.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    self.heartbeat()
                    continue

                t_start = time.time()

                # Feed to both tracks (they'll be consumed by aiortc's send loop)
                self.video_track.feed(pair)
                self.audio_track.feed(pair)

                self._pairs_processed += 1

                # Record metrics
                latency_ms = (time.time() - t_start) * 1000
                self.record_processed(latency_ms)
                await self.report_metric("mux_latency_ms", latency_ms)

                # Periodic logging
                now = time.time()
                if now - self._last_stats_time >= 3.0:
                    elapsed = now - self._last_stats_time
                    output_fps = self._pairs_processed / elapsed
                    logger.info(
                        f"[{self.session_id}] MuxEncode: "
                        f"output={output_fps:.1f}fps  "
                        f"v_queue={self.video_track._queue.qsize()}  "
                        f"a_queue={self.audio_track._queue.qsize()}"
                    )
                    await self.report_metric("output_fps", output_fps)
                    self._pairs_processed = 0
                    self._last_stats_time = now

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._consecutive_errors += 1
                logger.warning(f"[{self.session_id}] MuxEncode error: {exc}")
                await asyncio.sleep(0.01)

    async def cleanup(self) -> None:
        """Stop output tracks on shutdown."""
        try:
            self.video_track.stop()
        except Exception:
            pass
        try:
            self.audio_track.stop()
        except Exception:
            pass
        logger.info(f"[{self.session_id}] MuxEncode agent cleaned up")
