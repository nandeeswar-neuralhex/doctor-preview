"""
Agent 4 — PTS Alignment Agent ⭐ (Most Critical)

Responsibilities:
- Pull processed video frames from Agent 2
- For each video frame, request matching audio from Agent 3's time-indexed buffer
- Compute synchronized PTS values for both audio and video
- Pair them into SyncedPair with matching PTS (= perfect lip sync)
- Handle edge cases: audio not yet available, video dropped, clock drift
- Drift correction via periodic recalibration every N frames
- Push SyncedPairs → Agent 5 (Mux & Encode)
- Report AVO (Audio-Video Offset) metrics to Agent 6

This is the BRAIN of the sync system. If PTS alignment works within
±20ms, the entire pipeline delivers broadcast-quality lip sync.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from fractions import Fraction
from typing import Optional

import numpy as np

from agents.base_agent import BaseAgent
from agents.models import (
    PipelineConfig,
    ProcessedVideoFrame,
    SyncedPair,
)

logger = logging.getLogger("agents.pts_align")

# Output timebase — 90kHz is the standard for RTP/WebRTC video
OUTPUT_TIMEBASE = Fraction(1, 90000)
# Audio timebase — 48kHz for Opus
AUDIO_TIMEBASE = Fraction(1, 48000)


class PTSAlignAgent(BaseAgent):
    """
    Core synchronization engine.

    Algorithm:
      FOR each processed_video_frame:
        1. Get capture_ts from the original frame
        2. Calculate which audio samples were captured at the same time
        3. Request those exact audio samples from Agent 3's buffer
        4. If audio available → pair them → assign same PTS → push to Agent 5
        5. If audio NOT available → wait up to 20ms → if still missing → insert silence
        6. Track clock drift and recalibrate periodically
    """

    def __init__(
        self,
        config: PipelineConfig,
        processed_in_queue: asyncio.Queue,
        audio_buffer_agent,      # Agent 3 instance
        synced_out_queue: asyncio.Queue,
        face_swap_agent=None,    # Agent 2 — for processing time queries
        metrics_queue: Optional[asyncio.Queue] = None,
        session_id: str = "",
    ):
        super().__init__("pts_align", metrics_queue)
        self.config = config
        self.session_id = session_id

        # Queues
        self._processed_in = processed_in_queue
        self._synced_out = synced_out_queue

        # Agent references
        self._audio_buffer = audio_buffer_agent
        self._face_swap_agent = face_swap_agent

        # PTS state
        self._pair_counter = 0
        self._video_pts_counter = 0
        self._audio_pts_counter = 0
        self._frame_duration_s = 1.0 / config.target_fps
        self._video_pts_step = int(self._frame_duration_s / float(OUTPUT_TIMEBASE))
        self._audio_samples_per_frame = int(
            config.target_audio_sample_rate * self._frame_duration_s
        )

        # Drift tracking
        self._drift_history: deque[float] = deque(maxlen=100)
        self._drift_correction_ms = 0.0
        self._frames_since_recalibrate = 0
        self._reference_offset: Optional[float] = None  # NTP offset anchor

        # AVO (Audio-Video Offset) tracking for quality metrics
        self._avo_history: deque[float] = deque(maxlen=300)  # Last 300 pairs (~10s)
        self._silence_insertions = 0

        # Stats
        self._pairs_created = 0
        self._last_stats_time = 0.0

    async def run(self) -> None:
        """Main alignment loop."""
        self._last_stats_time = time.time()
        logger.info(
            f"[{self.session_id}] PTS Align started: "
            f"fps={self.config.target_fps}, "
            f"frame_dur={self._frame_duration_s * 1000:.1f}ms, "
            f"audio_samples/frame={self._audio_samples_per_frame}"
        )

        while not self.should_stop:
            try:
                # Wait for next processed video frame
                try:
                    video_frame: ProcessedVideoFrame = await asyncio.wait_for(
                        self._processed_in.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    self.heartbeat()
                    continue

                t_start = time.time()

                # ── Step 1: Request matching audio ────────────
                audio_samples, audio_sr, actual_audio_ts = await self._get_matching_audio(
                    video_frame.capture_ts
                )

                # ── Step 2: Compute AVO before alignment ──────
                original_offset_ms = 0.0
                if audio_samples is not None and actual_audio_ts > 0:
                    original_offset_ms = (actual_audio_ts - video_frame.capture_ts) * 1000
                    # Store SIGNED offset for drift direction detection
                    self._avo_history.append(original_offset_ms)

                # ── Step 3: Apply drift correction ────────────
                applied_delay_ms = video_frame.processing_duration_ms + self._drift_correction_ms

                # ── Step 4: Assign synchronized PTS ───────────
                video_pts = self._video_pts_counter
                # Apply drift correction to audio PTS to compensate for clock drift
                drift_samples = int(
                    (self._drift_correction_ms / 1000.0)
                    * self.config.target_audio_sample_rate
                )
                audio_pts = self._audio_pts_counter + drift_samples

                # ── Step 5: Build synced pair ─────────────────
                if audio_samples is None:
                    # Generate silence for this frame duration
                    audio_samples = np.zeros(
                        self._audio_samples_per_frame, dtype=np.float32
                    )
                    audio_sr = self.config.target_audio_sample_rate
                    self._silence_insertions += 1
                    await self.report_metric("silence_inserted", 1.0)

                pair = SyncedPair(
                    pair_id=self._pair_counter,
                    synced_pts=video_pts,
                    synced_ts=time.time(),
                    video_frame=video_frame,
                    video_pts=video_pts,
                    audio_samples=audio_samples,
                    audio_sample_rate=audio_sr,
                    audio_channels=1,
                    audio_pts=audio_pts,
                    original_av_offset_ms=original_offset_ms,
                    applied_delay_ms=applied_delay_ms,
                )

                # ── Step 6: Push to Agent 5 ───────────────────
                try:
                    self._synced_out.put_nowait(pair)
                except asyncio.QueueFull:
                    try:
                        self._synced_out.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self._synced_out.put_nowait(pair)

                # Advance PTS counters
                self._video_pts_counter += self._video_pts_step
                self._audio_pts_counter += self._audio_samples_per_frame
                self._pair_counter += 1
                self._pairs_created += 1
                self._frames_since_recalibrate += 1

                # ── Step 7: Drift recalibration ───────────────
                if self._frames_since_recalibrate >= self.config.drift_recalibrate_frames:
                    self._recalibrate_drift()

                # Record metrics
                align_latency_ms = (time.time() - t_start) * 1000
                self.record_processed(align_latency_ms)
                await self.report_metric("align_latency_ms", align_latency_ms)
                await self.report_metric("avo_ms", abs(original_offset_ms))

                # Periodic logging
                await self._maybe_log_stats()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._consecutive_errors += 1
                logger.exception(f"[{self.session_id}] PTS Align error: {exc}")
                await asyncio.sleep(0.01)

    # ── Audio Matching ────────────────────────────────────────

    async def _get_matching_audio(
        self, video_capture_ts: float
    ) -> tuple:
        """
        Request audio from Agent 3 that matches the video frame's capture time.

        If audio isn't immediately available, wait up to max_audio_wait_ms.
        If still unavailable, return None (caller inserts silence).
        """
        # Try immediate fetch
        audio_samples, sr, actual_ts = self._audio_buffer.get_audio_for_timestamp(
            video_capture_ts, self._frame_duration_s
        )

        if audio_samples is not None:
            return audio_samples, sr, actual_ts

        # Audio not available yet — wait with exponential backoff
        total_wait = 0.0
        wait_step_ms = 2.0  # Start with 2ms waits

        while total_wait < self.config.max_audio_wait_ms:
            # Clamp to remaining budget to prevent overshoot
            remaining = self.config.max_audio_wait_ms - total_wait
            actual_wait = min(wait_step_ms, remaining)
            await asyncio.sleep(actual_wait / 1000.0)
            total_wait += actual_wait

            audio_samples, sr, actual_ts = self._audio_buffer.get_audio_for_timestamp(
                video_capture_ts, self._frame_duration_s
            )

            if audio_samples is not None:
                return audio_samples, sr, actual_ts

            # Double wait step (exponential backoff, cap at 10ms)
            wait_step_ms = min(wait_step_ms * 2, 10.0)

        # Audio unavailable after max wait — return None
        logger.debug(
            f"[{self.session_id}] Audio unavailable for ts={video_capture_ts:.3f} "
            f"(waited {total_wait:.0f}ms)"
        )
        return None, self.config.target_audio_sample_rate, 0.0

    # ── Drift Correction ──────────────────────────────────────

    def _recalibrate_drift(self) -> None:
        """
        Recalibrate the drift correction based on recent AVO measurements.

        Clock drift between audio and video capture clocks is real
        (~1ms per minute). We detect and compensate for it here.
        """
        if len(self._avo_history) < 10:
            self._frames_since_recalibrate = 0
            return

        # Compute trend: compare first half vs second half of recent AVO
        half = len(self._avo_history) // 2
        first_half = list(self._avo_history)[:half]
        second_half = list(self._avo_history)[half:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        drift_trend_ms = avg_second - avg_first

        # If drift is growing (> 1ms trend), apply correction
        if abs(drift_trend_ms) > 1.0:
            # Apply correction in the opposite direction of drift
            correction = -drift_trend_ms * 0.5  # Apply 50% correction (conservative)
            self._drift_correction_ms += correction
            self._drift_history.append(self._drift_correction_ms)

            logger.info(
                f"[{self.session_id}] Drift recalibrated: "
                f"trend={drift_trend_ms:+.2f}ms  "
                f"correction={correction:+.2f}ms  "
                f"total_correction={self._drift_correction_ms:+.2f}ms"
            )

        self._frames_since_recalibrate = 0

    # ── Stats ─────────────────────────────────────────────────

    async def _maybe_log_stats(self) -> None:
        """Log alignment stats every 3 seconds."""
        now = time.time()
        if now - self._last_stats_time < 3.0:
            return

        elapsed = now - self._last_stats_time
        pairs_per_sec = self._pairs_created / elapsed

        # Compute AVO percentiles (use abs values for quality metrics)
        avo_p50 = avo_p95 = avo_p99 = 0.0
        if self._avo_history:
            sorted_avo = sorted(abs(v) for v in self._avo_history)
            n = len(sorted_avo)
            avo_p50 = sorted_avo[int(n * 0.50)]
            avo_p95 = sorted_avo[min(int(n * 0.95), n - 1)]
            avo_p99 = sorted_avo[min(int(n * 0.99), n - 1)]

        logger.info(
            f"[{self.session_id}] PTSAlign: "
            f"pairs/s={pairs_per_sec:.1f}  "
            f"AVO P50={avo_p50:.1f}ms  P95={avo_p95:.1f}ms  P99={avo_p99:.1f}ms  "
            f"drift_corr={self._drift_correction_ms:+.1f}ms  "
            f"silence={self._silence_insertions}"
        )

        await self.report_metric("avo_p50_ms", avo_p50)
        await self.report_metric("avo_p95_ms", avo_p95)
        await self.report_metric("avo_p99_ms", avo_p99)
        await self.report_metric("drift_correction_ms", self._drift_correction_ms)
        await self.report_metric("silence_insertions", float(self._silence_insertions))
        await self.report_metric("pairs_per_sec", pairs_per_sec)

        self._pairs_created = 0
        self._last_stats_time = now

    # ── Properties ────────────────────────────────────────────

    @property
    def avo_p95_ms(self) -> float:
        """Current P95 Audio-Video Offset (for quality checks)."""
        if not self._avo_history:
            return 0.0
        sorted_avo = sorted(abs(v) for v in self._avo_history)
        idx = min(int(len(sorted_avo) * 0.95), len(sorted_avo) - 1)
        return sorted_avo[idx]

    @property
    def avo_mean_ms(self) -> float:
        """Mean AVO (absolute value)."""
        if not self._avo_history:
            return 0.0
        return sum(abs(v) for v in self._avo_history) / len(self._avo_history)

    async def cleanup(self) -> None:
        """Clean up on shutdown."""
        logger.info(
            f"[{self.session_id}] PTS Align cleaned up — "
            f"total pairs={self._pair_counter}, "
            f"silence insertions={self._silence_insertions}, "
            f"final drift correction={self._drift_correction_ms:+.1f}ms"
        )
