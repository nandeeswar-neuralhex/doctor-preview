"""
Agent 3 — Audio Buffer Agent

Responsibilities:
- Pull decoded audio chunks from Agent 1's output queue
- Store in a time-indexed ring buffer keyed by capture_ts
- On request from Agent 4: return audio samples for a given timestamp range
- Handle buffer overflow (drop oldest) and underflow (return silence)
- Provide recent audio bytes for lip sync (used by Agent 2)
- Report buffer fill level, underflow/overflow counts to Agent 6
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from agents.base_agent import BaseAgent
from agents.models import AudioChunkData, PipelineConfig

logger = logging.getLogger("agents.audio_buffer")


class AudioBufferAgent(BaseAgent):
    """
    Time-indexed ring buffer for audio frames.

    Holds audio until the corresponding video frame finishes
    processing on the GPU, then Agent 4 requests the matching
    audio for PTS alignment.

    Thread-safe: Agent 2 (GPU thread) calls get_recent_audio_bytes()
    while the async loop writes to the buffer.
    """

    def __init__(
        self,
        config: PipelineConfig,
        audio_in_queue: asyncio.Queue,
        metrics_queue: Optional[asyncio.Queue] = None,
        session_id: str = "",
    ):
        super().__init__("audio_buffer", metrics_queue)
        self.config = config
        self.session_id = session_id
        self._audio_in = audio_in_queue

        # Ring buffer: list of (capture_ts, AudioChunkData) sorted by time
        self._buffer: deque[AudioChunkData] = deque()
        self._lock = threading.Lock()

        # Buffer limits in samples
        self._max_duration_s = config.audio_buffer_capacity_ms / 1000.0

        # Sample rate (detected from first chunk)
        self._sample_rate = config.target_audio_sample_rate

        # Stats
        self._underflow_count = 0
        self._overflow_count = 0
        self._total_chunks = 0
        self._last_stats_time = 0.0

        # Read pointer for Agent 4 queries
        self._read_ts = 0.0

    async def run(self) -> None:
        """Main loop — continuously consume audio chunks from Agent 1."""
        self._last_stats_time = time.time()

        while not self.should_stop:
            try:
                chunk: AudioChunkData = await asyncio.wait_for(
                    self._audio_in.get(), timeout=0.5
                )
            except asyncio.TimeoutError:
                self.heartbeat()
                continue

            with self._lock:
                self._buffer.append(chunk)
                self._sample_rate = chunk.sample_rate
                self._total_chunks += 1

                # Enforce max buffer capacity — drop oldest
                total_duration = self._get_buffer_duration_s()
                while total_duration > self._max_duration_s and len(self._buffer) > 1:
                    self._buffer.popleft()
                    self._overflow_count += 1
                    total_duration = self._get_buffer_duration_s()

            self.heartbeat()
            self.record_processed(0.0)

            # Periodic stats
            now = time.time()
            if now - self._last_stats_time >= 3.0:
                fill_pct = self.buffer_fill_pct
                logger.info(
                    f"[{self.session_id}] AudioBuffer: fill={fill_pct:.0f}%  "
                    f"chunks={len(self._buffer)}  "
                    f"underflows={self._underflow_count}  "
                    f"overflows={self._overflow_count}"
                )
                await self.report_metric("buffer_fill_pct", fill_pct)
                await self.report_metric("buffer_underflows", float(self._underflow_count))
                await self.report_metric("buffer_overflows", float(self._overflow_count))
                self._last_stats_time = now

    # ── Public API (called by Agent 4 — PTS Align) ────────────

    def get_audio_for_timestamp(
        self,
        target_ts: float,
        duration_s: float,
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Get audio samples that correspond to a given video capture timestamp.

        Args:
            target_ts: The capture_ts of the video frame.
            duration_s: How much audio to return (typically 1/fps seconds).

        Returns:
            (samples, sample_rate, actual_ts) or (None, sample_rate, 0.0) if unavailable.
            samples: float32 numpy array of PCM audio.
        """
        with self._lock:
            if not self._buffer:
                self._underflow_count += 1
                return None, self._sample_rate, 0.0

            # Find chunks that overlap [target_ts, target_ts + duration_s]
            target_end = target_ts + duration_s
            matching_chunks: List[AudioChunkData] = []

            for chunk in self._buffer:
                chunk_end = chunk.capture_ts + (chunk.duration_ms / 1000.0)
                # Check overlap
                if chunk.capture_ts <= target_end and chunk_end >= target_ts:
                    matching_chunks.append(chunk)

            if not matching_chunks:
                # No matching audio — check if it's a timing issue
                buffer_start = self._buffer[0].capture_ts if self._buffer else 0
                buffer_end = (
                    self._buffer[-1].capture_ts + self._buffer[-1].duration_ms / 1000.0
                    if self._buffer
                    else 0
                )

                if target_ts < buffer_start:
                    # Audio already evicted (video took too long)
                    self._underflow_count += 1
                    return None, self._sample_rate, 0.0
                elif target_ts > buffer_end:
                    # Audio hasn't arrived yet
                    self._underflow_count += 1
                    return None, self._sample_rate, 0.0
                else:
                    self._underflow_count += 1
                    return None, self._sample_rate, 0.0

            # Concatenate matching audio samples
            all_samples = np.concatenate([c.samples for c in matching_chunks])
            actual_ts = matching_chunks[0].capture_ts

            # Trim to requested duration
            needed_samples = int(duration_s * self._sample_rate)
            if len(all_samples) > needed_samples:
                # Calculate offset into the first chunk
                offset_s = max(0, target_ts - actual_ts)
                offset_samples = int(offset_s * self._sample_rate)
                start = min(offset_samples, len(all_samples) - needed_samples)
                all_samples = all_samples[start:start + needed_samples]
                actual_ts = target_ts
            elif len(all_samples) < needed_samples:
                # Pad with silence
                pad = np.zeros(needed_samples - len(all_samples), dtype=np.float32)
                all_samples = np.concatenate([all_samples, pad])

            return all_samples, self._sample_rate, actual_ts

    def get_recent_audio_bytes(self) -> Tuple[bytes, int]:
        """
        Get recent audio as PCM int16 bytes for lip sync (Agent 2).

        Returns the last ~300ms of audio in the buffer.
        This is thread-safe (called from GPU thread).
        """
        with self._lock:
            if not self._buffer:
                return b'', self._sample_rate

            # Collect last 300ms worth of chunks
            target_duration_s = 0.3
            chunks = []
            total_duration = 0.0

            for chunk in reversed(self._buffer):
                chunks.insert(0, chunk)
                total_duration += chunk.duration_ms / 1000.0
                if total_duration >= target_duration_s:
                    break

            if not chunks:
                return b'', self._sample_rate

            all_samples = np.concatenate([c.samples for c in chunks])

            # Convert float32 → int16 bytes for lip syncer compatibility
            pcm_int16 = (all_samples * 32768.0).clip(-32768, 32767).astype(np.int16)
            return pcm_int16.tobytes(), self._sample_rate

    # ── Properties ────────────────────────────────────────────

    @property
    def buffer_fill_pct(self) -> float:
        """Current buffer fill level as a percentage of capacity."""
        with self._lock:
            if self._max_duration_s <= 0:
                return 0.0
            current = self._get_buffer_duration_s()
            return min(100.0, (current / self._max_duration_s) * 100.0)

    @property
    def buffer_duration_ms(self) -> float:
        """Current buffered audio duration in milliseconds."""
        with self._lock:
            return self._get_buffer_duration_s() * 1000.0

    @property
    def oldest_timestamp(self) -> float:
        """Oldest audio chunk's capture timestamp."""
        with self._lock:
            if self._buffer:
                return self._buffer[0].capture_ts
            return 0.0

    @property
    def newest_timestamp(self) -> float:
        """Newest audio chunk's capture timestamp."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1].capture_ts
            return 0.0

    # ── Internal ──────────────────────────────────────────────

    def _get_buffer_duration_s(self) -> float:
        """Total buffered audio duration (must hold lock)."""
        if not self._buffer:
            return 0.0
        return sum(c.duration_ms for c in self._buffer) / 1000.0

    async def cleanup(self) -> None:
        """Clear buffer on shutdown."""
        with self._lock:
            self._buffer.clear()
        logger.info(f"[{self.session_id}] Audio buffer agent cleaned up")
