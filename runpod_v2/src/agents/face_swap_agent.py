"""
Agent 2 — Face Swap Processing Agent

Responsibilities:
- Pull raw video frames from Agent 1's output queue
- Run face detection → face swap → optional lip sync
- Stamp each frame with processing_start_ts, processing_end_ts, processing_duration_ms
- Push processed frames → Agent 4 (PTS Align) input queue
- Report processing time per frame to Agent 6
- Downscale to 720p if needed (face swap uses 128×128 crops regardless)
- Frame skip logic if queue backs up
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from agents.base_agent import BaseAgent
from agents.models import (
    PipelineConfig,
    ProcessedVideoFrame,
    VideoFrameData,
)

logger = logging.getLogger("agents.face_swap")


class FaceSwapAgent(BaseAgent):
    """
    GPU-bound face swap processing agent.

    Uses a dedicated thread for GPU inference (ONNX/CUDA cannot be
    awaited) with an async wrapper for queue integration.
    """

    def __init__(
        self,
        swapper,          # FaceSwapper instance
        lip_syncer,       # LipSyncer instance (or None)
        config: PipelineConfig,
        video_in_queue: asyncio.Queue,
        audio_buffer_agent,  # Agent 3 — for lip sync audio
        processed_out_queue: asyncio.Queue,
        metrics_queue: Optional[asyncio.Queue] = None,
        session_id: str = "",
        session_settings: Optional[dict] = None,
        enable_lipsync: bool = True,
    ):
        super().__init__("face_swap", metrics_queue)
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.config = config
        self.session_id = session_id
        self.session_settings = session_settings or {}
        self.enable_lipsync = enable_lipsync

        # Queues
        self._video_in = video_in_queue
        self._processed_out = processed_out_queue
        self._audio_buffer_agent = audio_buffer_agent

        # Processing thread
        self._thread: Optional[threading.Thread] = None
        self._thread_stop = threading.Event()
        self._thread_input_queue = asyncio.Queue(maxsize=5)

        # Stats
        self._processing_times: deque = deque(maxlen=100)
        self._swap_count = 0
        self._drop_count = 0
        self._last_stats_time = 0.0

    async def run(self) -> None:
        """Main loop — read from video_in queue and dispatch to GPU thread."""
        self._last_stats_time = time.time()
        self._thread_stop.clear()

        # Start GPU processing thread
        loop = asyncio.get_event_loop()
        self._thread = threading.Thread(
            target=self._gpu_process_loop,
            args=(loop,),
            daemon=True,
            name=f"face-swap-{self.session_id}",
        )
        self._thread.start()
        logger.info(f"[{self.session_id}] Face swap GPU thread started")

        # Async reader — pull from ingest queue and feed to GPU thread
        try:
            while not self.should_stop:
                try:
                    frame: VideoFrameData = await asyncio.wait_for(
                        self._video_in.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    self.heartbeat()
                    continue

                # Backpressure: if GPU thread is backed up, drop this frame
                if self._thread_input_queue.full():
                    self._drop_count += 1
                    await self.report_metric("frames_dropped", 1.0)
                    continue

                await self._thread_input_queue.put(frame)
                self.heartbeat()

        except asyncio.CancelledError:
            pass
        finally:
            self._thread_stop.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)

    def _gpu_process_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Background thread: process video frames on GPU.

        Runs face swap + optional lip sync, then pushes results
        back to the async processed_out queue via the event loop.
        """
        logger.info(f"[{self.session_id}] GPU processing thread running")

        while not self._thread_stop.is_set():
            # Get next frame from the thread-safe queue
            try:
                # Use asyncio to get from async queue in this thread
                future = asyncio.run_coroutine_threadsafe(
                    self._thread_input_queue.get(), loop
                )
                frame: VideoFrameData = future.result(timeout=0.5)
            except Exception:
                continue

            processing_start = time.time()

            try:
                img = frame.image

                # Downscale to 720p if larger
                h, w = img.shape[:2]
                if h > self.config.max_process_height:
                    scale = self.config.max_process_height / h
                    new_w = int(w * scale)
                    new_h = self.config.max_process_height
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                # Determine if lip sync will run
                will_lipsync = (
                    self.enable_lipsync
                    and self.lip_syncer
                    and self.lip_syncer.is_ready()
                )

                # Face swap (GPU) — skip mouth preservation if lip sync will overwrite it
                result, faces = self.swapper.swap_face_with_faces(
                    self.session_id, img, skip_mouth_preservation=will_lipsync
                )

                # Lip sync — apply using audio from Agent 3's buffer
                lipsync_applied = False
                if will_lipsync and len(faces) > 0 and self._audio_buffer_agent:
                    try:
                        audio_pcm, sample_rate = self._audio_buffer_agent.get_recent_audio_bytes()
                        if audio_pcm and len(audio_pcm) > 0:
                            mel = self.lip_syncer.audio_to_mel(audio_pcm, sample_rate)
                            if mel is not None:
                                face = max(
                                    faces,
                                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                                )
                                x1, y1, x2, y2 = face.bbox.astype(int)
                                x1, y1 = max(0, x1), max(0, y1)
                                x2 = min(result.shape[1], x2)
                                y2 = min(result.shape[0], y2)
                                if x2 > x1 and y2 > y1:
                                    face_crop = result[y1:y2, x1:x2]
                                    synced = self.lip_syncer.infer(face_crop, mel, self.session_id)
                                    if synced is not None:
                                        result = self.lip_syncer.apply_mouth_only(
                                            result, (x1, y1, x2, y2), synced
                                        )
                                        lipsync_applied = True
                    except Exception as exc:
                        logger.debug(f"[{self.session_id}] Lip sync error: {exc}")

                processing_end = time.time()
                processing_duration_ms = (processing_end - processing_start) * 1000

                # Build processed frame
                out_h, out_w = result.shape[:2]
                processed = ProcessedVideoFrame(
                    frame_id=frame.frame_id,
                    capture_ts=frame.capture_ts,
                    decode_ts=frame.decode_ts,
                    processing_start_ts=processing_start,
                    processing_end_ts=processing_end,
                    processing_duration_ms=processing_duration_ms,
                    image=result,
                    width=out_w,
                    height=out_h,
                    faces_detected=len(faces),
                    lipsync_applied=lipsync_applied,
                )

                # Push to processed output queue (non-blocking)
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._put_processed(processed), loop
                    )
                    future.result(timeout=0.1)
                except Exception:
                    # Queue full — drop processed frame
                    self._drop_count += 1

                # Record stats
                self._processing_times.append(processing_duration_ms)
                self._swap_count += 1

                # Report metrics
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.report_metric("swap_latency_ms", processing_duration_ms),
                        loop,
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.report_metric("faces_detected", float(len(faces))),
                        loop,
                    )
                except Exception:
                    pass

                self.record_processed(processing_duration_ms)

                # Periodic logging
                now = time.time()
                if now - self._last_stats_time >= 3.0:
                    elapsed = now - self._last_stats_time
                    swap_fps = self._swap_count / elapsed
                    avg_ms = (
                        sum(self._processing_times) / len(self._processing_times)
                        if self._processing_times
                        else 0
                    )
                    stddev_ms = (
                        (sum((x - avg_ms) ** 2 for x in self._processing_times) / len(self._processing_times)) ** 0.5
                        if len(self._processing_times) > 1
                        else 0
                    )
                    logger.info(
                        f"[{self.session_id}] FaceSwap: "
                        f"fps={swap_fps:.1f}  avg={avg_ms:.1f}ms  "
                        f"stddev={stddev_ms:.1f}ms  drops={self._drop_count}"
                    )
                    self._swap_count = 0
                    self._drop_count = 0
                    self._last_stats_time = now

            except Exception as exc:
                self._consecutive_errors += 1
                logger.warning(f"[{self.session_id}] Face swap error: {exc}")

        logger.info(f"[{self.session_id}] GPU processing thread stopped")

    async def _put_processed(self, frame: ProcessedVideoFrame) -> None:
        """Push processed frame to output queue with backpressure."""
        try:
            self._processed_out.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest to make room
            try:
                self._processed_out.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._processed_out.put_nowait(frame)

    async def cleanup(self) -> None:
        """Signal thread to stop."""
        self._thread_stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info(f"[{self.session_id}] Face swap agent cleaned up")

    @property
    def avg_processing_time_ms(self) -> float:
        """Rolling average of processing time (for Agent 4 to query)."""
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)

    @property
    def processing_time_stddev_ms(self) -> float:
        """Standard deviation of processing time."""
        if len(self._processing_times) < 2:
            return 0.0
        avg = self.avg_processing_time_ms
        variance = sum((x - avg) ** 2 for x in self._processing_times) / len(self._processing_times)
        return variance ** 0.5
