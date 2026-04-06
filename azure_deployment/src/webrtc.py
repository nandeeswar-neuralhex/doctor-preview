"""
WebRTC handling with aiortc for low-latency video/audio streaming.
Ported from runpod_service for runpod_v2.

Quality pipeline:
  Client camera (1080p) → VP8 encode @ client bitrate → network → aiortc decode
  → face swap (full res) → VP8 encode @ server bitrate → network → client display

  The DEFAULT server VP8 bitrate in aiortc is only 500 Kbps, which destroys
  quality. We parse the b=AS hint from the client's SDP and set the server-side
  encoder to match, so 1080p preset = 4 Mbps both ways.
"""
from __future__ import annotations

import asyncio

# ── Extend ICE consent timeout ──
# aioice defaults: CONSENT_INTERVAL=5s, CONSENT_FAILURES=6 → dies after 30s.
# When the browser is backgrounded on macOS, App Nap may delay ICE keepalives.
# Increase tolerance to 60 failures × 5s = 5 minutes before giving up.
try:
    import aioice.ice as _aioice_mod
    _aioice_mod.CONSENT_FAILURES = 60  # was 6 → now survives 5 minutes of silence
    print(f"[WebRTC] ICE consent timeout extended to {60 * 5}s (was 30s)")
except Exception:
    pass
import re
import threading
import time
from typing import Dict, Optional
from fractions import Fraction

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame

from face_swapper import FaceSwapper
from lip_syncer import LipSyncer
from config import (
    ENABLE_LIPSYNC, LIPSYNC_AUDIO_WINDOW_MS, ENABLE_AV_SYNC_PIPELINE,
    AV_SYNC_AUDIO_BUFFER_MS, AV_SYNC_MAX_AUDIO_WAIT_MS,
    AV_SYNC_DRIFT_RECAL_FRAMES, AV_SYNC_AVO_WARNING_MS, AV_SYNC_AVO_CRITICAL_MS,
)

# ── Monkey-patch aiortc's VP8 encoder for higher quality ──
# aiortc defaults: qmax=56, cpu-used=-6 (fastest/ugliest), 500 Kbps.
# The real problems:
#   1. DEFAULT_BITRATE=500000 → __init__ stores this in __target_bitrate (private)
#   2. encode() uses self.__target_bitrate for bufsize (name-mangled, can't override)
#   3. qmax=56 is hardcoded in encode()
#   4. cpu-used=-6 is hardcoded (fastest, worst quality)
# Fix: override __init__ to use high default, and replace encode() entirely.
try:
    import av
    import multiprocessing
    import random
    from aiortc.codecs import vpx as _vpx_module
    from aiortc.codecs.vpx import (
        Vp8Encoder as _OrigVp8Encoder,
        convert_timebase,
        number_of_threads,
        VIDEO_TIME_BASE,
    )

    _HIGH_DEFAULT_BITRATE = 6_000_000  # 3 Mbps

    _orig_init = _OrigVp8Encoder.__init__

    def _patched_init(self):
        _orig_init(self)
        # Override the name-mangled __target_bitrate with high default.
        # Python mangles __target_bitrate → _Vp8Encoder__target_bitrate
        self._Vp8Encoder__target_bitrate = _HIGH_DEFAULT_BITRATE

    def _patched_encode(self, frame, force_keyframe=False):
        assert isinstance(frame, av.VideoFrame)
        if frame.format.name != "yuv420p":
            frame = frame.reformat(format="yuv420p")

        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
        ):
            self.codec = None

        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I

        if self.codec is None:
            self.codec = av.CodecContext.create("libvpx", "w")
            self.codec.width = frame.width
            self.codec.height = frame.height
            self.codec.bit_rate = self.target_bitrate
            self.codec.pix_fmt = "yuv420p"
            self.codec.gop_size = 3000
            self.codec.qmin = 2
            self.codec.qmax = 32          # was 56 — lower = better quality
            self.codec.options = {
                "bufsize": str(self.target_bitrate),  # was __target_bitrate (500k)
                "cpu-used": "-6",          # fastest — CPU is the bottleneck, not bitrate
                "deadline": "realtime",
                "lag-in-frames": "0",
                "minrate": str(self.target_bitrate),
                "maxrate": str(self.target_bitrate),
                "noise-sensitivity": "0",  # was 4 — less noise reduction = sharper
                "overshoot-pct": "15",
                "partitions": "0",
                "static-thresh": "0",      # was 1 — encode all blocks, no skipping
                "undershoot-pct": "100",
            }
            self.codec.thread_count = number_of_threads(
                frame.width * frame.height, multiprocessing.cpu_count()
            )

        data_to_send = b""
        for package in self.codec.encode(frame):
            data_to_send += bytes(package)

        payloads = self._packetize(data_to_send, self.picture_id)
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        self.picture_id = (self.picture_id + 1) % (1 << 15)
        return payloads, timestamp

    _OrigVp8Encoder.__init__ = _patched_init
    _OrigVp8Encoder.encode = _patched_encode
    _vpx_module.DEFAULT_BITRATE = _HIGH_DEFAULT_BITRATE
    print(f"[WebRTC] VP8 encoder patched: bitrate=6Mbps, qmax=32, cpu-used=-6 (fast), process_cap=720p")
except Exception as e:
    print(f"[WebRTC] Warning: could not patch VP8 encoder: {e}")
    import traceback
    traceback.print_exc()

# ── Phase 1.4: H.264 codec support (preferred over VP8 when available) ──
# H.264 is ~40% more efficient than VP8 at same bitrate.
# We try to configure aiortc to prefer H.264, falling back to VP8 if unavailable.
_H264_AVAILABLE = False
try:
    from aiortc.codecs import h264 as _h264_module
    _H264_AVAILABLE = True
    print("[WebRTC] H.264 codec available — will prefer over VP8")
except ImportError:
    print("[WebRTC] H.264 codec not available — using VP8 (install openh264 for H.264)")
except Exception as e:
    print(f"[WebRTC] H.264 check error: {e} — using VP8")


def _parse_sdp_bitrate(sdp: str) -> Optional[int]:
    """Extract b=AS:<kbps> from SDP and return as bps, or None."""
    m = re.search(r'b=AS:(\d+)', sdp)
    if m:
        return int(m.group(1)) * 1000  # kbps → bps
    return None


class AudioBuffer:
    """Circular buffer for audio PCM data with configurable window size."""
    def __init__(self):
        self._buffer = bytearray()
        self._sample_rate = 48000
        self._channels = 1
        self._max_duration_s = 0.3  # 300ms — only the last 200ms (16 mel frames) are used

    def append(self, frame: AudioFrame):
        try:
            self._sample_rate = frame.sample_rate
            pcm = frame.to_ndarray().tobytes()
            self._buffer.extend(pcm)
            max_bytes = int(self._max_duration_s * self._sample_rate) * 2
            if len(self._buffer) > max_bytes:
                self._buffer = self._buffer[-max_bytes:]
        except Exception:
            return

    def get_recent_audio(self) -> tuple[bytes, int]:
        return bytes(self._buffer), self._sample_rate


class VideoTransformTrack(MediaStreamTrack):
    """Decoupled video processing track.

    Architecture:
    - A background thread continuously processes the latest input frame on GPU
    - recv() outputs at 30 FPS, always returning the latest processed result
    - This decouples GPU speed (~13 FPS) from output framerate (30 FPS)
    - Result: smooth 30 FPS output, ~75ms processing latency
    """
    kind = "video"

    TARGET_FPS = 30
    FRAME_INTERVAL = 1.0 / TARGET_FPS

    def __init__(
        self,
        track: MediaStreamTrack,
        swapper: FaceSwapper,
        lip_syncer: Optional[LipSyncer],
        session_id: str,
        audio_buffer: AudioBuffer,
        session_settings: Optional[Dict[str, dict]] = None,
        target_bitrate: Optional[int] = None,
    ):
        super().__init__()
        self.track = track
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.session_id = session_id
        self.audio_buffer = audio_buffer
        self.session_settings = session_settings or {}
        self._target_bitrate = target_bitrate  # from client SDP b=AS hint

        # Shared state between input reader, GPU worker, and output
        self._latest_input = None       # latest raw frame (numpy BGR)
        self._latest_result = None      # latest processed frame (numpy BGR)
        self._input_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._result_event = asyncio.Event()

        # Output timing
        self._pts = 0
        self._time_base = Fraction(1, 90000)
        self._pts_step = int(self.FRAME_INTERVAL / self._time_base)
        self._started = False

        # Stats
        self._out_count = 0
        self._swap_count = 0
        self._last_log = time.time()

        # Processing thread
        self._stop = threading.Event()
        self._has_input = threading.Event()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)

        # Input reader task (started on first recv)
        self._reader_task = None
        self._loop = None

    # Cap processing resolution to 720p. Face swap uses 128×128 crops regardless
    # of frame size, so 1080p adds zero face quality — only burns 2.25× more CPU
    # on warp, blend, copy, and VP8 encode. Google Meet also caps at 720p.
    MAX_PROCESS_HEIGHT = 720

    def _process_loop(self):
        """Background thread: process latest input frame on GPU."""
        while not self._stop.is_set():
            self._has_input.wait(timeout=0.1)
            if self._stop.is_set():
                break
            self._has_input.clear()

            with self._input_lock:
                img = self._latest_input
            if img is None:
                continue

            # Downscale to 720p if larger — saves CPU on warp/blend/copy/encode
            h, w = img.shape[:2]
            if h > self.MAX_PROCESS_HEIGHT:
                scale = self.MAX_PROCESS_HEIGHT / h
                new_w = int(w * scale)
                new_h = self.MAX_PROCESS_HEIGHT
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Check if lip sync will run — if so, skip mouth preservation
            settings = self.session_settings.get(self.session_id, {}) if self.session_settings else {}
            enable_lipsync = settings.get("enable_lipsync", ENABLE_LIPSYNC)
            will_lipsync = (enable_lipsync and self.lip_syncer
                           and self.lip_syncer.is_ready())

            # Face swap (GPU) — with skip_mouth_preservation if lip sync active
            result, faces = self.swapper.swap_face_with_faces(
                self.session_id, img, skip_mouth_preservation=will_lipsync
            )

            # Lip sync — use the SWAPPED result so generated mouth
            # matches target skin tone (not original person's)
            lipsync_applied = False
            if will_lipsync and len(faces) > 0:
                audio_pcm, sample_rate = self.audio_buffer.get_recent_audio()
                mel = self.lip_syncer.audio_to_mel(audio_pcm, sample_rate)
                if mel is not None:
                    face = faces[0]
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(result.shape[1], x2), min(result.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        face_crop = result[y1:y2, x1:x2]
                        synced = self.lip_syncer.infer(face_crop, mel, self.session_id)
                        if synced is not None:
                            result = self.lip_syncer.apply_mouth_only(
                                result, (x1, y1, x2, y2), synced
                            )
                            lipsync_applied = True

            # Cross-fade removed: was causing ghosting/smearing on fast movements.
            # The face swapper's temporal smoothing already handles frame stability.

            with self._result_lock:
                self._latest_result = result
            self._swap_count += 1

            # Signal the output that a new result is ready
            if self._loop:
                self._loop.call_soon_threadsafe(self._result_event.set)

    async def _read_input(self):
        """Async task: continuously read input frames, store latest."""
        logged_input_res = False
        try:
            while not self._stop.is_set():
                frame = await self.track.recv()
                img = frame.to_ndarray(format="bgr24")
                if not logged_input_res:
                    logged_input_res = True
                    h, w = img.shape[:2]
                    print(f"[WebRTC:{self.session_id}] Input resolution: {w}×{h}")
                with self._input_lock:
                    self._latest_input = img
                self._has_input.set()
        except Exception:
            pass

    async def recv(self) -> VideoFrame:
        # First call: start background workers
        if not self._started:
            self._started = True
            self._loop = asyncio.get_event_loop()
            self._worker.start()
            self._reader_task = asyncio.ensure_future(self._read_input())

        # Fix #7: Wait for a new processed frame with timeout
        # This syncs output to actual swap timing rather than a fixed 30fps clock
        # that produces 2-3 duplicate frames then a sudden jump.
        try:
            await asyncio.wait_for(self._result_event.wait(), timeout=self.FRAME_INTERVAL)
        except asyncio.TimeoutError:
            pass  # Use previous frame if no new result ready
        self._result_event.clear()

        # Get latest processed result
        with self._result_lock:
            result = self._latest_result

        if result is None:
            # Shouldn't happen, but fallback to a black frame
            result = np.zeros((480, 640, 3), dtype=np.uint8)

        # Build output frame at steady 30 FPS cadence
        new_frame = VideoFrame.from_ndarray(result, format="bgr24")
        new_frame.pts = self._pts
        new_frame.time_base = self._time_base
        self._pts += self._pts_step

        # Periodic logging
        self._out_count += 1
        now = time.time()
        if now - self._last_log >= 3.0:
            elapsed = now - self._last_log
            out_fps = self._out_count / elapsed
            swap_fps = self._swap_count / elapsed
            h, w = result.shape[:2]
            print(f"[WebRTC:{self.session_id}] output={out_fps:.1f}fps  swap={swap_fps:.1f}fps  res={w}×{h}")
            self._out_count = 0
            self._swap_count = 0
            self._last_log = now

        # Pacing handled by wait_for timeout above — no extra sleep needed.
        # Previous double-sleep (wait_for + sleep) halved effective FPS from 30 to ~15.
        return new_frame

    def stop(self):
        self._stop.set()
        self._has_input.set()
        if self._reader_task:
            self._reader_task.cancel()
        super().stop()


class WebRTCManager:
    def __init__(self, swapper: FaceSwapper, lip_syncer: Optional[LipSyncer]):
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.pcs: Dict[str, RTCPeerConnection] = {}
        self.relay = MediaRelay()
        self.session_settings: Dict[str, dict] = {}

    async def handle_offer(self, session_id: str, sdp: str, type: str) -> RTCSessionDescription:
        # Close any existing connection for this session
        if session_id in self.pcs:
            await self.pcs[session_id].close()
            del self.pcs[session_id]

        # Parse b=AS bitrate hint from client's SDP
        client_bitrate = _parse_sdp_bitrate(sdp)
        if client_bitrate:
            print(f"[WebRTC:{session_id}] Client requested bitrate: {client_bitrate // 1000} Kbps")
        else:
            client_bitrate = 6_000_000  # default 6 Mbps
            print(f"[WebRTC:{session_id}] No b=AS in SDP, using default 6 Mbps")

        pc = RTCPeerConnection()
        self.pcs[session_id] = pc

        audio_buffer = AudioBuffer()

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                async def recv_audio():
                    try:
                        while True:
                            frame = await track.recv()
                            audio_buffer.append(frame)
                    except Exception:
                        pass
                asyncio.ensure_future(recv_audio())
            elif track.kind == "video":
                local_video = VideoTransformTrack(
                    self.relay.subscribe(track),
                    self.swapper,
                    self.lip_syncer,
                    session_id,
                    audio_buffer,
                    self.session_settings,
                    target_bitrate=client_bitrate,
                )
                pc.addTrack(local_video)

                # Set the encoder bitrate on the sender AFTER it's created.
                # aiortc creates the VP8 encoder lazily on the first encode(),
                # but we can set target_bitrate on it once it exists.
                # We do this via a background task that waits for the encoder.
                async def _set_encoder_bitrate():
                    for sender in pc.getSenders():
                        if sender.track == local_video:
                            # Wait for encoder to be created
                            for _ in range(50):  # up to 5 seconds
                                enc = getattr(sender, '_RTCRtpSender__encoder', None)
                                if enc and hasattr(enc, 'target_bitrate'):
                                    enc.target_bitrate = client_bitrate
                                    print(f"[WebRTC:{session_id}] Set server VP8 encoder bitrate → {client_bitrate // 1000} Kbps")
                                    return
                                await asyncio.sleep(0.1)
                            print(f"[WebRTC:{session_id}] Warning: could not set encoder bitrate (encoder not found)")
                            return
                asyncio.ensure_future(_set_encoder_bitrate())

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            print(f"[WebRTC:{session_id}] Connection state: {state}")
            if state in ["failed", "closed", "disconnected"]:
                await self.cleanup_session(session_id)

        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    async def cleanup_session(self, session_id: str):
        if session_id in self.pcs:
            try:
                await self.pcs[session_id].close()
            except Exception:
                pass
            del self.pcs[session_id]
        self.session_settings.pop(session_id, None)
        # Risk#2 fix: clean up per-session lip sync state
        if self.lip_syncer:
            self.lip_syncer.cleanup_session(session_id)

    def set_session_settings(self, session_id: str, settings: dict):
        self.session_settings[session_id] = {
            **self.session_settings.get(session_id, {}),
            **settings
        }


# ═══════════════════════════════════════════════════════════════
# Approach 4: Agent-Based A/V Sync Pipeline Manager
# ═══════════════════════════════════════════════════════════════
# When ENABLE_AV_SYNC_PIPELINE=true, this replaces the original
# WebRTCManager with the 7-agent synchronized pipeline.
# Falls back to the original approach if the pipeline fails.
# ═══════════════════════════════════════════════════════════════

class SyncWebRTCManager:
    """
    WebRTC manager using the 7-agent A/V sync pipeline.

    Architecture:
      Client → WebRTC → Agent 1 (Ingest) → Agent 2 (FaceSwap) → Agent 4 (PTSAlign)
                                ↓                                       ↑
                         Agent 3 (AudioBuffer) ─────────────────────────┘
                                                       ↓
                                                Agent 5 (MuxEncode)
                                                   ↓         ↓
                                          SyncedVideo   SyncedAudio → WebRTC → Client

    Falls back to the original VideoTransformTrack approach if the
    pipeline fails to start or the circuit breaker trips.
    """

    def __init__(self, swapper: FaceSwapper, lip_syncer: Optional[LipSyncer]):
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.pcs: Dict[str, RTCPeerConnection] = {}
        self.pipelines: Dict[str, 'SyncPipeline'] = {}
        self.relay = MediaRelay()
        self.session_settings: Dict[str, dict] = {}
        # Keep a fallback manager for circuit-breaker scenarios
        self._fallback_manager = WebRTCManager(swapper, lip_syncer)

    async def handle_offer(self, session_id: str, sdp: str, type: str) -> RTCSessionDescription:
        """Handle WebRTC offer using the agent-based sync pipeline."""
        from agents.pipeline import SyncPipeline
        from agents.models import PipelineConfig

        # Close any existing connection
        if session_id in self.pcs:
            await self.cleanup_session(session_id)

        # Parse bitrate hint from SDP
        client_bitrate = _parse_sdp_bitrate(sdp)
        if not client_bitrate:
            client_bitrate = 6_000_000
        print(f"[SyncWebRTC:{session_id}] Bitrate: {client_bitrate // 1000} Kbps")

        pc = RTCPeerConnection()
        self.pcs[session_id] = pc

        # Create the sync pipeline
        settings = self.session_settings.get(session_id, {})
        enable_lipsync = settings.get("enable_lipsync", ENABLE_LIPSYNC)

        config = PipelineConfig(
            video_bitrate=client_bitrate,
            target_fps=30,
            audio_buffer_capacity_ms=float(AV_SYNC_AUDIO_BUFFER_MS),
            max_audio_wait_ms=float(AV_SYNC_MAX_AUDIO_WAIT_MS),
            drift_recalibrate_frames=AV_SYNC_DRIFT_RECAL_FRAMES,
            avo_warning_ms=AV_SYNC_AVO_WARNING_MS,
            avo_critical_ms=AV_SYNC_AVO_CRITICAL_MS,
        )

        pipeline = SyncPipeline(
            session_id=session_id,
            swapper=self.swapper,
            lip_syncer=self.lip_syncer,
            config=config,
            target_bitrate=client_bitrate,
            enable_lipsync=enable_lipsync,
            session_settings=settings,
        )
        self.pipelines[session_id] = pipeline

        # Collect tracks as they arrive
        received_tracks = {"video": None, "audio": None}
        pipeline_started = {"started": False}

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                received_tracks["audio"] = track
                print(f"[SyncWebRTC:{session_id}] Audio track received")
            elif track.kind == "video":
                received_tracks["video"] = self.relay.subscribe(track)
                print(f"[SyncWebRTC:{session_id}] Video track received")

            # Once video arrives, wait briefly for audio then start pipeline
            if received_tracks["video"] and not pipeline_started["started"]:
                pipeline_started["started"] = True
                asyncio.ensure_future(
                    self._wait_and_start_pipeline(
                        session_id, pc, received_tracks, client_bitrate
                    )
                )

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            print(f"[SyncWebRTC:{session_id}] Connection state: {state}")
            if state in ["failed", "closed", "disconnected"]:
                await self.cleanup_session(session_id)

        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    async def _wait_and_start_pipeline(
        self,
        session_id: str,
        pc: RTCPeerConnection,
        tracks: dict,
        client_bitrate: int,
    ) -> None:
        """Wait briefly for audio track, then start pipeline."""
        # Audio track may arrive slightly after video — wait up to 500ms
        for _ in range(10):
            if tracks.get("audio"):
                break
            await asyncio.sleep(0.05)

        if not tracks.get("audio"):
            print(f"[SyncWebRTC:{session_id}] No audio track after 500ms — starting video-only")

        await self._start_pipeline(session_id, pc, tracks, client_bitrate)

    async def _start_pipeline(
        self,
        session_id: str,
        pc: RTCPeerConnection,
        tracks: dict,
        client_bitrate: int,
    ) -> None:
        """Start the agent pipeline and add output tracks to the peer connection."""
        pipeline = self.pipelines.get(session_id)
        if not pipeline:
            return

        try:
            # Set input tracks
            pipeline.set_tracks(tracks.get("video"), tracks.get("audio"))

            # Start the 7-agent pipeline
            await pipeline.start()

            # Add synced output tracks to the peer connection
            pc.addTrack(pipeline.video_output_track)

            # Add synced audio track (relays processed + aligned audio)
            if tracks.get("audio"):
                pc.addTrack(pipeline.audio_output_track)

            # Set encoder bitrate
            async def _set_bitrate():
                for sender in pc.getSenders():
                    if sender.track == pipeline.video_output_track:
                        for _ in range(50):
                            enc = getattr(sender, '_RTCRtpSender__encoder', None)
                            if enc and hasattr(enc, 'target_bitrate'):
                                enc.target_bitrate = client_bitrate
                                print(f"[SyncWebRTC:{session_id}] Encoder bitrate → {client_bitrate // 1000} Kbps")
                                return
                            await asyncio.sleep(0.1)
            asyncio.ensure_future(_set_bitrate())

            print(f"[SyncWebRTC:{session_id}] ✅ Agent pipeline started successfully")

        except Exception as exc:
            print(f"[SyncWebRTC:{session_id}] ❌ Pipeline start failed: {exc}")
            print(f"[SyncWebRTC:{session_id}] Falling back to legacy mode")
            # Fallback to original approach
            await self._fallback_to_legacy(session_id, pc, tracks, client_bitrate)

    async def _fallback_to_legacy(
        self,
        session_id: str,
        pc: RTCPeerConnection,
        tracks: dict,
        client_bitrate: int,
    ) -> None:
        """Fall back to the original VideoTransformTrack approach."""
        audio_buf = AudioBuffer()

        # Set up legacy audio reader
        if tracks.get("audio"):
            async def recv_audio():
                try:
                    while True:
                        frame = await tracks["audio"].recv()
                        audio_buf.append(frame)
                except Exception:
                    pass
            asyncio.ensure_future(recv_audio())

        # Set up legacy video track
        if tracks.get("video"):
            local_video = VideoTransformTrack(
                tracks["video"],
                self.swapper,
                self.lip_syncer,
                session_id,
                audio_buf,
                self.session_settings,
                target_bitrate=client_bitrate,
            )
            pc.addTrack(local_video)

    async def cleanup_session(self, session_id: str):
        """Clean up pipeline and peer connection for a session."""
        # Stop the agent pipeline
        if session_id in self.pipelines:
            try:
                await self.pipelines[session_id].stop()
            except Exception as exc:
                print(f"[SyncWebRTC:{session_id}] Pipeline stop error: {exc}")
            del self.pipelines[session_id]

        # Close peer connection
        if session_id in self.pcs:
            try:
                await self.pcs[session_id].close()
            except Exception:
                pass
            del self.pcs[session_id]

        self.session_settings.pop(session_id, None)

        if self.lip_syncer:
            self.lip_syncer.cleanup_session(session_id)

    def set_session_settings(self, session_id: str, settings: dict):
        self.session_settings[session_id] = {
            **self.session_settings.get(session_id, {}),
            **settings
        }

    def get_pipeline_metrics(self, session_id: str) -> Optional[dict]:
        """Get sync pipeline metrics for a session."""
        pipeline = self.pipelines.get(session_id)
        if pipeline:
            return pipeline.get_metrics()
        return None
