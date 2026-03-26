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
from config import ENABLE_LIPSYNC, LIPSYNC_AUDIO_WINDOW_MS

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

    _HIGH_DEFAULT_BITRATE = 3_000_000  # 3 Mbps

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
    print(f"[WebRTC] VP8 encoder patched: bitrate=3Mbps, qmax=32, cpu-used=-6 (fast), process_cap=720p")
except Exception as e:
    print(f"[WebRTC] Warning: could not patch VP8 encoder: {e}")
    import traceback
    traceback.print_exc()


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
        self._max_duration_s = 0.5

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
        """Background thread: process latest input frame on GPU.
        Fix #7: Cross-fade between consecutive swap results to eliminate stutter."""
        prev_result = None  # For cross-fade between swap outputs
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

            # Save original for lip sync (clean mouth pixels, no swap artifacts)
            original_img = img.copy()

            # Fix #3: Check if lip sync will run — if so, skip mouth preservation
            settings = self.session_settings.get(self.session_id, {}) if self.session_settings else {}
            enable_lipsync = settings.get("enable_lipsync", ENABLE_LIPSYNC)
            will_lipsync = (enable_lipsync and self.lip_syncer
                           and self.lip_syncer.is_ready())

            # Face swap (GPU) — with skip_mouth_preservation if lip sync active
            result, faces = self.swapper.swap_face_with_faces(
                self.session_id, img, skip_mouth_preservation=will_lipsync
            )

            # Lip sync — use original frame for Wav2Lip input so it gets
            # clean teeth/tongue instead of swapped artifacts
            if will_lipsync and len(faces) > 0:
                audio_pcm, sample_rate = self.audio_buffer.get_recent_audio()
                mel = self.lip_syncer.audio_to_mel(audio_pcm, sample_rate)
                if mel is not None:
                    face = faces[0]
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(result.shape[1], x2), min(result.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        face_crop = original_img[y1:y2, x1:x2]
                        synced = self.lip_syncer.infer(face_crop, mel)
                        if synced is not None:
                            result = self.lip_syncer.apply_mouth_only(
                                result, (x1, y1, x2, y2), synced
                            )

            # Fix #7: Cross-fade with previous swap result to eliminate
            # the "jump" when a new swap result replaces repeated frames.
            # 70% new + 30% old = smooth transition over 1-2 frames.
            if prev_result is not None and prev_result.shape == result.shape:
                result = cv2.addWeighted(result, 0.7, prev_result, 0.3, 0)
            prev_result = result.copy()

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

        # Pace output to ~30 FPS
        await asyncio.sleep(self.FRAME_INTERVAL)
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
            client_bitrate = 3_000_000  # default 3 Mbps
            print(f"[WebRTC:{session_id}] No b=AS in SDP, using default 3 Mbps")

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

    def set_session_settings(self, session_id: str, settings: dict):
        self.session_settings[session_id] = {
            **self.session_settings.get(session_id, {}),
            **settings
        }
