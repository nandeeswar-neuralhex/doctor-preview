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
from collections import deque
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
    ENABLE_LIPSYNC, LIPSYNC_AUDIO_WINDOW_MS,
    ENABLE_AUDIO_RELAY, AUDIO_RELAY_EXTRA_DELAY_MS,
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
        latency_state: Optional[dict] = None,
    ):
        super().__init__()
        self.track = track
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.session_id = session_id
        self.audio_buffer = audio_buffer
        self.session_settings = session_settings or {}
        self._target_bitrate = target_bitrate  # from client SDP b=AS hint

        # A/V sync: shared latency estimate (ms) — written here, read by
        # DelayedAudioRelayTrack so relayed audio matches video timing.
        self._latency_state = latency_state if latency_state is not None else {"ms": 100.0}
        self._latest_input_ts = 0.0

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
                input_ts = self._latest_input_ts
            if img is None:
                continue

            # Downscale to 720p if larger — saves CPU on warp/blend/copy/encode
            h, w = img.shape[:2]
            if h > self.MAX_PROCESS_HEIGHT:
                scale = self.MAX_PROCESS_HEIGHT / h
                new_w = int(w * scale)
                new_h = self.MAX_PROCESS_HEIGHT
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Check if lip sync will run
            settings = self.session_settings.get(self.session_id, {}) if self.session_settings else {}
            enable_lipsync = settings.get("enable_lipsync", ENABLE_LIPSYNC)
            will_lipsync = (enable_lipsync and self.lip_syncer
                           and self.lip_syncer.is_ready())

            # Face swap (GPU)
            result, faces = self.swapper.swap_face_with_faces(self.session_id, img)

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
                        synced = self.lip_syncer.infer(face_crop, mel)
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

            # A/V sync: measure input→result latency and keep an asymmetric
            # EMA. Rises fast (audio must never lead the mouth), falls slowly
            # (avoids oscillating delay that clips words).
            if input_ts > 0:
                total_ms = (time.monotonic() - input_ts) * 1000.0 \
                    + (self.FRAME_INTERVAL * 500.0)  # + avg output pacing wait
                prev = self._latency_state.get("ms", 100.0)
                alpha = 0.30 if total_ms > prev else 0.05
                self._latency_state["ms"] = prev + alpha * (total_ms - prev)

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
                    self._latest_input_ts = time.monotonic()
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


class DelayedAudioRelayTrack(MediaStreamTrack):
    """Relays the client's mic audio back as a smooth jitter-buffered stream.

    A/V sync (Option A): this track shares the peer connection with the
    processed video track, so the receiving browser aligns both via RTCP
    sender reports. The small jitter buffer (~2 frames) plus the video
    pipeline's own ~30-60ms latency land well inside the ITU lip-sync
    tolerance; residual offset is tuned with AUDIO_RELAY_EXTRA_DELAY_MS.

    Design (classic VoIP jitter buffer — NO dynamic delay chasing):
    - A reader task queues incoming frames and feeds the lipsync buffer.
    - recv() emits one frame per frame-duration at a steady cadence.
    - Frames are popped in order whenever the buffer is primed; silence is
      emitted ONLY on true underrun (empty buffer), never interleaved
      between available frames.
    - After an underrun the buffer re-primes to target depth before
      resuming (hysteresis prevents silence/audio flapping).
    - Latency is bounded by dropping QUIET frames when the buffer grows
      too deep (never cuts words), with a hard drop only past 500ms.

    Previous design chased the video-latency EMA with per-frame age gating
    — latency spikes thrashed the delay (33ms→1844ms→29ms) and jitter
    interleaved silence mid-speech, destroying the audio. Do not revisit.
    """

    kind = "audio"

    BASE_DEPTH_FRAMES = 4        # ~80ms priming depth
    MAX_DEPTH_FRAMES = 12        # adaptive growth cap (~240ms)
    EXCESS_SOFT_FRAMES = 8       # >target+8 (~160ms): drop quiet head frames
    EXCESS_HARD_FRAMES = 25      # >target+25 (~500ms): force-drop to target
    QUIET_RMS = 300              # int16 RMS below this = droppable silence

    def __init__(
        self,
        source: MediaStreamTrack,
        audio_buffer: AudioBuffer,
        latency_state: dict,
        extra_delay_ms: int = 0,
    ):
        super().__init__()
        self._source = source
        self._audio_buffer = audio_buffer
        self._latency_state = latency_state  # logging only — no delay coupling

        self._queue: deque = deque()
        self._reader_task = None
        self._started = False

        # Output format template. CRITICAL: every frame this track emits
        # (silence or relayed) MUST share one format — aiortc's OpusEncoder
        # keeps a persistent AudioResampler that hard-crashes on any
        # format/layout change, silently killing the audio sender.
        # aiortc's Opus decoder always outputs s16/stereo/48kHz, so that is
        # the default; it is re-locked from the first queued source frame
        # before anything is emitted.
        self._sample_rate = 48000
        self._layout = "stereo"
        self._format = "s16"
        self._samples = 960  # 20 ms @ 48 kHz
        self._template_locked = False
        self._first_real_sent = False
        self._pts = 0
        self._next_emit: Optional[float] = None

        # Jitter buffer state
        frame_ms = 20.0
        self._target_depth = self.BASE_DEPTH_FRAMES + int(max(0, extra_delay_ms) / frame_ms)
        self._primed = False
        self._underruns = 0
        self._quiet_drops = 0
        self._hard_drops = 0
        self._last_log = time.monotonic()

    async def _read_source(self):
        """Continuously pull frames from the client's mic track."""
        try:
            while True:
                frame = await self._source.recv()
                self._audio_buffer.append(frame)  # feed lipsync window
                self._queue.append(frame)
        except Exception:
            pass

    def _make_silence(self) -> AudioFrame:
        frame = AudioFrame(format=self._format, layout=self._layout, samples=self._samples)
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = self._sample_rate
        return frame

    @staticmethod
    def _frame_rms(frame: AudioFrame) -> float:
        try:
            arr = frame.to_ndarray().astype(np.float32)
            return float(np.sqrt((arr ** 2).mean()))
        except Exception:
            return 1e9  # treat unreadable frames as loud → never dropped

    def _bound_latency(self):
        """Keep buffer depth near target without ever cutting words."""
        depth = len(self._queue)
        if depth > self._target_depth + self.EXCESS_HARD_FRAMES:
            # Way too deep (stall recovery) — force-drop to target
            while len(self._queue) > self._target_depth:
                self._queue.popleft()
                self._hard_drops += 1
        elif depth > self._target_depth + self.EXCESS_SOFT_FRAMES:
            # Slightly deep — shed only quiet head frames (max 2 per tick)
            for _ in range(2):
                if (len(self._queue) > self._target_depth
                        and self._frame_rms(self._queue[0]) < self.QUIET_RMS):
                    self._queue.popleft()
                    self._quiet_drops += 1
                else:
                    break

    async def recv(self) -> AudioFrame:
        try:
            return await self._recv_impl()
        except Exception:
            # aiortc kills the sender silently on track errors — make sure
            # any failure is visible in the logs before propagating.
            import traceback
            print("[AudioRelay] recv() FAILED — audio sender will stop:")
            traceback.print_exc()
            raise

    async def _recv_impl(self) -> AudioFrame:
        if not self._started:
            self._started = True
            self._reader_task = asyncio.ensure_future(self._read_source())

        # Steady output cadence (one frame per frame-duration)
        frame_dur = self._samples / float(self._sample_rate)
        now = time.monotonic()
        if self._next_emit is None:
            self._next_emit = now
        wait = self._next_emit - now
        if wait > 0:
            await asyncio.sleep(wait)
        elif wait < -0.5:
            self._next_emit = time.monotonic()  # resync after a stall
        self._next_emit += frame_dur

        # Lock the output format to the source's format BEFORE emitting
        # anything, so silence and relayed frames always match.
        if not self._template_locked and self._queue:
            first = self._queue[0]
            self._sample_rate = first.sample_rate or self._sample_rate
            self._layout = first.layout.name
            self._format = first.format.name
            self._samples = first.samples
            self._template_locked = True
            print(f"[AudioRelay] output format locked: "
                  f"{self._format}/{self._layout}/{self._sample_rate}Hz/{self._samples}spf")

        self._bound_latency()

        # Priming / underrun hysteresis: only play once target depth is
        # buffered, so we never flap between silence and audio per-frame.
        # Each underrun adaptively deepens the buffer (classic jitter
        # buffer) so repeated network bursts stop causing gaps.
        depth = len(self._queue)
        if not self._primed:
            if depth >= self._target_depth:
                self._primed = True
        elif depth == 0:
            self._primed = False
            self._underruns += 1
            self._target_depth = min(self._target_depth + 2, self.MAX_DEPTH_FRAMES)

        out: Optional[AudioFrame] = None
        if self._primed and self._queue:
            out = self._queue.popleft()

        if out is not None:
            if not self._first_real_sent:
                self._first_real_sent = True
                print(f"[AudioRelay] first source frame relayed "
                      f"({out.format.name}/{out.layout.name}/{out.sample_rate}Hz)")
            # Guard: a mid-stream format change would crash the Opus
            # encoder's resampler. Should never happen with a single
            # decoder — warn and re-lock if it somehow does.
            if (out.format.name != self._format
                    or out.layout.name != self._layout
                    or (out.sample_rate or self._sample_rate) != self._sample_rate):
                print(f"[AudioRelay] WARNING: source format changed to "
                      f"{out.format.name}/{out.layout.name}/{out.sample_rate}Hz")
                self._sample_rate = out.sample_rate or self._sample_rate
                self._layout = out.layout.name
                self._format = out.format.name
                self._samples = out.samples
        else:
            out = self._make_silence()

        # Re-stamp with a monotonic PTS (silence insertion breaks source PTS)
        out.pts = self._pts
        out.time_base = Fraction(1, self._sample_rate)
        self._pts += out.samples

        now = time.monotonic()
        if now - self._last_log >= 10.0:
            self._last_log = now
            print(f"[AudioRelay] depth={len(self._queue)}/{self._target_depth} "
                  f"underruns={self._underruns} quiet_drops={self._quiet_drops} "
                  f"hard_drops={self._hard_drops} "
                  f"video_latency={self._latency_state.get('ms', 0):.0f}ms")

        return out

    def stop(self):
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
        # Shared video-latency estimate (ms): written by VideoTransformTrack,
        # read by DelayedAudioRelayTrack to keep relayed audio in sync.
        latency_state = {"ms": 100.0}

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                if ENABLE_AUDIO_RELAY:
                    # A/V sync (Option A): relay mic audio back, delayed to
                    # match video latency. Same PC as video → browser syncs
                    # both tracks via RTCP sender reports.
                    relay_audio = DelayedAudioRelayTrack(
                        track,
                        audio_buffer,
                        latency_state,
                        extra_delay_ms=AUDIO_RELAY_EXTRA_DELAY_MS,
                    )
                    pc.addTrack(relay_audio)
                    print(f"[WebRTC:{session_id}] Audio relay enabled — "
                          f"delayed to match video (+{AUDIO_RELAY_EXTRA_DELAY_MS}ms extra)")
                else:
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
                    latency_state=latency_state,
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




