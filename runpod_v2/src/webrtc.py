"""
WebRTC handling with aiortc for low-latency video/audio streaming.
Ported from runpod_service for runpod_v2.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame

from face_swapper import FaceSwapper
from lip_syncer import LipSyncer
from config import ENABLE_LIPSYNC, LIPSYNC_AUDIO_WINDOW_MS


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
    kind = "video"

    def __init__(
        self,
        track: MediaStreamTrack,
        swapper: FaceSwapper,
        lip_syncer: Optional[LipSyncer],
        session_id: str,
        audio_buffer: AudioBuffer,
        session_settings: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.track = track
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.session_id = session_id
        self.audio_buffer = audio_buffer
        self.session_settings = session_settings or {}
        self._frame_count = 0
        self._last_log = time.time()

    async def recv(self) -> VideoFrame:
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Face swap
        result, faces = self.swapper.swap_face_with_faces(self.session_id, img)

        # Lip sync (optional)
        settings = self.session_settings.get(self.session_id, {}) if self.session_settings else {}
        enable_lipsync = settings.get("enable_lipsync", ENABLE_LIPSYNC)
        if enable_lipsync and self.lip_syncer and self.lip_syncer.is_ready() and len(faces) > 0:
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

        # Periodic logging
        self._frame_count += 1
        now = time.time()
        if now - self._last_log >= 3.0:
            fps = self._frame_count / (now - self._last_log)
            print(f"[WebRTC:{self.session_id}] FPS: {fps:.1f}")
            self._frame_count = 0
            self._last_log = now

        new_frame = VideoFrame.from_ndarray(result, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


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

        pc = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        ))
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
                    self.session_settings
                )
                pc.addTrack(local_video)

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
