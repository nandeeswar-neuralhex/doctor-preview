"""
WebRTC handling with aiortc for low-latency video/audio streaming.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame, AudioFrame

from face_swapper import FaceSwapper
from lip_syncer import LipSyncer
from config import ENABLE_LIPSYNC, LIPSYNC_AUDIO_WINDOW_MS


class AudioBuffer:
    def __init__(self):
        self._buffer = bytearray()
        self._sample_rate = 48000
        self._channels = 1
        self._max_samples = int(0.5 * self._sample_rate)

    def append(self, frame: AudioFrame):
        try:
            self._sample_rate = frame.sample_rate
            pcm = frame.to_ndarray().tobytes()
            self._buffer.extend(pcm)
            if len(self._buffer) > self._max_samples * 2:
                self._buffer = self._buffer[-self._max_samples * 2 :]
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
        frame_queue: Optional[asyncio.Queue] = None,
        session_settings: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.track = track
        self.swapper = swapper
        self.lip_syncer = lip_syncer
        self.session_id = session_id
        self.audio_buffer = audio_buffer
        self.frame_queue = frame_queue
        self.session_settings = session_settings or {}

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
                        synced = cv2.resize(synced, (x2 - x1, y2 - y1))
                        result[y1:y2, x1:x2] = synced

        if self.frame_queue is not None:
            try:
                if self.frame_queue.full():
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(result)
            except Exception:
                pass

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
        self.frame_queues: Dict[str, asyncio.Queue] = {}
        self.session_settings: Dict[str, dict] = {}

    async def handle_offer(self, session_id: str, sdp: str, type: str) -> RTCSessionDescription:
        pc = RTCPeerConnection()
        self.pcs[session_id] = pc

        if session_id not in self.frame_queues:
            self.frame_queues[session_id] = asyncio.Queue(maxsize=2)

        audio_buffer = AudioBuffer()

        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            if track.kind == "audio":
                async def recv_audio():
                    while True:
                        frame = await track.recv()
                        audio_buffer.append(frame)
                asyncio.ensure_future(recv_audio())
            elif track.kind == "video":
                local_video = VideoTransformTrack(
                    self.relay.subscribe(track),
                    self.swapper,
                    self.lip_syncer,
                    session_id,
                    audio_buffer,
                    self.frame_queues.get(session_id),
                    self.session_settings
                )
                pc.addTrack(local_video)

        @pc.on("connectionstatechange")
        async def on_state_change():
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                await pc.close()
                if session_id in self.pcs:
                    del self.pcs[session_id]
                if session_id in self.frame_queues:
                    del self.frame_queues[session_id]
                if session_id in self.session_settings:
                    del self.session_settings[session_id]

        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription

    def set_session_settings(self, session_id: str, settings: dict):
        """Store per-session settings (e.g. enable_lipsync)."""
        self.session_settings[session_id] = {
            **self.session_settings.get(session_id, {}),
            **settings
        }

    def get_latest_frame_queue(self, session_id: str) -> Optional[asyncio.Queue]:
        return self.frame_queues.get(session_id)

    def set_session_settings(self, session_id: str, settings: dict):
        self.session_settings[session_id] = {
            **self.session_settings.get(session_id, {}),
            **settings
        }