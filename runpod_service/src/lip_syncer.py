"""
LipSyncer - Real-time lip sync using Wav2Lip ONNX
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
import librosa
import cv2

from typing import Optional

from config import WAV2LIP_MODEL_PATH, LIPSYNC_AUDIO_WINDOW_MS


class LipSyncer:
    def __init__(self, providers: list[str]):
        self.session: Optional[ort.InferenceSession] = None
        self.input_names = []

        try:
            self.session = ort.InferenceSession(WAV2LIP_MODEL_PATH, providers=providers)
            self.input_names = [i.name for i in self.session.get_inputs()]
            print(f"LipSyncer loaded model: {WAV2LIP_MODEL_PATH}")
        except Exception as e:
            print(f"LipSyncer disabled (model load failed): {e}")
            self.session = None

    def is_ready(self) -> bool:
        return self.session is not None

    def audio_to_mel(self, audio_pcm: bytes, sample_rate: int) -> Optional[np.ndarray]:
        if not self.session or not audio_pcm:
            return None
        try:
            audio = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) == 0:
                return None
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=800,
                hop_length=200,
                win_length=800,
                n_mels=80
            )
            mel = np.log10(np.maximum(mel, 1e-5))
            return mel
        except Exception as e:
            print(f"Mel extraction failed: {e}")
            return None

    def infer(self, face: np.ndarray, mel: np.ndarray) -> Optional[np.ndarray]:
        if not self.session or mel is None:
            return None
        try:
            # Wav2Lip expects (B, T, 80) mel and (B, 3, 96, 96) face
            face_resized = cv2.resize(face, (96, 96))
            face_rgb = face_resized[:, :, ::-1].astype(np.float32) / 255.0
            face_input = np.transpose(face_rgb, (2, 0, 1))[None, ...]

            # Use last window of mel (16 frames)
            mel_t = mel.T
            mel_win = mel_t[-16:]
            if mel_win.shape[0] < 16:
                mel_win = np.pad(mel_win, ((0, 16 - mel_win.shape[0]), (0, 0)), mode="edge")
            mel_input = mel_win[None, ...].astype(np.float32)

            # Heuristic: map input names by expected shapes
            inputs = {}
            for name, inp in zip(self.input_names, self.session.get_inputs()):
                shape = inp.shape
                if len(shape) == 4:
                    inputs[name] = face_input
                else:
                    inputs[name] = mel_input
            pred = self.session.run(None, inputs)[0]
            pred = np.transpose(pred[0], (1, 2, 0))
            pred = (pred * 255).clip(0, 255).astype(np.uint8)
            pred = pred[:, :, ::-1]
            return pred
        except Exception as e:
            print(f"LipSync inference failed: {e}")
            return None