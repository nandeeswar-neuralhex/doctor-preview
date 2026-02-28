"""
LipSyncer - Real-time lip sync using Wav2Lip ONNX

Improvements over v1:
- Resamples audio to 16kHz (Wav2Lip training rate) regardless of input rate
- Wider audio buffer for better mel boundary handling
- Mouth-only blending: preserves upper face from face-swap, only modifies mouth/chin
- Feathered edge blending to prevent visible seams
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import librosa
except ImportError:
    librosa = None

from config import WAV2LIP_MODEL_PATH

# Wav2Lip was trained at 16kHz
_WAV2LIP_SR = 16000


class LipSyncer:
    def __init__(self, providers: list[str]):
        self.session: Optional[ort.InferenceSession] = None
        self.input_names = []

        if ort is None:
            print("LipSyncer disabled: onnxruntime not installed")
            return

        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = 2   # Limit CPU threads to avoid starving event loop
            sess_opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(WAV2LIP_MODEL_PATH, sess_options=sess_opts, providers=providers)
            self.input_names = [i.name for i in self.session.get_inputs()]
            print(f"LipSyncer loaded model: {WAV2LIP_MODEL_PATH}")
        except Exception as e:
            print(f"LipSyncer disabled (model load failed): {e}")
            self.session = None

    def is_ready(self) -> bool:
        if self.session is None:
            return False
        if librosa is None:
            print("LipSyncer: librosa not installed — pip install librosa")
            return False
        return True

    def audio_to_mel(self, audio_pcm: bytes, sample_rate: int) -> Optional[np.ndarray]:
        """
        Convert raw PCM int16 audio to log-mel spectrogram.
        Automatically resamples to 16kHz for Wav2Lip compatibility.
        """
        if not self.session or not audio_pcm or librosa is None:
            return None
        try:
            audio = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio) == 0:
                return None

            # Resample to 16kHz if needed (Wav2Lip was trained at 16kHz)
            if sample_rate != _WAV2LIP_SR and sample_rate > 0:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_WAV2LIP_SR)

            # Mel spectrogram with Wav2Lip-compatible parameters at 16kHz
            # n_fft=800  → 50ms analysis window
            # hop_length=200 → 12.5ms hop between frames
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=_WAV2LIP_SR,
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
        """
        Run Wav2Lip inference on a face crop + mel spectrogram.
        Uses the last 16 mel frames (~200ms) as the model expects.
        """
        if not self.session or mel is None:
            return None
        try:
            # Wav2Lip expects (B, 3, 96, 96) face
            face_resized = cv2.resize(face, (96, 96))
            face_rgb = face_resized[:, :, ::-1].astype(np.float32) / 255.0
            face_input = np.transpose(face_rgb, (2, 0, 1))[None, ...]

            # Use last 16 mel frames (model's expected input size)
            # With a wider audio buffer, these 16 frames have proper STFT
            # context instead of edge artifacts
            mel_t = mel.T
            n_frames = 16
            mel_win = mel_t[-n_frames:]
            if mel_win.shape[0] < n_frames:
                mel_win = np.pad(
                    mel_win,
                    ((0, n_frames - mel_win.shape[0]), (0, 0)),
                    mode="edge"
                )
            mel_input = mel_win[None, ...].astype(np.float32)

            # Map inputs by expected shapes (4D = face image, other = mel)
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
            pred = pred[:, :, ::-1]  # RGB → BGR
            return pred
        except Exception as e:
            print(f"LipSync inference failed: {e}")
            return None

    def apply_mouth_only(
        self,
        result: np.ndarray,
        face_bbox: tuple,
        lip_output: np.ndarray,
        landmarks_68: np.ndarray = None,
    ) -> np.ndarray:
        """
        Blend Wav2Lip output into the mouth region only with feathered edges.

        Improvements:
        - Optional 68-point landmarks for precise mouth contour masking
        - Post-sharpening to counteract Wav2Lip's 96x96 resolution blur
        - Adaptive blend zone based on actual lip positions
        """
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(result.shape[1], x2)
        y2 = min(result.shape[0], y2)
        face_h = y2 - y1
        face_w = x2 - x1
        if face_h <= 0 or face_w <= 0:
            return result

        lip_resized = cv2.resize(lip_output, (face_w, face_h))

        # ── Sharpen lip output to counteract Wav2Lip's 96x96 blur ──
        usm_blur = cv2.GaussianBlur(lip_resized, (0, 0), 1.5)
        lip_resized = cv2.addWeighted(lip_resized, 1.3, usm_blur, -0.3, 0)

        mask = np.zeros((face_h, face_w), dtype=np.float32)
        used_landmarks = False

        # ── PRIMARY: Landmark-based mouth mask ──
        if landmarks_68 is not None:
            try:
                # Mouth landmarks (48-67) in frame coordinates → face ROI coordinates
                mouth_pts = landmarks_68[48:68].copy()
                mouth_pts[:, 0] -= x1
                mouth_pts[:, 1] -= y1

                mouth_min_y = mouth_pts[:, 1].min()
                mouth_max_y = mouth_pts[:, 1].max()
                mouth_h = max(1, mouth_max_y - mouth_min_y)

                # Blend zone: from above mouth to below chin
                blend_top = max(0, int(mouth_min_y - mouth_h * 1.2))
                full_top = max(0, int(mouth_min_y - mouth_h * 0.3))
                full_bottom = min(face_h, int(mouth_max_y + mouth_h * 0.4))
                blend_bottom = min(face_h, int(mouth_max_y + mouth_h * 0.8))

                for y_idx in range(blend_top, blend_bottom):
                    if y_idx < full_top:
                        alpha = (y_idx - blend_top) / max(1, full_top - blend_top)
                    elif y_idx > full_bottom:
                        alpha = 1.0 - (y_idx - full_bottom) / max(1, blend_bottom - full_bottom)
                    else:
                        alpha = 1.0
                    mask[y_idx, :] = alpha

                # Horizontal: fade based on mouth width + margin
                mouth_left = max(0, int(mouth_pts[:, 0].min() - mouth_h * 0.5))
                mouth_right = min(face_w, int(mouth_pts[:, 0].max() + mouth_h * 0.5))
                if mouth_left > 0:
                    for x_idx in range(mouth_left):
                        mask[:, x_idx] *= x_idx / max(1, mouth_left)
                if mouth_right < face_w:
                    for x_idx in range(mouth_right, face_w):
                        mask[:, x_idx] *= 1.0 - (x_idx - mouth_right) / max(1, face_w - mouth_right)

                used_landmarks = True
            except Exception:
                used_landmarks = False

        # ── FALLBACK: Vertical gradient mask ──
        if not used_landmarks:
            blend_start = int(face_h * 0.48)
            blend_end = int(face_h * 0.62)

            for y_idx in range(blend_start, face_h):
                if y_idx < blend_end:
                    alpha = (y_idx - blend_start) / max(1, blend_end - blend_start)
                else:
                    alpha = 1.0
                mask[y_idx, :] = alpha

            # Feather horizontal edges to avoid hard vertical seams
            edge = max(3, int(face_w * 0.08))
            for x_idx in range(edge):
                factor = x_idx / edge
                mask[:, x_idx] *= factor
                mask[:, face_w - 1 - x_idx] *= factor

        # Gaussian blur for smooth transitions
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3)

        # Alpha-blend mouth region
        face_region = result[y1:y2, x1:x2].astype(np.float32)
        lip_region = lip_resized.astype(np.float32)
        mask_3d = mask[:, :, np.newaxis]
        blended = face_region * (1.0 - mask_3d) + lip_region * mask_3d
        result[y1:y2, x1:x2] = blended.astype(np.uint8)
        return result
