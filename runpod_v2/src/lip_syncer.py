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
from typing import Dict, Optional

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
        # Fix #3: Cache previous output for temporal smoothing (per session)
        self._prev_outputs: Dict[str, np.ndarray] = {}

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

            # Apply Tukey window (cosine-tapered) to smooth only the edges.
            # Unlike Hann which zeroes ~25% at each end (killing lip amplitude),
            # Tukey(alpha=0.25) keeps the middle 75% at full strength and only
            # tapers the outer 12.5% on each side — preserving speech energy.
            if len(audio) > 64:
                from scipy.signal import windows as _wins
                audio = audio * _wins.tukey(len(audio), alpha=0.25)

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

    def infer(self, face: np.ndarray, mel: np.ndarray, session_id: str = "") -> Optional[np.ndarray]:
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

            # Map inputs: match by name first (reliable), fall back to shape heuristic
            inputs = {}
            for name, inp in zip(self.input_names, self.session.get_inputs()):
                name_lower = name.lower()
                if 'mel' in name_lower or 'audio' in name_lower:
                    inputs[name] = mel_input
                elif 'face' in name_lower or 'img' in name_lower or 'image' in name_lower:
                    inputs[name] = face_input
                elif len(inp.shape) == 4:
                    inputs[name] = face_input
                else:
                    inputs[name] = mel_input

            pred = self.session.run(None, inputs)[0]
            pred = np.transpose(pred[0], (1, 2, 0))
            pred = (pred * 255).clip(0, 255).astype(np.uint8)
            pred = pred[:, :, ::-1]  # RGB → BGR

            # Light temporal smoothing — 85% current + 15% previous
            # Higher current weight preserves crisp lip articulation (m/b/p closures)
            # while still removing single-frame jitter
            prev = self._prev_outputs.get(session_id) if session_id else None
            if prev is not None and prev.shape == pred.shape:
                pred = cv2.addWeighted(pred, 0.85, prev, 0.15, 0)
            if session_id:
                self._prev_outputs[session_id] = pred.copy()

            return pred
        except Exception as e:
            print(f"LipSync inference failed: {e}")
            return None

    def apply_mouth_only(
        self,
        result: np.ndarray,
        face_bbox: tuple,
        lip_output: np.ndarray,
    ) -> np.ndarray:
        """
        Blend Wav2Lip output into the mouth region only with feathered edges.

        Instead of replacing the entire face crop (which destroys the
        high-quality face-swap output), this preserves the upper face
        (eyes, nose, forehead) and only modifies the mouth/chin area
        using a vertical gradient mask with feathered edges.
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

        # Build a vertical gradient mask:
        #   0 at top (preserve face-swap output)
        #   gradual blend from 52-68% of face height
        #   1 at bottom (use Wav2Lip lip-synced output)
        # Shifted lower to better match actual mouth position and reduce dark seams
        mask = np.zeros((face_h, face_w), dtype=np.float32)
        blend_start = int(face_h * 0.52)
        blend_end = int(face_h * 0.68)

        for y_idx in range(blend_start, face_h):
            if y_idx < blend_end:
                alpha = (y_idx - blend_start) / max(1, blend_end - blend_start)
            else:
                alpha = 1.0
            mask[y_idx, :] = alpha

        # Feather horizontal edges to avoid hard vertical seams
        # Wider feather absorbs bbox pixel shifts and reduces dark edge artifacts
        edge = max(5, int(face_w * 0.20))
        for x_idx in range(edge):
            factor = x_idx / edge
            mask[:, x_idx] *= factor
            mask[:, face_w - 1 - x_idx] *= factor

        # Gaussian blur for smooth transitions
        # Larger sigma makes boundary shifts invisible, reducing dark edge artifacts
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=11, sigmaY=11)

        # Alpha-blend mouth region
        face_region = result[y1:y2, x1:x2].astype(np.float32)
        lip_region = lip_resized.astype(np.float32)
        mask_3d = mask[:, :, np.newaxis]
        blended = face_region * (1.0 - mask_3d) + lip_region * mask_3d
        result[y1:y2, x1:x2] = blended.astype(np.uint8)
        return result

    def cleanup_session(self, session_id: str):
        """Clean up per-session lip sync state."""
        self._prev_outputs.pop(session_id, None)
