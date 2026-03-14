"""
Symmetry Analyzer — Computes facial symmetry scores.
Compares left vs right halves using landmark distances and appearance.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import BaseAnalyzer


class SymmetryAnalyzer(BaseAnalyzer):
    """
    Facial symmetry analysis by:
    1. Landmark-based geometric symmetry (L/R distance comparison)
    2. Appearance-based symmetry (SSIM of mirrored halves)
    3. Per-feature symmetry (eyes, cheeks, jawline, lips)
    """

    # Landmark pairs: (left_idx, right_idx) for symmetry comparison
    SYMMETRY_PAIRS = {
        "eye_outer": (33, 263),
        "eye_inner": (133, 362),
        "eyebrow_outer": (46, 276),
        "eyebrow_inner": (105, 334),
        "cheek_upper": (116, 345),
        "cheek_lower": (187, 411),
        "mouth_corner": (61, 291),
        "jaw_angle": (132, 361),
        "jaw_mid": (58, 288),
        "nostril": (48, 278),
    }

    def __init__(self, device: str = "cpu"):
        super().__init__(name="symmetry_analyzer", device=device)

    def load_model(self) -> None:
        self._is_loaded = True

    def analyze(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._is_loaded:
            self.load_model()

        if landmarks is None:
            return {
                "score": 0,
                "detections": [],
                "heatmap": None,
                "error": "Landmarks required for symmetry analysis",
            }

        h, w = face_image.shape[:2]

        # Compute midline from nose bridge landmarks
        midline_top = landmarks[10]    # Forehead center
        midline_bottom = landmarks[152]  # Chin
        midline_x = (midline_top[0] + midline_bottom[0]) / 2

        # 1. Geometric symmetry from landmark pairs
        geometric = self._geometric_symmetry(landmarks, midline_x)

        # 2. Appearance symmetry (flip and compare)
        appearance = self._appearance_symmetry(face_image, int(midline_x), mask)

        # 3. Per-feature scores
        eye_sym = self._feature_symmetry(landmarks, midline_x,
                                          [(33, 263), (133, 362), (46, 276), (105, 334)])
        cheek_sym = self._feature_symmetry(landmarks, midline_x,
                                            [(116, 345), (187, 411)])
        jaw_sym = self._feature_symmetry(landmarks, midline_x,
                                          [(132, 361), (58, 288)])
        lip_sym = self._feature_symmetry(landmarks, midline_x,
                                          [(61, 291)])

        # Midline deviation
        nose_tip = landmarks[1]
        chin = landmarks[152]
        midline_deviation = abs(nose_tip[0] - chin[0])

        # Combined score
        score = (
            geometric * 0.4
            + appearance * 0.3
            + (eye_sym + cheek_sym + jaw_sym + lip_sym) / 4 * 0.3
        )

        # Generate symmetry heatmap (difference between L/R)
        heatmap = self._generate_heatmap(face_image, int(midline_x), mask)

        return {
            "score": round(score, 1),
            "detections": self._classify(score),
            "heatmap": heatmap,
            "geometric_score": round(geometric, 1),
            "appearance_score": round(appearance, 1),
            "eye_alignment": round(eye_sym, 1),
            "cheek_balance": round(cheek_sym, 1),
            "jawline_symmetry": round(jaw_sym, 1),
            "lip_symmetry": round(lip_sym, 1),
            "midline_deviation_mm": round(midline_deviation * 0.26, 2),  # Approx px-to-mm
        }

    def _geometric_symmetry(self, landmarks: np.ndarray, midline_x: float) -> float:
        """Compare left/right landmark distances from midline."""
        deviations = []
        for name, (left_idx, right_idx) in self.SYMMETRY_PAIRS.items():
            if left_idx >= len(landmarks) or right_idx >= len(landmarks):
                continue
            left_dist = abs(landmarks[left_idx][0] - midline_x)
            right_dist = abs(landmarks[right_idx][0] - midline_x)
            avg = (left_dist + right_dist) / 2
            if avg > 0:
                deviation = abs(left_dist - right_dist) / avg
                deviations.append(deviation)

        if not deviations:
            return 50.0

        mean_deviation = np.mean(deviations)
        # 0 deviation = 100 score, 0.5 deviation = 0 score
        return max(0, min(100, (1 - mean_deviation * 2) * 100))

    def _appearance_symmetry(
        self, image: np.ndarray, midline_x: int, mask: Optional[np.ndarray]
    ) -> float:
        """Compare left half vs mirrored right half using normalized correlation."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        if midline_x <= 0 or midline_x >= w:
            return 50.0

        left_half = gray[:, :midline_x]
        right_half = gray[:, midline_x:]

        # Mirror the right half
        right_flipped = cv2.flip(right_half, 1)

        # Make same size
        min_w = min(left_half.shape[1], right_flipped.shape[1])
        if min_w == 0:
            return 50.0

        left_crop = left_half[:, -min_w:]
        right_crop = right_flipped[:, :min_w]

        # Normalized cross-correlation
        l_norm = left_crop - np.mean(left_crop)
        r_norm = right_crop - np.mean(right_crop)
        l_std = np.std(l_norm) + 1e-7
        r_std = np.std(r_norm) + 1e-7

        ncc = float(np.mean(l_norm * r_norm) / (l_std * r_std))
        # NCC range [-1, 1] → map to [0, 100]
        return max(0, min(100, (ncc + 1) * 50))

    def _feature_symmetry(
        self, landmarks: np.ndarray, midline_x: float,
        pairs: list
    ) -> float:
        """Compute symmetry for a specific feature group."""
        deviations = []
        for left_idx, right_idx in pairs:
            if left_idx >= len(landmarks) or right_idx >= len(landmarks):
                continue
            left_d = abs(landmarks[left_idx][0] - midline_x)
            right_d = abs(landmarks[right_idx][0] - midline_x)
            avg = (left_d + right_d) / 2
            if avg > 0:
                deviations.append(abs(left_d - right_d) / avg)

        if not deviations:
            return 50.0
        mean_dev = np.mean(deviations)
        return max(0, min(100, (1 - mean_dev * 2) * 100))

    def _generate_heatmap(
        self, image: np.ndarray, midline_x: int, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Generate pixel-level symmetry difference heatmap."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        heatmap = np.zeros((h, w), dtype=np.float32)

        min_side = min(midline_x, w - midline_x)
        if min_side <= 0:
            return heatmap

        left = gray[:, midline_x - min_side:midline_x]
        right = gray[:, midline_x:midline_x + min_side]
        right_flipped = cv2.flip(right, 1)

        diff = np.abs(left - right_flipped) / 255.0
        diff_smoothed = cv2.GaussianBlur(diff.astype(np.float32), (15, 15), 0)

        heatmap[:, midline_x - min_side:midline_x] = diff_smoothed
        heatmap[:, midline_x:midline_x + min_side] = cv2.flip(diff_smoothed, 1)

        if mask is not None:
            heatmap = heatmap * (mask.astype(np.float32) / 255.0)

        return heatmap

    def _classify(self, score: float) -> list:
        detections = []
        if score < 85:
            severity = "severe" if score < 70 else "moderate" if score < 80 else "mild"
            detections.append({
                "type": "asymmetry",
                "severity": severity,
                "count": 0,
                "zone": "full_face",
            })
        return detections
