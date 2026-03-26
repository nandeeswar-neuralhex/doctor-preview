"""
Pigmentation Analyzer — Detects brown spots, melasma, UV damage.
Uses LAB color space decomposition for melanin channel extraction.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from scipy import ndimage

from .base import BaseAnalyzer


class PigmentationAnalyzer(BaseAnalyzer):
    """
    Pigmentation analysis using:
    1. LAB color space — L (lightness) + B channel (blue-yellow = melanin proxy)
    2. ITA (Individual Typology Angle) for Fitzpatrick classification
    3. Adaptive thresholding for spot detection
    4. Connected component analysis for spot counting/sizing
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(name="pigmentation_analyzer", device=device)

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

        h, w = face_image.shape[:2]

        # Convert to LAB color space
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB).astype(np.float64)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Melanin index (simplified): based on L and B channels
        # Lower L + higher B = more melanin
        melanin_map = (255 - L) * 0.6 + (B - 128) * 0.4
        melanin_map = np.clip(melanin_map, 0, 255)

        if mask is not None:
            melanin_map = melanin_map * (mask.astype(np.float64) / 255.0)

        # Normalize as deviation from person's baseline (not absolute melanin)
        # This prevents dark skin tones from showing as entirely "problematic"
        if mask is not None:
            skin_pixels = melanin_map[mask > 0]
            if len(skin_pixels) > 0:
                baseline = float(np.median(skin_pixels))
            else:
                baseline = float(np.median(melanin_map))
        else:
            baseline = float(np.median(melanin_map))
        deviation_map = np.abs(melanin_map - baseline)
        if deviation_map.max() > 0:
            melanin_norm = deviation_map / deviation_map.max()
        else:
            melanin_norm = np.zeros_like(melanin_map)

        # Detect hyperpigmentation spots
        spots_mask = self._detect_spots(L, mask)
        spot_count, spot_sizes = self._count_spots(spots_mask)

        # UV damage estimation (uniformity of pigmentation)
        uv_score = self._estimate_uv_damage(melanin_norm, mask)

        # Compute ITA for Fitzpatrick type
        ita = self._compute_ita(L, B, mask)
        fitzpatrick = self._ita_to_fitzpatrick(ita)

        # Score: fewer spots + more uniform = higher score
        area = np.sum(mask > 0) if mask is not None else h * w
        spot_density = spot_count / max(area / 100000, 1)
        uniformity = 1.0 - np.std(melanin_norm[mask > 0] if mask is not None else melanin_norm)
        score = max(0, min(100, (uniformity * 60) + max(0, 40 - spot_density * 10)))

        detections = self._classify(spot_count, spot_sizes, uv_score, zone)

        return {
            "score": round(score, 1),
            "detections": detections,
            "heatmap": melanin_norm.astype(np.float32),
            "spot_count": spot_count,
            "spot_sizes": spot_sizes,
            "melanin_index": float(np.mean(melanin_norm[mask > 0]) if mask is not None else np.mean(melanin_norm)),
            "uv_damage_score": round(uv_score, 1),
            "ita": round(ita, 1),
            "fitzpatrick_estimate": fitzpatrick,
        }

    def _detect_spots(self, L_channel: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Detect hyperpigmentation spots using adaptive thresholding on L channel."""
        L_uint8 = L_channel.astype(np.uint8)

        # Local mean — spots are darker than surrounding area
        local_mean = cv2.GaussianBlur(L_uint8, (51, 51), 0).astype(np.float64)
        diff = local_mean - L_uint8.astype(np.float64)

        # Threshold: spots are significantly darker than local mean
        threshold = max(np.std(diff) * 1.5, 8.0)
        spots = (diff > threshold).astype(np.uint8) * 255

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, kernel)
        spots = cv2.morphologyEx(spots, cv2.MORPH_CLOSE, kernel)

        if mask is not None:
            spots = spots & mask

        return spots

    def _count_spots(self, spots_mask: np.ndarray):
        """Count and measure individual spots."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            spots_mask, connectivity=8
        )
        spot_sizes = []
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if 10 < area < 5000:  # Filter noise and too-large regions
                spot_sizes.append(int(area))

        return len(spot_sizes), sorted(spot_sizes, reverse=True)[:20]

    def _estimate_uv_damage(self, melanin_norm: np.ndarray, mask: Optional[np.ndarray]) -> float:
        """Estimate UV damage from pigmentation non-uniformity."""
        if mask is not None:
            region = melanin_norm[mask > 0]
        else:
            region = melanin_norm.flatten()

        if len(region) == 0:
            return 0.0

        # Higher variance = more UV damage
        variance = float(np.var(region))
        return min(100, variance * 500)

    def _compute_ita(self, L: np.ndarray, B: np.ndarray, mask: Optional[np.ndarray]) -> float:
        """Compute Individual Typology Angle for skin type classification."""
        if mask is not None:
            L_mean = float(np.mean(L[mask > 0]))
            B_mean = float(np.mean(B[mask > 0]))
        else:
            L_mean = float(np.mean(L))
            B_mean = float(np.mean(B))

        # OpenCV LAB outputs L in [0,255]; ITA formula requires CIELAB L* in [0,100]
        L_scaled = L_mean * 100.0 / 255.0
        ita = np.degrees(np.arctan2(L_scaled - 50, B_mean - 128))
        return ita

    def _ita_to_fitzpatrick(self, ita: float) -> int:
        """Convert ITA angle to Fitzpatrick skin type."""
        if ita > 55:
            return 1
        elif ita > 41:
            return 2
        elif ita > 28:
            return 3
        elif ita > 10:
            return 4
        elif ita > -30:
            return 5
        else:
            return 6

    def _classify(self, count: int, sizes: list, uv_score: float, zone: Optional[str]) -> list:
        detections = []
        if count > 0:
            severity = "severe" if count > 15 else "moderate" if count > 5 else "mild"
            detections.append({
                "type": "brown_spots",
                "severity": severity,
                "count": count,
                "zone": zone or "full_face",
            })
        if uv_score > 40:
            detections.append({
                "type": "uv_damage",
                "severity": "severe" if uv_score > 70 else "moderate",
                "count": 0,
                "zone": zone or "full_face",
            })
        return detections
