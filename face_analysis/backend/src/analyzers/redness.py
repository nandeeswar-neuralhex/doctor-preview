"""
Redness Analyzer — Detects erythema, rosacea, inflammation.
Uses hemoglobin channel extraction from RGB decomposition.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from scipy import ndimage

from .base import BaseAnalyzer


class RednessAnalyzer(BaseAnalyzer):
    """
    Redness detection using:
    1. Hemoglobin index extraction (optical density of red channel)
    2. A-channel from LAB (red-green axis)
    3. Color ratio analysis (R / (G+B))
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(name="redness_analyzer", device=device)

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
        bgr = face_image.astype(np.float64) + 1e-7  # Avoid log(0)
        B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

        # 1. Hemoglobin index via optical density
        # OD = -log10(I/I0), hemoglobin absorbs green more than red
        hemo_map = self._extract_hemoglobin(R, G, B)

        # 2. LAB A-channel (positive A = red)
        lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB).astype(np.float64)
        a_channel = lab[:, :, 1] - 128  # Center at 0
        a_norm = np.clip(a_channel / 30.0, 0, 1)  # Normalize positives

        # 3. Red ratio: R / (R + G + B)
        total = R + G + B
        red_ratio = R / total
        red_excess = np.clip((red_ratio - 0.33) * 10, 0, 1)  # > 0.33 is reddish

        # Combined redness map
        redness_map = np.clip(
            0.4 * hemo_map + 0.35 * a_norm + 0.25 * red_excess, 0, 1
        ).astype(np.float32)

        if mask is not None:
            redness_map = redness_map * (mask.astype(np.float32) / 255.0)

        # Detect redness zones
        redness_zones = self._detect_zones(redness_map, mask)

        # Score: less redness = higher score
        # Factor in both intensity AND coverage for accurate scoring
        red_pixels = redness_map[redness_map > 0.1]
        skin_pixels = np.sum(mask > 0) if mask is not None else h * w
        red_intensity = float(np.mean(red_pixels)) if len(red_pixels) > 0 else 0.0
        red_coverage = len(red_pixels) / max(skin_pixels, 1)
        mean_redness = red_intensity * (red_coverage ** 0.5)  # sqrt to soften coverage impact
        score = max(0, min(100, 100 - mean_redness * 120))

        detections = self._classify(mean_redness, redness_zones, zone)

        return {
            "score": round(score, 1),
            "detections": detections,
            "heatmap": redness_map,
            "hemoglobin_index": float(np.mean(hemo_map[mask > 0]) if mask is not None else np.mean(hemo_map)),
            "redness_zones": redness_zones,
        }

    def _extract_hemoglobin(self, R: np.ndarray, G: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Extract hemoglobin concentration proxy.
        Based on Beer-Lambert law applied to skin optics.
        Hemoglobin absorbs strongly at 540-580nm (green channel).
        """
        # Optical density of green relative to red
        od_g = -np.log10(G / 255.0 + 1e-7)
        od_r = -np.log10(R / 255.0 + 1e-7)

        # Hemoglobin index: OD(green) - OD(red) — positive = more hemoglobin
        hemo = od_g - od_r
        hemo = np.clip(hemo, 0, None)

        if hemo.max() > 0:
            hemo = hemo / hemo.max()
        return hemo.astype(np.float32)

    def _detect_zones(self, redness_map: np.ndarray, mask: Optional[np.ndarray]) -> list:
        """Identify contiguous redness zones."""
        binary = (redness_map > 0.4).astype(np.uint8)
        if mask is not None:
            binary = binary & (mask > 0).astype(np.uint8)

        labeled, num_features = ndimage.label(binary)
        zones = []
        for i in range(1, min(num_features + 1, 20)):
            region = labeled == i
            area = int(np.sum(region))
            if area > 100:  # Minimum area threshold
                intensity = float(np.mean(redness_map[region]))
                y_coords, x_coords = np.where(region)
                zones.append({
                    "area_px": area,
                    "intensity": round(intensity, 3),
                    "center": (int(np.mean(x_coords)), int(np.mean(y_coords))),
                })
        return zones

    def _classify(self, mean_redness: float, zones: list, zone: Optional[str]) -> list:
        detections = []
        if mean_redness > 0.3:
            severity = "severe" if mean_redness > 0.6 else "moderate" if mean_redness > 0.4 else "mild"
            detections.append({
                "type": "redness",
                "severity": severity,
                "count": len(zones),
                "zone": zone or "full_face",
            })
        if len(zones) > 3 and mean_redness > 0.35:
            detections.append({
                "type": "possible_rosacea",
                "severity": "moderate",
                "count": len(zones),
                "zone": zone or "cheeks",
            })
        return detections
