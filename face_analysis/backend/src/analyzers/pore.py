"""
Pore Analyzer — Detects and scores skin pore visibility.
Uses DoG (Difference of Gaussians) + local maxima detection.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from scipy import ndimage

from .base import BaseAnalyzer


class PoreAnalyzer(BaseAnalyzer):
    """
    Pore detection using multi-scale Difference of Gaussians (DoG)
    and local extrema detection. Classifies pores as fine/medium/enlarged.
    """

    # Scale ranges for different pore sizes (sigma pairs)
    SCALES = [
        (1.0, 2.0, "fine"),       # Fine pores
        (2.0, 4.0, "medium"),     # Medium pores
        (4.0, 8.0, "enlarged"),   # Enlarged pores
    ]

    def __init__(self, device: str = "cpu"):
        super().__init__(name="pore_analyzer", device=device)

    def load_model(self) -> None:
        """No external model needed — pure CV approach."""
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

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = gray.shape

        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray.astype(np.uint8)).astype(np.float64)

        # Multi-scale DoG for pore detection
        pore_map = np.zeros((h, w), dtype=np.float64)
        pore_counts = {"fine": 0, "medium": 0, "enlarged": 0}
        total_pores = 0

        for sigma1, sigma2, size_class in self.SCALES:
            dog = self._difference_of_gaussians(enhanced, sigma1, sigma2)

            # Find local minima (dark spots = pores)
            local_min = ndimage.minimum_filter(dog, size=int(sigma2 * 3))
            candidates = (dog == local_min) & (dog < -np.std(dog) * 1.5)

            if mask is not None:
                candidates = candidates & (mask > 0)

            count = int(np.sum(candidates))
            pore_counts[size_class] = count
            total_pores += count

            # Weight by severity
            weight = {"fine": 0.3, "medium": 0.6, "enlarged": 1.0}[size_class]
            pore_map += candidates.astype(np.float64) * weight

        # Generate density heatmap using Gaussian kernel
        if total_pores > 0:
            density_map = ndimage.gaussian_filter(pore_map, sigma=15)
            if density_map.max() > 0:
                density_map = density_map / density_map.max()
        else:
            density_map = np.zeros((h, w), dtype=np.float32)

        # Score: fewer/smaller pores = higher score
        area = np.sum(mask > 0) if mask is not None else h * w
        pore_density = total_pores / max(area / 10000, 1)  # Pores per 100x100 patch
        score = max(0, min(100, 100 - pore_density * 5))

        detections = self._classify_pores(pore_counts, zone)

        return {
            "score": round(score, 1),
            "detections": detections,
            "heatmap": density_map.astype(np.float32),
            "pore_counts": pore_counts,
            "total_pores": total_pores,
            "density_per_cm2": round(pore_density, 1),
        }

    def _difference_of_gaussians(
        self, image: np.ndarray, sigma1: float, sigma2: float
    ) -> np.ndarray:
        """Compute Difference of Gaussians."""
        g1 = ndimage.gaussian_filter(image, sigma=sigma1)
        g2 = ndimage.gaussian_filter(image, sigma=sigma2)
        return g1 - g2

    def _classify_pores(self, counts: Dict[str, int], zone: Optional[str]) -> list:
        """Classify pore detections into severity categories."""
        detections = []
        total = sum(counts.values())

        if counts["enlarged"] > 10:
            detections.append({
                "type": "enlarged_pores",
                "severity": "severe" if counts["enlarged"] > 30 else "moderate",
                "count": counts["enlarged"],
                "zone": zone or "full_face",
            })
        if counts["medium"] > 20:
            detections.append({
                "type": "visible_pores",
                "severity": "moderate" if counts["medium"] > 50 else "mild",
                "count": counts["medium"],
                "zone": zone or "full_face",
            })

        return detections
