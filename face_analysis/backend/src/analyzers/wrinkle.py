"""
Wrinkle Analyzer — Detects and scores wrinkles and fine lines.
Uses Canny edge detection + Gabor filters + morphological analysis.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np
from scipy import ndimage

from .base import BaseAnalyzer


class WrinkleAnalyzer(BaseAnalyzer):
    """
    Multi-scale wrinkle detection using:
    1. Gabor filter bank (oriented edge detection)
    2. Frangi vesselness filter (ridge detection)
    3. Morphological line extraction
    """

    # Gabor filter parameters for different wrinkle orientations
    GABOR_PARAMS = [
        {"theta": t, "sigma": s, "lambd": l, "gamma": 0.5}
        for t in np.arange(0, np.pi, np.pi / 8)  # 8 orientations
        for s in [2.0, 3.0]  # 2 scales
        for l in [4.0, 8.0]  # 2 wavelengths
    ]

    def __init__(self, device: str = "cpu"):
        super().__init__(name="wrinkle_analyzer", device=device)
        self._gabor_kernels = []

    def load_model(self) -> None:
        """Pre-compute Gabor filter bank."""
        self._gabor_kernels = []
        for params in self.GABOR_PARAMS:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=params["sigma"],
                theta=params["theta"],
                lambd=params["lambd"],
                gamma=params["gamma"],
                psi=0,
            )
            kernel /= kernel.sum() + 1e-7
            self._gabor_kernels.append(kernel)
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

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Apply CLAHE for contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 1. Gabor filter bank response (fine lines)
        gabor_response = self._apply_gabor_bank(enhanced)

        # 2. Frangi-like ridge detection (deep wrinkles)
        ridge_map = self._detect_ridges(enhanced)

        # 3. Morphological line detection
        line_map = self._detect_lines(enhanced)

        # Combine all detections
        combined = np.clip(
            0.4 * gabor_response + 0.35 * ridge_map + 0.25 * line_map,
            0, 1
        ).astype(np.float32)

        # Apply skin mask if provided
        if mask is not None:
            combined = combined * (mask.astype(np.float32) / 255.0)

        # Classify wrinkles
        detections = self._classify_wrinkles(combined, landmarks, zone)

        # Score: higher = fewer wrinkles = healthier
        wrinkle_density = np.mean(combined[combined > 0.3])
        if np.isnan(wrinkle_density):
            wrinkle_density = 0.0
        score = max(0, min(100, 100 - wrinkle_density * 200))

        return {
            "score": round(score, 1),
            "detections": detections,
            "heatmap": combined,
            "wrinkle_density": float(wrinkle_density),
            "deep_wrinkle_count": int(np.sum(ridge_map > 0.6)),
            "fine_line_count": int(np.sum(gabor_response > 0.4) / 100),
        }

    def _apply_gabor_bank(self, gray: np.ndarray) -> np.ndarray:
        """Apply Gabor filter bank and return max response."""
        responses = np.zeros_like(gray, dtype=np.float64)
        for kernel in self._gabor_kernels:
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            responses = np.maximum(responses, np.abs(filtered))

        # Normalize to 0-1
        if responses.max() > 0:
            responses = responses / responses.max()
        return responses.astype(np.float32)

    def _detect_ridges(self, gray: np.ndarray) -> np.ndarray:
        """Frangi-like ridge detection for deep wrinkles."""
        blurred = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), sigmaX=2.0)

        # Second-order derivatives (Hessian)
        Ixx = ndimage.gaussian_filter(blurred, sigma=2, order=[0, 2])
        Iyy = ndimage.gaussian_filter(blurred, sigma=2, order=[2, 0])
        Ixy = ndimage.gaussian_filter(blurred, sigma=2, order=[1, 1])

        # Eigenvalues of Hessian
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2

        discriminant = np.sqrt(np.maximum(trace**2 - 4 * det, 0))
        lambda1 = (trace + discriminant) / 2
        lambda2 = (trace - discriminant) / 2

        # Frangi vesselness: strong when one eigenvalue >> other
        Rb2 = np.where(lambda1 != 0, (lambda2 / (lambda1 + 1e-10))**2, 0)
        S2 = lambda1**2 + lambda2**2
        beta = 0.5
        c = 0.5 * S2.max() if S2.max() > 0 else 1.0

        vesselness = np.exp(-Rb2 / (2 * beta**2)) * (1 - np.exp(-S2 / (2 * c)))
        vesselness = np.where(lambda2 < 0, vesselness, 0)  # Dark ridges only

        if vesselness.max() > 0:
            vesselness = vesselness / vesselness.max()
        return vesselness.astype(np.float32)

    def _detect_lines(self, gray: np.ndarray) -> np.ndarray:
        """Morphological line extraction."""
        result = np.zeros_like(gray, dtype=np.float32)

        for angle in range(0, 180, 15):
            length = 15
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
            M = cv2.getRotationMatrix2D((length // 2, 0), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (length, length))

            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            diff = cv2.absdiff(gray, opened).astype(np.float32)
            result = np.maximum(result, diff)

        if result.max() > 0:
            result = result / result.max()
        return result

    def _classify_wrinkles(
        self, heatmap: np.ndarray, landmarks: Optional[np.ndarray], zone: Optional[str]
    ) -> list:
        """Classify detected wrinkles into categories."""
        detections = []
        deep_threshold = 0.6
        fine_threshold = 0.3

        deep_count = int(np.sum(heatmap > deep_threshold) / 50)
        fine_count = int(np.sum((heatmap > fine_threshold) & (heatmap <= deep_threshold)) / 100)

        if deep_count > 0:
            severity = "severe" if deep_count > 10 else "moderate" if deep_count > 5 else "mild"
            detections.append({
                "type": "deep_wrinkles",
                "severity": severity,
                "count": deep_count,
                "zone": zone or "full_face",
            })

        if fine_count > 0:
            severity = "moderate" if fine_count > 20 else "mild"
            detections.append({
                "type": "fine_lines",
                "severity": severity,
                "count": fine_count,
                "zone": zone or "full_face",
            })

        return detections
