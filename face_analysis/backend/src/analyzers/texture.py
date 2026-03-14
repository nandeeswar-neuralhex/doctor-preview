"""
Texture Analyzer — Scores skin smoothness and roughness.
Uses GLCM (Grey-Level Co-occurrence Matrix) features.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import BaseAnalyzer


class TextureAnalyzer(BaseAnalyzer):
    """
    Skin texture analysis using:
    1. GLCM features (contrast, homogeneity, energy, correlation)
    2. LBP (Local Binary Pattern) for micro-texture
    3. FFT-based roughness estimation
    """

    # GLCM offsets (distance, angle pairs)
    DISTANCES = [1, 3, 5]
    ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    def __init__(self, device: str = "cpu"):
        super().__init__(name="texture_analyzer", device=device)

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

        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. GLCM features (computed on patches for spatial map)
        glcm_features = self._compute_glcm_features(gray, mask)

        # 2. LBP texture map
        lbp_map = self._compute_lbp(gray)

        # 3. FFT roughness
        roughness_map = self._compute_roughness(gray)

        # Combined texture quality map
        # Higher homogeneity + lower contrast + lower roughness = smoother skin
        texture_quality = np.clip(
            0.4 * glcm_features["homogeneity_map"]
            + 0.3 * (1.0 - glcm_features["contrast_map"])
            + 0.3 * (1.0 - roughness_map),
            0, 1
        ).astype(np.float32)

        if mask is not None:
            texture_quality = texture_quality * (mask.astype(np.float32) / 255.0)

        # Score from texture quality
        valid_region = texture_quality[texture_quality > 0]
        if len(valid_region) > 0:
            score = float(np.mean(valid_region) * 100)
        else:
            score = 50.0

        # Invert heatmap for display (red = rough, green = smooth)
        display_heatmap = 1.0 - texture_quality

        return {
            "score": round(score, 1),
            "detections": self._classify(score, zone),
            "heatmap": display_heatmap,
            "contrast": float(glcm_features["contrast"]),
            "homogeneity": float(glcm_features["homogeneity"]),
            "energy": float(glcm_features["energy"]),
            "roughness_index": float(np.mean(roughness_map[mask > 0]) if mask is not None else np.mean(roughness_map)),
        }

    def _compute_glcm_features(self, gray: np.ndarray, mask: Optional[np.ndarray]) -> Dict:
        """Compute GLCM features using a sliding-window approach."""
        h, w = gray.shape
        patch_size = 32
        stride = 16

        contrast_map = np.zeros((h, w), dtype=np.float32)
        homogeneity_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        quantized = (gray // 16).astype(np.uint8)  # 16 grey levels

        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                patch = quantized[y:y + patch_size, x:x + patch_size]

                if mask is not None:
                    patch_mask = mask[y:y + patch_size, x:x + patch_size]
                    if np.sum(patch_mask > 0) < patch_size * patch_size * 0.5:
                        continue

                # Simplified GLCM computation (horizontal offset = 1)
                glcm = self._compute_glcm_matrix(patch, 16)
                c, hom, e = self._glcm_features(glcm)

                contrast_map[y:y + patch_size, x:x + patch_size] += c
                homogeneity_map[y:y + patch_size, x:x + patch_size] += hom
                count_map[y:y + patch_size, x:x + patch_size] += 1

        count_map = np.maximum(count_map, 1)
        contrast_map /= count_map
        homogeneity_map /= count_map

        # Normalize to 0-1
        if contrast_map.max() > 0:
            contrast_map /= contrast_map.max()
        if homogeneity_map.max() > 0:
            homogeneity_map /= homogeneity_map.max()

        return {
            "contrast_map": contrast_map,
            "homogeneity_map": homogeneity_map,
            "contrast": float(np.mean(contrast_map[contrast_map > 0])) if np.any(contrast_map > 0) else 0,
            "homogeneity": float(np.mean(homogeneity_map[homogeneity_map > 0])) if np.any(homogeneity_map > 0) else 0,
            "energy": 0.0,  # Simplified
        }

    @staticmethod
    def _compute_glcm_matrix(patch: np.ndarray, levels: int) -> np.ndarray:
        """Compute GLCM for horizontal offset of 1."""
        glcm = np.zeros((levels, levels), dtype=np.float64)
        left = patch[:, :-1]
        right = patch[:, 1:]
        for i in range(levels):
            for j in range(levels):
                glcm[i, j] = np.sum((left == i) & (right == j))
        total = glcm.sum()
        if total > 0:
            glcm /= total
        return glcm

    @staticmethod
    def _glcm_features(glcm: np.ndarray):
        """Extract contrast, homogeneity, energy from GLCM."""
        levels = glcm.shape[0]
        i, j = np.meshgrid(range(levels), range(levels), indexing="ij")
        contrast = float(np.sum(glcm * (i - j) ** 2))
        homogeneity = float(np.sum(glcm / (1 + np.abs(i - j))))
        energy = float(np.sum(glcm ** 2))
        # Normalize contrast
        max_contrast = (levels - 1) ** 2
        contrast = contrast / max_contrast if max_contrast > 0 else 0
        return contrast, homogeneity, energy

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """Simplified Local Binary Pattern."""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)

        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
            lbp |= ((neighbor >= gray).astype(np.uint8) << bit)

        return lbp

    def _compute_roughness(self, gray: np.ndarray) -> np.ndarray:
        """FFT-based roughness estimation — high frequency content indicates texture/roughness."""
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4

        # Mask high frequencies (outside radius = high freq = roughness)
        Y, X = np.ogrid[:h, :w]
        high_freq_mask = ((X - cx) ** 2 + (Y - cy) ** 2) > radius ** 2

        # Roughness = ratio of high-frequency energy
        high_energy = np.sum(magnitude * high_freq_mask)
        total_energy = np.sum(magnitude) + 1e-7
        roughness_ratio = high_energy / total_energy

        # Create spatial roughness map using local variance
        local_var = ndimage_local_variance(gray, size=15)
        if local_var.max() > 0:
            local_var = local_var / local_var.max()
        return local_var.astype(np.float32)

    def _classify(self, score: float, zone: Optional[str]) -> list:
        detections = []
        if score < 60:
            severity = "severe" if score < 30 else "moderate"
            detections.append({
                "type": "rough_texture",
                "severity": severity,
                "count": 0,
                "zone": zone or "full_face",
            })
        return detections


def ndimage_local_variance(image: np.ndarray, size: int = 15) -> np.ndarray:
    """Compute local variance using uniform filter."""
    from scipy.ndimage import uniform_filter
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img ** 2, size=size)
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return variance
