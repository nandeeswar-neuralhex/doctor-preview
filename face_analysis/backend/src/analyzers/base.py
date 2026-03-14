"""
Abstract Base Analyzer — All skin analyzers implement this interface.
Follows Open/Closed Principle: extend by adding new analyzers, not modifying existing ones.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseAnalyzer(ABC):
    """
    Contract for all facial skin analyzers.

    Every analyzer receives a pre-processed face ROI (aligned, normalized)
    and returns a structured result dict with scores and detection data.
    """

    def __init__(self, name: str, device: str = "cpu"):
        self._name = name
        self._device = device
        self._is_loaded = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load_model(self) -> None:
        """Load ML model weights into memory. Called once at startup."""
        ...

    @abstractmethod
    def analyze(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run analysis on a face image.

        Args:
            face_image: BGR face crop, shape (H, W, 3), dtype uint8
            landmarks: 468-point facial landmarks, shape (468, 2)
            mask: Binary skin mask, shape (H, W), dtype uint8
            zone: Optional zone name to restrict analysis

        Returns:
            Dict with at minimum:
                - "score": float 0-100  (100 = healthy)
                - "detections": list of detected issues
                - "heatmap": np.ndarray or None (H, W) float32 0-1
        """
        ...

    def unload(self) -> None:
        """Release model from memory."""
        self._is_loaded = False

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "unloaded"
        return f"<{self.__class__.__name__}('{self._name}', {status})>"
