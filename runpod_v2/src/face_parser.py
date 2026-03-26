"""
BiSeNet Face Parsing — Pixel-perfect face segmentation for mask generation.

Replaces the oval ellipse mask with a real semantic segmentation mask that
follows actual face contours: jawline, ears, hair, beard, neck.

Model: BiSeNet trained on CelebAMask-HQ (19 classes)
Input: 512×512 RGB
Output: 512×512 segmentation map

Class labels (CelebAMask-HQ):
  0  = Background
  1  = Skin
  2  = Left eyebrow
  3  = Right eyebrow
  4  = Left eye
  5  = Right eye
  6  = Eyeglasses
  7  = Left ear
  8  = Right ear
  9  = Earring
  10 = Nose
  11 = Mouth (inner)
  12 = Upper lip
  13 = Lower lip
  14 = Neck
  15 = Necklace
  16 = Cloth
  17 = Hair
  18 = Hat
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logger.warning("onnxruntime not available — FaceParser disabled")

from config import (
    ENABLE_FACE_PARSING,
    FACE_PARSING_MODEL,
    PARSING_CLASSES,
    SMOOTHING_ALPHA,
    ENABLE_TEMPORAL_SMOOTHING,
)

# Map class names → label indices
CLASS_MAP = {
    "background": 0,
    "skin": 1,
    "left_eyebrow": 2,
    "right_eyebrow": 3,
    "left_eye": 4,
    "right_eye": 5,
    "eyeglasses": 6,
    "left_ear": 7,
    "right_ear": 8,
    "earring": 9,
    "nose": 10,
    "mouth": 11,
    "upper_lip": 12,
    "lower_lip": 13,
    "neck": 14,
    "necklace": 15,
    "cloth": 16,
    "hair": 17,
    "hat": 18,
}

# Default classes to include in the face mask
DEFAULT_FACE_CLASSES = [
    "skin", "left_eyebrow", "right_eyebrow",
    "left_eye", "right_eye", "eyeglasses",
    "left_ear", "right_ear",
    "nose", "mouth", "upper_lip", "lower_lip",
]

# Additional classes that can be optionally included
OPTIONAL_CLASSES = {
    "hair": ["hair", "hat"],
    "ears": ["left_ear", "right_ear", "earring"],
    "neck": ["neck"],
    "face": DEFAULT_FACE_CLASSES,
}


class FaceParser:
    """
    BiSeNet-based face parsing for pixel-perfect mask generation.

    Usage:
        parser = FaceParser(providers)
        mask = parser.parse(frame, face_bbox, session_id)
        # mask is a float32 (H, W) array in [0, 1]
    """

    # BiSeNet input size
    INPUT_SIZE = 512

    def __init__(self, providers: list):
        """Initialize FaceParser with ONNX model.

        Args:
            providers: ONNX execution providers (e.g., CUDAExecutionProvider)
        """
        self.session: Optional[ort.InferenceSession] = None
        self._enabled = ENABLE_FACE_PARSING
        self._include_labels: List[int] = []
        self._smooth_masks: Dict[str, np.ndarray] = {}
        self._parse_every_n = 1  # Parse every N frames (1 = every frame)
        self._frame_counters: Dict[str, int] = {}
        self._cached_masks: Dict[str, np.ndarray] = {}

        if not self._enabled:
            logger.info("FaceParser disabled by config (ENABLE_FACE_PARSING=false)")
            return

        if ort is None:
            logger.warning("FaceParser disabled: onnxruntime not available")
            self._enabled = False
            return

        if not os.path.exists(FACE_PARSING_MODEL):
            logger.warning(
                "FaceParser disabled: model not found at %s", FACE_PARSING_MODEL
            )
            self._enabled = False
            return

        # Build label set from config
        self._include_labels = self._resolve_labels(PARSING_CLASSES)
        logger.info(
            "FaceParser including classes: %s → labels %s",
            PARSING_CLASSES,
            self._include_labels,
        )

        # Load ONNX model
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1

            self.session = ort.InferenceSession(
                FACE_PARSING_MODEL, sess_options=sess_opts, providers=providers
            )
            input_info = self.session.get_inputs()[0]
            logger.info(
                "FaceParser loaded: %s, input=%s %s",
                FACE_PARSING_MODEL,
                input_info.name,
                input_info.shape,
            )
        except Exception as e:
            logger.error("FaceParser model load failed: %s", e)
            self.session = None
            self._enabled = False

    def is_ready(self) -> bool:
        """Check if parser is loaded and ready."""
        return self._enabled and self.session is not None

    def _resolve_labels(self, class_names: List[str]) -> List[int]:
        """Convert class name list to label indices.

        Supports both individual class names and group names
        (e.g., 'face', 'hair', 'ears', 'neck').
        """
        labels = set()
        for name in class_names:
            name_lower = name.lower().strip()
            if name_lower in OPTIONAL_CLASSES:
                # Group name — expand to individual classes
                for cls in OPTIONAL_CLASSES[name_lower]:
                    if cls in CLASS_MAP:
                        labels.add(CLASS_MAP[cls])
            elif name_lower in CLASS_MAP:
                labels.add(CLASS_MAP[name_lower])
            else:
                logger.warning("Unknown parsing class: '%s' — skipping", name)
        return sorted(labels)

    def parse(
        self,
        frame: np.ndarray,
        face_bbox: np.ndarray,
        session_id: str = "",
    ) -> Optional[np.ndarray]:
        """Generate a pixel-perfect face mask using BiSeNet segmentation.

        Args:
            frame: Full BGR frame
            face_bbox: Face bounding box [x1, y1, x2, y2]
            session_id: For temporal smoothing cache

        Returns:
            Float32 mask (roi_h, roi_w) in [0, 1], or None on failure
        """
        if not self.is_ready():
            return None

        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = face_bbox.astype(int)

            # Expand bbox by 50% to capture hair, ears, neck
            face_w = x2 - x1
            face_h = y2 - y1
            expand_x = int(face_w * 0.5)
            expand_y = int(face_h * 0.5)

            crop_x1 = max(0, x1 - expand_x)
            crop_y1 = max(0, y1 - expand_y)
            crop_x2 = min(w, x2 + expand_x)
            crop_y2 = min(h, y2 + int(expand_y * 0.3))  # Less expansion below chin

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                return None

            # Frame-skip: reuse cached mask if not due for re-parse
            counter_key = f"{session_id}_parse_counter"
            self._frame_counters[counter_key] = (
                self._frame_counters.get(counter_key, 0) + 1
            )
            if (
                self._frame_counters[counter_key] % self._parse_every_n != 0
                and counter_key in self._cached_masks
            ):
                return self._cached_masks[counter_key]

            # Preprocess: resize to 512×512, normalize
            input_img = cv2.resize(crop, (self.INPUT_SIZE, self.INPUT_SIZE))
            input_img = input_img[:, :, ::-1].astype(np.float32)  # BGR → RGB
            # Normalize with ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            input_img = (input_img / 255.0 - mean) / std
            input_img = np.transpose(input_img, (2, 0, 1))  # HWC → CHW
            input_img = input_img[np.newaxis, ...]  # Add batch dim

            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_img})

            # Parse output: argmax over classes → segmentation map
            seg_map = outputs[0]  # (1, 19, 512, 512) or (1, 512, 512)
            if len(seg_map.shape) == 4:
                seg_map = np.argmax(seg_map[0], axis=0)  # (512, 512)
            else:
                seg_map = seg_map[0]  # (512, 512)

            # Build binary mask from selected classes
            mask_512 = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE), dtype=np.float32)
            for label in self._include_labels:
                mask_512[seg_map == label] = 1.0

            # Resize mask back to crop size
            crop_h, crop_w = crop.shape[:2]
            mask_crop = cv2.resize(
                mask_512, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
            )

            # Create full-frame-sized mask
            full_mask = np.zeros((h, w), dtype=np.float32)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = mask_crop

            # Apply light Gaussian blur for smooth edges (2px feather)
            full_mask = cv2.GaussianBlur(full_mask, (5, 5), 1.0)

            # Temporal smoothing to prevent mask boundary flicker
            if ENABLE_TEMPORAL_SMOOTHING and session_id:
                smooth_key = f"{session_id}_parse_mask"
                if smooth_key in self._smooth_masks:
                    prev = self._smooth_masks[smooth_key]
                    if prev.shape == full_mask.shape:
                        full_mask = (
                            SMOOTHING_ALPHA * prev
                            + (1.0 - SMOOTHING_ALPHA) * full_mask
                        )
                self._smooth_masks[smooth_key] = full_mask.copy()

            # Cache for frame-skip
            self._cached_masks[counter_key] = full_mask

            return full_mask

        except Exception as e:
            logger.error("FaceParser.parse() failed: %s", e, exc_info=True)
            return None

    def parse_roi(
        self,
        frame: np.ndarray,
        face_bbox: np.ndarray,
        roi_bounds: Tuple[int, int, int, int],
        session_id: str = "",
    ) -> Optional[np.ndarray]:
        """Generate a mask cropped to the ROI region.

        Args:
            frame: Full BGR frame
            face_bbox: Face bounding box [x1, y1, x2, y2]
            roi_bounds: (roi_x1, roi_y1, roi_x2, roi_y2) for the output crop
            session_id: For temporal smoothing

        Returns:
            Float32 mask (roi_h, roi_w) in [0, 1], or None
        """
        full_mask = self.parse(frame, face_bbox, session_id)
        if full_mask is None:
            return None

        roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
        roi_mask = full_mask[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi_mask

    def set_parse_frequency(self, every_n_frames: int = 1):
        """Set how often to re-run parsing (1=every frame, 3=every 3rd frame).

        On slower GPUs like T4, set to 3 to reduce overhead.
        The cached mask is used for intermediate frames.
        """
        self._parse_every_n = max(1, every_n_frames)
        logger.info("FaceParser parse frequency: every %d frames", self._parse_every_n)

    def cleanup_session(self, session_id: str):
        """Free cached data for a session."""
        keys_to_remove = [
            k for k in self._smooth_masks if k.startswith(session_id)
        ]
        for k in keys_to_remove:
            del self._smooth_masks[k]

        keys_to_remove = [
            k for k in self._cached_masks if k.startswith(session_id)
        ]
        for k in keys_to_remove:
            del self._cached_masks[k]

        keys_to_remove = [
            k for k in self._frame_counters if k.startswith(session_id)
        ]
        for k in keys_to_remove:
            del self._frame_counters[k]

        logger.debug("FaceParser cleaned up session %s", session_id)
