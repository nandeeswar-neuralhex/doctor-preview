"""
LivePortrait — Motion-driven portrait animation for real-time face generation.

Takes a single target photo + webcam driving motion and GENERATES a new face
of the target person with the source person's expressions, head pose, and eye gaze.

Architecture:
  - Appearance Encoder: extracts identity/texture features from target photo (one-time)
  - Motion Extractor: extracts keypoints/expression/pose from each webcam frame
  - Warping + Generator: generates new face from appearance + motion
  - Stitching Module: seamlessly blends generated face into the scene

Based on LivePortrait (Kuaishou) — Apache 2.0 license.
"""
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logger.warning("onnxruntime not available — LivePortrait disabled")

from config import (
    LIVEPORTRAIT_MODEL_DIR,
    LIVEPORTRAIT_RESOLUTION,
    ENABLE_EYE_GAZE_CORRECTION,
    MOTION_SMOOTHING_ALPHA,
    ENABLE_TEMPORAL_SMOOTHING,
)


class LivePortrait:
    """
    Motion-driven face generation using LivePortrait ONNX models.

    Pipeline:
      1. extract_appearance(target_photo) → appearance_features (one-time)
      2. extract_motion(source_frame) → motion_keypoints (per-frame)
      3. generate(appearance, motion) → generated_face (per-frame)
      4. stitch(generated, background) → final_frame (per-frame)

    Usage:
        lp = LivePortrait(providers)
        appearance = lp.extract_appearance(target_photo)
        # Per webcam frame:
        motion = lp.extract_motion(webcam_frame)
        face = lp.generate(appearance, motion)
    """

    def __init__(self, providers: list):
        """Initialize LivePortrait with ONNX model sessions.

        Args:
            providers: ONNX execution providers (e.g., CUDAExecutionProvider)
        """
        self._providers = providers
        self._ready = False

        # ONNX sessions
        self.appearance_encoder: Optional[ort.InferenceSession] = None
        self.motion_extractor: Optional[ort.InferenceSession] = None
        self.generator: Optional[ort.InferenceSession] = None
        self.stitcher: Optional[ort.InferenceSession] = None

        # Cached state
        self._appearance_cache: Dict[str, np.ndarray] = {}
        self._smooth_motion: Dict[str, np.ndarray] = {}
        self._resolution = LIVEPORTRAIT_RESOLUTION

        if ort is None:
            logger.warning("LivePortrait disabled: onnxruntime not available")
            return

        # Model paths
        model_dir = LIVEPORTRAIT_MODEL_DIR
        self._model_paths = {
            "appearance": os.path.join(model_dir, "liveportrait_appearance.onnx"),
            "motion": os.path.join(model_dir, "liveportrait_motion.onnx"),
            "generator": os.path.join(model_dir, "liveportrait_generator.onnx"),
            "stitcher": os.path.join(model_dir, "liveportrait_stitching.onnx"),
        }

        # Check if all models exist
        missing = [
            name
            for name, path in self._model_paths.items()
            if not os.path.exists(path)
        ]
        if missing:
            logger.warning(
                "LivePortrait disabled: missing models: %s",
                ", ".join(missing),
            )
            return

        # Load all models
        self._load_models()

    def _load_models(self):
        """Load all ONNX model sessions."""
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_opts.intra_op_num_threads = 2
            sess_opts.inter_op_num_threads = 1

            t0 = time.time()

            self.appearance_encoder = ort.InferenceSession(
                self._model_paths["appearance"],
                sess_options=sess_opts,
                providers=self._providers,
            )
            logger.info(
                "LivePortrait appearance encoder loaded → %s",
                self.appearance_encoder.get_providers()[0],
            )

            self.motion_extractor = ort.InferenceSession(
                self._model_paths["motion"],
                sess_options=sess_opts,
                providers=self._providers,
            )
            logger.info(
                "LivePortrait motion extractor loaded → %s",
                self.motion_extractor.get_providers()[0],
            )

            self.generator = ort.InferenceSession(
                self._model_paths["generator"],
                sess_options=sess_opts,
                providers=self._providers,
            )
            logger.info(
                "LivePortrait generator loaded → %s",
                self.generator.get_providers()[0],
            )

            self.stitcher = ort.InferenceSession(
                self._model_paths["stitcher"],
                sess_options=sess_opts,
                providers=self._providers,
            )
            logger.info(
                "LivePortrait stitcher loaded → %s",
                self.stitcher.get_providers()[0],
            )

            elapsed = time.time() - t0
            logger.info(
                "LivePortrait all models loaded in %.1fs, resolution=%d",
                elapsed,
                self._resolution,
            )
            self._ready = True

        except Exception as e:
            logger.error("LivePortrait model loading failed: %s", e, exc_info=True)
            self._ready = False

    def is_ready(self) -> bool:
        """Check if all models are loaded and ready."""
        return self._ready

    def _preprocess_face(
        self, image: np.ndarray, target_size: int = 256
    ) -> np.ndarray:
        """Preprocess face image for model input.

        Args:
            image: BGR face crop
            target_size: resize to this size

        Returns:
            Float32 tensor (1, 3, target_size, target_size) normalized to [-1, 1]
        """
        img = cv2.resize(image, (target_size, target_size))
        img = img[:, :, ::-1].astype(np.float32)  # BGR → RGB
        img = img / 127.5 - 1.0  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = img[np.newaxis, ...]  # Add batch dim
        return img

    def extract_appearance(
        self, target_photo: np.ndarray, session_id: str = ""
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract appearance features from target photo (one-time per target).

        Args:
            target_photo: BGR image of the target person's face (cropped)
            session_id: Cache key for this appearance

        Returns:
            Dict with appearance feature tensors, or None on failure
        """
        if not self.is_ready():
            return None

        try:
            t0 = time.time()

            # Preprocess target face
            face_input = self._preprocess_face(
                target_photo, self._resolution
            )

            # Run appearance encoder
            input_name = self.appearance_encoder.get_inputs()[0].name
            outputs = self.appearance_encoder.run(None, {input_name: face_input})

            # Cache appearance features
            appearance = {
                f"feat_{i}": out for i, out in enumerate(outputs)
            }
            appearance["_source_image"] = face_input  # Keep for generator

            if session_id:
                self._appearance_cache[session_id] = appearance

            elapsed = (time.time() - t0) * 1000
            logger.info(
                "LivePortrait appearance extracted in %.1fms (session=%s)",
                elapsed,
                session_id,
            )
            return appearance

        except Exception as e:
            logger.error(
                "LivePortrait appearance extraction failed: %s", e, exc_info=True
            )
            return None

    def extract_motion(
        self, source_frame: np.ndarray, face_bbox: np.ndarray, session_id: str = ""
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract motion (expression, pose, gaze) from a webcam frame.

        Args:
            source_frame: BGR webcam frame
            face_bbox: Face bounding box [x1, y1, x2, y2]
            session_id: For temporal smoothing

        Returns:
            Dict with motion keypoints/parameters, or None on failure
        """
        if not self.is_ready():
            return None

        try:
            # Crop face from frame
            x1, y1, x2, y2 = face_bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            h, w = source_frame.shape[:2]
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = source_frame[y1:y2, x1:x2]
            if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                return None

            # Preprocess
            face_input = self._preprocess_face(face_crop, self._resolution)

            # Run motion extractor
            input_name = self.motion_extractor.get_inputs()[0].name
            outputs = self.motion_extractor.run(None, {input_name: face_input})

            motion = {
                f"motion_{i}": out for i, out in enumerate(outputs)
            }

            # Temporal smoothing on motion keypoints
            if ENABLE_TEMPORAL_SMOOTHING and session_id:
                motion = self._smooth_motion_data(session_id, motion)

            return motion

        except Exception as e:
            logger.error(
                "LivePortrait motion extraction failed: %s", e, exc_info=True
            )
            return None

    def _smooth_motion_data(
        self, session_id: str, motion: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply EMA smoothing to motion keypoints for stability.

        Args:
            session_id: Session key
            motion: Current motion data

        Returns:
            Smoothed motion data
        """
        smooth_key = f"{session_id}_motion"
        if smooth_key in self._smooth_motion:
            prev = self._smooth_motion[smooth_key]
            smoothed = {}
            for key in motion:
                if key in prev and isinstance(motion[key], np.ndarray):
                    if prev[key].shape == motion[key].shape:
                        smoothed[key] = (
                            MOTION_SMOOTHING_ALPHA * prev[key]
                            + (1.0 - MOTION_SMOOTHING_ALPHA) * motion[key]
                        )
                    else:
                        smoothed[key] = motion[key]
                else:
                    smoothed[key] = motion[key]
            self._smooth_motion[smooth_key] = smoothed
            return smoothed
        else:
            self._smooth_motion[smooth_key] = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in motion.items()
            }
            return motion

    def generate(
        self,
        appearance: Dict[str, np.ndarray],
        motion: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Generate a face image from appearance + motion.

        Args:
            appearance: From extract_appearance()
            motion: From extract_motion()

        Returns:
            BGR image (resolution × resolution) or None on failure
        """
        if not self.is_ready():
            return None

        try:
            # Build generator inputs from appearance + motion features
            gen_inputs = {}
            input_specs = self.generator.get_inputs()

            # Map model inputs — the exact mapping depends on the exported model
            # Typically: source_image, appearance_features, motion_keypoints
            all_features = list(appearance.values()) + list(motion.values())
            feature_idx = 0

            for inp in input_specs:
                if inp.name == "source_image" and "_source_image" in appearance:
                    gen_inputs[inp.name] = appearance["_source_image"]
                elif feature_idx < len(all_features):
                    feat = all_features[feature_idx]
                    if isinstance(feat, np.ndarray):
                        gen_inputs[inp.name] = feat
                    feature_idx += 1

            # Run generator
            outputs = self.generator.run(None, gen_inputs)
            generated = outputs[0]  # (1, 3, H, W) in [-1, 1]

            # Postprocess: [-1, 1] → [0, 255] BGR
            generated = generated[0]  # Remove batch dim
            generated = np.transpose(generated, (1, 2, 0))  # CHW → HWC
            generated = ((generated + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            generated = generated[:, :, ::-1]  # RGB → BGR

            return generated

        except Exception as e:
            logger.error(
                "LivePortrait generation failed: %s", e, exc_info=True
            )
            return None

    def stitch(
        self,
        generated_face: np.ndarray,
        background: np.ndarray,
        face_bbox: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Paste generated face onto background with seamless blending.

        Args:
            generated_face: Generated face from generate()
            background: Original frame (background)
            face_bbox: Face bounding box for placement
            mask: Optional parsing mask for precise blending

        Returns:
            Composited frame
        """
        try:
            h, w = background.shape[:2]
            x1, y1, x2, y2 = face_bbox.astype(int)

            # Expand placement region slightly
            face_w = x2 - x1
            face_h = y2 - y1
            pad_x = int(face_w * 0.15)
            pad_y = int(face_h * 0.15)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(w, x2 + pad_x)
            py2 = min(h, y2 + pad_y)

            region_w = px2 - px1
            region_h = py2 - py1

            # Resize generated face to placement region
            face_resized = cv2.resize(
                generated_face, (region_w, region_h),
                interpolation=cv2.INTER_LINEAR,
            )

            # Build blending mask
            if mask is not None and mask.shape[:2] == background.shape[:2]:
                blend_mask = mask[py1:py2, px1:px2]
            else:
                # Fallback: soft elliptical mask
                blend_mask = np.zeros((region_h, region_w), dtype=np.float32)
                cv2.ellipse(
                    blend_mask,
                    (region_w // 2, region_h // 2),
                    (int(region_w * 0.45), int(region_h * 0.48)),
                    0, 0, 360, 1.0, -1,
                )
                blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 8.0)

            # Alpha blend
            alpha = blend_mask[..., np.newaxis]
            result = background.copy()
            roi = result[py1:py2, px1:px2].astype(np.float32)
            blended = roi * (1.0 - alpha) + face_resized.astype(np.float32) * alpha
            result[py1:py2, px1:px2] = blended.astype(np.uint8)

            return result

        except Exception as e:
            logger.error("LivePortrait stitch failed: %s", e, exc_info=True)
            return background

    def get_cached_appearance(self, session_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Retrieve cached appearance features for a session."""
        return self._appearance_cache.get(session_id)

    def cleanup_session(self, session_id: str):
        """Free cached data for a session."""
        self._appearance_cache.pop(session_id, None)

        keys_to_remove = [
            k for k in self._smooth_motion if k.startswith(session_id)
        ]
        for k in keys_to_remove:
            del self._smooth_motion[k]

        logger.debug("LivePortrait cleaned up session %s", session_id)
