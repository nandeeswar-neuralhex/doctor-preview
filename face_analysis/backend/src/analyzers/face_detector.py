"""
Face Detector & Landmark Extractor
Uses MediaPipe Face Mesh for 468-point landmarks + face parsing for skin mask.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseAnalyzer


class FaceDetector(BaseAnalyzer):
    """
    Detects faces, extracts 468 landmarks, generates skin segmentation mask,
    and computes per-zone masks for downstream analyzers.
    """

    # MediaPipe Face Mesh zone landmark indices (approximate regions)
    ZONE_LANDMARKS = {
        "forehead": list(range(54, 104)) + [10, 338, 297, 332, 284, 251, 389],
        "left_cheek": [36, 50, 187, 123, 116, 117, 118, 119, 120, 121, 47, 126],
        "right_cheek": [266, 280, 411, 352, 345, 346, 347, 348, 349, 350, 277, 355],
        "nose": [1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 45, 275, 440, 220],
        "chin": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127],
        "under_eye_left": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        "under_eye_right": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        "jawline_left": [132, 58, 172, 136, 150, 149, 176, 148, 152],
        "jawline_right": [361, 288, 397, 365, 379, 378, 400, 377, 152],
        "left_temple": [54, 103, 67, 109, 10],
        "right_temple": [284, 332, 297, 338, 10],
        "neck": [152, 148, 176, 149, 150, 136, 172, 58],
    }

    def __init__(self, device: str = "cpu"):
        super().__init__(name="face_detector", device=device)
        self._face_mesh = None
        self._face_detection = None

    def load_model(self) -> None:
        """Initialize MediaPipe Face Mesh."""
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
        self._is_loaded = True

    def analyze(
        self,
        face_image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Detect face and extract landmarks.

        Returns:
            - landmarks_468: (468, 2) array of normalized landmark coords
            - face_bbox: (x, y, w, h)
            - zone_masks: dict of zone_name -> binary mask
            - skin_mask: binary mask of skin region
            - face_crop: aligned face crop
            - confidence: detection confidence
        """
        if not self._is_loaded:
            self.load_model()

        h, w = face_image.shape[:2]
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {
                "score": 0,
                "detections": [],
                "heatmap": None,
                "landmarks_468": None,
                "face_bbox": None,
                "zone_masks": {},
                "skin_mask": None,
                "confidence": 0.0,
                "error": "No face detected",
            }

        face_lm = results.multi_face_landmarks[0]

        # Convert to pixel coordinates
        landmarks_px = np.array([
            [lm.x * w, lm.y * h] for lm in face_lm.landmark
        ], dtype=np.float32)

        # 3D normalized landmarks (x, y in 0-1 range, z = depth)
        landmarks_3d = np.array([
            [lm.x, lm.y, lm.z] for lm in face_lm.landmark
        ], dtype=np.float32)

        # Bounding box
        x_min, y_min = landmarks_px.min(axis=0).astype(int)
        x_max, y_max = landmarks_px.max(axis=0).astype(int)
        padding = int(0.15 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Skin mask from convex hull of face outline landmarks
        face_outline_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                            323, 361, 288, 397, 365, 379, 378, 400, 377,
                            152, 148, 176, 149, 150, 136, 172, 58, 132,
                            93, 234, 127, 162, 21, 54, 103, 67, 109]
        outline_pts = landmarks_px[face_outline_idx].astype(np.int32)
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(skin_mask, outline_pts, 255)

        # Generate per-zone masks
        zone_masks = self._compute_zone_masks(landmarks_px, h, w)

        return {
            "score": 100,  # Detection score
            "detections": [],
            "heatmap": None,
            "landmarks_468": landmarks_px,
            "landmarks_3d": landmarks_3d,
            "face_bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "zone_masks": zone_masks,
            "skin_mask": skin_mask,
            "confidence": 0.95,
        }

    def _compute_zone_masks(
        self, landmarks: np.ndarray, h: int, w: int
    ) -> Dict[str, np.ndarray]:
        """Generate binary masks for each facial zone from landmarks."""
        zone_masks = {}
        for zone_name, indices in self.ZONE_LANDMARKS.items():
            safe_indices = [i for i in indices if i < len(landmarks)]
            if len(safe_indices) < 3:
                continue
            pts = landmarks[safe_indices].astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
            zone_masks[zone_name] = mask
        return zone_masks

    def get_aligned_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        target_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face using eye centers for rotation-invariant analysis.

        Returns:
            (aligned_face, inverse_transform_matrix)
        """
        left_eye = landmarks[[33, 133, 160, 158, 153, 144]].mean(axis=0)
        right_eye = landmarks[[362, 263, 387, 385, 380, 373]].mean(axis=0)

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))

        eye_center = ((left_eye + right_eye) / 2).astype(np.float32)
        desired_eye_dist = target_size * 0.35
        current_eye_dist = np.sqrt(dx**2 + dy**2)
        scale = desired_eye_dist / max(current_eye_dist, 1.0)

        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
        M[0, 2] += target_size / 2 - eye_center[0]
        M[1, 2] += target_size * 0.38 - eye_center[1]

        aligned = cv2.warpAffine(image, M, (target_size, target_size))
        M_inv = cv2.invertAffineTransform(M)

        return aligned, M_inv

    def unload(self) -> None:
        if self._face_mesh:
            self._face_mesh.close()
        if self._face_detection:
            self._face_detection.close()
        super().unload()
