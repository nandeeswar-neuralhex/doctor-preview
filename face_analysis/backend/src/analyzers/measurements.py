"""
Facial Measurements — Precise proportional analysis.
Computes distances, angles, ratios, and golden-ratio compliance.
"""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import BaseAnalyzer


class MeasurementAnalyzer(BaseAnalyzer):
    """
    Computes clinical facial measurements from 468 MediaPipe landmarks.
    All distances reported in mm (estimated using interpupillary distance calibration).
    """

    # Average interpupillary distance for calibration (mm)
    AVG_IPD_MM = 63.0

    # Key landmark indices for measurements
    LM = {
        "left_eye_outer": 33,
        "left_eye_inner": 133,
        "right_eye_outer": 263,
        "right_eye_inner": 362,
        "left_eyebrow_outer": 46,
        "right_eyebrow_outer": 276,
        "nose_tip": 1,
        "nose_bridge": 6,
        "nose_left": 48,
        "nose_right": 278,
        "nasion": 168,
        "upper_lip_top": 0,
        "lower_lip_bottom": 17,
        "lip_left": 61,
        "lip_right": 291,
        "chin": 152,
        "forehead_top": 10,
        "left_jaw": 132,
        "right_jaw": 361,
        "left_cheek": 234,
        "right_cheek": 454,
        "left_ear": 234,
        "right_ear": 454,
    }

    def __init__(self, device: str = "cpu"):
        super().__init__(name="measurement_analyzer", device=device)

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

        if landmarks is None or len(landmarks) < 468:
            return {
                "score": 0,
                "detections": [],
                "heatmap": None,
                "error": "468 landmarks required",
            }

        # Calibrate pixel-to-mm using interpupillary distance
        px_to_mm = self._calibrate(landmarks)

        # Compute all measurements
        measurements = {}

        # Interpupillary distance
        ipd_px = self._dist(landmarks, "left_eye_outer", "right_eye_outer")
        measurements["interpupillary_distance"] = {
            "value": round(ipd_px * px_to_mm, 1),
            "unit": "mm",
            "reference": "60-66mm",
        }

        # Nasal width
        nasal_w = self._dist(landmarks, "nose_left", "nose_right")
        measurements["nasal_width"] = {
            "value": round(nasal_w * px_to_mm, 1),
            "unit": "mm",
            "reference": "31-36mm",
        }

        # Nasal length
        nasal_l = self._dist(landmarks, "nasion", "nose_tip")
        measurements["nasal_length"] = {
            "value": round(nasal_l * px_to_mm, 1),
            "unit": "mm",
            "reference": "45-55mm",
        }

        # Lip width
        lip_w = self._dist(landmarks, "lip_left", "lip_right")
        measurements["lip_width"] = {
            "value": round(lip_w * px_to_mm, 1),
            "unit": "mm",
            "reference": "48-56mm",
        }

        # Lip height (upper lip to lower lip)
        lip_h = self._dist(landmarks, "upper_lip_top", "lower_lip_bottom")
        measurements["lip_height"] = {
            "value": round(lip_h * px_to_mm, 1),
            "unit": "mm",
            "reference": "18-24mm",
        }

        # Face width (bizygomatic)
        face_w = self._dist(landmarks, "left_cheek", "right_cheek")
        measurements["face_width"] = {
            "value": round(face_w * px_to_mm, 1),
            "unit": "mm",
            "reference": "130-150mm",
        }

        # Face height (trichion to menton)
        face_h = self._dist(landmarks, "forehead_top", "chin")
        measurements["face_height"] = {
            "value": round(face_h * px_to_mm, 1),
            "unit": "mm",
            "reference": "170-200mm",
        }

        # Jawline angles
        jaw_angle_l = self._angle_at(landmarks, "left_ear", "left_jaw", "chin")
        jaw_angle_r = self._angle_at(landmarks, "right_ear", "right_jaw", "chin")
        measurements["jawline_angle_left"] = {
            "value": round(jaw_angle_l, 1),
            "unit": "degrees",
            "reference": "120-135°",
        }
        measurements["jawline_angle_right"] = {
            "value": round(jaw_angle_r, 1),
            "unit": "degrees",
            "reference": "120-135°",
        }

        # Golden ratio
        golden = self._golden_ratio(landmarks, px_to_mm)
        measurements["golden_ratio"] = {
            "value": round(golden, 3),
            "unit": "ratio",
            "reference": "1.618 (ideal)",
        }

        # Facial thirds
        thirds = self._facial_thirds(landmarks)
        measurements["facial_thirds"] = thirds

        # Nasofrontal angle
        nf_angle = self._nasofrontal_angle(landmarks)
        measurements["nasofrontal_angle"] = {
            "value": round(nf_angle, 1),
            "unit": "degrees",
            "reference": "115-135°",
        }

        # Nasolabial angle
        nl_angle = self._nasolabial_angle(landmarks)
        measurements["nasolabial_angle"] = {
            "value": round(nl_angle, 1),
            "unit": "degrees",
            "reference": "90-110°",
        }

        # Proportional score (how close to ideal proportions)
        score = self._proportional_score(measurements)

        # Generate measurement overlay
        heatmap = self._generate_overlay(face_image, landmarks, measurements, px_to_mm)

        return {
            "score": round(score, 1),
            "detections": [],
            "heatmap": heatmap,
            "measurements": measurements,
            "px_to_mm": round(px_to_mm, 4),
        }

    def _calibrate(self, landmarks: np.ndarray) -> float:
        """Calibrate pixel-to-mm ratio using interpupillary distance."""
        left_eye = landmarks[self.LM["left_eye_outer"]]
        right_eye = landmarks[self.LM["right_eye_outer"]]
        ipd_px = np.sqrt(np.sum((left_eye - right_eye) ** 2))
        return self.AVG_IPD_MM / max(ipd_px, 1.0)

    def _dist(self, landmarks: np.ndarray, name1: str, name2: str) -> float:
        """Euclidean distance between two named landmarks."""
        p1 = landmarks[self.LM[name1]]
        p2 = landmarks[self.LM[name2]]
        return float(np.sqrt(np.sum((p1 - p2) ** 2)))

    def _angle_at(self, landmarks: np.ndarray, a: str, vertex: str, b: str) -> float:
        """Angle at vertex point between rays to a and b."""
        pa = landmarks[self.LM[a]]
        pv = landmarks[self.LM[vertex]]
        pb = landmarks[self.LM[b]]

        va = pa - pv
        vb = pb - pv
        cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-7)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    def _golden_ratio(self, landmarks: np.ndarray, px_to_mm: float) -> float:
        """Compute facial golden ratio approximation."""
        face_h = self._dist(landmarks, "forehead_top", "chin")
        face_w = self._dist(landmarks, "left_cheek", "right_cheek")
        if face_w > 0:
            return face_h / face_w
        return 0

    def _facial_thirds(self, landmarks: np.ndarray) -> Dict:
        """Compute facial thirds (upper, middle, lower)."""
        top = landmarks[self.LM["forehead_top"]][1]
        nasion = landmarks[self.LM["nasion"]][1]
        nose_base = landmarks[self.LM["nose_tip"]][1]
        chin = landmarks[self.LM["chin"]][1]

        total = chin - top
        if total <= 0:
            return {"upper": 33.3, "middle": 33.3, "lower": 33.3}

        upper = (nasion - top) / total * 100
        middle = (nose_base - nasion) / total * 100
        lower = (chin - nose_base) / total * 100

        return {
            "upper": round(upper, 1),
            "middle": round(middle, 1),
            "lower": round(lower, 1),
        }

    def _nasofrontal_angle(self, landmarks: np.ndarray) -> float:
        """Angle between forehead slope and nose bridge."""
        forehead = landmarks[self.LM["forehead_top"]]
        nasion = landmarks[self.LM["nasion"]]
        nose_bridge = landmarks[self.LM["nose_bridge"]]

        v1 = forehead - nasion
        v2 = nose_bridge - nasion
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    def _nasolabial_angle(self, landmarks: np.ndarray) -> float:
        """Angle between nose base and upper lip."""
        nose_tip = landmarks[self.LM["nose_tip"]]
        nose_base = landmarks[self.LM["nose_bridge"]]
        upper_lip = landmarks[self.LM["upper_lip_top"]]

        v1 = nose_base - nose_tip
        v2 = upper_lip - nose_tip
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    def _proportional_score(self, measurements: Dict) -> float:
        """Score how close measurements are to ideal proportions."""
        scores = []

        # Golden ratio (ideal = 1.618)
        gr = measurements.get("golden_ratio", {}).get("value", 0)
        if gr > 0:
            gr_score = max(0, 100 - abs(gr - 1.618) / 1.618 * 200)
            scores.append(gr_score)

        # Facial thirds balance (ideal = 33.3% each)
        thirds = measurements.get("facial_thirds", {})
        if thirds:
            deviation = sum(abs(v - 33.3) for v in [
                thirds.get("upper", 33.3),
                thirds.get("middle", 33.3),
                thirds.get("lower", 33.3),
            ])
            thirds_score = max(0, 100 - deviation * 3)
            scores.append(thirds_score)

        # Jawline symmetry
        jl = measurements.get("jawline_angle_left", {}).get("value", 0)
        jr = measurements.get("jawline_angle_right", {}).get("value", 0)
        if jl > 0 and jr > 0:
            jaw_sym = max(0, 100 - abs(jl - jr) * 5)
            scores.append(jaw_sym)

        return np.mean(scores) if scores else 50.0

    def _generate_overlay(
        self, image: np.ndarray, landmarks: np.ndarray,
        measurements: Dict, px_to_mm: float
    ) -> np.ndarray:
        """Generate measurement lines overlay image."""
        h, w = image.shape[:2]
        overlay = np.zeros((h, w), dtype=np.float32)

        # Draw measurement lines as a heatmap
        pairs_to_draw = [
            ("left_eye_outer", "right_eye_outer"),
            ("nose_left", "nose_right"),
            ("lip_left", "lip_right"),
            ("forehead_top", "chin"),
            ("left_cheek", "right_cheek"),
        ]

        for p1_name, p2_name in pairs_to_draw:
            p1 = landmarks[self.LM[p1_name]].astype(int)
            p2 = landmarks[self.LM[p2_name]].astype(int)
            cv2.line(overlay, tuple(p1), tuple(p2), 0.8, 2)

        # Draw landmarks
        for name, idx in self.LM.items():
            pt = landmarks[idx].astype(int)
            cv2.circle(overlay, tuple(pt), 3, 1.0, -1)

        return overlay
