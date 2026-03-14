"""
Analysis Pipeline — Orchestrates all analyzers in sequence.
Single Responsibility: coordinates the flow, does not perform analysis itself.
"""

import base64
import io
import time
import uuid
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from ..analyzers.face_detector import FaceDetector
from ..analyzers.measurements import MeasurementAnalyzer
from ..analyzers.pigmentation import PigmentationAnalyzer
from ..analyzers.pore import PoreAnalyzer
from ..analyzers.redness import RednessAnalyzer
from ..analyzers.symmetry import SymmetryAnalyzer
from ..analyzers.texture import TextureAnalyzer
from ..analyzers.wrinkle import WrinkleAnalyzer
from ..config import get_config
from .depth_enhancer import DepthEnhancer


class AnalysisPipeline:
    """
    Orchestrates the full face analysis pipeline.
    Follows Dependency Inversion: depends on BaseAnalyzer abstractions.
    """

    def __init__(self):
        config = get_config()
        device = config.gpu.device

        # Initialize all analyzers (lazy model loading)
        self._face_detector = FaceDetector(device=device)
        self._analyzers = {
            "wrinkle": WrinkleAnalyzer(device=device),
            "pore": PoreAnalyzer(device=device),
            "pigmentation": PigmentationAnalyzer(device=device),
            "redness": RednessAnalyzer(device=device),
            "texture": TextureAnalyzer(device=device),
            "symmetry": SymmetryAnalyzer(device=device),
            "measurements": MeasurementAnalyzer(device=device),
        }
        self._depth_enhancer = DepthEnhancer(
            sr_scale=config.analysis.depth_sr_scale,
            n_frames=config.analysis.multi_frame_count,
        )
        self._config = config
        self._loaded = False

    def load_models(self) -> List[str]:
        """Load all ML models. Returns list of loaded model names."""
        loaded = []
        self._face_detector.load_model()
        loaded.append(self._face_detector.name)

        for name, analyzer in self._analyzers.items():
            analyzer.load_model()
            loaded.append(analyzer.name)

        self._loaded = True
        return loaded

    def analyze_single(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        depth_frames: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Run full analysis on a single face image.

        Args:
            image: BGR image (H, W, 3) uint8
            depth_map: Optional depth map from LiDAR
            depth_frames: Optional multiple depth frames for averaging

        Returns:
            Complete analysis results dict
        """
        if not self._loaded:
            self.load_models()

        start_time = time.time()
        session_id = str(uuid.uuid4())

        # Resize to target resolution
        target_res = self._config.analysis.target_resolution
        image = self._resize_preserve_aspect(image, target_res)

        # Enhance depth if multiple frames provided (Technique 3)
        if depth_frames and len(depth_frames) > 1:
            depth_map = self._depth_enhancer.average_depth_frames(depth_frames, image)

        # Super-resolve depth if provided (Technique 4)
        if depth_map is not None:
            depth_map = self._depth_enhancer.super_resolve_depth(
                depth_map, image, scale=self._config.analysis.depth_sr_scale
            )

        # Step 1: Face detection & landmarks
        detection = self._face_detector.analyze(image)
        if detection.get("error"):
            return {
                "session_id": session_id,
                "error": detection["error"],
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        landmarks = detection["landmarks_468"]
        skin_mask = detection["skin_mask"]
        zone_masks = detection["zone_masks"]

        # Step 2: Align face for consistent analysis
        aligned_face, inv_matrix = self._face_detector.get_aligned_face(
            image, landmarks, target_size=512
        )

        # Step 3: Run all analyzers
        results = {
            "session_id": session_id,
            "zone_scores": {},
            "conditions": [],
            "heatmaps": {},
        }

        # Per-analyzer results
        analyzer_results = {}
        for name, analyzer in self._analyzers.items():
            try:
                result = analyzer.analyze(
                    face_image=image,
                    landmarks=landmarks,
                    mask=skin_mask,
                )
                analyzer_results[name] = result

                # Collect heatmap
                if result.get("heatmap") is not None:
                    results["heatmaps"][name] = self._heatmap_to_base64(
                        result["heatmap"], image
                    )

                # Collect detections
                results["conditions"].extend(result.get("detections", []))

            except Exception as e:
                analyzer_results[name] = {"score": 50, "error": str(e)}

        # Step 4: Compute per-zone scores
        for zone_name, zone_mask in zone_masks.items():
            zone_score = self._compute_zone_score(
                image, landmarks, zone_mask, zone_name
            )
            results["zone_scores"][zone_name] = zone_score

        # Step 5: Compute overall score
        analyzer_scores = {
            name: r.get("score", 50) for name, r in analyzer_results.items()
        }
        weights = {
            "wrinkle": 0.20, "pore": 0.15, "pigmentation": 0.20,
            "redness": 0.15, "texture": 0.15, "symmetry": 0.10,
            "measurements": 0.05,
        }
        overall = sum(
            analyzer_scores.get(k, 50) * w for k, w in weights.items()
        )
        results["overall_score"] = round(overall, 1)

        # Step 6: Estimate skin age
        results["skin_age"] = self._estimate_skin_age(analyzer_results)

        # Step 7: Generate recommendations
        results["recommendations"] = self._generate_recommendations(
            analyzer_results, results["conditions"]
        )

        # Step 8: Copy specific analyzer data
        results["symmetry"] = {
            "overall_score": analyzer_results.get("symmetry", {}).get("score", 50),
            "eye_alignment": analyzer_results.get("symmetry", {}).get("eye_alignment", 50),
            "cheek_balance": analyzer_results.get("symmetry", {}).get("cheek_balance", 50),
            "jawline_symmetry": analyzer_results.get("symmetry", {}).get("jawline_symmetry", 50),
            "lip_symmetry": analyzer_results.get("symmetry", {}).get("lip_symmetry", 50),
            "midline_deviation_mm": analyzer_results.get("symmetry", {}).get("midline_deviation_mm", 0),
        }
        results["measurements"] = analyzer_results.get("measurements", {}).get("measurements", {})
        results["analyzer_scores"] = analyzer_scores

        # Annotated image with all overlays
        results["annotated_image"] = self._generate_annotated_image(
            image, landmarks, skin_mask, analyzer_results
        )

        # Original image as base64 for frontend visual overlay
        _, orig_buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results["original_image"] = base64.b64encode(orig_buf).decode('utf-8')

        # Normalized landmarks for frontend rendering (0-1 range)
        h, w = image.shape[:2]
        results["landmarks"] = [
            {"x": round(float(pt[0]) / w, 4), "y": round(float(pt[1]) / h, 4)}
            for pt in landmarks
        ]

        # 3D landmarks for Three.js face model (x, y, z normalized)
        landmarks_3d = detection.get("landmarks_3d")
        if landmarks_3d is not None:
            results["landmarks_3d"] = [
                {"x": round(float(pt[0]), 5), "y": round(float(pt[1]), 5), "z": round(float(pt[2]), 5)}
                for pt in landmarks_3d
            ]

        results["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)
        results["resolution"] = f"{image.shape[1]}x{image.shape[0]}"

        return results

    def analyze_multi(
        self,
        images: Dict[str, np.ndarray],
        depth_maps: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze multiple angles and merge results.

        Args:
            images: Dict of angle_name -> BGR image
            depth_maps: Optional dict of angle_name -> depth map
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())

        angle_results = {}
        for angle, image in images.items():
            dm = depth_maps.get(angle) if depth_maps else None
            result = self.analyze_single(image, depth_map=dm)
            result.pop("session_id", None)  # Will use merged session ID
            angle_results[angle] = result

        # Merge results: average scores, union detections
        merged = self._merge_angle_results(angle_results)
        merged["session_id"] = session_id
        merged["angles_analyzed"] = list(images.keys())
        merged["processing_time_ms"] = round((time.time() - start_time) * 1000, 1)

        return merged

    def compare_sessions(
        self, session_before: Dict, session_after: Dict
    ) -> Dict[str, Any]:
        """Compare two analysis sessions for before/after reporting."""
        improvements = []
        regressions = []

        # Compare zone scores
        for zone in session_after.get("zone_scores", {}):
            if zone in session_before.get("zone_scores", {}):
                before = session_before["zone_scores"][zone]
                after = session_after["zone_scores"][zone]
                for metric in ["wrinkles", "pores", "pigmentation", "redness", "texture", "overall"]:
                    b_val = before.get(metric, 50)
                    a_val = after.get(metric, 50)
                    delta = a_val - b_val
                    entry = {
                        "zone": zone, "metric": metric,
                        "before": b_val, "after": a_val, "delta": round(delta, 1)
                    }
                    if delta > 1:
                        improvements.append(entry)
                    elif delta < -1:
                        regressions.append(entry)

        overall_delta = (
            session_after.get("overall_score", 50)
            - session_before.get("overall_score", 50)
        )

        return {
            "session_before": session_before,
            "session_after": session_after,
            "improvements": sorted(improvements, key=lambda x: -x["delta"]),
            "regressions": sorted(regressions, key=lambda x: x["delta"]),
            "overall_delta": round(overall_delta, 1),
        }

    # ── Private Helpers ─────────────────────────────────────────────────────

    def _compute_zone_score(
        self, image: np.ndarray, landmarks: np.ndarray,
        zone_mask: np.ndarray, zone_name: str
    ) -> Dict[str, float]:
        """Run key analyzers for a specific zone and return scores."""
        scores = {}
        for key in ["wrinkle", "pore", "pigmentation", "redness", "texture"]:
            try:
                result = self._analyzers[key].analyze(
                    image, landmarks=landmarks, mask=zone_mask, zone=zone_name
                )
                scores[key] = result.get("score", 50)
            except Exception:
                scores[key] = 50.0

        # Weighted overall for zone
        weights = {"wrinkle": 0.25, "pore": 0.20, "pigmentation": 0.20,
                    "redness": 0.15, "texture": 0.20}
        scores["overall"] = round(
            sum(scores.get(k, 50) * w for k, w in weights.items()), 1
        )
        return scores

    def _estimate_skin_age(self, analyzer_results: Dict) -> int:
        """Estimate biological skin age from analysis scores."""
        wrinkle_score = analyzer_results.get("wrinkle", {}).get("score", 50)
        texture_score = analyzer_results.get("texture", {}).get("score", 50)
        pigment_score = analyzer_results.get("pigmentation", {}).get("score", 50)

        # Simple linear model: lower scores = older skin
        avg_score = (wrinkle_score * 0.5 + texture_score * 0.3 + pigment_score * 0.2)
        estimated_age = int(80 - avg_score * 0.5)  # Score 100 → age 30, Score 0 → age 80
        return max(18, min(90, estimated_age))

    def _generate_recommendations(
        self, analyzer_results: Dict, conditions: List
    ) -> List[Dict]:
        """Generate prioritized treatment recommendations."""
        recommendations = []
        priority = 1

        # Sort conditions by severity
        severity_order = {"severe": 0, "moderate": 1, "mild": 2}
        sorted_conditions = sorted(
            conditions,
            key=lambda c: severity_order.get(c.get("severity", "mild"), 3),
        )

        recommendation_map = {
            "deep_wrinkles": ("Wrinkle Treatment", "Consider retinoid therapy or micro-needling for deep wrinkle reduction"),
            "fine_lines": ("Fine Line Prevention", "Hyaluronic acid serum and daily SPF 50+ recommended"),
            "enlarged_pores": ("Pore Refinement", "Salicylic acid cleanser and weekly clay mask recommended"),
            "brown_spots": ("Pigmentation Treatment", "Chemical peel (glycolic acid) or IPL therapy for spot reduction"),
            "uv_damage": ("UV Damage Repair", "Vitamin C serum daily + strict SPF 50+ sun protection"),
            "redness": ("Redness Management", "Azelaic acid cream and gentle cleanser for inflammation reduction"),
            "possible_rosacea": ("Rosacea Assessment", "Clinical rosacea evaluation recommended — avoid triggers"),
            "rough_texture": ("Texture Improvement", "AHA/BHA exfoliation routine with moisturizer"),
            "asymmetry": ("Symmetry Note", "Minor asymmetry is normal; severe cases may benefit from targeted treatment"),
        }

        seen_types = set()
        for condition in sorted_conditions:
            ctype = condition.get("type", "")
            if ctype in seen_types:
                continue
            seen_types.add(ctype)

            if ctype in recommendation_map:
                area, suggestion = recommendation_map[ctype]
                recommendations.append({
                    "priority": priority,
                    "area": area,
                    "condition": ctype,
                    "severity": condition.get("severity", "mild"),
                    "suggestion": suggestion,
                    "zones": condition.get("zone", "full_face"),
                })
                priority += 1

        return recommendations[:10]  # Top 10

    def _merge_angle_results(self, angle_results: Dict) -> Dict:
        """Merge results from multiple angles into one consolidated result."""
        if not angle_results:
            return {}

        # Use front as primary, supplement with side angles
        primary = angle_results.get("front_0", list(angle_results.values())[0])
        merged = primary.copy()

        # Average scores across angles
        all_scores = [r.get("overall_score", 50) for r in angle_results.values()]
        merged["overall_score"] = round(np.mean(all_scores), 1)

        # Merge conditions (union, deduplicate by type+zone)
        all_conditions = []
        seen = set()
        for result in angle_results.values():
            for c in result.get("conditions", []):
                key = f"{c.get('type')}_{c.get('zone')}"
                if key not in seen:
                    seen.add(key)
                    all_conditions.append(c)
        merged["conditions"] = all_conditions

        return merged

    @staticmethod
    def _resize_preserve_aspect(image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image so the larger dimension equals target_size."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        if scale >= 1.0:
            return image
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _heatmap_to_base64(heatmap: np.ndarray, original: np.ndarray) -> str:
        """Convert heatmap to colored overlay and encode as base64 PNG."""
        h, w = original.shape[:2]
        hm = cv2.resize(heatmap, (w, h))

        # Normalize to 0-255
        hm_uint8 = (np.clip(hm, 0, 1) * 255).astype(np.uint8)

        # Apply colormap (COLORMAP_JET: blue=low, red=high)
        colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)

        # Blend with original
        alpha = 0.4
        blended = cv2.addWeighted(original, 1 - alpha, colored, alpha, 0)

        # Encode to PNG base64
        _, buffer = cv2.imencode(".png", blended)
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _generate_annotated_image(
        image: np.ndarray, landmarks: np.ndarray,
        skin_mask: np.ndarray, results: Dict
    ) -> str:
        """Generate annotated image with landmarks, zones, and scores."""
        annotated = image.copy()

        # Draw landmarks (small dots)
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(annotated, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Draw face outline
        face_outline_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                            323, 361, 288, 397, 365, 379, 378, 400, 377,
                            152, 148, 176, 149, 150, 136, 172, 58, 132,
                            93, 234, 127, 162, 21, 54, 103, 67, 109]
        pts = landmarks[face_outline_idx].astype(np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)

        # Encode
        _, buffer = cv2.imencode(".png", annotated)
        return base64.b64encode(buffer).decode("utf-8")
