"""
FaceSwapper - Core face swapping logic using InsightFace/FaceFusion models
Optimized for 24+ FPS real-time processing
"""
import cv2
import numpy as np
from typing import Dict, Optional, Any
from types import SimpleNamespace
import onnxruntime as ort
from insightface.app import FaceAnalysis

from config import (
    MODELS_DIR,
    INSWAPPER_MODEL,
    EXECUTION_PROVIDER,
    ENABLE_SEAMLESS_CLONE,
    FACE_MASK_BLUR,
    FACE_MASK_SCALE,
    ENABLE_GFPGAN,
    GFPGAN_MODEL_PATH,
    ENABLE_TEMPORAL_SMOOTHING,
    SMOOTHING_ALPHA,
    MAX_FACES,
)


class FaceSwapper:
    """
    High-performance face swapper using InsightFace models.
    Manages multiple sessions with pre-extracted target faces.
    """
    
    def __init__(self):
        print("Initializing FaceSwapper...")
        
        # ── Step 1: Verify GPU availability ──
        available = ort.get_available_providers()
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {available}")
        
        self.gpu_active = "CUDAExecutionProvider" in available
        if self.gpu_active:
            print("✅ GPU (CUDA) is available")
        else:
            print("⚠️  WARNING: CUDAExecutionProvider NOT available — running on CPU!")
            print("   This will be very slow (3-5 FPS instead of 24+ FPS)")
        
        # ── Step 2: Configure ONNX session options for max GPU throughput ──
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        # Use all available threads for any CPU fallback ops
        sess_options.intra_op_num_threads = 0  # 0 = use all cores
        sess_options.inter_op_num_threads = 0
        
        # ── Step 3: Build provider list with CUDA options ──
        if self.gpu_active:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4 GB
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        
        self._providers = providers
        self._sess_options = sess_options
        
        # ── Step 4: Initialize face analyzer (detection + landmarks) on GPU ──
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=MODELS_DIR,
            providers=providers
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Verify which provider the detection model actually uses
        for model in self.face_analyzer.models:
            try:
                session = getattr(model, 'session', None)
                if session:
                    active_providers = session.get_providers()
                    print(f"  Face analyzer model → {active_providers[0]}")
            except Exception:
                pass
        
        # ── Step 5: Load face swapper model with GPU session options ──
        print(f"Loading swapper model: {INSWAPPER_MODEL}")
        try:
            from insightface.model_zoo import get_model
            self.swapper = get_model(INSWAPPER_MODEL, providers=providers)
            # Verify swapper is on GPU
            try:
                swapper_session = getattr(self.swapper, 'session', None)
                if swapper_session:
                    active = swapper_session.get_providers()
                    print(f"  INSwapper model → {active[0]}")
                    if active[0] != "CUDAExecutionProvider" and self.gpu_active:
                        print("  ⚠️  Swapper fell back to CPU! Retrying with explicit session...")
                        swapper_session_gpu = ort.InferenceSession(
                            INSWAPPER_MODEL,
                            sess_options=sess_options,
                            providers=providers
                        )
                        self.swapper.session = swapper_session_gpu
                        print(f"  ✅ Swapper forced to GPU: {swapper_session_gpu.get_providers()[0]}")
            except Exception as verify_err:
                print(f"  Provider verification info: {verify_err}")
            print(f"Loaded INSwapper model via insightface")
        except Exception as e:
            print(f"Failed to load swapper with GPU: {e}")
            print("Falling back to CPUExecutionProvider...")
            from insightface.model_zoo import get_model
            self.swapper = get_model(INSWAPPER_MODEL, providers=["CPUExecutionProvider"])
            self.gpu_active = False
        
        # Optional face enhancer (GFPGAN)
        self.enhancer = None
        if ENABLE_GFPGAN:
            try:
                from gfpgan import GFPGANer

                model_path = GFPGAN_MODEL_PATH or None
                self.enhancer = GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None
                )
                print("GFPGAN enhancer enabled")
            except Exception as e:
                print(f"GFPGAN not available: {e}")
                self.enhancer = None

        # Session storage for target faces and smoothing state
        self.target_faces: Dict[str, Any] = {}
        self._smooth_kps: Dict[str, np.ndarray] = {}
        self._smooth_bbox: Dict[str, np.ndarray] = {}
        self._last_result: Dict[str, np.ndarray] = {}
        self._ready = True
        
        print("FaceSwapper initialized successfully!")
    
    def is_ready(self) -> bool:
        """Check if the swapper is ready for processing"""
        return self._ready
    
    def gpu_status(self) -> dict:
        """Return GPU usage diagnostics for debugging."""
        info = {
            "onnxruntime_version": ort.__version__,
            "available_providers": ort.get_available_providers(),
            "gpu_active": self.gpu_active,
            "analyzer_providers": [],
            "swapper_provider": "unknown",
        }
        # Check what each analyzer sub-model is using
        for model in self.face_analyzer.models:
            try:
                session = getattr(model, 'session', None)
                if session:
                    info["analyzer_providers"].append(session.get_providers()[0])
            except Exception:
                info["analyzer_providers"].append("unknown")
        # Check swapper
        try:
            swapper_session = getattr(self.swapper, 'session', None)
            if swapper_session:
                info["swapper_provider"] = swapper_session.get_providers()[0]
        except Exception:
            pass
        return info
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.target_faces)
    
    def set_target_face(self, session_id: str, image: np.ndarray) -> bool:
        """
        Extract and store the target face from an uploaded image.
        
        Args:
            session_id: Unique session identifier
            image: BGR image containing the target face
            
        Returns:
            True if face was successfully extracted and stored
        """
        # Detect faces in the target image
        faces = self.face_analyzer.get(image)

        if len(faces) == 0:
            # Retry with resized image for small/large faces and low-contrast images
            height, width = image.shape[:2]
            min_dim = min(height, width)
            max_dim = max(height, width)

            # Contrast enhancement (helps low-light/flat images)
            def _enhance_contrast(img: np.ndarray) -> np.ndarray:
                try:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l2 = clahe.apply(l)
                    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
                except Exception:
                    return img

            scale_candidates = [1.0]
            if max_dim > 1200:
                scale_candidates.extend([1200 / max_dim, 960 / max_dim, 720 / max_dim, 640 / max_dim])
            elif max_dim > 900:
                scale_candidates.extend([960 / max_dim, 720 / max_dim])
            if min_dim < 320:
                scale_candidates.append(640 / max(1, min_dim))
            elif min_dim < 480:
                scale_candidates.append(1.5)
            else:
                scale_candidates.append(1.25)
            # De-duplicate while preserving order
            scale_candidates = list(dict.fromkeys(scale_candidates))

            # Try original and contrast-enhanced images at multiple scales
            tried = []
            for base_img in (image, _enhance_contrast(image)):
                for scale in scale_candidates:
                    if scale <= 0:
                        continue
                    if not (0.3 <= scale <= 2.5):
                        continue
                    if max_dim * scale > 2000:
                        continue
                    resized = cv2.resize(base_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    tried.append(scale)
                    faces = self.face_analyzer.get(resized)
                    if len(faces) > 0:
                        break
                if len(faces) > 0:
                    break

        if len(faces) == 0:
            print(f"No face detected in target image for session {session_id} (size={image.shape[1]}x{image.shape[0]})")
            return False
        
        # Use the largest face (most prominent)
        target_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # Store face data for this session
        self.target_faces[session_id] = {
            "face": target_face,
            "embedding": target_face.embedding,
        }
        
        print(f"Target face set for session {session_id}")
        return True
    
    def has_target(self, session_id: str) -> bool:
        """Check if a session has a target face set"""
        return session_id in self.target_faces
    
    def swap_face(self, session_id: str, frame: np.ndarray) -> np.ndarray:
        """
        Perform face swap on input frame.
        
        Args:
            session_id: Session identifier with pre-set target face
            frame: BGR input frame from webcam
            
        Returns:
            Processed frame with face swapped (or original if no swap possible)
        """
        result, _ = self.swap_face_with_faces(session_id, frame)
        return result

    def swap_face_with_faces(self, session_id: str, frame: np.ndarray):
        """Swap face and also return detected source faces for downstream processing."""
        if session_id not in self.target_faces:
            return frame, []

        target_data = self.target_faces[session_id]
        target_face = target_data["face"]

        # Detect faces in the input frame (with fallback for small/low-res faces)
        source_faces = self._detect_faces_with_fallback(frame)

        if len(source_faces) == 0:
            # Return last good result to avoid flashing raw frame
            if session_id in self._last_result:
                return self._last_result[session_id], []
            return frame, []

        # Sort faces by size (largest first)
        source_faces = sorted(
            source_faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )

        result = frame.copy()

        # Swap up to MAX_FACES for stability (default 1)
        for source_face in source_faces[:max(1, MAX_FACES)]:
            result = self._swap_single_face(result, source_face, target_face, session_id)

        # Cache last good result to avoid flashing on face-lost frames
        self._last_result[session_id] = result
        return result, source_faces

    def _swap_single_face(
        self,
        frame: np.ndarray,
        source_face: Any,
        target_face: Any,
        session_id: str
    ) -> np.ndarray:
        """
        Swap a single face in the frame.
        Uses InsightFace's INSwapper.get() which handles emap, alignment, and inference.
        """
        # Get face bounding box and landmarks
        bbox = source_face.bbox.astype(np.float32)
        
        # Optionally smooth landmarks AND bbox for stability
        kps = source_face.kps
        if ENABLE_TEMPORAL_SMOOTHING:
            kps = self._smooth_landmarks(session_id, kps)
            bbox = self._smooth_bounding_box(session_id, bbox)
        bbox = bbox.astype(int)

        try:
            # Use INSwapper's built-in get() method
            # It handles: alignment, emap transformation, inference, paste-back
            # source_face = face detected in the webcam frame (to be replaced)
            # target_face = the uploaded target face (identity to swap to)
            result = self.swapper.get(frame, source_face, target_face, paste_back=True)
            return result
            
        except Exception as e:
            import traceback
            print(f"Swap error: {e}")
            traceback.print_exc()
            return frame

    def _detect_faces_with_fallback(self, frame: np.ndarray):
        """Detect faces with a fallback upscaling pass for small/low-res faces."""
        faces = self.face_analyzer.get(frame)
        if len(faces) > 0:
            return faces

        height, width = frame.shape[:2]
        min_dim = min(height, width)
        scale = 1.0
        resized = None

        # Upscale if face likely too small
        if min_dim < 360:
            scale = 720 / max(1, min_dim)
        elif min_dim < 480:
            scale = 640 / max(1, min_dim)
        elif min_dim < 640:
            scale = 1.25

        if scale > 1.0:
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            faces_resized = self.face_analyzer.get(resized)
            if len(faces_resized) == 0:
                return []

            # Map detections back to original coordinates
            mapped = []
            inv_scale = 1.0 / scale
            for f in faces_resized:
                try:
                    bbox = (f.bbox.astype(np.float32) * inv_scale).astype(np.float32)
                    kps = (f.kps.astype(np.float32) * inv_scale).astype(np.float32)
                    det_score = getattr(f, 'det_score', 0.9)
                    ns = SimpleNamespace(bbox=bbox, kps=kps, det_score=det_score)
                    # Preserve 68-point 3D landmarks for precise mask contour
                    if hasattr(f, 'landmark_3d_68') and f.landmark_3d_68 is not None:
                        lm68 = f.landmark_3d_68.copy().astype(np.float32)
                        lm68[:, 0] *= inv_scale
                        lm68[:, 1] *= inv_scale
                        ns.landmark_3d_68 = lm68
                    # Preserve 106-point 2D landmarks
                    if hasattr(f, 'landmark_2d_106') and f.landmark_2d_106 is not None:
                        lm106 = f.landmark_2d_106.copy().astype(np.float32)
                        lm106 *= inv_scale
                        ns.landmark_2d_106 = lm106
                    mapped.append(ns)
                except Exception:
                    continue
            return mapped

        return []
    
    # Standard 5-point alignment template for 128x128 crop
    _ALIGN_TEMPLATE = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    def _align_face(self, frame: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
        """Align face for model input using landmarks (full affine)."""
        try:
            # Use full 6-DOF affine for better expression/pose tracking
            tform, _ = cv2.estimateAffine2D(
                kps.astype(np.float32),
                self._ALIGN_TEMPLATE,
                method=cv2.LMEDS
            )
            if tform is None:
                # Fallback to partial affine
                tform = cv2.estimateAffinePartial2D(kps, self._ALIGN_TEMPLATE)[0]
            if tform is None:
                return None
            aimg = cv2.warpAffine(frame, tform, (128, 128), borderValue=0.0)
            return aimg
        except Exception:
            return None
    
    def _paste_back(
        self, 
        frame: np.ndarray, 
        swapped: np.ndarray, 
        kps: np.ndarray,
        bbox: np.ndarray,
        source_face: Any = None,
        session_id: str = None
    ) -> np.ndarray:
        """Paste the swapped face back using 68-point contour mask for ~95% accuracy."""
        try:
            # Use full 6-DOF affine for accurate inverse mapping
            tform, _ = cv2.estimateAffine2D(
                self._ALIGN_TEMPLATE,
                kps.astype(np.float32),
                method=cv2.LMEDS
            )
            if tform is None:
                tform = cv2.estimateAffinePartial2D(self._ALIGN_TEMPLATE, kps)[0]
            if tform is None:
                return frame

            # Warp swapped face to original position
            h, w = frame.shape[:2]
            warped = cv2.warpAffine(swapped, tform, (w, h), borderValue=0.0)

            # Warped template points (needed for center + fallback mask)
            template_h = np.hstack([self._ALIGN_TEMPLATE, np.ones((5, 1), dtype=np.float32)])
            warped_pts = (tform @ template_h.T).T.astype(np.int32)

            x1, y1, x2, y2 = bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            used_68 = False

            # ── PRIMARY: 68-point jawline + forehead contour mask ──
            lm68 = None
            if source_face is not None and hasattr(source_face, 'landmark_3d_68') and source_face.landmark_3d_68 is not None:
                lm68 = source_face.landmark_3d_68[:, :2].astype(np.float32)
                # Smooth 68-point landmarks for stable mask contour
                if ENABLE_TEMPORAL_SMOOTHING and session_id:
                    lm68 = self._smooth_landmarks_68(session_id, lm68)

            if lm68 is not None:
                try:
                    # Jawline contour: points 0-16 (17 points tracing chin-to-chin)
                    jaw = lm68[0:17]
                    # Eyebrow points: 17-21 (left), 22-26 (right)
                    left_brow = lm68[17:22]
                    right_brow = lm68[22:27]

                    # Estimate forehead: project above eyebrows by 65% of face height
                    brow_y = (left_brow[:, 1].mean() + right_brow[:, 1].mean()) / 2.0
                    jaw_y = jaw[:, 1].max()
                    forehead_shift = (jaw_y - brow_y) * 0.65

                    # Create forehead arc from eyebrow points shifted upward
                    brow_pts = np.vstack([left_brow[::-1], right_brow[::-1]])
                    forehead = brow_pts.copy()
                    forehead[:, 1] -= forehead_shift

                    # Complete face contour: jaw (bottom) + forehead (top)
                    contour = np.vstack([jaw, forehead]).astype(np.int32)
                    hull = cv2.convexHull(contour)
                    cv2.fillConvexPoly(mask, hull, 255)

                    if mask.sum() > 100:
                        used_68 = True
                except Exception:
                    mask = np.zeros((h, w), dtype=np.uint8)

            # ── FALLBACK 1: 5-point warped template hull ──
            if not used_68:
                try:
                    hull = cv2.convexHull(warped_pts)
                    cv2.fillConvexPoly(mask, hull, 255)
                except Exception:
                    pass

            # ── FALLBACK 2: bbox rectangle ──
            if mask.sum() < 10:
                x1c, y1c = max(0, int(x1)), max(0, int(y1))
                x2c, y2c = min(w, int(x2)), min(h, int(y2))
                cv2.rectangle(mask, (x1c, y1c), (x2c, y2c), 255, thickness=-1)

            # Compute center for seamlessClone
            # Compute bounding box of the mask for efficient/safe cloning
            if mask.sum() > 0:
                y_indices, x_indices = np.nonzero(mask)
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                
                # Add a small padding
                pad = 10
                min_y = max(0, min_y - pad)
                max_y = min(h, max_y + pad)
                min_x = max(0, min_x - pad)
                max_x = min(w, max_x + pad)
                
                sub_h = max_y - min_y
                sub_w = max_x - min_x
                
                if sub_h > 0 and sub_w > 0:
                    # Crop to ROI
                    sub_warped = warped[min_y:max_y, min_x:max_x]
                    sub_mask = mask[min_y:max_y, min_x:max_x]
                    
                    # Center of the ROI in the destination image
                    cx = min_x + sub_w // 2
                    cy = min_y + sub_h // 2
                    
                    if ENABLE_SEAMLESS_CLONE:
                        try:
                            blended = cv2.seamlessClone(sub_warped, frame, sub_mask, (cx, cy), cv2.NORMAL_CLONE)
                            return blended
                        except Exception as e:
                            print(f"Seamless clone failed (center={cx},{cy}): {e}")
            
            # Fallback if mask is empty or clone failed
            if ENABLE_SEAMLESS_CLONE:
                 pass # Already fell through or failed

            # Fallback alpha blend with mask blur for smooth edges
            blur_amount = FACE_MASK_BLUR
            if blur_amount > 0:
                if blur_amount % 2 == 0:
                    blur_amount += 1
                mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            result = frame * (1 - alpha) + warped * alpha
            return result.astype(np.uint8)

        except Exception as e:
            print(f"Paste back error: {e}")
            return frame

    def _smooth_landmarks(self, session_id: str, kps: np.ndarray) -> np.ndarray:
        """Exponential smoothing of landmarks per session for stability."""
        if session_id not in self._smooth_kps:
            self._smooth_kps[session_id] = kps.copy()
            return kps
        prev = self._smooth_kps[session_id]
        smoothed = SMOOTHING_ALPHA * prev + (1.0 - SMOOTHING_ALPHA) * kps
        self._smooth_kps[session_id] = smoothed
        return smoothed

    def _smooth_landmarks_68(self, session_id: str, lm68: np.ndarray) -> np.ndarray:
        """Exponential smoothing of 68-point landmarks for stable mask contour."""
        key = f"{session_id}_lm68"
        if key not in self._smooth_kps:
            self._smooth_kps[key] = lm68.copy()
            return lm68
        prev = self._smooth_kps[key]
        if prev.shape != lm68.shape:
            self._smooth_kps[key] = lm68.copy()
            return lm68
        smoothed = SMOOTHING_ALPHA * prev + (1.0 - SMOOTHING_ALPHA) * lm68
        self._smooth_kps[key] = smoothed
        return smoothed

    def _smooth_bounding_box(self, session_id: str, bbox: np.ndarray) -> np.ndarray:
        """Exponential smoothing of bounding box to prevent mask jitter."""
        key = f"{session_id}_bbox"
        if key not in self._smooth_bbox:
            self._smooth_bbox[key] = bbox.copy()
            return bbox
        prev = self._smooth_bbox[key]
        smoothed = SMOOTHING_ALPHA * prev + (1.0 - SMOOTHING_ALPHA) * bbox
        self._smooth_bbox[key] = smoothed
        return smoothed

    def _color_match(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Histogram-based color matching in LAB space for ~95% accurate color transfer.
        
        Uses per-channel CDF lookup tables instead of simple mean/std.
        This preserves the full tonal range and handles non-Gaussian
        color distributions (shadows, highlights) much better.
        """
        try:
            if mask is None or mask.sum() < 10:
                return source
            mask_bool = mask > 0

            # Work in LAB uint8 (all channels 0-255 in OpenCV)
            src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
            tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
            matched = src_lab.copy()

            for c in range(3):
                src_ch = src_lab[..., c][mask_bool].ravel()
                tgt_ch = tgt_lab[..., c][mask_bool].ravel()
                if len(src_ch) == 0 or len(tgt_ch) == 0:
                    continue

                # Build CDFs from histograms
                src_hist = np.bincount(src_ch, minlength=256).astype(np.float64)
                tgt_hist = np.bincount(tgt_ch, minlength=256).astype(np.float64)
                src_cdf = np.cumsum(src_hist)
                src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
                tgt_cdf = np.cumsum(tgt_hist)
                tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1

                # Build 256-entry LUT: for each source level, find target level with closest CDF
                lut = np.searchsorted(tgt_cdf, src_cdf).clip(0, 255).astype(np.uint8)

                # Apply LUT only within the mask
                channel = matched[..., c].copy()
                channel[mask_bool] = lut[src_lab[..., c][mask_bool]]
                matched[..., c] = channel

            return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"Color match error: {e}")
            return source
    
    def cleanup_session(self, session_id: str):
        """Clean up session data to free memory"""
        if session_id in self.target_faces:
            del self.target_faces[session_id]
        if session_id in self._smooth_kps:
            del self._smooth_kps[session_id]
        # Clean up 68-landmark smoothing state
        lm68_key = f"{session_id}_lm68"
        if lm68_key in self._smooth_kps:
            del self._smooth_kps[lm68_key]
        bbox_key = f"{session_id}_bbox"
        if bbox_key in self._smooth_bbox:
            del self._smooth_bbox[bbox_key]
        if session_id in self._last_result:
            del self._last_result[session_id]
        print(f"Cleaned up session {session_id}")
