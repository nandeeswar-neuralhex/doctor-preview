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
    ENABLE_FACE_PARSING,
    ENABLE_REALTIME_ENHANCE,
    ENHANCE_EVERY_N_FRAMES,
    DETECTION_SIZE,
    SWAP_ENGINE,
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
        # CRITICAL: Limit CPU threads for ONNX pre/post-processing ops
        # Setting 0 = ALL cores, which starves asyncio event loop, base64, numpy
        # 2 threads is enough for the small CPU-side ops (transpose, gather, etc.)
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        
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
        
        # ── Step 4a: FULL face analyzer for target image upload (640×640, all models) ──
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=MODELS_DIR,
            providers=providers
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print(f"  Full analyzer: {len(self.face_analyzer.models)} models at 640×640 (for target upload)")
        
        # ── Step 4b: FAST face analyzer for real-time webcam (320×320, detection only) ──
        # KEY INSIGHT: INSwapper.get() only needs bbox + kps (5-point) from detection.
        # Landmark_3d_68 and landmark_2d_106 are NOT used by the swapper — they were
        # only needed for our custom _paste_back (which paste_back=True bypasses).
        # Dropping them saves ~10-15ms CPU per frame.
        self.face_analyzer_fast = FaceAnalysis(
            name="buffalo_l",
            root=MODELS_DIR,
            providers=providers,
            allowed_modules=["detection"]
        )
        self.face_analyzer_fast.prepare(ctx_id=0, det_size=(320, 320))
        print(f"  Fast analyzer: {len(self.face_analyzer_fast.models)} models at 320×320 (for real-time)")
        
        # Verify which provider the models actually use
        for analyzer_name, analyzer in [("full", self.face_analyzer), ("fast", self.face_analyzer_fast)]:
            for model in analyzer.models:
                try:
                    session = getattr(model, 'session', None)
                    if session:
                        active_providers = session.get_providers()
                        model_name = getattr(model, 'taskname', 'unknown')
                        print(f"  {analyzer_name}/{model_name} → {active_providers[0]}")
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

        # ── Phase 1.1: BiSeNet Face Parser ──
        self.face_parser = None
        if ENABLE_FACE_PARSING:
            try:
                from face_parser import FaceParser
                self.face_parser = FaceParser(providers)
                if self.face_parser.is_ready():
                    # On T4 (< 24GB), parse every 3rd frame to save latency
                    if not self._is_high_end_gpu():
                        self.face_parser.set_parse_frequency(3)
                    print("✅ BiSeNet face parser enabled")
                else:
                    print("⚠️  BiSeNet face parser not ready (model missing?)")
                    self.face_parser = None
            except Exception as e:
                print(f"BiSeNet face parser not available: {e}")
                self.face_parser = None

        # ── Phase 1.2: Real-time enhancement frame counter ──
        self._enhance_frame_counter: Dict[str, int] = {}

        # ── Phase 1.3: GPU capability detection for 640×640 detection ──
        self._use_high_res_detection = DETECTION_SIZE >= 640 or self._is_high_end_gpu()
        if self._use_high_res_detection:
            print(f"  Detection: using 640×640 full analyzer (high-end GPU detected)")
        else:
            print(f"  Detection: using 320×320 fast analyzer")

        # ── Phase 3: LivePortrait engine ──
        self.live_portrait = None
        if SWAP_ENGINE == "liveportrait":
            try:
                from live_portrait import LivePortrait
                self.live_portrait = LivePortrait(providers)
                if self.live_portrait.is_ready():
                    print("✅ LivePortrait engine enabled (swap_engine=liveportrait)")
                else:
                    print("⚠️  LivePortrait not ready — falling back to INSwapper")
                    self.live_portrait = None
            except Exception as e:
                print(f"LivePortrait not available: {e} — using INSwapper")
                self.live_portrait = None

        # Session storage for target faces and smoothing state
        self.target_faces: Dict[str, Any] = {}
        self._smooth_kps: Dict[str, np.ndarray] = {}
        self._smooth_bbox: Dict[str, np.ndarray] = {}
        self._last_result: Dict[str, np.ndarray] = {}
        # Fix #1: Affine matrix smoothing for sub-pixel face position stability
        self._smooth_affine: Dict[str, np.ndarray] = {}
        # Fix #1: Detection miss counter for graceful fade-out
        self._miss_count: Dict[str, int] = {}
        # Fix #2: Mouth MSE history for temporal averaging (removes noise)
        self._mouth_mse_history: Dict[str, list] = {}
        # Fix #2: Mouth mask activation state for hysteresis
        self._mouth_mask_active: Dict[str, bool] = {}
        # Fix #4: Target selection persistence
        self._current_target_idx: Dict[str, int] = {}
        self._target_hold_frames: Dict[str, int] = {}
        # Fix #5: Color LUT temporal smoothing cache
        self._color_lut_cache: Dict[str, np.ndarray] = {}
        # Fix #6: Pre-compute and cache the 128x128 soft oval mask (identical every frame)
        self._face_mask_128 = np.zeros((128, 128), dtype=np.float32)
        cv2.ellipse(self._face_mask_128, (64, 64), (52, 58), 0, 0, 360, 1.0, -1)
        self._face_mask_128 = cv2.GaussianBlur(self._face_mask_128, (31, 31), 0)
        self._face_mask_128 = (self._face_mask_128 * 255).astype(np.uint8)
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
            "full_analyzer_models": len(self.face_analyzer.models),
            "fast_analyzer_models": len(self.face_analyzer_fast.models),
            "analyzer_providers": [],
            "fast_analyzer_providers": [],
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
        for model in self.face_analyzer_fast.models:
            try:
                session = getattr(model, 'session', None)
                if session:
                    info["fast_analyzer_providers"].append(session.get_providers()[0])
            except Exception:
                info["fast_analyzer_providers"].append("unknown")
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
    
    def set_target_faces(self, session_id: str, images: list) -> dict:
        """
        Extract and store multiple target faces from uploaded images.
        Each image captures a different expression for expression matching.
        
        Args:
            session_id: Unique session identifier
            images: List of BGR images containing the target face
            
        Returns:
            dict with status and count of successful faces
        """
        # Clear previous faces for this session
        if session_id in self.target_faces:
            del self.target_faces[session_id]
        
        results = []
        for i, image in enumerate(images):
            success = self._extract_and_store_target(session_id, image, index=i)
            results.append(success)
        
        success_count = sum(results)
        if success_count == 0:
            return {"success": False, "count": 0, "message": "No faces detected in any image"}
        
        print(f"  Stored {success_count}/{len(images)} target faces for session {session_id}")
        return {"success": True, "count": success_count, "total": len(images)}

    def set_target_face(self, session_id: str, image: np.ndarray) -> bool:
        """
        Extract and store a single target face (appends to multi-face list).
        
        Args:
            session_id: Unique session identifier
            image: BGR image containing the target face
            
        Returns:
            True if face was successfully extracted and stored
        """
        return self._extract_and_store_target(session_id, image, index=None)

    def _extract_and_store_target(self, session_id: str, image: np.ndarray, index: int = None) -> bool:
        """
        Internal: Extract and store a target face from an image.
        Appends to the session's target face list for expression matching.
        """
        height, width = image.shape[:2]
        min_dim = min(height, width)
        max_dim = max(height, width)
        print(f"[set_target_face] session={session_id}, input size={width}x{height}")

        # Contrast enhancement helper
        def _enhance_contrast(img: np.ndarray) -> np.ndarray:
            try:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l2 = clahe.apply(l)
                return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
            except Exception:
                return img

        # Pad image helper — adds background so face isn't touching edges
        # Critical for close-up selfies where face fills 80%+ of the frame
        def _pad_image(img: np.ndarray, pad_pct: float = 0.4) -> np.ndarray:
            h, w = img.shape[:2]
            pad_y = int(h * pad_pct)
            pad_x = int(w * pad_pct)
            # Use border replication (or solid color) to create context around face
            padded = cv2.copyMakeBorder(
                img, pad_y, pad_y, pad_x, pad_x,
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            return padded

        # Build list of images to try: original, padded, contrast-enhanced, padded+enhanced
        base_images = [
            ("original", image),
            ("padded_40pct", _pad_image(image, 0.4)),
            ("padded_70pct", _pad_image(image, 0.7)),
            ("contrast", _enhance_contrast(image)),
            ("contrast_padded", _pad_image(_enhance_contrast(image), 0.5)),
        ]

        # Build scale candidates based on image dimensions
        scale_candidates = [1.0]
        if max_dim > 1200:
            scale_candidates.extend([
                1200 / max_dim,
                960 / max_dim,
                720 / max_dim,
                640 / max_dim,
                480 / max_dim,  # aggressive downscale for very close selfies
            ])
        elif max_dim > 900:
            scale_candidates.extend([960 / max_dim, 720 / max_dim, 640 / max_dim])
        elif max_dim > 640:
            scale_candidates.extend([640 / max_dim])
        # Upscale for small images
        if min_dim < 320:
            scale_candidates.append(640 / max(1, min_dim))
        elif min_dim < 480:
            scale_candidates.append(1.5)
        # De-duplicate while preserving order
        scale_candidates = list(dict.fromkeys(scale_candidates))

        faces = []
        # Try each base image variant at each scale
        for variant_name, base_img in base_images:
            bh, bw = base_img.shape[:2]
            b_max = max(bh, bw)
            for scale in scale_candidates:
                if scale <= 0 or not (0.15 <= scale <= 3.0):
                    continue
                target_max = b_max * scale
                if target_max > 2500 or target_max < 200:
                    continue
                if abs(scale - 1.0) < 0.01:
                    resized = base_img
                else:
                    resized = cv2.resize(base_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                
                faces = self.face_analyzer.get(resized)
                if len(faces) > 0:
                    print(f"  ✅ Face detected via variant='{variant_name}' scale={scale:.2f} (resized={resized.shape[1]}x{resized.shape[0]})")
                    break
            if len(faces) > 0:
                break

        if len(faces) == 0:
            label = f"image #{index}" if index is not None else "target image"
            print(f"  ❌ No face detected in {label} for session {session_id} after all attempts")
            return False
        
        # Use the largest face (most prominent)
        target_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # Extract expression features for matching:
        # - pose (yaw/pitch/roll) from bbox geometry
        # - landmark positions encode mouth open/smile/etc.
        expression_features = self._extract_expression_features(target_face)
        
        face_entry = {
            "face": target_face,
            "embedding": target_face.embedding,
            "expression_features": expression_features,
            "index": index,
        }
        
        # Initialize or append to the session's face list
        if session_id not in self.target_faces:
            self.target_faces[session_id] = []
        self.target_faces[session_id].append(face_entry)
        
        label = f"image #{index}" if index is not None else "target"
        print(f"  Target face ({label}) stored for session {session_id} (total: {len(self.target_faces[session_id])})")
        return True
    
    def _extract_expression_features(self, face) -> np.ndarray:
        """
        Extract a compact expression feature vector from a face.
        Uses landmark geometry to capture mouth openness, smile, eye state, and head pose.
        """
        features = []
        
        # From 5-point kps: eye distance, mouth-to-eye ratio
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps
            # Inter-eye distance (normalization reference)
            eye_dist = np.linalg.norm(kps[0] - kps[1]) + 1e-6
            # Nose-to-mouth distance (relative)
            nose_mouth = np.linalg.norm(kps[2] - np.mean([kps[3], kps[4]], axis=0)) / eye_dist
            # Mouth width
            mouth_w = np.linalg.norm(kps[3] - kps[4]) / eye_dist
            # Face symmetry (nose offset from eye midpoint)
            eye_mid = (kps[0] + kps[1]) / 2
            nose_offset = (kps[2][0] - eye_mid[0]) / eye_dist  # yaw proxy
            nose_v_offset = (kps[2][1] - eye_mid[1]) / eye_dist  # pitch proxy
            # Mouth corner vertical positions relative to nose — captures subtle
            # smile/frown that shifts corners by only 1-2 pixels at 320×320.
            mouth_left_v = (kps[3][1] - kps[2][1]) / eye_dist
            mouth_right_v = (kps[4][1] - kps[2][1]) / eye_dist
            # Mouth midpoint vertical — captures jaw-open (mouth drops down)
            mouth_mid_y = ((kps[3][1] + kps[4][1]) / 2.0 - kps[2][1]) / eye_dist
            features.extend([nose_mouth, mouth_w, nose_offset, nose_v_offset,
                             mouth_left_v, mouth_right_v, mouth_mid_y])
        
        # From 68-point landmarks (if available): detailed expression
        if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            lm = face.landmark_3d_68[:, :2]
            eye_dist = np.linalg.norm(lm[36] - lm[45]) + 1e-6
            # Mouth openness (vertical)
            mouth_open = np.linalg.norm(lm[62] - lm[66]) / eye_dist
            # Mouth width
            mouth_width = np.linalg.norm(lm[48] - lm[54]) / eye_dist
            # Smile: corner height relative to center
            mouth_center_y = (lm[62][1] + lm[66][1]) / 2
            smile_left = (mouth_center_y - lm[48][1]) / eye_dist
            smile_right = (mouth_center_y - lm[54][1]) / eye_dist
            # Eye openness
            left_eye_open = np.linalg.norm(lm[37] - lm[41]) / eye_dist
            right_eye_open = np.linalg.norm(lm[43] - lm[47]) / eye_dist
            # Brow raise
            left_brow = np.linalg.norm(lm[19] - lm[37]) / eye_dist
            right_brow = np.linalg.norm(lm[24] - lm[43]) / eye_dist
            # Jaw angle (head tilt)
            jaw_angle = np.arctan2(lm[16][1] - lm[0][1], lm[16][0] - lm[0][0])
            features.extend([
                mouth_open, mouth_width, smile_left, smile_right,
                left_eye_open, right_eye_open, left_brow, right_brow, jaw_angle
            ])
        
        return np.array(features, dtype=np.float32) if features else np.zeros(4, dtype=np.float32)
    
    def _match_best_target(self, session_id: str, source_face) -> dict:
        """
        Match the source face's expression to the best target face from the uploaded set.
        Returns the target face entry that best matches the current expression.
        Uses hysteresis + hold timer to prevent rapid target switching (Fix #4).
        """
        target_list = self.target_faces.get(session_id, [])
        if not target_list:
            return None
        if len(target_list) == 1:
            return target_list[0]
        
        # Extract expression features from the source (webcam) face
        source_features = self._extract_expression_features(source_face)
        
        # Find the target with the most similar expression
        best_match_idx = 0
        best_score = float('inf')
        
        for idx, entry in enumerate(target_list):
            target_features = entry.get("expression_features")
            if target_features is None:
                continue
            # Compare features (use only the overlapping dimensions)
            min_len = min(len(source_features), len(target_features))
            if min_len == 0:
                continue
            dist = np.linalg.norm(source_features[:min_len] - target_features[:min_len])
            if dist < best_score:
                best_score = dist
                best_match_idx = idx
        
        # ── Fix #4: Target persistence with hysteresis ──
        current_idx = self._current_target_idx.get(session_id)
        hold_count = self._target_hold_frames.get(session_id, 0)
        
        if current_idx is not None and current_idx < len(target_list):
            # Only switch if new match is significantly better (>30% lower distance)
            # AND we've held the current target for at least 15 frames (~0.75s)
            current_features = target_list[current_idx].get("expression_features")
            if current_features is not None:
                min_len_c = min(len(source_features), len(current_features))
                if min_len_c > 0:
                    current_dist = np.linalg.norm(source_features[:min_len_c] - current_features[:min_len_c])
                    # Keep current unless new is 30% better OR held for 15+ frames
                    if best_score > current_dist * 0.70 or hold_count < 15:
                        self._target_hold_frames[session_id] = hold_count + 1
                        return target_list[current_idx]
        
        # Switch to new target
        self._current_target_idx[session_id] = best_match_idx
        self._target_hold_frames[session_id] = 0
        return target_list[best_match_idx]
    
    def has_target(self, session_id: str) -> bool:
        """Check if a session has at least one target face set"""
        targets = self.target_faces.get(session_id, [])
        return len(targets) > 0
    
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

    def swap_face_with_faces(self, session_id: str, frame: np.ndarray, skip_mouth_preservation: bool = False):
        """Swap face using expression-matched target and return detected faces.
        
        Supports two engines:
          - "inswapper": Traditional INSwapper 128×128 pipeline
          - "liveportrait": LivePortrait motion-driven generation (Phase 3)
        
        Args:
            skip_mouth_preservation: If True, skip mouth interior preservation
                                     (used when lip sync will run afterwards)
        """
        import time as _t
        target_list = self.target_faces.get(session_id, [])
        if not target_list:
            return frame, []

        _t0 = _t.time()
        # Detect faces in the input frame (with fallback for small/low-res faces)
        source_faces = self._detect_faces_with_fallback(frame)
        _t1 = _t.time()

        if len(source_faces) == 0:
            # ── Fix #1: Graceful fade-out on detection miss ──
            # Keep returning cached swap for up to 90 frames (4.5s at 20fps) before
            # slowly fading back — this prevents rapid original↔swapped flicker when
            # detection momentarily drops (pose change, motion blur, lighting).
            miss_count = self._miss_count.get(session_id, 0) + 1
            self._miss_count[session_id] = miss_count
            if session_id in self._last_result:
                if miss_count <= 90:
                    # First 90 missed frames: return cached result (no visible flicker)
                    return self._last_result[session_id], []
                else:
                    # After 90 misses: slowly blend toward raw frame (2% per frame)
                    # Takes 50 more frames (~2.5s) to fully return to original
                    blend = min(1.0, (miss_count - 90) * 0.02)
                    cached = self._last_result[session_id]
                    if cached.shape == frame.shape:
                        faded = cv2.addWeighted(cached, 1.0 - blend, frame, blend, 0)
                        return faded, []
                    # Resolution changed — resize cached to match current frame
                    resized_cached = cv2.resize(cached, (frame.shape[1], frame.shape[0]))
                    faded = cv2.addWeighted(resized_cached, 1.0 - blend, frame, blend, 0)
                    return faded, []
            return frame, []
        
        # ── Smooth recovery from detection miss ──
        # Save miss count before resetting — used for cross-fade below
        _prev_miss_count = self._miss_count.get(session_id, 0)
        self._miss_count[session_id] = 0

        # ── Phase 3: LivePortrait engine path ──
        if self.live_portrait and self.live_portrait.is_ready():
            try:
                best = max(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                best.bbox = self._smooth_bounding_box(session_id, best.bbox)

                # Get cached appearance (extracted on target upload)
                appearance = self.live_portrait.get_cached_appearance(session_id)
                if appearance is not None:
                    # Extract motion from source face
                    motion = self.live_portrait.extract_motion(frame, best.bbox, session_id)
                    if motion is not None:
                        # Generate new face
                        generated = self.live_portrait.generate(appearance, motion)
                        if generated is not None:
                            # Get parsing mask if available
                            parse_mask = None
                            if self.face_parser and self.face_parser.is_ready():
                                parse_mask = self.face_parser.parse(frame, best.bbox, session_id)
                            # Stitch onto background
                            result = self.live_portrait.stitch(
                                generated, frame, best.bbox, parse_mask
                            )
                            self._last_result[session_id] = result
                            return result, source_faces
            except Exception as e:
                print(f"LivePortrait swap failed: {e} — falling back to INSwapper")
            # Fall through to INSwapper if LivePortrait fails

        result = frame
        n_swap = max(1, MAX_FACES)

        _t2 = _t.time()
        if n_swap == 1:
            best = max(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            # ── Fix #1: Smooth bounding box to prevent ROI jitter ──
            best.bbox = self._smooth_bounding_box(session_id, best.bbox)
            matched_target = self._match_best_target(session_id, best)
            _t3 = _t.time()
            if matched_target:
                result = self._swap_single_face(result, best, matched_target["face"], session_id, skip_mouth_preservation)
            _t4 = _t.time()
        else:
            source_faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            _t3 = _t.time()
            for i, source_face in enumerate(source_faces[:n_swap]):
                # Smooth bbox per face
                face_key = f"{session_id}_f{i}"
                source_face.bbox = self._smooth_bounding_box(face_key, source_face.bbox)
                matched_target = self._match_best_target(session_id, source_face)
                if matched_target:
                    result = self._swap_single_face(result, source_face, matched_target["face"], session_id, skip_mouth_preservation)
            _t4 = _t.time()

        # Print per-step breakdown every 60 frames (every ~3 seconds at 20fps)
        if not hasattr(self, '_dbg_count'):
            self._dbg_count = {}
        self._dbg_count[session_id] = self._dbg_count.get(session_id, 0) + 1
        if self._dbg_count[session_id] % 60 == 0:
            print(f"  [PROFILE] detect={(_t1-_t0)*1000:.1f}ms  match={(_t3-_t2)*1000:.1f}ms  swap={(_t4-_t3)*1000:.1f}ms  total={(_t4-_t0)*1000:.1f}ms")

        # Cross-fade when recovering from detection miss to prevent visible jump
        # Only activates after 3+ missed frames (~150ms gap)
        if _prev_miss_count > 3 and session_id in self._last_result:
            cached = self._last_result[session_id]
            if cached.shape == result.shape:
                blend = min(0.35, _prev_miss_count * 0.02)
                result = cv2.addWeighted(result, 1.0 - blend, cached, blend, 0)

        # Cache last good result to avoid flashing on face-lost frames
        self._last_result[session_id] = result
        return result, source_faces

    def _swap_single_face(
        self,
        frame: np.ndarray,
        source_face: Any,
        target_face: Any,
        session_id: str,
        skip_mouth_preservation: bool = False
    ) -> np.ndarray:
        """
        Swap a single face using ROI-direct warping for maximum CPU efficiency.

        Key optimization:
        - M maps frame→128px, M_inv maps 128px→full frame
        - Instead of warpAffine over entire 1080p frame (657K pixels), we adjust
          M_inv's translation to warp DIRECTLY into the face ROI (~300x300 = 90K pixels)
        - This is a 7x CPU speedup for the paste-back step with identical quality

        Smoothness fixes applied:
        - Fix #1: Affine matrix M_inv is temporally smoothed → sub-pixel stable face position
        - Fix #2: Mouth preservation uses heavy EMA + hysteresis → zero mouth flicker
        - Fix #5: Color LUT is temporally smoothed → no skin-tone flicker
        - Fix #6: Cached mask, larger padding, rounded ROI coords → stable boundaries
        """
        if ENABLE_TEMPORAL_SMOOTHING and hasattr(source_face, 'kps') and source_face.kps is not None:
            source_face.kps = self._smooth_landmarks(session_id, source_face.kps)

        try:
            # GPU: face alignment + ONNX inference → 128x128 swapped face + affine matrix M
            res = self.swapper.get(frame, source_face, target_face, paste_back=False)
            if res is None:
                return frame
            bgr_fake, M = res  # bgr_fake: (128,128,3), M: (2,3) affine frame→128px

            # ── Phase 1.2: Real-time GFPGAN enhancement on swapped face ──
            if ENABLE_REALTIME_ENHANCE and self.enhancer is not None:
                enhance_key = f"{session_id}_enhance"
                self._enhance_frame_counter[enhance_key] = (
                    self._enhance_frame_counter.get(enhance_key, 0) + 1
                )
                if self._enhance_frame_counter[enhance_key] % ENHANCE_EVERY_N_FRAMES == 0:
                    try:
                        # Upscale 128→512 for GFPGAN, then downscale back
                        face_up = cv2.resize(bgr_fake, (512, 512), interpolation=cv2.INTER_CUBIC)
                        _, _, enhanced = self.enhancer.enhance(
                            face_up, has_aligned=True, only_center_face=True, paste_back=False
                        )
                        if enhanced is not None:
                            bgr_fake = cv2.resize(enhanced, (128, 128), interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass  # Enhancement failure is non-critical

            h, w = frame.shape[:2]

            # Inverse affine for paste-back. NOT smoothed separately because
            # the source kps that produce M are already EMA-smoothed.
            # Double-smoothing caused ~150-250ms face lag on head turns.
            M_inv = cv2.invertAffineTransform(M)

            # ── Fix #6: Build face ROI with larger padding + rounded coords ──
            bbox = source_face.bbox.astype(int)
            x1 = max(0, bbox[0]); y1 = max(0, bbox[1])
            x2 = min(w, bbox[2]); y2 = min(h, bbox[3])
            pad = 45  # Increased from 30 → wider boundary hides edge artifacts
            # Round to even pixels to prevent 1px oscillation
            roi_y1 = max(0, (y1 - pad) & ~1); roi_y2 = min(h, (y2 + pad + 1) & ~1)
            roi_x1 = max(0, (x1 - pad) & ~1); roi_x2 = min(w, (x2 + pad + 1) & ~1)
            roi_h = roi_y2 - roi_y1; roi_w = roi_x2 - roi_x1
            if roi_h < 10 or roi_w < 10:
                return frame

            # ── ROI-direct warp: adjust M_inv translation by ROI origin ──
            M_roi_inv = M_inv.copy()
            M_roi_inv[0, 2] -= roi_x1   # shift x translation
            M_roi_inv[1, 2] -= roi_y1   # shift y translation

            # Warp 128×128 swapped face → ROI-sized (not 1080p!)
            roi_warped = cv2.warpAffine(bgr_fake, M_roi_inv, (roi_w, roi_h),
                                        borderMode=cv2.BORDER_REPLICATE)

            # ── Phase 1.1: BiSeNet parsing mask OR fallback to oval ──
            parsing_mask = None
            if self.face_parser and self.face_parser.is_ready():
                try:
                    parsing_mask = self.face_parser.parse_roi(
                        frame, source_face.bbox,
                        (roi_x1, roi_y1, roi_x2, roi_y2),
                        session_id,
                    )
                except Exception:
                    parsing_mask = None

            if parsing_mask is not None and parsing_mask.shape == (roi_h, roi_w):
                # Use BiSeNet mask (already float32 in [0, 1])
                roi_mask = (parsing_mask * 255).astype(np.uint8)
            else:
                # Fallback: cached 128×128 soft oval mask
                roi_mask = cv2.warpAffine(self._face_mask_128, M_roi_inv, (roi_w, roi_h))

            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

            # ── Fix #5: Color match with temporally-smoothed LUT ──
            if ENABLE_SEAMLESS_CLONE and roi_mask.sum() > 0:
                try:
                    roi_warped = self._color_match(roi_warped, roi_frame, roi_mask, session_id)
                except Exception:
                    pass

            # ── Smooth alpha blend on ROI ──
            blur_k = FACE_MASK_BLUR if FACE_MASK_BLUR % 2 == 1 else FACE_MASK_BLUR + 1
            blurred_mask = cv2.GaussianBlur(roi_mask, (blur_k, blur_k), 0) if blur_k > 1 else roi_mask
            alpha = (blurred_mask.astype(np.float32) / 255.0)[..., None]
            blended_roi = (roi_frame * (1 - alpha) + roi_warped * alpha).astype(np.uint8)

            # ── Fix #2: Mouth Interior Preservation (Stabilized) ──
            # Skip when lip sync is active (Fix #3) to avoid double-overwrite conflict
            if not skip_mouth_preservation and hasattr(source_face, 'kps') and source_face.kps is not None:
                try:
                    kps = source_face.kps
                    mouth_left = kps[3]
                    mouth_right = kps[4]
                    mouth_center = (mouth_left + mouth_right) / 2.0
                    mouth_w = np.linalg.norm(mouth_right - mouth_left)
                    eye_dist = np.linalg.norm(kps[1] - kps[0])

                    cx = mouth_center[0]
                    cy = mouth_center[1] + eye_dist * 0.08

                    # Convert to ROI coordinates
                    cx_roi = int(cx) - roi_x1
                    cy_roi = int(cy) - roi_y1

                    # ── Step 1: Color-difference detection with MSE averaging ──
                    sample_rx = max(6, int(mouth_w * 0.32))
                    sample_ry = max(6, int(eye_dist * 0.16))
                    sy1 = max(0, cy_roi - sample_ry)
                    sy2 = min(roi_h, cy_roi + sample_ry)
                    sx1 = max(0, cx_roi - sample_rx)
                    sx2 = min(roi_w, cx_roi + sample_rx)

                    mask_strength = 0.0
                    if sy2 > sy1 + 2 and sx2 > sx1 + 2 and 0 < cx_roi < roi_w and 0 < cy_roi < roi_h:
                        swap_patch = blended_roi[sy1:sy2, sx1:sx2].astype(np.float32)
                        orig_patch = roi_frame[sy1:sy2, sx1:sx2].astype(np.float32)

                        mse = np.mean((swap_patch - orig_patch) ** 2)

                        # Fix #2: Average MSE over last 5 frames to remove noise
                        mse_key = f"{session_id}_mse"
                        if mse_key not in self._mouth_mse_history:
                            self._mouth_mse_history[mse_key] = []
                        self._mouth_mse_history[mse_key].append(mse)
                        if len(self._mouth_mse_history[mse_key]) > 5:
                            self._mouth_mse_history[mse_key] = self._mouth_mse_history[mse_key][-5:]
                        mse = np.mean(self._mouth_mse_history[mse_key])

                        mse_conf = np.clip((mse - 250.0) / 1000.0, 0.0, 1.0)

                        swap_std = np.std(swap_patch)
                        orig_std = np.std(orig_patch)
                        uniformity_ratio = swap_std / max(orig_std, 1.0)
                        uniform_conf = np.clip((0.6 - uniformity_ratio) / 0.4, 0.0, 0.5)

                        # Cap at 0.70 — never fully replace the swap with original.
                        # This preserves subtle expression differences that the swap
                        # model generated (small smiles, teeth showing slightly).
                        mask_strength = min(0.70, mse_conf + uniform_conf)

                    # ── Step 2: Heavy temporal smoothing (Fix #2) ──
                    smooth_key = f"{session_id}_mouth_str"
                    if smooth_key in self._smooth_bbox:
                        prev_str = self._smooth_bbox[smooth_key]
                        # Slow rise (0.7/0.3) and very slow decay (0.92/0.08)
                        # Takes ~10 frames to appear, ~25 frames to disappear → zero flicker
                        if mask_strength > prev_str:
                            mask_strength = 0.7 * prev_str + 0.3 * mask_strength
                        else:
                            mask_strength = 0.92 * prev_str + 0.08 * mask_strength
                    self._smooth_bbox[smooth_key] = mask_strength

                    # ── Step 3: Hysteresis thresholds (Fix #2) ──
                    active_key = f"{session_id}_mouth_active"
                    is_active = self._mouth_mask_active.get(active_key, False)
                    if is_active:
                        # Deactivate only when strength drops well below threshold
                        if mask_strength < 0.10:
                            is_active = False
                    else:
                        # Activate only when strength is clearly above threshold
                        if mask_strength > 0.30:
                            is_active = True
                    self._mouth_mask_active[active_key] = is_active

                    # ── Step 4: Apply mouth mask (Fixed geometry — only alpha varies) ──
                    if is_active and mask_strength > 0.05:
                        # Tighter ellipse focused on inner mouth/teeth area
                        # Smaller region reduces dark halo at skin-mouth boundary
                        ell_rx = int(mouth_w * 0.28)
                        ell_ry = int(eye_dist * 0.14)

                        if (ell_rx > 2 and ell_ry > 2 and
                                0 < cx_roi < roi_w and 0 < cy_roi < roi_h):
                            mouth_excl = np.zeros((roi_h, roi_w), dtype=np.float32)
                            cv2.ellipse(mouth_excl, (cx_roi, cy_roi),
                                        (ell_rx, ell_ry), 0, 0, 360, 1.0, -1)
                            # Wider blur for softer boundary to hide skin-mouth seam
                            blur_sz = max(71, int(max(ell_rx, ell_ry) * 3.5) | 1)
                            mouth_excl = cv2.GaussianBlur(mouth_excl, (blur_sz, blur_sz), 0)
                            mouth_excl *= mask_strength
                            m3 = mouth_excl[..., None]
                            blended_roi = (blended_roi.astype(np.float32) * (1.0 - m3) +
                                           roi_frame.astype(np.float32) * m3).astype(np.uint8)
                except Exception:
                    pass  # If landmark estimation fails, skip — swap still works

            result = frame  # modify in-place — caller doesn't reuse the input
            result[roi_y1:roi_y2, roi_x1:roi_x2] = blended_roi
            return result

        except Exception as e:
            print(f"Swap error: {e}")
            return frame

    def _detect_faces_with_fallback(self, frame: np.ndarray):
        """Detect faces with automatic analyzer selection.

        Phase 1.3: On high-end GPUs (H100/A100), use 640×640 full analyzer
        for better keypoint accuracy. On T4/A10, use 320×320 fast analyzer.
        Fix #1: Filter low-confidence detections to avoid noisy kps → jitter."""
        # Phase 1.3: Use full 640×640 analyzer on high-end GPUs
        if self._use_high_res_detection:
            faces = self.face_analyzer.get(frame)
        else:
            faces = self.face_analyzer_fast.get(frame)
        # Filter out low-confidence detections that give noisy landmarks
        # Raised from 0.55 → 0.65 to eliminate jittery bboxes from borderline detections
        faces = [f for f in faces if getattr(f, 'det_score', 0.9) >= 0.65]
        if len(faces) > 0:
            return faces

        # Fallback: upscale tiny frames so detector can find small faces
        height, width = frame.shape[:2]
        min_dim = min(height, width)
        scale = 1.0

        if min_dim < 360:
            scale = 720 / max(1, min_dim)
        elif min_dim < 480:
            scale = 640 / max(1, min_dim)
        elif min_dim < 640:
            scale = 1.25

        if scale > 1.0:
            resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            faces_resized = self.face_analyzer_fast.get(resized)
            if len(faces_resized) == 0:
                return []

            # Map detections back to original coordinates
            inv_scale = 1.0 / scale
            mapped = []
            for f in faces_resized:
                try:
                    bbox = (f.bbox * inv_scale).astype(np.float32)
                    kps = (f.kps * inv_scale).astype(np.float32)
                    det_score = getattr(f, 'det_score', 0.9)
                    mapped.append(SimpleNamespace(bbox=bbox, kps=kps, det_score=det_score))
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

            # ── FAST ROI-BASED BLENDING ──
            # Critical optimization: instead of running seamlessClone on the
            # ENTIRE 1080p frame (slow, ~100ms), we crop to just the face
            # bounding box region and run it only there (~5-8ms).
            if mask.sum() > 0:
                y_indices, x_indices = np.nonzero(mask)
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                min_x, max_x = np.min(x_indices), np.max(x_indices)

                # Padding so seamlessClone has context pixels at the border
                pad = 20
                roi_y1 = max(0, min_y - pad)
                roi_y2 = min(h, max_y + pad)
                roi_x1 = max(0, min_x - pad)
                roi_x2 = min(w, max_x + pad)

                roi_h = roi_y2 - roi_y1
                roi_w = roi_x2 - roi_x1

                if roi_h > 10 and roi_w > 10:
                    # Crop both the warped face and the mask to the face ROI
                    roi_warped  = warped[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                    roi_mask    = mask[roi_y1:roi_y2, roi_x1:roi_x2].copy()
                    roi_frame   = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

                    # SeamlessClone center relative to the small ROI (not full frame)
                    sc_cx = (min_x - roi_x1) + (max_x - min_x) // 2
                    sc_cy = (min_y - roi_y1) + (max_y - min_y) // 2
                    sc_cx = int(np.clip(sc_cx, 1, roi_w - 2))
                    sc_cy = int(np.clip(sc_cy, 1, roi_h - 2))

                    if ENABLE_SEAMLESS_CLONE:
                        try:
                            blended_roi = cv2.seamlessClone(
                                roi_warped, roi_frame, roi_mask,
                                (sc_cx, sc_cy), cv2.NORMAL_CLONE
                            )
                            result = frame.copy()
                            result[roi_y1:roi_y2, roi_x1:roi_x2] = blended_roi
                            return result
                        except Exception as e:
                            print(f"SeamlessClone failed: {e} — using alpha blend")

                    # Alpha-blend fallback (also ROI-only, very fast)
                    blur_amount = FACE_MASK_BLUR
                    if blur_amount % 2 == 0:
                        blur_amount += 1
                    if blur_amount > 1:
                        roi_mask_blur = cv2.GaussianBlur(roi_mask, (blur_amount, blur_amount), 0)
                    else:
                        roi_mask_blur = roi_mask
                    alpha = (roi_mask_blur.astype(np.float32) / 255.0)[..., None]
                    blended_roi = (roi_frame * (1 - alpha) + roi_warped * alpha).astype(np.uint8)
                    result = frame.copy()
                    result[roi_y1:roi_y2, roi_x1:roi_x2] = blended_roi
                    return result

            # Absolute fallback: return un-blended warped frame
            return frame

        except Exception as e:
            print(f"Paste back error: {e}")
            return frame

    def _smooth_landmarks(self, session_id: str, kps: np.ndarray) -> np.ndarray:
        """Per-point exponential smoothing of landmarks.

        Eye points (0, 1) and nose (2) use the standard alpha for position
        stability.  Mouth corners (3, 4) use a lighter alpha (50% of standard)
        so that subtle lip movements — small smiles, slight mouth opening —
        pass through with minimal damping.
        """
        if session_id not in self._smooth_kps:
            self._smooth_kps[session_id] = kps.copy()
            return kps
        prev = self._smooth_kps[session_id]
        # Per-point alpha: lighter smoothing on mouth corners
        alpha = np.full(kps.shape, SMOOTHING_ALPHA, dtype=np.float32)
        alpha[3] = SMOOTHING_ALPHA * 0.5   # left mouth corner — more responsive
        alpha[4] = SMOOTHING_ALPHA * 0.5   # right mouth corner — more responsive
        smoothed = alpha * prev + (1.0 - alpha) * kps
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

    def _color_match(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray, session_id: str = "") -> np.ndarray:
        """Histogram-based color matching in LAB space with temporal LUT smoothing.
        
        Fix #5: The CDF LUT is blended with the previous frame's LUT (70% old + 30% new)
        to prevent frame-to-frame skin tone flicker caused by varying pixel populations.
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

                # Build 256-entry LUT
                lut = np.searchsorted(tgt_cdf, src_cdf).clip(0, 255).astype(np.float64)

                # Reduce L-channel (lightness) matching intensity to prevent
                # mouth interior darkening — dark cavity pixels in the histogram
                # pull the LUT toward black, making teeth/lips even darker.
                # Blending 50% with identity preserves original brightness structure.
                if c == 0:
                    identity = np.arange(256, dtype=np.float64)
                    lut = 0.5 * identity + 0.5 * lut

                # Fix #5: Temporal smoothing of LUT (70% old + 30% new)
                if session_id:
                    lut_key = f"{session_id}_color_lut_{c}"
                    if lut_key in self._color_lut_cache:
                        lut = 0.7 * self._color_lut_cache[lut_key] + 0.3 * lut
                    self._color_lut_cache[lut_key] = lut.copy()

                lut_u8 = lut.clip(0, 255).astype(np.uint8)

                # Apply LUT only within the mask
                channel = matched[..., c].copy()
                channel[mask_bool] = lut_u8[src_lab[..., c][mask_bool]]
                matched[..., c] = channel

            return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"Color match error: {e}")
            return source
    
    def _is_high_end_gpu(self) -> bool:
        """Detect if running on H100/A100 (>= 40GB VRAM).

        Returns True for high-end GPUs that can handle 640×640 detection
        and real-time GFPGAN enhancement without FPS drop.
        """
        try:
            import torch
            if torch.cuda.is_available():
                vram_bytes = torch.cuda.get_device_properties(0).total_mem
                vram_gb = vram_bytes / (1024 ** 3)
                gpu_name = torch.cuda.get_device_properties(0).name
                is_high_end = vram_gb >= 40.0
                print(f"  GPU: {gpu_name}, VRAM: {vram_gb:.1f}GB, high_end={is_high_end}")
                return is_high_end
        except ImportError:
            pass
        except Exception as e:
            print(f"  GPU detection error: {e}")
        # Fallback: check config
        return DETECTION_SIZE >= 640

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
        mouth_key = f"{session_id}_mouth_str"
        if mouth_key in self._smooth_bbox:
            del self._smooth_bbox[mouth_key]
        if session_id in self._last_result:
            del self._last_result[session_id]
        # Fix #1: Clean up affine smoothing
        affine_key = f"{session_id}_affine"
        if affine_key in self._smooth_affine:
            del self._smooth_affine[affine_key]
        # Fix #1: Clean up miss counter
        self._miss_count.pop(session_id, None)
        # Fix #2: Clean up mouth MSE history and activation state
        mse_key = f"{session_id}_mse"
        self._mouth_mse_history.pop(mse_key, None)
        active_key = f"{session_id}_mouth_active"
        self._mouth_mask_active.pop(active_key, None)
        # Fix #4: Clean up target persistence
        self._current_target_idx.pop(session_id, None)
        self._target_hold_frames.pop(session_id, None)
        # Fix #5: Clean up color LUT cache
        for c in range(3):
            lut_key = f"{session_id}_color_lut_{c}"
            self._color_lut_cache.pop(lut_key, None)
        # Clean up multi-face bbox keys
        for i in range(10):
            face_key = f"{session_id}_f{i}"
            self._smooth_bbox.pop(f"{face_key}_bbox", None)
        # Phase 1.1: Clean up face parser cache
        if self.face_parser:
            self.face_parser.cleanup_session(session_id)
        # Phase 1.2: Clean up enhance frame counter
        enhance_key = f"{session_id}_enhance"
        self._enhance_frame_counter.pop(enhance_key, None)
        # Phase 3: Clean up LivePortrait cache
        if self.live_portrait:
            self.live_portrait.cleanup_session(session_id)
        print(f"Cleaned up session {session_id}")
