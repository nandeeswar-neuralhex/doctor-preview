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
        
        # Configure ONNX for GPU
        providers = [EXECUTION_PROVIDER, "CPUExecutionProvider"]
        
        # Initialize face analyzer (detection + landmarks)
        self.face_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=MODELS_DIR,
            providers=providers
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load face swapper model
        print(f"Loading swapper model: {INSWAPPER_MODEL}")
        self.swapper = ort.InferenceSession(
            INSWAPPER_MODEL,
            providers=providers
        )
        
        # Get model input/output info
        self.input_name = self.swapper.get_inputs()[0].name
        self.input_shape = self.swapper.get_inputs()[0].shape
        
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
        self._ready = True
        
        print("FaceSwapper initialized successfully!")
    
    def is_ready(self) -> bool:
        """Check if the swapper is ready for processing"""
        return self._ready
    
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
            # Retry with resized image for small faces
            height, width = image.shape[:2]
            resized = None
            min_dim = min(height, width)
            if min_dim < 320:
                scale = 640 / max(1, min_dim)
                resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                scale = 1.5
                max_dim = max(height, width)
                if max_dim * scale <= 1600:
                    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            if resized is not None:
                faces = self.face_analyzer.get(resized)

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
        Uses InsightFace's inswapper model for high-quality swapping.
        """
        # Get face bounding box and landmarks
        bbox = source_face.bbox.astype(int)
        
        # Optionally smooth landmarks for stability
        kps = source_face.kps
        if ENABLE_TEMPORAL_SMOOTHING:
            kps = self._smooth_landmarks(session_id, kps)

        # Prepare input for the swapper model
        # The inswapper expects aligned face crops
        aimg = self._align_face(frame, kps)
        
        if aimg is None:
            return frame
        
        # Prepare model input
        blob = cv2.dnn.blobFromImage(
            aimg, 
            1.0 / 255.0, 
            (128, 128), 
            (0.0, 0.0, 0.0), 
            swapRB=True
        )
        
        # Get target embedding
        target_emb = target_face.embedding.reshape(1, -1).astype(np.float32)
        
        # Run inference
        try:
            pred = self.swapper.run(
                None, 
                {
                    self.input_name: blob,
                    self.swapper.get_inputs()[1].name: target_emb
                }
            )[0]
            
            # Post-process output
            pred = pred.squeeze().transpose(1, 2, 0)
            pred = (pred * 255).clip(0, 255).astype(np.uint8)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            # Optional enhancement
            if self.enhancer is not None:
                try:
                    _, _, pred = self.enhancer.enhance(pred, has_aligned=True, only_center_face=True, paste_back=False)
                except Exception as e:
                    print(f"Enhancer error: {e}")
            
            # Paste back to original frame
            result = self._paste_back(frame, pred, kps, bbox)
            return result
            
        except Exception as e:
            print(f"Swap error: {e}")
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
                    mapped.append(SimpleNamespace(bbox=bbox, kps=kps))
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
        bbox: np.ndarray
    ) -> np.ndarray:
        """Paste the swapped face back onto the original frame"""
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

            # Create mask from facial landmarks (convex hull)
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            try:
                hull = cv2.convexHull(kps.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 255)
            except Exception:
                mask = np.zeros((h, w), dtype=np.uint8)

            # Fallback: if mask is empty, use bbox rectangle
            if mask.sum() < 10:
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                cv2.rectangle(mask, (x1c, y1c), (x2c, y2c), 255, thickness=-1)

            # Optional dilation (scale)
            if FACE_MASK_SCALE > 1.0:
                scale = FACE_MASK_SCALE
                kx = max(3, int((x2 - x1) * (scale - 1.0)) | 1)
                ky = max(3, int((y2 - y1) * (scale - 1.0)) | 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky))
                mask = cv2.dilate(mask, kernel, iterations=1)

            if FACE_MASK_BLUR > 0:
                k = FACE_MASK_BLUR if FACE_MASK_BLUR % 2 == 1 else FACE_MASK_BLUR + 1
                mask = cv2.GaussianBlur(mask, (k, k), 0)

            # Color match for more natural blending
            warped = self._color_match(warped, frame, mask)

            if ENABLE_SEAMLESS_CLONE:
                center = (cx, cy)
                try:
                    blended = cv2.seamlessClone(warped, frame, mask, center, cv2.NORMAL_CLONE)
                    return blended
                except Exception as e:
                    print(f"Seamless clone failed: {e}")

            # Fallback alpha blend
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

    def _color_match(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Match color statistics of source to target within mask region."""
        try:
            if mask is None:
                return source
            mask_bool = mask > 0
            if mask_bool.sum() < 10:
                return source
            src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

            matched = src_lab.copy()
            for c in range(3):
                src_vals = src_lab[..., c][mask_bool]
                tgt_vals = tgt_lab[..., c][mask_bool]
                src_mean, src_std = src_vals.mean(), src_vals.std()
                tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std()
                if src_std < 1e-6:
                    continue
                matched[..., c] = (matched[..., c] - src_mean) * (tgt_std / src_std) + tgt_mean
            matched = np.clip(matched, 0, 255).astype(np.uint8)
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
            print(f"Cleaned up session {session_id}")
