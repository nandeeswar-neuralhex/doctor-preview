"""
FaceSwapper - Core face swapping logic using InsightFace/FaceFusion models
Optimized for 24+ FPS real-time processing
"""
import cv2
import numpy as np
from typing import Dict, Optional, Any
import onnxruntime as ort
from insightface.app import FaceAnalysis

from config import MODELS_DIR, INSWAPPER_MODEL, EXECUTION_PROVIDER


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
        
        # Session storage for target faces
        self.target_faces: Dict[str, Any] = {}
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
            print(f"No face detected in target image for session {session_id}")
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
        if session_id not in self.target_faces:
            return frame
        
        target_data = self.target_faces[session_id]
        target_face = target_data["face"]
        
        # Detect faces in the input frame
        source_faces = self.face_analyzer.get(frame)
        
        if len(source_faces) == 0:
            return frame
        
        result = frame.copy()
        
        # Swap each detected face with the target
        for source_face in source_faces:
            result = self._swap_single_face(result, source_face, target_face)
        
        return result
    
    def _swap_single_face(
        self, 
        frame: np.ndarray, 
        source_face: Any, 
        target_face: Any
    ) -> np.ndarray:
        """
        Swap a single face in the frame.
        Uses InsightFace's inswapper model for high-quality swapping.
        """
        # Get face bounding box and landmarks
        bbox = source_face.bbox.astype(int)
        
        # Prepare input for the swapper model
        # The inswapper expects aligned face crops
        aimg = self._align_face(frame, source_face.kps)
        
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
            
            # Paste back to original frame
            result = self._paste_back(frame, pred, source_face.kps)
            return result
            
        except Exception as e:
            print(f"Swap error: {e}")
            return frame
    
    def _align_face(self, frame: np.ndarray, kps: np.ndarray) -> Optional[np.ndarray]:
        """Align face for model input using landmarks"""
        # Standard face alignment for 128x128 input
        src_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        try:
            tform = cv2.estimateAffinePartial2D(kps, src_pts)[0]
            aimg = cv2.warpAffine(frame, tform, (128, 128), borderValue=0.0)
            return aimg
        except:
            return None
    
    def _paste_back(
        self, 
        frame: np.ndarray, 
        swapped: np.ndarray, 
        kps: np.ndarray
    ) -> np.ndarray:
        """Paste the swapped face back onto the original frame"""
        # Inverse alignment
        src_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        try:
            tform = cv2.estimateAffinePartial2D(src_pts, kps)[0]
            
            # Warp swapped face to original position
            h, w = frame.shape[:2]
            warped = cv2.warpAffine(swapped, tform, (w, h), borderValue=0.0)
            
            # Create mask for blending
            mask = np.ones((128, 128), dtype=np.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask = cv2.warpAffine(mask, tform, (w, h), borderValue=0.0)
            mask = np.expand_dims(mask, axis=2)
            
            # Blend
            result = frame * (1 - mask) + warped * mask
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Paste back error: {e}")
            return frame
    
    def cleanup_session(self, session_id: str):
        """Clean up session data to free memory"""
        if session_id in self.target_faces:
            del self.target_faces[session_id]
            print(f"Cleaned up session {session_id}")
