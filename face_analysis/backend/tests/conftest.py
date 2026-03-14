"""
Test fixtures shared across all test modules.
"""

import os
import sys

import cv2
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_face_image():
    """Generate a synthetic face-like image for testing (512x512 BGR)."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Light skin tone

    # Simulate face oval
    cv2.ellipse(img, (256, 256), (150, 200), 0, 0, 360, (210, 185, 170), -1)

    # Eyes (dark circles)
    cv2.circle(img, (200, 220), 20, (80, 60, 50), -1)  # Left eye
    cv2.circle(img, (312, 220), 20, (80, 60, 50), -1)  # Right eye

    # Nose
    cv2.line(img, (256, 230), (256, 300), (190, 170, 155), 3)

    # Mouth
    cv2.ellipse(img, (256, 340), (40, 15), 0, 0, 180, (130, 100, 120), 2)

    # Add some texture noise (simulates skin texture)
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def sample_landmarks():
    """Generate synthetic 468-point landmarks for a 512x512 image."""
    landmarks = np.zeros((468, 2), dtype=np.float32)

    # Set key landmark positions on 512x512 image
    # Face outline
    landmarks[10] = [256, 60]     # Forehead top
    landmarks[152] = [256, 460]   # Chin
    landmarks[234] = [100, 256]   # Left cheek
    landmarks[454] = [412, 256]   # Right cheek

    # Eyes
    landmarks[33] = [180, 220]    # Left eye outer
    landmarks[133] = [220, 220]   # Left eye inner
    landmarks[263] = [330, 220]   # Right eye outer
    landmarks[362] = [290, 220]   # Right eye inner
    landmarks[160] = [195, 210]
    landmarks[158] = [195, 215]
    landmarks[153] = [195, 225]
    landmarks[144] = [195, 230]
    landmarks[387] = [315, 210]
    landmarks[385] = [315, 215]
    landmarks[380] = [315, 225]
    landmarks[373] = [315, 230]

    # Eyebrows
    landmarks[46] = [170, 190]    # Left eyebrow outer
    landmarks[276] = [340, 190]   # Right eyebrow outer
    landmarks[105] = [210, 195]   # Left eyebrow inner
    landmarks[334] = [300, 195]   # Right eyebrow inner

    # Nose
    landmarks[1] = [256, 300]     # Nose tip
    landmarks[6] = [256, 260]     # Nose bridge
    landmarks[168] = [256, 210]   # Nasion
    landmarks[48] = [240, 295]    # Nose left
    landmarks[278] = [272, 295]   # Nose right
    landmarks[197] = [256, 270]
    landmarks[195] = [256, 280]

    # Lips
    landmarks[0] = [256, 335]     # Upper lip top
    landmarks[17] = [256, 365]    # Lower lip bottom
    landmarks[61] = [225, 345]    # Lip left
    landmarks[291] = [287, 345]   # Lip right

    # Jaw
    landmarks[132] = [145, 380]   # Left jaw
    landmarks[361] = [367, 380]   # Right jaw
    landmarks[58] = [160, 420]
    landmarks[288] = [352, 420]

    # Face outline landmarks for mask generation
    face_outline_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                        323, 361, 288, 397, 365, 379, 378, 400, 377,
                        152, 148, 176, 149, 150, 136, 172, 58, 132,
                        93, 234, 127, 162, 21, 54, 103, 67, 109]

    # Set approximate positions for outline
    outline_positions = np.array([
        [256, 60], [300, 70], [340, 90], [360, 110], [380, 140],
        [395, 170], [405, 200], [410, 240], [412, 256],
        [410, 300], [367, 380], [352, 420], [340, 440],
        [320, 450], [300, 455], [280, 458], [270, 460], [260, 460],
        [256, 460], [250, 458], [240, 455], [220, 450], [200, 440],
        [180, 420], [160, 380], [145, 380], [100, 256], [105, 200],
        [110, 170], [120, 140], [140, 110], [160, 90], [200, 70],
        [220, 65], [240, 62],
    ], dtype=np.float32)

    for i, idx in enumerate(face_outline_idx):
        if i < len(outline_positions):
            landmarks[idx] = outline_positions[i]

    # Fill remaining landmarks with interpolated positions
    for i in range(468):
        if landmarks[i][0] == 0 and landmarks[i][1] == 0:
            # Random position within face area
            landmarks[i] = [
                np.random.uniform(120, 392),
                np.random.uniform(80, 440),
            ]

    return landmarks


@pytest.fixture
def sample_skin_mask():
    """Generate a face-shaped skin mask for 512x512."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.ellipse(mask, (256, 256), (150, 200), 0, 0, 360, 255, -1)
    return mask


@pytest.fixture
def sample_depth_map():
    """Generate a synthetic depth map (256x192, float32, meters)."""
    depth = np.ones((192, 256), dtype=np.float32) * 0.3  # 30cm base distance

    # Face protrusion (nose closer)
    y, x = np.ogrid[:192, :256]
    nose_mask = ((x - 128) ** 2 + (y - 96) ** 2) < 30 ** 2
    depth[nose_mask] -= 0.02  # Nose 2cm closer

    # Add slight noise
    depth += np.random.normal(0, 0.001, depth.shape).astype(np.float32)

    return depth


@pytest.fixture
def sample_depth_frames(sample_depth_map):
    """Generate multiple depth frames with slight variations (for multi-frame averaging)."""
    frames = []
    for _ in range(10):
        noise = np.random.normal(0, 0.002, sample_depth_map.shape).astype(np.float32)
        frames.append(sample_depth_map + noise)
    return frames
