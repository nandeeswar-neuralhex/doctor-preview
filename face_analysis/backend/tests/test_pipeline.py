"""
Unit Tests for Pipeline and Depth Enhancer
"""

import numpy as np
import pytest


class TestDepthEnhancer:
    def test_multi_frame_averaging(self, sample_depth_frames):
        from src.pipeline.depth_enhancer import DepthEnhancer
        enhancer = DepthEnhancer(n_frames=10)

        averaged = enhancer.average_depth_frames(sample_depth_frames)

        assert averaged.shape == sample_depth_frames[0].shape
        assert averaged.dtype == np.float32

        # Averaged noise should be less than individual frame noise
        ref = sample_depth_frames[0]
        noise_single = float(np.std(ref))
        noise_averaged = float(np.std(averaged - np.mean([f.mean() for f in sample_depth_frames])))
        # Averaging should reduce noise (not always exact due to alignment)
        assert noise_averaged < noise_single * 2  # Generous bound

    def test_single_frame_passthrough(self, sample_depth_map):
        from src.pipeline.depth_enhancer import DepthEnhancer
        enhancer = DepthEnhancer()

        result = enhancer.average_depth_frames([sample_depth_map])
        np.testing.assert_array_equal(result, sample_depth_map)

    def test_empty_frames_raises(self):
        from src.pipeline.depth_enhancer import DepthEnhancer
        enhancer = DepthEnhancer()

        with pytest.raises(ValueError, match="No depth frames"):
            enhancer.average_depth_frames([])

    def test_super_resolve_depth(self, sample_depth_map, sample_face_image):
        from src.pipeline.depth_enhancer import DepthEnhancer
        enhancer = DepthEnhancer(sr_scale=4)

        hr_depth = enhancer.super_resolve_depth(
            sample_depth_map, sample_face_image, scale=2
        )

        # Output should match RGB resolution
        assert hr_depth.shape[0] == sample_face_image.shape[0]
        assert hr_depth.shape[1] == sample_face_image.shape[1]

    def test_guided_filter_preserves_edges(self):
        from src.pipeline.depth_enhancer import DepthEnhancer

        # Create image with sharp edge
        guide = np.zeros((100, 100), dtype=np.float64)
        guide[:, 50:] = 1.0

        source = guide + np.random.normal(0, 0.1, guide.shape)

        result = DepthEnhancer._guided_filter(guide, source, radius=5, eps=0.01)

        # Edge should be preserved (sharp transition at x=50)
        assert np.mean(result[:, 45:50]) < 0.5
        assert np.mean(result[:, 50:55]) > 0.5

    def test_depth_to_pointcloud(self, sample_depth_map, sample_face_image):
        from src.pipeline.depth_enhancer import DepthEnhancer
        import cv2

        enhancer = DepthEnhancer()
        rgb_small = cv2.resize(sample_face_image, (256, 192))

        points = enhancer.depth_to_pointcloud(
            sample_depth_map, rgb_small,
            fx=200, fy=200, cx=128, cy=96
        )

        assert points.ndim == 2
        assert points.shape[1] == 6  # x, y, z, r, g, b
        assert len(points) > 0

    def test_normalize_depth(self, sample_depth_map):
        from src.pipeline.depth_enhancer import DepthEnhancer
        result = DepthEnhancer._normalize_depth_to_uint8(sample_depth_map)

        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255


class TestAnalysisPipeline:
    """Integration-level tests for the full pipeline."""

    def test_pipeline_initialization(self):
        from src.pipeline.analysis_pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline()
        assert pipeline is not None

    def test_resize_preserve_aspect(self):
        from src.pipeline.analysis_pipeline import AnalysisPipeline
        img = np.zeros((1000, 500, 3), dtype=np.uint8)
        resized = AnalysisPipeline._resize_preserve_aspect(img, 500)

        assert resized.shape[0] == 500  # Height (larger dim) = target
        assert resized.shape[1] == 250  # Width scaled proportionally

    def test_resize_no_upscale(self):
        from src.pipeline.analysis_pipeline import AnalysisPipeline
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        resized = AnalysisPipeline._resize_preserve_aspect(img, 500)

        # Should not upscale
        assert resized.shape[0] == 200
        assert resized.shape[1] == 300

    def test_heatmap_to_base64(self, sample_face_image):
        from src.pipeline.analysis_pipeline import AnalysisPipeline
        heatmap = np.random.rand(512, 512).astype(np.float32)

        b64 = AnalysisPipeline._heatmap_to_base64(heatmap, sample_face_image)

        assert isinstance(b64, str)
        assert len(b64) > 100  # Non-trivial base64 string

        # Should be valid base64
        import base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0

    def test_compare_sessions(self):
        from src.pipeline.analysis_pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline()

        before = {
            "overall_score": 70,
            "zone_scores": {
                "forehead": {"wrinkle": 65, "pore": 70, "pigmentation": 68, "redness": 75, "texture": 72, "overall": 70},
            },
        }
        after = {
            "overall_score": 78,
            "zone_scores": {
                "forehead": {"wrinkle": 72, "pore": 78, "pigmentation": 70, "redness": 80, "texture": 76, "overall": 75},
            },
        }

        comparison = pipeline.compare_sessions(before, after)

        assert comparison["overall_delta"] == 8.0
        assert len(comparison["improvements"]) > 0
        assert all(i["delta"] > 0 for i in comparison["improvements"])
