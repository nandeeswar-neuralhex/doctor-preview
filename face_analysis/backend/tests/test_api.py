"""
API Integration Tests — Tests all REST endpoints.
Uses httpx TestClient for async FastAPI testing.
"""

import io
import json

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

# We need to import after conftest sets up the path
from src.server import app


@pytest.fixture
def sample_image_bytes():
    """Create a JPEG image in memory for upload testing."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 180
    cv2.ellipse(img, (256, 256), (150, 200), 0, 0, 360, (210, 185, 170), -1)
    cv2.circle(img, (200, 220), 20, (80, 60, 50), -1)
    cv2.circle(img, (312, 220), 20, (80, 60, 50), -1)

    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()


@pytest.fixture
def sample_depth_bytes():
    """Create a depth map binary for upload testing."""
    depth = np.ones((192, 256), dtype=np.float32) * 0.3
    return depth.tobytes()


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_check(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "uptime_seconds" in data


@pytest.mark.asyncio
class TestAnalyzeEndpoint:
    async def test_single_analysis(self, sample_image_bytes):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/analyze",
                files={"image": ("face.jpg", sample_image_bytes, "image/jpeg")},
                data={"patient_name": "Test Patient", "patient_age": "30", "angle": "front_0"},
            )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "session_id" in data
        assert "overall_score" in data or "error" in data
        assert "processing_time_ms" in data

    async def test_multi_analysis(self, sample_image_bytes):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/analyze/multi",
                files={
                    "front": ("front.jpg", sample_image_bytes, "image/jpeg"),
                    "left_45": ("left.jpg", sample_image_bytes, "image/jpeg"),
                },
                data={"patient_name": "Multi Test"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    async def test_invalid_image(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/analyze",
                files={"image": ("bad.jpg", b"not an image", "image/jpeg")},
            )

        assert response.status_code == 400


@pytest.mark.asyncio
class TestSessionEndpoints:
    async def test_list_sessions(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/sessions")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    async def test_get_nonexistent_session(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/sessions/nonexistent-id")

        assert response.status_code == 404


@pytest.mark.asyncio
class TestReportEndpoints:
    async def test_report_for_nonexistent_session(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/sessions/nonexistent/report")

        assert response.status_code == 404

    async def test_full_flow_analyze_then_report(self, sample_image_bytes):
        """End-to-end: analyze → get session → generate report."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Step 1: Analyze
            resp1 = await client.post(
                "/analyze",
                files={"image": ("face.jpg", sample_image_bytes, "image/jpeg")},
                data={"patient_name": "E2E Test", "patient_age": "35"},
            )
            assert resp1.status_code == 200
            session_id = resp1.json().get("session_id")

            if session_id:
                # Step 2: Get session
                resp2 = await client.get(f"/sessions/{session_id}")
                assert resp2.status_code == 200

                # Step 3: Get report
                resp3 = await client.get(f"/sessions/{session_id}/report")
                assert resp3.status_code == 200
                assert "<!DOCTYPE html>" in resp3.text


@pytest.mark.asyncio
class TestCompareEndpoint:
    async def test_compare_nonexistent(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/compare/fake1/fake2")

        assert response.status_code == 404


@pytest.mark.asyncio
class TestValidateEndpoint:
    async def test_validate_single_photo(self, sample_image_bytes):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/validate",
                files=[("images", ("front.jpg", sample_image_bytes, "image/jpeg"))],
                data={"angles": "front_0"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "results" in data
        assert "passed" in data["results"][0]
        assert "quality_score" in data["results"][0]
        assert "face_detected" in data["results"][0]
        assert "issues" in data["results"][0]

    async def test_validate_multiple_photos(self, sample_image_bytes):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/validate",
                files=[
                    ("images", ("front.jpg", sample_image_bytes, "image/jpeg")),
                    ("images", ("left45.jpg", sample_image_bytes, "image/jpeg")),
                    ("angles", (None, "front_0")),
                    ("angles", (None, "left_45")),
                ],
            )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        assert "overall_passed" in data

    async def test_validate_invalid_image(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/validate",
                files=[("images", ("bad.jpg", b"not an image", "image/jpeg"))],
                data={"angles": "front_0"},
            )
        assert response.status_code == 400
