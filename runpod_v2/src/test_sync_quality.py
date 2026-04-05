#!/usr/bin/env python3
"""
A/V Sync Pipeline Quality Test

Simulates video+audio frames flowing through the 7-agent pipeline
and measures Audio-Video Offset (AVO), buffer health, and throughput.

Usage:
    # Quick test (5 seconds, no GPU)
    python test_sync_quality.py

    # Extended test with custom duration
    python test_sync_quality.py --duration 30

    # Test against a running server
    python test_sync_quality.py --server http://localhost:8765 --session test-session

Requirements:
    pip install numpy aiohttp
    (For server mode: the server must be running with ENABLE_AV_SYNC_PIPELINE=true)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_sync")


# ─────────────────────────────────────────────────────────────
# Test Result Model
# ─────────────────────────────────────────────────────────────

@dataclass
class SyncTestResult:
    """Aggregated sync quality metrics from the test run."""
    duration_s: float = 0.0
    frames_produced: int = 0
    frames_received: int = 0
    frame_drop_pct: float = 0.0

    # AVO stats (milliseconds)
    avo_mean_ms: float = 0.0
    avo_p50_ms: float = 0.0
    avo_p95_ms: float = 0.0
    avo_p99_ms: float = 0.0
    avo_max_ms: float = 0.0

    # Throughput
    input_fps: float = 0.0
    output_fps: float = 0.0

    # Buffer health
    buffer_fill_avg_pct: float = 0.0
    buffer_underflows: int = 0
    buffer_overflows: int = 0

    # Swap performance
    swap_latency_mean_ms: float = 0.0
    swap_latency_p95_ms: float = 0.0

    # Overall
    e2e_latency_ms: float = 0.0
    pipeline_healthy: bool = True
    passed: bool = True
    failure_reasons: list = None

    def __post_init__(self):
        if self.failure_reasons is None:
            self.failure_reasons = []


# ─────────────────────────────────────────────────────────────
# Local Pipeline Test (no server needed)
# ─────────────────────────────────────────────────────────────

class MockFaceSwapper:
    """Simulates face swap with configurable latency."""

    def __init__(self, latency_ms: float = 15.0, jitter_ms: float = 5.0):
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.model_loaded = True

    def swap_face(self, source_frame, target_face=None, session_id=None,
                  audio_buffer=None, lipsync_enabled=False, **kwargs):
        """Simulate face swap with realistic timing."""
        delay = (self.latency_ms + np.random.uniform(-self.jitter_ms, self.jitter_ms)) / 1000.0
        time.sleep(max(0.001, delay))
        # Return frame unchanged (swap simulation)
        return source_frame

    def get_target_face(self, session_id: str):
        """Return a mock target face."""
        return "mock_target_face"

    def cleanup_session(self, session_id: str):
        pass


class MockVideoTrack:
    """Simulates a WebRTC video track producing frames at target FPS."""

    kind = "video"

    def __init__(self, fps: int = 30, width: int = 640, height: int = 480):
        self.fps = fps
        self.width = width
        self.height = height
        self._frame_count = 0
        self._start_time = None
        self._stopped = False

    async def recv(self):
        """Produce a synthetic video frame at the target frame rate."""
        if self._stopped:
            raise Exception("Track stopped")

        if self._start_time is None:
            self._start_time = time.time()

        # Pace frame delivery to match target FPS
        expected_time = self._start_time + (self._frame_count / self.fps)
        now = time.time()
        if expected_time > now:
            await asyncio.sleep(expected_time - now)

        self._frame_count += 1

        # Create a synthetic av.VideoFrame-like object
        return MockAVVideoFrame(
            width=self.width,
            height=self.height,
            frame_id=self._frame_count,
            pts=int(self._frame_count * (90000 / self.fps)),  # 90kHz timebase
            time=time.time(),
        )

    def stop(self):
        self._stopped = True


class MockAudioTrack:
    """Simulates a WebRTC audio track producing Opus-like chunks."""

    kind = "audio"

    def __init__(self, sample_rate: int = 48000, chunk_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self._chunk_count = 0
        self._start_time = None
        self._stopped = False

    async def recv(self):
        """Produce a synthetic audio chunk at the standard 20ms Opus cadence."""
        if self._stopped:
            raise Exception("Track stopped")

        if self._start_time is None:
            self._start_time = time.time()

        expected_time = self._start_time + (self._chunk_count * self.chunk_duration_ms / 1000.0)
        now = time.time()
        if expected_time > now:
            await asyncio.sleep(expected_time - now)

        self._chunk_count += 1
        samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)

        return MockAVAudioFrame(
            samples=np.random.uniform(-0.1, 0.1, samples_per_chunk).astype(np.float32),
            sample_rate=self.sample_rate,
            pts=int(self._chunk_count * samples_per_chunk),
            time=time.time(),
        )

    def stop(self):
        self._stopped = True


@dataclass
class MockAVVideoFrame:
    """Mimics an av.VideoFrame for the IngestAgent."""
    width: int
    height: int
    frame_id: int
    pts: int
    time: float
    key_frame: bool = False

    def to_ndarray(self, format="bgr24"):
        """Return a synthetic BGR image."""
        return np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


@dataclass
class MockAVAudioFrame:
    """Mimics an av.AudioFrame for the IngestAgent."""
    samples: np.ndarray
    sample_rate: int
    pts: int
    time: float

    @property
    def format(self):
        class _Fmt:
            name = "flt"
        return _Fmt()

    @property
    def layout(self):
        class _Layout:
            name = "mono"
        return _Layout()

    def to_ndarray(self):
        return self.samples.reshape(1, -1)


async def run_local_pipeline_test(
    duration_s: float = 5.0,
    target_fps: int = 30,
    swap_latency_ms: float = 15.0,
) -> SyncTestResult:
    """
    Run the full 7-agent pipeline locally with mock tracks.

    This test does NOT require a GPU or running server — it uses MockFaceSwapper
    to simulate face-swap latency and verifies that the PTS alignment agent
    correctly matches audio to video within broadcast-standard tolerances.
    """
    # Import pipeline components (same code as production)
    sys.path.insert(0, ".")
    try:
        from agents.pipeline import SyncPipeline
        from agents.models import PipelineConfig
    except ImportError as exc:
        logger.error(f"Cannot import pipeline: {exc}")
        logger.error("Run this script from runpod_v2/src/")
        return SyncTestResult(passed=False, failure_reasons=[f"Import error: {exc}"])

    logger.info(f"Starting local pipeline test: duration={duration_s}s, fps={target_fps}, swap_latency={swap_latency_ms}ms")

    # Create mock components
    mock_swapper = MockFaceSwapper(latency_ms=swap_latency_ms)
    mock_video = MockVideoTrack(fps=target_fps)
    mock_audio = MockAudioTrack(sample_rate=48000, chunk_duration_ms=20)

    # Create pipeline with test config
    config = PipelineConfig(
        target_fps=target_fps,
        target_audio_sample_rate=48000,
        audio_buffer_capacity_ms=500,
        max_audio_wait_ms=20,
        drift_recalibrate_frames=300,
        avo_warning_ms=30,
        avo_critical_ms=60,
    )

    pipeline = SyncPipeline(
        session_id="test-local",
        swapper=mock_swapper,
        lip_syncer=None,
        config=config,
        target_bitrate=3_000_000,
        enable_lipsync=False,
    )

    # Set tracks and start
    pipeline.set_tracks(mock_video, mock_audio)

    try:
        await pipeline.start()
        logger.info("Pipeline started — collecting metrics...")

        # Let the pipeline run for the specified duration
        metrics_history = []
        start = time.time()

        while time.time() - start < duration_s:
            await asyncio.sleep(3.0)  # Quality monitor reports every 3s
            metrics = pipeline.get_metrics()
            if metrics:
                metrics_history.append(metrics)
                logger.info(
                    f"  AVO P95={metrics.get('avo_p95_ms', 0):.1f}ms  "
                    f"Buf={metrics.get('buffer_fill_pct', 0):.0f}%  "
                    f"Swap={metrics.get('swap_latency_ms', 0):.1f}ms  "
                    f"FPS in={metrics.get('input_video_fps', 0):.1f} out={metrics.get('output_fps', 0):.1f}"
                )

        # Collect final metrics
        final_metrics = pipeline.get_metrics()
        if final_metrics:
            metrics_history.append(final_metrics)

    finally:
        # Stop pipeline and mock tracks
        mock_video.stop()
        mock_audio.stop()
        await pipeline.stop()

    # Analyze results
    result = SyncTestResult(duration_s=duration_s)

    if not metrics_history:
        result.passed = False
        result.failure_reasons.append("No metrics collected — pipeline may have failed to produce output")
        return result

    # Use last metrics snapshot
    last = metrics_history[-1]
    result.avo_p95_ms = last.get("avo_p95_ms", 0)
    result.input_fps = last.get("input_video_fps", 0)
    result.output_fps = last.get("output_fps", 0)
    result.buffer_fill_avg_pct = last.get("buffer_fill_pct", 0)
    result.swap_latency_mean_ms = last.get("swap_latency_ms", 0)
    result.e2e_latency_ms = last.get("e2e_latency_ms", 0)
    result.frame_drop_pct = last.get("frame_drop_pct", 0)
    result.pipeline_healthy = last.get("pipeline_healthy", True)

    # Check quality thresholds (ITU-R BT.1359 broadcast standard)
    if result.avo_p95_ms > 45:
        result.passed = False
        result.failure_reasons.append(f"AVO P95 too high: {result.avo_p95_ms:.1f}ms > 45ms threshold")

    if result.frame_drop_pct > 5.0:
        result.passed = False
        result.failure_reasons.append(f"Frame drops too high: {result.frame_drop_pct:.1f}% > 5% threshold")

    if result.output_fps < target_fps * 0.7:
        result.passed = False
        result.failure_reasons.append(
            f"Output FPS too low: {result.output_fps:.1f} < {target_fps * 0.7:.1f} "
            f"(70% of target {target_fps})"
        )

    if not result.pipeline_healthy:
        result.passed = False
        result.failure_reasons.append("Pipeline reported unhealthy status")

    return result


# ─────────────────────────────────────────────────────────────
# Remote Server Test (HTTP polling)
# ─────────────────────────────────────────────────────────────

async def run_server_metrics_test(
    server_url: str,
    session_id: str,
    duration_s: float = 10.0,
    poll_interval_s: float = 3.0,
) -> SyncTestResult:
    """
    Poll the /session/{id}/sync-metrics endpoint of a running server
    and evaluate the sync quality.

    Requires: server running with ENABLE_AV_SYNC_PIPELINE=true and an active WebRTC session.
    """
    try:
        import aiohttp
    except ImportError:
        logger.error("aiohttp required for server test: pip install aiohttp")
        return SyncTestResult(passed=False, failure_reasons=["aiohttp not installed"])

    logger.info(f"Polling sync metrics from {server_url}/session/{session_id}/sync-metrics")
    logger.info(f"Duration: {duration_s}s, poll interval: {poll_interval_s}s")

    metrics_history = []
    start = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start < duration_s:
            try:
                async with session.get(
                    f"{server_url}/session/{session_id}/sync-metrics",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        metrics_history.append(data)
                        logger.info(
                            f"  AVO P95={data.get('avo_p95_ms', 0):.1f}ms  "
                            f"Buf={data.get('buffer_fill_pct', 0):.0f}%  "
                            f"Swap={data.get('swap_latency_ms', 0):.1f}ms  "
                            f"FPS={data.get('output_fps', 0):.1f}  "
                            f"Healthy={data.get('pipeline_healthy', '?')}"
                        )
                    elif resp.status == 404:
                        logger.warning("No active pipeline — is a WebRTC session running?")
                    else:
                        body = await resp.text()
                        logger.warning(f"Server returned {resp.status}: {body}")
            except Exception as exc:
                logger.warning(f"Poll failed: {exc}")

            await asyncio.sleep(poll_interval_s)

    result = SyncTestResult(duration_s=duration_s)

    if not metrics_history:
        result.passed = False
        result.failure_reasons.append("No metrics received from server")
        return result

    # Analyze collected metrics
    avo_values = [m.get("avo_p95_ms", 0) for m in metrics_history if m.get("avo_p95_ms") is not None]
    fps_values = [m.get("output_fps", 0) for m in metrics_history if m.get("output_fps") is not None]
    buf_values = [m.get("buffer_fill_pct", 0) for m in metrics_history if m.get("buffer_fill_pct") is not None]
    swap_values = [m.get("swap_latency_ms", 0) for m in metrics_history if m.get("swap_latency_ms") is not None]

    if avo_values:
        result.avo_mean_ms = np.mean(avo_values)
        result.avo_p50_ms = np.percentile(avo_values, 50)
        result.avo_p95_ms = np.percentile(avo_values, 95)
        result.avo_p99_ms = np.percentile(avo_values, 99)
        result.avo_max_ms = np.max(avo_values)

    if fps_values:
        result.output_fps = np.mean(fps_values)

    if buf_values:
        result.buffer_fill_avg_pct = np.mean(buf_values)

    if swap_values:
        result.swap_latency_mean_ms = np.mean(swap_values)
        result.swap_latency_p95_ms = np.percentile(swap_values, 95)

    last = metrics_history[-1]
    result.pipeline_healthy = last.get("pipeline_healthy", True)
    result.e2e_latency_ms = last.get("e2e_latency_ms", 0)
    result.frame_drop_pct = last.get("frame_drop_pct", 0)

    # Quality checks
    if result.avo_p95_ms > 45:
        result.passed = False
        result.failure_reasons.append(f"AVO P95 too high: {result.avo_p95_ms:.1f}ms > 45ms")

    if result.frame_drop_pct > 5.0:
        result.passed = False
        result.failure_reasons.append(f"Drops too high: {result.frame_drop_pct:.1f}%")

    if not result.pipeline_healthy:
        result.passed = False
        result.failure_reasons.append("Pipeline unhealthy")

    return result


# ─────────────────────────────────────────────────────────────
# Report Printer
# ─────────────────────────────────────────────────────────────

def print_report(result: SyncTestResult) -> None:
    """Print a formatted test report."""
    border = "═" * 60
    print(f"\n╔{border}╗")
    print(f"║{'A/V SYNC QUALITY TEST REPORT':^60}║")
    print(f"╠{border}╣")

    status = "✅ PASSED" if result.passed else "❌ FAILED"
    color_start = "\033[92m" if result.passed else "\033[91m"
    color_end = "\033[0m"
    print(f"║  Status: {color_start}{status}{color_end}")
    print(f"║  Duration: {result.duration_s:.1f}s")
    print(f"╠{border}╣")

    print(f"║  {'SYNC QUALITY':^56}  ║")
    print(f"║    AVO Mean:     {result.avo_mean_ms:>8.1f} ms                          ║")
    print(f"║    AVO P50:      {result.avo_p50_ms:>8.1f} ms                          ║")
    print(f"║    AVO P95:      {result.avo_p95_ms:>8.1f} ms  {'✅' if result.avo_p95_ms <= 45 else '❌'} (target ≤45ms)     ║")
    print(f"║    AVO P99:      {result.avo_p99_ms:>8.1f} ms                          ║")
    print(f"║    AVO Max:      {result.avo_max_ms:>8.1f} ms                          ║")
    print(f"╠{border}╣")

    print(f"║  {'THROUGHPUT':^56}  ║")
    print(f"║    Input FPS:    {result.input_fps:>8.1f}                              ║")
    print(f"║    Output FPS:   {result.output_fps:>8.1f}                              ║")
    print(f"║    Frame Drops:  {result.frame_drop_pct:>7.1f}%  {'✅' if result.frame_drop_pct <= 5 else '❌'} (target ≤5%)       ║")
    print(f"╠{border}╣")

    print(f"║  {'PIPELINE HEALTH':^56}  ║")
    print(f"║    Swap Latency: {result.swap_latency_mean_ms:>8.1f} ms                          ║")
    print(f"║    Buffer Fill:  {result.buffer_fill_avg_pct:>7.1f}%                              ║")
    print(f"║    E2E Latency:  {result.e2e_latency_ms:>8.1f} ms                          ║")
    health_str = "✅ Healthy" if result.pipeline_healthy else "❌ Unhealthy"
    print(f"║    Status:       {health_str:<42}  ║")

    if result.failure_reasons:
        print(f"╠{border}╣")
        print(f"║  {'FAILURE REASONS':^56}  ║")
        for reason in result.failure_reasons:
            print(f"║    • {reason:<53} ║")

    print(f"╚{border}╝\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test A/V sync pipeline quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick local test (no server needed)
  python test_sync_quality.py

  # Extended local test with slower face-swap
  python test_sync_quality.py --duration 20 --swap-latency 35

  # Poll a running server
  python test_sync_quality.py --server http://gpu-server:8765 --session session-12345 --duration 30
""",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds (default: 5)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--swap-latency", type=float, default=15.0, help="Simulated face-swap latency in ms (default: 15)")
    parser.add_argument("--server", type=str, default=None, help="Server URL for remote test (e.g. http://localhost:8765)")
    parser.add_argument("--session", type=str, default=None, help="Session ID for remote test")
    parser.add_argument("--json", action="store_true", help="Output result as JSON (for CI)")

    args = parser.parse_args()

    if args.server:
        # Remote server test
        if not args.session:
            parser.error("--session is required when using --server")
        result = asyncio.run(
            run_server_metrics_test(
                server_url=args.server,
                session_id=args.session,
                duration_s=args.duration,
            )
        )
    else:
        # Local pipeline test
        result = asyncio.run(
            run_local_pipeline_test(
                duration_s=args.duration,
                target_fps=args.fps,
                swap_latency_ms=args.swap_latency,
            )
        )

    if args.json:
        output = {
            "passed": result.passed,
            "duration_s": result.duration_s,
            "avo_p95_ms": round(result.avo_p95_ms, 2),
            "output_fps": round(result.output_fps, 1),
            "frame_drop_pct": round(result.frame_drop_pct, 1),
            "swap_latency_ms": round(result.swap_latency_mean_ms, 1),
            "e2e_latency_ms": round(result.e2e_latency_ms, 1),
            "pipeline_healthy": result.pipeline_healthy,
            "failure_reasons": result.failure_reasons,
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(result)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
