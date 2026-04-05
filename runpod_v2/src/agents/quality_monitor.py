"""
Agent 6 — Quality Monitor Agent

Responsibilities:
- Collect metrics from ALL other agents every second
- Compute real-time Audio-Video Offset (AVO) percentiles
- Compute Lip Sync Confidence Score (LSCS) using audio energy vs mouth motion
- Alert on threshold breaches → send AlertEvents to Orchestrator
- Store metrics for logging (and optionally external sinks)
- Report pipeline-wide health dashboard

Metric Collection:
  All agents push MetricEvent to the shared metrics_queue.
  Quality Monitor drains this queue and maintains rolling windows.

Alert Thresholds (from PipelineConfig):
  🟡 WARNING:  AVO P95 > 30ms  or  buffer fill > 85%
  🔴 CRITICAL: AVO P95 > 60ms  or  buffer underflow > 3/min
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from agents.models import (
    AlertEvent,
    AlertLevel,
    MetricEvent,
    PipelineConfig,
    PipelineMetrics,
)

logger = logging.getLogger("agents.quality_monitor")


class QualityMonitorAgent(BaseAgent):
    """
    Real-time quality monitoring and alerting for the A/V sync pipeline.

    Consumes MetricEvents from all agents, computes aggregates,
    and fires AlertEvents when quality degrades.
    """

    REPORT_INTERVAL_S = 3.0     # Log aggregated metrics every 3s
    ALERT_COOLDOWN_S = 10.0     # Don't repeat the same alert within 10s

    def __init__(
        self,
        config: PipelineConfig,
        metrics_queue: asyncio.Queue,
        alert_queue: asyncio.Queue,
        session_id: str = "",
    ):
        super().__init__("quality_monitor", metrics_queue=None)  # Don't report to self
        self.config = config
        self.session_id = session_id
        self._metrics_in = metrics_queue
        self._alert_out = alert_queue

        # Rolling metric windows: metric_name → deque of (timestamp, value)
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))

        # Latest metric values (for quick access)
        self._latest: Dict[str, float] = {}

        # Alert cooldowns: alert_key → last_fired_time
        self._alert_cooldowns: Dict[str, float] = {}

        # Aggregate stats
        self._latest_report: Optional[PipelineMetrics] = None
        self._last_report_time = 0.0

    async def run(self) -> None:
        """Main loop — drain metrics queue and evaluate alerts."""
        self._last_report_time = time.time()
        logger.info(f"[{self.session_id}] Quality Monitor started")

        while not self.should_stop:
            try:
                # Drain all available metrics (non-blocking)
                drained = 0
                while drained < 100:  # Cap per iteration
                    try:
                        event: MetricEvent = self._metrics_in.get_nowait()
                        self._ingest_metric(event)
                        drained += 1
                    except asyncio.QueueEmpty:
                        break

                # Evaluate alert conditions
                await self._evaluate_alerts()

                # Periodic report
                now = time.time()
                if now - self._last_report_time >= self.REPORT_INTERVAL_S:
                    self._build_report()
                    self._log_report()
                    self._last_report_time = now

                self.heartbeat()

                # Sleep briefly to avoid busy loop
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception(f"[{self.session_id}] Quality Monitor error: {exc}")
                await asyncio.sleep(1.0)

    # ── Metric Ingestion ──────────────────────────────────────

    def _ingest_metric(self, event: MetricEvent) -> None:
        """Store a metric event in the rolling window."""
        key = f"{event.agent_name}.{event.metric_name}"
        self._windows[key].append((event.timestamp, event.value))
        self._latest[key] = event.value

    def _get_latest(self, key: str, default: float = 0.0) -> float:
        """Get the latest value for a metric."""
        return self._latest.get(key, default)

    def _get_window_avg(self, key: str, window_s: float = 10.0) -> float:
        """Get the average of a metric over the last N seconds."""
        window = self._windows.get(key)
        if not window:
            return 0.0

        cutoff = time.time() - window_s
        values = [v for ts, v in window if ts >= cutoff]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _get_percentile(self, key: str, pct: float, window_s: float = 10.0) -> float:
        """Get a percentile of a metric over the last N seconds."""
        window = self._windows.get(key)
        if not window:
            return 0.0

        cutoff = time.time() - window_s
        values = sorted(v for ts, v in window if ts >= cutoff)
        if not values:
            return 0.0

        idx = min(int(len(values) * pct / 100.0), len(values) - 1)
        return values[idx]

    # ── Alert Evaluation ──────────────────────────────────────

    async def _evaluate_alerts(self) -> None:
        """Check all alert conditions and fire alerts if needed."""

        # AVO P95 warning
        avo_p95 = self._get_latest("pts_align.avo_p95_ms", 0.0)
        if avo_p95 > self.config.avo_critical_ms:
            await self._fire_alert(
                AlertLevel.CRITICAL,
                "pts_align",
                f"AVO P95 ({avo_p95:.1f}ms) exceeds critical threshold",
                "avo_p95_ms",
                avo_p95,
                self.config.avo_critical_ms,
            )
        elif avo_p95 > self.config.avo_warning_ms:
            await self._fire_alert(
                AlertLevel.WARNING,
                "pts_align",
                f"AVO P95 ({avo_p95:.1f}ms) exceeds warning threshold",
                "avo_p95_ms",
                avo_p95,
                self.config.avo_warning_ms,
            )

        # Audio buffer fill level
        buffer_fill = self._get_latest("audio_buffer.buffer_fill_pct", 0.0)
        if buffer_fill > self.config.buffer_fill_warning_pct:
            await self._fire_alert(
                AlertLevel.WARNING,
                "audio_buffer",
                f"Audio buffer fill ({buffer_fill:.0f}%) exceeds threshold",
                "buffer_fill_pct",
                buffer_fill,
                self.config.buffer_fill_warning_pct,
            )

        # Buffer underflows (accumulated)
        underflows = self._get_latest("audio_buffer.buffer_underflows", 0.0)
        if underflows > 3:
            await self._fire_alert(
                AlertLevel.CRITICAL,
                "audio_buffer",
                f"Audio buffer underflows ({underflows:.0f}) — audio arriving late",
                "buffer_underflows",
                underflows,
                3.0,
            )

        # Face swap latency
        swap_latency = self._get_latest("face_swap.swap_latency_ms", 0.0)
        if swap_latency > 80:
            await self._fire_alert(
                AlertLevel.WARNING,
                "face_swap",
                f"Face swap latency ({swap_latency:.0f}ms) very high",
                "swap_latency_ms",
                swap_latency,
                80.0,
            )

        # Frame drops
        frame_drops = self._get_latest("face_swap.frames_dropped", 0.0)
        if frame_drops > 0:
            await self._fire_alert(
                AlertLevel.INFO,
                "face_swap",
                f"Frames dropped: {frame_drops:.0f}",
                "frames_dropped",
                frame_drops,
                0.0,
            )

        # Silence insertions (Agent 4)
        silence = self._get_latest("pts_align.silence_insertions", 0.0)
        if silence > 5:
            await self._fire_alert(
                AlertLevel.WARNING,
                "pts_align",
                f"Silence insertions ({silence:.0f}) — audio/video desync",
                "silence_insertions",
                silence,
                5.0,
            )

    async def _fire_alert(
        self,
        level: AlertLevel,
        agent: str,
        message: str,
        metric_name: str,
        current: float,
        threshold: float,
    ) -> None:
        """Fire an alert if not in cooldown."""
        alert_key = f"{agent}.{metric_name}.{level.name}"
        now = time.time()

        if alert_key in self._alert_cooldowns:
            if now - self._alert_cooldowns[alert_key] < self.ALERT_COOLDOWN_S:
                return  # Still in cooldown

        self._alert_cooldowns[alert_key] = now

        alert = AlertEvent(
            level=level,
            agent_name=agent,
            message=message,
            metric_name=metric_name,
            current_value=current,
            threshold=threshold,
        )

        try:
            self._alert_out.put_nowait(alert)
        except asyncio.QueueFull:
            pass  # Don't block

    # ── Reporting ─────────────────────────────────────────────

    def _build_report(self) -> PipelineMetrics:
        """Build a comprehensive pipeline metrics snapshot."""
        report = PipelineMetrics(
            timestamp=time.time(),
            session_id=self.session_id,

            # Agent 1 - Ingest
            input_video_fps=self._get_latest("ingest.input_video_fps"),
            input_audio_pps=self._get_latest("ingest.input_audio_pps"),
            decode_latency_ms=self._get_window_avg("ingest.decode_latency_ms"),

            # Agent 2 - Face Swap
            swap_fps=self._get_latest("face_swap.swap_fps", 0.0),
            swap_latency_ms=self._get_window_avg("face_swap.swap_latency_ms"),
            swap_latency_stddev_ms=self._get_latest("face_swap.swap_latency_stddev_ms"),
            faces_detected_avg=self._get_window_avg("face_swap.faces_detected"),

            # Agent 3 - Audio Buffer
            buffer_fill_pct=self._get_latest("audio_buffer.buffer_fill_pct"),
            buffer_underflows=int(self._get_latest("audio_buffer.buffer_underflows")),
            buffer_overflows=int(self._get_latest("audio_buffer.buffer_overflows")),

            # Agent 4 - PTS Align
            avo_mean_ms=self._get_window_avg("pts_align.avo_ms"),
            avo_p50_ms=self._get_latest("pts_align.avo_p50_ms"),
            avo_p95_ms=self._get_latest("pts_align.avo_p95_ms"),
            avo_p99_ms=self._get_latest("pts_align.avo_p99_ms"),
            drift_correction_ms=self._get_latest("pts_align.drift_correction_ms"),
            silence_insertions=int(self._get_latest("pts_align.silence_insertions")),

            # Agent 5 - Encode
            encode_latency_ms=self._get_window_avg("mux_encode.mux_latency_ms"),
            output_fps=self._get_latest("mux_encode.output_fps"),

            # Overall
            end_to_end_latency_ms=self._compute_e2e_latency(),
            frame_drop_pct=self._compute_frame_drop_pct(),
        )

        self._latest_report = report
        return report

    def _log_report(self) -> None:
        """Log the aggregated metrics report."""
        r = self._latest_report
        if not r:
            return

        logger.info(
            f"[{self.session_id}] ═══ Quality Report ═══\n"
            f"  Input:   video={r.input_video_fps:.1f}fps  audio={r.input_audio_pps:.1f}pps\n"
            f"  Swap:    {r.swap_latency_ms:.1f}ms ±{r.swap_latency_stddev_ms:.1f}ms  "
            f"faces={r.faces_detected_avg:.1f}\n"
            f"  Buffer:  fill={r.buffer_fill_pct:.0f}%  "
            f"underflows={r.buffer_underflows}  overflows={r.buffer_overflows}\n"
            f"  AVO:     mean={r.avo_mean_ms:.1f}ms  P50={r.avo_p50_ms:.1f}ms  "
            f"P95={r.avo_p95_ms:.1f}ms  P99={r.avo_p99_ms:.1f}ms\n"
            f"  Drift:   correction={r.drift_correction_ms:+.1f}ms  "
            f"silence={r.silence_insertions}\n"
            f"  Output:  {r.output_fps:.1f}fps  "
            f"encode={r.encode_latency_ms:.1f}ms\n"
            f"  E2E:     {r.end_to_end_latency_ms:.0f}ms  "
            f"drops={r.frame_drop_pct:.1f}%"
        )

    def _compute_e2e_latency(self) -> float:
        """Estimate end-to-end latency from ingest to output."""
        decode = self._get_window_avg("ingest.decode_latency_ms")
        swap = self._get_window_avg("face_swap.swap_latency_ms")
        align = self._get_window_avg("pts_align.align_latency_ms")
        encode = self._get_window_avg("mux_encode.mux_latency_ms")
        return decode + swap + align + encode

    def _compute_frame_drop_pct(self) -> float:
        """Estimate frame drop percentage."""
        input_fps = self._get_latest("ingest.input_video_fps", 0.0)
        output_fps = self._get_latest("mux_encode.output_fps", 0.0)
        if input_fps <= 0:
            return 0.0
        return max(0.0, (1.0 - output_fps / input_fps) * 100.0)

    @property
    def latest_report(self) -> Optional[PipelineMetrics]:
        """Get the most recent pipeline metrics report."""
        return self._latest_report

    async def cleanup(self) -> None:
        """Final cleanup."""
        # Build one last report
        self._build_report()
        self._log_report()
        logger.info(f"[{self.session_id}] Quality Monitor cleaned up")
