"""
Base Agent class for the A/V Sync Pipeline.

All agents inherit from this. Provides:
- Lifecycle management (start / stop / health)
- Logging with agent name prefix
- Metrics reporting to the shared metrics queue
- Error counting for circuit breaker
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from agents.models import AgentState, AgentHealth, MetricEvent

logger = logging.getLogger("agents")


class BaseAgent(ABC):
    """Abstract base for all pipeline agents."""

    def __init__(self, name: str, metrics_queue: Optional[asyncio.Queue] = None):
        self.name = name
        self.state = AgentState.IDLE
        self._metrics_queue = metrics_queue
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Health tracking
        self._last_heartbeat = 0.0
        self._consecutive_errors = 0
        self._total_processed = 0
        self._latency_sum = 0.0
        self._latency_count = 0

        self._logger = logging.getLogger(f"agents.{name}")

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        """Start the agent's main loop as an asyncio task."""
        if self.state == AgentState.RUNNING:
            self._logger.warning(f"{self.name} already running")
            return

        self._stop_event.clear()
        self.state = AgentState.RUNNING
        self._last_heartbeat = time.time()
        self._task = asyncio.create_task(self._run_wrapper(), name=f"agent-{self.name}")
        self._logger.info(f"{self.name} started")

    async def stop(self) -> None:
        """Signal the agent to stop and wait for cleanup."""
        if self.state in (AgentState.STOPPED, AgentState.STOPPING):
            return

        self.state = AgentState.STOPPING
        self._stop_event.set()

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        await self.cleanup()
        self.state = AgentState.STOPPED
        self._logger.info(f"{self.name} stopped")

    @property
    def is_running(self) -> bool:
        return self.state == AgentState.RUNNING

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def health(self) -> AgentHealth:
        """Return current health snapshot."""
        avg_latency = (
            self._latency_sum / self._latency_count
            if self._latency_count > 0
            else 0.0
        )
        return AgentHealth(
            agent_name=self.name,
            state=self.state,
            last_heartbeat=self._last_heartbeat,
            consecutive_errors=self._consecutive_errors,
            total_processed=self._total_processed,
            avg_latency_ms=avg_latency,
        )

    # ── Abstract methods ──────────────────────────────────────

    @abstractmethod
    async def run(self) -> None:
        """Main processing loop. Must check self.should_stop periodically."""
        ...

    async def cleanup(self) -> None:
        """Override for resource cleanup on shutdown."""
        pass

    # ── Internal ──────────────────────────────────────────────

    async def _run_wrapper(self) -> None:
        """Wraps run() with error handling and heartbeat updates."""
        try:
            await self.run()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._consecutive_errors += 1
            self.state = AgentState.ERROR
            self._logger.exception(f"{self.name} crashed: {exc}")
            await self.report_metric("agent_error", 1.0, tags={"error": str(exc)})

    # ── Helpers ───────────────────────────────────────────────

    def heartbeat(self) -> None:
        """Update heartbeat timestamp — call in main loop."""
        self._last_heartbeat = time.time()

    def record_processed(self, latency_ms: float) -> None:
        """Record a successful processing event."""
        self._total_processed += 1
        self._latency_sum += latency_ms
        self._latency_count += 1
        self._consecutive_errors = 0
        self.heartbeat()

    async def report_metric(
        self,
        name: str,
        value: float,
        tags: Optional[dict] = None,
    ) -> None:
        """Send a metric event to Agent 6 (Quality Monitor)."""
        if self._metrics_queue is None:
            return
        event = MetricEvent(
            agent_name=self.name,
            metric_name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
        )
        try:
            self._metrics_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop metric rather than block pipeline
