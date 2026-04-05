"""
Agent 0 — Orchestrator Agent

Responsibilities:
- Start / stop all child agents in the correct order
- Maintain shared pipeline state (session, health, config)
- Supervisor loop (1 Hz) checking agent health
- Circuit breaker: if any agent fails N times → fallback to adaptive delay
- Route configuration changes to all agents
- Manage session lifecycle
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.models import (
    AgentState,
    AlertEvent,
    AlertLevel,
    PipelineConfig,
    PipelineMetrics,
)

logger = logging.getLogger("agents.orchestrator")


class OrchestratorAgent(BaseAgent):
    """
    Master coordinator for the A/V sync pipeline.

    Owns all child agents and supervises their health.
    Triggers fallback to simple adaptive-delay mode if the
    full muxing pipeline cannot maintain quality targets.
    """

    SUPERVISOR_INTERVAL_S = 1.0  # Health check every second

    def __init__(
        self,
        session_id: str,
        config: PipelineConfig,
        metrics_queue: asyncio.Queue,
        alert_queue: asyncio.Queue,
    ):
        super().__init__("orchestrator", metrics_queue)
        self.session_id = session_id
        self.config = config
        self._alert_queue = alert_queue

        # Child agents (set by pipeline.py after creation)
        self._agents: List[BaseAgent] = []

        # Pipeline state
        self._pipeline_healthy = True
        self._fallback_active = False
        self._start_time = 0.0
        self._latest_metrics: Optional[PipelineMetrics] = None

    # ── Public API ────────────────────────────────────────────

    def register_agents(self, agents: List[BaseAgent]) -> None:
        """Register child agents for supervision."""
        self._agents = list(agents)
        logger.info(
            f"[{self.session_id}] Registered {len(self._agents)} agents: "
            f"{[a.name for a in self._agents]}"
        )

    @property
    def is_pipeline_healthy(self) -> bool:
        return self._pipeline_healthy

    @property
    def is_fallback_active(self) -> bool:
        return self._fallback_active

    async def start_pipeline(self) -> None:
        """Start all agents in dependency order, then start self."""
        self._start_time = time.time()
        logger.info(f"[{self.session_id}] Starting pipeline with {len(self._agents)} agents")

        # Start agents in order: ingest first, quality last
        for agent in self._agents:
            try:
                await agent.start()
                logger.info(f"[{self.session_id}]   ✅ {agent.name} started")
            except Exception as exc:
                logger.error(f"[{self.session_id}]   ❌ {agent.name} failed to start: {exc}")
                self._pipeline_healthy = False
                await self._emergency_shutdown(f"{agent.name} failed to start")
                return

        # Start orchestrator's own supervisor loop
        await self.start()
        logger.info(f"[{self.session_id}] Pipeline fully started")

    async def stop_pipeline(self) -> None:
        """Gracefully stop all agents in reverse order."""
        logger.info(f"[{self.session_id}] Stopping pipeline")

        # Stop self first (supervisor loop)
        await self.stop()

        # Stop agents in reverse order (quality → encode → align → buffer → swap → ingest)
        for agent in reversed(self._agents):
            try:
                await agent.stop()
                logger.info(f"[{self.session_id}]   ⏹  {agent.name} stopped")
            except Exception as exc:
                logger.warning(f"[{self.session_id}]   ⚠️  {agent.name} stop error: {exc}")

        elapsed = time.time() - self._start_time
        logger.info(f"[{self.session_id}] Pipeline stopped after {elapsed:.1f}s")

    # ── Main loop ─────────────────────────────────────────────

    async def run(self) -> None:
        """Supervisor loop — check agent health every second."""
        logger.info(f"[{self.session_id}] Supervisor loop started")

        while not self.should_stop:
            try:
                await asyncio.sleep(self.SUPERVISOR_INTERVAL_S)
                self.heartbeat()

                # Check each agent's health
                for agent in self._agents:
                    health = agent.health()

                    # Stale heartbeat (no activity for 10s)
                    if (
                        health.state == AgentState.RUNNING
                        and health.last_heartbeat > 0
                        and time.time() - health.last_heartbeat > 10.0
                    ):
                        logger.warning(
                            f"[{self.session_id}] {agent.name} heartbeat stale "
                            f"({time.time() - health.last_heartbeat:.1f}s)"
                        )
                        await self._handle_agent_issue(agent, "stale_heartbeat")

                    # Agent crashed
                    if health.state == AgentState.ERROR:
                        logger.error(
                            f"[{self.session_id}] {agent.name} in ERROR state "
                            f"(errors={health.consecutive_errors})"
                        )
                        await self._handle_agent_issue(agent, "error_state")

                    # Too many consecutive errors
                    if health.consecutive_errors >= self.config.max_consecutive_failures:
                        logger.error(
                            f"[{self.session_id}] {agent.name} exceeded max failures "
                            f"({health.consecutive_errors})"
                        )
                        await self._handle_agent_issue(agent, "max_failures")

                # Process alerts from Agent 6
                await self._process_alerts()

                # Report supervisor metrics
                await self.report_metric("pipeline_healthy", 1.0 if self._pipeline_healthy else 0.0)
                await self.report_metric("fallback_active", 1.0 if self._fallback_active else 0.0)
                await self.report_metric("uptime_s", time.time() - self._start_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception(f"[{self.session_id}] Supervisor error: {exc}")

    # ── Internal ──────────────────────────────────────────────

    async def _handle_agent_issue(self, agent: BaseAgent, reason: str) -> None:
        """Handle an unhealthy agent — restart or trigger fallback."""
        logger.warning(
            f"[{self.session_id}] Handling issue for {agent.name}: {reason}"
        )

        # Try to restart the agent first
        try:
            await agent.stop()
            await asyncio.sleep(0.5)
            await agent.start()
            logger.info(f"[{self.session_id}] {agent.name} restarted successfully")
            return
        except Exception as exc:
            logger.error(f"[{self.session_id}] {agent.name} restart failed: {exc}")

        # If restart fails and it's a critical agent, trigger fallback
        critical_agents = {"ingest", "face_swap", "pts_align", "mux_encode"}
        if agent.name in critical_agents and self.config.fallback_to_adaptive_delay:
            await self._activate_fallback(
                f"{agent.name} unrecoverable ({reason})"
            )

    async def _activate_fallback(self, reason: str) -> None:
        """Switch from full muxing pipeline to simple adaptive delay."""
        if self._fallback_active:
            return

        logger.warning(
            f"[{self.session_id}] ⚠️ ACTIVATING FALLBACK — reason: {reason}"
        )
        self._fallback_active = True
        self._pipeline_healthy = False

        await self.report_metric("fallback_activated", 1.0, tags={"reason": reason})

    async def _process_alerts(self) -> None:
        """Drain alerts from Agent 6 and act on critical ones."""
        alerts_processed = 0
        while not self._alert_queue.empty() and alerts_processed < 10:
            try:
                alert: AlertEvent = self._alert_queue.get_nowait()
                alerts_processed += 1

                if alert.level == AlertLevel.CRITICAL:
                    logger.error(
                        f"[{self.session_id}] 🔴 CRITICAL: {alert.message} "
                        f"({alert.metric_name}={alert.current_value:.1f}, "
                        f"threshold={alert.threshold:.1f})"
                    )
                    # Multiple critical alerts → consider fallback
                    if alerts_processed >= 3:
                        await self._activate_fallback("Multiple critical alerts")

                elif alert.level == AlertLevel.WARNING:
                    logger.warning(
                        f"[{self.session_id}] 🟡 WARNING: {alert.message}"
                    )

            except asyncio.QueueEmpty:
                break

    async def _emergency_shutdown(self, reason: str) -> None:
        """Emergency stop — kill everything."""
        logger.error(f"[{self.session_id}] EMERGENCY SHUTDOWN: {reason}")
        self._pipeline_healthy = False

        for agent in reversed(self._agents):
            try:
                await agent.stop()
            except Exception:
                pass
