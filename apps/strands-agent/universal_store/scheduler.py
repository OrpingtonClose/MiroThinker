"""Scheduler daemon — event loop that drives actors.

Manages the priority event queue, health monitoring, and phase transitions.
This is NOT an actor; it is the event loop that drives actors.

Generic routing is used so that no actor module (swarm, flock, etc.) is
imported directly.  Actors are registered at runtime and event types can be
mapped to actor ids via ``register_route``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from universal_store.protocols import (
    Event,
    HealthCheck,
    OrchestratorEvent,
    OrchestratorPhase,
    UserInterrupt,
)
from universal_store.trace import TraceStore
from universal_store.config import UnifiedConfig


class Scheduler:
    """Daemon scheduler that drives the priority event queue, health checks,
    and phase-guard enforcement.  Routes events to registered actors without
    blocking on their responses.
    """

    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config
        self.event_queue: asyncio.PriorityQueue[tuple[int, Event]] = asyncio.PriorityQueue(
            maxsize=config.scheduler.event_queue_size
        )
        self.orchestrator_outbox: asyncio.Queue[OrchestratorEvent] = asyncio.Queue()

        self._actors: dict[str, Any] = {}
        self._routes: dict[str, str] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._trace: TraceStore | None = None

        # Phase / convergence state
        self._current_phase: OrchestratorPhase = OrchestratorPhase.IDLE
        self._phase_start_time: float = 0.0
        self._round_count = 0
        self._convergence_stuck_rounds = 0
        self._last_convergence_score: float | None = None

        # Backpressure threshold = 80 % of queue capacity
        self._backpressure_threshold = int(config.scheduler.event_queue_size * 0.8)

        # Spam-guard flags for phase-guard decisions
        self._phase_timeout_emitted = False
        self._max_rounds_reached_emitted = False
        self._convergence_stuck_emitted = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _ensure_trace(self) -> TraceStore:
        if self._trace is None:
            self._trace = await TraceStore.get()
        return self._trace

    def _reset_phase_flags(self) -> None:
        """Reset spam-guard flags when entering a new phase."""
        self._phase_timeout_emitted = False
        self._convergence_stuck_emitted = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the scheduler loop, health monitor, and phase guard."""
        if self._running:
            return
        self._running = True
        self._phase_start_time = time.monotonic()

        trace = await self._ensure_trace()
        await trace.record(
            actor_id="scheduler",
            event_type="scheduler_start",
            phase=self._current_phase.value,
        )

        self._tasks = [
            asyncio.create_task(self._run_loop(), name="scheduler:run_loop"),
            asyncio.create_task(self._health_monitor(), name="scheduler:health"),
            asyncio.create_task(self._phase_guard(), name="scheduler:phase_guard"),
        ]

    async def stop(self) -> None:
        """Graceful shutdown. Cancels background tasks and drains the queue."""
        if not self._running:
            return
        self._running = False

        for task in self._tasks:
            task.cancel()
        self._tasks = []

        # Drain queue so waiting producers don't deadlock
        dropped = 0
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break

        trace = await self._ensure_trace()
        await trace.record(
            actor_id="scheduler",
            event_type="scheduler_stop",
            phase=self._current_phase.value,
            payload={"dropped_events": dropped},
        )

    # ------------------------------------------------------------------ #
    # Actor registration & routing
    # ------------------------------------------------------------------ #

    def register_actor(self, actor_id: str, actor: Any) -> None:
        """Register an actor so the scheduler can route events to it."""
        self._actors[actor_id] = actor

    def register_route(self, event_type: str, actor_id: str) -> None:
        """Map an ``event_type`` to a destination ``actor_id``.

        This enables generic routing without importing actor modules.
        """
        self._routes[event_type] = actor_id

    async def submit(self, event: Event, priority: int = 2) -> None:
        """Submit an event to the priority queue.

        Priority is clamped to the range **0–4** (0 = highest).
        Backpressure is applied automatically when the queue exceeds its
        threshold: lowest-priority events are dropped.
        """
        clamped = max(0, min(self.config.scheduler.priority_levels - 1, priority))
        trace = await self._ensure_trace()

        # Backpressure: if queue is over threshold, drop lowest-priority events
        if self.event_queue.qsize() >= self._backpressure_threshold:
            await self._drop_lowest_priority_events()

        try:
            self.event_queue.put_nowait((clamped, event))
            await trace.record(
                actor_id="scheduler",
                event_type="event_submitted",
                phase=self._current_phase.value,
                payload={
                    "event_type": event.event_type,
                    "priority": clamped,
                    "source_actor": event.source_actor,
                },
            )
        except asyncio.QueueFull:
            await trace.record(
                actor_id="scheduler",
                event_type="event_dropped",
                phase=self._current_phase.value,
                payload={
                    "event_type": event.event_type,
                    "priority": clamped,
                    "reason": "queue_full",
                },
            )

    async def _drop_lowest_priority_events(self) -> None:
        """Remove lowest-priority events until the queue is below threshold."""
        if self.event_queue.empty():
            return

        # Drain all items, sort by priority ascending, trim the tail
        items: list[tuple[int, Event]] = []
        while not self.event_queue.empty():
            try:
                items.append(self.event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        items.sort(key=lambda x: x[0])
        keep_count = int(self.config.scheduler.event_queue_size * 0.6)
        keep = items[:keep_count]
        drop = items[keep_count:]

        for item in keep:
            self.event_queue.put_nowait(item)

        trace = await self._ensure_trace()
        for _, event in drop:
            await trace.record(
                actor_id="scheduler",
                event_type="event_dropped_backpressure",
                phase=self._current_phase.value,
                payload={
                    "event_type": event.event_type,
                    "source_actor": event.source_actor,
                },
            )

    # ------------------------------------------------------------------ #
    # Main run loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self) -> None:
        """Main loop: pop the highest-priority event and route it."""
        while self._running:
            try:
                priority, event = await asyncio.wait_for(
                    self.event_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            trace = await self._ensure_trace()
            await trace.record(
                actor_id="scheduler",
                event_type="event_routed",
                phase=self._current_phase.value,
                payload={
                    "event_type": event.event_type,
                    "priority": priority,
                    "source_actor": event.source_actor,
                },
            )

            # Update tracking from well-known events
            self._update_tracking(event)

            # Route without blocking on actor response
            await self._route_event(event)

    def _update_tracking(self, event: Event) -> None:
        """Update round counters, convergence state, and phase from events."""
        payload = event.payload
        score: float | None = None

        # Round counting
        if event.event_type in {
            "SwarmPhaseComplete",
            "GossipRoundComplete",
            "FlockRoundComplete",
        }:
            self._round_count += 1

        # Convergence score extraction
        if event.event_type == "ConvergenceDetected":
            score = payload.get("score")
        elif event.event_type == "FlockRoundComplete":
            score = payload.get("convergence_score")
        elif event.event_type == "GossipRoundComplete":
            info_gain = payload.get("info_gain")
            if info_gain is not None:
                score = 1.0 - float(info_gain)

        if score is not None:
            if (
                self._last_convergence_score is not None
                and abs(score - self._last_convergence_score)
                < self.config.scheduler.convergence_threshold
            ):
                self._convergence_stuck_rounds += 1
            else:
                self._convergence_stuck_rounds = 0
                self._convergence_stuck_emitted = False
            self._last_convergence_score = score

        # Generic phase transition via payload
        new_phase = payload.get("phase")
        if new_phase and new_phase in {p.value for p in OrchestratorPhase}:
            if OrchestratorPhase(new_phase) != self._current_phase:
                self._current_phase = OrchestratorPhase(new_phase)
                self._phase_start_time = time.monotonic()
                self._reset_phase_flags()
                self._max_rounds_reached_emitted = False

    async def _route_event(self, event: Event) -> None:
        """Dispatch an event to the appropriate registered actor."""
        target_id: str | None = None

        # 1. Explicit target in payload
        target_id = event.payload.get("target_actor")

        # 2. Route table lookup by event_type
        if target_id is None:
            target_id = self._routes.get(event.event_type)

        # 3. Reply to source actor
        if target_id is None and event.source_actor:
            target_id = event.source_actor

        actor = self._actors.get(target_id) if target_id else None
        trace = await self._ensure_trace()

        if actor is not None:
            # Non-blocking: actor.send uses put_nowait internally
            try:
                await actor.send(event)
            except Exception as exc:
                await trace.record(
                    actor_id="scheduler",
                    event_type="route_send_error",
                    phase=self._current_phase.value,
                    payload={
                        "target_actor": target_id,
                        "event_type": event.event_type,
                    },
                    error=exc,
                )
        else:
            await trace.record(
                actor_id="scheduler",
                event_type="route_not_found",
                phase=self._current_phase.value,
                payload={
                    "event_type": event.event_type,
                    "source_actor": event.source_actor,
                    "target_actor": target_id,
                },
            )

    # ------------------------------------------------------------------ #
    # Health monitor
    # ------------------------------------------------------------------ #

    async def _health_monitor(self) -> None:
        """Query all registered actors every 10 s.

        If aggregate memory exceeds the configured threshold or GPU utilisation
        exceeds the configured threshold, an auto-pause event is submitted to
        the queue.  All results are traced.
        """
        interval = self.config.actor.health_check_interval_s
        while self._running:
            await asyncio.sleep(interval)

            trace = await self._ensure_trace()
            health_records: list[dict[str, Any]] = []

            # Query all actors concurrently with a short timeout so that one
            # slow actor cannot block the scheduler.
            if self._actors:
                coros = [
                    self._actor_health_with_timeout(aid, actor)
                    for aid, actor in self._actors.items()
                ]
                results = await asyncio.gather(*coros, return_exceptions=True)

                for (actor_id, _), result in zip(self._actors.items(), results):
                    if isinstance(result, Exception):
                        health_records.append(
                            {
                                "actor_id": actor_id,
                                "status": "failed",
                                "error": str(result),
                                "memory_mb": 0.0,
                                "gpu_util": 0.0,
                            }
                        )
                    else:
                        health_records.append(result)

            # Aggregate metrics
            total_memory_mb = sum(r.get("memory_mb", 0.0) for r in health_records)
            max_gpu_util = max(
                (r.get("gpu_util", 0.0) for r in health_records), default=0.0
            )

            await trace.record(
                actor_id="scheduler",
                event_type="health_check",
                phase=self._current_phase.value,
                payload={
                    "actor_count": len(self._actors),
                    "total_memory_mb": total_memory_mb,
                    "max_gpu_util": max_gpu_util,
                    "details": health_records,
                },
            )

            # Submit formal HealthCheck events for downstream consumers
            for rec in health_records:
                hc = HealthCheck(
                    actor_id=rec["actor_id"],
                    status=rec.get("status", "healthy"),
                    memory_mb=rec.get("memory_mb", 0.0),
                )
                await self.submit(hc, priority=3)

            # Auto-pause thresholds
            mem_threshold = self.config.scheduler.auto_pause_on_memory_mb
            gpu_threshold = self.config.scheduler.auto_pause_on_gpu_util

            if total_memory_mb > mem_threshold or max_gpu_util > gpu_threshold:
                pause_event = UserInterrupt(
                    message=(
                        f"Auto-pause triggered: memory={total_memory_mb:.0f}MB, "
                        f"gpu={max_gpu_util:.1f}%"
                    ),
                    action="pause",
                )
                await self.submit(pause_event, priority=0)

                await self.orchestrator_outbox.put(
                    OrchestratorEvent(
                        phase=self._current_phase,
                        message="Auto-pause triggered due to resource exhaustion",
                        data={
                            "memory_mb": total_memory_mb,
                            "gpu_util": max_gpu_util,
                        },
                    )
                )

                await trace.record(
                    actor_id="scheduler",
                    event_type="auto_pause_triggered",
                    phase=self._current_phase.value,
                    payload={
                        "memory_mb": total_memory_mb,
                        "gpu_util": max_gpu_util,
                    },
                )

    async def _actor_health_with_timeout(
        self, actor_id: str, actor: Any
    ) -> dict[str, Any]:
        """Call ``actor.health()`` with a 5-second timeout."""
        try:
            return await asyncio.wait_for(actor.health(), timeout=5.0)
        except asyncio.TimeoutError:
            return {
                "actor_id": actor_id,
                "status": "degraded",
                "error": "health_check_timeout",
                "memory_mb": 0.0,
                "gpu_util": 0.0,
            }

    # ------------------------------------------------------------------ #
    # Phase guard
    # ------------------------------------------------------------------ #

    async def _phase_guard(self) -> None:
        """Enforce max rounds, max time per phase, and convergence-stuck
        recovery.  Runs every second.
        """
        while self._running:
            await asyncio.sleep(1.0)

            trace = await self._ensure_trace()
            now = time.monotonic()
            elapsed = now - self._phase_start_time if self._phase_start_time else 0.0

            # --- Max time per phase ---
            max_time = self.config.scheduler.default_round_time_s
            if (
                elapsed > max_time
                and self._current_phase
                not in {
                    OrchestratorPhase.IDLE,
                    OrchestratorPhase.CONVERGED,
                    OrchestratorPhase.ERROR,
                }
                and not self._phase_timeout_emitted
            ):
                self._phase_timeout_emitted = True
                await trace.record(
                    actor_id="scheduler",
                    event_type="phase_guard_max_time",
                    phase=self._current_phase.value,
                    payload={"elapsed_s": elapsed, "max_s": max_time},
                )
                await self.submit(
                    Event(
                        event_type="PhaseTimeout",
                        payload={
                            "phase": self._current_phase.value,
                            "elapsed_s": elapsed,
                        },
                    ),
                    priority=0,
                )
                await self.orchestrator_outbox.put(
                    OrchestratorEvent(
                        phase=self._current_phase,
                        message=f"Phase {self._current_phase.value} timed out",
                        data={"elapsed_s": elapsed, "max_s": max_time},
                    )
                )

            # --- Max total rounds ---
            max_rounds = self.config.scheduler.max_total_rounds
            if (
                self._round_count >= max_rounds
                and not self._max_rounds_reached_emitted
            ):
                self._max_rounds_reached_emitted = True
                await trace.record(
                    actor_id="scheduler",
                    event_type="phase_guard_max_rounds",
                    phase=self._current_phase.value,
                    payload={"round_count": self._round_count},
                )
                await self.submit(
                    Event(
                        event_type="MaxRoundsReached",
                        payload={"round_count": self._round_count},
                    ),
                    priority=0,
                )
                await self.orchestrator_outbox.put(
                    OrchestratorEvent(
                        phase=self._current_phase,
                        message="Maximum round count reached",
                        data={"round_count": self._round_count},
                    )
                )

            # --- Convergence stuck ---
            stuck_threshold = self.config.scheduler.max_convergence_stuck_rounds
            if (
                self._convergence_stuck_rounds >= stuck_threshold
                and not self._convergence_stuck_emitted
            ):
                self._convergence_stuck_emitted = True
                await trace.record(
                    actor_id="scheduler",
                    event_type="phase_guard_convergence_stuck",
                    phase=self._current_phase.value,
                    payload={"stuck_rounds": self._convergence_stuck_rounds},
                )
                # Trigger external research (lower priority than operator alert)
                await self.submit(
                    Event(
                        event_type="ExternalResearchTrigger",
                        payload={
                            "reason": "convergence_stuck",
                            "stuck_rounds": self._convergence_stuck_rounds,
                        },
                    ),
                    priority=1,
                )
                # Operator alert (highest priority)
                await self.submit(
                    Event(
                        event_type="OperatorAlert",
                        payload={
                            "reason": "convergence_stuck",
                            "stuck_rounds": self._convergence_stuck_rounds,
                        },
                    ),
                    priority=0,
                )
                await self.orchestrator_outbox.put(
                    OrchestratorEvent(
                        phase=self._current_phase,
                        message="Convergence stuck — operator alert raised",
                        data={"stuck_rounds": self._convergence_stuck_rounds},
                    )
                )
