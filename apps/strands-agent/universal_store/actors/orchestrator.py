"""Orchestrator actor — the root of the supervision tree.

Manages the 9-state machine and routes events from all child supervisors
according to the current phase.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator

from universal_store.actors.base import Actor
from universal_store.actors.supervisor import RootSupervisor
from universal_store.config import UnifiedConfig
from universal_store.protocols import (
    BudgetOverride,
    ConvergenceDetected,
    Event,
    FlockComplete,
    McpResearchComplete,
    OrchestratorEvent,
    OrchestratorPhase,
    StoreDelta,
    SwarmComplete,
    SwarmPhaseComplete,
    UserInterrupt,
)
from universal_store.trace import TraceStore, trace_block


class OrchestratorActor(RootSupervisor):
    """Root actor that owns the 9-phase state machine.

    Receives events from all child supervisors, transitions phases,
    and emits :class:`OrchestratorEvent` instances for UI consumption.
    """

    def __init__(
        self,
        actor_id: str = "orchestrator",
        config: UnifiedConfig | None = None,
    ) -> None:
        super().__init__(actor_id)
        self.phase: OrchestratorPhase = OrchestratorPhase.IDLE
        self.config = config or UnifiedConfig.from_env()
        self.output_queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue(
            maxsize=self.config.actor.mailbox_queue_size
        )
        self._query: str = ""
        self._run_id: str = ""
        self._previous_phase: OrchestratorPhase = OrchestratorPhase.IDLE
        self._pending_gaps: list[str] = []
        self._paused_children: list[str] = []
        # Register bootstrap child so the pipeline can complete end-to-end
        from universal_store.actors.bootstrap import BootstrapActor
        self.register_child("bootstrap", lambda: BootstrapActor())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit(
        self,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Put an :class:`OrchestratorEvent` onto ``self.output_queue``."""
        event = OrchestratorEvent(
            phase=self.phase,
            message=message,
            data=data or {},
            trace_id=self._run_id,
        )
        try:
            self.output_queue.put_nowait(event)
        except asyncio.QueueFull:
            trace = await self._ensure_trace()
            await trace.record(
                actor_id=self.actor_id,
                event_type="output_queue_dropped",
                phase=self.phase,
                payload={"message": message},
            )

    async def _transition_to(
        self,
        new_phase: OrchestratorPhase,
        message: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Trace and broadcast a phase transition."""
        if self.phase == new_phase:
            return

        old_phase = self.phase
        payload = {
            "from_phase": old_phase,
            "to_phase": new_phase,
            "message": message,
            **(data or {}),
        }

        async with trace_block(
            self.actor_id,
            "phase_transition",
            phase=old_phase,
            payload=payload,
        ):
            self.phase = new_phase
            await self._emit(
                f"Transition {old_phase} → {new_phase}"
                + (f": {message}" if message else ""),
                payload,
            )

    # ------------------------------------------------------------------
    # Child supervision override
    # ------------------------------------------------------------------

    async def _handle_child_crash(self, child_id: str) -> None:
        """Override to emit an event before applying the restart strategy."""
        await self._emit(
            "Child crash detected",
            {"child_id": child_id, "strategy": self.strategy},
        )
        await super()._handle_child_crash(child_id)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _on_ingest_query(self, event: Event) -> None:
        """Bootstrap the orchestrator from an IngestQuery event sent by the entrypoint."""
        query = event.payload.get("query", self._query)
        run_id = event.payload.get("run_id", self._run_id)
        if query:
            self._query = query
        if run_id:
            self._run_id = run_id
        if self.phase != OrchestratorPhase.IDLE:
            await self._emit(
                f"IngestQuery received in non-IDLE phase {self.phase!r}; ignoring duplicate",
                {"phase": str(self.phase)},
            )
            return
        await self.spawn_all_registered()
        await self._transition_to(
            OrchestratorPhase.INGESTING,
            message=f"IngestQuery received; starting ingestion for: {self._query}",
            data={"query": self._query, "run_id": self._run_id},
        )
        await self.broadcast_to_children(
            Event("ingest", {"query": self._query, "run_id": self._run_id})
        )

    async def _on_user_interrupt(self, event: Event) -> None:
        action = event.payload.get("action", "pause")
        message = event.payload.get("message", "")

        if action == "pause":
            self._previous_phase = self.phase
            await self._transition_to(
                OrchestratorPhase.USER_INTERRUPTION,
                message=f"User pause: {message}",
            )
            self._paused_children = []
            for child_id, child in list(self._children.items()):
                await child.stop(graceful=True)
                self._paused_children.append(child_id)
            await self._emit(
                "Children paused",
                {"paused_children": self._paused_children},
            )

        elif action == "stop":
            await self._transition_to(
                OrchestratorPhase.ERROR,
                message=f"User stop: {message}",
            )
            await self.stop(graceful=True)

        elif action == "inject":
            target_phase = self._previous_phase
            await self._transition_to(
                target_phase,
                message=f"User resume/inject: {message}",
            )
            for child_id in self._paused_children:
                factory = self._child_specs.get(child_id)
                if factory:
                    new_child = factory()
                    self.spawn_child(new_child)
            self._paused_children.clear()
            await self._emit("Children resumed")

        else:
            await self._emit(
                f"Unknown UserInterrupt action: {action}",
                {"action": action, "message": message},
            )

    async def _on_convergence_detected(self, event: Event) -> None:
        layer = event.payload.get("layer", "unknown")
        score = event.payload.get("score", 0.0)

        if self._pending_gaps:
            await self._transition_to(
                OrchestratorPhase.SWARMING,
                message=f"Gaps remain ({len(self._pending_gaps)}); restarting swarm",
                data={"layer": layer, "score": score, "gaps": self._pending_gaps},
            )
            await self.broadcast_to_children(
                Event(
                    "swarm_restart",
                    {"reason": "gaps_remain", "gaps": self._pending_gaps},
                )
            )
        else:
            await self._transition_to(
                OrchestratorPhase.CONVERGED,
                message=f"Converged at layer {layer} with score {score}",
                data={"layer": layer, "score": score},
            )
            # Graceful shutdown after convergence so the event stream terminates
            # Use create_task to avoid deadlocking: stop() awaits self._task
            # which is the task currently running this handler.
            asyncio.create_task(self.stop(graceful=True))

    async def _on_store_delta(self, event: Event) -> None:
        row_types = event.payload.get("row_types", [])
        rows_added = event.payload.get("rows_added", 0)

        if "raw" in row_types and self.phase in {
            OrchestratorPhase.IDLE,
            OrchestratorPhase.CONVERGED,
        }:
            await self._transition_to(
                OrchestratorPhase.SWARMING,
                message=f"New raw data ({rows_added} rows); restarting swarm",
                data={"rows_added": rows_added, "row_types": row_types},
            )
            await self.broadcast_to_children(
                Event(
                    "swarm_restart",
                    {"reason": "new_raw_data", "rows_added": rows_added},
                )
            )
        else:
            await self._emit(
                "Store delta received",
                {"rows_added": rows_added, "row_types": row_types},
            )

    async def _on_flock_complete(self, event: Event) -> None:
        directions = event.payload.get("directions", [])
        convergence_reason = event.payload.get("convergence_reason", "")

        if directions:
            await self._transition_to(
                OrchestratorPhase.FETCHING_EXTERNAL,
                message=f"Flock complete; {len(directions)} research directions",
                data={
                    "directions": directions,
                    "convergence_reason": convergence_reason,
                },
            )
            research_event = Event(
                "research_directions",
                {
                    "directions": directions,
                    "convergence_reason": convergence_reason,
                },
            )
            mcp_child = self._children.get("mcp_researcher")
            if mcp_child is not None:
                await mcp_child.send(research_event)
            else:
                await self.broadcast_to_children(research_event)
        else:
            await self._transition_to(
                OrchestratorPhase.SYNTHESIZING,
                message=f"Flock complete; no directions needed: {convergence_reason}",
                data={"convergence_reason": convergence_reason},
            )
            await self.broadcast_to_children(
                Event("synthesize", {"reason": convergence_reason})
            )

    async def _on_swarm_complete(self, event: Event) -> None:
        findings = event.payload.get("findings", [])
        gaps = event.payload.get("gaps", [])
        self._pending_gaps = gaps

        await self._transition_to(
            OrchestratorPhase.FLOCKING,
            message=f"Swarm complete; {len(findings)} findings, {len(gaps)} gaps",
            data={"findings_count": len(findings), "gaps": gaps},
        )
        await self.broadcast_to_children(
            Event("flock_start", {"findings": findings, "gaps": gaps})
        )

    async def _on_mcp_research_complete(self, event: Event) -> None:
        findings_added = event.payload.get("findings_added", 0)
        cost_usd = event.payload.get("cost_usd", 0.0)
        source_type = event.payload.get("source_type", "unknown")

        await self._transition_to(
            OrchestratorPhase.SWARMING,
            message=f"External research complete; {findings_added} findings added",
            data={
                "findings_added": findings_added,
                "cost_usd": cost_usd,
                "source_type": source_type,
            },
        )
        await self.broadcast_to_children(
            Event(
                "swarm_restart",
                {
                    "reason": "external_research_complete",
                    "new_findings": findings_added,
                },
            )
        )

    async def _on_swarm_phase_complete(self, event: Event) -> None:
        phase = event.payload.get("phase", "")

        if self.phase == OrchestratorPhase.INGESTING:
            await self._transition_to(
                OrchestratorPhase.SWARMING,
                message=f"Ingestion complete; swarm phase {phase} starting",
                data={"swarm_phase": phase},
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        trace = await self._ensure_trace()
        await trace.set_run(self._run_id)

        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
            except asyncio.TimeoutError:
                await self._health_check_all()
                continue

            await trace.record(
                actor_id=self.actor_id,
                event_type="orchestrator_received",
                phase=self.phase,
                payload={
                    "event_type": event.event_type,
                    "source": event.source_actor,
                },
            )

            try:
                async with trace_block(
                    self.actor_id,
                    "orchestrator_handle_event",
                    phase=self.phase,
                    payload={
                        "event_type": event.event_type,
                        "source": event.source_actor,
                    },
                ):
                    if event.event_type == "IngestQuery":
                        await self._on_ingest_query(event)
                    elif event.event_type == "UserInterrupt":
                        await self._on_user_interrupt(event)
                    elif event.event_type == "ConvergenceDetected":
                        await self._on_convergence_detected(event)
                    elif event.event_type == "StoreDelta":
                        await self._on_store_delta(event)
                    elif event.event_type == "FlockComplete":
                        await self._on_flock_complete(event)
                    elif event.event_type == "SwarmComplete":
                        await self._on_swarm_complete(event)
                    elif event.event_type == "McpResearchComplete":
                        await self._on_mcp_research_complete(event)
                    elif event.event_type == "SwarmPhaseComplete":
                        await self._on_swarm_phase_complete(event)
                    elif event.event_type == "actor_crashed":
                        await self._handle_child_crash(
                            event.payload.get("actor_id", event.source_actor)
                        )
                    else:
                        await self._emit(
                            f"Unhandled event: {event.event_type}",
                            {
                                "source": event.source_actor,
                                "payload": event.payload,
                            },
                        )
            except Exception as e:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="orchestrator_event_error",
                    phase=self.phase,
                    error=e,
                )
                await self._transition_to(
                    OrchestratorPhase.ERROR,
                    message=f"Error handling {event.event_type}: {e!s}",
                    data={
                        "event_type": event.event_type,
                        "source": event.source_actor,
                    },
                )
                # Exception is traced and emitted — not swallowed silently.
                # We continue the loop so the orchestrator stays alive.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def events(self) -> AsyncIterator[OrchestratorEvent]:
        """Async iterator yielding :class:`OrchestratorEvent` for UI consumption."""
        while True:
            if self._shutdown and self.output_queue.empty():
                return
            try:
                event = await asyncio.wait_for(self.output_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

    def run(self, query: str) -> asyncio.Queue[OrchestratorEvent]:
        """Set up the tree, start children, and begin the orchestrator loop.

        Args:
            query: The research query that drives this run.

        Returns:
            The output queue of :class:`OrchestratorEvent` instances.
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("Orchestrator is already running")

        self._query = query
        self._run_id = str(uuid.uuid4())[:16]

        async def _bootstrap() -> None:
            try:
                await self.spawn_all_registered()
                trace = await self._ensure_trace()
                await trace.set_run(self._run_id)
                await self._transition_to(
                    OrchestratorPhase.INGESTING,
                    message=f"Starting run for query: {self._query}",
                    data={"query": self._query, "run_id": self._run_id},
                )
                await self.broadcast_to_children(
                    Event("ingest", {"query": self._query, "run_id": self._run_id})
                )
            except Exception as e:
                trace = await self._ensure_trace()
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="bootstrap_error",
                    phase=self.phase,
                    error=e,
                )
                await self._transition_to(
                    OrchestratorPhase.ERROR,
                    message=f"Bootstrap failed: {e!s}",
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "OrchestratorActor.run() must be called within a running event loop"
            )

        loop.create_task(_bootstrap())
        self.start()
        return self.output_queue
