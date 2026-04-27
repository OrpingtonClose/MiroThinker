"""Bootstrap actor — drives the orchestrator state machine to completion.

This is a temporary stand-in while real swarm / flock / MCP actors are being
wired up.  It receives the initial ``ingest`` event and synthesises the
sequence of events that the orchestrator expects:

1. ``StoreDelta``   → SWARMING
2. ``SwarmComplete`` → FLOCKING
3. ``FlockComplete`` (no directions) → SYNTHESIZING
4. ``ConvergenceDetected`` → CONVERGED

All actions are traced so the bootstrap behaviour is fully observable.
"""
from __future__ import annotations

import asyncio

from universal_store.actors.base import Actor
from universal_store.protocols import (
    ConvergenceDetected,
    Event,
    FlockComplete,
    StoreDelta,
    SwarmComplete,
)
from universal_store.trace import TraceStore, trace_block


class BootstrapActor(Actor):
    """Mock actor that walks the orchestrator through every phase."""

    def __init__(self, actor_id: str = "bootstrap") -> None:
        super().__init__(actor_id)

    async def _run(self) -> None:
        trace = await TraceStore.get()
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            async with trace_block(
                self.actor_id,
                "bootstrap_received",
                payload={"event_type": event.event_type},
            ):
                if event.event_type == "ingest":
                    await self._drive_pipeline(event, trace)
                else:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="bootstrap_ignored",
                        payload={"event_type": event.event_type},
                    )

    async def _drive_pipeline(self, event: Event, trace: TraceStore) -> None:
        """Emit the event sequence that moves the orchestrator to CONVERGED."""
        query = event.payload.get("query", "")

        # 1. Pretend we ingested some raw data
        await asyncio.sleep(0.1)
        await self.send_to_parent(
            StoreDelta(rows_added=3, row_types=["raw"])
        )

        # 2. Pretend swarm produced findings
        await asyncio.sleep(0.1)
        await self.send_to_parent(
            SwarmComplete(findings=[1, 2, 3], gaps=[])
        )

        # 3. Pretend flock converged with no new directions needed
        await asyncio.sleep(0.1)
        await self.send_to_parent(
            FlockComplete(convergence_reason="bootstrap", directions=[])
        )

        # 4. Pretend we converged
        await asyncio.sleep(0.1)
        await self.send_to_parent(
            ConvergenceDetected(layer="bootstrap", score=0.001)
        )
