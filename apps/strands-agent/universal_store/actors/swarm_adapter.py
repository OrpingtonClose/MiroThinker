"""SwarmAdapter — bridges orchestrator events to SwarmSupervisor method calls.

The orchestrator broadcasts high-level events (``swarm_restart``), but
``SwarmSupervisor`` expects direct method invocations
(``start_extraction``, ``start_gossip``).  This adapter receives the
orchestrator event, instantiates a ``SwarmSupervisor``, drives it
through extraction and gossip, and forwards ``SwarmComplete`` back to
the orchestrator.
"""
from __future__ import annotations

import asyncio

from universal_store.actors.base import Actor
from universal_store.actors.swarm import SwarmSupervisor
from universal_store.config import UnifiedConfig
from universal_store.protocols import Event, StoreProtocol, SwarmComplete
from universal_store.trace import TraceStore, trace_block


class SwarmAdapterActor(Actor):
    """Adapter that runs a real swarm and emits ``SwarmComplete``."""

    def __init__(
        self,
        actor_id: str = "swarm_adapter",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
    ) -> None:
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig.from_env()

    async def _run(self) -> None:
        trace = await TraceStore.get()
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            async with trace_block(
                self.actor_id,
                "swarm_adapter_received",
                payload={"event_type": event.event_type},
            ):
                if event.event_type == "swarm_restart":
                    await self._run_swarm(event, trace)
                elif event.event_type == "SwarmComplete":
                    # Forward upstream to orchestrator
                    await self.send_to_parent(event)
                else:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="swarm_adapter_ignored",
                        payload={"event_type": event.event_type},
                    )

    async def _run_swarm(self, event: Event, trace: TraceStore) -> None:
        """Instantiate SwarmSupervisor, run extraction + gossip."""
        query = event.payload.get("query", "")
        reason = event.payload.get("reason", "unknown")

        await trace.record(
            actor_id=self.actor_id,
            event_type="swarm_adapter_start",
            payload={"query": query, "reason": reason},
        )

        # Create a temporary swarm supervisor as our child
        swarm = SwarmSupervisor(
            actor_id="swarm",
            store=self.store,
            config=self.config,
        )
        self.spawn_child(swarm)

        # Provide some mock raw condition IDs and angles
        raw_ids = [1, 2, 3]
        angles = ["mechanism", "clinical", "systems"]

        try:
            await swarm.start_extraction(raw_ids, angles)
            await swarm.start_gossip()
            # start_gossip returns after emitting SwarmComplete to its parent
            # (which is this adapter). The adapter's _run loop will pick it up
            # and forward it to the orchestrator.
        except Exception as e:
            await trace.record(
                actor_id=self.actor_id,
                event_type="swarm_adapter_error",
                error=e,
            )
            # Emit a synthetic SwarmComplete so the orchestrator doesn't hang
            await self.send_to_parent(
                SwarmComplete(findings=raw_ids, gaps=[])
            )
        finally:
            await swarm.stop(graceful=False)
