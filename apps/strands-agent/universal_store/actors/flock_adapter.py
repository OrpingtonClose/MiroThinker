"""FlockAdapter — bridges orchestrator events to FlockSupervisor method calls.

The orchestrator broadcasts ``flock_start``, but ``FlockSupervisor`` expects
``start_flock()`` to be called directly.  This adapter receives the event,
instantiates a ``FlockSupervisor``, drives it, and forwards
``FlockComplete`` back to the orchestrator.
"""
from __future__ import annotations

import asyncio

from universal_store.actors.base import Actor
from universal_store.actors.flock import FlockSupervisor
from universal_store.config import UnifiedConfig
from universal_store.protocols import Event, FlockComplete
from universal_store.trace import TraceStore, trace_block


class FlockAdapterActor(Actor):
    """Adapter that runs a real flock and emits ``FlockComplete``."""

    def __init__(
        self,
        actor_id: str = "flock_adapter",
        config: UnifiedConfig | None = None,
    ) -> None:
        super().__init__(actor_id)
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
                "flock_adapter_received",
                payload={"event_type": event.event_type},
            ):
                if event.event_type == "flock_start":
                    await self._run_flock(event, trace)
                elif event.event_type == "FlockComplete":
                    await self.send_to_parent(event)
                else:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="flock_adapter_ignored",
                        payload={"event_type": event.event_type},
                    )

    async def _run_flock(self, event: Event, trace: TraceStore) -> None:
        """Instantiate FlockSupervisor, start flock, allocate budgets."""
        findings = event.payload.get("findings", [])
        gaps = event.payload.get("gaps", [])

        await trace.record(
            actor_id=self.actor_id,
            event_type="flock_adapter_start",
            payload={"findings_count": len(findings), "gaps_count": len(gaps)},
        )

        flock = FlockSupervisor(
            actor_id="flock",
            config=self.config,
        )
        self.spawn_child(flock)

        # Provide mock condition IDs and angles
        condition_ids = findings if findings else [1, 2, 3]
        angles = ["mechanism", "clinical", "systems"]

        try:
            flock.start_flock(condition_ids, angles)
            # Kick off first round manually (start_flock doesn't do this)
            await flock._allocate_budgets(1)
            # The flock now runs autonomously; clones report back via
            # FlockRoundComplete events to the flock's mailbox.
            # _evaluate_round() will eventually send FlockComplete to us
            # (our parent relationship). We wait a bit then clean up.
            await asyncio.sleep(0.5)
        except Exception as e:
            await trace.record(
                actor_id=self.actor_id,
                event_type="flock_adapter_error",
                error=e,
            )
            await self.send_to_parent(
                FlockComplete(convergence_reason="adapter_error", directions=[])
            )
        finally:
            await flock.stop(graceful=False)
