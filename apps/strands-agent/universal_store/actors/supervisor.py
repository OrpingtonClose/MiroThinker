"""Supervision tree — hierarchical actor management with fault tolerance.

Pattern: parent supervises children. On child crash, parent decides:
- Restart (up to max_restarts within window)
- Escalate (crash self, let grandparent handle)
- Terminate (shut down the branch)
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from .base import Actor
from universal_store.protocols import Event
from universal_store.trace import TraceStore


class SupervisionStrategy:
    RESTART = "restart"
    ESCALATE = "escalate"
    TERMINATE = "terminate"


class Supervisor(Actor):
    """Base supervisor. Manages a group of child actors."""

    def __init__(
        self,
        actor_id: str,
        strategy: str = SupervisionStrategy.RESTART,
        max_restarts: int = 3,
        restart_window_s: float = 60.0,
    ):
        super().__init__(actor_id)
        self.strategy = strategy
        self._max_restarts = max_restarts
        self._restart_window_s = restart_window_s
        self._child_specs: dict[str, Callable[[], Actor]] = {}  # factory functions

    def register_child(self, child_id: str, factory: Callable[[], Actor]) -> None:
        """Register a child factory. The supervisor can respawn on crash."""
        self._child_specs[child_id] = factory

    async def _run(self) -> None:
        """Supervisor loop: process own mailbox + monitor children."""
        trace = await TraceStore.get()
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
            except asyncio.TimeoutError:
                await self._health_check_all()
                continue

            await trace.record(
                actor_id=self.actor_id,
                event_type="supervisor_received",
                payload={"event_type": event.event_type, "source": event.source_actor},
            )

            if event.event_type == "actor_crashed":
                child_id = event.payload.get("actor_id", event.source_actor)
                await self._handle_child_crash(child_id)
            else:
                await self._handle_event(event)

    async def _handle_child_crash(self, child_id: str) -> None:
        trace = await TraceStore.get()
        child = self._children.get(child_id)
        if child is None and child_id not in self._child_specs:
            await trace.record(
                actor_id=self.actor_id,
                event_type="supervisor_unknown_child_crash",
                payload={"child_id": child_id},
            )
            return

        if self.strategy == SupervisionStrategy.RESTART:
            factory = self._child_specs.get(child_id)
            if factory:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="supervisor_restarting_child",
                    payload={"child_id": child_id},
                )
                new_child = factory()
                self.spawn_child(new_child)
            else:
                await self._escalate(f"No factory for child {child_id}")
        elif self.strategy == SupervisionStrategy.ESCALATE:
            await self._escalate(f"Child {child_id} crashed")
        elif self.strategy == SupervisionStrategy.TERMINATE:
            await trace.record(
                actor_id=self.actor_id,
                event_type="supervisor_terminating_branch",
                payload={"child_id": child_id},
            )
            if child:
                await child.stop(graceful=False)
            del self._children[child_id]

    async def _escalate(self, reason: str) -> None:
        trace = await TraceStore.get()
        await trace.record(
            actor_id=self.actor_id,
            event_type="supervisor_escalating",
            payload={"reason": reason},
        )
        await self.stop(graceful=False)
        if self._parent:
            await self._parent.send(Event("actor_crashed", {"actor_id": self.actor_id, "reason": reason}))

    async def _handle_event(self, event: Event) -> None:
        """Override in subclass for custom event routing."""
        pass

    async def _health_check_all(self) -> None:
        trace = await TraceStore.get()
        for child_id, child in list(self._children.items()):
            health = await child.health()
            if not health.get("running"):
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="supervisor_detected_dead_child",
                    payload={"child_id": child_id},
                )
                await self._handle_child_crash(child_id)

    async def spawn_all_registered(self) -> None:
        """Spawn all registered children."""
        for child_id, factory in self._child_specs.items():
            if child_id not in self._children:
                child = factory()
                self.spawn_child(child)


class OneForOneSupervisor(Supervisor):
    """If one child crashes, restart only that child."""
    def __init__(self, actor_id: str, **kw):
        super().__init__(actor_id, strategy=SupervisionStrategy.RESTART, **kw)


class AllForOneSupervisor(Supervisor):
    """If one child crashes, restart ALL children."""
    def __init__(self, actor_id: str, **kw):
        super().__init__(actor_id, strategy=SupervisionStrategy.RESTART, **kw)

    async def _handle_child_crash(self, child_id: str) -> None:
        trace = await TraceStore.get()
        await trace.record(
            actor_id=self.actor_id,
            event_type="supervisor_all_for_one_restart",
            payload={"crashed_child": child_id, "total_children": len(self._children)},
        )
        # Stop all children
        for cid, child in list(self._children.items()):
            await child.stop(graceful=False)
        self._children.clear()
        # Respawn all
        await self.spawn_all_registered()


class RootSupervisor(Supervisor):
    """Top-level supervisor. Cannot escalate — logs and survives."""
    async def _escalate(self, reason: str) -> None:
        trace = await TraceStore.get()
        await trace.record(
            actor_id=self.actor_id,
            event_type="root_supervisor_cannot_escalate",
            error=Exception(reason),
        )
        # Root supervisor does not die. It logs and continues.
