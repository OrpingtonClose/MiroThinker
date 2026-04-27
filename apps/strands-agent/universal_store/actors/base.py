"""Base Actor class — the foundation of the entire supervision tree.

Every actor in the system inherits from this. It provides:
- Async mailbox (Queue)
- Graceful lifecycle (start, stop, restart)
- Automatic trace recording (no silent anything)
- Health check emission
- Exception isolation (crash one actor, not the tree)
"""
from __future__ import annotations

import asyncio
import inspect
from abc import abstractmethod
from typing import Any

from universal_store.protocols import ActorProtocol, Event
from universal_store.trace import TraceStore, trace_block


class Actor(ActorProtocol):
    """Base class for all actors. Do not instantiate directly."""

    def __init__(self, actor_id: str, mailbox_size: int = 10_000):
        self.actor_id = actor_id
        self.mailbox: asyncio.Queue[Event] = asyncio.Queue(maxsize=mailbox_size)
        self._task: asyncio.Task | None = None
        self._shutdown = False
        self._restart_count = 0
        self._max_restarts = 3
        self._restart_window_s = 60.0
        self._restart_timestamps: list[float] = []
        self._parent: Actor | None = None
        self._children: dict[str, Actor] = {}
        self._trace: TraceStore | None = None

    async def _ensure_trace(self) -> TraceStore:
        if self._trace is None:
            self._trace = await TraceStore.get()
        return self._trace

    async def send(self, event: Event) -> None:
        """Send an event to this actor's mailbox. Non-blocking."""
        try:
            self.mailbox.put_nowait(event.with_source(self.actor_id))
        except asyncio.QueueFull:
            trace = await self._ensure_trace()
            await trace.record(
                actor_id=self.actor_id,
                event_type="mailbox_dropped",
                payload={"event_type": event.event_type, "reason": "queue_full"},
            )

    def start(self) -> None:
        """Start the actor's event loop. Idempotent."""
        if self._task is None or self._task.done():
            self._shutdown = False
            self._task = asyncio.create_task(self._run_wrapper(), name=f"actor:{self.actor_id}")

    async def _run_wrapper(self) -> None:
        """Wraps _run with trace, error isolation, and restart logic."""
        trace = await self._ensure_trace()
        await trace.record(
            actor_id=self.actor_id,
            event_type="actor_started",
            payload={"children": list(self._children.keys())},
        )
        while not self._shutdown:
            try:
                async with trace_block(self.actor_id, "actor_run_cycle", ""):
                    await self._run()
            except asyncio.CancelledError:
                await trace.record(actor_id=self.actor_id, event_type="actor_cancelled")
                break
            except Exception as e:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="actor_crashed",
                    error=e,
                )
                if await self._should_restart():
                    self._restart_count += 1
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="actor_restarting",
                        payload={"restart_count": self._restart_count},
                    )
                    await asyncio.sleep(1.0)  # backoff
                else:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="actor_permanently_failed",
                        payload={"restart_count": self._restart_count},
                    )
                    break
        await trace.record(actor_id=self.actor_id, event_type="actor_stopped")

    async def _should_restart(self) -> bool:
        now = asyncio.get_event_loop().time()
        # Remove old restart timestamps outside the window
        self._restart_timestamps = [t for t in self._restart_timestamps if now - t < self._restart_window_s]
        if len(self._restart_timestamps) >= self._max_restarts:
            return False
        self._restart_timestamps.append(now)
        return True

    @abstractmethod
    async def _run(self) -> None:
        """Override in subclass. The main actor loop."""
        raise NotImplementedError

    async def stop(self, graceful: bool = True) -> None:
        """Stop the actor. If graceful, let in-flight work finish."""
        self._shutdown = True
        if self._task and not self._task.done():
            if not graceful:
                self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                # Force cancel if graceful shutdown took too long
                self._task.cancel()
                try:
                    await asyncio.wait_for(self._task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            except asyncio.CancelledError:
                pass
        # Stop children
        for child in list(self._children.values()):
            await child.stop(graceful=graceful)

    def spawn_child(self, child: Actor) -> None:
        """Spawn a child actor under this actor's supervision."""
        child._parent = self
        self._children[child.actor_id] = child
        child.start()

    async def send_to_parent(self, event: Event) -> None:
        if self._parent:
            await self._parent.send(event)

    async def broadcast_to_children(self, event: Event) -> None:
        for child in self._children.values():
            await child.send(event)

    async def health(self) -> dict[str, Any]:
        """Return health metrics. Override for richer data."""
        return {
            "actor_id": self.actor_id,
            "running": self._task is not None and not self._task.done(),
            "mailbox_size": self.mailbox.qsize(),
            "restart_count": self._restart_count,
            "children_count": len(self._children),
        }
