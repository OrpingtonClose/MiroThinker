"""User-facing actor — buffers messages, handles interrupts, and routes operator queries.

The ``UserProxyActor`` sits between the user/UI and the supervision tree. It never
blocks on user input; all communication is async via the mailbox. It is responsible for:

- Buffering user messages in ``self.user_queue``.
- Handling ``UserInterrupt`` events (pause / stop / inject).
- Routing ``QueryRequest`` events to the ``OperatorQueryEngine``.
- Re-hydrating state from the trace store on resume.
- Tracing every interrupt, pause/resume, and query routed.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from universal_store.actors.base import Actor
from universal_store.protocols import Event, OrchestratorPhase, UserInterrupt
from universal_store.config import UnifiedConfig
from universal_store.trace import TraceStore, trace_block


class QueryRequest(Event):
    """Event representing an operator query to be routed to the query engine."""

    def __init__(self, query_text: str, query_type: str = "generic", **kw: Any) -> None:
        super().__init__("QueryRequest", {"query_text": query_text, "query_type": query_type}, **kw)


class OperatorQueryEngine:
    """Placeholder for the operator query engine.

    In a full implementation this would dispatch to the appropriate query handler.
    For now it simply emits a ``QueryResponse`` event so the routing path can be
    exercised end-to-end.
    """

    async def handle(self, request: QueryRequest) -> Event:
        """Process a query request and return a response event."""
        return Event(
            "QueryResponse",
            {
                "query_text": request.payload.get("query_text"),
                "query_type": request.payload.get("query_type"),
                "status": "routed",
            },
        )


class UserProxyActor(Actor):
    """Buffers user messages, handles interrupts, and routes operator queries.

    Parameters
    ----------
    actor_id:
        Unique identifier for this actor in the supervision tree.
    config:
        ``UnifiedConfig`` instance. If ``None``, loaded from environment.
    orchestrator:
        Optional reference to the ``OrchestratorActor``. When ``None``, events are
        forwarded to the actor's parent via :meth:`send_to_parent`.
    """

    def __init__(
        self,
        actor_id: str,
        config: UnifiedConfig | None = None,
        orchestrator: Actor | None = None,
    ) -> None:
        cfg = config or UnifiedConfig.from_env()
        super().__init__(actor_id, mailbox_size=cfg.actor.mailbox_queue_size)
        self.config = cfg
        self.orchestrator = orchestrator
        self.user_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pending_messages: dict[str, dict] = {}
        self._paused = False
        self._stopped = False
        self._injections: list[dict] = []
        self._query_engine = OperatorQueryEngine()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Read events from the mailbox and dispatch them to handlers."""
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            async with trace_block(
                self.actor_id,
                "user_proxy_handle_event",
                phase="",
                payload={"event_type": event.event_type, "source": event.source_actor},
            ):
                if event.event_type == "UserInterrupt":
                    await self._handle_interrupt(event)
                elif event.event_type == "QueryRequest":
                    await self._route_query(event)
                elif event.event_type == "user_message":
                    await self._buffer_message(event.payload)
                elif event.event_type == "resume":
                    await self._handle_resume()
                else:
                    trace = await self._ensure_trace()
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="user_proxy_unknown_event",
                        payload={"event_type": event.event_type},
                    )

    # ------------------------------------------------------------------
    # Interrupt handling
    # ------------------------------------------------------------------

    async def _handle_interrupt(self, event: Event) -> None:
        """Dispatch a ``UserInterrupt`` event based on its ``action`` field."""
        trace = await self._ensure_trace()
        action = event.payload.get("action")
        message = event.payload.get("message", "")

        await trace.record(
            actor_id=self.actor_id,
            event_type="user_proxy_interrupt_received",
            phase=str(OrchestratorPhase.USER_INTERRUPTION),
            payload={"action": action, "message": message},
        )

        if action == "pause":
            await self._do_soft_pause()
        elif action == "stop":
            await self._do_hard_stop()
        elif action == "inject":
            await self._do_inject(message)
        else:
            await trace.record(
                actor_id=self.actor_id,
                event_type="user_proxy_unknown_interrupt",
                phase=str(OrchestratorPhase.USER_INTERRUPTION),
                payload={"action": action, "message": message},
            )

    async def _do_soft_pause(self) -> None:
        """Soft pause: stop launching new workers, let in-flight work finish."""
        trace = await self._ensure_trace()
        self._paused = True
        await self._snapshot_state()
        async with trace_block(
            self.actor_id,
            "user_proxy_soft_pause",
            phase=str(OrchestratorPhase.USER_INTERRUPTION),
            payload={"paused": True},
        ):
            await self._emit_to_orchestrator(
                Event("orchestrator_pause", {"mode": "soft", "source": self.actor_id})
            )

    async def _do_hard_stop(self) -> None:
        """Hard stop: cancel all tasks and shut down children."""
        trace = await self._ensure_trace()
        self._stopped = True
        await self._snapshot_state()
        async with trace_block(
            self.actor_id,
            "user_proxy_hard_stop",
            phase=str(OrchestratorPhase.USER_INTERRUPTION),
            payload={"stopped": True},
        ):
            await self._emit_to_orchestrator(
                Event("orchestrator_stop", {"mode": "hard", "source": self.actor_id})
            )
            for child in list(self._children.values()):
                await child.stop(graceful=False)

    async def _do_inject(self, message: str) -> None:
        """Queue a new angle/question for injection into the next swarm cycle."""
        trace = await self._ensure_trace()
        injection = {
            "message": message,
            "timestamp": time.time(),
        }
        self._injections.append(injection)
        await trace.record(
            actor_id=self.actor_id,
            event_type="user_proxy_injected",
            payload={"injection": injection, "total_injections": len(self._injections)},
        )
        await self._emit_to_orchestrator(
            Event("injection_queued", {"injection": injection, "count": len(self._injections)})
        )

    # ------------------------------------------------------------------
    # Resume / re-hydration
    # ------------------------------------------------------------------

    async def _handle_resume(self) -> None:
        """Resume from pause: re-hydrate state from store and signal orchestrator."""
        trace = await self._ensure_trace()
        async with trace_block(self.actor_id, "user_proxy_resume", ""):
            self._paused = False
            restored = await self._rehydrate_state()
            await trace.record(
                actor_id=self.actor_id,
                event_type="user_proxy_resumed",
                payload={"restored": restored, "injections": len(self._injections)},
            )
            await self._emit_to_orchestrator(
                Event(
                    "orchestrator_resume",
                    {
                        "source": self.actor_id,
                        "restored": restored,
                        "injections": len(self._injections),
                    },
                )
            )

    # ------------------------------------------------------------------
    # Message buffering
    # ------------------------------------------------------------------

    async def _buffer_message(self, payload: dict[str, Any]) -> None:
        """Buffer a user message in both the async queue and the indexed pending map."""
        trace = await self._ensure_trace()
        msg_id = payload.get("msg_id", f"msg_{time.time()}")
        message = {"msg_id": msg_id, **payload}
        await self.user_queue.put(message)
        self._pending_messages[msg_id] = message
        await trace.record(
            actor_id=self.actor_id,
            event_type="user_proxy_message_buffered",
            payload={"msg_id": msg_id, "queue_size": self.user_queue.qsize()},
        )

    def get_pending_messages(self) -> list[dict]:
        """Return all buffered user messages **without** clearing them.

        Returns
        -------
        list[dict]
            Copy of the pending message buffer.
        """
        return list(self._pending_messages.values())

    async def acknowledge_message(self, msg_id: str) -> bool:
        """Remove a message from the buffer by its ``msg_id``.

        Parameters
        ----------
        msg_id:
            Identifier of the message to acknowledge.

        Returns
        -------
        bool
            ``True`` if the message was found and removed, ``False`` otherwise.
        """
        trace = await self._ensure_trace()
        if msg_id not in self._pending_messages:
            await trace.record(
                actor_id=self.actor_id,
                event_type="user_proxy_message_acknowledge_failed",
                payload={"msg_id": msg_id, "reason": "not_found"},
            )
            return False

        del self._pending_messages[msg_id]

        # Best-effort removal from the async queue (rebuild without the target).
        temp: list[dict] = []
        found = False
        while not self.user_queue.empty():
            item = self.user_queue.get_nowait()
            if item.get("msg_id") == msg_id and not found:
                found = True
                continue
            temp.append(item)
        for item in temp:
            await self.user_queue.put(item)

        await trace.record(
            actor_id=self.actor_id,
            event_type="user_proxy_message_acknowledged",
            payload={"msg_id": msg_id, "found": True, "queue_size": self.user_queue.qsize()},
        )
        return True

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    async def _route_query(self, event: Event) -> None:
        """Forward a ``QueryRequest`` event to the ``OperatorQueryEngine``."""
        trace = await self._ensure_trace()
        query_text = event.payload.get("query_text", "")
        query_type = event.payload.get("query_type", "generic")
        request = QueryRequest(query_text=query_text, query_type=query_type)

        async with trace_block(
            self.actor_id,
            "user_proxy_route_query",
            payload={"query_text": query_text, "query_type": query_type},
        ):
            response = await self._query_engine.handle(request)
            await trace.record(
                actor_id=self.actor_id,
                event_type="user_proxy_query_routed",
                payload={
                    "query_text": query_text,
                    "response_event_type": response.event_type,
                },
            )
            await self._emit_to_orchestrator(response)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _emit_to_orchestrator(self, event: Event) -> None:
        """Send *event* to the orchestrator if known, otherwise to the parent actor."""
        if self.orchestrator is not None:
            await self.orchestrator.send(event)
        else:
            await self.send_to_parent(event)

    async def _snapshot_state(self) -> None:
        """Persist the current actor state to the trace store for later re-hydration."""
        trace = await self._ensure_trace()
        await trace.record(
            actor_id=self.actor_id,
            event_type="user_proxy_state_snapshot",
            payload={
                "paused": self._paused,
                "stopped": self._stopped,
                "injections": self._injections,
                "pending_message_ids": list(self._pending_messages.keys()),
            },
        )

    async def _rehydrate_state(self) -> bool:
        """Re-hydrate state from the most recent trace snapshot.

        Returns
        -------
        bool
            ``True`` if a snapshot was found and restored, ``False`` otherwise.
        """
        trace = await self._ensure_trace()
        try:
            records = await trace.query(
                actor_id=self.actor_id,
                event_type="user_proxy_state_snapshot",
                limit=1,
            )
            if not records:
                return False

            payload = json.loads(records[0].get("payload_json", "{}"))
            self._paused = payload.get("paused", False)
            self._stopped = payload.get("stopped", False)
            self._injections = payload.get("injections", [])
            return True
        except Exception as e:
            await trace.record(
                actor_id=self.actor_id,
                event_type="user_proxy_rehydrate_failed",
                error=e,
            )
            return False

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Return health metrics including pause state and queue sizes."""
        base = await super().health()
        base.update({
            "paused": self._paused,
            "stopped": self._stopped,
            "pending_messages": len(self._pending_messages),
            "injections_queued": len(self._injections),
            "user_queue_size": self.user_queue.qsize(),
        })
        return base
