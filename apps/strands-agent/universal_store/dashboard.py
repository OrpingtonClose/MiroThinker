"""FastAPI dashboard for the Universal Store Architecture.

Provides:
- Real-time SSE event stream fed by the OrchestratorActor output queue
- Trace viewer with server-side filtering
- Operator controls (user interrupts)
- Curator digests and semantic graph queries

**No actor modules are imported here** — the dashboard receives events via a
module-level :class:`asyncio.Queue` that OrchestratorActor pushes to.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from universal_store.protocols import OrchestratorEvent, OrchestratorPhase
from universal_store.trace import TraceStore
from universal_store.config import UnifiedConfig


# ---------------------------------------------------------------------------
# Global Orchestrator output queue
# ---------------------------------------------------------------------------

_orchestrator_queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue(maxsize=10_000)


def get_event_queue() -> asyncio.Queue[OrchestratorEvent]:
    """Return the dashboard event queue.

    OrchestratorActor should publish :class:`OrchestratorEvent` objects to
    this queue; the dashboard relays them to all connected SSE clients.
    """
    return _orchestrator_queue


# ---------------------------------------------------------------------------
# Internal broadcaster — fan-out one Orchestrator queue → many SSE clients
# ---------------------------------------------------------------------------

class _EventBroadcaster:
    """Relays events from the single Orchestrator queue to per-client queues."""

    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[OrchestratorEvent]] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[OrchestratorEvent]:
        """Create a new subscriber queue."""
        q: asyncio.Queue[OrchestratorEvent] = asyncio.Queue(maxsize=1_000)
        async with self._lock:
            self._subs.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[OrchestratorEvent]) -> None:
        async with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    async def publish(self, event: OrchestratorEvent) -> None:
        """Broadcast *event* to all subscribers, dropping those with full queues."""
        dead: list[asyncio.Queue[OrchestratorEvent]] = []
        async with self._lock:
            for q in self._subs:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subs.remove(q)


_broadcaster = _EventBroadcaster()


async def _relay_orchestrator_events() -> None:
    """Background task: drain the Orchestrator queue and broadcast."""
    while True:
        try:
            event = await _orchestrator_queue.get()
            await _broadcaster.publish(event)
        except asyncio.CancelledError:
            break
        except Exception:
            # Prevent a malformed event from killing the relay.
            await asyncio.sleep(0.5)


async def event_stream() -> AsyncIterator[str]:
    """SSE stream of :class:`OrchestratorEvent` objects.

    Subscribes to the OrchestratorActor output queue (via the internal
    broadcaster).  Each event is serialised as ``data: {json}\n\n``.  A
    heartbeat comment is emitted every 15 seconds so reverse proxies do
    not drop idle connections.
    """
    q = await _broadcaster.subscribe()
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=15.0)
                payload = {
                    "phase": event.phase.value,
                    "message": event.message,
                    "data": event.data,
                    "timestamp": event.timestamp,
                    "trace_id": event.trace_id,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            except asyncio.TimeoutError:
                yield ":heartbeat\n\n"
    finally:
        await _broadcaster.unsubscribe(q)


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class InterruptRequest(BaseModel):
    """User interrupt submitted via the dashboard."""

    action: str = Field(..., pattern="^(pause|stop|inject)$")
    message: str = ""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_dashboard_app(trace_store: TraceStore) -> FastAPI:
    """Factory that creates a FastAPI dashboard wired to *trace_store*.

    Parameters
    ----------
    trace_store:
        The active :class:`TraceStore` instance (usually obtained via
        ``await TraceStore.get()``).

    Returns
    -------
    FastAPI
        Configured application with all dashboard endpoints.
    """
    config = UnifiedConfig.from_env()

    # Connection to the main DuckDB store for lessons / semantic graph.
    import duckdb

    store_conn = duckdb.connect(config.store.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        relay_task = asyncio.create_task(_relay_orchestrator_events())
        yield
        relay_task.cancel()
        try:
            await relay_task
        except asyncio.CancelledError:
            pass
        store_conn.close()

    app = FastAPI(
        title="Universal Store Dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — wide open for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Middleware: trace every request / response
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        start = time.time()
        trace_id = await trace_store.record(
            actor_id="dashboard",
            event_type="http_request",
            payload={
                "method": request.method,
                "path": request.url.path,
                "query": str(request.query_params),
            },
        )
        try:
            response = await call_next(request)
            latency_ms = (time.time() - start) * 1000
            await trace_store.record(
                actor_id="dashboard",
                event_type="http_response",
                payload={
                    "trace_id": trace_id,
                    "status_code": response.status_code,
                    "path": request.url.path,
                },
                latency_ms=latency_ms,
            )
            return response
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            await trace_store.record(
                actor_id="dashboard",
                event_type="http_error",
                payload={"trace_id": trace_id, "path": request.url.path},
                latency_ms=latency_ms,
                error=exc,
            )
            raise

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> dict[str, Any]:
        """System health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # SSE events
    # ------------------------------------------------------------------
    @app.get("/events")
    async def events() -> StreamingResponse:
        """Server-Sent Events stream of :class:`OrchestratorEvent` objects."""
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Trace queries
    # ------------------------------------------------------------------
    @app.get("/trace")
    async def trace(
        run_id: str | None = None,
        actor_id: str | None = None,
        event_type: str | None = None,
        since: str | None = None,
        limit: int = Query(1000, ge=1, le=10_000),
    ) -> list[dict]:
        """Query trace records with optional filters."""
        return await trace_store.query(
            run_id=run_id,
            actor_id=actor_id,
            event_type=event_type,
            since=since,
            limit=limit,
        )

    @app.get("/trace/stats/{run_id}")
    async def trace_stats(run_id: str) -> dict[str, Any]:
        """Aggregate statistics for a single run."""
        return await trace_store.get_stats(run_id)

    @app.get("/trace/errors/{run_id}")
    async def trace_errors(
        run_id: str,
        limit: int = Query(100, ge=1, le=1000),
    ) -> list[dict]:
        """Error traces for a single run."""
        return await trace_store.get_errors(run_id, limit=limit)

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------
    @app.get("/lessons")
    async def lessons(
        type: str | None = None,
        angle: str | None = None,
        since: str | None = None,
        min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    ) -> list[dict]:
        """Query recorded lessons."""
        conditions = ["1=1"]
        params: list[Any] = []

        if type is not None:
            conditions.append("lesson_type = ?")
            params.append(type)
        if angle is not None:
            conditions.append("angle = ?")
            params.append(angle)
        if since is not None:
            conditions.append("created_at >= ?")
            params.append(since)
        if min_confidence > 0.0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)

        sql = f"""
            SELECT * FROM lessons
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
        """
        return store_conn.execute(sql, params).fetchdf().to_dict("records")

    # ------------------------------------------------------------------
    # Operator controls
    # ------------------------------------------------------------------
    @app.post("/interrupt")
    async def interrupt(body: InterruptRequest) -> dict[str, Any]:
        """Submit a user interrupt (pause, stop, or inject)."""
        await trace_store.record(
            actor_id="dashboard",
            event_type="user_interrupt",
            payload={"action": body.action, "message": body.message},
        )

        event = OrchestratorEvent(
            phase=OrchestratorPhase.USER_INTERRUPTION,
            message=body.message,
            data={"action": body.action, "source": "dashboard"},
        )
        try:
            _orchestrator_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

        return {"status": "accepted", "action": body.action}

    # ------------------------------------------------------------------
    # Digests
    # ------------------------------------------------------------------
    @app.get("/digest")
    async def digest(
        limit: int = Query(50, ge=1, le=500),
    ) -> list[dict]:
        """Latest curator digests."""
        return await trace_store.query(
            event_type="CuratorDigest",
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Semantic graph
    # ------------------------------------------------------------------
    @app.get("/semantic-graph")
    async def semantic_graph(
        limit: int = Query(500, ge=1, le=5000),
    ) -> list[dict]:
        """Semantic connections graph."""
        sql = """
            SELECT * FROM semantic_connections
            ORDER BY created_at DESC
            LIMIT ?
        """
        return store_conn.execute(sql, [limit]).fetchdf().to_dict("records")

    # ------------------------------------------------------------------
    # Minimal HTML trace viewer
    # ------------------------------------------------------------------
    @app.get("/trace/viewer", response_class=HTMLResponse)
    async def trace_viewer() -> str:
        """Filterable trace viewer with live SSE feed."""
        return _TRACE_VIEWER_HTML

    return app


# ---------------------------------------------------------------------------
# Minimal trace viewer HTML
# ---------------------------------------------------------------------------

_TRACE_VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trace Viewer — Universal Store</title>
<style>
  :root { --bg: #fafafa; --fg: #111; --accent: #2563eb; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 2rem; background: var(--bg); color: var(--fg); }
  h1 { margin-top: 0; }
  .filters { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .filters input, .filters button { padding: .5rem .75rem; font-size: 1rem; border: 1px solid #ccc; border-radius: .375rem; }
  .filters button { background: var(--accent); color: #fff; border-color: var(--accent); cursor: pointer; }
  table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
  th, td { padding: .6rem .75rem; border-bottom: 1px solid #e5e5e5; text-align: left; font-size: .875rem; }
  th { background: #f3f4f6; font-weight: 600; }
  tr:hover { background: #f9fafb; }
  .live { margin-top: 2rem; }
  #events { background: #111; color: #0f0; padding: 1rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem; white-space: pre-wrap; max-height: 20rem; overflow-y: auto; border-radius: .375rem; }
  .muted { color: #666; font-size: .8rem; }
</style>
</head>
<body>
<h1>Trace Viewer</h1>
<div class="filters">
  <input id="runId" placeholder="run_id">
  <input id="actorId" placeholder="actor_id">
  <input id="eventType" placeholder="event_type">
  <input id="since" type="datetime-local">
  <button onclick="load()">Load</button>
</div>
<table id="table">
  <thead>
    <tr>
      <th>trace_id</th><th>run_id</th><th>actor_id</th><th>event_type</th>
      <th>phase</th><th>timestamp</th><th>latency_ms</th><th>error</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
<div class="live">
  <h2>Live Events <span class="muted">(SSE /events)</span></h2>
  <div id="events">Connecting…</div>
</div>
<script>
const es = new EventSource('/events');
const eventsDiv = document.getElementById('events');
es.onmessage = e => {
  const data = JSON.parse(e.data);
  eventsDiv.textContent = JSON.stringify(data, null, 2) + '\n---\n' + eventsDiv.textContent.slice(0, 4000);
};
es.onerror = () => { eventsDiv.textContent = 'Disconnected — reconnecting…\n' + eventsDiv.textContent; };

async function load() {
  const p = new URLSearchParams();
  if (runId.value) p.set('run_id', runId.value);
  if (actorId.value) p.set('actor_id', actorId.value);
  if (eventType.value) p.set('event_type', eventType.value);
  if (since.value) p.set('since', new Date(since.value).toISOString());
  p.set('limit', '1000');
  const rows = await fetch('/trace?' + p).then(r => r.json());
  const tbody = document.querySelector('#table tbody');
  tbody.innerHTML = rows.map(r => `<tr>
    <td>${esc(r.trace_id)}</td>
    <td>${esc(r.run_id)}</td>
    <td>${esc(r.actor_id)}</td>
    <td>${esc(r.event_type)}</td>
    <td>${esc(r.phase)}</td>
    <td>${esc(r.timestamp)}</td>
    <td>${r.latency_ms ?? ''}</td>
    <td>${esc(r.error)}</td>
  </tr>`).join('');
}
function esc(s) { return (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
</script>
</body>
</html>"""
