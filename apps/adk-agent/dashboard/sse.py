# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""SSE and REST endpoints for the real-time pipeline dashboard.

Mount these on the FastAPI app alongside the AG-UI endpoint:

    from dashboard.sse import mount_dashboard_routes
    mount_dashboard_routes(app)

Endpoints:
  GET /dashboard/stream   — SSE stream pushing collector snapshots every 500ms
  GET /dashboard/latest   — most recent finalized run (JSON)
  GET /dashboard/runs     — list all saved dashboard JSONs
  GET /dashboard/run/{id} — fetch a specific run by session_id prefix
  GET /dashboard/html/{id} — render a run as self-contained HTML
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import glob
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from dashboard import get_any_active_collector
from dashboard import event_store
from dashboard.html_report import generate_dashboard_html

logger = logging.getLogger(__name__)

_DASHBOARD_LOGS_DIR = os.environ.get(
    "DASHBOARD_LOGS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard_logs"),
)

# Thread pool for SQLite reads — completely decoupled from the async event loop
_db_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="dashboard-db"
)


async def _sse_generator(request: Request):
    """Yield SSE events with snapshots every 500ms.

    Reads from SQLite via a thread-pool executor so the SSE stream
    works even when the async event loop is saturated by LLM calls.
    Falls back to the in-memory collector if SQLite has no data.
    """
    loop = asyncio.get_event_loop()
    while True:
        if await request.is_disconnected():
            break

        # Try SQLite first (works even when event loop is busy)
        snapshot = await loop.run_in_executor(
            _db_executor, event_store.get_latest_snapshot, None
        )

        if snapshot:
            yield f"data: {json.dumps(snapshot, default=str)}\n\n"
        else:
            # Fall back to in-memory collector
            collector = get_any_active_collector()
            if collector:
                snapshot = collector.snapshot()
                yield f"data: {json.dumps(snapshot, default=str)}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'idle', 'message': 'No active pipeline run'})}\n\n"
        await asyncio.sleep(0.5)


async def dashboard_stream(request: Request) -> StreamingResponse:
    """SSE stream that pushes collector snapshots while pipeline runs."""
    return StreamingResponse(
        _sse_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _load_runs() -> list[dict[str, Any]]:
    """Load all dashboard JSON files, sorted newest first."""
    pattern = os.path.join(_DASHBOARD_LOGS_DIR, "pipeline_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    runs = []
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
            runs.append({
                "file": os.path.basename(path),
                "session_id": data.get("session_id", ""),
                "query": data.get("query", "")[:100],
                "elapsed_secs": data.get("elapsed_secs", 0),
                "started_at": data.get("started_at", 0),
                "kpi": data.get("kpi", {}),
            })
        except Exception:
            logger.warning("Failed to load %s", path, exc_info=True)
    return runs


def _load_run_by_id(session_prefix: str) -> dict[str, Any] | None:
    """Load a specific run by session_id prefix match."""
    pattern = os.path.join(_DASHBOARD_LOGS_DIR, "pipeline_*.json")
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("session_id", "").startswith(session_prefix):
                return data
        except Exception:
            continue
    return None


async def dashboard_runs(request: Request) -> JSONResponse:
    """GET /dashboard/runs — list all saved dashboard JSONs + SQLite runs."""
    loop = asyncio.get_event_loop()

    # Get runs from SQLite (includes currently running ones)
    db_runs = await loop.run_in_executor(_db_executor, event_store.get_all_runs)

    # Also get legacy JSON file runs
    file_runs = _load_runs()

    # Merge: SQLite runs take priority, add file-only runs
    seen_ids = {r["session_id"] for r in db_runs}
    merged = db_runs + [r for r in file_runs if r.get("session_id") not in seen_ids]

    return JSONResponse({"runs": merged})


async def dashboard_latest(request: Request) -> JSONResponse:
    """GET /dashboard/latest — most recent snapshot (from SQLite or memory)."""
    loop = asyncio.get_event_loop()

    # Try SQLite first (works even under load)
    snapshot = await loop.run_in_executor(
        _db_executor, event_store.get_latest_snapshot, None
    )
    if snapshot:
        return JSONResponse(snapshot)

    # Fall back to in-memory collector
    collector = get_any_active_collector()
    if collector:
        return JSONResponse(collector.snapshot())

    # Otherwise return the most recent saved run
    runs = _load_runs()
    if not runs:
        return JSONResponse({"error": "No dashboard runs found"}, status_code=404)

    # Load the full data for the most recent run
    data = _load_run_by_id(runs[0]["session_id"][:8])
    if data:
        return JSONResponse(data)
    return JSONResponse({"error": "Run data not found"}, status_code=404)


async def dashboard_run_detail(request: Request) -> JSONResponse:
    """GET /dashboard/run/{session_id} — fetch a specific run."""
    session_id = request.path_params.get("session_id", "")
    data = _load_run_by_id(session_id)
    if data:
        return JSONResponse(data)
    return JSONResponse({"error": f"Run {session_id} not found"}, status_code=404)


async def dashboard_html(request: Request) -> HTMLResponse:
    """GET /dashboard/html/{session_id} — render a run as HTML."""
    session_id = request.path_params.get("session_id", "")
    data = _load_run_by_id(session_id)
    if not data:
        return HTMLResponse(
            f"<h1>Run {session_id} not found</h1>", status_code=404
        )
    html_content = generate_dashboard_html(data)
    return HTMLResponse(html_content)


async def dashboard_html_latest(request: Request) -> HTMLResponse:
    """GET /dashboard/html — render the most recent run as HTML."""
    runs = _load_runs()
    if not runs:
        return HTMLResponse("<h1>No dashboard runs found</h1>", status_code=404)
    data = _load_run_by_id(runs[0]["session_id"][:8])
    if not data:
        return HTMLResponse("<h1>Run data not found</h1>", status_code=404)
    return HTMLResponse(generate_dashboard_html(data))


def mount_dashboard_routes(app: FastAPI) -> None:
    """Mount all dashboard endpoints on the FastAPI app."""
    app.add_api_route("/dashboard/stream", dashboard_stream, methods=["GET"])
    app.add_api_route("/dashboard/latest", dashboard_latest, methods=["GET"])
    app.add_api_route("/dashboard/runs", dashboard_runs, methods=["GET"])
    app.add_api_route("/dashboard/run/{session_id}", dashboard_run_detail, methods=["GET"])
    app.add_api_route("/dashboard/html/{session_id}", dashboard_html, methods=["GET"])
    app.add_api_route("/dashboard/html", dashboard_html_latest, methods=["GET"])
    logger.info("Dashboard routes mounted at /dashboard/*")
