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
    os.path.join(os.path.expanduser("~"), ".mirothinker", "dashboard_logs"),
)

# Thread pool for SQLite reads — completely decoupled from the async event loop
_db_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="dashboard-db"
)


# Grace period (seconds) after a run completes during which the SSE stream
# continues to serve the finalized snapshot instead of immediately showing idle.
_POST_RUN_GRACE_SECS = 30.0

# Tracks the most recently finalized snapshot and when it was captured.
_last_finalized_snapshot: dict | None = None
_last_finalized_at: float = 0.0


async def _sse_generator(request: Request):
    """Yield SSE events with snapshots every 500ms.

    Prefers the in-memory collector (always up-to-date) over SQLite
    snapshots (which may lag during Flock processing).  Falls back to
    SQLite only when there is no active in-memory collector.
    """
    global _last_finalized_snapshot, _last_finalized_at
    loop = asyncio.get_running_loop()
    while True:
        if await request.is_disconnected():
            break

        snapshot = None

        # 1) Prefer in-memory collector — always has real-time data
        collector = get_any_active_collector()
        if collector:
            snapshot = collector.snapshot()
            # Cache a copy so we can serve it during the grace period
            if snapshot and snapshot.get("finalized"):
                _last_finalized_snapshot = snapshot
                _last_finalized_at = asyncio.get_event_loop().time()

        # 2) Fall back to SQLite — try running runs first
        if not snapshot:
            try:
                snapshot = await loop.run_in_executor(
                    _db_executor,
                    lambda: event_store.get_latest_snapshot(None, only_running=True),
                )
            except Exception:
                snapshot = None

        # 3) Fall back to SQLite without only_running filter to get the
        #    most recently completed run (mirrors /dashboard/latest logic).
        if not snapshot:
            try:
                snapshot = await loop.run_in_executor(
                    _db_executor,
                    lambda: event_store.get_latest_snapshot(None),
                )
            except Exception:
                snapshot = None

        # 4) During the grace period after finalization, continue serving
        #    the last finalized snapshot so the dashboard doesn't flicker.
        if not snapshot and _last_finalized_snapshot:
            elapsed = asyncio.get_event_loop().time() - _last_finalized_at
            if elapsed < _POST_RUN_GRACE_SECS:
                snapshot = _last_finalized_snapshot
            else:
                _last_finalized_snapshot = None

        if snapshot:
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
                "query": data.get("query", ""),
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
    loop = asyncio.get_running_loop()

    # Get runs from SQLite (includes currently running ones)
    try:
        db_runs = await loop.run_in_executor(_db_executor, event_store.get_all_runs)
    except Exception:
        db_runs = []

    # Also get legacy JSON file runs
    file_runs = _load_runs()

    # Merge: SQLite runs take priority, add file-only runs
    seen_ids = {r["session_id"] for r in db_runs}
    merged = db_runs + [r for r in file_runs if r.get("session_id") not in seen_ids]

    return JSONResponse({"runs": merged})


async def dashboard_latest(request: Request) -> JSONResponse:
    """GET /dashboard/latest — most recent snapshot (from memory, SQLite, or files).

    Priority mirrors the SSE generator: prefer in-memory collector (real-time),
    fall back to SQLite, then to saved JSON files.
    """
    loop = asyncio.get_running_loop()

    # 1) Prefer in-memory collector — always has real-time data
    collector = get_any_active_collector()
    if collector:
        return JSONResponse(collector.snapshot())

    # 2) Fall back to SQLite
    try:
        snapshot = await loop.run_in_executor(
            _db_executor, event_store.get_latest_snapshot, None
        )
    except Exception:
        snapshot = None
    if snapshot:
        return JSONResponse(snapshot)

    # 3) Otherwise return the most recent saved run
    runs = _load_runs()
    if not runs:
        return JSONResponse({"status": "idle", "message": "No active pipeline run"})

    data = _load_run_by_id(runs[0]["session_id"][:8])
    if data:
        return JSONResponse(data)
    return JSONResponse({"status": "idle", "message": "No active pipeline run"})


async def dashboard_run_detail(request: Request) -> JSONResponse:
    """GET /dashboard/run/{session_id} — fetch a specific run."""
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    data = _load_run_by_id(session_id)
    if data is None:
        # Fall back to SQLite (covers currently-running and finalized runs)
        try:
            data = await loop.run_in_executor(
                _db_executor, event_store.get_run_detail, session_id
            )
        except Exception:
            data = None
    if data:
        return JSONResponse(data)
    return JSONResponse({"error": f"Run {session_id} not found"}, status_code=404)


async def dashboard_html(request: Request) -> HTMLResponse:
    """GET /dashboard/html/{session_id} — render a run as HTML."""
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    data = _load_run_by_id(session_id)
    if not data:
        # Fall back to SQLite
        try:
            data = await loop.run_in_executor(
                _db_executor, event_store.get_run_detail, session_id
            )
        except Exception:
            data = None
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


# ── Tracing REST endpoints ────────────────────────────────────────────


def _parse_int_param(value: str | None, default: int | None = None) -> int | None:
    """Safely parse an integer query parameter, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


async def dashboard_trace_summary(request: Request) -> JSONResponse:
    """GET /dashboard/traces/{session_id} — compact trace summary.

    Returns counts of algorithm traces, LLM traces, corpus snapshots,
    quality regressions, and per-iteration algorithm breakdowns.
    Designed for the improvement-loop consumer.
    """
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    try:
        summary = await loop.run_in_executor(
            _db_executor, event_store.get_trace_summary, session_id,
        )
    except Exception:
        summary = None
    if summary:
        return JSONResponse(summary)
    return JSONResponse({"error": f"No traces for {session_id}"}, status_code=404)


async def dashboard_algorithm_traces(request: Request) -> JSONResponse:
    """GET /dashboard/traces/{session_id}/algorithms — per-algorithm traces.

    Optional query param: ?iteration=N to filter by iteration.
    Returns the full before/after snapshots and details for each algorithm.
    """
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    iteration = _parse_int_param(request.query_params.get("iteration"))
    try:
        traces = await loop.run_in_executor(
            _db_executor,
            lambda: event_store.get_algorithm_traces(session_id, iteration),
        )
    except Exception:
        traces = []
    return JSONResponse({"session_id": session_id, "traces": traces})


async def dashboard_llm_traces(request: Request) -> JSONResponse:
    """GET /dashboard/traces/{session_id}/llm — Flock LLM prompt/response log.

    Optional query params: ?iteration=N&limit=500
    Returns captured prompts, responses, callers, and durations.
    """
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    iteration = _parse_int_param(request.query_params.get("iteration"))
    limit = _parse_int_param(request.query_params.get("limit"), default=500) or 500
    try:
        traces = await loop.run_in_executor(
            _db_executor,
            lambda: event_store.get_llm_traces(session_id, iteration, limit),
        )
    except Exception:
        traces = []
    return JSONResponse({"session_id": session_id, "traces": traces})


async def dashboard_corpus_snapshots(request: Request) -> JSONResponse:
    """GET /dashboard/traces/{session_id}/corpus — corpus state snapshots.

    Optional query param: ?iteration=N
    Returns iteration-level corpus snapshots with status counts,
    score summaries, and condition lists for iteration-over-iteration diffs.
    """
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    iteration = _parse_int_param(request.query_params.get("iteration"))
    try:
        snaps = await loop.run_in_executor(
            _db_executor,
            lambda: event_store.get_corpus_snapshots(session_id, iteration),
        )
    except Exception:
        snaps = []
    return JSONResponse({"session_id": session_id, "snapshots": snaps})


async def dashboard_quality_regressions(request: Request) -> JSONResponse:
    """GET /dashboard/traces/{session_id}/regressions — quality regression flags.

    Optional query param: ?severity=warning (filters to warning+critical)
    Returns flagged quality decreases after algorithm runs.
    """
    loop = asyncio.get_running_loop()
    session_id = request.path_params.get("session_id", "")
    severity = request.query_params.get("severity", "info")
    try:
        regs = await loop.run_in_executor(
            _db_executor,
            lambda: event_store.get_quality_regressions(session_id, severity),
        )
    except Exception:
        regs = []
    return JSONResponse({"session_id": session_id, "regressions": regs})


def mount_dashboard_routes(app: FastAPI) -> None:
    """Mount all dashboard endpoints on the FastAPI app."""
    app.add_api_route("/dashboard/stream", dashboard_stream, methods=["GET"])
    app.add_api_route("/dashboard/latest", dashboard_latest, methods=["GET"])
    app.add_api_route("/dashboard/runs", dashboard_runs, methods=["GET"])
    app.add_api_route("/dashboard/run/{session_id}", dashboard_run_detail, methods=["GET"])
    app.add_api_route("/dashboard/html/{session_id}", dashboard_html, methods=["GET"])
    app.add_api_route("/dashboard/html", dashboard_html_latest, methods=["GET"])

    # Tracing endpoints (for after-run improvement loops)
    app.add_api_route(
        "/dashboard/traces/{session_id}",
        dashboard_trace_summary, methods=["GET"],
    )
    app.add_api_route(
        "/dashboard/traces/{session_id}/algorithms",
        dashboard_algorithm_traces, methods=["GET"],
    )
    app.add_api_route(
        "/dashboard/traces/{session_id}/llm",
        dashboard_llm_traces, methods=["GET"],
    )
    app.add_api_route(
        "/dashboard/traces/{session_id}/corpus",
        dashboard_corpus_snapshots, methods=["GET"],
    )
    app.add_api_route(
        "/dashboard/traces/{session_id}/regressions",
        dashboard_quality_regressions, methods=["GET"],
    )
    logger.info("Dashboard routes mounted at /dashboard/* (including tracing endpoints)")
