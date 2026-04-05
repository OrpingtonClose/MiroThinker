# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI SSE server + REST API for the ADK execution dashboard.

Provides:
* ``/`` — main dashboard HTML page
* ``/api/events/{session_id}`` — SSE stream of real-time events
* ``/api/metrics/{session_id}`` — accumulated metrics JSON
* ``/api/sessions`` — list of tracked sessions
* ``/api/reports`` — list of saved post-hoc report files
* ``/api/reports/{filename}`` — load a specific saved report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

_DASHBOARD_DIR = Path(__file__).parent

app = FastAPI(title="MiroThinker ADK Dashboard")

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=str(_DASHBOARD_DIR / "static")),
    name="static",
)

# Templates
templates = Jinja2Templates(directory=str(_DASHBOARD_DIR / "templates"))

# Global collector registry  (session_id -> DashboardCollector)
# Populated by main.py when it creates a collector
collectors: dict[str, Any] = {}


# ── HTML ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main dashboard page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "sessions": list(collectors.keys())},
    )


# ── SSE ───────────────────────────────────────────────────────────────────────

@app.get("/api/events/{session_id}")
async def stream_events(session_id: str):
    """SSE endpoint for real-time events."""
    collector = collectors.get(session_id)
    if not collector:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    async def event_generator():
        while True:
            event = await collector.event_queue.get()
            if event is None:
                # Sentinel: session ended
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                break
            payload = {
                "type": event.event_type.value,
                "timestamp": event.timestamp,
                "agent": event.agent_name,
                "turn": event.turn,
                "attempt": event.attempt,
                "data": event.data,
            }
            yield f"data: {json.dumps(payload, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── REST ──────────────────────────────────────────────────────────────────────

@app.get("/api/metrics/{session_id}")
async def get_metrics(session_id: str):
    """Get accumulated metrics for a session (post-hoc or live)."""
    collector = collectors.get(session_id)
    if not collector:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return collector.to_metrics_dict()


@app.get("/api/sessions")
async def list_sessions():
    """List all active/completed sessions."""
    return [
        {
            "session_id": sid,
            "query": c.query,
            "turns": c.current_turn,
            "attempt": c.current_attempt,
            "elapsed_secs": c.elapsed_secs(),
            "tool_calls": c.total_tool_calls,
            "llm_calls": c.total_llm_calls,
        }
        for sid, c in collectors.items()
    ]


@app.get("/api/reports")
async def list_reports():
    """List saved post-hoc report files."""
    from dashboard.collector import DashboardCollector

    reports = DashboardCollector.list_saved_reports()
    return [{"filename": r, "name": Path(r).stem} for r in reports]


@app.get("/api/reports/{filename:path}")
async def load_report(filename: str):
    """Load a specific saved report."""
    from dashboard.collector import DashboardCollector

    # Validate that the file is within the expected reports directory
    reports_dir = Path("dashboard_logs").resolve()
    target = (reports_dir / Path(filename).name).resolve()
    if not str(target).startswith(str(reports_dir)):
        return JSONResponse({"error": "Invalid path"}, status_code=400)

    try:
        data = DashboardCollector.load(str(target))
        return data
    except FileNotFoundError:
        return JSONResponse({"error": "Report not found"}, status_code=404)
