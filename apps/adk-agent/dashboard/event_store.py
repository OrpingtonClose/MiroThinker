# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""SQLite-backed event store for pipeline dashboard observability.

Uses WAL mode so writers (the pipeline collector) and readers (the SSE
endpoint) never block each other.  All writes happen synchronously from
the collector thread; all reads happen from a separate connection in a
thread-pool executor so the async event loop is never blocked.

Schema:
    runs    — one row per pipeline invocation
    events  — append-only log of every collector event
    snapshots — periodic KPI snapshots written by the collector

The SSE endpoint reads the latest snapshot + recent events from SQLite
instead of polling the in-memory collector, so it works even when the
event loop is saturated by LLM calls.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH = os.environ.get(
    "DASHBOARD_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard.db"),
)

# Module-level lock for writer connections
_writer_lock = threading.Lock()
_writer_conn: sqlite3.Connection | None = None


def _get_writer() -> sqlite3.Connection:
    """Get or create the singleton writer connection (thread-safe)."""
    global _writer_conn
    with _writer_lock:
        if _writer_conn is None:
            _writer_conn = sqlite3.connect(_DB_PATH, timeout=10)
            _writer_conn.execute("PRAGMA journal_mode=WAL")
            _writer_conn.execute("PRAGMA synchronous=NORMAL")
            _writer_conn.execute("PRAGMA busy_timeout=5000")
            _init_schema(_writer_conn)
            logger.info("Dashboard SQLite writer opened: %s", _DB_PATH)
        return _writer_conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            session_id   TEXT PRIMARY KEY,
            query        TEXT NOT NULL DEFAULT '',
            started_at   REAL NOT NULL,
            finalized_at REAL,
            elapsed_secs REAL,
            status       TEXT NOT NULL DEFAULT 'running',  -- running | completed | error
            result_json  TEXT  -- full finalized data (JSON)
        );

        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            event_type   TEXT NOT NULL,
            agent        TEXT NOT NULL DEFAULT '',
            phase        TEXT NOT NULL DEFAULT '',
            data_json    TEXT NOT NULL DEFAULT '{}',
            timestamp    REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES runs(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_session
            ON events(session_id, id);

        CREATE INDEX IF NOT EXISTS idx_events_session_time
            ON events(session_id, timestamp);

        CREATE TABLE IF NOT EXISTS snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            timestamp    REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES runs(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_session
            ON snapshots(session_id, id DESC);
    """)
    conn.commit()


# ── Writer API (called from collector, may be sync) ──────────────


def insert_run(session_id: str, query: str) -> None:
    """Record the start of a new pipeline run."""
    conn = _get_writer()
    conn.execute(
        "INSERT OR REPLACE INTO runs (session_id, query, started_at, status) "
        "VALUES (?, ?, ?, 'running')",
        (session_id, query, time.time()),
    )
    conn.commit()


def insert_event(
    session_id: str,
    event_type: str,
    agent: str,
    phase: str,
    data: dict[str, Any],
    timestamp: float,
) -> None:
    """Append an event to the log."""
    conn = _get_writer()
    conn.execute(
        "INSERT INTO events (session_id, event_type, agent, phase, data_json, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, event_type, agent, phase, json.dumps(data, default=str), timestamp),
    )
    conn.commit()


def insert_snapshot(session_id: str, snapshot: dict[str, Any]) -> None:
    """Write a periodic snapshot (called by the collector)."""
    conn = _get_writer()
    conn.execute(
        "INSERT INTO snapshots (session_id, snapshot_json, timestamp) "
        "VALUES (?, ?, ?)",
        (session_id, json.dumps(snapshot, default=str), time.time()),
    )
    conn.commit()


def finalize_run(
    session_id: str,
    status: str = "completed",
    elapsed_secs: float = 0.0,
    result_json: str = "",
) -> None:
    """Mark a run as complete and store the finalized data."""
    conn = _get_writer()
    conn.execute(
        "UPDATE runs SET status = ?, finalized_at = ?, elapsed_secs = ?, result_json = ? "
        "WHERE session_id = ?",
        (status, time.time(), elapsed_secs, result_json, session_id),
    )
    conn.commit()


# ── Reader API (called from SSE endpoint, thread-safe) ───────────


def _get_reader() -> sqlite3.Connection:
    """Create a new read-only connection for the current thread.

    Each reader gets its own connection so there's no contention with
    the writer.  WAL mode allows concurrent reads.
    """
    conn = sqlite3.connect(_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_snapshot(session_id: str | None = None) -> dict[str, Any] | None:
    """Read the most recent snapshot for a session (or the most recent run)."""
    conn = _get_reader()
    try:
        if session_id:
            row = conn.execute(
                "SELECT snapshot_json FROM snapshots "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        else:
            # Get the most recent running session's latest snapshot
            row = conn.execute(
                "SELECT s.snapshot_json FROM snapshots s "
                "JOIN runs r ON s.session_id = r.session_id "
                "WHERE r.status = 'running' "
                "ORDER BY s.id DESC LIMIT 1",
            ).fetchone()
            if row is None:
                # Fall back to most recent session regardless of status
                row = conn.execute(
                    "SELECT snapshot_json FROM snapshots "
                    "ORDER BY id DESC LIMIT 1",
                ).fetchone()
        if row:
            return json.loads(row[0])
        return None
    finally:
        conn.close()


def get_recent_events(
    session_id: str, since_id: int = 0, limit: int = 500
) -> list[dict[str, Any]]:
    """Read events newer than since_id for a session."""
    conn = _get_reader()
    try:
        rows = conn.execute(
            "SELECT id, event_type, agent, phase, data_json, timestamp "
            "FROM events WHERE session_id = ? AND id > ? "
            "ORDER BY id ASC LIMIT ?",
            (session_id, since_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "event_type": r[1],
                "agent": r[2],
                "phase": r[3],
                "data": json.loads(r[4]),
                "timestamp": r[5],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_active_session_id() -> str | None:
    """Get the session_id of the currently running pipeline."""
    conn = _get_reader()
    try:
        row = conn.execute(
            "SELECT session_id FROM runs WHERE status = 'running' "
            "ORDER BY started_at DESC LIMIT 1",
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_all_runs() -> list[dict[str, Any]]:
    """List all runs, newest first."""
    conn = _get_reader()
    try:
        rows = conn.execute(
            "SELECT session_id, query, started_at, finalized_at, "
            "elapsed_secs, status FROM runs ORDER BY started_at DESC",
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "query": r[1],
                "started_at": r[2],
                "finalized_at": r[3],
                "elapsed_secs": r[4],
                "status": r[5],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_run_detail(session_id: str) -> dict[str, Any] | None:
    """Load full finalized data for a run."""
    conn = _get_reader()
    try:
        row = conn.execute(
            "SELECT result_json FROM runs WHERE session_id LIKE ? || '%' "
            "AND result_json IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row and row[0]:
            return json.loads(row[0])
        return None
    finally:
        conn.close()


def get_event_count(session_id: str) -> int:
    """Get the total number of events for a session."""
    conn = _get_reader()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()
