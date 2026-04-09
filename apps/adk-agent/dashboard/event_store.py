# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""SQLite-backed event store for pipeline dashboard observability.

Uses WAL mode so writers (the pipeline collector) and readers (the SSE
endpoint) never block each other.  All writes happen synchronously from
the collector thread; all reads happen from a separate connection in a
thread-pool executor so the async event loop is never blocked.

Schema:
    runs    -- one row per pipeline invocation
    events  -- append-only log of every collector event
    snapshots -- periodic KPI snapshots written by the collector
    algorithm_traces -- per-algorithm before/after trace from Flock battery
    llm_traces -- Flock LLM prompt/response capture
    corpus_snapshots -- iteration-level corpus state snapshots
    quality_regressions -- flagged quality decreases after algorithm runs

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
            _writer_conn = sqlite3.connect(
                _DB_PATH, timeout=10, check_same_thread=False
            )
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

        -- Per-algorithm trace: what each Flock battery algorithm did
        CREATE TABLE IF NOT EXISTS algorithm_traces (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            iteration       INTEGER NOT NULL DEFAULT 0,
            algorithm_name  TEXT NOT NULL,
            affected_count  INTEGER NOT NULL DEFAULT 0,
            before_snapshot TEXT NOT NULL DEFAULT '{}',  -- JSON: score distributions before
            after_snapshot  TEXT NOT NULL DEFAULT '{}',   -- JSON: score distributions after
            details_json    TEXT NOT NULL DEFAULT '{}',   -- algorithm-specific decisions
            duration_ms     REAL NOT NULL DEFAULT 0.0,
            timestamp       REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES runs(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_algo_traces_session
            ON algorithm_traces(session_id, iteration, algorithm_name);

        -- Flock LLM prompt/response capture
        CREATE TABLE IF NOT EXISTS llm_traces (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            iteration       INTEGER NOT NULL DEFAULT 0,
            caller          TEXT NOT NULL DEFAULT '',     -- e.g. 'score_single', 'detect_contradictions'
            prompt          TEXT NOT NULL,
            response        TEXT NOT NULL DEFAULT '',
            model           TEXT NOT NULL DEFAULT '',
            duration_ms     REAL NOT NULL DEFAULT 0.0,
            timestamp       REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES runs(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_llm_traces_session
            ON llm_traces(session_id, iteration);

        -- Corpus state snapshots at iteration boundaries
        CREATE TABLE IF NOT EXISTS corpus_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            iteration       INTEGER NOT NULL,
            phase           TEXT NOT NULL DEFAULT '',      -- 'pre_battery', 'post_battery', 'post_synthesis'
            total_conditions INTEGER NOT NULL DEFAULT 0,
            status_counts   TEXT NOT NULL DEFAULT '{}',   -- JSON: {ready: N, merged: N, ...}
            score_summary   TEXT NOT NULL DEFAULT '{}',   -- JSON: {mean_quality: X, median: X, ...}
            conditions_json  TEXT NOT NULL DEFAULT '[]',  -- JSON array of condition summaries
            timestamp       REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES runs(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_corpus_snap_session
            ON corpus_snapshots(session_id, iteration);

        -- Quality regression flags
        CREATE TABLE IF NOT EXISTS quality_regressions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            iteration       INTEGER NOT NULL,
            algorithm_name  TEXT NOT NULL,
            metric_name     TEXT NOT NULL,                -- e.g. 'mean_composite_quality'
            before_value    REAL NOT NULL,
            after_value     REAL NOT NULL,
            delta           REAL NOT NULL,
            severity        TEXT NOT NULL DEFAULT 'info', -- info | warning | critical
            timestamp       REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES runs(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_quality_reg_session
            ON quality_regressions(session_id, iteration);
    """)
    conn.commit()


# ── Writer API (called from collector, may be sync) ──────────────


def insert_run(session_id: str, query: str) -> None:
    """Record the start of a new pipeline run."""
    conn = _get_writer()
    with _writer_lock:
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
    with _writer_lock:
        conn.execute(
            "INSERT INTO events (session_id, event_type, agent, phase, data_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, event_type, agent, phase, json.dumps(data, default=str), timestamp),
        )
        conn.commit()


def insert_snapshot(session_id: str, snapshot: dict[str, Any]) -> None:
    """Write a periodic snapshot (called by the collector)."""
    conn = _get_writer()
    with _writer_lock:
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
    with _writer_lock:
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
    the writer.  WAL mode allows concurrent reads.  We also ensure the
    schema exists so readers work even if no pipeline has run yet.
    """
    conn = sqlite3.connect(_DB_PATH, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(conn)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_latest_snapshot(
    session_id: str | None = None,
    only_running: bool = False,
) -> dict[str, Any] | None:
    """Read the most recent snapshot for a session (or the most recent run).

    If *only_running* is True, only return snapshots from sessions whose
    status is 'running'.  This prevents the SSE stream from returning
    stale completed-run data when no pipeline is active.
    """
    conn = _get_reader()
    try:
        if session_id:
            if only_running:
                row = conn.execute(
                    "SELECT s.snapshot_json FROM snapshots s "
                    "JOIN runs r ON s.session_id = r.session_id "
                    "WHERE s.session_id = ? AND r.status = 'running' "
                    "ORDER BY s.id DESC LIMIT 1",
                    (session_id,),
                ).fetchone()
            else:
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
            if row is None and not only_running:
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
    """List all runs, newest first.  Includes KPI data from latest snapshot."""
    conn = _get_reader()
    try:
        rows = conn.execute(
            "SELECT session_id, query, started_at, finalized_at, "
            "elapsed_secs, status FROM runs ORDER BY started_at DESC",
        ).fetchall()
        result = []
        for r in rows:
            sid = r[0]
            run: dict[str, Any] = {
                "session_id": sid,
                "query": r[1],
                "started_at": r[2],
                "finalized_at": r[3],
                "elapsed_secs": r[4],
                "status": r[5],
                "kpi": {},
            }
            # Enrich with KPI from latest snapshot
            snap_row = conn.execute(
                "SELECT snapshot_json FROM snapshots "
                "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
            if snap_row:
                try:
                    snap = json.loads(snap_row[0])
                    run["kpi"] = snap.get("kpi", {})
                except (json.JSONDecodeError, TypeError):
                    pass
            result.append(run)
        return result
    finally:
        conn.close()


def get_run_detail(session_id: str) -> dict[str, Any] | None:
    """Load full finalized data for a run."""
    conn = _get_reader()
    try:
        escaped = session_id.replace("%", "\\%").replace("_", "\\_")
        row = conn.execute(
            "SELECT result_json FROM runs WHERE session_id LIKE ? || '%' ESCAPE '\\' "
            "AND result_json IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (escaped,),
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


# ── Tracing writer API ────────────────────────────────────────────


def insert_algorithm_trace(
    session_id: str,
    iteration: int,
    algorithm_name: str,
    affected_count: int,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    details: dict[str, Any],
    duration_ms: float,
) -> None:
    """Record a single algorithm's execution trace."""
    conn = _get_writer()
    with _writer_lock:
        conn.execute(
            "INSERT INTO algorithm_traces "
            "(session_id, iteration, algorithm_name, affected_count, "
            "before_snapshot, after_snapshot, details_json, duration_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, iteration, algorithm_name, affected_count,
                json.dumps(before_snapshot, default=str),
                json.dumps(after_snapshot, default=str),
                json.dumps(details, default=str),
                duration_ms, time.time(),
            ),
        )
        conn.commit()


def insert_llm_trace(
    session_id: str,
    iteration: int,
    caller: str,
    prompt: str,
    response: str,
    model: str,
    duration_ms: float,
) -> None:
    """Record a Flock LLM prompt/response pair."""
    conn = _get_writer()
    with _writer_lock:
        conn.execute(
            "INSERT INTO llm_traces "
            "(session_id, iteration, caller, prompt, response, model, "
            "duration_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, iteration, caller, prompt, response,
                model, duration_ms, time.time(),
            ),
        )
        conn.commit()


def insert_corpus_snapshot(
    session_id: str,
    iteration: int,
    phase: str,
    total_conditions: int,
    status_counts: dict[str, int],
    score_summary: dict[str, float],
    conditions: list[dict[str, Any]],
) -> None:
    """Record a corpus state snapshot at an iteration boundary."""
    conn = _get_writer()
    with _writer_lock:
        conn.execute(
            "INSERT INTO corpus_snapshots "
            "(session_id, iteration, phase, total_conditions, "
            "status_counts, score_summary, conditions_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, iteration, phase, total_conditions,
                json.dumps(status_counts, default=str),
                json.dumps(score_summary, default=str),
                json.dumps(conditions, default=str),
                time.time(),
            ),
        )
        conn.commit()


def insert_quality_regression(
    session_id: str,
    iteration: int,
    algorithm_name: str,
    metric_name: str,
    before_value: float,
    after_value: float,
    severity: str = "info",
) -> None:
    """Flag a quality regression after an algorithm run."""
    conn = _get_writer()
    with _writer_lock:
        conn.execute(
            "INSERT INTO quality_regressions "
            "(session_id, iteration, algorithm_name, metric_name, "
            "before_value, after_value, delta, severity, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, iteration, algorithm_name, metric_name,
                before_value, after_value, after_value - before_value,
                severity, time.time(),
            ),
        )
        conn.commit()


# ── Tracing reader API ────────────────────────────────────────────


def get_algorithm_traces(
    session_id: str, iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Read algorithm traces for a session, optionally filtered by iteration."""
    conn = _get_reader()
    try:
        if iteration is not None:
            rows = conn.execute(
                "SELECT id, iteration, algorithm_name, affected_count, "
                "before_snapshot, after_snapshot, details_json, "
                "duration_ms, timestamp "
                "FROM algorithm_traces WHERE session_id = ? AND iteration = ? "
                "ORDER BY id ASC",
                (session_id, iteration),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, iteration, algorithm_name, affected_count, "
                "before_snapshot, after_snapshot, details_json, "
                "duration_ms, timestamp "
                "FROM algorithm_traces WHERE session_id = ? "
                "ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "id": r[0], "iteration": r[1],
                "algorithm_name": r[2], "affected_count": r[3],
                "before": json.loads(r[4]), "after": json.loads(r[5]),
                "details": json.loads(r[6]),
                "duration_ms": r[7], "timestamp": r[8],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_llm_traces(
    session_id: str, iteration: int | None = None, limit: int = 500,
) -> list[dict[str, Any]]:
    """Read Flock LLM traces for a session."""
    conn = _get_reader()
    try:
        if iteration is not None:
            rows = conn.execute(
                "SELECT id, iteration, caller, prompt, response, model, "
                "duration_ms, timestamp "
                "FROM llm_traces WHERE session_id = ? AND iteration = ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, iteration, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, iteration, caller, prompt, response, model, "
                "duration_ms, timestamp "
                "FROM llm_traces WHERE session_id = ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "id": r[0], "iteration": r[1], "caller": r[2],
                "prompt": r[3], "response": r[4], "model": r[5],
                "duration_ms": r[6], "timestamp": r[7],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_corpus_snapshots(
    session_id: str, iteration: int | None = None,
) -> list[dict[str, Any]]:
    """Read corpus snapshots for a session."""
    conn = _get_reader()
    try:
        if iteration is not None:
            rows = conn.execute(
                "SELECT id, iteration, phase, total_conditions, "
                "status_counts, score_summary, conditions_json, timestamp "
                "FROM corpus_snapshots WHERE session_id = ? AND iteration = ? "
                "ORDER BY id ASC",
                (session_id, iteration),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, iteration, phase, total_conditions, "
                "status_counts, score_summary, conditions_json, timestamp "
                "FROM corpus_snapshots WHERE session_id = ? "
                "ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "id": r[0], "iteration": r[1], "phase": r[2],
                "total_conditions": r[3],
                "status_counts": json.loads(r[4]),
                "score_summary": json.loads(r[5]),
                "conditions": json.loads(r[6]),
                "timestamp": r[7],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_quality_regressions(
    session_id: str, min_severity: str = "info",
) -> list[dict[str, Any]]:
    """Read quality regression flags for a session."""
    severities = ["info", "warning", "critical"]
    min_idx = severities.index(min_severity) if min_severity in severities else 0
    allowed = severities[min_idx:]
    placeholders = ",".join("?" * len(allowed))
    conn = _get_reader()
    try:
        rows = conn.execute(
            f"SELECT id, iteration, algorithm_name, metric_name, "
            f"before_value, after_value, delta, severity, timestamp "
            f"FROM quality_regressions WHERE session_id = ? "
            f"AND severity IN ({placeholders}) "
            f"ORDER BY id ASC",
            (session_id, *allowed),
        ).fetchall()
        return [
            {
                "id": r[0], "iteration": r[1],
                "algorithm_name": r[2], "metric_name": r[3],
                "before_value": r[4], "after_value": r[5],
                "delta": r[6], "severity": r[7], "timestamp": r[8],
            }
            for r in rows
        ]
    finally:
        conn.close()


def get_trace_summary(session_id: str) -> dict[str, Any]:
    """Return a compact summary of all traces for a session.

    Designed for the improvement-loop consumer: one JSON object with
    counts, regressions, and per-iteration algorithm summaries.
    """
    conn = _get_reader()
    try:
        algo_count = conn.execute(
            "SELECT COUNT(*) FROM algorithm_traces WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        llm_count = conn.execute(
            "SELECT COUNT(*) FROM llm_traces WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        snap_count = conn.execute(
            "SELECT COUNT(*) FROM corpus_snapshots WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        reg_count = conn.execute(
            "SELECT COUNT(*) FROM quality_regressions WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

        # Per-iteration algorithm summaries
        iter_rows = conn.execute(
            "SELECT iteration, algorithm_name, affected_count, duration_ms "
            "FROM algorithm_traces WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        iterations: dict[int, list[dict[str, Any]]] = {}
        for r in iter_rows:
            it = r[0]
            iterations.setdefault(it, []).append({
                "algorithm": r[1], "affected": r[2], "duration_ms": r[3],
            })

        # Total Flock LLM time
        total_llm_ms = conn.execute(
            "SELECT COALESCE(SUM(duration_ms), 0) FROM llm_traces "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]

        return {
            "session_id": session_id,
            "algorithm_trace_count": algo_count,
            "llm_trace_count": llm_count,
            "corpus_snapshot_count": snap_count,
            "quality_regression_count": reg_count,
            "total_flock_llm_ms": total_llm_ms,
            "iterations": iterations,
        }
    finally:
        conn.close()
