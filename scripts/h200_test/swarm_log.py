#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Structured SQLite logging for the swarm pipeline.

Captures EVERYTHING to a queryable SQLite database:
  - All Python logging events (every logger.info/debug/warning/error call)
  - Full LLM call traces (prompt, response, model, tokens, timing)
  - Enrichment results (title, URL, snippet, angle, admitted/rejected)

Usage:
    from swarm_log import init_logging, log_llm_call, log_enrichment_result

    # At startup — replaces logging.basicConfig()
    init_logging("swarm_log.db")

    # Existing logger.info() calls automatically land in SQLite.
    # No changes needed to existing code.

    # For structured LLM tracking:
    log_llm_call(db, phase="map", worker="tren", prompt=..., response=..., ...)

    # For enrichment tracking:
    log_enrichment_result(db, angle="tren", query="...", title="...", ...)

Query examples:
    SELECT * FROM log_events WHERE logger = 'swarm.engine' ORDER BY timestamp;
    SELECT phase, worker, input_chars, output_chars, elapsed_s FROM llm_calls;
    SELECT angle, query, title, admitted FROM enrichment_results WHERE admitted = 0;
    SELECT * FROM llm_calls WHERE phase = 'queen_merge';
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path


# ── SQLite logging handler ────────────────────────────────────────────
# Hooks into Python's standard logging module. Every logger.*() call
# in the process lands here automatically.

class SQLiteHandler(logging.Handler):
    """Logging handler that writes all log records to a SQLite database."""

    def __init__(self, db_path: str) -> None:
        super().__init__()
        self._db_path = db_path
        self._local = threading.local()
        # Create table on init (using a one-off connection)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS log_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL    NOT NULL,
                level     TEXT    NOT NULL,
                logger    TEXT    NOT NULL,
                message   TEXT    NOT NULL,
                pathname  TEXT,
                lineno    INTEGER,
                func_name TEXT,
                thread    TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    REAL    NOT NULL,
                phase        TEXT    NOT NULL,
                worker       TEXT,
                model        TEXT,
                prompt       TEXT    NOT NULL,
                response     TEXT    NOT NULL,
                input_chars  INTEGER,
                output_chars INTEGER,
                max_tokens   INTEGER,
                temperature  REAL,
                elapsed_s    REAL,
                error        TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS enrichment_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                angle       TEXT    NOT NULL,
                query       TEXT    NOT NULL,
                backend     TEXT,
                title       TEXT,
                url         TEXT,
                snippet     TEXT,
                admitted    INTEGER NOT NULL DEFAULT 1,
                reject_reason TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS worker_outputs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   REAL    NOT NULL,
                phase       TEXT    NOT NULL,
                worker      TEXT    NOT NULL,
                angle       TEXT,
                round       INTEGER,
                output      TEXT    NOT NULL,
                output_chars INTEGER,
                info_gain   REAL
            )
        """)
        conn.commit()
        conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Thread-local connection — SQLite connections aren't thread-safe."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
        return self._local.conn

    def emit(self, record: logging.LogRecord) -> None:
        try:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO log_events
                   (timestamp, level, logger, message, pathname, lineno, func_name, thread)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.created,
                    record.levelname,
                    record.name,
                    self.format(record),
                    record.pathname,
                    record.lineno,
                    record.funcName,
                    record.threadName,
                ),
            )
            conn.commit()
        except Exception:
            self.handleError(record)


# ── Initialization ────────────────────────────────────────────────────

_db_path: str = ""
_handler: SQLiteHandler | None = None


def init_logging(
    db_path: str = "swarm_log.db",
    console_level: int = logging.INFO,
    sqlite_level: int = logging.DEBUG,
) -> str:
    """Initialize logging to both console and SQLite.

    Replaces logging.basicConfig(). Call once at startup.
    Returns the SQLite database path.
    """
    global _db_path, _handler

    _db_path = str(Path(db_path).resolve())

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s",
    ))

    # SQLite handler — captures everything including DEBUG
    _handler = SQLiteHandler(_db_path)
    _handler.setLevel(sqlite_level)
    _handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(_handler)

    logging.getLogger(__name__).info(
        "db_path=<%s> | structured SQLite logging initialized", _db_path,
    )
    return _db_path


def get_db_path() -> str:
    """Return the current SQLite log database path."""
    return _db_path


# ── Structured logging helpers ────────────────────────────────────────
# These write to dedicated tables for easy querying.
# They use the same SQLite database as the logging handler.

def _get_conn() -> sqlite3.Connection:
    """Get a connection to the log database."""
    return sqlite3.connect(_db_path)


def log_llm_call(
    phase: str,
    prompt: str,
    response: str,
    *,
    worker: str = "",
    model: str = "",
    max_tokens: int = 0,
    temperature: float = 0.0,
    elapsed_s: float = 0.0,
    error: str = "",
) -> None:
    """Log a full LLM call with prompt and response text."""
    if not _db_path:
        return
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO llm_calls
               (timestamp, phase, worker, model, prompt, response,
                input_chars, output_chars, max_tokens, temperature, elapsed_s, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), phase, worker, model, prompt, response,
                len(prompt), len(response), max_tokens, temperature,
                elapsed_s, error,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "error=<%s> | failed to log LLM call", exc,
        )


def log_enrichment_result(
    angle: str,
    query: str,
    *,
    backend: str = "",
    title: str = "",
    url: str = "",
    snippet: str = "",
    admitted: bool = True,
    reject_reason: str = "",
) -> None:
    """Log an enrichment search result (admitted or rejected)."""
    if not _db_path:
        return
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO enrichment_results
               (timestamp, angle, query, backend, title, url, snippet, admitted, reject_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), angle, query, backend, title, url, snippet,
                1 if admitted else 0, reject_reason,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "error=<%s> | failed to log enrichment result", exc,
        )


def log_worker_output(
    phase: str,
    worker: str,
    output: str,
    *,
    angle: str = "",
    round_num: int = 0,
    info_gain: float = 0.0,
) -> None:
    """Log a worker's output for a given phase/round."""
    if not _db_path:
        return
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO worker_outputs
               (timestamp, phase, worker, angle, round, output, output_chars, info_gain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), phase, worker, angle, round_num,
                output, len(output), info_gain,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "error=<%s> | failed to log worker output", exc,
        )
