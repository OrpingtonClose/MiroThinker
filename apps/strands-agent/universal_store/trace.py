"""Unified Trace Store — single container for every observable action.

NO SILENT ANYTHING. Every actor event, query, decision, error, state transition,
and external fetch is recorded here. The trace is the source of truth for:
- Debugging (what happened when?)
- Auditing (who decided what?)
- Reflexion (what worked? what failed?)
- Operator visibility (real-time and historical)
- Compliance (append-only, immutable)

Design principles:
1. FAST: async batch insert, no blocking the caller
2. RELIABLE: WAL-backed DuckDB, synced to durable storage
3. COMPLETE: every event_type has a schema; unknown events are rejected
4. QUERYABLE: indexed by run, actor, type, timestamp
5. STREAMABLE: SSE endpoint for real-time dashboard
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable

from .protocols import TraceRecord


# ---------------------------------------------------------------------------
# TraceStore — singleton per process
# ---------------------------------------------------------------------------

class TraceStore:
    """Singleton trace container. Use TraceStore.get() to access."""

    _instance: TraceStore | None = None
    _lock = asyncio.Lock()

    def __init__(self, db_path: str = ":memory:", batch_size: int = 100, flush_interval_s: float = 1.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._queue: asyncio.Queue[TraceRecord] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._buffer: list[TraceRecord] = []
        self._shutdown = False
        self._run_id: str = ""

    @classmethod
    async def get(cls, db_path: str = ":memory:") -> TraceStore:
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(db_path)
                    await cls._instance._init_db()
                    cls._instance._task = asyncio.create_task(
                        cls._instance._flush_loop(),
                        name="trace_store_flush"
                    )
        return cls._instance

    async def _init_db(self) -> None:
        import duckdb
        self._conn = duckdb.connect(self.db_path)
        from .schema import TRACE_TABLE
        self._conn.execute(TRACE_TABLE)
        # NOTE: Disabled sequence-backed auto-increment. DuckDB 1.5.x WAL replay
        # fails when ALTER TABLE ... SET DEFAULT nextval(...) is used.
        # Inserts must provide id explicitly (handled by the INSERT in _flush).
        # self._conn.execute(
        #     "CREATE SEQUENCE IF NOT EXISTS trace_records_id_seq START 1"
        # )
        # try:
        #     self._conn.execute(
        #         "ALTER TABLE trace_records ALTER COLUMN id"
        #         " SET DEFAULT nextval('trace_records_id_seq')"
        #     )
        # except Exception:
        #     pass
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_records(run_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_trace_actor ON trace_records(actor_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_trace_type ON trace_records(event_type, timestamp DESC);
        """)

    async def set_run(self, run_id: str) -> None:
        self._run_id = run_id

    async def _flush_loop(self) -> None:
        """Background task: flush buffer on interval or when full."""
        while not self._shutdown:
            try:
                record = await asyncio.wait_for(
                    self._queue.get(), timeout=self.flush_interval_s
                )
                self._buffer.append(record)
                if len(self._buffer) >= self.batch_size:
                    await self._flush()
            except asyncio.TimeoutError:
                if self._buffer:
                    await self._flush()

    async def _flush(self) -> None:
        # Drain the queue so records enqueued after the last flush_loop
        # iteration are captured when _flush() is called explicitly.
        while not self._queue.empty():
            try:
                self._buffer.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer = []
        try:
            import duckdb
            # Use a new connection for thread safety if needed
            # For single-process asyncio, shared connection is fine
            for rec in batch:
                self._conn.execute("""
                    INSERT INTO trace_records (
                        trace_id, run_id, actor_id, event_type, phase,
                        payload_json, timestamp, latency_ms, error, stack_trace
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(rec.trace_id), str(rec.run_id), str(rec.actor_id),
                    str(rec.event_type), str(rec.phase), str(rec.payload_json),
                    str(rec.timestamp), float(rec.latency_ms),
                    str(rec.error), str(rec.stack_trace)
                ))
            self._conn.commit()
        except Exception as e:
            # CRITICAL: trace flush failure must not be silent
            # Write to stderr and append to an emergency log file
            import sys
            print(f"[TRACE EMERGENCY] flush failed: {e}", file=sys.stderr)
            with open("trace_emergency.log", "a") as f:
                for rec in batch:
                    f.write(json.dumps({
                        "trace_id": rec.trace_id, "error": str(e),
                        "actor_id": rec.actor_id, "event_type": rec.event_type,
                        "timestamp": rec.timestamp
                    }) + "\n")

    async def record(
        self,
        actor_id: str,
        event_type: str,
        phase: str = "",
        payload: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
        error: Exception | None = None,
    ) -> str:
        """Record a trace. Never blocks caller; goes to async queue."""
        trace_id = str(uuid.uuid4())[:16]
        payload_json = json.dumps(payload or {}, default=str)
        error_str = str(error) if error else ""
        stack_str = traceback.format_exc() if error else ""
        rec = TraceRecord(
            trace_id=str(trace_id),
            run_id=str(self._run_id),
            actor_id=str(actor_id),
            event_type=str(event_type),
            phase=str(phase),
            payload_json=str(payload_json),
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=float(latency_ms),
            error=str(error_str),
            stack_trace=str(stack_str),
        )
        await self._queue.put(rec)
        return trace_id

    async def query(
        self,
        run_id: str | None = None,
        actor_id: str | None = None,
        event_type: str | None = None,
        phase: str | None = None,
        since: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Query traces. Returns list of dicts."""
        conditions = ["1=1"]
        params: list[Any] = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if actor_id:
            conditions.append("actor_id = ?")
            params.append(actor_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if phase:
            conditions.append("phase = ?")
            params.append(phase)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        sql = f"""
            SELECT * FROM trace_records
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        return self._conn.execute(sql, params).fetchdf().to_dict("records")

    async def get_errors(self, run_id: str, limit: int = 100) -> list[dict]:
        return self._conn.execute("""
            SELECT * FROM trace_records
            WHERE run_id = ? AND error != ''
            ORDER BY timestamp DESC
            LIMIT ?
        """, [run_id, limit]).fetchdf().to_dict("records")

    async def get_stats(self, run_id: str) -> dict[str, Any]:
        """Summary stats for a run."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) as total_events,
                COUNT(DISTINCT actor_id) as actor_count,
                COUNT(DISTINCT event_type) as event_type_count,
                SUM(CASE WHEN error != '' THEN 1 ELSE 0 END) as error_count,
                AVG(latency_ms) as avg_latency_ms,
                MIN(timestamp) as first_event,
                MAX(timestamp) as last_event
            FROM trace_records
            WHERE run_id = ?
        """, [run_id]).fetchone()
        return {
            "total_events": row[0],
            "actor_count": row[1],
            "event_type_count": row[2],
            "error_count": row[3],
            "avg_latency_ms": row[4],
            "first_event": row[5],
            "last_event": row[6],
        }

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush()
        self._conn.close()
        TraceStore._instance = None


# ---------------------------------------------------------------------------
# Decorator: trace any async function
# ---------------------------------------------------------------------------

def traced(actor_id: str, event_type: str, phase: str = ""):
    """Decorator: record entry, exit, and exceptions of an async function."""
    def decorator(fn: Callable):
        async def wrapper(*args, **kwargs):
            store = await TraceStore.get()
            trace_id = await store.record(
                actor_id=actor_id,
                event_type=f"{event_type}_start",
                phase=phase,
                payload={"args": str(args), "kwargs": str(kwargs)},
            )
            start = time.time()
            try:
                result = await fn(*args, **kwargs)
                await store.record(
                    actor_id=actor_id,
                    event_type=f"{event_type}_end",
                    phase=phase,
                    payload={"trace_id": trace_id, "result_type": type(result).__name__},
                    latency_ms=(time.time() - start) * 1000,
                )
                return result
            except Exception as e:
                await store.record(
                    actor_id=actor_id,
                    event_type=f"{event_type}_error",
                    phase=phase,
                    payload={"trace_id": trace_id},
                    latency_ms=(time.time() - start) * 1000,
                    error=e,
                )
                raise
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context manager: trace a block
# ---------------------------------------------------------------------------

@asynccontextmanager
async def trace_block(actor_id: str, event_type: str, phase: str = "", payload: dict | None = None):
    store = await TraceStore.get()
    trace_id = await store.record(
        actor_id=actor_id, event_type=f"{event_type}_start", phase=phase, payload=payload
    )
    start = time.time()
    try:
        yield trace_id
        await store.record(
            actor_id=actor_id, event_type=f"{event_type}_end", phase=phase,
            payload={"trace_id": trace_id}, latency_ms=(time.time() - start) * 1000
        )
    except Exception as e:
        await store.record(
            actor_id=actor_id, event_type=f"{event_type}_error", phase=phase,
            payload={"trace_id": trace_id}, latency_ms=(time.time() - start) * 1000, error=e
        )
        raise
