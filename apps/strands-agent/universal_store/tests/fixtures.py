"""Test fixtures for the Universal Store Architecture end-to-end suite.

All fixtures are async-safe and clean up DuckDB files on teardown.
A lightweight :class:`MinimalDuckDBStore` is provided so that actors can be
exercised without importing the full production store stack.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from typing import Any

import duckdb
import pytest
import pytest_asyncio
import re

from universal_store.config import UnifiedConfig
from universal_store.schema import get_all_ddl
from universal_store.trace import TraceStore


# ---------------------------------------------------------------------------
# Minimal DuckDB store — satisfies the duck-typed interface actors expect
# ---------------------------------------------------------------------------

def _strip_include_clauses(ddl: str) -> str:
    """Remove SQL-Server ``INCLUDE`` clauses that DuckDB does not support."""
    return re.sub(r"\)\s*INCLUDE\s*\([^)]*\)", ")", ddl, flags=re.IGNORECASE)


def _fix_autoincrement(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    """Add a sequence-backed default so ``id INTEGER PRIMARY KEY`` auto-increments."""
    seq_name = f"{table}_id_seq"
    conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq_name} START 1")
    conn.execute(
        f"ALTER TABLE {table} ALTER COLUMN id SET DEFAULT nextval('{seq_name}')"
    )


class MinimalDuckDBStore:
    """In-memory/file-backed DuckDB wrapper with the methods actors call.

    Mirrors the surface area of ``ConditionStore`` (``admit``, ``conn``,
    ``get_findings``, ``insert``, ``query``, ``execute``) so that swarm,
    flock, semantic, curation and MCP actors can be wired directly.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._next_id = 1
        self._create_base_conditions()
        self._run_migrations()
        # DuckDB does not auto-increment INTEGER PRIMARY KEY by default
        for table in (
            "score_history",
            "lessons",
            "lesson_applications",
            "semantic_connections",
            "source_fingerprints",
            "chunks",
            "source_utility_log",
            "source_quality_registry",
            "trace_records",
        ):
            _fix_autoincrement(self.conn, table)
        self.conn.commit()

    def _create_base_conditions(self) -> None:
        """Create the base ``conditions`` table before applying ALTERs."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conditions (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                source_type TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',
                row_type TEXT DEFAULT 'finding',
                parent_id INTEGER,
                related_id INTEGER,
                consider_for_use BOOLEAN DEFAULT TRUE,
                obsolete_reason TEXT DEFAULT '',
                angle TEXT DEFAULT '',
                strategy TEXT DEFAULT '',
                expansion_depth INTEGER DEFAULT 0,
                created_at TEXT DEFAULT '',
                iteration INTEGER DEFAULT 0,
                confidence FLOAT DEFAULT 0.5,
                trust_score FLOAT DEFAULT 0.5,
                novelty_score FLOAT DEFAULT 0.5,
                specificity_score FLOAT DEFAULT 0.5,
                relevance_score FLOAT DEFAULT 0.5,
                actionability_score FLOAT DEFAULT 0.5,
                duplication_score FLOAT DEFAULT -1.0,
                fabrication_risk FLOAT DEFAULT 0.0,
                verification_status TEXT DEFAULT '',
                scored_at TEXT DEFAULT '',
                score_version INTEGER DEFAULT 0,
                composite_quality FLOAT DEFAULT -1.0,
                information_density FLOAT DEFAULT -1.0,
                cross_ref_boost FLOAT DEFAULT 0.0,
                processing_status TEXT DEFAULT 'raw',
                expansion_tool TEXT DEFAULT 'none',
                expansion_hint TEXT DEFAULT '',
                expansion_fulfilled BOOLEAN DEFAULT FALSE,
                expansion_gap TEXT DEFAULT '',
                expansion_priority FLOAT DEFAULT 0.0,
                cluster_id INTEGER DEFAULT -1,
                cluster_rank INTEGER DEFAULT 0,
                contradiction_flag BOOLEAN DEFAULT FALSE,
                contradiction_partner INTEGER DEFAULT -1,
                staleness_penalty FLOAT DEFAULT 0.0,
                relationship_score FLOAT DEFAULT 0.0,
                phase TEXT DEFAULT '',
                parent_ids TEXT DEFAULT '',
                source_model TEXT DEFAULT '',
                source_run TEXT DEFAULT '',
                evaluation_count INTEGER DEFAULT 0,
                last_evaluated_at TEXT DEFAULT '',
                evaluator_angles TEXT DEFAULT '',
                mcp_research_status TEXT DEFAULT '',
                information_gain FLOAT DEFAULT 0.0
            )
            """
        )

    def _run_migrations(self) -> None:
        """Run the additive schema DDL from ``universal_store.schema``."""
        self.conn.execute(_strip_include_clauses(get_all_ddl()))

    # -- Actor-facing API ----------------------------------------------------

    def admit(self, fact: str, **kwargs: Any) -> int | None:
        """Insert a single condition row and return its id.

        Accepts the full surface area of ``ConditionStore.admit`` plus any
        extra keyword arguments that correspond to columns on the
        ``conditions`` table (e.g. ``novelty_score``, ``cluster_id``).
        """
        fact = fact.strip()
        if not fact:
            return None
        cid = self._next_id
        self._next_id += 1
        now = datetime.now(timezone.utc).isoformat()

        row: dict[str, Any] = {
            "id": cid,
            "fact": fact,
            "source_url": kwargs.get("source_url", ""),
            "source_type": kwargs.get("source_type", "researcher"),
            "source_ref": kwargs.get("source_ref", ""),
            "row_type": kwargs.get("row_type", "finding"),
            "related_id": kwargs.get("related_id"),
            "consider_for_use": kwargs.get("consider_for_use", True),
            "confidence": kwargs.get("confidence", 0.5),
            "verification_status": kwargs.get("verification_status", ""),
            "angle": kwargs.get("angle", ""),
            "parent_id": kwargs.get("parent_id"),
            "strategy": kwargs.get("strategy", ""),
            "expansion_depth": kwargs.get("expansion_depth", 0),
            "created_at": kwargs.get("created_at", now),
            "iteration": kwargs.get("iteration", 0),
            "source_model": kwargs.get("source_model", ""),
            "source_run": kwargs.get("source_run", ""),
            "phase": kwargs.get("phase", ""),
        }

        # Merge any extra columns that callers provide (e.g. novelty_score, cluster_id)
        for k, v in kwargs.items():
            if k not in row:
                row[k] = v

        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        values = list(row.values())
        self.conn.execute(
            f"INSERT INTO conditions ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        return cid

    def get_findings(
        self,
        min_confidence: float = 0.0,
        angle: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return active findings with optional filters."""
        query = """
            SELECT id, fact, source_url, source_type, confidence,
                   angle, iteration, verification_status
            FROM conditions
            WHERE consider_for_use = TRUE
              AND row_type = 'finding'
              AND confidence >= ?
        """
        params: list[Any] = [min_confidence]
        if angle:
            query += " AND angle = ?"
            params.append(angle)
        query += f" ORDER BY confidence DESC, id ASC LIMIT {int(limit)}"
        rows = self.conn.execute(query, params).fetchall()
        cols = [
            "id",
            "fact",
            "source_url",
            "source_type",
            "confidence",
            "angle",
            "iteration",
            "verification_status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def insert(self, table: str, row: dict[str, Any]) -> int:
        """Generic insert for any table (used by MCP researcher)."""
        cols = list(row.keys())
        placeholders = ", ".join(["?"] * len(cols))
        values = list(row.values())
        self.conn.execute(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
            values,
        )
        result = self.conn.execute("SELECT last_insert_rowid()").fetchone()
        return result[0] if result else 0

    def query(self, sql: str, params: tuple | None = None) -> list[dict]:
        """Execute a read query and return dict rows."""
        result = self.conn.execute(sql, params or ())
        if result.description is None:
            return []
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def execute(self, sql: str, params: tuple | None = None) -> list[dict]:
        """Execute raw SQL (returns empty list for write statements)."""
        self.conn.execute(sql, params or ())
        return []

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self.conn.close()


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def temp_duckdb() -> str:
    """Yield a temporary DuckDB file path and delete it on teardown."""
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(path)  # Remove empty stub so DuckDB can create a valid file
    try:
        yield path
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


@pytest_asyncio.fixture
async def temp_trace_db() -> str:
    """Yield a temporary trace DB path and delete it on teardown."""
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.unlink(path)  # Remove empty stub so DuckDB can create a valid file
    try:
        yield path
    finally:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


@pytest_asyncio.fixture
async def minimal_config() -> UnifiedConfig:
    """Return a minimal :class:`UnifiedConfig` tuned for fast unit tests."""
    cfg = UnifiedConfig()
    cfg.store.db_path = ":memory:"
    cfg.trace.db_path = ":memory:"
    cfg.actor.mailbox_queue_size = 100
    cfg.scheduler.event_queue_size = 100
    cfg.scheduler.default_round_time_s = 5.0
    cfg.scheduler.convergence_threshold = 0.02
    cfg.scheduler.max_convergence_stuck_rounds = 2
    cfg.scheduler.max_total_rounds = 5
    cfg.swarm.default_bee_count = 2
    cfg.swarm.max_gossip_rounds = 3
    cfg.swarm.gossip_info_gain_threshold = 0.5
    cfg.swarm.min_gossip_rounds = 1
    cfg.swarm.max_workers_per_phase = 2
    cfg.flock.default_clone_count = 2
    cfg.flock.default_rounds = 3
    cfg.flock.magnitude_convergence_threshold = 0.05
    cfg.flock.ucb_alpha = 1.0
    cfg.flock.priority_decay_rate = 0.1
    cfg.flock.serendipity_floor = 0.4
    cfg.external.red_tier_usd = 1.0
    cfg.external.yellow_tier_usd = 0.5
    cfg.external.green_tier_usd = 0.05
    cfg.external.red_tier_tokens = 5000
    cfg.external.red_tier_latency_s = 10.0
    cfg.external.operator_override_timeout_s = 0.1
    cfg.external.max_targets_per_round = 5
    cfg.semantic.heuristic_max_candidates = 100
    cfg.semantic.embedding_threshold = 0.5
    cfg.semantic.llm_verification_batch_size = 32
    cfg.semantic.min_confidence_for_storage = 0.70
    cfg.curation.clone_context_max_items = 10
    cfg.curation.angle_bundle_max_items = 20
    cfg.curation.global_health_interval_s = 0.1
    cfg.curation.contradiction_digest_interval_s = 0.1
    cfg.curation.gap_digest_interval_s = 0.1
    cfg.curation.narrative_interval_s = 0.1
    cfg.reflexion.halflife_runs_default = 2
    cfg.reflexion.min_lesson_confidence = 0.5
    return cfg


@pytest.fixture
def sample_findings() -> list[dict]:
    """Return 10 sample findings with all required gradient fields."""
    return [
        {
            "id": i + 1,
            "fact": (
                f"Finding {i + 1}: Quantum error-correction codes "
                f"require {i + 2} physical qubits per logical qubit."
            ),
            "confidence": 0.7 + (i * 0.02),
            "novelty_score": 0.6,
            "specificity_score": 0.7,
            "relevance_score": 0.8,
            "actionability_score": 0.5,
            "fabrication_risk": 0.1,
            "angle": "physics" if i % 2 == 0 else "engineering",
            "source_type": "paper",
            "source_url": f"https://example.com/finding-{i + 1}",
            "created_at": "2024-01-01T00:00:00",
            "row_type": "finding",
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_raw_rows() -> list[dict]:
    """Return 5 sample raw input rows."""
    return [
        {
            "id": 100 + i,
            "fact": f"Raw input {i + 1}: Lorem ipsum dolor sit amet.",
            "source_url": f"https://source.com/raw-{i + 1}",
            "source_type": "scraped",
            "confidence": 0.6,
            "angle": "general",
            "row_type": "raw",
            "strategy": "",
            "created_at": "2024-01-01T00:00:00",
        }
        for i in range(5)
    ]


@pytest_asyncio.fixture
async def populated_store(
    temp_duckdb: str,
    sample_findings: list[dict],
    sample_raw_rows: list[dict],
) -> MinimalDuckDBStore:
    """Create a store, run DDL, insert raw rows + findings, and yield it."""
    store = MinimalDuckDBStore(temp_duckdb)
    for row in sample_raw_rows:
        store.admit(**{k: v for k, v in row.items() if k != "id"})
    for finding in sample_findings:
        store.admit(**{k: v for k, v in finding.items() if k != "id"})
    store.conn.commit()
    try:
        yield store
    finally:
        store.close()


@pytest_asyncio.fixture
async def trace_store(temp_trace_db: str) -> TraceStore:
    """Yield a fresh :class:`TraceStore` singleton tied to *temp_trace_db*."""
    # Hard-reset singleton so tests cannot pollute each other.
    if TraceStore._instance is not None:
        try:
            await TraceStore._instance.shutdown()
        except Exception:
            pass
    TraceStore._instance = None

    ts = await TraceStore.get(temp_trace_db)
    _fix_autoincrement(ts._conn, "trace_records")
    try:
        yield ts
    finally:
        await ts.shutdown()


@pytest_asyncio.fixture(autouse=True)
async def reset_trace_store() -> None:
    """Autouse fixture that guarantees the trace singleton is clean after every test."""
    yield
    if TraceStore._instance is not None:
        try:
            await TraceStore._instance.shutdown()
        except Exception:
            pass
    TraceStore._instance = None


# ---------------------------------------------------------------------------
# Mock actor for scheduler / routing tests
# ---------------------------------------------------------------------------

# Patch LessonStore so that its ``lessons`` table auto-increments ``id``
# (DuckDB does not do this by default for INTEGER PRIMARY KEY).
from universal_store.actors.reflexion import LessonStore

_original_lesson_ensure_conn = LessonStore._ensure_conn

async def _patched_lesson_ensure_conn(self: LessonStore) -> Any:
    conn = await _original_lesson_ensure_conn(self)
    _fix_autoincrement(conn, "lessons")
    return conn

LessonStore._ensure_conn = _patched_lesson_ensure_conn  # type: ignore[method-assign]


class MockActor:
    """Minimal stand-in actor that records every event it receives."""

    def __init__(self, actor_id: str) -> None:
        self.actor_id = actor_id
        self.mailbox: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.received: list = []
        self._task: asyncio.Task | None = None
        self._shutdown = False

    async def send(self, event: Any) -> None:
        try:
            self.mailbox.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._shutdown = False
            self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            self.received.append(event)

    async def stop(self, graceful: bool = True) -> None:
        self._shutdown = True
        if self._task and not self._task.done():
            if not graceful:
                self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

    async def health(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "running": self._task is not None and not self._task.done(),
            "mailbox_size": self.mailbox.qsize(),
        }
