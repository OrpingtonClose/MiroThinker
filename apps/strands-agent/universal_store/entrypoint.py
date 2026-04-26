"""One-liner entry point for the Universal Store Architecture.

Usage::

    config = UnifiedConfig.from_env()
    orch = UniversalOrchestrator(config)
    async for event in orch.run("What are the side-effects of X?"):
        print(event.phase, event.message)

Or with context-manager lifecycle::

    async with UniversalOrchestrator(config) as orch:
        async for event in orch.run("..."):
            ...
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import AsyncIterator

import duckdb

from universal_store.actors.orchestrator import OrchestratorActor
from universal_store.config import UnifiedConfig
from universal_store.protocols import Event, OrchestratorEvent, OrchestratorPhase
from universal_store.schema import get_all_ddl
from universal_store.trace import TraceStore, trace_block


class UniversalOrchestrator:
    """Top-level orchestrator entry point.

    Responsibilities:
    1. Bootstrap DuckDB and run schema migrations.
    2. Initialise the :class:`TraceStore`.
    3. Instantiate and manage the :class:`OrchestratorActor` tree.
    4. Stream :class:`OrchestratorEvent` objects back to the caller.
    5. Guarantee cleanup on both success and failure.
    """

    def __init__(self, config: UnifiedConfig) -> None:
        self.config = config
        self._db: duckdb.DuckDBPyConnection | None = None
        self._trace: TraceStore | None = None
        self._orchestrator: OrchestratorActor | None = None

    async def setup(self) -> None:
        """Initialise DuckDB, run schema migrations, init TraceStore, create OrchestratorActor.

        Every step is traced. Safe to call multiple times (idempotent for DB/TraceStore,
        but will recreate the OrchestratorActor reference).
        """
        # 1. DuckDB
        self._db = duckdb.connect(self.config.store.db_path)
        self._db.execute(get_all_ddl())
        self._db.commit()

        # 2. TraceStore (singleton – first call wins the real db_path)
        self._trace = await TraceStore.get(self.config.trace.db_path)

        await self._trace.record(
            actor_id="UniversalOrchestrator",
            event_type="setup_start",
            phase="setup",
            payload={"store_db": self.config.store.db_path, "trace_db": self.config.trace.db_path},
        )

        # 3. Actor tree root (not started yet)
        self._orchestrator = OrchestratorActor(self.config)

        await self._trace.record(
            actor_id="UniversalOrchestrator",
            event_type="setup_end",
            phase="setup",
            payload={"orchestrator_id": self._orchestrator.actor_id},
        )

    async def run(self, query: str) -> AsyncIterator[OrchestratorEvent]:
        """Accept a user query, spin up the actor tree, and yield the event stream.

        Flow:
        1. Insert a ``runs`` row.
        2. Bind the current ``run_id`` to the trace store.
        3. Start the :class:`OrchestratorActor`.
        4. Submit an initial ``IngestQuery`` event.
        5. Yield every :class:`OrchestratorEvent` produced by ``orchestrator.events()``.
        6. On normal completion **or** exception, update the ``runs`` row.
        7. On exception, emit an ``ERROR`` phase event, flush traces, and re-raise.

        Args:
            query: Free-text user research query.

        Yields:
            :class:`OrchestratorEvent` objects for real-time dashboards / UIs.

        Raises:
            RuntimeError: If :meth:`setup` has not been called.
        """
        if self._db is None or self._trace is None or self._orchestrator is None:
            raise RuntimeError(
                "UniversalOrchestrator not initialised. "
                "Call setup() first or use 'async with' context manager."
            )

        run_id = uuid.uuid4().hex
        started_at = datetime.utcnow().isoformat()

        await self._trace.record(
            actor_id="UniversalOrchestrator",
            event_type="run_start",
            phase="run",
            payload={"query": query, "run_id": run_id},
        )

        # 1. Persist run metadata
        self._db.execute(
            """
            INSERT INTO runs (run_id, started_at, source_query, status)
            VALUES (?, ?, ?, ?)
            """,
            [run_id, started_at, query, "running"],
        )
        self._db.commit()

        # 2. Bind trace context
        await self._trace.set_run(run_id)

        # 3. Start actor tree
        self._orchestrator.start()

        # 4. Seed the recursive loop
        await self._orchestrator.send(Event("IngestQuery", {"query": query}))

        try:
            # 5. Stream events to caller
            async for event in self._orchestrator.events():
                yield event

            # 6a. Normal completion
            ended_at = datetime.utcnow().isoformat()
            self._db.execute(
                """
                UPDATE runs
                SET ended_at = ?, status = ?
                WHERE run_id = ?
                """,
                [ended_at, "completed", run_id],
            )
            self._db.commit()

            await self._trace.record(
                actor_id="UniversalOrchestrator",
                event_type="run_end",
                phase="run",
                payload={"run_id": run_id, "status": "completed"},
            )

        except Exception as exc:
            # 6b/7. Graceful degradation on error
            yield OrchestratorEvent(
                phase=OrchestratorPhase.ERROR,
                message=str(exc),
                data={
                    "run_id": run_id,
                    "query": query,
                    "error_type": type(exc).__name__,
                },
            )

            ended_at = datetime.utcnow().isoformat()
            self._db.execute(
                """
                UPDATE runs
                SET ended_at = ?, status = ?, convergence_reason = ?
                WHERE run_id = ?
                """,
                [ended_at, "error", str(exc), run_id],
            )
            self._db.commit()

            await self._trace.record(
                actor_id="UniversalOrchestrator",
                event_type="run_error",
                phase="run",
                payload={"run_id": run_id, "query": query},
                error=exc,
            )

            # Flush any buffered traces before propagating the exception
            await self._trace._flush()
            raise

    async def shutdown(self) -> None:
        """Tear down the actor tree, flush traces, and close DuckDB.

        Called automatically when used as an ``async with`` context manager.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._trace is not None:
            async with trace_block("UniversalOrchestrator", "shutdown", phase="shutdown"):
                if self._orchestrator is not None:
                    await self._orchestrator.stop(graceful=True)
                await self._trace.shutdown()
                self._trace = None
        else:
            # Best-effort actor stop even if trace store is gone
            if self._orchestrator is not None:
                try:
                    await self._orchestrator.stop(graceful=True)
                except Exception:
                    pass

        if self._db is not None:
            try:
                self._db.commit()
                self._db.close()
            except Exception:
                pass
            self._db = None

        self._orchestrator = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> UniversalOrchestrator:
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.shutdown()
