"""Production DuckDB store implementing StoreProtocol.

Wraps a DuckDB connection and provides the methods actors expect:
- admit()     — insert a condition row
- insert()    — generic insert for any table
- query()     — read queries returning dict rows
- execute()   — raw SQL execution
- get_findings() — active findings with optional filters
- close()     — close the connection
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import duckdb

from universal_store.schema import get_all_ddl
from universal_store.protocols import StoreProtocol


class DuckDBStore:
    """DuckDB-backed store for the Universal Store Architecture.

    Satisfies the duck-typed interface used by swarm, flock, semantic,
    curation and MCP actors.  Methods are synchronous; actors that need
    async access should wrap calls with ``asyncio.to_thread()``.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._next_id = 1
        self._ensure_schema()
        self.conn.commit()

    def _ensure_schema(self) -> None:
        """Create or migrate all tables and indices."""
        self.conn.execute(get_all_ddl())

    # -- Actor-facing API ----------------------------------------------------

    def admit(self, fact: str, **kwargs: Any) -> int | None:
        """Insert a single condition row and return its id."""
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
            "id", "fact", "source_url", "source_type",
            "confidence", "angle", "iteration", "verification_status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def insert(self, table: str, row: dict[str, Any]) -> int:
        """Generic insert for any table."""
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
