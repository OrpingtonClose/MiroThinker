# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
DuckDB-backed corpus store for AtomicConditions.

Provides structured, queryable storage for research findings.  The corpus
grows over LoopAgent iterations as the researcher produces new findings
and the condition manager decomposes them into atoms.

The thinker reads the corpus each iteration (via ``format_for_thinker``)
and reasons about gaps, contradictions, and next steps.  The synthesiser
reads it at the end (via ``format_for_synthesiser``) to write the report.

Embedding-based cosine similarity is used as a cheap pre-filter for
deduplication.  Final judgement about true duplicates is left to the
thinker agent (agentic, not numeric).
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb

from models.atomic_condition import AtomicCondition

logger = logging.getLogger(__name__)


class CorpusStore:
    """In-memory DuckDB store for AtomicConditions."""

    def __init__(self) -> None:
        self.conn = duckdb.connect()
        self._setup_tables()
        self._next_id = 1

    def _setup_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE conditions (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                confidence FLOAT DEFAULT 0.5,
                verification_status TEXT DEFAULT '',
                angle TEXT DEFAULT '',
                parent_id INTEGER,
                strategy TEXT DEFAULT '',
                expansion_depth INTEGER DEFAULT 0,
                created_at TEXT DEFAULT ''
            )
        """)
        self.conn.execute("""
            CREATE TABLE similarity_flags (
                condition_a INTEGER,
                condition_b INTEGER,
                flag TEXT DEFAULT 'pending',
                PRIMARY KEY (condition_a, condition_b)
            )
        """)

    def admit(self, condition: AtomicCondition) -> Optional[int]:
        """Insert a condition into the corpus.

        Returns the assigned ID, or ``None`` if the fact is empty/trivial.
        """
        fact = condition.fact.strip()
        if not fact or len(fact) < 10:
            return None

        cid = self._next_id
        self._next_id += 1

        self.conn.execute(
            """INSERT INTO conditions
               (id, fact, source_url, confidence, verification_status,
                angle, parent_id, strategy, expansion_depth, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                cid,
                fact,
                condition.source_url,
                condition.confidence,
                condition.verification_status,
                condition.angle,
                condition.parent_id,
                condition.strategy,
                condition.expansion_depth,
                condition.created_at,
            ],
        )
        logger.debug("Admitted condition #%d: %.80s", cid, fact)
        return cid

    def admit_batch(self, conditions: list[AtomicCondition]) -> list[int]:
        """Insert multiple conditions.  Returns list of assigned IDs."""
        ids = []
        for c in conditions:
            cid = self.admit(c)
            if cid is not None:
                ids.append(cid)
        return ids

    def count(self) -> int:
        """Total number of conditions in the corpus."""
        return self.conn.execute("SELECT COUNT(*) FROM conditions").fetchone()[0]

    def count_by_status(self) -> dict[str, int]:
        """Count conditions grouped by verification_status."""
        rows = self.conn.execute(
            "SELECT verification_status, COUNT(*) FROM conditions GROUP BY verification_status"
        ).fetchall()
        return {row[0] or "(unset)": row[1] for row in rows}

    def get_all(self) -> list[dict]:
        """Return all conditions as dicts, ordered by confidence DESC."""
        rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, verification_status,
                      angle, parent_id, strategy, expansion_depth
               FROM conditions
               WHERE verification_status != 'fabricated'
               ORDER BY confidence DESC, id ASC"""
        ).fetchall()
        cols = [
            "id", "fact", "source_url", "confidence", "verification_status",
            "angle", "parent_id", "strategy", "expansion_depth",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def format_for_thinker(self) -> str:
        """Format the corpus as structured text for the thinker to read.

        Groups conditions by angle and includes metadata the thinker needs
        for gap analysis: confidence, verification status, expansion depth.
        """
        conditions = self.get_all()
        if not conditions:
            return "(no findings yet)"

        lines: list[str] = []
        lines.append(f"CORPUS: {len(conditions)} conditions\n")

        # Group by angle
        by_angle: dict[str, list[dict]] = {}
        for c in conditions:
            angle = c["angle"] or "general"
            by_angle.setdefault(angle, []).append(c)

        for angle, conds in sorted(by_angle.items()):
            lines.append(f"## Angle: {angle} ({len(conds)} findings)")
            for c in conds:
                status = c["verification_status"] or "unverified"
                src = f" [{c['source_url']}]" if c["source_url"] else ""
                depth = f" (depth={c['expansion_depth']})" if c["expansion_depth"] > 0 else ""
                lines.append(
                    f"  [{c['id']}] (conf={c['confidence']:.1f}, {status}{depth}) "
                    f"{c['fact']}{src}"
                )
            lines.append("")

        # Summary stats
        status_counts = self.count_by_status()
        lines.append("STATUS SUMMARY: " + ", ".join(
            f"{k}={v}" for k, v in sorted(status_counts.items())
        ))

        return "\n".join(lines)

    def format_for_synthesiser(self) -> str:
        """Format the corpus for the synthesiser — all verified/unverified
        conditions with full detail, ordered by confidence."""
        conditions = self.get_all()
        if not conditions:
            return "(no findings)"

        lines: list[str] = []
        for c in conditions:
            src = f"\n   Source: {c['source_url']}" if c["source_url"] else ""
            status = c["verification_status"] or "unverified"
            lines.append(
                f"- [{status}, confidence={c['confidence']:.2f}] "
                f"{c['fact']}{src}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
