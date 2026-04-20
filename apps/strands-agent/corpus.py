"""DuckDB-backed corpus store for research ↔ gossip data flow.

Adapted from apps/adk-agent/models/corpus_store.py with:
- Flock dependency removed (atomisation via direct LLM calls in atomizer.py)
- Dashboard hooks removed
- Swarm-specific export/import methods added
- No v1 migration needed (fresh schema only)

Every factual claim is an AtomicCondition row with provenance,
gradient-flag scoring, and lineage (parent_id DAG). No data passes
between research and gossip as strings — everything goes through
this store.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import duckdb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AtomicCondition dataclass
# ---------------------------------------------------------------------------

@dataclass
class AtomicCondition:
    """A single compressed research finding (Atom of Thought).

    Each atom is an indivisible unit of research output carrying:
    - The factual claim itself
    - Source provenance (URL)
    - Confidence score (0.0-1.0)
    - Verification status lifecycle: "" -> "speculative" -> "verified" | "fabricated"
    - Expansion metadata (parent lineage, strategy, depth)
    - Angle tracking (which research angle produced it)
    """

    fact: str
    source_url: str = ""
    confidence: float = 0.5
    verification_status: str = ""
    angle: str = ""
    parent_id: int | None = None
    strategy: str = ""
    expansion_depth: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    row_type: str = "finding"
    related_id: int | None = None
    consider_for_use: bool = True
    source_type: str = "researcher"
    source_ref: str = ""
    iteration: int = 0

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, self.confidence))


# ---------------------------------------------------------------------------
# ConditionStore
# ---------------------------------------------------------------------------

class ConditionStore:
    """DuckDB-backed corpus backbone for research ↔ gossip data flow.

    Every factual claim is an AtomicCondition row with provenance,
    gradient-flag scoring, and lineage (parent_id DAG).

    Thread-safe via _lock for ALL connection access (reads + writes)
    from async event loop and sync research thread.
    """

    def __init__(self, db_path: str = "") -> None:
        """Initialize DuckDB connection and schema.

        Args:
            db_path: Path to DuckDB file. Empty string = in-memory.
        """
        if db_path:
            self.conn = duckdb.connect(db_path)
        else:
            self.conn = duckdb.connect()
        self._lock = threading.RLock()  # Reentrant: methods call each other
        self._next_id = 1
        self.user_query: str = ""  # Set by _run_job for trigger_gossip context
        self.version: int = 0  # Monotonic counter, incremented on every write
        self.research_complete: bool = False  # Signal from research to gossip loop
        self._setup_tables()

    def _setup_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conditions (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                source_type TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',

                -- Row type: 'finding' | 'similarity' | 'contradiction'
                --           | 'raw' | 'synthesis' | 'thought' | 'insight'
                row_type TEXT DEFAULT 'finding',

                -- Hierarchical relationships
                parent_id INTEGER,
                related_id INTEGER,

                -- Universal exclusion flag
                consider_for_use BOOLEAN DEFAULT TRUE,
                obsolete_reason TEXT DEFAULT '',

                -- Core metadata
                angle TEXT DEFAULT '',
                strategy TEXT DEFAULT '',
                expansion_depth INTEGER DEFAULT 0,
                created_at TEXT DEFAULT '',
                iteration INTEGER DEFAULT 0,

                -- GRADIENT FLAGS (0.0-1.0)
                confidence FLOAT DEFAULT 0.5,
                trust_score FLOAT DEFAULT 0.5,
                novelty_score FLOAT DEFAULT 0.5,
                specificity_score FLOAT DEFAULT 0.5,
                relevance_score FLOAT DEFAULT 0.5,
                actionability_score FLOAT DEFAULT 0.5,
                duplication_score FLOAT DEFAULT -1.0,
                fabrication_risk FLOAT DEFAULT 0.0,

                -- Categorical
                verification_status TEXT DEFAULT '',

                -- Scoring metadata
                scored_at TEXT DEFAULT '',
                score_version INTEGER DEFAULT 0,

                -- Composite & derived scores
                composite_quality FLOAT DEFAULT -1.0,
                information_density FLOAT DEFAULT -1.0,
                cross_ref_boost FLOAT DEFAULT 0.0,

                -- Processing state
                processing_status TEXT DEFAULT 'raw',

                -- Expansion system
                expansion_tool TEXT DEFAULT 'none',
                expansion_hint TEXT DEFAULT '',
                expansion_fulfilled BOOLEAN DEFAULT FALSE,
                expansion_gap TEXT DEFAULT '',
                expansion_priority FLOAT DEFAULT 0.0,

                -- Clustering
                cluster_id INTEGER DEFAULT -1,
                cluster_rank INTEGER DEFAULT 0,

                -- Contradiction
                contradiction_flag BOOLEAN DEFAULT FALSE,
                contradiction_partner INTEGER DEFAULT -1,

                -- Staleness
                staleness_penalty FLOAT DEFAULT 0.0,

                -- For relationship rows
                relationship_score FLOAT DEFAULT 0.0
            )
        """)
        # Seed next_id from existing rows
        result = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM conditions").fetchone()
        if result:
            self._next_id = result[0] + 1

    # ------------------------------------------------------------------
    # Core write methods
    # ------------------------------------------------------------------

    def admit(
        self,
        fact: str,
        source_url: str = "",
        source_type: str = "researcher",
        source_ref: str = "",
        row_type: str = "finding",
        angle: str = "",
        parent_id: int | None = None,
        related_id: int | None = None,
        confidence: float = 0.5,
        verification_status: str = "",
        strategy: str = "",
        expansion_depth: int = 0,
        iteration: int = 0,
        consider_for_use: bool = True,
    ) -> int | None:
        """Insert a single condition row.

        Returns the assigned condition ID, or None if fact is empty.
        """
        fact = fact.strip()
        if not fact:
            return None

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cid = self._next_id
            self._next_id += 1
            self.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_url, source_type, source_ref,
                    row_type, related_id, consider_for_use,
                    confidence, verification_status, angle,
                    parent_id, strategy,
                    expansion_depth, created_at, iteration)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    cid, fact, source_url, source_type, source_ref,
                    row_type, related_id, consider_for_use,
                    confidence, verification_status, angle,
                    parent_id, strategy,
                    expansion_depth, now, iteration,
                ],
            )
            self.version += 1
        logger.debug("admitted condition #%d: %.80s", cid, fact)
        return cid

    def admit_condition(self, condition: AtomicCondition) -> int | None:
        """Insert an AtomicCondition instance."""
        return self.admit(
            fact=condition.fact,
            source_url=condition.source_url,
            source_type=condition.source_type,
            source_ref=condition.source_ref,
            row_type=condition.row_type,
            angle=condition.angle,
            parent_id=condition.parent_id,
            related_id=condition.related_id,
            confidence=condition.confidence,
            verification_status=condition.verification_status,
            strategy=condition.strategy,
            expansion_depth=condition.expansion_depth,
            iteration=condition.iteration,
            consider_for_use=condition.consider_for_use,
        )

    def admit_batch(self, conditions: list[AtomicCondition]) -> list[int]:
        """Insert multiple conditions. Returns list of assigned IDs."""
        ids = []
        for c in conditions:
            cid = self.admit_condition(c)
            if cid is not None:
                ids.append(cid)
        return ids

    def admit_synthesis(
        self,
        report: str,
        iteration: int,
        source_type: str = "gossip_swarm",
        metrics: dict[str, Any] | None = None,
    ) -> int:
        """Store a full gossip synthesis report as row_type='synthesis'.

        No truncation. Full report text stored. Metrics serialized as JSON
        in the strategy field.

        Returns the condition ID of the synthesis row.
        """
        strategy = json.dumps(metrics) if metrics else ""
        cid = self.admit(
            fact=report,
            source_type=source_type,
            row_type="synthesis",
            strategy=strategy,
            iteration=iteration,
            confidence=0.8,
        )
        if cid is None:
            raise ValueError("cannot admit empty synthesis report")
        logger.info(
            "admitted synthesis #%d: %d chars (iteration=%d)",
            cid, len(report), iteration,
        )
        return cid

    def admit_thought(
        self,
        reasoning: str,
        parent_thought_id: int | None = None,
        angle: str = "",
        strategy: str = "",
        iteration: int = 0,
        expansion_depth: int = 0,
    ) -> int:
        """Persist a reasoning trace as row_type='thought'."""
        cid = self.admit(
            fact=reasoning,
            row_type="thought",
            parent_id=parent_thought_id,
            angle=angle,
            strategy=strategy,
            iteration=iteration,
            expansion_depth=expansion_depth,
        )
        if cid is None:
            raise ValueError("cannot admit empty thought")
        return cid

    # ------------------------------------------------------------------
    # Ingestion (raw text -> atomised conditions)
    # ------------------------------------------------------------------

    def ingest_raw(
        self,
        raw_text: str,
        source_type: str = "researcher",
        source_ref: str = "",
        angle: str = "",
        iteration: int = 0,
        user_query: str = "",
    ) -> list[int]:
        """Ingest raw text via chunk-aware atomisation.

        Preserves full lineage: raw -> chunks -> atoms.
        Every row in the corpus is a node in a DAG with traceable parents.
        No truncation. Full text stored as row_type='raw'.

        Atomisation is deferred to the caller (atomizer.py) or done
        inline via _atomise_simple() for basic paragraph splitting.

        Returns list of admitted condition IDs.
        """
        if not raw_text or not raw_text.strip():
            return []

        now = datetime.now(timezone.utc).isoformat()

        # Layer 0: store FULL raw text (no truncation)
        with self._lock:
            raw_id = self._next_id
            self._next_id += 1
            self.conn.execute(
                "INSERT INTO conditions "
                "(id, fact, source_type, source_ref, row_type, "
                "consider_for_use, created_at, iteration) "
                "VALUES (?, ?, ?, ?, 'raw', FALSE, ?, ?)",
                [raw_id, raw_text, source_type, source_ref, now, iteration],
            )

        # Layer 1: split into paragraph-level chunks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text.strip()]

        chunk_ids: list[int] = []
        with self._lock:
            for seq, para in enumerate(paragraphs):
                if not para:
                    continue
                chunk_id = self._next_id
                self._next_id += 1
                self.conn.execute(
                    "INSERT INTO conditions "
                    "(id, fact, source_type, source_ref, row_type, "
                    "parent_id, consider_for_use, created_at, iteration, "
                    "expansion_depth, angle) "
                    "VALUES (?, ?, ?, ?, 'finding', ?, TRUE, ?, ?, ?, ?)",
                    [
                        chunk_id, para, source_type, source_ref,
                        raw_id, now, iteration, seq,
                        angle or f"iteration_{iteration}",
                    ],
                )
                chunk_ids.append(chunk_id)

        # Mark raw row as atomised
        with self._lock:
            self.conn.execute(
                "UPDATE conditions SET "
                "obsolete_reason = 'atomised into ' || ? || ' findings' "
                "WHERE id = ? AND row_type = 'raw'",
                [str(len(chunk_ids)), raw_id],
            )

        with self._lock:
            self.version += 1  # Single bump per ingest_raw (covers raw + chunks)
        logger.info(
            "ingested raw: %d paragraphs from %d chars (source=%s, iteration=%d)",
            len(chunk_ids), len(raw_text), source_type, iteration,
        )
        return chunk_ids

    # ------------------------------------------------------------------
    # Export methods (for gossip swarm and orchestrator)
    # ------------------------------------------------------------------

    def export_for_swarm(
        self,
        iteration: int | None = None,
        min_confidence: float = 0.0,
        max_rows: int | None = None,
    ) -> str:
        """Export corpus as structured text for gossip swarm input.

        Queries conditions (row_type='finding', consider_for_use=TRUE),
        formats as structured text with source attribution per fact.

        Args:
            iteration: Filter to specific iteration, or None for all.
            min_confidence: Minimum confidence threshold.
            max_rows: Cap on number of conditions to export.

        Returns:
            Structured text corpus for swarm consumption.
        """
        with self._lock:
            query = """
                SELECT id, fact, source_url, source_type, confidence,
                       angle, iteration, verification_status
                FROM conditions
                WHERE consider_for_use = TRUE
                  AND row_type = 'finding'
                  AND confidence >= ?
            """
            params: list[Any] = [min_confidence]

            if iteration is not None:
                query += " AND iteration = ?"
                params.append(iteration)

            query += " ORDER BY confidence DESC, id ASC"

            if max_rows is not None:
                query += f" LIMIT {int(max_rows)}"

            rows = self.conn.execute(query, params).fetchall()

        if not rows:
            return "(corpus is empty — no findings yet)"

        lines: list[str] = []
        for row in rows:
            cid, fact, src_url, src_type, conf, angle, itr, vstatus = row
            source_tag = f"[{src_type}]" if src_type else ""
            url_tag = f" ({src_url})" if src_url else ""
            conf_tag = f" [conf={conf:.2f}]" if conf != 0.5 else ""
            vstatus_tag = f" [{vstatus}]" if vstatus else ""
            lines.append(
                f"[#{cid}] {source_tag}{conf_tag}{vstatus_tag} "
                f"{fact}{url_tag}"
            )

        header = f"=== CORPUS: {len(rows)} findings ==="
        return header + "\n" + "\n".join(lines)

    def get_synthesis(self, iteration: int) -> str | None:
        """Get the gossip synthesis report for a specific iteration.

        Returns full text, no truncation. None if no synthesis exists.
        """
        with self._lock:
            row = self.conn.execute(
                """SELECT fact FROM conditions
                   WHERE row_type = 'synthesis' AND iteration = ?
                   ORDER BY id DESC LIMIT 1""",
                [iteration],
            ).fetchone()
        return row[0] if row else None

    def get_all_syntheses(self) -> list[dict[str, Any]]:
        """Return all synthesis rows ordered by iteration."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, fact, iteration, strategy, created_at
                   FROM conditions
                   WHERE row_type = 'synthesis'
                   ORDER BY iteration ASC, id ASC"""
            ).fetchall()
        cols = ["id", "fact", "iteration", "strategy", "created_at"]
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------
    # Gap analysis (replaces truncated string feedback)
    # ------------------------------------------------------------------

    def query_gaps(
        self,
        user_query: str,
        iteration: int,
    ) -> str:
        """Produce structured gap analysis for the next research iteration.

        Queries the corpus for:
        - Total coverage stats
        - Low-confidence claims (confidence < 0.4)
        - Unverified claims (verification_status='speculative')
        - Unfulfilled expansion hints
        - Contradictions
        - Synthesis insights from previous iteration

        Returns structured text that becomes the researcher's next prompt.
        """
        with self._lock:
            lines: list[str] = []

            # Overall stats
            total = self.count()
            by_type = self.count_by_type()
            lines.append(f"=== GAP ANALYSIS (after iteration {iteration}) ===")
            lines.append(f"Total conditions: {total}")
            lines.append(f"By type: {by_type}")

            # Low-confidence findings
            low_conf = self.conn.execute(
                """SELECT id, fact, confidence, angle
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND confidence < 0.4
                   ORDER BY confidence ASC
                   LIMIT 20"""
            ).fetchall()
            if low_conf:
                lines.append(f"\n--- LOW CONFIDENCE ({len(low_conf)} claims need verification) ---")
                for cid, fact, conf, angle in low_conf:
                    lines.append(f"  [#{cid}] conf={conf:.2f} [{angle}]: {fact[:200]}")

            # Unverified/speculative
            speculative = self.conn.execute(
                """SELECT id, fact, angle
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND verification_status = 'speculative'
                   LIMIT 20"""
            ).fetchall()
            if speculative:
                lines.append(f"\n--- SPECULATIVE ({len(speculative)} unverified claims) ---")
                for cid, fact, angle in speculative:
                    lines.append(f"  [#{cid}] [{angle}]: {fact[:200]}")

            # Unfulfilled expansion hints
            unfulfilled = self.conn.execute(
                """SELECT id, expansion_gap, expansion_priority
                   FROM conditions
                   WHERE expansion_gap != ''
                     AND expansion_fulfilled = FALSE
                   ORDER BY expansion_priority DESC
                   LIMIT 10"""
            ).fetchall()
            if unfulfilled:
                lines.append(f"\n--- EXPANSION GAPS ({len(unfulfilled)} unfulfilled) ---")
                for cid, gap, priority in unfulfilled:
                    lines.append(f"  [#{cid}] priority={priority:.2f}: {gap}")

            # Contradictions
            contradictions = self.conn.execute(
                """SELECT c1.id, c1.fact, c2.id, c2.fact
                   FROM conditions c1
                   JOIN conditions c2 ON c1.contradiction_partner = c2.id
                   WHERE c1.contradiction_flag = TRUE
                     AND c1.id < c2.id
                   LIMIT 10"""
            ).fetchall()
            if contradictions:
                lines.append(f"\n--- CONTRADICTIONS ({len(contradictions)} pairs) ---")
                for c1_id, c1_fact, c2_id, c2_fact in contradictions:
                    lines.append(f"  [#{c1_id}]: {c1_fact[:150]}")
                    lines.append(f"  vs [#{c2_id}]: {c2_fact[:150]}")

            # Previous synthesis highlights
            prev_synthesis = self.get_synthesis(iteration)
            if prev_synthesis:
                lines.append(f"\n--- PRIOR SYNTHESIS (iteration {iteration}) ---")
                lines.append(prev_synthesis[:2000])
                if len(prev_synthesis) > 2000:
                    lines.append(f"... ({len(prev_synthesis) - 2000} more chars in full synthesis)")

            # Angle coverage
            angles = self.conn.execute(
                """SELECT angle, COUNT(*) as cnt,
                          AVG(confidence) as avg_conf
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND angle != ''
                   GROUP BY angle
                   ORDER BY cnt DESC"""
            ).fetchall()
            if angles:
                lines.append("\n--- ANGLE COVERAGE ---")
                for angle, cnt, avg_conf in angles:
                    lines.append(f"  [{angle}]: {cnt} findings, avg_conf={avg_conf:.2f}")

            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def build_report(
        self,
        user_query: str,
        iteration: int | None = None,
        include_sources: bool = True,
    ) -> str:
        """Build a full report from corpus state.

        Queries all synthesis rows + high-confidence findings.
        No truncation. Full text.

        Args:
            user_query: Original query for context.
            iteration: Filter to specific iteration, or None for all.
            include_sources: Whether to append source URLs.
        """
        with self._lock:
            lines: list[str] = []

            # Synthesis reports (the main content)
            syntheses = self.get_all_syntheses()
            if syntheses:
                for s in syntheses:
                    lines.append(f"\n{'='*60}")
                    lines.append(f"SYNTHESIS — Iteration {s['iteration']}")
                    lines.append(f"{'='*60}")
                    lines.append(s["fact"])
            else:
                lines.append("(No gossip synthesis completed — raw findings below)")

            # High-confidence findings not covered by synthesis
            high_conf = self.conn.execute(
                """SELECT fact, source_url, confidence, angle
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND confidence >= 0.7
                   ORDER BY confidence DESC
                   LIMIT 100"""
            ).fetchall()
            if high_conf:
                lines.append(f"\n{'='*60}")
                lines.append(f"HIGH-CONFIDENCE FINDINGS ({len(high_conf)})")
                lines.append(f"{'='*60}")
                for fact, src_url, conf, angle in high_conf:
                    src = f" — {src_url}" if src_url and include_sources else ""
                    lines.append(f"• [{conf:.2f}] {fact}{src}")

            # Source catalogue
            if include_sources:
                sources = self.conn.execute(
                    """SELECT DISTINCT source_url, source_type
                       FROM conditions
                       WHERE source_url != '' AND source_url IS NOT NULL
                       ORDER BY source_type, source_url"""
                ).fetchall()
                if sources:
                    lines.append(f"\n{'='*60}")
                    lines.append(f"SOURCES ({len(sources)})")
                    lines.append(f"{'='*60}")
                    for url, stype in sources:
                        lines.append(f"  [{stype}] {url}")

            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Count / stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of conditions in the corpus."""
        with self._lock:
            result = self.conn.execute("SELECT COUNT(*) FROM conditions").fetchone()
        return result[0] if result else 0

    def count_by_type(self) -> dict[str, int]:
        """Count conditions grouped by row_type."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT row_type, COUNT(*) FROM conditions GROUP BY row_type"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def count_since(self, since_id: int) -> int:
        """Count findings added after a given condition ID.

        Used by the gossip loop to determine if enough new material
        has accumulated to justify another gossip round.

        Args:
            since_id: Count findings with id > this value.

        Returns:
            Number of new findings since the given ID.
        """
        with self._lock:
            result = self.conn.execute(
                """SELECT COUNT(*) FROM conditions
                   WHERE id > ?
                     AND row_type = 'finding'
                     AND consider_for_use = TRUE""",
                [since_id],
            ).fetchone()
        return result[0] if result else 0

    def max_id(self) -> int:
        """Return the current maximum condition ID (0 if empty)."""
        with self._lock:
            result = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM conditions"
            ).fetchone()
        return result[0] if result else 0

    def count_findings(self, iteration: int | None = None) -> int:
        """Count active findings, optionally filtered by iteration."""
        with self._lock:
            if iteration is not None:
                result = self.conn.execute(
                    """SELECT COUNT(*) FROM conditions
                       WHERE row_type = 'finding'
                         AND consider_for_use = TRUE
                         AND iteration = ?""",
                    [iteration],
                ).fetchone()
            else:
                result = self.conn.execute(
                    """SELECT COUNT(*) FROM conditions
                       WHERE row_type = 'finding'
                         AND consider_for_use = TRUE"""
                ).fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict[str, Any]]:
        """Return ALL conditions — nothing excluded."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, fact, source_url, source_type, confidence,
                          trust_score, novelty_score, specificity_score,
                          relevance_score, actionability_score,
                          duplication_score, fabrication_risk,
                          verification_status, angle, parent_id, strategy,
                          expansion_depth, iteration, row_type
                   FROM conditions
                   ORDER BY id ASC"""
            ).fetchall()
        cols = [
            "id", "fact", "source_url", "source_type", "confidence",
            "trust_score", "novelty_score", "specificity_score",
            "relevance_score", "actionability_score", "duplication_score",
            "fabrication_risk", "verification_status", "angle", "parent_id",
            "strategy", "expansion_depth", "iteration", "row_type",
        ]
        return [dict(zip(cols, row)) for row in rows]

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

        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        cols = [
            "id", "fact", "source_url", "source_type", "confidence",
            "angle", "iteration", "verification_status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_contradictions(self) -> list[dict[str, Any]]:
        """Return all contradiction pairs."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT c1.id, c1.fact, c1.confidence,
                          c2.id, c2.fact, c2.confidence
                   FROM conditions c1
                   JOIN conditions c2 ON c1.contradiction_partner = c2.id
                   WHERE c1.contradiction_flag = TRUE
                     AND c1.id < c2.id
                   ORDER BY c1.id ASC"""
            ).fetchall()
        result = []
        for c1_id, c1_fact, c1_conf, c2_id, c2_fact, c2_conf in rows:
            result.append({
                "claim_a": {"id": c1_id, "fact": c1_fact, "confidence": c1_conf},
                "claim_b": {"id": c2_id, "fact": c2_fact, "confidence": c2_conf},
            })
        return result

    def get_expansion_hints(self) -> list[dict[str, Any]]:
        """Return unfulfilled expansion hints."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT id, fact, expansion_gap, expansion_priority, angle
                   FROM conditions
                   WHERE expansion_gap != ''
                     AND expansion_fulfilled = FALSE
                   ORDER BY expansion_priority DESC"""
            ).fetchall()
        cols = ["id", "fact", "expansion_gap", "expansion_priority", "angle"]
        return [dict(zip(cols, r)) for r in rows]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        with self._lock:
            try:
                self.conn.close()
            except Exception:
                pass
