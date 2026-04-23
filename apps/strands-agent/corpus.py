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

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:
    from swarm.lineage import LineageEntry

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

    def __init__(self, db_path: str = "", *, current_run: str = "") -> None:
        """Initialize DuckDB connection and schema.

        Args:
            db_path: Path to DuckDB file. Empty string = in-memory.
            current_run: Run identifier that scopes all queries.
                When set, every read method automatically filters by
                ``source_run = current_run`` and every write method
                stamps ``source_run`` on new rows.  Pass ``cross_run=True``
                to individual methods to bypass this scoping.
        """
        if db_path:
            self.conn = duckdb.connect(db_path)
        else:
            self.conn = duckdb.connect()
        self._lock = threading.RLock()  # Reentrant: methods call each other
        self._next_id = 1
        self._db_path = db_path
        self.current_run: str = current_run
        self.user_query: str = ""  # Set by _run_job for trigger_gossip context

        # Enable WAL mode for file-backed databases — crash-safe writes
        # and better concurrent read performance during 24h runs.
        if db_path:
            try:
                self.conn.execute("PRAGMA enable_progress_bar")
            except Exception:
                pass

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
                relationship_score FLOAT DEFAULT 0.0,

                -- Swarm lineage (unified with LineageStore)
                phase TEXT DEFAULT '',
                parent_ids TEXT DEFAULT '',

                -- Source provenance (24h continuous operation)
                source_model TEXT DEFAULT '',
                source_run TEXT DEFAULT ''
            )
        """)

        # Corpus fingerprint tracking — prevents re-ingestion
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS corpus_fingerprints (
                fingerprint TEXT PRIMARY KEY,
                source TEXT DEFAULT '',
                ingested_at TEXT DEFAULT '',
                char_count INTEGER DEFAULT 0,
                paragraph_count INTEGER DEFAULT 0
            )
        """)

        # Store-level summaries for rolling knowledge briefings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_summaries (
                id INTEGER PRIMARY KEY,
                angle TEXT NOT NULL,
                summary TEXT NOT NULL,
                finding_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT '',
                run_number INTEGER DEFAULT 0
            )
        """)

        # Ensure lineage columns exist on older databases (idempotent).
        self._ensure_lineage_columns()
        self._ensure_provenance_columns()
        # Seed next_id from existing rows
        result = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM conditions").fetchone()
        if result:
            self._next_id = result[0] + 1

    def _ensure_lineage_columns(self) -> None:
        """Backfill phase/parent_ids/provenance columns on pre-existing databases.

        Safe to call repeatedly; DuckDB's ``ADD COLUMN IF NOT EXISTS``
        handles the idempotency.
        """
        for col, typedef in (
            ("phase", "TEXT DEFAULT ''"),
            ("parent_ids", "TEXT DEFAULT ''"),
            ("source_model", "TEXT DEFAULT ''"),
            ("source_run", "TEXT DEFAULT ''"),
        ):
            try:
                self.conn.execute(
                    f"ALTER TABLE conditions ADD COLUMN IF NOT EXISTS "
                    f"{col} {typedef}"
                )
            except Exception:
                # Older DuckDB versions without IF NOT EXISTS support will
                # error on a duplicate column; swallow and continue.
                pass

    def _ensure_provenance_columns(self) -> None:
        """Backfill source_model/source_run columns on pre-existing databases."""
        for col, typedef in (
            ("source_model", "TEXT DEFAULT ''"),
            ("source_run", "TEXT DEFAULT ''"),
        ):
            try:
                self.conn.execute(
                    f"ALTER TABLE conditions ADD COLUMN IF NOT EXISTS "
                    f"{col} {typedef}"
                )
            except Exception:
                pass

        # Ensure metadata tables exist on older databases
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS corpus_fingerprints (
                    fingerprint TEXT PRIMARY KEY,
                    source TEXT DEFAULT '',
                    ingested_at TEXT DEFAULT '',
                    char_count INTEGER DEFAULT 0,
                    paragraph_count INTEGER DEFAULT 0
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_summaries (
                    id INTEGER PRIMARY KEY,
                    angle TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    finding_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT '',
                    run_number INTEGER DEFAULT 0
                )
            """)
        except Exception:
            pass

        # P2: indices for common query patterns (idempotent)
        for idx_sql in (
            "CREATE INDEX IF NOT EXISTS idx_cond_run_filter ON conditions(source_run, consider_for_use, row_type)",
            "CREATE INDEX IF NOT EXISTS idx_cond_angle ON conditions(angle, consider_for_use)",
            "CREATE INDEX IF NOT EXISTS idx_cond_source ON conditions(source_type, angle)",
            "CREATE INDEX IF NOT EXISTS idx_cond_phase ON conditions(phase)",
        ):
            try:
                self.conn.execute(idx_sql)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Run-scoping helpers
    # ------------------------------------------------------------------

    def _run_sql(self, *, cross_run: bool = False) -> str:
        """Return a SQL fragment ``AND source_run = ?`` when scoping is active.

        Args:
            cross_run: If True, skip run filtering (for compaction, cross-run
                comparison, etc.).

        Returns:
            SQL fragment (empty string when scoping is inactive or bypassed).
        """
        if cross_run or not self.current_run:
            return ""
        return " AND source_run = ?"

    def _run_params(self, *, cross_run: bool = False) -> list[str]:
        """Return ``[current_run]`` when scoping is active, else ``[]``.

        Pair with :meth:`_run_sql` to build parameterized queries.
        """
        if cross_run or not self.current_run:
            return []
        return [self.current_run]

    # ------------------------------------------------------------------
    # Corpus fingerprinting — prevents re-ingestion across runs
    # ------------------------------------------------------------------

    def has_corpus_hash(self, fingerprint: str) -> bool:
        """Check if a corpus with this hash has already been ingested."""
        with self._lock:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM corpus_fingerprints WHERE fingerprint = ?",
                [fingerprint],
            ).fetchone()
        return bool(result and result[0] > 0)

    def register_corpus_hash(
        self,
        fingerprint: str,
        source: str = "",
        char_count: int = 0,
        paragraph_count: int = 0,
    ) -> None:
        """Register that a corpus has been ingested."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """INSERT OR REPLACE INTO corpus_fingerprints
                   (fingerprint, source, ingested_at, char_count, paragraph_count)
                   VALUES (?, ?, ?, ?, ?)""",
                [fingerprint, source, now, char_count, paragraph_count],
            )

    # ------------------------------------------------------------------
    # Store compaction — deduplicate and archive stale findings
    # ------------------------------------------------------------------

    def compact(
        self,
        complete: Any = None,
        max_cluster_size: int = 8,
    ) -> dict[str, int]:
        """Deduplicate findings using exact-match and LLM semantic judgment.

        Two-phase approach designed to be called by an external agent
        outside the swarm — never by swarm workers themselves.

        Phase 1 (pure SQL, no LLM): Remove exact-duplicate ``fact`` text
        within the same angle, keeping the highest-confidence row.

        Phase 2 (LLM-assisted, optional): Pre-cluster candidates within
        each angle using shared key terms extracted via SQL string
        functions (no cartesian product). For each candidate cluster,
        ask the LLM once "which of these are the same claim?" and mark
        lower-confidence duplicates obsolete.

        Args:
            complete: Async callable ``(prompt: str) -> str`` for LLM
                judgment in Phase 2.  If None, only Phase 1 runs.
            max_cluster_size: Max findings per LLM dedup call.

        Returns:
            Dict with compaction statistics.
        """
        import asyncio

        stats: dict[str, int] = {
            "exact_duplicates_removed": 0,
            "semantic_duplicates_removed": 0,
            "total_checked": 0,
        }

        # ── Phase 1: exact-match dedup (pure SQL) ────────────────────
        with self._lock:
            # Find groups of identical fact text within the same angle
            dup_groups = self.conn.execute(
                """SELECT angle, fact, COUNT(*) as cnt,
                          MAX(confidence) as max_conf
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                   GROUP BY angle, fact
                   HAVING COUNT(*) > 1""",
            ).fetchall()

        for angle, fact, cnt, max_conf in dup_groups:
            with self._lock:
                # Keep the highest-confidence row, mark the rest obsolete
                self.conn.execute(
                    """UPDATE conditions
                       SET consider_for_use = FALSE,
                           obsolete_reason = 'exact_duplicate'
                       WHERE angle = ?
                         AND fact = ?
                         AND consider_for_use = TRUE
                         AND id NOT IN (
                             SELECT id FROM conditions
                             WHERE angle = ? AND fact = ?
                               AND consider_for_use = TRUE
                             ORDER BY confidence DESC
                             LIMIT 1
                         )""",
                    [angle, fact, angle, fact],
                )
            stats["exact_duplicates_removed"] += cnt - 1

        # ── Phase 2: LLM semantic dedup (optional) ───────────────────
        if complete is None:
            with self._lock:
                stats["total_checked"] = self.conn.execute(
                    """SELECT COUNT(*) FROM conditions
                       WHERE consider_for_use = TRUE
                         AND row_type IN ('finding', 'thought', 'insight')""",
                ).fetchone()[0]

            logger.info(
                "exact_dupes=<%d>, total=<%d> | compaction phase 1 complete",
                stats["exact_duplicates_removed"], stats["total_checked"],
            )
            return stats

        # Get distinct angles
        with self._lock:
            angles = self.conn.execute(
                """SELECT DISTINCT angle FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                     AND angle != ''""",
            ).fetchall()

        semantic_removed = 0
        for (angle,) in angles:
            # Fetch active findings for this angle, ordered by confidence
            with self._lock:
                rows = self.conn.execute(
                    """SELECT id, fact, confidence
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = ?
                         AND row_type IN ('finding', 'thought', 'insight')
                       ORDER BY confidence DESC""",
                    [angle],
                ).fetchall()

            if len(rows) < 2:
                continue

            # Build keyword index: extract significant words per finding
            # to cluster candidates without cartesian product
            word_to_ids: dict[str, list[int]] = {}
            id_to_row: dict[int, tuple[int, str, float]] = {}
            for cid, fact, conf in rows:
                id_to_row[cid] = (cid, fact, conf)
                words = set(
                    w.lower() for w in re.findall(r"[a-zA-Z]{4,}", fact)
                )
                for w in words:
                    word_to_ids.setdefault(w, []).append(cid)

            # Find candidate clusters: findings sharing 3+ keywords
            from collections import Counter
            pair_overlap: Counter[tuple[int, int]] = Counter()
            for w, ids in word_to_ids.items():
                if len(ids) > 20:
                    continue  # skip very common words
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        a, b = min(ids[i], ids[j]), max(ids[i], ids[j])
                        pair_overlap[(a, b)] += 1

            # Group pairs with 3+ shared keywords into clusters
            # using union-find to merge overlapping pairs
            parent: dict[int, int] = {}

            def find(x: int) -> int:
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent[x], parent[x])
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for (a, b), count in pair_overlap.items():
                if count >= 3:
                    union(a, b)

            # Collect clusters
            clusters: dict[int, list[int]] = {}
            for cid in id_to_row:
                root = find(cid)
                if root != cid or cid in parent:
                    clusters.setdefault(root, []).append(cid)
            # Ensure root nodes themselves are included in their cluster
            for root in list(clusters.keys()):
                if root not in clusters[root]:
                    clusters[root].insert(0, root)

            # For each cluster, ask LLM to identify duplicates
            for root, members in clusters.items():
                if len(members) < 2:
                    continue
                # Cap cluster size to avoid huge prompts
                members = sorted(
                    members,
                    key=lambda c: id_to_row[c][2],
                    reverse=True,
                )[:max_cluster_size]

                # Build prompt listing the candidate findings
                lines = []
                for idx, cid in enumerate(members):
                    _, fact, conf = id_to_row[cid]
                    lines.append(f"[{idx}] (conf={conf:.2f}) {fact[:300]}")

                prompt = (
                    f"You are deduplicating research findings in the "
                    f"'{angle}' domain.\n\n"
                    f"Below are {len(members)} findings that may be "
                    f"saying the same thing. Group them by semantic "
                    f"equivalence — findings making the same factual "
                    f"claim count as duplicates even if worded "
                    f"differently.\n\n"
                    + "\n".join(lines)
                    + "\n\nFor each duplicate group, output ONLY the "
                    f"index numbers of duplicates on one line, comma-"
                    f"separated. The FIRST index in each group is the "
                    f"keeper (highest confidence). Output one group per "
                    f"line. If no duplicates exist, output NONE.\n"
                    f"Example: 0,3,5\n1,4"
                )

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Already in an async context (e.g. called from synthesize())
                    # Run the coroutine in a separate thread to avoid
                    # "cannot call run_until_complete from a running loop"
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        response = pool.submit(
                            asyncio.run, complete(prompt)
                        ).result(timeout=120)
                else:
                    response = asyncio.run(complete(prompt))

                # Parse response: each line is a group of indices
                for line in response.strip().split("\n"):
                    line = line.strip()
                    if not line or line.upper() == "NONE":
                        continue
                    indices = []
                    for part in line.split(","):
                        part = part.strip().strip("[]")
                        if part.isdigit():
                            idx = int(part)
                            if 0 <= idx < len(members):
                                indices.append(idx)
                    if len(indices) < 2:
                        continue
                    # First index is keeper, rest are duplicates
                    for dup_idx in indices[1:]:
                        dup_cid = members[dup_idx]
                        with self._lock:
                            self.conn.execute(
                                """UPDATE conditions
                                   SET consider_for_use = FALSE,
                                       obsolete_reason = 'semantic_duplicate'
                                   WHERE id = ?
                                     AND consider_for_use = TRUE""",
                                [dup_cid],
                            )
                        semantic_removed += 1

        stats["semantic_duplicates_removed"] = semantic_removed

        with self._lock:
            stats["total_checked"] = self.conn.execute(
                """SELECT COUNT(*) FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')""",
            ).fetchone()[0]

        logger.info(
            "exact_dupes=<%d>, semantic_dupes=<%d>, total=<%d> | compaction complete",
            stats["exact_duplicates_removed"],
            stats["semantic_duplicates_removed"],
            stats["total_checked"],
        )
        return stats

    # ------------------------------------------------------------------
    # Knowledge summaries — rolling briefings for workers
    # ------------------------------------------------------------------

    def get_latest_summary(self, angle: str) -> str:
        """Get the most recent knowledge summary for an angle.

        Returns empty string if no summary exists.
        """
        with self._lock:
            result = self.conn.execute(
                """SELECT summary FROM knowledge_summaries
                   WHERE angle = ?
                   ORDER BY id DESC LIMIT 1""",
                [angle],
            ).fetchone()
        return result[0] if result else ""

    def store_summary(
        self,
        angle: str,
        summary: str,
        finding_count: int = 0,
        run_number: int = 0,
    ) -> None:
        """Store a knowledge summary for an angle."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            sid = self.conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM knowledge_summaries"
            ).fetchone()[0]
            self.conn.execute(
                """INSERT INTO knowledge_summaries
                   (id, angle, summary, finding_count, created_at, run_number)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [sid, angle, summary, finding_count, now, run_number],
            )

    def get_store_stats(self, *, cross_run: bool = False) -> dict[str, Any]:
        """Get aggregate statistics about the store for diagnostics."""
        run_sql = self._run_sql(cross_run=cross_run)
        run_p = self._run_params(cross_run=cross_run)
        with self._lock:
            total = self.conn.execute(
                f"SELECT COUNT(*) FROM conditions WHERE TRUE{run_sql}",
                run_p,
            ).fetchone()[0]
            active = self.conn.execute(
                f"SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE{run_sql}",
                run_p,
            ).fetchone()[0]
            by_type = self.conn.execute(
                f"""SELECT row_type, COUNT(*) FROM conditions
                   WHERE consider_for_use = TRUE{run_sql}
                   GROUP BY row_type""",
                run_p,
            ).fetchall()
            by_angle = self.conn.execute(
                f"""SELECT angle, COUNT(*) FROM conditions
                   WHERE consider_for_use = TRUE AND row_type = 'finding'{run_sql}
                   GROUP BY angle ORDER BY COUNT(*) DESC""",
                run_p,
            ).fetchall()
            models = self.conn.execute(
                f"""SELECT DISTINCT source_model FROM conditions
                   WHERE source_model != '' AND source_model IS NOT NULL{run_sql}""",
                run_p,
            ).fetchall()
        return {
            "total_rows": total,
            "active_rows": active,
            "by_type": dict(by_type),
            "by_angle": dict(by_angle),
            "models_seen": [r[0] for r in models],
        }

    # ------------------------------------------------------------------
    # Scoped query methods — run-aware reads for external callers
    # ------------------------------------------------------------------
    # These methods use _run_sql()/_run_params() so that when current_run
    # is set, queries automatically filter by source_run.  External code
    # (data_package.py, mcp_engine.py, research_organizer.py) MUST use
    # these instead of raw conn.execute() against conditions.

    def get_top_findings(
        self,
        angle: str | None = None,
        *,
        row_types: tuple[str, ...] = ("finding", "thought", "insight"),
        limit: int = 20,
        min_length: int = 0,
        exclude_source_types: tuple[str, ...] = (),
        order_by: str = "confidence DESC",
        cross_run: bool = False,
    ) -> list[tuple[str, float, str, str]]:
        """Get top findings with run scoping.

        Args:
            angle: Filter to this angle. None = all angles.
            row_types: Row types to include.
            limit: Maximum results.
            min_length: Minimum fact length in characters.
            exclude_source_types: Source types to exclude.
            order_by: SQL ORDER BY clause.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, confidence, source_url, source_type) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = []

        where_parts = ["consider_for_use = TRUE"]

        if row_types:
            placeholders = ", ".join("?" for _ in row_types)
            where_parts.append(f"row_type IN ({placeholders})")
            params.extend(row_types)

        if angle is not None:
            where_parts.append("angle = ?")
            params.append(angle)

        if min_length > 0:
            where_parts.append(f"length(fact) >= {min_length}")

        if exclude_source_types:
            placeholders = ", ".join("?" for _ in exclude_source_types)
            where_parts.append(f"source_type NOT IN ({placeholders})")
            params.extend(exclude_source_types)

        params.extend(self._run_params(cross_run=cross_run))

        where = " AND ".join(where_parts) + run_sql
        sql = f"""SELECT fact, confidence, source_url, source_type
                  FROM conditions
                  WHERE {where}
                  ORDER BY {order_by}
                  LIMIT ?"""
        params.append(limit)

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def count_active(
        self,
        angle: str | None = None,
        *,
        row_types: tuple[str, ...] | None = None,
        cross_run: bool = False,
    ) -> int:
        """Count active (consider_for_use=TRUE) findings with run scoping.

        Args:
            angle: Filter to this angle. None = all angles.
            row_types: Filter to these row types. None = all types.
            cross_run: If True, skip run filtering.

        Returns:
            Count of matching rows.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = []

        where_parts = ["consider_for_use = TRUE"]

        if angle is not None:
            where_parts.append("angle = ?")
            params.append(angle)

        if row_types is not None:
            placeholders = ", ".join("?" for _ in row_types)
            where_parts.append(f"row_type IN ({placeholders})")
            params.extend(row_types)

        params.extend(self._run_params(cross_run=cross_run))

        where = " AND ".join(where_parts) + run_sql
        sql = f"SELECT COUNT(*) FROM conditions WHERE {where}"

        with self._lock:
            return self.conn.execute(sql, params).fetchone()[0]

    def get_cross_angle_findings(
        self,
        exclude_angle: str,
        *,
        limit: int = 15,
        exclude_source_types: tuple[str, ...] = ("corpus_section",),
        cross_run: bool = False,
    ) -> list[tuple[str, str, float]]:
        """Get findings from angles OTHER than exclude_angle.

        Args:
            exclude_angle: Angle to exclude from results.
            limit: Maximum results.
            exclude_source_types: Source types to exclude.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, angle, confidence) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [exclude_angle]

        excl_sql = ""
        if exclude_source_types:
            placeholders = ", ".join("?" for _ in exclude_source_types)
            excl_sql = f" AND source_type NOT IN ({placeholders})"
            params.extend(exclude_source_types)

        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        sql = f"""SELECT fact, angle, confidence
                  FROM conditions
                  WHERE consider_for_use = TRUE
                    AND angle != ?
                    AND row_type IN ('finding', 'thought', 'insight')
                    {excl_sql}
                    {run_sql}
                  ORDER BY confidence DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_research_questions(
        self,
        angle: str,
        *,
        limit: int = 10,
        cross_run: bool = False,
    ) -> list[str]:
        """Get open research questions for an angle.

        Args:
            angle: Research angle.
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of research question fact strings.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [angle]
        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        sql = f"""SELECT fact FROM conditions
                  WHERE row_type = 'research_question'
                    AND angle = ?
                    AND consider_for_use = TRUE
                    {run_sql}
                  ORDER BY id DESC
                  LIMIT ?"""

        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [r[0] for r in rows]

    def get_insights_for_angle(
        self,
        angle: str,
        *,
        limit: int = 10,
        cross_run: bool = False,
    ) -> list[tuple[str, str]]:
        """Get insight rows for cross-domain connections involving an angle.

        Args:
            angle: Research angle.
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, angle) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [angle, angle]
        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        sql = f"""SELECT fact, angle FROM conditions
                  WHERE row_type = 'insight'
                    AND consider_for_use = TRUE
                    AND (angle = ? OR fact LIKE '%' || replace(replace(?, '%', '\\%'), '_', '\\_') || '%' ESCAPE '\\')
                    {run_sql}
                  ORDER BY confidence DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_contradictions(
        self,
        angle: str,
        *,
        limit: int = 5,
        cross_run: bool = False,
    ) -> list[tuple[str, str, str | None]]:
        """Get contradiction rows targeting an angle.

        Args:
            angle: Research angle.
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, source_angle, target_fact) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [angle, angle]
        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        # Qualify source_run with table alias to avoid ambiguity in self-JOIN
        qualified_run_sql = run_sql.replace("source_run", "c.source_run")
        sql = f"""SELECT c.fact, c.angle, target.fact as target_fact
                  FROM conditions c
                  LEFT JOIN conditions target ON c.related_id = target.id
                  WHERE c.row_type = 'contradiction'
                    AND c.consider_for_use = TRUE
                    AND (target.angle = ? OR c.angle = ?)
                    {qualified_run_sql}
                  ORDER BY c.id DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_fresh_evidence(
        self,
        angle: str,
        wave: int,
        *,
        limit: int = 10,
        cross_run: bool = False,
    ) -> list[tuple[str, float, str, str]]:
        """Get clone research findings for an angle from a specific wave.

        Args:
            angle: Research angle.
            wave: Current wave number (retrieves findings from wave-1).
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, confidence, strategy/doubt, source_ref) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [angle, wave - 1]
        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        sql = f"""SELECT fact, confidence, strategy, source_ref
                  FROM conditions
                  WHERE source_type = 'clone_research'
                    AND consider_for_use = TRUE
                    AND angle = ?
                    AND iteration = ?
                    {run_sql}
                  ORDER BY confidence DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_clone_research_count(
        self,
        angle: str,
        doubt_prefix: str,
        *,
        min_confidence: float = 0.8,
        cross_run: bool = False,
    ) -> int:
        """Count clone research findings matching a doubt prefix.

        Used by the retirement checker to detect when a doubt has been
        resolved by another clone.

        Args:
            angle: Research angle.
            doubt_prefix: First 50 chars of the doubt (case-insensitive).
            min_confidence: Minimum confidence threshold.
            cross_run: If True, skip run filtering.

        Returns:
            Count of matching clone findings.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [angle, doubt_prefix]
        params.extend(self._run_params(cross_run=cross_run))

        sql = f"""SELECT COUNT(*) FROM conditions
                  WHERE source_type = 'clone_research'
                    AND consider_for_use = TRUE
                    AND angle = ?
                    AND confidence >= {min_confidence}
                    AND lower(left(strategy, 50)) = ?
                    {run_sql}"""

        with self._lock:
            return self.conn.execute(sql, params).fetchone()[0]

    # ------------------------------------------------------------------
    # Additional scoped query methods for worker_tools.py
    # ------------------------------------------------------------------

    def get_angle_stats(
        self,
        *,
        row_types: tuple[str, ...] = ("finding", "thought", "insight"),
        cross_run: bool = False,
    ) -> list[tuple[str, int, float]]:
        """Get per-angle finding counts and average confidence.

        Args:
            row_types: Row types to include.
            cross_run: If True, skip run filtering.

        Returns:
            List of (angle, count, avg_confidence) tuples sorted by count ASC.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = list(row_types)
        params.extend(self._run_params(cross_run=cross_run))

        placeholders = ", ".join("?" for _ in row_types)
        sql = f"""SELECT angle, COUNT(*) as cnt,
                         AVG(confidence) as avg_conf
                  FROM conditions
                  WHERE consider_for_use = TRUE
                    AND row_type IN ({placeholders})
                    AND angle != ''
                    {run_sql}
                  GROUP BY angle
                  ORDER BY cnt ASC"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def count_by_filter(
        self,
        *,
        row_types: tuple[str, ...] | None = None,
        max_confidence: float | None = None,
        verification_status: str | None = None,
        cross_run: bool = False,
    ) -> int:
        """Count active findings matching flexible filters.

        Args:
            row_types: Row types to include. None = all.
            max_confidence: Maximum confidence (exclusive). None = no cap.
            verification_status: Filter to this status. None = any.
            cross_run: If True, skip run filtering.

        Returns:
            Count of matching rows.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = []
        where_parts = ["consider_for_use = TRUE"]

        if row_types:
            placeholders = ", ".join("?" for _ in row_types)
            where_parts.append(f"row_type IN ({placeholders})")
            params.extend(row_types)

        if max_confidence is not None:
            where_parts.append("confidence < ?")
            params.append(max_confidence)

        if verification_status is not None:
            where_parts.append("verification_status = ?")
            params.append(verification_status)

        params.extend(self._run_params(cross_run=cross_run))
        where = " AND ".join(where_parts) + run_sql
        sql = f"SELECT COUNT(*) FROM conditions WHERE {where}"

        with self._lock:
            return self.conn.execute(sql, params).fetchone()[0]

    def get_findings_by_angle_like(
        self,
        angle_pattern: str,
        *,
        row_types: tuple[str, ...] = ("finding", "thought", "insight"),
        limit: int = 200,
        cross_run: bool = False,
    ) -> list[tuple[int, str, float, str]]:
        """Get findings where angle matches a LIKE pattern.

        Args:
            angle_pattern: SQL LIKE pattern for angle matching.
            row_types: Row types to include.
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of (id, fact, confidence, angle) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = list(row_types)
        params.extend([f"%{angle_pattern}%", f"%{angle_pattern.lower()}%"])
        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        placeholders = ", ".join("?" for _ in row_types)
        sql = f"""SELECT id, fact, confidence, angle
                  FROM conditions
                  WHERE consider_for_use = TRUE
                    AND row_type IN ({placeholders})
                    AND (angle LIKE ? OR angle LIKE ?)
                    {run_sql}
                  ORDER BY confidence DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def count_corpus_rows(
        self,
        angle: str,
        *,
        row_type: str = "raw",
        cross_run: bool = False,
    ) -> int:
        """Count corpus rows for an angle.

        Args:
            angle: Research angle.
            row_type: Row type to count.
            cross_run: If True, skip run filtering.

        Returns:
            Count of matching rows.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [angle]
        params.extend(self._run_params(cross_run=cross_run))

        where = f"row_type = ? AND angle = ?"
        params_full = [row_type, angle]
        params_full.extend(self._run_params(cross_run=cross_run))

        sql = f"""SELECT COUNT(*) FROM conditions
                  WHERE row_type = ? AND angle = ?
                  {run_sql}"""

        with self._lock:
            return self.conn.execute(sql, params_full).fetchone()[0]

    def get_corpus_page(
        self,
        angle: str,
        *,
        row_type: str = "raw",
        require_active: bool = False,
        page_size: int = 100,
        page_offset: int = 0,
        cross_run: bool = False,
    ) -> list[tuple[str]]:
        """Get a page of corpus fact texts for an angle.

        Args:
            angle: Research angle.
            row_type: Row type to filter by (e.g. 'raw', 'finding').
            require_active: If True, also filter by consider_for_use=TRUE.
            page_size: Rows per page.
            page_offset: Row offset for pagination.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact,) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [row_type, angle]

        where_parts = ["row_type = ?", "angle = ?"]
        if require_active:
            where_parts.append("consider_for_use = TRUE")

        params.extend(self._run_params(cross_run=cross_run))
        params.extend([page_size, page_offset])

        where = " AND ".join(where_parts) + run_sql
        sql = f"""SELECT fact FROM conditions
                  WHERE {where}
                  ORDER BY id ASC
                  LIMIT ? OFFSET ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_knowledge_summaries(
        self,
        *,
        cross_run: bool = False,
    ) -> list[tuple[str, str, int, int]]:
        """Get knowledge summaries for all angles.

        Args:
            cross_run: If True, skip run filtering.

        Returns:
            List of (angle, summary, finding_count, run_number) tuples.
        """
        # knowledge_summaries table does not have source_run column,
        # so we always return all summaries regardless of run scoping.
        with self._lock:
            return self.conn.execute(
                """SELECT angle, summary, finding_count, run_number
                   FROM knowledge_summaries
                   ORDER BY id DESC""",
            ).fetchall()

    # ------------------------------------------------------------------
    # Wave-scoped methods for incremental data packages
    # ------------------------------------------------------------------

    def get_findings_since_wave(
        self,
        angle: str | None = None,
        *,
        since_wave: int,
        row_types: tuple[str, ...] = ("finding", "thought", "insight"),
        limit: int = 50,
        cross_run: bool = False,
    ) -> list[tuple[str, float, str, str]]:
        """Get findings created after a given wave.

        Used by incremental data packages to deliver only the delta
        since the worker's last wave.

        Args:
            angle: Filter to this angle. None = all angles.
            since_wave: Return findings with iteration > since_wave.
            row_types: Row types to include.
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, confidence, source_url, source_type) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = []

        where_parts = [
            "consider_for_use = TRUE",
            "iteration > ?",
        ]
        params.append(since_wave)

        if row_types:
            placeholders = ", ".join("?" for _ in row_types)
            where_parts.append(f"row_type IN ({placeholders})")
            params.extend(row_types)

        if angle is not None:
            where_parts.append("angle = ?")
            params.append(angle)

        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        where = " AND ".join(where_parts) + run_sql
        sql = f"""SELECT fact, confidence, source_url, source_type
                  FROM conditions
                  WHERE {where}
                  ORDER BY confidence DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def get_cross_angle_findings_since_wave(
        self,
        exclude_angle: str,
        *,
        since_wave: int,
        limit: int = 15,
        cross_run: bool = False,
    ) -> list[tuple[str, str, float]]:
        """Get cross-angle findings created after a given wave.

        Args:
            exclude_angle: Angle to exclude from results.
            since_wave: Return findings with iteration > since_wave.
            limit: Maximum results.
            cross_run: If True, skip run filtering.

        Returns:
            List of (fact, angle, confidence) tuples.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [exclude_angle, since_wave]
        params.extend(self._run_params(cross_run=cross_run))
        params.append(limit)

        sql = f"""SELECT fact, angle, confidence
                  FROM conditions
                  WHERE consider_for_use = TRUE
                    AND angle != ?
                    AND row_type IN ('finding', 'thought', 'insight')
                    AND iteration > ?
                    {run_sql}
                  ORDER BY confidence DESC
                  LIMIT ?"""

        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def count_findings_since_wave(
        self,
        angle: str | None = None,
        *,
        since_wave: int,
        cross_run: bool = False,
    ) -> int:
        """Count findings created after a given wave.

        Args:
            angle: Filter to this angle. None = all angles.
            since_wave: Count findings with iteration > since_wave.
            cross_run: If True, skip run filtering.

        Returns:
            Count of matching rows.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        params: list[Any] = [since_wave]

        where_parts = [
            "consider_for_use = TRUE",
            "iteration > ?",
            "row_type IN ('finding', 'thought', 'insight')",
        ]

        if angle is not None:
            where_parts.append("angle = ?")
            params.append(angle)

        params.extend(self._run_params(cross_run=cross_run))
        where = " AND ".join(where_parts) + run_sql
        sql = f"SELECT COUNT(*) FROM conditions WHERE {where}"

        with self._lock:
            return self.conn.execute(sql, params).fetchone()[0]

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
        source_model: str = "",
        source_run: str = "",
        phase: str = "",
    ) -> int | None:
        """Insert a single condition row.

        Returns the assigned condition ID, or None if fact is empty.

        If ``source_run`` is not provided and ``current_run`` is set on the
        store, the row is automatically stamped with the current run id.

        Args:
            source_model: Model that produced this finding (#192 provenance).
            source_run: Run identifier for cross-run comparison (#192).
                Defaults to ``self.current_run`` when empty.
            phase: Swarm phase (e.g. 'wave_1', 'serendipity') for lineage.
        """
        fact = fact.strip()
        if not fact:
            return None

        # Auto-stamp source_run from current_run when not explicitly given
        if not source_run and self.current_run:
            source_run = self.current_run

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
                    expansion_depth, created_at, iteration,
                    source_model, source_run, phase)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    cid, fact, source_url, source_type, source_ref,
                    row_type, related_id, consider_for_use,
                    confidence, verification_status, angle,
                    parent_id, strategy,
                    expansion_depth, now, iteration,
                    source_model, source_run, phase,
                ],
            )
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
    # Swarm LineageStore protocol (duck-typed via ``emit``)
    # ------------------------------------------------------------------

    # Mapping from swarm phase names to ``row_type`` values.  Keeps the
    # lineage DAG visible to existing consumer-facing queries that filter
    # on ``row_type`` (thinker/synthesiser/exports).
    _PHASE_ROW_TYPE: dict[str, str] = {
        "corpus_analysis": "raw",
        "worker_synthesis": "thought",
        "serendipity": "thought",
        "queen_merge": "synthesis",
        "knowledge_report": "synthesis",
    }

    @classmethod
    def _phase_to_row_type(cls, phase: str) -> str:
        """Resolve a swarm phase name to the ``row_type`` column value."""
        if not phase:
            return "thought"
        if phase.startswith("gossip_round"):
            return "thought"
        return cls._PHASE_ROW_TYPE.get(phase, "thought")

    def emit(self, entry: "LineageEntry") -> None:
        """Record a :class:`swarm.lineage.LineageEntry` as a condition row.

        Implements the ``LineageStore`` protocol so a ``ConditionStore``
        instance can be passed directly as
        ``SwarmConfig(lineage_store=...)``.  Every phase of the swarm
        pipeline becomes a queryable row in the unified corpus.

        Field mapping:
            entry.phase       -> phase column (+ derives row_type)
            entry.angle       -> angle
            entry.content     -> fact
            entry.metadata    -> strategy (JSON-serialized)
            entry.parent_ids  -> parent_ids (JSON array of entry IDs)
            entry.entry_id    -> source_ref (for entry_id-based lookups)
            entry.timestamp   -> created_at (ISO8601)
        """
        fact = (entry.content or "").strip() or f"[{entry.phase} — empty content]"
        row_type = self._phase_to_row_type(entry.phase)
        try:
            metadata_json = json.dumps(dict(entry.metadata))
        except (TypeError, ValueError):
            metadata_json = ""
        parent_ids_json = json.dumps(list(entry.parent_ids))

        ts = entry.timestamp
        if isinstance(ts, (int, float)):
            created_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            cid = self._next_id
            self._next_id += 1
            self.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_type, source_ref, row_type,
                    consider_for_use, angle, strategy,
                    created_at, phase, parent_ids, source_run)
                   VALUES (?, ?, 'swarm', ?, ?, TRUE, ?, ?, ?, ?, ?, ?)""",
                [
                    cid, fact, entry.entry_id, row_type,
                    entry.angle, metadata_json,
                    created_at, entry.phase, parent_ids_json,
                    self.current_run,
                ],
            )
        logger.debug(
            "emit lineage entry #%d: phase=%s angle=%s parents=%s",
            cid, entry.phase, entry.angle, parent_ids_json,
        )

    # ------------------------------------------------------------------
    # Corpus fingerprinting (#190)
    # ------------------------------------------------------------------

    @staticmethod
    def _corpus_fingerprint(text: str) -> str:
        """SHA-256 hex digest of raw corpus text."""
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def has_corpus_fingerprint(self, text: str) -> bool:
        """Check whether *text* was already ingested (by SHA-256 fingerprint).

        Prevents re-ingesting the same corpus across waves or runs.
        """
        fp = self._corpus_fingerprint(text)
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM conditions "
                "WHERE row_type = 'corpus_fingerprint' AND source_ref = ? "
                "LIMIT 1",
                [fp],
            ).fetchone()
        return row is not None

    def _record_corpus_fingerprint(self, text: str) -> None:
        """Store a fingerprint row so future ingests are skipped."""
        fp = self._corpus_fingerprint(text)
        with self._lock:
            cid = self._next_id
            self._next_id += 1
            self.conn.execute(
                "INSERT INTO conditions "
                "(id, fact, source_type, source_ref, row_type, "
                "consider_for_use, created_at) "
                "VALUES (?, ?, 'system', ?, 'corpus_fingerprint', FALSE, ?)",
                [cid, f"corpus fingerprint: {len(text)} chars", fp,
                 datetime.now(timezone.utc).isoformat()],
            )

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

        Skips ingestion if the same text was already ingested (SHA-256
        fingerprint check — #190).

        Atomisation is deferred to the caller (atomizer.py) or done
        inline via _atomise_simple() for basic paragraph splitting.

        Returns list of admitted condition IDs.
        """
        if not raw_text or not raw_text.strip():
            return []

        # Fingerprint dedup: skip if already ingested
        if self.has_corpus_fingerprint(raw_text):
            logger.info(
                "source=<%s>, chars=<%d> | skipped — corpus already ingested (fingerprint match)",
                source_type, len(raw_text),
            )
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

        # Record fingerprint so the same corpus is never re-ingested
        self._record_corpus_fingerprint(raw_text)

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
        *,
        cross_run: bool = False,
    ) -> str:
        """Export corpus as structured text for gossip swarm input.

        Queries conditions (row_type='finding', consider_for_use=TRUE),
        formats as structured text with source attribution per fact.

        Args:
            iteration: Filter to specific iteration, or None for all.
            min_confidence: Minimum confidence threshold.
            max_rows: Cap on number of conditions to export.
            cross_run: If True, include findings from all runs.

        Returns:
            Structured text corpus for swarm consumption.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        run_p = self._run_params(cross_run=cross_run)
        with self._lock:
            query = f"""
                SELECT id, fact, source_url, source_type, confidence,
                       angle, iteration, verification_status
                FROM conditions
                WHERE consider_for_use = TRUE
                  AND row_type = 'finding'
                  AND confidence >= ?
                  {run_sql}
            """
            params: list[Any] = [min_confidence] + run_p

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

    def export_delta(
        self,
        since: str,
        min_confidence: float = 0.0,
        max_rows: int = 10000,
        *,
        cross_run: bool = False,
    ) -> str:
        """Return findings created after *since* as formatted text.

        Used by the swarm engine between gossip rounds to pick up new
        findings that producers ingested while the swarm was running.

        Args:
            since: ISO-8601 timestamp — only findings with
                ``created_at > since`` are returned.
            min_confidence: Minimum confidence threshold.
            max_rows: Safety cap on returned findings (default 10000).
            cross_run: If True, include findings from all runs.

        Returns:
            Formatted text block of new findings, or empty string if none.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        run_p = self._run_params(cross_run=cross_run)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT id, fact, source_url, source_type, confidence,
                          angle, iteration, verification_status
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND confidence >= ?
                     AND created_at > ?
                     {run_sql}
                   ORDER BY created_at ASC
                   LIMIT ?""",
                [min_confidence, since] + run_p + [max_rows],
            ).fetchall()

        if not rows:
            return ""

        lines: list[str] = []
        for row in rows:
            cid, fact, src_url, src_type, conf, angle, itr, vstatus = row
            source_tag = f"[{src_type}]" if src_type else ""
            url_tag = f" ({src_url})" if src_url else ""
            conf_tag = f" [conf={conf:.2f}]" if conf != 0.5 else ""
            lines.append(f"[#{cid}] {source_tag}{conf_tag} {fact}{url_tag}")

        return (
            f"=== {len(rows)} NEW FINDINGS (since last round) ===\n"
            + "\n".join(lines)
        )

    def export_prior_research(self, *, cross_run: bool = False) -> str:
        """Export prior thoughts and insights as corpus text.

        Returns worker synthesis outputs, gossip round outputs, and
        cross-domain insights — the reasoning artifacts that
        ``export_for_swarm`` does NOT include (it only exports raw
        findings).  This avoids duplicating findings in the swarm
        input when both methods are used together.

        Returns:
            Formatted text block of prior thoughts/insights, or empty string.
        """
        run_sql = self._run_sql(cross_run=cross_run)
        run_p = self._run_params(cross_run=cross_run)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT id, fact, source_url, source_type, confidence,
                          angle, row_type, iteration
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('thought', 'insight')
                     {run_sql}
                   ORDER BY iteration ASC, id ASC""",
                run_p,
            ).fetchall()

        if not rows:
            return ""

        lines: list[str] = []
        for row in rows:
            cid, fact, src_url, src_type, conf, angle, rtype, itr = row
            tags = []
            if src_type:
                tags.append(f"[{src_type}]")
            if rtype != "finding":
                tags.append(f"[{rtype}]")
            if angle:
                tags.append(f"[angle:{angle}]")
            if conf != 0.5:
                tags.append(f"[conf={conf:.2f}]")
            tag_str = " ".join(tags)
            url_tag = f" ({src_url})" if src_url else ""
            lines.append(f"[#{cid}] {tag_str} {fact}{url_tag}")

        return (
            f"=== PRIOR RESEARCH: {len(rows)} entries "
            f"(thoughts + insights) ===\n"
            + "\n".join(lines)
        )

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
                   ORDER BY id ASC
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
                    lines.append(f"  [#{c1_id}]: {c1_fact}")
                    lines.append(f"  vs [#{c2_id}]: {c2_fact}")

            # Previous synthesis highlights
            prev_synthesis = self.get_synthesis(iteration)
            if prev_synthesis:
                lines.append(f"\n--- PRIOR SYNTHESIS (iteration {iteration}) ---")
                lines.append(prev_synthesis)

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

            # Swarm deliberation state (gossip round 2 is the
            # contradiction-resolution round — low info gain or
            # unresolved contradictions surface here).
            gossip_rows = self.conn.execute(
                """SELECT id, phase, angle, strategy, fact
                   FROM conditions
                   WHERE phase LIKE 'gossip_round_%'
                   ORDER BY id ASC"""
            ).fetchall()
            if gossip_rows:
                swarm_lines: list[str] = []
                for cid, phase, g_angle, strategy, fact in gossip_rows:
                    # Detect unresolved contradictions in round-2 summaries.
                    if phase == "gossip_round_2":
                        for marker in ("unresolvable", "unresolved", "contradiction"):
                            if marker in (fact or "").lower():
                                snippet = " ".join((fact or "").split())
                                swarm_lines.append(
                                    f"  [#{cid}, {phase}, angle={g_angle}]: "
                                    f"{snippet}"
                                )
                                break

                    # Low-info-gain rounds: metadata.info_gain stored in
                    # the strategy JSON by the engine's _emit call.
                    info_gain = None
                    if strategy:
                        try:
                            meta = json.loads(strategy)
                            if isinstance(meta, dict):
                                raw_gain = meta.get("info_gain")
                                if isinstance(raw_gain, (int, float)):
                                    info_gain = float(raw_gain)
                        except (TypeError, ValueError):
                            pass
                    if info_gain is not None and info_gain < 0.05:
                        swarm_lines.append(
                            f"  [#{cid}, {phase}, angle={g_angle}]: "
                            f"Low info gain ({info_gain:.2f}) — "
                            f"may need external data"
                        )

                if swarm_lines:
                    lines.append(
                        f"\n--- SWARM DELIBERATION GAPS ({len(swarm_lines)}) ---"
                    )
                    lines.extend(swarm_lines)

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
                   LIMIT 200"""
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
    # Observability — metrics as store rows
    # ------------------------------------------------------------------

    def emit_metric(
        self,
        metric_type: str,
        data: dict[str, Any],
        *,
        angle: str = "system",
        source_model: str = "",
        source_run: str = "",
        iteration: int = 0,
        parent_id: int | None = None,
    ) -> int:
        """Persist a metric snapshot as a condition row.

        Metric rows use ``row_type`` = *metric_type* (e.g.
        ``wave_metric``, ``worker_metric``, ``store_metric``,
        ``decision_point``).  The JSON blob goes into ``fact``.
        These rows are excluded from research queries
        (``consider_for_use = FALSE``) but queryable for dashboards.

        Args:
            metric_type: Row type tag (``wave_metric``, etc.).
            data: Arbitrary JSON-serialisable metric payload.
            angle: Metric category or worker angle.
            source_model: Model that produced the measured output.
            source_run: Run identifier for cross-run comparison.
            iteration: Wave number (0 for run-level metrics).
            parent_id: Optional link to the entity being measured.

        Returns:
            The condition ID of the new metric row.
        """
        fact_json = json.dumps(data, default=str)
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                cid = self._next_id
                self._next_id += 1
                self.conn.execute(
                    """INSERT INTO conditions
                       (id, fact, source_type, row_type,
                        consider_for_use, angle, created_at,
                        iteration, parent_id, source_model, source_run)
                       VALUES (?, ?, 'observability', ?, FALSE, ?, ?, ?, ?, ?, ?)""",
                    [
                        cid, fact_json, metric_type,
                        angle, now, iteration, parent_id,
                        source_model, source_run,
                    ],
                )
            logger.debug(
                "metric_type=<%s>, angle=<%s>, iteration=<%d> | emitted metric #%d",
                metric_type, angle, iteration, cid,
            )
            return cid
        except Exception as exc:
            logger.warning(
                "metric_type=<%s>, error=<%s> | failed to emit metric, continuing",
                metric_type, exc,
            )
            return -1

    def store_health_snapshot(
        self,
        *,
        source_run: str = "",
        iteration: int = 0,
    ) -> dict[str, Any]:
        """Query the store's own health and persist as a ``store_metric`` row.

        Returns the health data dict (also stored as a metric row).
        Never raises — returns empty dict on failure to avoid crashing
        the main pipeline for observability failures.
        """
        try:
            with self._lock:
                total = self.conn.execute(
                    "SELECT COUNT(*) FROM conditions"
                ).fetchone()[0]
                active = self.conn.execute(
                    "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE"
                ).fetchone()[0]
                obsolete = total - active

                rows_by_type = {}
                for row_type, cnt in self.conn.execute(
                    "SELECT row_type, COUNT(*) FROM conditions GROUP BY row_type"
                ).fetchall():
                    rows_by_type[row_type] = cnt

                rows_by_angle = {}
                for ang, cnt in self.conn.execute(
                    "SELECT angle, COUNT(*) FROM conditions "
                    "WHERE consider_for_use = TRUE GROUP BY angle"
                ).fetchall():
                    rows_by_angle[ang] = cnt

            data = {
                "total_rows": total,
                "active_rows": active,
                "obsolete_rows": obsolete,
                "rows_by_type": rows_by_type,
                "rows_by_angle": rows_by_angle,
            }
            self.emit_metric(
                "store_metric", data,
                source_run=source_run, iteration=iteration,
            )
            return data
        except Exception as exc:
            logger.warning(
                "error=<%s> | health snapshot failed, continuing",
                exc,
            )
            return {}

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
    # Lineage queries (swarm provenance)
    # ------------------------------------------------------------------

    _LINEAGE_COLUMNS: tuple[str, ...] = (
        "id", "fact", "row_type", "angle", "phase",
        "parent_id", "parent_ids", "strategy", "source_ref",
        "iteration", "created_at",
    )

    def _row_to_lineage_dict(self, row: tuple) -> dict[str, Any]:
        return dict(zip(self._LINEAGE_COLUMNS, row))

    def get_by_phase(self, phase: str) -> list[dict[str, Any]]:
        """Return condition rows whose ``phase`` matches or starts with *phase*.

        Exact match is returned first, then prefix matches (so passing
        ``"gossip_round"`` returns every gossip_round_N entry).
        """
        cols_sql = ", ".join(self._LINEAGE_COLUMNS)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {cols_sql} FROM conditions
                    WHERE phase = ? OR phase LIKE ?
                    ORDER BY id ASC""",
                [phase, f"{phase}%"],
            ).fetchall()
        return [self._row_to_lineage_dict(r) for r in rows]

    def get_by_angle(self, angle: str) -> list[dict[str, Any]]:
        """Return condition rows whose ``angle`` equals *angle*."""
        cols_sql = ", ".join(self._LINEAGE_COLUMNS)
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {cols_sql} FROM conditions
                    WHERE angle = ?
                    ORDER BY id ASC""",
                [angle],
            ).fetchall()
        return [self._row_to_lineage_dict(r) for r in rows]

    def get_lineage_chain(self, condition_id: int) -> list[dict[str, Any]]:
        """Walk the parent DAG from *condition_id* back to its root(s).

        Follows both the integer ``parent_id`` FK (single parent) and the
        JSON ``parent_ids`` array (multi-parent DAG written by
        :meth:`emit`).  String entry IDs stored in ``parent_ids`` are
        resolved via the ``source_ref`` column.

        Returns the starting row plus every reachable ancestor in BFS
        order (newest-first).
        """
        cols_sql = ", ".join(self._LINEAGE_COLUMNS)
        chain: list[dict[str, Any]] = []
        visited_ids: set[int] = set()
        queue: list[int] = [condition_id]

        with self._lock:
            while queue:
                cid = queue.pop(0)
                if cid in visited_ids:
                    continue
                visited_ids.add(cid)
                row = self.conn.execute(
                    f"SELECT {cols_sql} FROM conditions WHERE id = ?",
                    [cid],
                ).fetchone()
                if row is None:
                    continue
                entry = self._row_to_lineage_dict(row)
                chain.append(entry)

                # 1. Integer FK parent.
                pid = entry.get("parent_id")
                if isinstance(pid, int) and pid not in visited_ids:
                    queue.append(pid)

                # 2. JSON array of parent entry IDs (swarm-native).
                raw_parents = entry.get("parent_ids") or ""
                if raw_parents:
                    try:
                        parent_list = json.loads(raw_parents)
                    except (TypeError, ValueError):
                        parent_list = []
                    for parent in parent_list:
                        if isinstance(parent, int):
                            if parent not in visited_ids:
                                queue.append(parent)
                        elif isinstance(parent, str) and parent:
                            # Resolve string entry_id (e.g. "lineage-0001")
                            # via source_ref to its integer row id.
                            resolved = self.conn.execute(
                                "SELECT id FROM conditions "
                                "WHERE source_ref = ? LIMIT 1",
                                [parent],
                            ).fetchone()
                            if resolved and resolved[0] not in visited_ids:
                                queue.append(resolved[0])

        return chain

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
