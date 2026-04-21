# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""DuckDB-backed persistent knowledge store for cross-session learning.

Stores facts, insights, and entities extracted from research conversations
so new queries can leverage accumulated knowledge instead of starting from
scratch every time.

Uses DuckDB Full Text Search (FTS) for BM25-ranked semantic retrieval
without requiring external embedding models or API calls.

Location: ~/.mirothinker/knowledge/knowledge.db

Architecture:
    - ``insights`` table: factual claims with provenance, topic tags,
      confidence scores, and access counts
    - ``entities`` table: named entities (people, compounds, orgs) with
      cross-reference counts for disambiguation
    - FTS index on insights.fact for BM25 retrieval
    - Thread-safe via RLock (same pattern as ConditionStore)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

KNOWLEDGE_DIR = Path(
    os.environ.get(
        "MIROTHINKER_KNOWLEDGE_DIR",
        os.path.expanduser("~/.mirothinker/knowledge"),
    )
)
DB_PATH = KNOWLEDGE_DIR / "knowledge.db"


# ── Data types ────────────────────────────────────────────────────────


@dataclass
class Insight:
    """A single knowledge entry extracted from a research conversation."""

    fact: str
    source_url: str = ""
    source_type: str = ""  # "research", "forum", "academic", "government", etc.
    topic: str = ""
    confidence: float = 0.7
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    access_count: int = 0
    last_accessed: str = ""
    query_context: str = ""  # the user query that produced this insight


@dataclass
class Entity:
    """A named entity seen across multiple conversations."""

    name: str
    entity_type: str = ""  # "person", "compound", "organization", "concept"
    description: str = ""
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    mention_count: int = 1


# ── KnowledgeStore ────────────────────────────────────────────────────


class KnowledgeStore:
    """Persistent knowledge store backed by DuckDB with FTS.

    Thread-safe via RLock. The store is a singleton — all agents and
    plugins share the same instance so knowledge accumulates globally.

    The FTS index uses DuckDB's built-in ``fts`` extension with BM25
    scoring for keyword-based retrieval. No external embedding model
    or API calls are needed.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the knowledge store.

        Args:
            db_path: Path to the DuckDB file. Defaults to
                ``~/.mirothinker/knowledge/knowledge.db``.
                Pass ``:memory:`` for in-memory testing.
        """
        if db_path is None:
            db_path = DB_PATH
        self._db_path = str(db_path)
        self._lock = threading.RLock()

        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = duckdb.connect(self._db_path)
        self._setup_tables()
        self._setup_fts()

        count = self.count_insights()
        entity_count = self.count_entities()
        logger.info(
            "insights=<%d>, entities=<%d> | knowledge store opened at %s",
            count,
            entity_count,
            self._db_path,
        )

    def _setup_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY,
                    fact TEXT NOT NULL,
                    source_url TEXT DEFAULT '',
                    source_type TEXT DEFAULT '',
                    topic TEXT DEFAULT '',
                    confidence FLOAT DEFAULT 0.7,
                    created_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT DEFAULT '',
                    query_context TEXT DEFAULT ''
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT DEFAULT '',
                    description TEXT DEFAULT '',
                    first_seen TEXT NOT NULL,
                    mention_count INTEGER DEFAULT 1
                )
            """)
            # Ensure we have a sequence for IDs
            self._conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS insights_id_seq START 1
            """)
            self._conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS entities_id_seq START 1
            """)

    def _setup_fts(self) -> None:
        """Set up Full Text Search index on insights.fact."""
        with self._lock:
            try:
                self._conn.execute("INSTALL fts")
                self._conn.execute("LOAD fts")
            except duckdb.CatalogException:
                # Already installed/loaded
                pass

            # Recreate the FTS index (DuckDB FTS requires rebuild to pick
            # up new rows — we rebuild on search, not on insert)
            self._fts_stale = True

    def _rebuild_fts_if_needed(self) -> None:
        """Rebuild the FTS index if it's stale (new inserts since last build)."""
        if not self._fts_stale:
            return
        with self._lock:
            if not self._fts_stale:
                return
            try:
                # Drop and recreate the index
                self._conn.execute(
                    "PRAGMA drop_fts_index('insights')"
                )
            except Exception:
                pass  # index doesn't exist yet
            try:
                self._conn.execute("""
                    PRAGMA create_fts_index(
                        'insights', 'id', 'fact', 'topic', 'query_context',
                        stemmer='english',
                        stopwords='english'
                    )
                """)
                self._fts_stale = False
            except Exception:
                logger.debug("FTS index rebuild failed, falling back to LIKE", exc_info=True)

    # ── Write operations ──────────────────────────────────────────────

    def store_insight(self, insight: Insight) -> int:
        """Store a new insight and return its ID.

        Args:
            insight: The insight to store.

        Returns:
            The auto-generated ID of the stored insight.
        """
        with self._lock:
            result = self._conn.execute(
                """
                INSERT INTO insights (id, fact, source_url, source_type, topic,
                    confidence, created_at, access_count, last_accessed, query_context)
                VALUES (nextval('insights_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                [
                    insight.fact,
                    insight.source_url,
                    insight.source_type,
                    insight.topic,
                    insight.confidence,
                    insight.created_at,
                    insight.access_count,
                    insight.last_accessed,
                    insight.query_context,
                ],
            ).fetchone()
            self._fts_stale = True
            row_id = result[0] if result else -1

        logger.debug(
            "id=<%d>, topic=<%s> | insight stored",
            row_id,
            insight.topic,
        )
        return row_id

    def store_entity(self, entity: Entity) -> int:
        """Store or update a named entity.

        If an entity with the same name and type already exists, its
        mention_count is incremented instead of creating a duplicate.

        Args:
            entity: The entity to store or update.

        Returns:
            The ID of the stored/updated entity.
        """
        with self._lock:
            existing = self._conn.execute(
                "SELECT id, mention_count FROM entities WHERE name = ? AND entity_type = ?",
                [entity.name, entity.entity_type],
            ).fetchone()

            if existing:
                entity_id = existing[0]
                self._conn.execute(
                    "UPDATE entities SET mention_count = mention_count + 1, description = ? WHERE id = ?",
                    [entity.description or "", entity_id],
                )
                return entity_id

            result = self._conn.execute(
                """
                INSERT INTO entities (id, name, entity_type, description, first_seen, mention_count)
                VALUES (nextval('entities_id_seq'), ?, ?, ?, ?, ?)
                RETURNING id
                """,
                [
                    entity.name,
                    entity.entity_type,
                    entity.description,
                    entity.first_seen,
                    entity.mention_count,
                ],
            ).fetchone()
            return result[0] if result else -1

    # ── Read operations ───────────────────────────────────────────────

    def search_insights(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
        topic: str = "",
    ) -> list[dict[str, Any]]:
        """Search insights using FTS (BM25) with optional filters.

        Falls back to LIKE-based search if FTS index is unavailable.

        Args:
            query: Search query text.
            limit: Maximum results to return.
            min_confidence: Minimum confidence threshold.
            topic: Filter by topic (empty = all topics).

        Returns:
            List of insight dicts ordered by relevance.
        """
        self._rebuild_fts_if_needed()

        with self._lock:
            try:
                # Try FTS first
                sql = """
                    SELECT i.*, fts.score
                    FROM insights i
                    JOIN (
                        SELECT *, fts_main_insights.match_bm25(id, ?, fields := 'fact,topic,query_context') AS score
                        FROM insights
                    ) fts ON i.id = fts.id
                    WHERE fts.score IS NOT NULL
                """
                params: list[Any] = [query]

                if min_confidence > 0:
                    sql += " AND i.confidence >= ?"
                    params.append(min_confidence)
                if topic:
                    sql += " AND i.topic LIKE ?"
                    params.append(f"%{topic}%")

                sql += " ORDER BY fts.score DESC LIMIT ?"
                params.append(limit)

                rows = self._conn.execute(sql, params).fetchall()
                columns = [desc[0] for desc in self._conn.description]
            except Exception:
                # Fallback to LIKE search
                logger.debug("FTS search failed, falling back to LIKE")
                rows, columns = self._like_search(query, limit, min_confidence, topic)

            # Update access counts
            if rows:
                ids = [row[0] for row in rows]
                now = datetime.now(timezone.utc).isoformat()
                placeholders = ",".join(["?"] * len(ids))
                self._conn.execute(
                    f"UPDATE insights SET access_count = access_count + 1, last_accessed = ? WHERE id IN ({placeholders})",
                    [now] + ids,
                )

        return [dict(zip(columns, row)) for row in rows]

    def _like_search(
        self,
        query: str,
        limit: int,
        min_confidence: float,
        topic: str,
    ) -> tuple[list[tuple], list[str]]:
        """Fallback keyword search using LIKE."""
        words = query.lower().split()
        if not words:
            return [], []

        conditions = []
        params: list[Any] = []
        for word in words[:5]:  # limit to 5 keywords
            conditions.append("(LOWER(fact) LIKE ? OR LOWER(topic) LIKE ? OR LOWER(query_context) LIKE ?)")
            params.extend([f"%{word}%", f"%{word}%", f"%{word}%"])

        sql = f"SELECT *, 0.0 AS score FROM insights WHERE ({' OR '.join(conditions)})"

        if min_confidence > 0:
            sql += " AND confidence >= ?"
            params.append(min_confidence)
        if topic:
            sql += " AND topic LIKE ?"
            params.append(f"%{topic}%")

        sql += " ORDER BY confidence DESC, access_count DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in self._conn.description]
        return rows, columns

    def get_recent_insights(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most recently stored insights.

        Args:
            limit: Maximum results to return.

        Returns:
            List of insight dicts ordered by creation time (newest first).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM insights ORDER BY created_at DESC LIMIT ?",
                [limit],
            ).fetchall()
            columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in rows]

    def get_top_entities(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get the most frequently mentioned entities.

        Args:
            limit: Maximum results to return.

        Returns:
            List of entity dicts ordered by mention count.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
                [limit],
            ).fetchall()
            columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in rows]

    def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search entities by name.

        Args:
            query: Entity name to search for.
            limit: Maximum results.

        Returns:
            List of matching entity dicts.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE LOWER(name) LIKE ? ORDER BY mention_count DESC LIMIT ?",
                [f"%{query.lower()}%", limit],
            ).fetchall()
            columns = [desc[0] for desc in self._conn.description]
        return [dict(zip(columns, row)) for row in rows]

    # ── Stats ─────────────────────────────────────────────────────────

    def count_insights(self) -> int:
        """Return total number of stored insights."""
        with self._lock:
            result = self._conn.execute("SELECT COUNT(*) FROM insights").fetchone()
            return result[0] if result else 0

    def count_entities(self) -> int:
        """Return total number of stored entities."""
        with self._lock:
            result = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
            return result[0] if result else 0

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about accumulated knowledge."""
        with self._lock:
            insight_count = self._conn.execute("SELECT COUNT(*) FROM insights").fetchone()[0]
            entity_count = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

            top_topics = self._conn.execute(
                "SELECT topic, COUNT(*) as cnt FROM insights WHERE topic != '' GROUP BY topic ORDER BY cnt DESC LIMIT 10"
            ).fetchall()

            avg_confidence = self._conn.execute(
                "SELECT AVG(confidence) FROM insights"
            ).fetchone()[0]

            most_accessed = self._conn.execute(
                "SELECT fact, access_count FROM insights ORDER BY access_count DESC LIMIT 5"
            ).fetchall()

        return {
            "total_insights": insight_count,
            "total_entities": entity_count,
            "top_topics": [{"topic": t[0], "count": t[1]} for t in top_topics],
            "avg_confidence": round(avg_confidence, 3) if avg_confidence else 0.0,
            "most_accessed": [
                {"fact": m[0][:100], "access_count": m[1]} for m in most_accessed
            ],
        }

    # ── Deduplication ─────────────────────────────────────────────────

    def has_similar_insight(self, fact: str, threshold: float = 0.8) -> bool:
        """Check if a substantially similar insight already exists.

        Uses simple word overlap ratio as a lightweight dedup check.

        Args:
            fact: The fact text to check for duplicates.
            threshold: Minimum word overlap ratio (0.0-1.0).

        Returns:
            True if a similar insight already exists.
        """
        fact_words = set(fact.lower().split())
        if not fact_words:
            return False

        # Check recent insights (last 500) for overlap
        with self._lock:
            rows = self._conn.execute(
                "SELECT fact FROM insights ORDER BY id DESC LIMIT 500"
            ).fetchall()

        for (existing_fact,) in rows:
            existing_words = set(existing_fact.lower().split())
            if not existing_words:
                continue
            overlap = len(fact_words & existing_words)
            ratio = overlap / max(len(fact_words), len(existing_words))
            if ratio >= threshold:
                return True

        return False

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
        logger.info("knowledge store closed")


# ── Singleton ─────────────────────────────────────────────────────────

_store: KnowledgeStore | None = None
_store_lock = threading.Lock()


def get_knowledge_store(db_path: str | Path | None = None) -> KnowledgeStore:
    """Get or create the singleton KnowledgeStore.

    Thread-safe. The first call creates the store; subsequent calls
    return the same instance.

    Args:
        db_path: Override path for the DuckDB file. Only used on
            first call (when the singleton is created).

    Returns:
        The global KnowledgeStore instance.
    """
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        _store = KnowledgeStore(db_path)
        return _store


def reset_knowledge_store() -> None:
    """Close and reset the singleton (for testing)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
            _store = None
