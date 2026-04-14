# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
DuckDB-backed corpus store for AtomicConditions -- Flock edition.

Replaces brittle programmatic text processing with DuckDB's Flock community
extension -- LLM calls executed directly in SQL.  Flock is MANDATORY for
pipeline operation; the pipeline will refuse to start without it.

Key features:
  - Gradient-flag columns on every row (nothing is ever dropped, only scored)
  - Flock-powered scoring (confidence, trust, specificity, fabrication risk, etc.)
  - Gossip-based synthesis via Flock for large corpora
  - Persistent file-backed DuckDB so knowledge accumulates across sessions
  - Unified ingestion path (all data enters identically via ``ingest_raw``)
  - Consumer-specific query methods (thinker sees almost everything,
    synthesiser is stricter but still sees low-confidence with hedging)
"""

from __future__ import annotations

import json as _json
import logging
import os
import re
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Optional
import duckdb

from models.atomic_condition import AtomicCondition
from utils.flock_proxy import get_flock_proxy_url, start_flock_proxy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flock model configuration (env-driven)
# ---------------------------------------------------------------------------
def _strip_litellm_prefix(model_str: str) -> str:
    """Strip the leading ``litellm/`` prefix if present.

    ADK_MODEL is typically ``litellm/openai/zai-org-glm-5-1``.  LiteLLM
    itself needs ``openai/zai-org-glm-5-1`` (no ``litellm/`` wrapper).
    We keep the provider prefix (openai/, ollama/, etc.) so LiteLLM
    routes to the right backend.
    """
    if model_str.startswith("litellm/"):
        model_str = model_str[len("litellm/"):]
    return model_str


_FLOCK_MODEL = os.environ.get(
    "FLOCK_MODEL",
    _strip_litellm_prefix(os.environ.get("ADK_MODEL", "gpt-4o")),
)
_FLOCK_KEY = os.environ.get(
    "FLOCK_API_KEY", os.environ.get("OPENAI_API_KEY", "")
)
_FLOCK_BASE = os.environ.get(
    "FLOCK_API_BASE", os.environ.get("OPENAI_API_BASE", "")
)

# Gossip threshold: corpora above this many conditions use gossip synthesis
GOSSIP_THRESHOLD = int(os.environ.get("GOSSIP_THRESHOLD", "80"))

# Provider name for CREATE MODEL (default openai; override if needed)
_FLOCK_PROVIDER = os.environ.get("FLOCK_PROVIDER", "openai")

# Default corpus storage directory (file-backed for session persistence)
_CORPUS_DIR = os.path.join(
    os.environ.get("FINDINGS_DIR", os.path.join(os.path.expanduser("~"), ".mirothinker")),
    "corpora",
)

# Explicit override: set CORPUS_DB_PATH to a full file path to use that
# instead of the auto-generated session-scoped path.
_DB_PATH = os.environ.get("CORPUS_DB_PATH", "")

# ---------------------------------------------------------------------------
# Direct HTTP path config (bypasses DuckDB's 2000-char llm_complete limit)
# ---------------------------------------------------------------------------
_FLOCK_PARALLELISM = int(os.environ.get("FLOCK_PARALLELISM", "6"))
_FLOCK_HTTP_TIMEOUT = int(os.environ.get("FLOCK_HTTP_TIMEOUT", "60"))


class CorpusStore:
    """DuckDB + Flock store for AtomicConditions with gradient scoring."""

    def __init__(self, db_path: str = "") -> None:
        if db_path:
            db = db_path
        elif _DB_PATH:
            db = _DB_PATH
        else:
            # Auto-generate a timestamped file in the corpus directory
            os.makedirs(_CORPUS_DIR, exist_ok=True)
            db = os.path.join(
                _CORPUS_DIR,
                f"corpus_{int(time.time())}_{os.getpid()}_{id(self):x}.duckdb",
            )
        self.db_path = db
        logger.info("CorpusStore opening DuckDB at %s", db)
        self.conn = duckdb.connect(db)

        # -- Load Flock extension (MANDATORY) --
        self._flock_available = False
        try:
            self.conn.execute("INSTALL flock FROM community")
            self.conn.execute("LOAD flock")
            self._flock_available = True
        except Exception as exc:
            raise RuntimeError(
                "Flock DuckDB extension is required but failed to load. "
                "Flock is only available for DuckDB <1.5.0. Current "
                f"DuckDB version: {duckdb.__version__}. "
                "Pin duckdb>=1.4.0,<1.5.0 in pyproject.toml. "
                f"Original error: {exc}"
            ) from exc

        # -- Configure Flock secrets & model --
        if self._flock_available:
            self._setup_flock()

        self._setup_tables()
        self._next_id = self._compute_next_id()
        self._write_lock = threading.Lock()

        # -- Tracing context (set by caller before battery runs) --
        self._trace_session_id: str = ""
        self._trace_iteration: int = 0
        self._trace_enabled: bool = False

    # ------------------------------------------------------------------
    # Flock configuration
    # ------------------------------------------------------------------

    def _setup_flock(self) -> None:
        """Create the DuckDB SECRET and MODEL that Flock needs.

        Flock sends ``response_format: json_schema`` on every request,
        which many providers (Venice, Ollama, vLLM) reject.  We route
        Flock through a LiteLLM-backed proxy on localhost that:
        - strips ``response_format`` before calling the real provider
        - uses ``litellm.acompletion()`` for provider-agnostic routing
        - wraps plain-text responses in JSON for Flock's C++ parser

        The proxy accepts the model/key/base from env vars
        (``FLOCK_MODEL``, ``FLOCK_API_KEY``, ``FLOCK_API_BASE``).
        """
        def _sql_escape(val: str) -> str:
            return val.replace("'", "''")

        # Start the LiteLLM-backed Flock proxy.  It handles:
        # - response_format stripping
        # - provider-specific request/response translation via LiteLLM
        # - JSON wrapping for Flock compatibility
        proxy_url = start_flock_proxy(
            litellm_model=_FLOCK_MODEL,
            litellm_api_key=_FLOCK_KEY,
            litellm_api_base=_FLOCK_BASE,
        )
        logger.info(
            "Flock LiteLLM proxy started: %s  model=%s  base=%s",
            proxy_url, _FLOCK_MODEL,
            _FLOCK_BASE or "(provider default)",
        )

        # Point Flock's SECRET at our local proxy.  The proxy handles
        # real auth to the provider, so we use a dummy key here.
        # Flock's llm_complete looks for the unnamed default secret
        # (__default_openai), NOT a named secret.
        try:
            self.conn.execute(
                f"CREATE SECRET "
                f"(TYPE OPENAI, "
                f"API_KEY 'flock-proxy', "
                f"BASE_URL '{_sql_escape(proxy_url)}')"
            )
        except Exception:
            logger.warning(
                "Failed to CREATE SECRET -- Flock will be "
                "unavailable.",
                exc_info=True,
            )

        # CREATE MODEL — the model name is just a label here; the proxy
        # overrides it with FLOCK_MODEL via LiteLLM.
        provider = _sql_escape(_FLOCK_PROVIDER)
        try:
            self.conn.execute("DROP MODEL IF EXISTS 'corpus_model'")
        except Exception:
            pass
        try:
            self.conn.execute(
                f"CREATE MODEL('corpus_model', 'flock-model', "
                f"'{provider}')"
            )
        except Exception:
            logger.warning(
                "Failed to CREATE MODEL -- Flock ingestion and "
                "scoring will be unavailable. Ensure the flock "
                "extension is installed.",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _setup_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conditions (
                id INTEGER PRIMARY KEY,
                fact TEXT NOT NULL,
                source_url TEXT DEFAULT '',
                source_type TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',

                -- Row type: 'finding' | 'similarity' | 'contradiction' | 'raw' | 'synthesis' | 'thought' | 'insight'
                row_type TEXT DEFAULT 'finding',

                -- Hierarchical relationships (parent-child in the SAME table)
                parent_id INTEGER,
                related_id INTEGER,

                -- THE universal exclusion flag
                consider_for_use BOOLEAN DEFAULT TRUE,
                obsolete_reason TEXT DEFAULT '',

                -- Core metadata
                angle TEXT DEFAULT '',
                strategy TEXT DEFAULT '',
                expansion_depth INTEGER DEFAULT 0,
                created_at TEXT DEFAULT '',
                iteration INTEGER DEFAULT 0,

                -- GRADIENT FLAGS (0.0-1.0, computed by Flock)
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

                -- Processing state (FSM state column)
                processing_status TEXT DEFAULT 'raw',

                -- Expansion system (MUST be closed-loop)
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

                -- For relationship rows: the score
                relationship_score FLOAT DEFAULT 0.0
            )
        """)

        # Migrate from v1 schema (satellite tables → single table)
        self._migrate_from_v1()

    def _migrate_from_v1(self) -> None:
        """Migrate from v1 schema (satellite tables) to single-table architecture.

        Checks if old satellite tables exist and migrates their data into
        the unified conditions table as relationship/raw rows, then drops
        the old tables.  Also converts expansion_strategy → expansion_tool.
        Safe to call repeatedly (idempotent).
        """
        # Check if any old tables exist
        old_tables = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name IN ('similarity_flags', 'contradiction_pairs', 'raw_ingestion')"
        ).fetchall()
        old_table_names = {r[0] for r in old_tables}

        if not old_table_names:
            # Also handle the expansion_strategy → expansion_tool rename
            # for databases that were created with v2 schema but still
            # have expansion_strategy column
            try:
                cols = self.conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'conditions'"
                ).fetchall()
                col_names = {r[0] for r in cols}
                if "expansion_strategy" in col_names and "expansion_tool" not in col_names:
                    self.conn.execute(
                        "ALTER TABLE conditions RENAME COLUMN expansion_strategy TO expansion_tool"
                    )
                    logger.info("Renamed expansion_strategy → expansion_tool")
            except Exception:
                pass
            return

        logger.info(
            "v1 migration: found old tables %s — migrating to single-table",
            old_table_names,
        )

        # Ensure new columns exist (ALTER TABLE ADD IF NOT EXISTS)
        _new_cols = {
            "row_type": "TEXT DEFAULT 'finding'",
            "related_id": "INTEGER",
            "consider_for_use": "BOOLEAN DEFAULT TRUE",
            "obsolete_reason": "TEXT DEFAULT ''",
            "expansion_tool": "TEXT DEFAULT 'none'",
            "expansion_fulfilled": "BOOLEAN DEFAULT FALSE",
            "relationship_score": "FLOAT DEFAULT 0.0",
        }
        existing_cols = {
            r[0] for r in self.conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'conditions'"
            ).fetchall()
        }
        for col, typedef in _new_cols.items():
            if col not in existing_cols:
                try:
                    self.conn.execute(
                        f"ALTER TABLE conditions ADD COLUMN {col} {typedef}"
                    )
                except Exception:
                    pass  # column may already exist

        # 1. Migrate similarity_flags → relationship rows
        if "similarity_flags" in old_table_names:
            try:
                sim_rows = self.conn.execute(
                    "SELECT condition_a, condition_b, similarity_score, relationship "
                    "FROM similarity_flags"
                ).fetchall()
                for cond_a, cond_b, sim_score, rel_text in sim_rows:
                    cid = self.conn.execute(
                        "SELECT COALESCE(MAX(id), 0) + 1 FROM conditions"
                    ).fetchone()[0]
                    self.conn.execute(
                        "INSERT INTO conditions "
                        "(id, fact, row_type, parent_id, related_id, "
                        "relationship_score, consider_for_use) "
                        "VALUES (?, ?, 'similarity', ?, ?, ?, FALSE)",
                        [cid, rel_text or "similarity", cond_a, cond_b, sim_score],
                    )
                logger.info("Migrated %d similarity_flags rows", len(sim_rows))
            except Exception:
                logger.warning("Failed to migrate similarity_flags", exc_info=True)

        # 2. Migrate contradiction_pairs → relationship rows
        if "contradiction_pairs" in old_table_names:
            try:
                contra_rows = self.conn.execute(
                    "SELECT condition_a, condition_b, contradiction_score, description "
                    "FROM contradiction_pairs"
                ).fetchall()
                for cond_a, cond_b, contra_score, desc in contra_rows:
                    cid = self.conn.execute(
                        "SELECT COALESCE(MAX(id), 0) + 1 FROM conditions"
                    ).fetchone()[0]
                    self.conn.execute(
                        "INSERT INTO conditions "
                        "(id, fact, row_type, parent_id, related_id, "
                        "relationship_score, consider_for_use) "
                        "VALUES (?, ?, 'contradiction', ?, ?, ?, FALSE)",
                        [cid, desc or "contradiction", cond_a, cond_b, contra_score],
                    )
                logger.info("Migrated %d contradiction_pairs rows", len(contra_rows))
            except Exception:
                logger.warning("Failed to migrate contradiction_pairs", exc_info=True)

        # 3. Migrate raw_ingestion → raw rows
        if "raw_ingestion" in old_table_names:
            try:
                raw_rows = self.conn.execute(
                    "SELECT id, raw_text, source_type, source_ref, ingested_at, atomised "
                    "FROM raw_ingestion"
                ).fetchall()
                for raw_id, raw_text, src_type, src_ref, ingested_at, atomised in raw_rows:
                    cid = self.conn.execute(
                        "SELECT COALESCE(MAX(id), 0) + 1 FROM conditions"
                    ).fetchone()[0]
                    self.conn.execute(
                        "INSERT INTO conditions "
                        "(id, fact, source_type, source_ref, row_type, "
                        "consider_for_use, created_at) "
                        "VALUES (?, ?, ?, ?, 'raw', ?, ?)",
                            [cid, raw_text, src_type, src_ref,
                             False, ingested_at or ""],
                    )
                logger.info("Migrated %d raw_ingestion rows", len(raw_rows))
            except Exception:
                logger.warning("Failed to migrate raw_ingestion", exc_info=True)

        # 4. Convert expansion_strategy → expansion_tool on existing conditions
        if "expansion_strategy" in existing_cols:
            try:
                if "expansion_tool" not in existing_cols:
                    self.conn.execute(
                        "ALTER TABLE conditions ADD COLUMN expansion_tool TEXT DEFAULT 'none'"
                    )
                self.conn.execute(
                    "UPDATE conditions SET expansion_tool = expansion_strategy "
                    "WHERE expansion_strategy IS NOT NULL AND expansion_strategy != ''"
                )
                logger.info("Converted expansion_strategy → expansion_tool")
            except Exception:
                logger.warning("Failed to convert expansion_strategy", exc_info=True)

        # 5. Set consider_for_use=FALSE on merged conditions
        try:
            self.conn.execute(
                "UPDATE conditions SET consider_for_use = FALSE, "
                "obsolete_reason = 'merged (v1 migration)' "
                "WHERE processing_status = 'merged'"
            )
        except Exception:
            pass

        # 6. DROP old tables
        for tbl in ("similarity_flags", "contradiction_pairs", "raw_ingestion"):
            if tbl in old_table_names:
                try:
                    self.conn.execute(f"DROP TABLE IF EXISTS {tbl}")
                    logger.info("Dropped old table: %s", tbl)
                except Exception:
                    logger.warning("Failed to drop %s", tbl, exc_info=True)

        logger.info("v1 migration complete")

    # ------------------------------------------------------------------
    # Expansion fulfillment
    # ------------------------------------------------------------------

    def _check_expansion_fulfillment(self, parent_id: int | None) -> None:
        """Mark a parent condition's expansion as fulfilled when a child arrives."""
        if parent_id is None:
            return
        self.conn.execute(
            "UPDATE conditions SET expansion_fulfilled = TRUE "
            "WHERE id = ? AND expansion_tool != 'none' AND expansion_fulfilled = FALSE",
            [parent_id],
        )

    def _compute_next_id(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM conditions"
        ).fetchone()
        return row[0] if row else 1

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def admit(self, condition: AtomicCondition) -> Optional[int]:
        """Insert a condition. Always succeeds unless fact is empty string.
        Scoring/dedup happens via Flock in score_new_conditions()."""
        fact = condition.fact.strip()
        if not fact:
            return None

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
                cid,
                fact,
                condition.source_url,
                getattr(condition, "source_type", "researcher"),
                getattr(condition, "source_ref", ""),
                getattr(condition, "row_type", "finding"),
                getattr(condition, "related_id", None),
                getattr(condition, "consider_for_use", True),
                condition.confidence,
                getattr(condition, "verification_status", ""),
                condition.angle,
                condition.parent_id,
                condition.strategy,
                condition.expansion_depth,
                condition.created_at,
                getattr(condition, "iteration", 0),
            ],
        )
        # Close the expansion loop: mark parent as fulfilled
        self._check_expansion_fulfillment(condition.parent_id)
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

    def admit_thought(
        self,
        reasoning: str,
        parent_thought_id: int | None = None,
        angle: str = "",
        strategy: str = "",
        iteration: int = 0,
        expansion_depth: int = 0,
    ) -> int:
        """Persist a full reasoning trace as a ``row_type='thought'`` row.

        Thought rows form parent-child lineage chains via ``parent_id``.
        They are IMMUTABLE — the maestro must never UPDATE, DELETE, or
        set ``consider_for_use=FALSE`` on them.

        Thread-safe: uses ``_write_lock`` so parallel specialist thinkers
        can call this concurrently without ID collisions.

        Returns the new row's ID.
        """
        fact = reasoning.strip()
        if not fact:
            raise ValueError("Cannot admit empty thought")

        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            cid = self._next_id
            self._next_id += 1
            self.conn.execute(
                """INSERT INTO conditions
                   (id, fact, row_type, parent_id, angle, strategy,
                    iteration, expansion_depth, consider_for_use, created_at)
                   VALUES (?, ?, 'thought', ?, ?, ?, ?, ?, TRUE, ?)""",
                [cid, fact, parent_thought_id, angle, strategy,
                 iteration, expansion_depth, now],
            )
        logger.debug("Admitted thought #%d: %d chars (parent=%s)", cid, len(fact), parent_thought_id)
        return cid

    def get_latest_thought(self, angle: str | None = None) -> dict | None:
        """Return the most recent ``row_type='thought'`` row, or None.

        Optionally filtered by *angle* (e.g. a specific agent name).
        """
        if angle is not None:
            rows = self.conn.execute(
                """SELECT * FROM conditions
                   WHERE row_type = 'thought' AND angle = ?
                   ORDER BY id DESC LIMIT 1""",
                [angle],
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM conditions
                   WHERE row_type = 'thought'
                   ORDER BY id DESC LIMIT 1""",
            ).fetchall()
        if not rows:
            return None
        cols = [d[0] for d in self.conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'conditions' ORDER BY ordinal_position"
        ).fetchall()]
        return dict(zip(cols, rows[0]))

    def admit_insight(
        self,
        conclusion: str,
        source_thought_id: int | None,
        angle: str = "",
        grounding_ids: list[int] | None = None,
        iteration: int = 0,
    ) -> int:
        """Materialize an evidence-grounded insight as ``row_type='insight'``.

        Insight rows bridge the epistemic gap between internal reasoning
        (thoughts) and the synthesiser's evidence base.  Unlike thoughts,
        insights are:
        - Visible to the synthesiser (they are evidence-grounded conclusions)
        - Grounded in specific finding-row IDs (stored in ``strategy`` field)
        - Linked to their source thought via ``parent_id``

        Thread-safe: uses ``_write_lock``.

        Args:
            conclusion: The materialized conclusion text.
            source_thought_id: The arbitration verdict thought that produced
                this insight.
            angle: Research angle this insight belongs to.
            grounding_ids: List of finding-row IDs that support this insight.
            iteration: Pipeline iteration number.

        Returns the new row's ID.
        """
        fact = conclusion.strip()
        if not fact:
            raise ValueError("Cannot admit empty insight")

        grounding_str = ",".join(str(i) for i in (grounding_ids or []))
        strategy = f"grounded_insight|findings:{grounding_str}"
        now = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            cid = self._next_id
            self._next_id += 1
            self.conn.execute(
                """INSERT INTO conditions
                   (id, fact, row_type, parent_id, angle, strategy,
                    iteration, expansion_depth, consider_for_use, created_at)
                   VALUES (?, ?, 'insight', ?, ?, ?, ?, 0, TRUE, ?)""",
                [cid, fact, source_thought_id, angle, strategy,
                 iteration, now],
            )
        logger.debug(
            "Admitted insight #%d from thought #%s (grounded in %d findings)",
            cid, source_thought_id if source_thought_id is not None else "N/A",
            len(grounding_ids or []),
        )
        return cid

    def get_thoughts_by_angle(self, angle: str) -> list[dict]:
        """Return all thought rows for a given *angle*, newest first."""
        rows = self.conn.execute(
            """SELECT id, fact, angle, strategy, iteration,
                      expansion_depth, parent_id, created_at
               FROM conditions
               WHERE row_type = 'thought' AND angle = ?
               ORDER BY id DESC""",
            [angle],
        ).fetchall()
        cols = [
            "id", "fact", "angle", "strategy", "iteration",
            "expansion_depth", "parent_id", "created_at",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def get_thought_chain(self, thought_id: int) -> list[dict]:
        """Walk the parent chain from *thought_id* up to the root.

        Returns a list ordered root-first (oldest ancestor first).
        """
        chain: list[dict] = []
        seen: set[int] = set()
        current_id: int | None = thought_id
        while current_id is not None and current_id not in seen:
            seen.add(current_id)
            row = self.conn.execute(
                """SELECT id, fact, angle, strategy, iteration,
                          expansion_depth, parent_id, created_at
                   FROM conditions
                   WHERE id = ? AND row_type = 'thought'""",
                [current_id],
            ).fetchone()
            if row is None:
                break
            cols = [
                "id", "fact", "angle", "strategy", "iteration",
                "expansion_depth", "parent_id", "created_at",
            ]
            d = dict(zip(cols, row))
            chain.append(d)
            current_id = d["parent_id"]
        chain.reverse()
        return chain

    def get_thought_children(self, parent_id: int) -> list[dict]:
        """Return all direct child thoughts of a given thought."""
        rows = self.conn.execute(
            """SELECT id, fact, angle, strategy, iteration,
                      expansion_depth, parent_id, created_at
               FROM conditions
               WHERE row_type = 'thought' AND parent_id = ?
               ORDER BY id ASC""",
            [parent_id],
        ).fetchall()
        cols = [
            "id", "fact", "angle", "strategy", "iteration",
            "expansion_depth", "parent_id", "created_at",
        ]
        return [dict(zip(cols, r)) for r in rows]

    def count_thoughts(self, angle: str | None = None) -> int:
        """Count thought rows, optionally filtered by angle."""
        if angle is not None:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE row_type = 'thought' AND angle = ?",
                [angle],
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM conditions WHERE row_type = 'thought'"
            ).fetchone()
        return row[0] if row else 0

    def get_distinct_thought_angles(self) -> list[str]:
        """Return all distinct angle values from thought rows."""
        rows = self.conn.execute(
            "SELECT DISTINCT angle FROM conditions "
            "WHERE row_type = 'thought' AND angle != '' "
            "ORDER BY angle"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Counting helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of conditions in the corpus."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM conditions"
        ).fetchone()[0]

    def count_by_status(self) -> dict[str, int]:
        """Count conditions grouped by verification_status."""
        rows = self.conn.execute(
            "SELECT verification_status, COUNT(*) "
            "FROM conditions GROUP BY verification_status"
        ).fetchall()
        return {row[0] or "(unset)": row[1] for row in rows}

    # ------------------------------------------------------------------
    # Flock-powered scoring
    # ------------------------------------------------------------------

    def set_trace_context(
        self, session_id: str, iteration: int,
    ) -> None:
        """Set the tracing context for subsequent operations.

        Called by the condition_manager before scoring/battery runs so
        that all Flock LLM calls and algorithm traces are tagged with
        the correct session and iteration.
        """
        self._trace_session_id = session_id
        self._trace_iteration = iteration
        self._trace_enabled = bool(session_id)

    def _flock_complete(
        self, prompt: str, caller: str = "",
    ) -> str:
        """Run a single Flock llm_complete call and return text.

        When tracing is enabled, captures the prompt, response, and
        duration to the dashboard SQLite for after-run analysis.

        Also emits dashboard events (llm_start / llm_end) so the
        live dashboard shows Flock battery progress in real time.
        """
        # Emit llm_start to the live dashboard
        from dashboard import get_any_active_collector
        collector = get_any_active_collector()
        flock_agent = f"flock/{caller}" if caller else "flock"
        try:
            if collector:
                prompt_tokens_est = len(prompt) // 3
                collector.llm_start(flock_agent, prompt_tokens_est)
        except Exception:
            pass  # never block pipeline for dashboard

        t0 = time.monotonic()
        row = self.conn.execute(
            "SELECT llm_complete("
            "{'model_name': 'corpus_model'}, "
            "{'prompt': ?})",
            [prompt],
        ).fetchone()
        result = str(row[0]) if row and row[0] is not None else ""
        duration_ms = (time.monotonic() - t0) * 1000

        # Emit llm_end to the live dashboard
        try:
            if collector:
                completion_tokens_est = len(result) // 3
                collector.llm_end(
                    flock_agent, duration_ms / 1000, completion_tokens_est,
                )
        except Exception:
            pass  # never block pipeline for dashboard

        if self._trace_enabled and caller:
            try:
                from dashboard import event_store
                event_store.insert_llm_trace(
                    session_id=self._trace_session_id,
                    iteration=self._trace_iteration,
                    caller=caller,
                    prompt=prompt[:4000],
                    response=result[:2000],
                    model=_FLOCK_MODEL,
                    duration_ms=duration_ms,
                )
            except Exception:
                pass  # never block pipeline for tracing

        return result

    # ------------------------------------------------------------------
    # Direct HTTP path — bypasses DuckDB's 2000-char llm_complete limit
    # ------------------------------------------------------------------

    def _http_complete(
        self, prompt: str, caller: str = "", max_tokens: int = 4096,
        *, _no_duckdb_fallback: bool = False,
    ) -> str:
        """Direct HTTP call to the Flock proxy, bypassing DuckDB's limit.

        Used for long-form generation (atomisation, synthesis, merges)
        where output may exceed DuckDB Flock's ~2000-char truncation.
        Falls back to ``_flock_complete()`` if the proxy URL is unknown,
        unless *_no_duckdb_fallback* is True (used by batch callers to
        avoid concurrent DuckDB access from worker threads).
        """
        proxy_url = get_flock_proxy_url()
        if not proxy_url:
            if _no_duckdb_fallback:
                return ""
            return self._flock_complete(prompt, caller)

        t0 = time.monotonic()
        body = _json.dumps({
            "model": "flock-model",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }).encode()
        req = urllib.request.Request(
            f"{proxy_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_FLOCK_HTTP_TIMEOUT) as resp:
                data = _json.loads(resp.read())
            choices = data.get("choices", [])
            result = choices[0]["message"]["content"] if choices else ""
            # Unwrap Flock JSON wrapper if present
            try:
                parsed = _json.loads(result)
                if isinstance(parsed, dict) and "items" in parsed:
                    items = parsed["items"]
                    if isinstance(items, list) and items:
                        result = str(items[0])
            except (_json.JSONDecodeError, TypeError, KeyError):
                pass
        except Exception:
            if _no_duckdb_fallback:
                logger.warning(
                    "Direct HTTP to proxy failed for %s (batch context, "
                    "no DuckDB fallback)", caller, exc_info=True,
                )
                return ""
            logger.warning(
                "Direct HTTP to proxy failed for %s — falling back to "
                "DuckDB llm_complete", caller, exc_info=True,
            )
            return self._flock_complete(prompt, caller)

        duration_ms = (time.monotonic() - t0) * 1000

        if self._trace_enabled and caller:
            try:
                from dashboard import event_store
                event_store.insert_llm_trace(
                    session_id=self._trace_session_id,
                    iteration=self._trace_iteration,
                    caller=caller,
                    prompt=prompt[:4000],
                    response=result[:4000],
                    model=_FLOCK_MODEL,
                    duration_ms=duration_ms,
                )
            except Exception:
                pass

        return result

    def _http_complete_batch(
        self,
        prompts: list[tuple[str, str]],
        max_tokens: int = 256,
    ) -> list[str]:
        """Run multiple HTTP completions in parallel via ThreadPoolExecutor.

        Args:
            prompts: list of ``(prompt_text, caller_label)`` tuples.
            max_tokens: per-request max tokens (scoring needs ~10).

        Returns:
            List of response strings in the same order as *prompts*.
        """
        results: list[str] = [""] * len(prompts)
        workers = min(_FLOCK_PARALLELISM, len(prompts))

        def _call(idx: int, prompt: str, caller: str) -> tuple[int, str]:
            return idx, self._http_complete(
                prompt, caller, max_tokens, _no_duckdb_fallback=True,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_call, i, p, c): i
                for i, (p, c) in enumerate(prompts)
            }
            for fut in as_completed(futures):
                try:
                    idx, text = fut.result()
                    results[idx] = text
                except Exception:
                    logger.warning(
                        "Batch HTTP completion failed for index %d",
                        futures[fut], exc_info=True,
                    )

        return results

    # ------------------------------------------------------------------
    # Junk atom filter — rejects LLM meta-commentary before admission
    # ------------------------------------------------------------------

    _JUNK_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"(?i)^here\s+are\s+the\s+(atomic\s+)?facts?"),
        re.compile(r"(?i)should\s+be\s+prioriti[sz]ed\s+in\s+round"),
        re.compile(r"(?i)^(based\s+on|according\s+to)\s+the\s+(text|passage|content|above|information)\s*(above|below|provided)?"),
        re.compile(r"(?i)^(note|disclaimer|caveat|warning)\s*:"),
        re.compile(r"(?i)^the\s+following\s+(are|is|lists?|contains?)"),
        re.compile(r"(?i)^(in\s+summary|to\s+summar|overall)\s*[,:]"),
        re.compile(r"(?i)^I\s+(have\s+)?(identified|extracted|found|listed)"),
        re.compile(r"(?i)^(here\s+is|below\s+is|the\s+above)"),
        re.compile(r"(?i)^(sure|certainly|of\s+course)[,!.]"),
        re.compile(r"(?i)^(let\s+me|I'll|I\s+will)\s+(extract|identify|list|break)"),
    ]

    @staticmethod
    def _is_junk_atom(line: str) -> bool:
        """Return True if *line* looks like LLM meta-commentary."""
        for pat in CorpusStore._JUNK_PATTERNS:
            if pat.search(line):
                return True
        return False

    def score_new_conditions(self, user_query: str = "") -> int:
        """Score all unscored conditions using genuine per-finding LLM assessment.

        Uses ``score_version = 0`` (not ``scored_at = ''``) to find unscored
        findings.  This is critical: if the maestro writes flat default scores
        via a bulk UPDATE (setting scored_at but NOT incrementing score_version),
        those findings will be RE-SCORED here with genuine per-finding LLM
        assessment.  This breaks the flat-scoring pre-emption chain.

        Returns count of findings scored.
        """
        unscored = self.conn.execute(
            "SELECT id, fact, source_url FROM conditions "
            "WHERE score_version = 0 AND consider_for_use = TRUE "
            "AND row_type = 'finding'"
        ).fetchall()
        if not unscored:
            return 0

        for cid, fact, source_url in unscored:
            try:
                self._score_single(cid, fact, source_url, user_query)
            except Exception:
                logger.warning(
                    "Flock scoring failed for condition #%d "
                    "-- keeping defaults",
                    cid,
                    exc_info=True,
                )

        return len(unscored)

    def _score_single(
        self, cid: int, fact: str, source_url: str, user_query: str,
    ) -> None:
        """Score a single condition across all gradient dimensions.

        All 7 scoring prompts are batched into a single parallel call
        via ``_http_complete_batch()`` so they execute concurrently
        instead of sequentially (7x speedup per condition).

        Every prompt includes the user's research question so the LLM
        can calibrate scores relative to the actual topic.  Without
        this context, all findings received identical scores because
        the LLM had no frame of reference for what "relevant" or
        "novel" meant.
        """
        url_text = source_url if source_url else "(no URL)"
        query_ctx = (
            f"Research question: {user_query}\n" if user_query else ""
        )

        # Build all scoring prompts up front — each one is query-aware
        prompts: list[tuple[str, str]] = [
            (
                f"{query_ctx}"
                "Rate the confidence of this research finding on 0.0-1.0.\n"
                "Consider: Does it make specific, verifiable claims? "
                "Does it cite named sources, studies, or data? "
                "Is the language hedged ('may', 'might') or definitive?\n"
                "0.2 = vague opinion with no evidence. "
                "0.5 = plausible claim with some support. "
                "0.8 = specific claim citing named studies/data. "
                "1.0 = verifiable fact with exact citations.\n"
                "Return ONLY a decimal number.\n"
                f"Finding: {fact}",
                "score_confidence",
            ),
            (
                f"{query_ctx}"
                "Rate the trustworthiness of this source on 0.0-1.0.\n"
                "0.1 = no URL or obviously unreliable. "
                "0.3 = blog/forum/social media. "
                "0.5 = established news outlet. "
                "0.7 = professional/institutional source. "
                "0.9 = peer-reviewed journal, government data, or "
                "primary academic source.\n"
                "Return ONLY a decimal number.\n"
                f"URL: {url_text}",
                "score_trust",
            ),
            (
                f"{query_ctx}"
                "Rate how specific and concrete this finding is on 0.0-1.0.\n"
                "0.1 = completely generic ('experts say it matters'). "
                "0.3 = names a concept but no data or specifics. "
                "0.5 = mentions specific mechanisms or named theories. "
                "0.7 = includes names, dates, numbers, or citations. "
                "1.0 = contains exact data points, measurements, or "
                "direct quotes with attribution.\n"
                "Return ONLY a decimal number.\n"
                f"Finding: {fact}",
                "score_specificity",
            ),
            (
                f"{query_ctx}"
                "Rate the fabrication risk of this finding on 0.0-1.0.\n"
                "0.0 = clearly grounded in verifiable, real sources. "
                "0.3 = plausible but hard to verify independently. "
                "0.6 = suspicious — mixes real concepts with "
                "unverifiable claims. "
                "1.0 = likely fabricated — names fake studies, "
                "impossible statistics, or hallucinates details.\n"
                "Return ONLY a decimal number.\n"
                f"Finding: {fact}\nSource: {url_text}",
                "score_fabrication",
            ),
            (
                f"{query_ctx}"
                "Rate how novel this finding is on 0.0-1.0.\n"
                "Consider novelty RELATIVE TO the research question — "
                "common knowledge about an obscure topic is still novel "
                "if it's not widely cross-referenced.\n"
                "0.1 = textbook knowledge anyone in the field knows. "
                "0.3 = well-known within the discipline. "
                "0.5 = known to specialists but crosses disciplines "
                "in a useful way. "
                "0.8 = unusual connection or rarely cited finding. "
                "1.0 = genuinely surprising or contrarian.\n"
                "Return ONLY a decimal number.\n"
                f"Finding: {fact}",
                "score_novelty",
            ),
            (
                f"{query_ctx}"
                "Rate how actionable this finding is for the research "
                "question on 0.0-1.0.\n"
                "0.1 = purely background context, no usable content. "
                "0.3 = informational but not directly applicable. "
                "0.5 = provides a useful reference or framework. "
                "0.8 = contains specific evidence or arguments that "
                "directly address the research question. "
                "1.0 = a key finding that could change the direction "
                "of the research.\n"
                "Return ONLY a decimal number.\n"
                f"Finding: {fact}",
                "score_actionability",
            ),
        ]

        # Conditionally add relevance scoring — needs a user query to
        # evaluate against; without one the LLM has no frame of reference
        has_relevance = bool(user_query)
        if has_relevance:
            prompts.append((
                f"{query_ctx}"
                "Rate how relevant this finding is to the research "
                "question on 0.0-1.0.\n"
                "Score based on genuine conceptual connection to the "
                "research question — not surface keyword overlap.\n"
                "0.0 = completely off-topic, no conceptual connection. "
                "0.2 = shares a keyword but different domain entirely. "
                "0.5 = tangentially related, same broad field. "
                "0.7 = directly addresses a sub-question. "
                "1.0 = core finding that precisely answers the query.\n"
                "Return ONLY a decimal number.\n"
                f"Finding: {fact}",
                "score_relevance",
            ))

        # Fire all prompts in parallel
        responses = self._http_complete_batch(prompts, max_tokens=32)

        # Parse results in the same order as prompts
        confidence = self._parse_float(responses[0], 0.5)
        trust = self._parse_float(responses[1], 0.5)
        specificity = self._parse_float(responses[2], 0.5)
        fabrication = self._parse_float(responses[3], 0.0)
        novelty = self._parse_float(responses[4], 0.5)
        actionability = self._parse_float(responses[5], 0.5)
        relevance = (
            self._parse_float(responses[6], 0.5)
            if has_relevance else 0.5
        )

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE conditions
               SET confidence = ?, trust_score = ?,
                   specificity_score = ?, fabrication_risk = ?,
                   relevance_score = ?, novelty_score = ?,
                   actionability_score = ?,
                   scored_at = ?,
                   score_version = score_version + 1,
                   processing_status = 'scored'
               WHERE id = ?""",
            [confidence, trust, specificity, fabrication, relevance,
             novelty, actionability, now, cid],
        )

    @staticmethod
    def _parse_float(text: str, default: float) -> float:
        """Extract a float from Flock's response text."""
        text = text.strip()
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        return default

    # ------------------------------------------------------------------
    # Flock-powered semantic dedup
    # ------------------------------------------------------------------

    def compute_duplications(self) -> int:
        """Compute pairwise duplication scores for new conditions.

        Updates duplication_score on each row and stores similarity
        results as relationship rows (row_type='similarity') in
        the conditions table.  Returns count of pairs evaluated.

        Uses a proper-noun / technical-term pre-filter to skip obviously
        unrelated pairs, then batches the remaining LLM comparisons in
        parallel via ``_http_complete_batch()``.
        """
        new_ids = self.conn.execute(
            "SELECT id FROM conditions "
            "WHERE duplication_score < 0.0 AND scored_at != '' "
            "AND consider_for_use = TRUE AND row_type = 'finding'"
        ).fetchall()
        if not new_ids:
            return 0

        pair_count = 0
        for (new_id,) in new_ids:
            new_fact = self.conn.execute(
                "SELECT fact FROM conditions WHERE id = ?", [new_id]
            ).fetchone()[0]

            others = self.conn.execute(
                "SELECT id, fact FROM conditions "
                "WHERE id != ? AND id < ? "
                "AND consider_for_use = TRUE AND row_type = 'finding'",
                [new_id, new_id],
            ).fetchall()

            # Pre-filter: use proper nouns & technical terms (not generic
            # words) to identify candidate pairs for LLM comparison.
            # Generic word overlap misses paraphrases that use different
            # vocabulary for the same claim.  Proper nouns, numbers, and
            # technical terms are much stronger signals of topical overlap.
            # Common sentence starters that get capitalised but are
            # NOT proper nouns or technical terms.
            _SIG_STOPS = {
                "the", "this", "that", "these", "there", "their",
                "a", "an", "in", "it", "is", "are", "for", "from",
                "with", "has", "have", "was", "were", "will", "been",
                "some", "many", "most", "several", "according",
                "however", "although", "research", "studies", "recent",
                "new", "our", "its", "other", "such", "both", "each",
                "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "more", "less", "much", "very",
                "while", "when", "where", "what", "which", "who",
                "how", "why", "also", "but", "yet", "nor", "not",
                "all", "any", "no", "only", "so", "too", "than",
                "they", "we", "he", "she", "you", "may", "can",
                "would", "could", "should", "must", "shall", "might",
            }

            def _signature_terms(text: str) -> set[str]:
                """Extract proper nouns, numbers, and technical terms.

                Filters out common sentence starters that happen to be
                capitalised at sentence boundaries.
                """
                terms: set[str] = set()
                for word in text.split():
                    clean = word.strip(".,;:!?\"'()[]")
                    if not clean:
                        continue
                    low = clean.lower()
                    if low in _SIG_STOPS:
                        continue
                    # Capitalised words (proper nouns, named theories)
                    if clean[0].isupper() and len(clean) > 1:
                        terms.add(low)
                    # Numbers and measurements
                    elif any(c.isdigit() for c in clean):
                        terms.add(low)
                    # Hyphenated compound terms (technical jargon)
                    elif "-" in clean and len(clean) > 5:
                        terms.add(low)
                return terms

            new_terms = _signature_terms(new_fact)
            candidates: list[tuple[int, str]] = []
            for other_id, other_fact in others:
                other_terms = _signature_terms(other_fact)
                # If either has very few signature terms, include as
                # candidate — the LLM should decide
                if len(new_terms) < 2 or len(other_terms) < 2:
                    candidates.append((other_id, other_fact))
                    continue
                # Require >=2 shared terms to tolerate one coincidental
                # match (e.g. a common methodology name)
                shared = len(new_terms & other_terms)
                if shared >= 2:
                    candidates.append((other_id, other_fact))

            if not candidates:
                self.conn.execute(
                    "UPDATE conditions SET duplication_score = 0.0 "
                    "WHERE id = ?", [new_id],
                )
                continue

            # Build batch of compare prompts
            prompts: list[tuple[str, str]] = []
            for other_id, other_fact in candidates:
                prompts.append((
                    "Are these two statements saying essentially the "
                    "same thing? Return a similarity score: 0.0 "
                    "(completely different) to 1.0 (identical meaning). "
                    "Return ONLY a decimal number. "
                    "A: " + new_fact + " B: " + other_fact,
                    "compare_pair",
                ))

            # Fire all comparisons in parallel
            responses = self._http_complete_batch(prompts, max_tokens=32)

            max_sim = 0.0
            for i, (other_id, other_fact) in enumerate(candidates):
                sim = self._parse_float(responses[i], 0.0)
                sim = min(max(sim, 0.0), 1.0)
                max_sim = max(max_sim, sim)
                pair_count += 1

                if sim > 0.3:
                    rel = (
                        "duplicate" if sim > 0.85
                        else "confirms" if sim > 0.5
                        else "related"
                    )
                    # Store as a relationship row in conditions table
                    rel_id = self._next_id
                    self._next_id += 1
                    self.conn.execute(
                        "INSERT INTO conditions "
                        "(id, fact, row_type, parent_id, related_id, "
                        "relationship_score, consider_for_use) "
                        "VALUES (?, ?, 'similarity', ?, ?, ?, FALSE)",
                        [rel_id, rel, new_id, other_id, sim],
                    )

            self.conn.execute(
                "UPDATE conditions SET duplication_score = ? "
                "WHERE id = ?",
                [max_sim, new_id],
            )

        return pair_count

    # ------------------------------------------------------------------
    # Algorithm battery — each is a small, composable processing step
    # ------------------------------------------------------------------

    def compute_angle_diversity_boost(self) -> int:
        """Boost findings from underrepresented angles/source types.

        Findings from rare angles get a small quality boost, nudging
        the corpus toward breadth without overriding depth.  This is
        the serendipity-aware scoring layer — it rewards diversity of
        perspective alongside traditional quality signals.

        Pure SQL — no LLM calls.

        Returns number of conditions with non-zero boost.
        """
        if not (os.environ.get("SERENDIPITY_ENABLED", "1") == "1"):
            return 0

        # Count findings per angle to detect underrepresentation
        angle_counts = self.conn.execute(
            "SELECT angle, COUNT(*) as cnt FROM conditions "
            "WHERE row_type = 'finding' AND consider_for_use = TRUE "
            "AND angle IS NOT NULL AND angle != '' AND scored_at != '' "
            "GROUP BY angle"
        ).fetchall()
        if not angle_counts or len(angle_counts) < 2:
            return 0

        max_count = max(cnt for _, cnt in angle_counts)
        if max_count <= 1:
            return 0

        # Apply diversity boost: findings from rare angles get up to
        # 0.08 extra composite quality.  The most common angle gets 0.
        # This is additive to cross_ref_boost (reused column to avoid
        # schema changes).
        updated = 0
        for angle, cnt in angle_counts:
            diversity_bonus = round(0.08 * (1.0 - cnt / max_count), 4)
            if diversity_bonus > 0.005:
                n = self.conn.execute(
                    "SELECT COUNT(*) FROM conditions "
                    "WHERE angle = ? AND row_type = 'finding' "
                    "AND consider_for_use = TRUE AND scored_at != ''",
                    [angle],
                ).fetchone()[0]
                self.conn.execute(
                    "UPDATE conditions SET cross_ref_boost = cross_ref_boost + ? "
                    "WHERE angle = ? AND row_type = 'finding' "
                    "AND consider_for_use = TRUE AND scored_at != ''",
                    [diversity_bonus, angle],
                )
                updated += n

        if updated:
            logger.info(
                "Angle diversity boost: %d conditions boosted across %d angles",
                updated, len(angle_counts),
            )
        return updated

    def compute_composite_quality(self) -> int:
        """Compute a weighted composite quality score for all scored
        conditions.  Pure SQL — no LLM calls.

        Formula: 0.25*confidence + 0.20*relevance + 0.15*trust
                 + 0.15*novelty + 0.10*specificity + 0.10*actionability
                 - 0.15*fabrication_risk - staleness_penalty
                 + cross_ref_boost

        The cross_ref_boost column accumulates multiple boosts:
        - Cross-reference boost (from compute_cross_ref_boost)
        - Source diversity bonus (from compute_source_diversity)
        - Angle diversity bonus (from compute_angle_diversity_boost)

        Recomputes every battery run so that changes to staleness_penalty
        and cross_ref_boost from earlier algorithms are reflected.

        Returns number of conditions updated.
        """
        # Count how many will be updated before running
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE scored_at != '' AND consider_for_use = TRUE"
        ).fetchone()[0]
        if count == 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET composite_quality = (
                   0.25 * confidence
                   + 0.20 * relevance_score
                   + 0.15 * trust_score
                   + 0.15 * novelty_score
                   + 0.10 * specificity_score
                   + 0.10 * actionability_score
                   - 0.15 * fabrication_risk
                   - staleness_penalty
                   + cross_ref_boost
               ),
               processing_status = CASE
                   WHEN processing_status IN ('raw', 'scored') THEN 'analysed'
                   ELSE processing_status
               END
               WHERE scored_at != ''
                 AND consider_for_use = TRUE
            """
        )
        logger.info("Composite quality: updated %d conditions", count)
        return count

    def apply_quality_gate(self, threshold: float = 0.25) -> int:
        """Mark low-quality conditions for expansion instead of
        passing them to the swarm.  Pure SQL.

        Conditions below *threshold* get expansion_tool set to
        'exa_search' so the researcher's next iteration can enrich
        them rather than starting from scratch.

        Returns number of conditions flagged for expansion.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE composite_quality >= 0.0 AND composite_quality < ? "
            "AND expansion_tool = 'none' AND consider_for_use = TRUE",
            [threshold],
        ).fetchone()[0]
        if count == 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET expansion_tool = 'web_search_advanced_exa',
                   expansion_hint = 'Low composite quality ('
                       || ROUND(composite_quality, 2) || ') — '
                       || 'search for more specific data on: '
                       || SUBSTR(fact, 1, 120)
               WHERE composite_quality >= 0.0
                 AND composite_quality < ?
                 AND expansion_tool = 'none'
                 AND consider_for_use = TRUE
            """,
            [threshold],
        )
        if count:
            logger.info(
                "Quality gate: flagged %d conditions for expansion "
                "(threshold=%.2f)", count, threshold,
            )
        return count

    def apply_specificity_gate(self) -> int:
        """Flag vague/generic conditions for enrichment via search.

        A condition is vague if specificity_score < 0.3 AND it has no
        source URL.  These get expansion_tool = 'brave_deep' to
        trigger a deep search for concrete data.  Pure SQL.

        Returns number flagged.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE specificity_score < 0.3 "
            "AND (source_url = '' OR source_url IS NULL) "
            "AND expansion_tool = 'none' AND scored_at != '' "
            "AND consider_for_use = TRUE"
        ).fetchone()[0]
        if count == 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET expansion_tool = 'brave_web_search',
                   expansion_hint = 'Vague finding — search for '
                       || 'specific names/numbers/dates: '
                       || SUBSTR(fact, 1, 120)
               WHERE specificity_score < 0.3
                 AND (source_url = '' OR source_url IS NULL)
                 AND expansion_tool = 'none'
                 AND scored_at != ''
                 AND consider_for_use = TRUE
            """
        )
        if count:
            logger.info(
                "Specificity gate: flagged %d vague conditions "
                "for deep search", count,
            )
        return count

    def apply_relevance_gate(self) -> int:
        """Exclude atoms with very low relevance scores.

        Defense-in-depth against noise that survived the prompt-level
        relevance filter.  Atoms with relevance_score < 0.25 are
        marked ``consider_for_use=FALSE``.  Pure SQL — no LLM calls.

        Returns number excluded.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE relevance_score < 0.25 "
            "AND relevance_score >= 0.0 "
            "AND scored_at != '' "
            "AND consider_for_use = TRUE "
            "AND row_type = 'finding'"
        ).fetchone()[0]
        if count == 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET consider_for_use = FALSE,
                   obsolete_reason = 'relevance_gate: score='
                       || ROUND(relevance_score, 2)
               WHERE relevance_score < 0.25
                 AND relevance_score >= 0.0
                 AND scored_at != ''
                 AND consider_for_use = TRUE
                 AND row_type = 'finding'
            """
        )
        if count:
            logger.info(
                "Relevance gate: excluded %d low-relevance atoms", count,
            )
        return count

    def build_narrative_chains(self) -> int:
        """Walk causal/temporal/support edges to build narrative chains.

        Queries relationship rows (row_type in NARRATIVE_REL_TYPES) to
        find connected sequences of atoms.  For each chain of length
        >= 2, creates a ``row_type='narrative_chain'`` row whose
        ``fact`` field lists the chain members in order.

        Pure SQL + Python graph walk — no LLM calls.

        Returns number of chains built.
        """
        # Fetch all narrative edges
        edge_types = tuple(self._NARRATIVE_REL_TYPES)
        placeholders = ", ".join("?" for _ in edge_types)
        edges = self.conn.execute(
            f"SELECT parent_id, related_id, row_type "
            f"FROM conditions "
            f"WHERE row_type IN ({placeholders}) "
            f"AND parent_id IS NOT NULL "
            f"AND related_id IS NOT NULL",
            list(edge_types),
        ).fetchall()

        if not edges:
            return 0

        # Build adjacency list
        graph: dict[int, list[tuple[int, str]]] = defaultdict(list)
        all_targets: set[int] = set()
        for src, tgt, rel in edges:
            graph[src].append((tgt, rel))
            all_targets.add(tgt)

        # Find chain roots (nodes with outgoing edges but not targeted
        # by any edge — these are natural starting points).
        roots = set(graph.keys()) - all_targets
        if not roots:
            # All nodes are part of cycles; pick all sources
            roots = set(graph.keys())

        # Walk from each root to build chains (DFS, no revisits)
        chains: list[list[int]] = []
        visited_global: set[int] = set()

        for root in roots:
            if root in visited_global:
                continue
            stack: list[tuple[int, list[int]]] = [(root, [root])]
            while stack:
                node, path = stack.pop()
                extended = False
                for tgt, _rel in graph.get(node, []):
                    if tgt not in path:  # avoid cycles within chain
                        stack.append((tgt, path + [tgt]))
                        extended = True
                if not extended and len(path) >= 2:
                    chains.append(path)
                    visited_global.update(path)

        # Deduplicate chains that are subsets of longer chains
        chains.sort(key=len, reverse=True)
        covered: set[int] = set()
        final_chains: list[list[int]] = []
        for chain in chains:
            if not covered.issuperset(chain):
                final_chains.append(chain)
                covered.update(chain)

        # Remove stale narrative chains before rebuilding.
        # This ensures chains reflect the current state of the corpus.
        self.conn.execute(
            "DELETE FROM conditions WHERE row_type = 'narrative_chain'"
        )

        # Store each chain as a narrative_chain row, tagged with the
        # max iteration of its members for cross-iteration tracking.
        chain_count = 0
        for chain in final_chains:
            # Verify all members still exist and are active
            placeholders_c = ", ".join("?" for _ in chain)
            active = self.conn.execute(
                f"SELECT COUNT(*) FROM conditions "
                f"WHERE id IN ({placeholders_c}) "
                f"AND consider_for_use = TRUE "
                f"AND row_type = 'finding'",
                chain,
            ).fetchone()[0]
            if active < 2:
                continue

            # Tag chain with the max iteration of its members
            max_iter = self.conn.execute(
                f"SELECT COALESCE(MAX(iteration), 0) FROM conditions "
                f"WHERE id IN ({placeholders_c})",
                chain,
            ).fetchone()[0]

            chain_id = self._next_id
            self._next_id += 1
            chain_fact = "chain:" + ",".join(str(x) for x in chain)
            self.conn.execute(
                "INSERT INTO conditions "
                "(id, fact, row_type, parent_id, related_id, "
                "consider_for_use, relationship_score, iteration) "
                "VALUES (?, ?, 'narrative_chain', ?, ?, FALSE, ?, ?)",
                [chain_id, chain_fact,
                 chain[0], chain[-1], float(len(chain)), max_iter],
            )
            chain_count += 1

        if chain_count:
            logger.info(
                "Narrative chains: built %d chains from %d edges",
                chain_count, len(edges),
            )
        return chain_count

    def compute_cross_ref_boost(self) -> int:
        """Boost conditions that are confirmed by multiple angles.

        Queries relationship rows (row_type='similarity') with
        relationship = 'confirms' from different angles to compute
        cross-reference boosts.

        Returns number of conditions boosted.
        """
        # Reset first to prevent accumulation from
        # compute_source_diversity (which adds to cross_ref_boost).
        self.conn.execute(
            "UPDATE conditions SET cross_ref_boost = 0.0 "
            "WHERE scored_at != '' AND consider_for_use = TRUE"
        )
        self.conn.execute(
            """UPDATE conditions
               SET cross_ref_boost = boost.val
               FROM (
                   SELECT c.id,
                          COALESCE(LEAST(0.2, COUNT(*) * 0.05), 0.0) AS val
                   FROM conditions c
                   JOIN conditions sf
                       ON sf.row_type = 'similarity'
                      AND (sf.parent_id = c.id OR sf.related_id = c.id)
                   JOIN conditions c2
                       ON (sf.related_id = c2.id OR sf.parent_id = c2.id)
                      AND c2.id != c.id
                      AND c2.angle != c.angle
                   WHERE sf.fact IN ('confirms', 'related')
                     AND c.scored_at != ''
                     AND c.consider_for_use = TRUE
                   GROUP BY c.id
               ) AS boost
               WHERE conditions.id = boost.id
            """
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE cross_ref_boost > 0.0 AND consider_for_use = TRUE"
        ).fetchone()[0]
        if count:
            logger.info(
                "Cross-reference boost: %d conditions boosted", count,
            )
        return count

    def compute_staleness_decay(self, current_iteration: int = 0) -> int:
        """Apply a small penalty to conditions from older iterations.

        Newer conditions are fresher; older ones decay slightly so
        the swarm prioritises recent findings.  Pure SQL.

        Decay = 0.02 * (current_iteration - condition.iteration)
        Capped at 0.10 so old conditions aren't killed, just deprioritised.

        Returns number of conditions with non-zero penalty.
        """
        if current_iteration <= 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET staleness_penalty = LEAST(
                   0.10,
                   0.02 * (? - iteration)
               )
               WHERE iteration < ?
                 AND scored_at != ''
                 AND consider_for_use = TRUE
            """,
            [current_iteration, current_iteration],
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE staleness_penalty > 0.0 AND consider_for_use = TRUE"
        ).fetchone()[0]
        if count:
            logger.info(
                "Staleness decay: %d conditions penalised "
                "(current_iteration=%d)", count, current_iteration,
            )
        return count

    def compute_source_diversity(self) -> int:
        """Boost conditions that have a source URL from a unique domain.

        If many conditions cite the same domain, each gets a smaller
        boost.  Conditions with unique domains get a larger boost.
        This encourages source diversity in the final synthesis.
        Pure SQL.

        Returns number of conditions with updated boost.
        """
        # DuckDB does not allow aggregate functions inside correlated
        # subqueries in UPDATE.  Compute per-domain counts first, then
        # join-update to apply the diversity bonus.
        self.conn.execute(
            """UPDATE conditions
               SET cross_ref_boost = cross_ref_boost + div.bonus
               FROM (
                   SELECT c.id,
                          0.05 / GREATEST(1, COUNT(*) OVER (
                              PARTITION BY SPLIT_PART(
                                  REPLACE(REPLACE(c.source_url,
                                      'https://', ''), 'http://', ''),
                                  '/', 1
                              )
                          )) AS bonus
                   FROM conditions c
                   WHERE c.scored_at != ''
                     AND c.source_url != ''
                     AND c.source_url IS NOT NULL
                     AND c.consider_for_use = TRUE
               ) AS div
               WHERE conditions.id = div.id
            """
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE scored_at != '' AND source_url != '' "
            "AND source_url IS NOT NULL AND consider_for_use = TRUE"
        ).fetchone()[0]
        if count:
            logger.info(
                "Source diversity: adjusted %d conditions", count,
            )
        return count

    def detect_contradictions(self) -> int:
        """Detect contradicting condition pairs via Flock LLM.

        Queries relationship rows (row_type='similarity') with
        similarity score between 0.3 and 0.7 as candidates for
        contradiction.  Results are stored as new relationship rows
        (row_type='contradiction').

        Batches all contradiction checks in parallel via
        ``_http_complete_batch()``.

        Returns number of contradiction pairs found.
        """
        candidates = self.conn.execute(
            """SELECT sf.parent_id, c1.fact,
                      sf.related_id, c2.fact
               FROM conditions sf
               JOIN conditions c1 ON sf.parent_id = c1.id
               JOIN conditions c2 ON sf.related_id = c2.id
               WHERE sf.row_type = 'similarity'
                 AND sf.fact = 'related'
                 AND sf.relationship_score BETWEEN 0.3 AND 0.7
                 AND c1.contradiction_flag = FALSE
                 AND c2.contradiction_flag = FALSE
                 AND c1.consider_for_use = TRUE
                 AND c2.consider_for_use = TRUE
            """
        ).fetchall()
        if not candidates:
            return 0

        # Build batch of contradiction prompts
        prompts: list[tuple[str, str]] = [
            (
                "Do these two statements contradict each other? "
                "Return a contradiction score: 0.0 (no "
                "contradiction) to 1.0 (direct contradiction). "
                "Return ONLY a decimal number. "
                "A: " + fact_a + " B: " + fact_b,
                "detect_contradictions",
            )
            for _, fact_a, _, fact_b in candidates
        ]
        responses = self._http_complete_batch(prompts, max_tokens=32)

        contradiction_count = 0
        for i, (id_a, fact_a, id_b, fact_b) in enumerate(candidates):
            score = self._parse_float(responses[i], 0.0)
            if score > 0.6:
                # Store as a contradiction relationship row
                rel_id = self._next_id
                self._next_id += 1
                self.conn.execute(
                    "INSERT INTO conditions "
                    "(id, fact, row_type, parent_id, related_id, "
                    "relationship_score, consider_for_use) "
                    "VALUES (?, 'auto-detected', 'contradiction', ?, ?, ?, FALSE)",
                    [rel_id, id_a, id_b, score],
                )
                self.conn.execute(
                    "UPDATE conditions "
                    "SET contradiction_flag = TRUE, "
                    "    contradiction_partner = ? "
                    "WHERE id = ?",
                    [id_b, id_a],
                )
                self.conn.execute(
                    "UPDATE conditions "
                    "SET contradiction_flag = TRUE, "
                    "    contradiction_partner = ? "
                    "WHERE id = ?",
                    [id_a, id_b],
                )
                contradiction_count += 1

        if contradiction_count:
            logger.info(
                "Contradiction detection: %d pairs found",
                contradiction_count,
            )
        return contradiction_count

    def compute_information_density(self) -> int:
        """Score conditions by information density via Flock LLM.

        High-density conditions pack many facts/data points per
        character.  Low-density conditions are verbose or repetitive.

        Batches all density prompts in parallel via
        ``_http_complete_batch()``.

        Returns number of conditions scored.
        """
        unscored = self.conn.execute(
            "SELECT id, fact FROM conditions "
            "WHERE information_density < 0.0 AND scored_at != '' "
            "AND consider_for_use = TRUE"
        ).fetchall()
        if not unscored:
            return 0

        prompts: list[tuple[str, str]] = [
            (
                "Rate the information density of this text on "
                "0.0 to 1.0. High density = many concrete facts, "
                "numbers, names, URLs per sentence. Low density "
                "= verbose, repetitive, or filler text. Return "
                "ONLY a decimal number. "
                "Text: " + fact,
                "info_density",
            )
            for _, fact in unscored
        ]
        responses = self._http_complete_batch(prompts, max_tokens=32)

        for i, (cid, _) in enumerate(unscored):
            density = self._parse_float(responses[i], 0.5)
            self.conn.execute(
                "UPDATE conditions "
                "SET information_density = ? WHERE id = ?",
                [density, cid],
            )

        logger.info(
            "Information density: scored %d conditions",
            len(unscored),
        )
        return len(unscored)

    def cluster_conditions(self) -> int:
        """Cluster similar conditions and rank within each cluster.

        Queries relationship rows (row_type='similarity') to build
        clusters via a simple union-find approach.  Within each
        cluster, the condition with the highest composite_quality
        becomes rank 0 (representative).  Others get rank 1+
        (supplementary).

        Returns number of clusters formed.
        """
        # Get all conditions and their similarity edges
        conditions = self.conn.execute(
            "SELECT id FROM conditions "
            "WHERE scored_at != '' AND consider_for_use = TRUE "
            "AND row_type = 'finding'"
        ).fetchall()
        if not conditions:
            return 0

        # Union-find for clustering
        parent: dict[int, int] = {row[0]: row[0] for row in conditions}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Build clusters from high-similarity relationship rows
        edges = self.conn.execute(
            "SELECT parent_id, related_id "
            "FROM conditions "
            "WHERE row_type = 'similarity' "
            "AND relationship_score > 0.5"
        ).fetchall()
        for a, b in edges:
            if a in parent and b in parent:
                union(a, b)

        # Group by cluster root
        clusters: dict[int, list[int]] = {}
        for cid in parent:
            root = find(cid)
            clusters.setdefault(root, []).append(cid)

        # Assign cluster IDs and ranks
        cluster_count = 0
        for cluster_idx, (_, members) in enumerate(
            sorted(clusters.items())
        ):
            if len(members) < 2:
                # Singleton — no clustering needed
                self.conn.execute(
                    "UPDATE conditions SET cluster_id = ?, "
                    "cluster_rank = 0 WHERE id = ?",
                    [cluster_idx, members[0]],
                )
                continue

            cluster_count += 1
            # Rank by composite_quality within cluster
            scored = self.conn.execute(
                "SELECT id, composite_quality FROM conditions "
                "WHERE id IN ({}) "
                "ORDER BY composite_quality DESC".format(
                    ",".join("?" * len(members))
                ),
                members,
            ).fetchall()
            for rank, (cid, _) in enumerate(scored):
                self.conn.execute(
                    "UPDATE conditions SET cluster_id = ?, "
                    "cluster_rank = ? WHERE id = ?",
                    [cluster_idx, rank, cid],
                )

        # Update processing status
        self.conn.execute(
            """UPDATE conditions
               SET processing_status = 'clustered'
               WHERE processing_status = 'analysed'
                 AND cluster_id >= 0
            """
        )

        logger.info(
            "Clustering: formed %d multi-member clusters from %d "
            "conditions", cluster_count, len(parent),
        )
        return cluster_count

    def compress_redundant(self) -> int:
        """Merge near-duplicate conditions via Flock LLM.

        Queries relationship rows (row_type='similarity') for
        duplicate pairs (score > 0.85).  Creates a NEW child row
        with the merged text and dual-parent lineage
        (parent_id=atom_a, related_id=atom_b).  Both originals are
        marked ``consider_for_use=FALSE`` — no mutation, full lineage.

        Returns number of merges performed.
        """
        dupes = self.conn.execute(
            """SELECT sf.parent_id, c1.fact, c1.source_url,
                      c1.composite_quality AS quality_a,
                      sf.related_id, c2.fact, c2.source_url,
                      c2.composite_quality AS quality_b
               FROM conditions sf
               JOIN conditions c1 ON sf.parent_id = c1.id
               JOIN conditions c2 ON sf.related_id = c2.id
               WHERE sf.row_type = 'similarity'
                 AND sf.fact = 'duplicate'
                 AND sf.relationship_score > 0.85
                 AND c1.consider_for_use = TRUE
                 AND c2.consider_for_use = TRUE
            """
        ).fetchall()
        if not dupes:
            return 0

        merge_count = 0
        for id_a, fact_a, url_a, q_a, id_b, fact_b, url_b, q_b in dupes:
            # Skip if either parent was already consumed by an earlier
            # merge in this batch (both must still be active).
            still_active = self.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE id IN (?, ?) AND consider_for_use = TRUE",
                [id_a, id_b],
            ).fetchone()[0]
            if still_active < 2:
                continue

            try:
                merged_fact = self._http_complete(
                    "Merge these two near-duplicate findings into "
                    "one stronger, more complete statement. Keep "
                    "ALL specific data from both. If one has a URL "
                    "and the other doesn't, keep the URL. Return "
                    "ONLY the merged statement, nothing else. "
                    "A: " + fact_a + " B: " + fact_b,
                    caller="compress_redundant",
                )
                if not merged_fact or not merged_fact.strip():
                    continue

                best_url = url_a or url_b

                # Create a NEW child row with dual-parent lineage
                child_id = self._next_id
                self._next_id += 1
                now = datetime.now(timezone.utc).isoformat()
                self.conn.execute(
                    "INSERT INTO conditions "
                    "(id, fact, source_url, row_type, parent_id, "
                    "related_id, consider_for_use, created_at, "
                    "source_type, source_ref, angle, iteration) "
                    "VALUES (?, ?, ?, 'finding', ?, ?, TRUE, ?, "
                    "'merge', 'compress_redundant', "
                    "(SELECT angle FROM conditions WHERE id = ?), "
                    "(SELECT MAX(iteration) FROM conditions "
                    " WHERE id IN (?, ?)))",
                    [child_id, merged_fact.strip(), best_url,
                     id_a, id_b, now, id_a, id_a, id_b],
                )

                # Mark BOTH originals as consumed — no mutation
                self.conn.execute(
                    "UPDATE conditions "
                    "SET consider_for_use = FALSE, "
                    "    obsolete_reason = 'merged into ' || ?, "
                    "    duplication_score = 1.0 "
                    "WHERE id IN (?, ?)",
                    [str(child_id), id_a, id_b],
                )
                merge_count += 1
            except Exception:
                logger.warning(
                    "Redundancy merge failed for (%d, %d)",
                    id_a, id_b, exc_info=True,
                )

        if merge_count:
            logger.info(
                "Redundancy compression: merged %d pairs into "
                "%d new children",
                merge_count, merge_count,
            )
        return merge_count

    def mark_ready(self) -> int:
        """Mark all fully-processed conditions as 'ready' for the
        swarm.  Pure SQL.

        A condition is ready if:
        - It's been scored (scored_at != '')
        - consider_for_use = TRUE
        - It's not flagged for expansion, or expansion is fulfilled

        Returns number marked ready.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE scored_at != '' "
            "AND consider_for_use = TRUE "
            "AND processing_status NOT IN ('merged', 'ready') "
            "AND (expansion_tool = 'none' OR expansion_fulfilled = TRUE)"
        ).fetchone()[0]
        self.conn.execute(
            """UPDATE conditions
               SET processing_status = 'ready'
               WHERE scored_at != ''
                 AND consider_for_use = TRUE
                 AND processing_status NOT IN ('merged', 'ready')
                 AND (expansion_tool = 'none' OR expansion_fulfilled = TRUE)
            """
        )
        logger.info("Marked %d conditions as ready for swarm", count)
        return count

    def get_expansion_targets(self) -> list[dict]:
        """Return conditions flagged for expansion with their hints.

        The researcher can use these to guide its next iteration's
        search strategy instead of starting from scratch.
        """
        rows = self.conn.execute(
            """SELECT id, fact, expansion_tool, expansion_hint,
                      specificity_score, composite_quality
               FROM conditions
               WHERE expansion_tool != 'none'
                 AND expansion_fulfilled = FALSE
                 AND consider_for_use = TRUE
               ORDER BY composite_quality ASC
            """
        ).fetchall()
        return [
            {
                "id": r[0], "fact": r[1],
                "strategy": r[2], "hint": r[3],
                "specificity": r[4], "quality": r[5],
            }
            for r in rows
        ]

    def get_cluster_representatives(self) -> list[dict]:
        """Return only cluster representatives (rank 0) for the swarm.

        This is the key optimisation: instead of processing all 97
        conditions, the swarm only processes ~20-30 cluster reps.
        """
        rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      novelty_score, specificity_score,
                      composite_quality, cluster_id,
                      (SELECT COUNT(*) FROM conditions c2
                       WHERE c2.cluster_id = conditions.cluster_id
                         AND c2.consider_for_use = TRUE)
                       AS cluster_size
               FROM conditions
               WHERE cluster_rank = 0
                 AND consider_for_use = TRUE
                 AND scored_at != ''
               ORDER BY composite_quality DESC
            """
        ).fetchall()
        return [
            {
                "id": r[0], "fact": r[1], "source_url": r[2],
                "confidence": r[3], "trust_score": r[4],
                "novelty_score": r[5], "specificity_score": r[6],
                "composite_quality": r[7], "cluster_id": r[8],
                "cluster_size": r[9],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Tracing helpers
    # ------------------------------------------------------------------

    def _score_snapshot(self) -> dict[str, Any]:
        """Capture a snapshot of score distributions for tracing.

        Returns a dict with mean/min/max for key score dimensions
        plus status counts.  Designed to be small enough to store
        as JSON in the algorithm_traces table.
        """
        row = self.conn.execute(
            "SELECT COUNT(*), "
            "  AVG(composite_quality), MIN(composite_quality), MAX(composite_quality), "
            "  AVG(confidence), AVG(trust_score), AVG(novelty_score), "
            "  AVG(specificity_score), AVG(fabrication_risk), "
            "  AVG(staleness_penalty), AVG(cross_ref_boost), "
            "  AVG(information_density) "
            "FROM conditions WHERE scored_at != ''"
        ).fetchone()
        if not row or row[0] == 0:
            return {"count": 0}
        return {
            "count": row[0],
            "mean_quality": round(float(row[1] or 0), 4),
            "min_quality": round(float(row[2] or 0), 4),
            "max_quality": round(float(row[3] or 0), 4),
            "mean_confidence": round(float(row[4] or 0), 4),
            "mean_trust": round(float(row[5] or 0), 4),
            "mean_novelty": round(float(row[6] or 0), 4),
            "mean_specificity": round(float(row[7] or 0), 4),
            "mean_fabrication_risk": round(float(row[8] or 0), 4),
            "mean_staleness": round(float(row[9] or 0), 4),
            "mean_cross_ref": round(float(row[10] or 0), 4),
            "mean_info_density": round(float(row[11] or 0), 4),
        }

    def _status_snapshot(self) -> dict[str, int]:
        """Capture processing_status counts for tracing."""
        rows = self.conn.execute(
            "SELECT processing_status, COUNT(*) "
            "FROM conditions GROUP BY processing_status"
        ).fetchall()
        return {str(r[0] or "raw"): int(r[1]) for r in rows}

    def _trace_algorithm(
        self,
        name: str,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> int:
        """Run an algorithm function with before/after tracing.

        Captures score snapshots before and after, measures duration,
        detects quality regressions, and writes trace to SQLite.
        Emits dashboard events so the live UI shows battery progress.
        Returns the algorithm's result count.
        """
        # Emit algorithm start to live dashboard
        from dashboard import get_any_active_collector
        collector = get_any_active_collector()
        try:
            if collector:
                collector.emit_event(
                    "algorithm_start",
                    agent="flock",
                    data={"algorithm": name, "iteration": self._trace_iteration},
                )
        except Exception:
            pass  # never block pipeline for dashboard

        before = self._score_snapshot() if self._trace_enabled else {}
        t0 = time.monotonic()

        result = func(*args, **kwargs)

        duration_ms = (time.monotonic() - t0) * 1000
        after = self._score_snapshot() if self._trace_enabled else {}

        # Emit algorithm end to live dashboard
        try:
            if collector:
                collector.emit_event(
                    "algorithm_end",
                    agent="flock",
                    data={
                        "algorithm": name,
                        "affected": result,
                        "duration_ms": round(duration_ms, 1),
                    },
                )
        except Exception:
            pass  # never block pipeline for dashboard

        if self._trace_enabled:
            try:
                from dashboard import event_store

                event_store.insert_algorithm_trace(
                    session_id=self._trace_session_id,
                    iteration=self._trace_iteration,
                    algorithm_name=name,
                    affected_count=result,
                    before_snapshot=before,
                    after_snapshot=after,
                    details={},
                    duration_ms=duration_ms,
                )

                # Quality regression detection
                before_q = before.get("mean_quality", 0.0)
                after_q = after.get("mean_quality", 0.0)
                if before_q > 0 and after_q < before_q:
                    delta = after_q - before_q
                    severity = (
                        "critical" if delta < -0.10
                        else "warning" if delta < -0.03
                        else "info"
                    )
                    event_store.insert_quality_regression(
                        session_id=self._trace_session_id,
                        iteration=self._trace_iteration,
                        algorithm_name=name,
                        metric_name="mean_composite_quality",
                        before_value=before_q,
                        after_value=after_q,
                        severity=severity,
                    )
                    if severity != "info":
                        logger.warning(
                            "Quality regression after %s: "
                            "%.4f -> %.4f (delta=%.4f, %s)",
                            name, before_q, after_q, delta, severity,
                        )
            except Exception:
                pass  # never block pipeline for tracing

        return result

    def _save_corpus_snapshot(
        self, phase: str,
    ) -> None:
        """Save a corpus state snapshot for iteration-over-iteration diffs.

        Captures total counts, status distribution, score summary, and
        a compact list of conditions (id, fact preview, key scores).
        """
        if not self._trace_enabled:
            return
        try:
            from dashboard import event_store

            total = self.count()
            status_counts = self._status_snapshot()
            score_summary = self._score_snapshot()

            # Compact condition list: id + first 120 chars + key scores
            rows = self.conn.execute(
                "SELECT id, SUBSTR(fact, 1, 120), composite_quality, "
                "confidence, processing_status, cluster_id, cluster_rank "
                "FROM conditions WHERE scored_at != '' "
                "ORDER BY composite_quality DESC LIMIT 200"
            ).fetchall()
            conditions = [
                {
                    "id": r[0], "fact_preview": str(r[1]),
                    "quality": round(float(r[2] or 0), 3),
                    "confidence": round(float(r[3] or 0), 3),
                    "status": str(r[4]), "cluster_id": r[5],
                    "cluster_rank": r[6],
                }
                for r in rows
            ]

            event_store.insert_corpus_snapshot(
                session_id=self._trace_session_id,
                iteration=self._trace_iteration,
                phase=phase,
                total_conditions=total,
                status_counts=status_counts,
                score_summary=score_summary,
                conditions=conditions,
            )
        except Exception:
            pass  # never block pipeline

    def run_algorithm_battery(
        self, user_query: str = "", iteration: int = 0,
    ) -> dict[str, int]:
        """Run the full battery of algorithms in sequence.

        This is the main entry point called by the condition manager
        after scoring and dedup.  Returns a dict of algorithm names
        to their result counts for logging.

        When tracing is enabled (set_trace_context was called), each
        algorithm is wrapped with before/after score snapshots, duration
        measurement, and quality regression detection.  A corpus snapshot
        is taken before and after the full battery for iteration diffs.

        Pipeline order:
        1. Staleness decay (SQL)
        2. Cross-reference boost (SQL)
        3. Source diversity (SQL)
        4. Composite quality (SQL)
        5. Quality gate (SQL)
        6. Specificity gate (SQL)
        7. Relevance gate (SQL)
        8. Information density (Flock LLM)
        9. Contradiction detection (Flock LLM)
        10. Clustering (SQL)
        11. Redundancy compression (Flock LLM)
        12. Narrative chain building (SQL + graph walk)
        13. Mark ready (SQL)
        """
        self._trace_iteration = iteration
        battery_start = time.monotonic()

        # Pre-battery corpus snapshot
        self._save_corpus_snapshot("pre_battery")

        results: dict[str, int] = {}

        results["staleness_decay"] = self._trace_algorithm(
            "staleness_decay", self.compute_staleness_decay, iteration,
        )
        results["cross_ref_boost"] = self._trace_algorithm(
            "cross_ref_boost", self.compute_cross_ref_boost,
        )
        results["source_diversity"] = self._trace_algorithm(
            "source_diversity", self.compute_source_diversity,
        )
        results["angle_diversity"] = self._trace_algorithm(
            "angle_diversity", self.compute_angle_diversity_boost,
        )
        results["composite_quality"] = self._trace_algorithm(
            "composite_quality", self.compute_composite_quality,
        )
        results["quality_gate"] = self._trace_algorithm(
            "quality_gate", self.apply_quality_gate,
        )
        results["specificity_gate"] = self._trace_algorithm(
            "specificity_gate", self.apply_specificity_gate,
        )
        results["relevance_gate"] = self._trace_algorithm(
            "relevance_gate", self.apply_relevance_gate,
        )

        # Clear stale expansion flags — prevent permanent limbo
        if iteration >= 2:
            stale = self.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE expansion_tool != 'none' AND expansion_fulfilled = FALSE "
                "AND consider_for_use = TRUE AND iteration <= ?",
                [iteration - 2],
            ).fetchone()[0]
            if stale:
                self.conn.execute(
                    "UPDATE conditions SET expansion_tool = 'none', expansion_hint = '' "
                    "WHERE expansion_tool != 'none' AND expansion_fulfilled = FALSE "
                    "AND consider_for_use = TRUE AND iteration <= ?",
                    [iteration - 2],
                )
            results["stale_expansions_cleared"] = stale

        results["info_density"] = self._trace_algorithm(
            "info_density", self.compute_information_density,
        )
        results["contradictions"] = self._trace_algorithm(
            "contradictions", self.detect_contradictions,
        )
        results["clusters"] = self._trace_algorithm(
            "clusters", self.cluster_conditions,
        )
        results["redundancy_merges"] = self._trace_algorithm(
            "redundancy_merges", self.compress_redundant,
        )
        results["narrative_chains"] = self._trace_algorithm(
            "narrative_chains", self.build_narrative_chains,
        )
        results["marked_ready"] = self._trace_algorithm(
            "mark_ready", self.mark_ready,
        )

        # Post-battery corpus snapshot
        self._save_corpus_snapshot("post_battery")

        battery_duration_ms = (time.monotonic() - battery_start) * 1000

        # Summary stats
        total = self.count()
        ready = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE processing_status = 'ready'"
        ).fetchone()[0]
        expansion = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE expansion_tool != 'none' AND expansion_fulfilled = FALSE "
            "AND consider_for_use = TRUE"
        ).fetchone()[0]
        excluded = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE consider_for_use = FALSE"
        ).fetchone()[0]
        cluster_reps = len(self.get_cluster_representatives())

        logger.info(
            "Algorithm battery complete in %.0fms: %d total, %d ready, "
            "%d for expansion, %d excluded, %d cluster reps "
            "(swarm will process %d instead of %d)",
            battery_duration_ms,
            total, ready, expansion, excluded, cluster_reps,
            cluster_reps, total,
        )

        results["total"] = total
        results["ready"] = ready
        results["expansion_targets"] = expansion
        results["excluded"] = excluded
        results["cluster_reps"] = cluster_reps
        results["battery_duration_ms"] = int(battery_duration_ms)

        return results

    # ------------------------------------------------------------------
    # Query methods (consumer-specific)
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict]:
        """Return ALL conditions -- nothing excluded, ever."""
        rows = self.conn.execute(
            """SELECT id, fact, source_url, source_type, confidence,
                      trust_score, novelty_score, specificity_score,
                      relevance_score, actionability_score,
                      duplication_score, fabrication_risk,
                      verification_status, angle, parent_id, strategy,
                      expansion_depth, iteration
               FROM conditions
               ORDER BY confidence DESC, id ASC"""
        ).fetchall()
        cols = [
            "id", "fact", "source_url", "source_type", "confidence",
            "trust_score", "novelty_score", "specificity_score",
            "relevance_score", "actionability_score", "duplication_score",
            "fabrication_risk", "verification_status", "angle", "parent_id",
            "strategy", "expansion_depth", "iteration",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_for_thinker(self) -> list[dict]:
        """Return findings for the thinker with all computed columns.

        Filters: consider_for_use=TRUE, row_type='finding'.
        Ordered by composite quality, then novelty.
        """
        rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      novelty_score, specificity_score, duplication_score,
                      fabrication_risk, verification_status, angle,
                      parent_id, related_id, expansion_depth, iteration,
                      row_type, consider_for_use,
                      composite_quality, information_density, cross_ref_boost,
                      relevance_score, actionability_score,
                      cluster_id, cluster_rank,
                      expansion_tool, expansion_hint, expansion_fulfilled,
                      contradiction_flag, contradiction_partner,
                      staleness_penalty, processing_status
               FROM conditions
               WHERE consider_for_use = TRUE
                 AND row_type = 'finding'
               ORDER BY composite_quality DESC, novelty_score DESC, id ASC"""
        ).fetchall()
        cols = [
            "id", "fact", "source_url", "confidence", "trust_score",
            "novelty_score", "specificity_score", "duplication_score",
            "fabrication_risk", "verification_status", "angle",
            "parent_id", "related_id", "expansion_depth", "iteration",
            "row_type", "consider_for_use",
            "composite_quality", "information_density", "cross_ref_boost",
            "relevance_score", "actionability_score",
            "cluster_id", "cluster_rank",
            "expansion_tool", "expansion_hint", "expansion_fulfilled",
            "contradiction_flag", "contradiction_partner",
            "staleness_penalty", "processing_status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_thoughts_for_thinker(self, max_per_angle: int = 5) -> list[dict]:
        """Return a budget-governed selection of thought rows for the thinker.

        Retrieval governance rules:
        1. **Leaf/verdict prioritization**: Arbitration verdicts and leaf
           thoughts (no children) are prioritized over intermediate thoughts.
        2. **Angle-scoped budgets**: At most *max_per_angle* thoughts per
           angle, preventing context window flooding from prolific angles.
        3. **Insight rows included**: ``row_type='insight'`` rows are always
           included (they are evidence-grounded and compact).
        4. **Depth cap**: Thoughts at expansion_depth > 2 are deprioritized
           (summaries preferred over deep tree branches).

        Returns a list of dicts with thought metadata.
        """
        # Gather all thought angles
        angles = self.get_distinct_thought_angles()

        selected: list[dict] = []
        for angle_name in angles:
            # Fetch thoughts for this angle, prioritizing verdicts and leaves
            rows = self.conn.execute(
                """SELECT id, fact, angle, strategy, iteration,
                          expansion_depth, parent_id, created_at
                   FROM conditions
                   WHERE row_type = 'thought' AND angle = ?
                     AND consider_for_use = TRUE
                   ORDER BY
                     -- Verdicts first, then leaf thoughts, then others
                     CASE
                       WHEN strategy = 'arbitration_verdict' THEN 0
                       WHEN id NOT IN (
                         SELECT DISTINCT parent_id FROM conditions
                         WHERE parent_id IS NOT NULL AND row_type = 'thought'
                       ) THEN 1
                       ELSE 2
                     END,
                     -- Prefer shallower depth
                     expansion_depth ASC,
                     -- Most recent first within priority tier
                     id DESC
                   LIMIT ?""",
                [angle_name, max_per_angle],
            ).fetchall()
            cols = [
                "id", "fact", "angle", "strategy", "iteration",
                "expansion_depth", "parent_id", "created_at",
            ]
            selected.extend(dict(zip(cols, r)) for r in rows)

        # Also include all insight rows (compact, evidence-grounded)
        insight_rows = self.conn.execute(
            """SELECT id, fact, angle, strategy, iteration,
                      expansion_depth, parent_id, created_at
               FROM conditions
               WHERE row_type = 'insight' AND consider_for_use = TRUE
               ORDER BY id DESC
               LIMIT 20""",
        ).fetchall()
        insight_cols = [
            "id", "fact", "angle", "strategy", "iteration",
            "expansion_depth", "parent_id", "created_at",
        ]
        selected.extend(dict(zip(insight_cols, r)) for r in insight_rows)

        return selected

    def get_for_synthesiser(self) -> list[dict]:
        """Return findings and insights for the synthesiser.

        Stricter than thinker: excludes high fabrication risk and
        thought rows (the synthesiser works from findings, not internal
        reasoning).  INCLUDES ``row_type='insight'`` rows — these are
        evidence-grounded conclusions materialized from arbitration,
        safe for the synthesiser to consume.
        """
        rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      novelty_score, specificity_score, duplication_score,
                      fabrication_risk, verification_status, angle,
                      parent_id, related_id, expansion_depth, iteration,
                      row_type, consider_for_use,
                      composite_quality, information_density, cross_ref_boost,
                      relevance_score, actionability_score,
                      cluster_id, cluster_rank,
                      expansion_tool, expansion_hint, expansion_fulfilled,
                      contradiction_flag, contradiction_partner,
                      staleness_penalty, processing_status
               FROM conditions
               WHERE consider_for_use = TRUE
                 AND row_type IN ('finding', 'insight')
                 AND fabrication_risk < 0.80
               ORDER BY composite_quality DESC, confidence DESC, id ASC"""
        ).fetchall()
        cols = [
            "id", "fact", "source_url", "confidence", "trust_score",
            "novelty_score", "specificity_score", "duplication_score",
            "fabrication_risk", "verification_status", "angle",
            "parent_id", "related_id", "expansion_depth", "iteration",
            "row_type", "consider_for_use",
            "composite_quality", "information_density", "cross_ref_boost",
            "relevance_score", "actionability_score",
            "cluster_id", "cluster_rank",
            "expansion_tool", "expansion_hint", "expansion_fulfilled",
            "contradiction_flag", "contradiction_partner",
            "staleness_penalty", "processing_status",
        ]
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _describe_finding(self, c: dict) -> str:
        """Convert a condition's algorithm scores into verbal prose."""
        parts = [f"[{c['id']}] {c['fact']}"]

        # Source attribution
        if c.get("source_url"):
            parts.append(f"  Source: {c['source_url']}")

        # Verbal quality assessment
        cq = c.get("composite_quality", -1)
        if cq >= 0.7:
            parts.append(
                "  This finding is well-established and strongly supported."
            )
        elif cq >= 0.4:
            parts.append(
                "  This finding has moderate support but could benefit "
                "from additional verification."
            )
        elif cq >= 0:
            parts.append(
                "  This finding is weakly supported and needs more "
                "evidence."
            )

        # Verbal specificity
        spec = c.get("specificity_score", 0.5)
        if spec < 0.3:
            parts.append(
                "  Note: This is vague — it lacks specific names, "
                "numbers, or dates."
            )

        # Fabrication risk
        fab = c.get("fabrication_risk", 0)
        if fab > 0.5:
            parts.append(
                "  Warning: This finding has elevated fabrication risk "
                "and should be independently verified."
            )
        elif fab > 0.3:
            parts.append(
                "  Note: Some fabrication risk detected — cross-check "
                "recommended."
            )

        # Contradiction
        if c.get("contradiction_flag"):
            partner = c.get("contradiction_partner", -1)
            if partner > 0:
                parts.append(
                    f"  Contradicts finding [{partner}] — both sides "
                    "need examination."
                )

        # Cross-reference boost
        xref = c.get("cross_ref_boost", 0)
        if xref > 0.1:
            parts.append(
                "  Corroborated by independent sources from "
                "different angles."
            )

        # Expansion status
        tool = c.get("expansion_tool", "none")
        if tool != "none" and not c.get("expansion_fulfilled"):
            parts.append(
                f"  Needs enrichment: "
                f"{c.get('expansion_hint', 'search for more specific data')}"
            )

        return "\n".join(parts)

    def _get_chunk_text(self, chunk_id: int) -> str:
        """Fetch the original chunk text for a given chunk row id."""
        row = self.conn.execute(
            "SELECT fact FROM conditions WHERE id = ? AND row_type = 'chunk'",
            [chunk_id],
        ).fetchone()
        return row[0] if row else ""

    def _get_narrative_chains_for_atoms(
        self, atom_ids: set[int],
    ) -> list[list[int]]:
        """Return narrative chains that involve any of the given atom ids.

        Each chain is a list of atom ids in narrative order.
        """
        chain_rows = self.conn.execute(
            "SELECT fact FROM conditions "
            "WHERE row_type = 'narrative_chain'"
        ).fetchall()
        chains: list[list[int]] = []
        for (fact,) in chain_rows:
            if not fact.startswith("chain:"):
                continue
            members = [int(x) for x in fact[6:].split(",") if x.strip()]
            if atom_ids.intersection(members):
                chains.append(members)
        return chains

    def _get_relationship_edges(
        self, atom_ids: set[int],
    ) -> list[tuple[int, int, str]]:
        """Return relationship edges (src, tgt, rel_type) among atoms."""
        if not atom_ids:
            return []
        edge_types = tuple(self._NARRATIVE_REL_TYPES)
        placeholders = ", ".join("?" for _ in edge_types)
        rows = self.conn.execute(
            f"SELECT parent_id, related_id, row_type "
            f"FROM conditions "
            f"WHERE row_type IN ({placeholders}) "
            f"AND parent_id IS NOT NULL "
            f"AND related_id IS NOT NULL",
            list(edge_types),
        ).fetchall()
        return [
            (src, tgt, rel)
            for src, tgt, rel in rows
            if src in atom_ids or tgt in atom_ids
        ]

    def format_for_thinker(self, current_iteration: int = 0) -> str:
        """Format the corpus as two-tier briefing for the thinker.

        **Tier 1 — DELTA**: New findings from the current iteration get
        the full 3-layer treatment (chunk text + atoms + relationships).

        **Tier 2 — SUMMARY**: Older findings are condensed into a
        compact digest (one line per finding, grouped by angle/topic).

        This dramatically reduces context size on later iterations
        while still giving the thinker full visibility into what's new.

        Args:
            current_iteration: The current corpus iteration number.
                When 0 or when all findings are from the same iteration,
                falls back to the full 3-layer view.
        """
        conditions = self.get_for_thinker()
        if not conditions:
            return "(no findings yet)"

        # Split into delta (current iteration) and prior findings
        if current_iteration > 0:
            delta = [c for c in conditions if c.get("iteration", 0) >= current_iteration]
            prior = [c for c in conditions if c.get("iteration", 0) < current_iteration]
        else:
            delta = conditions
            prior = []

        atom_ids = {c["id"] for c in conditions}
        cond_by_id = {c["id"]: c for c in conditions}

        lines: list[str] = [
            f"RESEARCH BRIEFING: {len(conditions)} findings "
            f"({len(delta)} new this iteration, {len(prior)} from prior iterations)\n",
        ]

        # ── Tier 2: Condensed summary of PRIOR findings ──
        if prior:
            lines.append("=" * 60)
            lines.append("PRIOR FINDINGS (condensed summary)")
            lines.append("=" * 60)
            lines.append("")

            # Group prior findings by angle/topic for compact display
            by_angle: dict[str, list[dict]] = defaultdict(list)
            for c in prior:
                angle = c.get("angle", "") or "general"
                by_angle[angle].append(c)

            for angle, findings in sorted(by_angle.items()):
                lines.append(f"[{angle}] ({len(findings)} findings):")
                # Show top findings by quality, one-line each
                top = sorted(
                    findings,
                    key=lambda x: x.get("composite_quality") or 0,
                    reverse=True,
                )[:15]  # Cap at 15 per angle to keep it compact
                for c in top:
                    q = c.get("composite_quality") or 0
                    fact = c.get("fact") or ""
                    lines.append(f"  [{c['id']}] (q={q:.2f}) {fact}")
                if len(findings) > 15:
                    lines.append(f"  ... and {len(findings) - 15} more findings")
                lines.append("")

        # ── Tier 1: Full detail for DELTA (new) findings ──
        # Group delta atoms by their parent chunk
        by_chunk: dict[int, list[dict]] = defaultdict(list)
        orphans: list[dict] = []
        for c in delta:
            pid = c.get("parent_id")
            if pid and pid > 0:
                by_chunk[pid].append(c)
            else:
                orphans.append(c)

        if delta:
            lines.append("=" * 60)
            lines.append("NEW FINDINGS (full detail)")
            lines.append("=" * 60)
            lines.append("")
        else:
            lines.append("=" * 60)
            lines.append("ALL FINDINGS (full detail)")
            lines.append("=" * 60)
            lines.append("")
            # No delta/prior split — show everything in full
            for c in conditions:
                pid = c.get("parent_id")
                if pid and pid > 0:
                    by_chunk[pid].append(c)
                else:
                    orphans.append(c)

        for chunk_id in sorted(by_chunk.keys()):
            atoms = by_chunk[chunk_id]
            chunk_text = self._get_chunk_text(chunk_id)

            lines.append(f"--- Source Passage (chunk {chunk_id}) ---")
            if chunk_text:
                lines.append(chunk_text)
            lines.append("")
            lines.append("Extracted findings:")

            for c in atoms:
                lines.append(self._describe_finding(c))
                lines.append("")

        # Orphan atoms (no chunk parent — legacy or merge children)
        if orphans:
            lines.append("--- Standalone Findings (no source passage) ---")
            for c in orphans:
                lines.append(self._describe_finding(c))
                lines.append("")

        # ------ Layer 3: Relationships and narrative chains ------
        edges = self._get_relationship_edges(atom_ids)
        chains = self._get_narrative_chains_for_atoms(atom_ids)

        if edges or chains:
            lines.append("=" * 60)
            lines.append("LAYER 3: RELATIONSHIPS AND NARRATIVE THREADS")
            lines.append("=" * 60)
            lines.append("")

        if edges:
            lines.append("RELATIONSHIPS BETWEEN FINDINGS:")
            for src, tgt, rel in edges:
                src_preview = cond_by_id[src]["fact"] if src in cond_by_id else f"[{src}]"
                tgt_preview = cond_by_id[tgt]["fact"] if tgt in cond_by_id else f"[{tgt}]"
                lines.append(
                    f"  [{src}] --{rel}--> [{tgt}]"
                )
                lines.append(f"    {src_preview}")
                lines.append(f"    → {tgt_preview}")
                lines.append("")

        if chains:
            lines.append("NARRATIVE THREADS (connected finding sequences):")
            for i, chain in enumerate(chains, 1):
                chain_facts = []
                for mid in chain:
                    if mid in cond_by_id:
                        chain_facts.append(
                            f"  {len(chain_facts) + 1}. [{mid}] "
                            f"{cond_by_id[mid]['fact']}"
                        )
                if chain_facts:
                    lines.append(f"Thread {i}:")
                    lines.extend(chain_facts)
                    lines.append("")

        # ------ Contradictions ------
        contradictions = [
            c for c in conditions if c.get("contradiction_flag")
        ]
        if contradictions:
            lines.append("CONTRADICTIONS DETECTED:")
            seen: set[tuple[int, int]] = set()
            for c in contradictions:
                partner = c.get("contradiction_partner", -1)
                pair = (min(c["id"], partner), max(c["id"], partner))
                if pair in seen or partner < 0:
                    continue
                seen.add(pair)
                partner_c = cond_by_id.get(partner)
                if partner_c:
                    lines.append(
                        f"  [{c['id']}] vs [{partner}]: "
                        f"{c['fact']} CONTRADICTS "
                        f"{partner_c['fact']}"
                    )
            lines.append("")

        # ------ Corpus health ------
        strong = sum(
            1 for c in conditions
            if (c.get("composite_quality") or 0) >= 0.6
        )
        moderate = sum(
            1 for c in conditions
            if 0.3 <= (c.get("composite_quality") or 0) < 0.6
        )
        weak = sum(
            1 for c in conditions
            if (c.get("composite_quality") or 0) < 0.3
        )
        awaiting = sum(
            1 for c in conditions
            if c.get("expansion_tool", "none") != "none"
            and not c.get("expansion_fulfilled")
        )
        lines.append(
            f"CORPUS HEALTH: {len(conditions)} findings. "
            f"{strong} strong, {moderate} moderate, {weak} weak. "
            f"{len(edges)} relationships mapped. "
            f"{len(chains)} narrative threads. "
            f"{len(contradictions)} contradictions. "
            f"{awaiting} awaiting enrichment."
        )

        # ------ Layer 4: Specialist thought briefing (budget-governed) --
        thought_rows = self.get_thoughts_for_thinker()
        if thought_rows:
            lines.append("")
            lines.append("=" * 60)
            lines.append("LAYER 4: SPECIALIST THOUGHTS AND INSIGHTS")
            lines.append("=" * 60)
            lines.append("")

            # Separate insights from thoughts
            insights = [t for t in thought_rows if "grounded_insight" in (t.get("strategy") or "")]
            thoughts = [t for t in thought_rows if "grounded_insight" not in (t.get("strategy") or "")]

            if insights:
                lines.append("EVIDENCE-GROUNDED INSIGHTS (safe for synthesis):")
                for ins in insights:
                    lines.append(
                        f"  [insight #{ins['id']}, angle={ins['angle']}]: "
                        f"{ins['fact']}"
                    )
                lines.append("")

            # Group thoughts by angle
            by_angle: dict[str, list[dict]] = defaultdict(list)
            for t in thoughts:
                by_angle[t.get("angle", "")].append(t)

            for angle_name, angle_thoughts in sorted(by_angle.items()):
                verdicts = [t for t in angle_thoughts if t["strategy"] == "arbitration_verdict"]
                specialists = [t for t in angle_thoughts if t["strategy"] != "arbitration_verdict"]

                lines.append(f"ANGLE: {angle_name} ({len(angle_thoughts)} thoughts)")
                if verdicts:
                    lines.append("  VERDICTS (arbitrated conclusions):")
                    for v in verdicts:
                        lines.append(
                            f"    [thought #{v['id']}, iter={v['iteration']}]: "
                            f"{v['fact']}"
                        )
                if specialists:
                    lines.append("  SPECIALIST ANALYSIS:")
                    for s in specialists:
                        depth_tag = f" depth={s['expansion_depth']}" if s.get("expansion_depth", 0) > 0 else ""
                        lines.append(
                            f"    [thought #{s['id']}, iter={s['iteration']}"
                            f"{depth_tag}]: {s['fact']}"
                        )
                lines.append("")

            lines.append(
                f"THOUGHT HEALTH: {len(thought_rows)} thoughts shown "
                f"({len(insights)} insights, "
                f"{sum(1 for t in thoughts if t['strategy'] == 'arbitration_verdict')} verdicts, "
                f"{sum(1 for t in thoughts if t['strategy'] == 'specialist_analysis')} specialist). "
                f"Budget: {len(by_angle)} angles."
            )

        return "\n".join(lines)

    def format_summary_for_maestro(self) -> str:
        """Format a compact structural summary for the maestro.

        The maestro has unrestricted SQL access to the full corpus via
        ``execute_flock_sql()``, so it does NOT need every finding's text
        in its instruction.  This method provides just enough orientation
        for the maestro to know what SQL operations to perform:

        - Total counts by processing status and row type
        - Unscored condition count (the maestro's primary job)
        - Expansion targets awaiting fulfilment
        - Per-angle statistics (count, avg quality)
        - Top contradictions and cluster info

        This keeps the maestro's context window lean, leaving room for
        SQL results and Flock LLM outputs.
        """
        lines: list[str] = []

        # ── Overall counts ──
        total = self.count()
        if total == 0:
            return "(empty corpus — no conditions to organise)"

        status_counts = self._status_snapshot()
        lines.append(f"CORPUS OVERVIEW: {total} total conditions")
        if status_counts:
            status_parts = [f"{s}={n}" for s, n in sorted(status_counts.items())]
            lines.append(f"  By processing_status: {', '.join(status_parts)}")

        # Row type breakdown
        try:
            rt_rows = self.conn.execute(
                "SELECT row_type, COUNT(*) FROM conditions "
                "WHERE consider_for_use = TRUE GROUP BY row_type "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
            if rt_rows:
                rt_parts = [f"{rt}={n}" for rt, n in rt_rows]
                lines.append(f"  By type: {', '.join(rt_parts)}")
        except Exception:
            pass

        # ── Unscored conditions (primary maestro task) ──
        try:
            unscored = self.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE score_version = 0 AND consider_for_use = TRUE "
                "AND row_type = 'finding'"
            ).fetchone()[0]
            if unscored:
                lines.append(f"\n⚠ UNSCORED: {unscored} findings need scoring (score_version=0)")
            else:
                lines.append("\n✓ All findings scored")
        except Exception:
            pass

        # ── Expansion targets ──
        try:
            exp_rows = self.conn.execute(
                "SELECT expansion_tool, COUNT(*), "
                "GROUP_CONCAT(CAST(id AS VARCHAR), ', ') "
                "FROM conditions "
                "WHERE expansion_tool != 'none' "
                "AND expansion_fulfilled = FALSE "
                "AND consider_for_use = TRUE "
                "GROUP BY expansion_tool"
            ).fetchall()
            if exp_rows:
                lines.append(f"\nEXPANSION TARGETS ({sum(r[1] for r in exp_rows)} pending):")
                for tool, cnt, ids in exp_rows:
                    lines.append(f"  {tool}: {cnt} conditions (ids: {ids})")
        except Exception:
            pass

        # ── Per-angle statistics ──
        try:
            angle_rows = self.conn.execute(
                "SELECT angle, COUNT(*), "
                "ROUND(AVG(composite_quality), 2), "
                "SUM(CASE WHEN score_version = 0 THEN 1 ELSE 0 END) "
                "FROM conditions "
                "WHERE consider_for_use = TRUE AND row_type = 'finding' "
                "GROUP BY angle ORDER BY COUNT(*) DESC"
            ).fetchall()
            if angle_rows:
                lines.append(f"\nANGLE BREAKDOWN ({len(angle_rows)} angles):")
                for angle, cnt, avg_q, unscored_n in angle_rows:
                    label = angle or "(no angle)"
                    unscored_note = f", {int(unscored_n)} unscored" if unscored_n else ""
                    lines.append(
                        f"  [{label}]: {cnt} findings, avg_quality={avg_q}{unscored_note}"
                    )
        except Exception:
            pass

        # ── Contradiction summary ──
        try:
            contra_count = self.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE contradiction_flag = TRUE AND consider_for_use = TRUE"
            ).fetchone()[0]
            lines.append(f"\nCONTRADICTIONS: {contra_count}")
        except Exception:
            pass

        # ── Cluster summary ──
        try:
            cluster_rows = self.conn.execute(
                "SELECT COUNT(DISTINCT cluster_id) FROM conditions "
                "WHERE cluster_id >= 0 AND consider_for_use = TRUE"
            ).fetchone()[0]
            lines.append(f"CLUSTERS: {cluster_rows}")
        except Exception:
            pass

        # ── Thought/insight counts ──
        try:
            thought_count = self.count_thoughts()
            if thought_count:
                lines.append(f"THOUGHTS: {thought_count} (immutable — do not modify)")
            insight_count = self.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE row_type = 'insight' AND consider_for_use = TRUE"
            ).fetchone()[0]
            if insight_count:
                lines.append(f"INSIGHTS: {insight_count}")
        except Exception:
            pass

        return "\n".join(lines)

    def format_for_synthesiser(self) -> str:
        """Format the corpus for the synthesiser — chunk-based layout.

        Groups findings by source chunk so the synthesiser can
        weave coherent paragraphs from related material.  Includes
        narrative chains as suggested structural threads.
        """
        conditions = self.get_for_synthesiser()
        if not conditions:
            return "(no findings)"

        atom_ids = {c["id"] for c in conditions}
        cond_by_id = {c["id"]: c for c in conditions}

        # Group atoms by parent chunk
        by_chunk: dict[int, list[dict]] = defaultdict(list)
        orphans: list[dict] = []
        for c in conditions:
            pid = c.get("parent_id")
            if pid and pid > 0:
                by_chunk[pid].append(c)
            else:
                orphans.append(c)

        lines: list[str] = [
            f"SYNTHESIS BRIEFING: {len(conditions)} findings from "
            f"{len(by_chunk)} source passages\n",
        ]

        # Chunk-based sections
        for chunk_id in sorted(by_chunk.keys()):
            atoms = by_chunk[chunk_id]
            chunk_text = self._get_chunk_text(chunk_id)

            lines.append(f"--- Source Passage (chunk {chunk_id}) ---")
            if chunk_text:
                lines.append(chunk_text)
            lines.append("")
            lines.append("Key findings for synthesis:")
            for c in atoms:
                url_part = f" [{c['source_url']}]" if c.get("source_url") else ""
                lines.append(f"  • [{c['id']}] {c['fact']}{url_part}")
            lines.append("")

        if orphans:
            lines.append("--- Additional Findings ---")
            for c in orphans:
                url_part = f" [{c['source_url']}]" if c.get("source_url") else ""
                lines.append(f"  • [{c['id']}] {c['fact']}{url_part}")
            lines.append("")

        # Narrative chains as suggested structure
        chains = self._get_narrative_chains_for_atoms(atom_ids)
        if chains:
            lines.append("SUGGESTED NARRATIVE THREADS:")
            for i, chain in enumerate(chains, 1):
                chain_items = []
                for mid in chain:
                    if mid in cond_by_id:
                        chain_items.append(
                            f"  {len(chain_items) + 1}. "
                            f"{cond_by_id[mid]['fact']}"
                        )
                if chain_items:
                    lines.append(f"Thread {i}:")
                    lines.extend(chain_items)
                    lines.append("")

        # Summary
        strong = sum(
            1 for c in conditions
            if (c.get("composite_quality") or 0) >= 0.6
        )
        contradictions = sum(
            1 for c in conditions
            if c.get("contradiction_flag")
        )
        lines.append(
            f"SYNTHESIS NOTES: {len(conditions)} findings. "
            f"{strong} strongly supported. "
            f"{len(chains)} narrative threads to weave. "
            f"{contradictions} contradictions to address."
        )

        return "\n".join(lines)

    def format_for_specialist(self, angle: str, user_query: str = "") -> str:
        """Format a FULL per-angle corpus view for a specialist thinker.

        Each specialist in the swarm receives a **complete**, untruncated
        view of the findings relevant to their assigned angle.  This is
        the swarm's corpus-partitioning mechanism — inspired by the
        deep-search-portal pattern of decomposing a query into targeted
        sub-probes.

        Instead of giving every specialist a truncated copy of the whole
        corpus, each specialist gets:

        1. **Full findings for their angle** — every finding tagged with
           this angle, rendered in full verbal prose via ``_describe_finding``.
        2. **Unassigned findings** — findings with no angle tag that may
           be relevant to any specialist.
        3. **Prior thoughts for this angle** — full text of prior
           specialist analysis, arbitration verdicts, and insights so the
           specialist can build on previous work.
        4. **Orientation** — a one-line count of what other angles exist
           in the corpus (the specialist doesn't need their data, just
           awareness that parallel investigation is happening).

        Because each specialist is spawned with a **clean context**, the
        per-angle slice is naturally smaller than the full corpus, making
        context overflow a non-issue for typical research corpora.

        Args:
            angle: The specialist's assigned research angle.
            user_query: The user's original query for context.
        """
        lines: list[str] = []

        # ── Query per-angle findings ──
        cols = [
            "id", "fact", "source_url", "confidence", "trust_score",
            "novelty_score", "specificity_score", "duplication_score",
            "fabrication_risk", "verification_status", "angle",
            "parent_id", "related_id", "expansion_depth", "iteration",
            "row_type", "consider_for_use",
            "composite_quality", "information_density", "cross_ref_boost",
            "relevance_score", "actionability_score",
            "cluster_id", "cluster_rank",
            "expansion_tool", "expansion_hint", "expansion_fulfilled",
            "contradiction_flag", "contradiction_partner",
            "staleness_penalty", "processing_status",
        ]
        angle_rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      novelty_score, specificity_score, duplication_score,
                      fabrication_risk, verification_status, angle,
                      parent_id, related_id, expansion_depth, iteration,
                      row_type, consider_for_use,
                      composite_quality, information_density, cross_ref_boost,
                      relevance_score, actionability_score,
                      cluster_id, cluster_rank,
                      expansion_tool, expansion_hint, expansion_fulfilled,
                      contradiction_flag, contradiction_partner,
                      staleness_penalty, processing_status
               FROM conditions
               WHERE consider_for_use = TRUE
                 AND row_type = 'finding'
                 AND angle = ?
               ORDER BY composite_quality DESC, novelty_score DESC""",
            [angle],
        ).fetchall()
        angle_findings = [dict(zip(cols, row)) for row in angle_rows]

        # ── Unassigned findings (no angle — potentially relevant) ──
        unassigned_rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      novelty_score, specificity_score, duplication_score,
                      fabrication_risk, verification_status, angle,
                      parent_id, related_id, expansion_depth, iteration,
                      row_type, consider_for_use,
                      composite_quality, information_density, cross_ref_boost,
                      relevance_score, actionability_score,
                      cluster_id, cluster_rank,
                      expansion_tool, expansion_hint, expansion_fulfilled,
                      contradiction_flag, contradiction_partner,
                      staleness_penalty, processing_status
               FROM conditions
               WHERE consider_for_use = TRUE
                 AND row_type = 'finding'
                 AND (angle IS NULL OR angle = '')
               ORDER BY composite_quality DESC""",
        ).fetchall()
        unassigned_findings = [dict(zip(cols, row)) for row in unassigned_rows]

        # ── Orientation: other angles ──
        other_angles = self.conn.execute(
            """SELECT angle, COUNT(*) FROM conditions
               WHERE row_type = 'finding' AND consider_for_use = TRUE
                 AND angle != ? AND angle IS NOT NULL AND angle != ''
               GROUP BY angle ORDER BY COUNT(*) DESC""",
            [angle],
        ).fetchall()

        # ── Prior thoughts for this angle ──
        prior_thoughts = self.get_thoughts_by_angle(angle)

        # ── Build the briefing ──
        lines.append(f"SPECIALIST BRIEFING: angle '{angle}'")
        if user_query:
            lines.append(f"USER QUERY: {user_query}")
        lines.append(f"  {len(angle_findings)} findings for your angle")
        lines.append(f"  {len(unassigned_findings)} unassigned findings")
        if other_angles:
            other_summary = ", ".join(f"{a} ({n})" for a, n in other_angles)
            lines.append(f"  Other angles under investigation: {other_summary}")
        lines.append("")

        # ── Full findings for this angle ──
        if angle_findings:
            lines.append("=" * 60)
            lines.append(f"FINDINGS FOR '{angle}' (full detail)")
            lines.append("=" * 60)
            lines.append("")
            for c in angle_findings:
                lines.append(self._describe_finding(c))
                lines.append("")

        # ── Unassigned findings ──
        if unassigned_findings:
            lines.append("=" * 60)
            lines.append("UNASSIGNED FINDINGS (potentially relevant)")
            lines.append("=" * 60)
            lines.append("")
            for c in unassigned_findings:
                lines.append(self._describe_finding(c))
                lines.append("")

        # ── Prior thoughts (full text) ──
        if prior_thoughts:
            lines.append("=" * 60)
            lines.append(f"PRIOR ANALYSIS FOR '{angle}' (build on this)")
            lines.append("=" * 60)
            lines.append("")
            for t in prior_thoughts:
                depth_tag = (
                    f", depth={t['expansion_depth']}"
                    if t.get("expansion_depth", 0) > 0 else ""
                )
                lines.append(
                    f"[thought #{t['id']}, strategy={t['strategy']}, "
                    f"iter={t['iteration']}{depth_tag}]:"
                )
                lines.append(t["fact"])
                lines.append("")

        # ── Corpus health for this angle ──
        strong = sum(
            1 for c in angle_findings
            if (c.get("composite_quality") or 0) >= 0.6
        )
        weak = sum(
            1 for c in angle_findings
            if (c.get("composite_quality") or 0) < 0.3
        )
        lines.append(
            f"ANGLE HEALTH: {len(angle_findings)} findings for '{angle}'. "
            f"{strong} strong, {weak} weak. "
            f"{len(prior_thoughts)} prior thoughts."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Unified ingestion (Flock atomisation)
    # ------------------------------------------------------------------

    _URL_RE = re.compile(r"\[(https?://[^\]]+)\]")
    _BARE_URL_RE = re.compile(r"(https?://[^\s)\]\"'>]+)")
    _CONF_RE = re.compile(r"\(confidence=([0-9.]+)\)")

    # Regex for RELATES lines in enhanced atomisation output
    _RELATES_RE = re.compile(
        r'RELATES:\s*(\d+)\s*->\s*(\d+)\s*:\s*(\w[\w_]*)'
    )

    # Valid narrative relationship types for lineage edges
    _NARRATIVE_REL_TYPES = frozenset({
        'causes', 'caused_by', 'supports', 'contradicts', 'extends',
        'temporal_sequence', 'part_of', 'example_of', 'mechanism_of',
    })

    def ingest_raw(
        self,
        raw_text: str,
        source_type: str = "researcher",
        source_ref: str = "",
        angle: str = "",
        iteration: int = 0,
        user_query: str = "",
    ) -> list[int]:
        """Ingest raw text via chunk-aware Flock atomisation.

        Preserves full lineage: raw → chunks → atoms → relationship edges.
        Every row in the corpus is a node in a DAG with traceable parents.

        Returns list of admitted atom condition IDs.
        """
        if not raw_text or not raw_text.strip():
            return []

        now = datetime.now(timezone.utc).isoformat()

        # ---- Layer 0: Store FULL raw text (no truncation) ----
        raw_id = self._next_id
        self._next_id += 1
        self.conn.execute(
            "INSERT INTO conditions "
            "(id, fact, source_type, source_ref, row_type, "
            "consider_for_use, created_at, iteration) "
            "VALUES (?, ?, ?, ?, 'raw', FALSE, ?, ?)",
            [raw_id, raw_text, source_type, source_ref, now, iteration],
        )

        # ---- Layer 1: Split into paragraph-level chunks ----
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if p.strip()]
        if not paragraphs:
            paragraphs = [raw_text.strip()]

        chunk_ids: list[int] = []
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
                "VALUES (?, ?, ?, ?, 'chunk', ?, FALSE, ?, ?, ?, ?)",
                [chunk_id, para, source_type, source_ref,
                 raw_id, now, iteration, seq,
                 angle or f"iteration_{iteration}"],
            )
            chunk_ids.append(chunk_id)

        # ---- Layer 2: Atomise each chunk with enhanced prompt ----
        all_ids: list[int] = []
        for chunk_idx, chunk_id in enumerate(chunk_ids):
            chunk_text = paragraphs[chunk_idx]
            chunk_atom_ids = self._atomise_chunk(
                chunk_text, chunk_id, source_type, source_ref,
                angle, iteration, user_query,
            )
            all_ids.extend(chunk_atom_ids)

        # Mark the raw row as atomised
        self.conn.execute(
            "UPDATE conditions SET "
            "obsolete_reason = 'atomised into ' || ? || ' findings' "
            "WHERE id = ? AND row_type = 'raw'",
            [str(len(all_ids)), raw_id],
        )

        logger.info(
            "Flock atomisation: %d atoms from %d chunks, %d chars "
            "(source=%s)",
            len(all_ids), len(chunk_ids), len(raw_text), source_type,
        )
        return all_ids

    def _atomise_chunk(
        self,
        chunk_text: str,
        chunk_id: int,
        source_type: str,
        source_ref: str,
        angle: str,
        iteration: int,
        user_query: str,
    ) -> list[int]:
        """Atomise a single chunk into findings + relationship edges.

        Uses an enhanced prompt that extracts facts relevant to the
        research query AND identifies causal/temporal/logical
        relationships between them.  All in a single LLM call.

        Every atom gets ``parent_id = chunk_id`` for lineage.
        Relationship edges stored as rows linking atom pairs.
        """
        query_clause = (
            f"Research query: {user_query}\n\n" if user_query else ""
        )
        try:
            atomised = self._http_complete(
                "You are a research analyst. Extract findings and their "
                "relationships from the text below.\n\n"
                + query_clause
                + "Rules:\n"
                "- Extract ONLY facts relevant to the research query "
                "(skip off-topic, promotional, or meta-commentary material)\n"
                "- One fact per FACT: line\n"
                "- Preserve ALL specific data: names, numbers, dates, "
                "prices, URLs\n"
                "- If a URL is associated with a fact, append it as "
                "[URL] at the end of the line\n"
                "- If the text expresses confidence/uncertainty, append "
                "(confidence=X.X) where X.X is 0.0-1.0\n"
                "- After all facts, list relationships between them "
                "using RELATES: lines\n"
                "- RELATES connects facts by their line number (1-based)\n"
                "- If the text is a single atomic fact already, return "
                "one FACT: line and no RELATES:\n\n"
                "Relationship types: causes, caused_by, supports, "
                "contradicts, extends, temporal_sequence, part_of, "
                "example_of, mechanism_of\n\n"
                "Output format (follow exactly):\n"
                "FACT: [statement] [URL] (confidence=X.X)\n"
                "FACT: [statement]\n"
                "RELATES: 1 -> 2 : causes\n"
                "RELATES: 2 -> 3 : supports\n\n"
                "Text to analyse:\n" + chunk_text,
                caller="ingest_atomise",
            )
        except Exception:
            logger.warning(
                "Flock atomisation failed -- falling back to raw "
                "text as single condition",
                exc_info=True,
            )
            atomised = chunk_text

        if not atomised or not atomised.strip():
            atomised = chunk_text

        # ---- Parse FACT lines ----
        atom_ids: list[int] = []
        relates_lines: list[str] = []

        for line in atomised.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Collect RELATES lines for later processing
            if stripped.upper().startswith("RELATES:"):
                relates_lines.append(stripped)
                continue

            # Strip FACT: prefix if present
            if stripped.upper().startswith("FACT:"):
                stripped = stripped[5:].strip()

            # Strip bullet/number prefixes
            stripped = re.sub(
                r'^\s*(?:\d+[.)\]]\s*|[-*\u2022]\s+)', '', stripped,
            ).strip()
            if not stripped:
                continue

            # Reject junk atoms
            if self._is_junk_atom(stripped):
                logger.debug("Rejected junk atom: %.80s", stripped)
                continue

            # Extract URL
            url_match = self._URL_RE.search(stripped)
            if url_match:
                source_url = url_match.group(1)
                stripped = self._URL_RE.sub("", stripped).strip()
            else:
                bare_match = self._BARE_URL_RE.search(stripped)
                if bare_match:
                    source_url = bare_match.group(1).rstrip(".,;:")
                else:
                    source_url = ""

            # Extract confidence
            conf_match = self._CONF_RE.search(stripped)
            try:
                confidence = (
                    max(0.0, min(1.0, float(conf_match.group(1))))
                    if conf_match else 0.5
                )
            except (ValueError, TypeError):
                confidence = 0.5
            if conf_match:
                stripped = self._CONF_RE.sub("", stripped).strip()

            if not stripped:
                continue

            # Dedup: skip exact-match facts already in the corpus.
            # Exclude row_type='raw' and 'chunk' so we don't match rows
            # just inserted earlier in this method.
            existing = self.conn.execute(
                "SELECT id FROM conditions WHERE fact = ? "
                "AND row_type NOT IN ('raw', 'chunk') LIMIT 1",
                [stripped],
            ).fetchone()
            if existing:
                logger.debug(
                    "Dedup: skipping exact duplicate (existing id=%d): %.80s",
                    existing[0], stripped,
                )
                continue

            # Semantic dedup: LLM-powered via compute_duplications()
            # downstream. We deliberately do NOT pre-filter by keyword
            # overlap here — word overlap is a bag-of-words heuristic
            # that destroys meaning.  Two findings can share 85% of
            # their vocabulary while making opposite claims (e.g.
            # "mTOR signals growth when resources are abundant" vs
            # "...when resources are scarce").  The downstream
            # compute_duplications() uses LLM comparison to detect
            # true semantic duplicates accurately.

            # Insert atom with parent_id → chunk for lineage
            cid = self._next_id
            self._next_id += 1
            ts = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_url, source_type, source_ref,
                    confidence, angle, expansion_depth,
                    parent_id, created_at, iteration)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
                [
                    cid, stripped, source_url, source_type,
                    source_ref, confidence,
                    angle or f"iteration_{iteration}",
                    chunk_id,
                    ts, iteration,
                ],
            )
            atom_ids.append(cid)

        # ---- Parse RELATES lines into relationship edges ----
        for rel_line in relates_lines:
            m = self._RELATES_RE.search(rel_line)
            if not m:
                continue
            src_idx = int(m.group(1)) - 1  # 0-based
            tgt_idx = int(m.group(2)) - 1
            rel_type = m.group(3).lower()

            if rel_type not in self._NARRATIVE_REL_TYPES:
                continue
            if not (0 <= src_idx < len(atom_ids)
                    and 0 <= tgt_idx < len(atom_ids)):
                continue

            rel_id = self._next_id
            self._next_id += 1
            self.conn.execute(
                "INSERT INTO conditions "
                "(id, fact, row_type, parent_id, related_id, "
                "relationship_score, consider_for_use, iteration) "
                "VALUES (?, ?, ?, ?, ?, 1.0, FALSE, ?)",
                [rel_id, rel_type, rel_type,
                 atom_ids[src_idx], atom_ids[tgt_idx], iteration],
            )

        return atom_ids

    # ------------------------------------------------------------------
    # Gossip-based synthesis
    # ------------------------------------------------------------------

    def synthesise(self, user_query: str) -> str:
        """Produce a Flock synthesis of the corpus.
        Uses gossip for large corpora (above GOSSIP_THRESHOLD)."""
        conditions = self.get_for_synthesiser()
        if not conditions:
            return "(no findings)"

        if len(conditions) <= GOSSIP_THRESHOLD:
            return self._synthesise_single(conditions, user_query)
        return self._synthesise_gossip(conditions, user_query)

    def _synthesise_single(
        self, conditions: list[dict], user_query: str,
    ) -> str:
        """Single-pass synthesis using narrative-structured input."""
        corpus_text = self.format_for_synthesiser()
        return self._http_complete(
            "You are a research synthesiser. You receive findings "
            "organised by SOURCE PASSAGE — each passage preserves "
            "the original tone, hedging, and nuance of the author. "
            "Produce a comprehensive, well-structured report. "
            "Rules: "
            "- Weave findings into flowing narrative paragraphs, "
            "not bullet-point lists "
            "- Follow the SUGGESTED NARRATIVE THREADS to structure "
            "your report — they show causal and temporal chains "
            "- Include ALL facts, names, numbers, URLs "
            "- Preserve the author's nuance and hedging where it "
            "matters (e.g., 'may', 'preliminary evidence suggests') "
            "- Cross-reference findings from different source "
            "passages "
            "- Structure with clear headings "
            "- Cite source URLs inline "
            "- Do NOT add disclaimers or moralising "
            "- Address contradictions by presenting both sides"
            "\nUser query: " + user_query
            + "\n\n" + corpus_text,
            caller="synthesise_single",
        )

    def _synthesise_gossip(
        self, conditions: list[dict], user_query: str,
    ) -> str:
        """Gossip-style synthesis for large corpora.

        Phase 1: Synthesise each chunk-group independently, using
                 the source passage text for narrative context.
        Phase 2: Each group summary refined with peer awareness.
        Phase 3: Queen merges all into final report, guided by
                 narrative chains.
        """
        # Group by parent chunk for narrative coherence
        by_chunk: dict[int, list[dict]] = defaultdict(list)
        orphans: list[dict] = []
        for c in conditions:
            pid = c.get("parent_id")
            if pid and pid > 0:
                by_chunk[pid].append(c)
            else:
                orphans.append(c)

        # Phase 1: per-chunk synthesis (workers)
        chunk_summaries: dict[str, str] = {}

        for chunk_id, atoms in by_chunk.items():
            chunk_text = self._get_chunk_text(chunk_id)
            facts_text = "\n".join(
                f"- [{c['id']}] {c['fact']}"
                + (f" [{c['source_url']}]" if c["source_url"] else "")
                for c in atoms
            )
            group_label = f"chunk_{chunk_id}"
            summary = self._http_complete(
                "You are a synthesis worker. You receive a "
                "SOURCE PASSAGE (the original text with its "
                "tone, hedging, and nuance) plus EXTRACTED "
                "FINDINGS from that passage. Write a focused "
                "narrative section that preserves the author's "
                "voice and nuance. Include ALL facts, names, "
                "numbers, URLs. This section will be merged "
                "with other sections. Stay under 6000 chars."
                "\nSource passage:\n" + (chunk_text or "(no source)")
                + "\n\nExtracted findings:\n" + facts_text
                + "\nUser query: " + user_query,
                caller="gossip_phase1_worker",
            )
            chunk_summaries[group_label] = summary

        # Orphan atoms as a single group
        if orphans:
            orphan_text = "\n".join(
                f"- [{c['id']}] {c['fact']}"
                + (f" [{c['source_url']}]" if c["source_url"] else "")
                for c in orphans
            )
            chunk_summaries["standalone"] = self._http_complete(
                "You are a synthesis worker. Synthesise these "
                "standalone findings into a focused narrative "
                "section. Include ALL facts, names, numbers, "
                "URLs. Stay under 6000 chars."
                "\nFindings:\n" + orphan_text
                + "\nUser query: " + user_query,
                caller="gossip_phase1_worker",
            )

        logger.info(
            "Gossip Phase 1: %d chunk summaries produced",
            len(chunk_summaries),
        )

        # Phase 2: gossip refinement
        phase1_summaries = dict(chunk_summaries)
        for group in list(chunk_summaries.keys()):
            peer_text = "\n\n".join(
                f"### {g}\n{s}"
                for g, s in phase1_summaries.items()
                if g != group
            )
            refined = self._http_complete(
                "You produced a section summary. Now you see "
                "summaries from peer workers who processed "
                "other source passages. Cross-reference your "
                "findings with peers. Note agreements and "
                "contradictions. Incorporate complementary "
                "findings. Remove redundancy. Preserve ALL "
                "unique findings and the original author's "
                "nuance. Stay under 6000 characters."
                "\nYour section: " + group
                + "\nUser query: " + user_query
                + "\n\nYour current summary:\n"
                + chunk_summaries[group]
                + "\n\nPeer summaries:\n" + peer_text,
                caller="gossip_phase2_refine",
            )
            chunk_summaries[group] = refined

        logger.info("Gossip Phase 2: refinement complete")

        # Phase 3: queen merges, guided by narrative chains
        atom_ids = {c["id"] for c in conditions}
        chains = self._get_narrative_chains_for_atoms(atom_ids)
        cond_by_id = {c["id"]: c for c in conditions}
        chain_guidance = ""
        if chains:
            chain_lines = ["NARRATIVE THREADS (use these to structure your report):"]
            for i, chain in enumerate(chains, 1):
                items = []
                for mid in chain:
                    if mid in cond_by_id:
                        items.append(
                            f"  {len(items) + 1}. {cond_by_id[mid]['fact']}"
                        )
                if items:
                    chain_lines.append(f"Thread {i}:")
                    chain_lines.extend(items)
            chain_guidance = "\n".join(chain_lines) + "\n\n"

        all_summaries = "\n\n".join(
            f"## {group}\n{summary}"
            for group, summary in sorted(chunk_summaries.items())
        )
        final = self._http_complete(
            "You are the queen synthesiser. Merge these section "
            "reports from specialist workers into a single "
            "comprehensive research report. "
            "Rules: "
            "- Weave sections into flowing narrative, not "
            "bullet-point lists "
            "- Follow the NARRATIVE THREADS to structure your "
            "report — they show causal and temporal chains "
            "- Include ALL findings from all sections "
            "- Preserve the original authors' tone and nuance "
            "- Resolve contradictions by presenting both sides "
            "- Structure with clear headings and sub-headings "
            "- Cite source URLs inline "
            "- Do NOT add disclaimers or moralising "
            "- Prioritise specificity: names, numbers, dates, "
            "URLs "
            "- The report should be self-contained and readable "
            "by someone with no prior context"
            "\nUser query: " + user_query
            + "\n\n" + chain_guidance + all_summaries,
            caller="gossip_phase3_queen",
        )

        logger.info("Gossip Phase 3: queen synthesis complete")
        return final

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
