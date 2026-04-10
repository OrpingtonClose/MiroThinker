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

import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional
import duckdb

from models.atomic_condition import AtomicCondition
from utils.flock_proxy import start_flock_proxy

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

                -- Core metadata (set at admission)
                angle TEXT DEFAULT '',
                parent_id INTEGER,
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

                -- Composite & derived scores (computed by algorithms)
                composite_quality FLOAT DEFAULT -1.0,
                information_density FLOAT DEFAULT -1.0,
                cross_ref_boost FLOAT DEFAULT 0.0,

                -- Enum-style processing columns
                processing_status TEXT DEFAULT 'raw',
                    -- raw → scored → analysed → clustered → ready
                expansion_strategy TEXT DEFAULT 'none',
                    -- none | exa_search | brave_deep | kagi_enrich
                expansion_hint TEXT DEFAULT '',
                cluster_id INTEGER DEFAULT -1,
                cluster_rank INTEGER DEFAULT 0,
                contradiction_flag BOOLEAN DEFAULT FALSE,
                contradiction_partner INTEGER DEFAULT -1,
                staleness_penalty FLOAT DEFAULT 0.0
            )
        """)

        # -- Contradiction pairs table --
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS contradiction_pairs (
                condition_a INTEGER,
                condition_b INTEGER,
                contradiction_score FLOAT DEFAULT 0.0,
                description TEXT DEFAULT '',
                PRIMARY KEY (condition_a, condition_b)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS raw_ingestion (
                id INTEGER PRIMARY KEY,
                raw_text TEXT NOT NULL,
                source_type TEXT DEFAULT '',
                source_ref TEXT DEFAULT '',
                ingested_at TEXT DEFAULT '',
                atomised BOOLEAN DEFAULT FALSE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS similarity_flags (
                condition_a INTEGER,
                condition_b INTEGER,
                similarity_score FLOAT DEFAULT 0.0,
                relationship TEXT DEFAULT '',
                PRIMARY KEY (condition_a, condition_b)
            )
        """)

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
                confidence, verification_status, angle,
                parent_id, strategy,
                expansion_depth, created_at, iteration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                cid,
                fact,
                condition.source_url,
                getattr(condition, "source_type", "researcher"),
                getattr(condition, "source_ref", ""),
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

    def score_new_conditions(self, user_query: str = "") -> int:
        """Score all unscored conditions using Flock. Returns count."""
        unscored = self.conn.execute(
            "SELECT id, fact, source_url FROM conditions "
            "WHERE scored_at = ''"
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
        """Score a single condition across all gradient dimensions."""
        url_text = source_url if source_url else "(no URL)"

        confidence = self._parse_float(self._flock_complete(
            "Rate the confidence of this research finding on a scale "
            "from 0.0 to 1.0. Consider: Is it specific? Does it cite "
            "sources? Is the language hedged or definitive? Return "
            "ONLY a decimal number, nothing else. "
            "Finding: " + fact,
            caller="score_confidence",
        ), 0.5)

        trust = self._parse_float(self._flock_complete(
            "Rate the trustworthiness of this source URL on a scale "
            "from 0.0 to 1.0. Academic/government = high, established "
            "news = medium-high, forums/social = medium, unknown/no "
            "URL = low. Return ONLY a decimal number. "
            "URL: " + url_text,
            caller="score_trust",
        ), 0.5)

        specificity = self._parse_float(self._flock_complete(
            "Rate how specific and concrete this finding is on 0.0 to "
            "1.0. 1.0 = contains exact names, numbers, dates, URLs. "
            "0.0 = vague generality with no concrete data. Return "
            "ONLY a decimal number. "
            "Finding: " + fact,
            caller="score_specificity",
        ), 0.5)

        fabrication = self._parse_float(self._flock_complete(
            "Rate the risk that this finding is "
            "fabricated/hallucinated on 0.0 to 1.0. "
            "0.0 = clearly grounded in real sources. "
            "1.0 = likely made up, no verifiable details. Consider: "
            "Does it cite a real URL? Are the claims verifiable? "
            "Return ONLY a decimal number. "
            "Finding: " + fact + " Source: " + url_text,
            caller="score_fabrication",
        ), 0.0)

        relevance = 0.5
        if user_query:
            relevance = self._parse_float(self._flock_complete(
                "Rate how relevant this finding is to the user query "
                "on 0.0 to 1.0. Return ONLY a decimal number. "
                "Query: " + user_query + " Finding: " + fact,
                caller="score_relevance",
            ), 0.5)

        novelty = self._parse_float(self._flock_complete(
            "Rate how novel this finding is on 0.0 to 1.0. "
            "1.0 = completely new information not commonly known. "
            "0.0 = widely known, obvious, or trivial. "
            "Return ONLY a decimal number. "
            "Finding: " + fact,
            caller="score_novelty",
        ), 0.5)

        actionability = self._parse_float(self._flock_complete(
            "Rate how actionable this finding is on 0.0 to 1.0. "
            "1.0 = directly usable, contains specific steps or data. "
            "0.0 = purely informational with no actionable content. "
            "Return ONLY a decimal number. "
            "Finding: " + fact,
            caller="score_actionability",
        ), 0.5)

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

        Updates duplication_score on each row and populates
        similarity_flags. Returns count of pairs evaluated.
        """
        new_ids = self.conn.execute(
            "SELECT id FROM conditions "
            "WHERE duplication_score < 0.0 AND scored_at != ''"
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
                "WHERE id != ? AND id < ?",
                [new_id, new_id],
            ).fetchall()

            max_sim = 0.0
            for other_id, other_fact in others:
                try:
                    sim = self._compare_pair(
                        new_id, new_fact, other_id, other_fact,
                    )
                    max_sim = max(max_sim, sim)
                    pair_count += 1
                except Exception:
                    logger.warning(
                        "Flock dedup failed for pair (%d, %d)",
                        new_id, other_id,
                        exc_info=True,
                    )

            self.conn.execute(
                "UPDATE conditions SET duplication_score = ? "
                "WHERE id = ?",
                [max_sim, new_id],
            )

        return pair_count

    def _compare_pair(
        self, id_a: int, fact_a: str, id_b: int, fact_b: str,
    ) -> float:
        """Compare two facts for semantic similarity via Flock."""
        sim_raw = self._flock_complete(
            "Are these two statements saying essentially the "
            "same thing? Return a similarity score: 0.0 "
            "(completely different) to 1.0 (identical meaning). "
            "Return ONLY a decimal number. "
            "A: " + fact_a + " B: " + fact_b,
            caller="compare_pair",
        )
        sim = self._parse_float(sim_raw, 0.0)
        sim = min(max(sim, 0.0), 1.0)

        if sim > 0.3:
            rel = (
                "duplicate" if sim > 0.85
                else "confirms" if sim > 0.5
                else "related"
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO similarity_flags "
                "(condition_a, condition_b, "
                "similarity_score, relationship) "
                "VALUES (?, ?, ?, ?)",
                [id_a, id_b, sim, rel],
            )
        return sim

    # ------------------------------------------------------------------
    # Algorithm battery — each is a small, composable processing step
    # ------------------------------------------------------------------

    def compute_composite_quality(self) -> int:
        """Compute a weighted composite quality score for all scored
        conditions.  Pure SQL — no LLM calls.

        Formula: 0.25*confidence + 0.20*relevance + 0.15*trust
                 + 0.15*novelty + 0.10*specificity + 0.10*actionability
                 - 0.15*fabrication_risk - staleness_penalty
                 + cross_ref_boost

        Recomputes every battery run so that changes to staleness_penalty
        and cross_ref_boost from earlier algorithms are reflected.

        Returns number of conditions updated.
        """
        # Count how many will be updated before running
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE scored_at != ''"
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
            """
        )
        logger.info("Composite quality: updated %d conditions", count)
        return count

    def apply_quality_gate(self, threshold: float = 0.25) -> int:
        """Mark low-quality conditions for expansion instead of
        passing them to the swarm.  Pure SQL.

        Conditions below *threshold* get expansion_strategy set to
        'exa_search' so the researcher's next iteration can enrich
        them rather than starting from scratch.

        Returns number of conditions flagged for expansion.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE composite_quality >= 0.0 AND composite_quality < ? "
            "AND expansion_strategy = 'none'",
            [threshold],
        ).fetchone()[0]
        if count == 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET expansion_strategy = 'exa_search',
                   expansion_hint = 'Low composite quality ('
                       || ROUND(composite_quality, 2) || ') — '
                       || 'search for more specific data on: '
                       || SUBSTR(fact, 1, 120)
               WHERE composite_quality >= 0.0
                 AND composite_quality < ?
                 AND expansion_strategy = 'none'
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
        source URL.  These get expansion_strategy = 'brave_deep' to
        trigger a deep search for concrete data.  Pure SQL.

        Returns number flagged.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE specificity_score < 0.3 "
            "AND (source_url = '' OR source_url IS NULL) "
            "AND expansion_strategy = 'none' AND scored_at != ''"
        ).fetchone()[0]
        if count == 0:
            return 0
        self.conn.execute(
            """UPDATE conditions
               SET expansion_strategy = 'brave_deep',
                   expansion_hint = 'Vague finding — search for '
                       || 'specific names/numbers/dates: '
                       || SUBSTR(fact, 1, 120)
               WHERE specificity_score < 0.3
                 AND (source_url = '' OR source_url IS NULL)
                 AND expansion_strategy = 'none'
                 AND scored_at != ''
            """
        )
        if count:
            logger.info(
                "Specificity gate: flagged %d vague conditions "
                "for deep search", count,
            )
        return count

    def compute_cross_ref_boost(self) -> int:
        """Boost conditions that are confirmed by multiple angles.

        If a condition has similarity_flags entries with relationship
        = 'confirms' from different angles, its cross_ref_boost
        increases.  Pure SQL.

        Returns number of conditions boosted.
        """
        # Find conditions with confirming pairs from different angles
        self.conn.execute(
            """UPDATE conditions SET cross_ref_boost = (
                   SELECT COALESCE(
                       MIN(0.2, COUNT(*) * 0.05), 0.0
                   )
                   FROM similarity_flags sf
                   JOIN conditions c2
                       ON (sf.condition_b = c2.id
                           OR sf.condition_a = c2.id)
                   WHERE (sf.condition_a = conditions.id
                          OR sf.condition_b = conditions.id)
                     AND sf.relationship IN ('confirms', 'related')
                     AND c2.angle != conditions.angle
                     AND c2.id != conditions.id
               )
               WHERE scored_at != ''
            """
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE cross_ref_boost > 0.0"
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
               SET staleness_penalty = MIN(
                   0.10,
                   0.02 * (? - iteration)
               )
               WHERE iteration < ?
                 AND scored_at != ''
            """,
            [current_iteration, current_iteration],
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE staleness_penalty > 0.0"
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
        # Extract domain from source_url and count occurrences
        # DuckDB doesn't have a built-in domain extractor, so we
        # use string functions to approximate it.
        self.conn.execute(
            """UPDATE conditions
               SET cross_ref_boost = cross_ref_boost + CASE
                   WHEN source_url != '' AND source_url IS NOT NULL THEN
                       -- Unique domain bonus: 0.05 / count of same domain
                       0.05 / GREATEST(1, (
                           SELECT COUNT(*)
                           FROM conditions c2
                           WHERE c2.source_url != ''
                             AND SPLIT_PART(
                                 REPLACE(REPLACE(c2.source_url,
                                     'https://', ''), 'http://', ''),
                                 '/', 1
                             ) = SPLIT_PART(
                                 REPLACE(REPLACE(conditions.source_url,
                                     'https://', ''), 'http://', ''),
                                 '/', 1
                             )
                       ))
                   ELSE 0.0
               END
               WHERE scored_at != ''
                 AND source_url != ''
                 AND source_url IS NOT NULL
            """
        )
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE scored_at != '' AND source_url != '' "
            "AND source_url IS NOT NULL"
        ).fetchone()[0]
        if count:
            logger.info(
                "Source diversity: adjusted %d conditions", count,
            )
        return count

    def detect_contradictions(self) -> int:
        """Detect contradicting condition pairs via Flock LLM.

        Only checks pairs that are already flagged as 'related' in
        similarity_flags (sim > 0.3) — these are the most likely
        candidates for contradiction.

        Returns number of contradiction pairs found.
        """
        candidates = self.conn.execute(
            """SELECT sf.condition_a, c1.fact,
                      sf.condition_b, c2.fact
               FROM similarity_flags sf
               JOIN conditions c1 ON sf.condition_a = c1.id
               JOIN conditions c2 ON sf.condition_b = c2.id
               WHERE sf.relationship = 'related'
                 AND sf.similarity_score BETWEEN 0.3 AND 0.7
                 AND c1.contradiction_flag = FALSE
                 AND c2.contradiction_flag = FALSE
            """
        ).fetchall()
        if not candidates:
            return 0

        contradiction_count = 0
        for id_a, fact_a, id_b, fact_b in candidates:
            try:
                result = self._flock_complete(
                    "Do these two statements contradict each other? "
                    "Return a contradiction score: 0.0 (no "
                    "contradiction) to 1.0 (direct contradiction). "
                    "Return ONLY a decimal number. "
                    "A: " + fact_a + " B: " + fact_b,
                    caller="detect_contradictions",
                )
                score = self._parse_float(result, 0.0)
                if score > 0.6:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO contradiction_pairs "
                        "(condition_a, condition_b, "
                        "contradiction_score, description) "
                        "VALUES (?, ?, ?, 'auto-detected')",
                        [id_a, id_b, score],
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
            except Exception:
                logger.warning(
                    "Contradiction check failed for (%d, %d)",
                    id_a, id_b, exc_info=True,
                )

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

        Returns number of conditions scored.
        """
        unscored = self.conn.execute(
            "SELECT id, fact FROM conditions "
            "WHERE information_density < 0.0 AND scored_at != ''"
        ).fetchall()
        if not unscored:
            return 0

        for cid, fact in unscored:
            try:
                result = self._flock_complete(
                    "Rate the information density of this text on "
                    "0.0 to 1.0. High density = many concrete facts, "
                    "numbers, names, URLs per sentence. Low density "
                    "= verbose, repetitive, or filler text. Return "
                    "ONLY a decimal number. "
                    "Text: " + fact,
                    caller="info_density",
                )
                density = self._parse_float(result, 0.5)
                self.conn.execute(
                    "UPDATE conditions "
                    "SET information_density = ? WHERE id = ?",
                    [density, cid],
                )
            except Exception:
                logger.warning(
                    "Info density scoring failed for #%d", cid,
                    exc_info=True,
                )

        logger.info(
            "Information density: scored %d conditions",
            len(unscored),
        )
        return len(unscored)

    def cluster_conditions(self) -> int:
        """Cluster similar conditions and rank within each cluster.

        Uses the existing similarity_flags to build clusters via a
        simple union-find approach.  Within each cluster, the
        condition with the highest composite_quality becomes rank 0
        (representative).  Others get rank 1+ (supplementary).

        The swarm can then process only cluster representatives
        (rank 0) and fold in supplementary context, dramatically
        reducing LLM calls.

        Returns number of clusters formed.
        """
        # Get all conditions and their similarity edges
        conditions = self.conn.execute(
            "SELECT id FROM conditions WHERE scored_at != ''"
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

        # Build clusters from high-similarity pairs
        edges = self.conn.execute(
            "SELECT condition_a, condition_b "
            "FROM similarity_flags "
            "WHERE similarity_score > 0.5"
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

        For pairs with duplication_score > 0.85, asks Flock to merge
        them into a single stronger condition.  The merged condition
        inherits the best scores from both parents.

        Returns number of merges performed.
        """
        dupes = self.conn.execute(
            """SELECT sf.condition_a, c1.fact, c1.source_url,
                      c1.composite_quality AS quality_a,
                      sf.condition_b, c2.fact, c2.source_url,
                      c2.composite_quality AS quality_b
               FROM similarity_flags sf
               JOIN conditions c1 ON sf.condition_a = c1.id
               JOIN conditions c2 ON sf.condition_b = c2.id
               WHERE sf.relationship = 'duplicate'
                 AND sf.similarity_score > 0.85
                 AND c1.processing_status != 'merged'
                 AND c2.processing_status != 'merged'
            """
        ).fetchall()
        if not dupes:
            return 0

        merge_count = 0
        for id_a, fact_a, url_a, q_a, id_b, fact_b, url_b, q_b in dupes:
            try:
                merged_fact = self._flock_complete(
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

                # Keep the higher-scored condition, merge away the lower
                if (q_b or 0.0) > (q_a or 0.0):
                    survivor_id, merged_id = id_b, id_a
                    best_url = url_b or url_a
                else:
                    survivor_id, merged_id = id_a, id_b
                    best_url = url_a or url_b

                self.conn.execute(
                    "UPDATE conditions SET fact = ?, "
                    "source_url = CASE "
                    "  WHEN source_url = '' THEN ? "
                    "  ELSE source_url END "
                    "WHERE id = ?",
                    [merged_fact.strip(), best_url, survivor_id],
                )
                self.conn.execute(
                    "UPDATE conditions "
                    "SET processing_status = 'merged', "
                    "    duplication_score = 1.0 "
                    "WHERE id = ?",
                    [merged_id],
                )
                merge_count += 1
            except Exception:
                logger.warning(
                    "Redundancy merge failed for (%d, %d)",
                    id_a, id_b, exc_info=True,
                )

        if merge_count:
            logger.info(
                "Redundancy compression: merged %d pairs",
                merge_count,
            )
        return merge_count

    def mark_ready(self) -> int:
        """Mark all fully-processed conditions as 'ready' for the
        swarm.  Pure SQL.

        A condition is ready if:
        - It's been scored (scored_at != '')
        - It hasn't been merged away (processing_status != 'merged')
        - It's not flagged for expansion (expansion_strategy = 'none')

        Returns number marked ready.
        """
        count = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE scored_at != '' "
            "AND processing_status NOT IN ('merged', 'ready') "
            "AND expansion_strategy = 'none'"
        ).fetchone()[0]
        self.conn.execute(
            """UPDATE conditions
               SET processing_status = 'ready'
               WHERE scored_at != ''
                 AND processing_status NOT IN ('merged', 'ready')
                 AND expansion_strategy = 'none'
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
            """SELECT id, fact, expansion_strategy, expansion_hint,
                      specificity_score, composite_quality
               FROM conditions
               WHERE expansion_strategy != 'none'
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
                       WHERE c2.cluster_id = conditions.cluster_id)
                       AS cluster_size
               FROM conditions
               WHERE cluster_rank = 0
                 AND processing_status NOT IN ('merged')
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
        7. Information density (Flock LLM)
        8. Contradiction detection (Flock LLM)
        9. Clustering (SQL)
        10. Redundancy compression (Flock LLM)
        11. Mark ready (SQL)
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
        results["composite_quality"] = self._trace_algorithm(
            "composite_quality", self.compute_composite_quality,
        )
        results["quality_gate"] = self._trace_algorithm(
            "quality_gate", self.apply_quality_gate,
        )
        results["specificity_gate"] = self._trace_algorithm(
            "specificity_gate", self.apply_specificity_gate,
        )
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
            "WHERE expansion_strategy != 'none'"
        ).fetchone()[0]
        merged = self.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE processing_status = 'merged'"
        ).fetchone()[0]
        cluster_reps = len(self.get_cluster_representatives())

        logger.info(
            "Algorithm battery complete in %.0fms: %d total, %d ready, "
            "%d for expansion, %d merged, %d cluster reps "
            "(swarm will process %d instead of %d)",
            battery_duration_ms,
            total, ready, expansion, merged, cluster_reps,
            cluster_reps, total,
        )

        results["total"] = total
        results["ready"] = ready
        results["expansion_targets"] = expansion
        results["merged"] = merged
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
        """Thinker sees almost everything -- only excludes near-exact
        duplicates so it can reason about gaps."""
        rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      novelty_score, specificity_score, duplication_score,
                      fabrication_risk, verification_status, angle,
                      parent_id, expansion_depth, iteration
               FROM conditions
               WHERE duplication_score < 0.85
               ORDER BY novelty_score DESC, confidence DESC, id ASC"""
        ).fetchall()
        cols = [
            "id", "fact", "source_url", "confidence", "trust_score",
            "novelty_score", "specificity_score", "duplication_score",
            "fabrication_risk", "verification_status", "angle",
            "parent_id", "expansion_depth", "iteration",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_for_synthesiser(self) -> list[dict]:
        """Synthesiser is stricter but still sees low-confidence with
        hedging. Excludes near-duplicates and high fabrication risk."""
        rows = self.conn.execute(
            """SELECT id, fact, source_url, confidence, trust_score,
                      specificity_score, duplication_score, fabrication_risk,
                      verification_status, angle, iteration
               FROM conditions
               WHERE duplication_score < 0.80
                 AND fabrication_risk < 0.80
               ORDER BY confidence DESC, trust_score DESC, id ASC"""
        ).fetchall()
        cols = [
            "id", "fact", "source_url", "confidence", "trust_score",
            "specificity_score", "duplication_score", "fabrication_risk",
            "verification_status", "angle", "iteration",
        ]
        return [dict(zip(cols, row)) for row in rows]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_thinker(self) -> str:
        """Format the corpus as structured text for the thinker.

        Groups conditions by angle and includes gradient flags so the
        thinker can reason about quality, gaps, and contradictions.
        """
        conditions = self.get_for_thinker()
        if not conditions:
            return "(no findings yet)"

        lines: list[str] = [
            f"CORPUS: {len(conditions)} conditions "
            f"(total stored: {self.count()})\n"
        ]

        by_angle: dict[str, list[dict]] = {}
        for c in conditions:
            angle = c["angle"] or "general"
            by_angle.setdefault(angle, []).append(c)

        for angle, conds in sorted(by_angle.items()):
            lines.append(
                f"## Angle: {angle} ({len(conds)} findings)"
            )
            for c in conds:
                status = c["verification_status"] or "unverified"
                flags = (
                    f"conf={c['confidence']:.1f}, "
                    f"trust={c['trust_score']:.1f}, "
                    f"novel={c['novelty_score']:.1f}, "
                    f"spec={c['specificity_score']:.1f}"
                )
                if c["fabrication_risk"] > 0.3:
                    flags += (
                        f", fab_risk={c['fabrication_risk']:.1f}"
                    )
                if c["duplication_score"] > 0.3:
                    flags += f", dup={c['duplication_score']:.1f}"
                src = (
                    f" [{c['source_url']}]"
                    if c["source_url"] else ""
                )
                depth = (
                    f" (depth={c['expansion_depth']})"
                    if c.get("expansion_depth", 0) > 0
                    else ""
                )
                lines.append(
                    f"  [{c['id']}] ({flags}, {status}{depth}) "
                    f"{c['fact']}{src}"
                )
            lines.append("")

        status_counts = Counter(
            c["verification_status"] or "(unset)"
            for c in conditions
        )
        lines.append(
            "STATUS SUMMARY: "
            + ", ".join(
                f"{k}={v}"
                for k, v in sorted(status_counts.items())
            )
        )

        return "\n".join(lines)

    def format_for_synthesiser(self) -> str:
        """Format the corpus for the synthesiser -- includes angle and
        gradient flags so it can weight claims appropriately."""
        conditions = self.get_for_synthesiser()
        if not conditions:
            return "(no findings)"

        by_angle: dict[str, list[dict]] = {}
        for c in conditions:
            angle = c["angle"] or "general"
            by_angle.setdefault(angle, []).append(c)

        lines: list[str] = [
            f"CORPUS: {len(conditions)} findings\n"
        ]
        for angle, conds in sorted(by_angle.items()):
            lines.append(
                f"## {angle} ({len(conds)} findings)"
            )
            for c in conds:
                src = (
                    f"\n   Source: {c['source_url']}"
                    if c["source_url"]
                    else ""
                )
                status = (
                    c["verification_status"] or "unverified"
                )
                fab = (
                    f", fabrication_risk="
                    f"{c['fabrication_risk']:.1f}"
                    if c["fabrication_risk"] > 0.2
                    else ""
                )
                lines.append(
                    f"- [#{c['id']}, {status}, "
                    f"confidence={c['confidence']:.2f}, "
                    f"trust={c['trust_score']:.2f}{fab}] "
                    f"{c['fact']}{src}"
                )
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Unified ingestion (Flock atomisation)
    # ------------------------------------------------------------------

    _URL_RE = re.compile(r"\[(https?://[^\]]+)\]")
    _BARE_URL_RE = re.compile(r"(https?://[^\s)\]\"'>]+)")
    _CONF_RE = re.compile(r"\(confidence=([0-9.]+)\)")

    def ingest_raw(
        self,
        raw_text: str,
        source_type: str = "researcher",
        source_ref: str = "",
        angle: str = "",
        iteration: int = 0,
    ) -> list[int]:
        """Ingest raw text via Flock atomisation.

        Returns list of admitted condition IDs. Flock decomposes the
        text into atomic facts, extracts source URLs, and estimates
        initial confidence -- all via a single LLM call.
        """
        if not raw_text or not raw_text.strip():
            return []

        raw_id = self.conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM raw_ingestion"
        ).fetchone()[0]
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO raw_ingestion "
            "(id, raw_text, source_type, source_ref, ingested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [raw_id, raw_text, source_type, source_ref, now],
        )

        try:
            atomised = self._flock_complete(
                "You are a research finding decomposer. Extract every "
                "atomic fact from the text below. An atomic fact is a "
                "single, self-contained claim that can be verified "
                "independently. "
                "Rules: "
                "- One fact per line "
                "- Preserve ALL specific data: names, numbers, dates, "
                "prices, URLs "
                "- If a URL is associated with a fact, append it as "
                "[URL] at the end of the line "
                "- If the text expresses confidence/uncertainty, append "
                "(confidence=X.X) where X.X is 0.0-1.0 "
                "- Do NOT summarise or generalise "
                "- Do NOT add commentary, disclaimers, or meta-text "
                "- Do NOT skip any fact, no matter how minor "
                "- Short facts (names, prices, dates) are valid "
                "- If the text is a single atomic fact already, return "
                "it as-is "
                "Text to decompose: " + raw_text,
                caller="ingest_atomise",
            )
        except Exception:
            logger.warning(
                "Flock atomisation failed -- falling back to raw "
                "text as single condition",
                exc_info=True,
            )
            atomised = raw_text

        # Fallback: if Flock returned empty (e.g. unavailable or blank
        # response), use the raw text so findings are never silently lost.
        if not atomised or not atomised.strip():
            atomised = raw_text

        ids: list[int] = []
        for line in atomised.split("\n"):
            line = re.sub(r'^\s*(?:\d+[.)\]]\s*|[-*\u2022]\s+)', '', line.strip())
            line = line.strip()
            if not line:
                continue

            url_match = self._URL_RE.search(line)
            if url_match:
                source_url = url_match.group(1)
                line = self._URL_RE.sub("", line).strip()
            else:
                bare_match = self._BARE_URL_RE.search(line)
                if bare_match:
                    source_url = bare_match.group(1).rstrip(".,;:")
                else:
                    source_url = ""

            conf_match = self._CONF_RE.search(line)
            try:
                confidence = (
                    max(0.0, min(1.0, float(conf_match.group(1))))
                    if conf_match else 0.5
                )
            except (ValueError, TypeError):
                confidence = 0.5
            if conf_match:
                line = self._CONF_RE.sub("", line).strip()

            if not line:
                continue

            cid = self._next_id
            self._next_id += 1
            ts = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_url, source_type, source_ref,
                    confidence, angle, expansion_depth,
                    created_at, iteration)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                [
                    cid, line, source_url, source_type,
                    source_ref, confidence,
                    angle or f"iteration_{iteration}",
                    ts, iteration,
                ],
            )
            ids.append(cid)

        self.conn.execute(
            "UPDATE raw_ingestion SET atomised = TRUE "
            "WHERE id = ?",
            [raw_id],
        )

        logger.info(
            "Flock atomisation: %d conditions from %d chars "
            "(source=%s)",
            len(ids), len(raw_text), source_type,
        )
        return ids

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
        """Single-pass Flock synthesis for small corpora."""
        corpus_text = "\n".join(
            f"- [#{c['id']}, "
            f"{c['verification_status'] or 'unverified'}, "
            f"conf={c['confidence']:.2f}, "
            f"trust={c['trust_score']:.2f}] "
            f"{c['fact']}"
            + (
                f" [Source: {c['source_url']}]"
                if c["source_url"] else ""
            )
            for c in conditions
        )
        return self._flock_complete(
            "You are a research synthesiser. Read ALL findings "
            "below and produce a comprehensive, well-structured "
            "report. "
            "Rules: "
            "- Include ALL facts, names, numbers, URLs "
            "- Weight claims by confidence and trust scores "
            "- Cross-reference sources "
            "- Structure with clear headings "
            "- Cite source URLs inline "
            "- Do NOT add disclaimers or moralising "
            "- fabrication_risk > 0.7: mention only if "
            "corroborated "
            "- duplication_score > 0.8: merge with "
            "higher-confidence duplicate"
            "\nUser query: " + user_query
            + "\n\n" + corpus_text,
            caller="synthesise_single",
        )

    def _synthesise_gossip(
        self, conditions: list[dict], user_query: str,
    ) -> str:
        """Gossip-style synthesis for large corpora.

        Phase 1: Flock synthesises each angle independently.
        Phase 2: Each angle summary refined with peer awareness.
        Phase 3: Flock merges all into final report (queen).
        """
        by_angle: dict[str, list[dict]] = {}
        for c in conditions:
            angle = c["angle"] or "general"
            by_angle.setdefault(angle, []).append(c)

        # Phase 1: per-angle synthesis (workers)
        angle_summaries: dict[str, str] = {}
        for angle, conds in by_angle.items():
            facts_text = "\n".join(
                f"- [#{c['id']}, "
                f"conf={c['confidence']:.2f}, "
                f"trust={c['trust_score']:.2f}] {c['fact']}"
                + (
                    f" [{c['source_url']}]"
                    if c["source_url"] else ""
                )
                for c in conds
            )
            summary = self._flock_complete(
                "You are a synthesis worker in a peer-to-peer "
                "research swarm. Synthesise these findings into "
                "a focused section report. Include ALL facts, "
                "names, numbers, URLs. Note contradictions. "
                "This section will be merged with other sections "
                "-- focus on what is unique and important in "
                "YOUR findings. Do NOT add disclaimers. Stay "
                "under 6000 characters."
                "\nResearch angle: " + angle
                + "\nUser query: " + user_query
                + "\n\n" + facts_text,
                caller="gossip_phase1_worker",
            )
            angle_summaries[angle] = summary

        logger.info(
            "Gossip Phase 1: %d angle summaries produced",
            len(angle_summaries),
        )

        # Phase 2: gossip refinement
        phase1_summaries = dict(angle_summaries)
        for angle in list(angle_summaries.keys()):
            peer_text = "\n\n".join(
                f"### {a}\n{s}"
                for a, s in phase1_summaries.items()
                if a != angle
            )
            refined = self._flock_complete(
                "You produced a section summary. Now you see "
                "summaries from peer workers who processed "
                "other research angles. Cross-reference your "
                "findings with peers. Note agreements and "
                "contradictions. Incorporate complementary "
                "findings. Remove redundancy. Preserve ALL "
                "unique findings from your section. Stay under "
                "6000 characters."
                "\nYour angle: " + angle
                + "\nUser query: " + user_query
                + "\n\nYour current summary:\n"
                + angle_summaries[angle]
                + "\n\nPeer summaries:\n" + peer_text,
                caller="gossip_phase2_refine",
            )
            angle_summaries[angle] = refined

        logger.info("Gossip Phase 2: refinement complete")

        # Phase 3: queen merges all angle summaries
        all_summaries = "\n\n".join(
            f"## {angle}\n{summary}"
            for angle, summary in sorted(
                angle_summaries.items()
            )
        )
        final = self._flock_complete(
            "You are the queen synthesiser. Merge these section "
            "reports from specialist workers into a single "
            "comprehensive research report. "
            "Rules: "
            "- Include ALL findings from all sections "
            "- Resolve contradictions by noting both sides "
            "- Structure with clear headings and sub-headings "
            "- Cite source URLs inline "
            "- Do NOT add disclaimers or moralising "
            "- Prioritise specificity: names, numbers, dates, "
            "URLs "
            "- The report should be self-contained and readable "
            "by someone with no prior context"
            "\nUser query: " + user_query
            + "\n\n" + all_summaries,
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
