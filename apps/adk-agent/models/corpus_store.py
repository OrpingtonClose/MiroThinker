# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
DuckDB-backed corpus store for AtomicConditions -- Flock edition.

Replaces brittle programmatic text processing (regex parsing, len checks,
keyword heuristics, hard filters, 1000-char truncation) with DuckDB's Flock
community extension -- LLM calls executed directly in SQL.

Key changes from the original:
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
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import duckdb

from models.atomic_condition import AtomicCondition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flock model configuration (env-driven)
# ---------------------------------------------------------------------------
_FLOCK_MODEL = os.environ.get(
    "FLOCK_MODEL",
    os.environ.get("ADK_MODEL", "openai/gpt-4o").replace("litellm/", ""),
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

# Persistent DB path (empty string -> in-memory)
_DB_PATH = os.environ.get("CORPUS_DB_PATH", "")


class CorpusStore:
    """DuckDB + Flock store for AtomicConditions with gradient scoring."""

    def __init__(self) -> None:
        db = _DB_PATH or ":memory:"
        self.conn = duckdb.connect(db)

        # -- Load Flock extension --
        self.conn.execute("INSTALL flock FROM community")
        self.conn.execute("LOAD flock")

        # -- Configure Flock secrets & model --
        self._setup_flock()

        self._setup_tables()
        self._next_id = self._compute_next_id()

    # ------------------------------------------------------------------
    # Flock configuration
    # ------------------------------------------------------------------

    def _setup_flock(self) -> None:
        """Create the DuckDB SECRET and MODEL that Flock needs."""
        def _sql_escape(val: str) -> str:
            return val.replace("'", "''")

        secret_parts = ["TYPE OPENAI"]
        if _FLOCK_KEY:
            secret_parts.append(f"API_KEY '{_sql_escape(_FLOCK_KEY)}'")
        if _FLOCK_BASE:
            secret_parts.append(f"BASE_URL '{_sql_escape(_FLOCK_BASE)}'")

        if _FLOCK_KEY:
            try:
                self.conn.execute("DROP SECRET IF EXISTS flock_secret")
            except Exception:
                pass
            try:
                self.conn.execute(
                    f"CREATE SECRET flock_secret "
                    f"({', '.join(secret_parts)})"
                )
            except Exception:
                logger.warning(
                    "Failed to CREATE SECRET -- Flock will be "
                    "unavailable (bad API key format?).",
                    exc_info=True,
                )

        model_name = _sql_escape(_FLOCK_MODEL or "gpt-4o")
        provider = _sql_escape(_FLOCK_PROVIDER)
        try:
            self.conn.execute("DROP MODEL IF EXISTS 'corpus_model'")
        except Exception:
            pass
        try:
            self.conn.execute(
                f"CREATE MODEL('corpus_model', '{model_name}', "
                f"'{provider}')"
            )
        except Exception:
            logger.warning(
                "Failed to CREATE MODEL -- Flock ingestion and "
                "scoring will be unavailable. Ensure the flock "
                "extension is installed and a valid API key is set.",
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
                score_version INTEGER DEFAULT 0
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

    def _flock_complete(self, prompt: str) -> str:
        """Run a single Flock llm_complete call and return text."""
        row = self.conn.execute(
            "SELECT llm_complete("
            "{'model_name': 'corpus_model'}, "
            "{'prompt': ?})",
            [prompt],
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else ""

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
            "Finding: " + fact
        ), 0.5)

        trust = self._parse_float(self._flock_complete(
            "Rate the trustworthiness of this source URL on a scale "
            "from 0.0 to 1.0. Academic/government = high, established "
            "news = medium-high, forums/social = medium, unknown/no "
            "URL = low. Return ONLY a decimal number. "
            "URL: " + url_text
        ), 0.5)

        specificity = self._parse_float(self._flock_complete(
            "Rate how specific and concrete this finding is on 0.0 to "
            "1.0. 1.0 = contains exact names, numbers, dates, URLs. "
            "0.0 = vague generality with no concrete data. Return "
            "ONLY a decimal number. "
            "Finding: " + fact
        ), 0.5)

        fabrication = self._parse_float(self._flock_complete(
            "Rate the risk that this finding is "
            "fabricated/hallucinated on 0.0 to 1.0. "
            "0.0 = clearly grounded in real sources. "
            "1.0 = likely made up, no verifiable details. Consider: "
            "Does it cite a real URL? Are the claims verifiable? "
            "Return ONLY a decimal number. "
            "Finding: " + fact + " Source: " + url_text
        ), 0.0)

        relevance = 0.5
        if user_query:
            relevance = self._parse_float(self._flock_complete(
                "Rate how relevant this finding is to the user query "
                "on 0.0 to 1.0. Return ONLY a decimal number. "
                "Query: " + user_query + " Finding: " + fact
            ), 0.5)

        novelty = self._parse_float(self._flock_complete(
            "Rate how novel this finding is on 0.0 to 1.0. "
            "1.0 = completely new information not commonly known. "
            "0.0 = widely known, obvious, or trivial. "
            "Return ONLY a decimal number. "
            "Finding: " + fact
        ), 0.5)

        actionability = self._parse_float(self._flock_complete(
            "Rate how actionable this finding is on 0.0 to 1.0. "
            "1.0 = directly usable, contains specific steps or data. "
            "0.0 = purely informational with no actionable content. "
            "Return ONLY a decimal number. "
            "Finding: " + fact
        ), 0.5)

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE conditions
               SET confidence = ?, trust_score = ?,
                   specificity_score = ?, fabrication_risk = ?,
                   relevance_score = ?, novelty_score = ?,
                   actionability_score = ?,
                   scored_at = ?,
                   score_version = score_version + 1
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
            "A: " + fact_a + " B: " + fact_b
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
                "Text to decompose: " + raw_text
            )
        except Exception:
            logger.warning(
                "Flock atomisation failed -- falling back to raw "
                "text as single condition",
                exc_info=True,
            )
            atomised = raw_text

        ids: list[int] = []
        for line in atomised.split("\n"):
            line = re.sub(r'^\s*(?:\d+[.)\]]\s*|[-*\u2022]\s+)', '', line.strip())
            line = line.strip()
            if not line:
                continue

            url_match = self._URL_RE.search(line)
            source_url = (
                url_match.group(1) if url_match else ""
            )
            if url_match:
                line = self._URL_RE.sub("", line).strip()

            conf_match = self._CONF_RE.search(line)
            confidence = (
                max(0.0, min(1.0, float(conf_match.group(1))))
                if conf_match else 0.5
            )
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
        """Produce a synthesis of the corpus.
        Uses gossip for large corpora."""
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
            + "\n\n" + corpus_text
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
                + "\n\n" + facts_text
            )
            angle_summaries[angle] = summary

        logger.info(
            "Gossip Phase 1: %d angle summaries produced",
            len(angle_summaries),
        )

        # Phase 2: gossip refinement
        for angle in list(angle_summaries.keys()):
            peer_text = "\n\n".join(
                f"### {a}\n{s}"
                for a, s in angle_summaries.items()
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
                + "\n\nPeer summaries:\n" + peer_text
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
            + "\n\n" + all_summaries
        )

        logger.info("Gossip Phase 3: queen synthesis complete")
        return final

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()
