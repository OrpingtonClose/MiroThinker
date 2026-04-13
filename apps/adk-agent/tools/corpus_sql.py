# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Maestro's corpus SQL tool — unrestricted Flock/DuckDB access.

The maestro agent gets a single tool: ``execute_flock_sql(query)``.
This gives the maestro free-form access to the corpus DuckDB database
with Flock's LLM-in-SQL extension loaded.  The maestro can:

  - Read any column, any row
  - Create new columns (ALTER TABLE)
  - Insert new rows
  - Run Flock LLM functions (llm_complete, llm_filter, etc.)
  - Update flags, scores, statuses
  - Compose arbitrary analytical queries

The 11-step algorithm battery templates are provided in the maestro's
instruction as *examples* — the maestro is free to invent new operations
based on corpus state.

Safety: the tool operates on the session-scoped corpus DuckDB file,
not any shared database.  Each pipeline run gets its own file.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from google.adk.tools import FunctionTool, ToolContext

from utils.flock_proxy import (
    get_flock_call_count,
    register_flock_progress,
    unregister_flock_progress,
)

logger = logging.getLogger(__name__)

# Flock LLM function names used to detect long-running queries
_FLOCK_LLM_FUNCTIONS = (
    "llm_complete", "llm_filter", "llm_sentiment", "llm_embed",
    "llm_complete_json", "llm_extract",
)

# Maximum rows to return in a single query result (prevents context blow-up)
_MAX_RESULT_ROWS = 200
# Maximum chars per cell in the result (prevents huge text fields)
_MAX_CELL_CHARS = 500


def _fix_unescaped_quotes(query: str) -> str:
    """Fix unescaped single quotes inside SQL string literals.

    LLMs routinely generate ``WHERE fact ILIKE '%Anna's Library%'``
    instead of the correct ``'%Anna''s Library%'``.  The Flock DuckDB
    extension's parser crashes catastrophically on this — instead of
    a normal syntax error it emits ``Unknown keyword: SELECT``.

    Strategy: walk the query character-by-character, tracking whether
    we are inside a string literal.  When we encounter a single quote
    that is preceded and followed by word characters (i.e. it looks
    like an apostrophe inside a word), double it.  This avoids
    corrupting valid SQL where a closing quote abuts a keyword
    (e.g. ``'value'AND``).
    """
    if "'" not in query:
        return query

    # Common English apostrophe suffixes — the characters *after* the
    # quote in patterns like  Anna's  /  don't  /  they're  /  I've
    _APOSTROPHE_SUFFIXES = {"s", "t", "re", "ve", "ll", "d", "m"}

    chars = list(query)
    n = len(chars)
    result: list[str] = []
    in_string = False

    i = 0
    while i < n:
        ch = chars[i]

        if ch == "'":
            if not in_string:
                # Opening a string literal
                in_string = True
                result.append(ch)
            elif i + 1 < n and chars[i + 1] == "'":
                # Already-escaped quote inside string — pass both through
                result.append(ch)
                result.append(chars[i + 1])
                i += 1
            else:
                # Single quote inside a string — is it an apostrophe
                # (letter'letter) or a closing quote?
                #
                # We only treat it as an apostrophe when the suffix
                # after the quote matches a known English contraction
                # pattern (e.g.  's  /  't  /  're  /  've  /  'll).
                # This avoids corrupting  'value'AND  while still
                # fixing  Anna's  →  Anna''s.
                prev_is_letter = i > 0 and chars[i - 1].isalpha()
                if prev_is_letter and i + 1 < n and chars[i + 1].isalpha():
                    # Extract the suffix word after the apostrophe
                    suffix_end = i + 1
                    while suffix_end < n and chars[suffix_end].isalpha():
                        suffix_end += 1
                    suffix = "".join(chars[i + 1 : suffix_end]).lower()
                    if suffix in _APOSTROPHE_SUFFIXES:
                        # Known contraction — double the apostrophe
                        result.append("'")
                        result.append("'")
                    else:
                        # Not a known contraction — treat as closing quote
                        in_string = False
                        result.append(ch)
                else:
                    # Not letter'letter — closing quote
                    in_string = False
                    result.append(ch)
        else:
            result.append(ch)

        i += 1

    fixed = "".join(result)
    if fixed != query:
        logger.debug("Fixed unescaped quotes in SQL: %s → %s", query[:200], fixed[:200])
    return fixed


def _get_corpus_connection(corpus_key: str = ""):
    """Get the DuckDB connection from the active CorpusStore.

    Returns the raw DuckDB connection so the maestro can execute
    arbitrary SQL including Flock functions.

    Args:
        corpus_key: Session-specific corpus key from state.  When provided,
            looks up the exact CorpusStore for this session.  Falls back
            to the most recently created store if the key is missing.
    """
    from callbacks.condition_manager import _corpus_stores
    if not _corpus_stores:
        return None
    # Prefer the session-specific corpus store
    if corpus_key and corpus_key in _corpus_stores:
        return _corpus_stores[corpus_key].conn
    # Fallback: most recent store (single-session case)
    key = list(_corpus_stores.keys())[-1]
    return _corpus_stores[key].conn


async def execute_flock_sql(query: str, tool_context: ToolContext) -> str:
    """Execute arbitrary SQL/Flock on the corpus DuckDB database.

    You have UNRESTRICTED access to the corpus.  The ``conditions``
    table is the main corpus table with all research findings.

    You can run ANY valid DuckDB SQL including Flock LLM functions:
      - ``SELECT ... FROM conditions WHERE ...``
      - ``UPDATE conditions SET ... WHERE ...``
      - ``ALTER TABLE conditions ADD COLUMN ...``
      - ``INSERT INTO conditions (...) VALUES (...)``
      - Flock: ``SELECT llm_complete('prompt', column) FROM ...``
      - Flock: ``SELECT llm_filter(column, 'criteria') FROM ...``

    Key columns in ``conditions``:
      id, fact, source_url, source_type, confidence, trust_score,
      novelty_score, specificity_score, relevance_score,
      actionability_score, composite_quality, duplication_score,
      fabrication_risk, information_density, processing_status,
      expansion_tool, expansion_hint, expansion_fulfilled,
      cluster_id, cluster_rank, consider_for_use, angle, strategy,
      row_type, parent_id, related_id, relationship, iteration,
      created_at, scored_at, staleness_penalty, cross_ref_boost,
      contradiction_flag, contradiction_partner, obsolete_reason

    Row types: 'finding', 'similarity', 'contradiction', 'raw', 'synthesis', 'thought', 'insight'
    Note: 'thought' rows are IMMUTABLE — do not UPDATE or DELETE them.
    Note: 'insight' rows are evidence-grounded conclusions from arbitration — treat as findings.

    Processing statuses: 'raw', 'scored', 'analysed', 'ready', 'merged'

    Returns the query result as formatted text, or the number of
    affected rows for DML statements.

    Args:
        query: Any valid DuckDB SQL query (may include Flock functions).

    Returns:
        Query results as formatted text.
    """
    corpus_key = tool_context.state.get("_corpus_key", "")
    conn = _get_corpus_connection(corpus_key)
    if conn is None:
        return "[ERROR] No active corpus connection. The pipeline must be running."

    # Fix unescaped single quotes (LLMs write Anna's not Anna''s)
    query = _fix_unescaped_quotes(query)

    # Soft guard: warn (but do not block) if the query mutates thought or insight rows.
    # The maestro is trusted, but violations should be visible in logs.
    _ql = query.lower()
    if re.search(r"update\s+conditions\b.*\brow_type\s*=\s*'thought'", _ql, re.DOTALL):
        logger.warning(
            "THOUGHT SOVEREIGNTY: UPDATE on thought rows detected — "
            "thought rows should be IMMUTABLE. Query: %.200s", query,
        )
    if re.search(r"delete\s+from\s+conditions\b.*\brow_type\s*=\s*'thought'", _ql, re.DOTALL):
        logger.warning(
            "THOUGHT SOVEREIGNTY: DELETE on thought rows detected — "
            "thought rows should be IMMUTABLE. Query: %.200s", query,
        )
    if re.search(r"update\s+conditions\b.*\brow_type\s*=\s*'insight'", _ql, re.DOTALL):
        logger.warning(
            "INSIGHT SOVEREIGNTY: UPDATE on insight rows detected — "
            "insight rows should be IMMUTABLE. Query: %.200s", query,
        )
    if re.search(r"delete\s+from\s+conditions\b.*\brow_type\s*=\s*'insight'", _ql, re.DOTALL):
        logger.warning(
            "INSIGHT SOVEREIGNTY: DELETE on insight rows detected — "
            "insight rows should be IMMUTABLE. Query: %.200s", query,
        )

    # ── FLAT-SCORING GUARD ──────────────────────────────────────────
    # Detect bulk UPDATEs that set score columns to literal float values
    # without using llm_complete() — this is the flat-scoring anti-pattern
    # that assigns identical scores to all findings.
    _SCORE_COLS = (
        "trust_score", "novelty_score", "specificity_score",
        "relevance_score", "actionability_score", "fabrication_risk",
    )
    if re.search(r"(?i)\bupdate\b.*\bconditions\b", _ql):
        # Check if any score column is set to a literal number (not an
        # llm_complete call or subquery).
        has_flat_score = False
        for col in _SCORE_COLS:
            # Match patterns like: trust_score = 0.6  or  trust_score=0.7
            # but NOT: trust_score = CAST(llm_complete(...) AS FLOAT)
            # and NOT: trust_score = (SELECT ...)
            pattern = rf"{col}\s*=\s*(\d+\.?\d*)"
            match = re.search(pattern, _ql)
            if match:
                # Verify it's a literal, not inside an llm_complete call
                # by checking there's no llm_complete between the column
                # name and the number on the same line
                pos = match.start()
                context_before = _ql[max(0, pos - 200):pos]
                if "llm_complete" not in context_before.split(col)[-1] if col in context_before else True:
                    has_flat_score = True
                    break
        if has_flat_score:
            logger.warning(
                "FLAT SCORING DETECTED: UPDATE sets score columns to literal "
                "values instead of using llm_complete() for per-row assessment. "
                "The safety-net scorer will override these with genuine "
                "per-finding LLM scores (via score_version=0 check). "
                "Query: %.300s", query,
            )

    # Detect Flock LLM queries — they can take minutes and must not block
    # the asyncio event loop (which would freeze SSE streaming).
    query_lower = query.lower()
    is_flock_query = any(fn in query_lower for fn in _FLOCK_LLM_FUNCTIONS)
    start = time.monotonic()
    try:
        if is_flock_query:
            # Run in thread pool so the event loop stays responsive.
            # Register a progress callback so we can log each Flock LLM call.
            def _on_flock_call(call_num: int, model: str) -> None:
                elapsed = time.monotonic() - start
                logger.info(
                    "Flock progress: LLM call #%d (%s) at %.1fs",
                    call_num, model, elapsed,
                )

            register_flock_progress(_on_flock_call)
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, conn.execute, query)
            finally:
                total_calls = get_flock_call_count()
                unregister_flock_progress()
                if total_calls:
                    logger.info(
                        "Flock query completed: %d LLM calls in %.1fs",
                        total_calls, time.monotonic() - start,
                    )
        else:
            result = conn.execute(query)

        elapsed_ms = (time.monotonic() - start) * 1000

        # Check if this is a SELECT (has results) or DML (returns count)
        description = result.description
        if description:
            # SELECT — fetch and format results
            rows = result.fetchall()
            col_names = [d[0] for d in description]

            if not rows:
                return f"Query returned 0 rows ({elapsed_ms:.0f}ms)"

            # Format as readable text table
            lines: list[str] = []
            lines.append(
                f"Query returned {len(rows)} rows, "
                f"{len(col_names)} columns ({elapsed_ms:.0f}ms)"
            )
            lines.append("")

            # Truncate if too many rows
            display_rows = rows[:_MAX_RESULT_ROWS]
            if len(rows) > _MAX_RESULT_ROWS:
                lines.append(
                    f"[Showing first {_MAX_RESULT_ROWS} of {len(rows)} rows]"
                )

            # Column headers
            lines.append(" | ".join(col_names))
            lines.append("-" * min(120, len(" | ".join(col_names))))

            # Data rows
            for row in display_rows:
                cells = []
                for val in row:
                    s = str(val) if val is not None else "NULL"
                    if len(s) > _MAX_CELL_CHARS:
                        s = s[:_MAX_CELL_CHARS] + "..."
                    cells.append(s)
                lines.append(" | ".join(cells))

            return "\n".join(lines)
        else:
            # DML — report affected rows
            try:
                affected = result.fetchone()
                count = affected[0] if affected else 0
            except Exception:
                count = "unknown"
            return (
                f"Query executed successfully ({elapsed_ms:.0f}ms). "
                f"Rows affected: {count}"
            )

    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        return (
            f"[SQL ERROR] ({elapsed_ms:.0f}ms) {type(exc).__name__}: {exc}"
        )


# Public FunctionTool instance for the maestro agent
execute_flock_sql_tool = FunctionTool(execute_flock_sql)

CORPUS_SQL_TOOLS = [execute_flock_sql_tool]
