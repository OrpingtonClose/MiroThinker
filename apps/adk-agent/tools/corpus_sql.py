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

import logging
import time

from google.adk.tools import FunctionTool, ToolContext

logger = logging.getLogger(__name__)

# Maximum rows to return in a single query result (prevents context blow-up)
_MAX_RESULT_ROWS = 200
# Maximum chars per cell in the result (prevents huge text fields)
_MAX_CELL_CHARS = 500


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

    Row types: 'finding', 'similarity', 'contradiction', 'raw', 'synthesis'

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

    start = time.monotonic()
    try:
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
