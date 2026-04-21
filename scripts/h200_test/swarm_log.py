#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Thin logging wrapper around ConditionStore — the universal event sink.

Instead of a separate SQLite database, every swarm event (LLM calls,
enrichment results, gossip exchanges, worker assignments, hive memory
hits) goes through the ConditionStore as AtomicCondition rows.  The
existing lineage DAG (parent_ids, source_ref) gives you the execution
graph for free.

This module provides:

1. ``init_logging(store)`` — configure Python's logging to also capture
   console output, and return the store for use as the lineage store.

2. ``log_llm_call`` / ``log_enrichment_result`` / ``log_worker_output``
   — convenience functions that delegate to the module-level store's
   ``emit_llm_call`` / ``emit_enrichment`` / ``emit_gossip_exchange``.

3. ``export_mermaid`` / ``print_graph_stats`` — graph export helpers
   that delegate to the store.

Query examples (DuckDB SQL via the ConditionStore):
    -- All LLM calls for gossip round 2
    SELECT * FROM conditions WHERE row_type = 'llm_call'
      AND phase = 'gossip_round_2';

    -- Enrichment results rejected by relevance gate
    SELECT * FROM conditions WHERE row_type = 'enrichment'
      AND strategy::JSON->>'admitted' = 'false';

    -- Full lineage chain for the queen merge
    SELECT * FROM conditions WHERE phase = 'queen_merge';
    -- then use store.get_lineage_chain(id) to walk parents

    -- Export as Mermaid
    python -c "from swarm_log import export_mermaid; print(export_mermaid())"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure repo root + strands-agent are importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_STRANDS_AGENT = str(Path(__file__).resolve().parents[2] / "apps" / "strands-agent")
if _STRANDS_AGENT not in sys.path:
    sys.path.insert(0, _STRANDS_AGENT)

from corpus import ConditionStore


# ── Module state ──────────────────────────────────────────────────────

_store: ConditionStore | None = None


def init_logging(
    store: ConditionStore,
    console_level: int = logging.INFO,
) -> ConditionStore:
    """Initialize logging with a ConditionStore as the universal event sink.

    Sets up console logging and returns the store.  The store should be
    passed to ``SwarmConfig(lineage_store=...)`` so the engine's
    ``LineageEntry`` emissions also land in the same store.

    Args:
        store: The ConditionStore to use as the universal event sink.
        console_level: Console logging level.

    Returns:
        The same store (for convenience in assignment).
    """
    global _store
    _store = store

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s | %(message)s",
    ))

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove existing handlers to avoid duplicates
    root.handlers.clear()
    root.addHandler(console)

    logging.getLogger(__name__).info(
        "store=<%s> | ConditionStore logging initialized (universal event sink)",
        type(store).__name__,
    )
    return store


def get_store() -> ConditionStore | None:
    """Return the current module-level ConditionStore."""
    return _store


# ── Convenience functions (delegate to store) ─────────────────────────

def log_llm_call(
    phase: str,
    prompt: str,
    response: str,
    *,
    worker: str = "",
    model: str = "",
    angle: str = "",
    max_tokens: int = 0,
    temperature: float = 0.0,
    elapsed_s: float = 0.0,
    error: str = "",
    parent_ids: list[str] | None = None,
) -> int | None:
    """Log a full LLM call to the ConditionStore.  Returns condition ID."""
    if _store is None:
        return None
    return _store.emit_llm_call(
        phase=phase,
        prompt=prompt,
        response=response,
        worker=worker,
        model=model,
        angle=angle,
        max_tokens=max_tokens,
        temperature=temperature,
        elapsed_s=elapsed_s,
        error=error,
        parent_ids=parent_ids,
    )


def log_enrichment_result(
    angle: str,
    query: str,
    *,
    backend: str = "",
    title: str = "",
    url: str = "",
    snippet: str = "",
    admitted: bool = True,
    reject_reason: str = "",
    parent_ids: list[str] | None = None,
) -> int | None:
    """Log an enrichment search result to the ConditionStore.  Returns condition ID."""
    if _store is None:
        return None
    return _store.emit_enrichment(
        angle=angle,
        query=query,
        backend=backend,
        title=title,
        url=url,
        snippet=snippet,
        admitted=admitted,
        reject_reason=reject_reason,
        parent_ids=parent_ids,
    )


def log_worker_output(
    phase: str,
    worker: str,
    output: str,
    *,
    angle: str = "",
    round_num: int = 0,
    info_gain: float = 0.0,
    parent_ids: list[str] | None = None,
) -> int | None:
    """Log a worker output to the ConditionStore.  Returns condition ID."""
    if _store is None:
        return None
    return _store.emit_gossip_exchange(
        worker_id=worker,
        angle=angle,
        round_num=round_num,
        output=output,
        info_gain=info_gain,
        parent_ids=parent_ids,
    )


# ── Graph export (delegate to store) ─────────────────────────────────

def export_mermaid(
    include_types: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> str:
    """Export the execution graph as a Mermaid flowchart."""
    if _store is None:
        return "graph TD\n  no_data[No store configured]"
    return _store.export_mermaid(
        include_types=include_types,
        exclude_types=exclude_types,
    )


def print_graph_stats() -> str:
    """Return summary statistics of the execution graph."""
    if _store is None:
        return "No store configured"
    return _store.graph_stats()
