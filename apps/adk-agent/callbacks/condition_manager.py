# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-agent callback for the researcher: decomposes findings into atoms.

After the researcher outputs its findings text (stored in
``state["research_findings"]``), this callback:

  1. Ingests the raw text via Flock atomisation (``corpus.ingest_raw()``)
  2. Stores the resulting atoms in the shared ``CorpusStore`` (DuckDB)
  3. Overwrites ``state["research_findings"]`` with the structured corpus
     formatted for the thinker's next iteration

This bridges the gap between the researcher's free-text output and the
structured corpus that the thinker reads.  The researcher writes prose;
Flock turns it into queryable atoms with gradient-flag scoring.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from dashboard import get_active_collector

if TYPE_CHECKING:
    from models.corpus_store import CorpusStore

logger = logging.getLogger(__name__)


def researcher_condition_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """After-agent callback: decompose researcher findings into atoms.

    Reads ``state["research_findings"]``, ingests via Flock atomisation,
    and updates state with formatted corpus.
    """
    state = callback_context.state

    findings_text = state.get("research_findings", "")
    corpus = _get_corpus(state)
    if not findings_text:
        # No new findings, but still restore corpus format so the thinker
        # doesn't lose context from previous iterations.
        state["research_findings"] = corpus.format_for_thinker()
        state["corpus_for_synthesis"] = corpus.format_for_synthesiser()
        return None

    # Track iteration number
    iteration = state.get("_corpus_iteration", 0)

    # Ingest raw findings via Flock atomisation (replaces regex parsing)
    ids = corpus.ingest_raw(
        raw_text=findings_text,
        source_type="researcher",
        source_ref="",
        angle=f"iteration_{iteration}",
        iteration=iteration,
    )
    admitted_count = len(ids)
    total_count = corpus.count()
    if admitted_count:
        logger.info(
            "Condition manager: ingested %d conditions "
            "(iteration %d, total %d)",
            admitted_count, iteration, total_count,
        )
        _c = get_active_collector()
        if _c:
            _c.corpus_update(admitted_count, total_count, iteration)

    # Update state with structured corpus for thinker
    state["research_findings"] = corpus.format_for_thinker()
    state["_corpus_iteration"] = iteration + 1

    # Also store synthesiser-formatted version for the final stage
    state["corpus_for_synthesis"] = corpus.format_for_synthesiser()

    return None  # preserve original researcher output


# ---------------------------------------------------------------------------
# Module-level corpus store singleton
# ---------------------------------------------------------------------------
# Each pipeline run creates a fresh CorpusStore.  The store lives for the
# duration of the LoopAgent execution.  We use a dict keyed by a session
# marker so concurrent pipelines don't collide.

_corpus_stores: dict[str, "CorpusStore"] = {}


def _get_corpus(state: dict) -> "CorpusStore":
    """Get or create the CorpusStore for this session."""
    from models.corpus_store import CorpusStore

    # Use session-level key to isolate concurrent runs
    key = state.get("_corpus_key", "default")
    if key not in _corpus_stores:
        _corpus_stores[key] = CorpusStore()
        logger.info("Created new CorpusStore for key=%s", key)
    return _corpus_stores[key]


def build_corpus_state() -> dict:
    """Return the initial session-state dict for a new pipeline run.

    ``InMemorySessionService`` deep-copies sessions on both ``create_session``
    and ``get_session``, so state set *after* creation is invisible to the
    Runner.  This function builds the dict that must be passed to
    ``create_session(state=…)`` so the Runner sees the keys.

    Call :func:`init_corpus` after session creation to register the
    CorpusStore singleton.
    """
    import uuid

    key = f"corpus_{uuid.uuid4().hex[:8]}"
    return {
        "_corpus_key": key,
        "_corpus_iteration": 0,
        # Fallbacks so agents never see raw template literals.
        # "(no findings yet)" matches the thinker's first-iteration check.
        "corpus_for_synthesis": "(no findings)",
        "research_findings": "(no findings yet)",
    }


def init_corpus(state: dict) -> None:
    """Register (or re-register) the CorpusStore singleton for a session.

    Call this *after* ``create_session`` with the state returned by
    :func:`build_corpus_state` so the module-level ``_corpus_stores`` dict
    knows about the new key.  Eagerly creates the CorpusStore so that
    ``get_corpus_text`` works even if the researcher callback never fires.
    Also tears down any existing store for this key as a safety net
    against leaked DuckDB connections.
    """
    from models.corpus_store import CorpusStore

    key = state.get("_corpus_key")
    if not key:
        logger.warning("init_corpus called without _corpus_key in state")
        return

    # Defensive cleanup: close existing store for this key if present
    if key in _corpus_stores:
        _corpus_stores[key].close()
    _corpus_stores[key] = CorpusStore()
    logger.info("Initialised corpus store for key=%s", key)


def get_corpus_text(state: dict) -> str:
    """Return the synthesiser-formatted corpus text, or empty string.

    Used to dump partial results when the pipeline stalls before
    the synthesiser can produce its report.
    """
    key = state.get("_corpus_key")
    if key and key in _corpus_stores:
        return _corpus_stores[key].format_for_synthesiser()
    return ""


def cleanup_corpus(state: dict) -> None:
    """Close and remove the CorpusStore for a completed pipeline run.

    Call this after ``run_pipeline`` returns to release the DuckDB
    connection and its in-memory data.  Without this, each pipeline
    run leaks a CorpusStore in the module-level ``_corpus_stores`` dict.
    """
    key = state.get("_corpus_key")
    if key and key in _corpus_stores:
        _corpus_stores[key].close()
        del _corpus_stores[key]
        logger.info("Cleaned up CorpusStore for key=%s", key)
