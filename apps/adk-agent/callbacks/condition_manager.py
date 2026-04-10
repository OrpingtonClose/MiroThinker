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
import threading
from typing import TYPE_CHECKING, Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from dashboard import get_active_collector

if TYPE_CHECKING:
    from models.corpus_store import CorpusStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe queue for search results collected by after_tool_callback.
# Parallel tool callbacks append here; researcher_condition_callback drains
# on the main thread where DuckDB access is safe.
# ---------------------------------------------------------------------------
_pending_search_results: list[tuple[str, str, str, int]] = []  # (corpus_key, text, source_type, iteration)
_pending_search_lock = threading.Lock()


def queue_search_result(
    corpus_key: str, text: str, source_type: str, iteration: int,
) -> None:
    """Thread-safe enqueue of a search result for later corpus ingestion.

    Called from ``after_tool_callback`` which may fire from parallel
    tool threads.  The actual ``ingest_raw()`` happens in
    ``researcher_condition_callback`` on the main thread.
    """
    with _pending_search_lock:
        _pending_search_results.append((corpus_key, text, source_type, iteration))
    logger.debug(
        "Queued search result for corpus ingestion: source_type=%s, %d chars",
        source_type, len(text),
    )


def _drain_search_queue(state: dict) -> int:
    """Drain queued search results into the corpus (single-threaded).

    Only takes items matching this session's ``_corpus_key``, leaving
    items from other concurrent sessions in the queue.

    Returns the total number of admitted conditions across all drained items.
    """
    corpus_key = state.get("_corpus_key", "")
    if not corpus_key:
        return 0

    # Partition under the lock: take ours, leave others
    with _pending_search_lock:
        mine = [item for item in _pending_search_results if item[0] == corpus_key]
        if not mine:
            return 0
        # Keep only items that don't belong to us
        _pending_search_results[:] = [
            item for item in _pending_search_results if item[0] != corpus_key
        ]

    total_admitted = 0
    for _key, text, source_type, iteration in mine:
        try:
            admitted = _ingest_text_into_corpus(state, text, source_type)
            total_admitted += admitted
        except Exception:
            logger.warning(
                "Failed to ingest queued search result (source_type=%s)",
                source_type, exc_info=True,
            )

    if total_admitted:
        logger.info(
            "Drained search queue: %d items, %d conditions admitted",
            len(mine), total_admitted,
        )
    return total_admitted


def _ingest_text_into_corpus(
    state: dict,
    text: str,
    source_type: str,
) -> int:
    """Shared helper: atomise *text* and ingest into the session corpus.

    Returns the number of newly admitted conditions.
    """
    corpus = _get_corpus(state)
    iteration = state.get("_corpus_iteration", 0)

    # Set tracing context so Flock LLM calls and algorithm traces
    # are tagged with the correct session and iteration.
    _c = get_active_collector()
    if _c:
        corpus.set_trace_context(
            session_id=_c.session_id,
            iteration=iteration,
        )

    ids = corpus.ingest_raw(
        raw_text=text,
        source_type=source_type,
        source_ref="",
        angle=f"iteration_{iteration}",
        iteration=iteration,
    )
    admitted_count = len(ids)
    total_count = corpus.count()
    if admitted_count:
        logger.info(
            "Condition manager: ingested %d conditions from %s "
            "(iteration %d, total %d)",
            admitted_count, source_type, iteration, total_count,
        )
        if _c:
            _c.corpus_update(admitted_count, total_count, iteration)

    # Score & dedup
    user_query = state.get("user_query", "")
    scored = corpus.score_new_conditions(user_query)
    if scored:
        logger.info("Scored %d conditions via Flock", scored)
    deduped = corpus.compute_duplications()
    if deduped:
        logger.info("Evaluated %d dedup pairs via Flock", deduped)

    # Run the full algorithm battery (SQL gates + LLM enrichment)
    battery_results = corpus.run_algorithm_battery(
        user_query=user_query,
        iteration=iteration,
    )
    logger.info(
        "Algorithm battery: %d ready, %d for expansion, "
        "%d excluded, %d cluster reps (%.0fms)",
        battery_results.get("ready", 0),
        battery_results.get("expansion_targets", 0),
        battery_results.get("excluded", 0),
        battery_results.get("cluster_reps", 0),
        battery_results.get("battery_duration_ms", 0),
    )

    # Emit battery results as a dashboard event for live observability
    if _c:
        _c.emit_event("algorithm_battery", data=battery_results)

    # Report expansion targets to dashboard
    if _c and battery_results.get("expansion_targets", 0):
        expansion_targets = corpus.get_expansion_targets()
        logger.info(
            "Expansion targets for next iteration: %s",
            [t["strategy"] for t in expansion_targets[:5]],
        )

    return admitted_count


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
    _SENTINELS = {"(no findings yet)", "(no findings)"}
    if not findings_text or findings_text.strip() in _SENTINELS:
        # No new findings, but still restore corpus format so the thinker
        # doesn't lose context from previous iterations.
        state["research_findings"] = corpus.format_for_thinker()
        state["corpus_for_synthesis"] = corpus.format_for_synthesiser()
        # Still advance iteration so stale-expansion cleanup and the
        # thinker see a fresh round number even when the researcher
        # produced no new findings.
        state["_corpus_iteration"] = state.get("_corpus_iteration", 0) + 1
        return None

    _ingest_text_into_corpus(state, findings_text, "researcher")

    # Drain any search results queued by parallel after_tool_callbacks.
    # This is the single-threaded point where DuckDB access is safe.
    _drain_search_queue(state)

    # Update state with structured corpus for thinker
    state["research_findings"] = corpus.format_for_thinker()

    # Also store synthesiser-formatted version for the final stage
    state["corpus_for_synthesis"] = corpus.format_for_synthesiser()

    # Inject expansion targets so the researcher acts on them
    expansion_targets = corpus.get_expansion_targets()
    if expansion_targets:
        lines = ["=== ENRICHMENT TASKS (from corpus analysis) ==="]
        for t in expansion_targets[:10]:
            lines.append(
                f"- Finding [{t['id']}] needs enrichment via "
                f"{t['strategy']}: {t['hint']}"
            )
        lines.append("=== END ENRICHMENT TASKS ===")
        state["_expansion_targets"] = "\n".join(lines)
    else:
        state["_expansion_targets"] = ""

    # Advance iteration at the loop boundary — the researcher is now
    # the last agent in each LoopAgent iteration, so incrementing here
    # ensures all agents in the same round share the same iteration tag.
    state["_corpus_iteration"] = state.get("_corpus_iteration", 0) + 1

    return None  # preserve original researcher output


# ---------------------------------------------------------------------------
# Module-level corpus store singleton
# ---------------------------------------------------------------------------
# Each pipeline run gets a file-backed CorpusStore keyed by a session
# marker so concurrent pipelines don't collide.  The DuckDB file persists
# after the run completes, enabling session continuation.

_corpus_stores: dict[str, "CorpusStore"] = {}


def _get_corpus(state: dict) -> "CorpusStore":
    """Get or create the CorpusStore for this session."""
    from models.corpus_store import CorpusStore

    # Use session-level key to isolate concurrent runs
    key = state.get("_corpus_key", "default")
    if key not in _corpus_stores:
        db_path = state.get("_corpus_db_path", "")
        _corpus_stores[key] = CorpusStore(db_path=db_path)
        logger.info("Created new CorpusStore for key=%s db=%s", key, db_path)
    return _corpus_stores[key]


def build_corpus_state(db_path: str = "") -> dict:
    """Return the initial session-state dict for a new pipeline run.

    ``InMemorySessionService`` deep-copies sessions on both ``create_session``
    and ``get_session``, so state set *after* creation is invisible to the
    Runner.  This function builds the dict that must be passed to
    ``create_session(state=…)`` so the Runner sees the keys.

    Call :func:`init_corpus` after session creation to register the
    CorpusStore singleton.

    Args:
        db_path: Optional explicit DuckDB file path.  When empty the
            CorpusStore will auto-generate a timestamped file under
            ``$FINDINGS_DIR/corpora/``.
    """
    import uuid

    key = f"corpus_{uuid.uuid4().hex[:8]}"
    return {
        "_corpus_key": key,
        "_corpus_db_path": db_path,
        "_corpus_iteration": 0,
        "_expansion_targets": "",
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

    db_path = state.get("_corpus_db_path", "")
    _corpus_stores[key] = CorpusStore(db_path=db_path)
    logger.info(
        "Initialised corpus store for key=%s db=%s",
        key, _corpus_stores[key].db_path,
    )


def get_corpus_text(state: dict) -> str:
    """Return the synthesiser-formatted corpus text, or empty string.

    Used to dump partial results when the pipeline stalls before
    the synthesiser can produce its report.
    """
    key = state.get("_corpus_key")
    if key and key in _corpus_stores:
        return _corpus_stores[key].format_for_synthesiser()
    return ""


def run_swarm_synthesis(state: dict) -> str:
    """Run Flock gossip-based swarm synthesis on the corpus.

    For small corpora (<=GOSSIP_THRESHOLD conditions), runs a single
    Flock synthesis pass.  For larger corpora, runs the 3-phase gossip
    swarm: per-angle workers -> peer refinement -> queen merge.

    Returns the synthesised report text, or empty string if the corpus
    is empty or Flock is unavailable.
    """
    key = state.get("_corpus_key")
    if not key or key not in _corpus_stores:
        logger.warning("run_swarm_synthesis: no corpus found for key=%s", key)
        return ""

    corpus = _corpus_stores[key]
    user_query = state.get("user_query", "")

    # Ensure trace context is set for swarm LLM calls
    _c = get_active_collector()
    if _c:
        corpus.set_trace_context(
            session_id=_c.session_id,
            iteration=state.get("_corpus_iteration", 0),
        )

    logger.info(
        "Starting swarm synthesis (corpus has %d conditions, query=%.80s)",
        corpus.count(), user_query,
    )
    result = corpus.synthesise(user_query)
    logger.info(
        "Swarm synthesis complete: %d chars produced",
        len(result),
    )
    return result


def synthesis_condition_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """After-agent callback for the in-loop synthesiser.

    Reads the synthesiser's output (``state["loop_synthesis"]``),
    ingests it back into the CorpusStore via Flock atomisation so
    the synthesised insights are treated equally to raw findings.

    This implements the *fermentation* step: brute expansion
    (researcher) → intelligent structuring (synthesiser) → corpus
    re-ingestion → fuels the next round of expansion.
    """
    state = callback_context.state
    synthesis_text = state.get("loop_synthesis", "")
    if not synthesis_text or not synthesis_text.strip():
        # Still advance the iteration counter even if synthesis is empty,
        # so the next thinker round sees a fresh iteration number.
        state["_corpus_iteration"] = state.get("_corpus_iteration", 0) + 1
        return None

    _ingest_text_into_corpus(state, synthesis_text, "synthesiser")

    # Refresh corpus views so the thinker sees the enriched corpus
    corpus = _get_corpus(state)
    state["research_findings"] = corpus.format_for_thinker()
    state["corpus_for_synthesis"] = corpus.format_for_synthesiser()

    # Save post-synthesis corpus snapshot for iteration diffs
    corpus._save_corpus_snapshot("post_synthesis")

    # Advance iteration at the loop boundary — the synthesiser is the
    # last agent in each LoopAgent iteration, so incrementing here
    # ensures researcher + synthesiser from the same round share the
    # same iteration tag.
    state["_corpus_iteration"] = state.get("_corpus_iteration", 0) + 1

    return None  # preserve original synthesiser output


def cleanup_corpus(state: dict) -> None:
    """Close the CorpusStore connection for a completed pipeline run.

    The DuckDB file is preserved on disk so the corpus can be reopened
    later for session continuation or post-hoc analysis.  Only the
    in-process connection and module-level reference are released.

    Also removes corpus-related keys from *state* so that
    :func:`_init_pipeline_state` properly re-initialises on the next
    pipeline run within the same session.
    """
    key = state.get("_corpus_key")
    if key and key in _corpus_stores:
        corpus = _corpus_stores[key]
        try:
            logger.info(
                "Closing CorpusStore for key=%s  db=%s  (%d conditions)",
                key, corpus.db_path, corpus.count(),
            )
        except Exception:
            logger.warning("Could not log corpus stats before close", exc_info=True)
        corpus.close()
        del _corpus_stores[key]
    # Clear corpus-related state keys so _init_pipeline_state
    # re-initialises cleanly on session reuse.
    # ADK State objects don't support .pop(); use del with guard.
    for k in ("_corpus_key", "_corpus_db_path", "_corpus_iteration",
              "_expansion_targets",
              "research_findings", "corpus_for_synthesis", "loop_synthesis"):
        try:
            del state[k]
        except (KeyError, TypeError, AttributeError):
            pass
