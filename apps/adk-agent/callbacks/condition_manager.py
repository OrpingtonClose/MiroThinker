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

import asyncio
import logging
import os
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
# Parallel tool callbacks append here; search_executor_callback drains
# on the main thread where DuckDB access is safe.
# ---------------------------------------------------------------------------
_pending_search_results: list[tuple[str, str, str, int]] = []  # (corpus_key, text, source_type, iteration)
_pending_search_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Module-level corpus continuity — survives across AG-UI HTTP requests.
#
# Phase 1: The ad-hoc variables are DEPRECATED and REMOVED.
# Corpus continuity is now handled by PipelineStateRegistry
# (models/pipeline_state.py).  If you need corpus continuity,
# use PipelineStateRegistry.instance().lookup() instead.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-corpus async locks — serialise all DuckDB writes through an
# asyncio.Lock so waiting coroutines yield to the event loop instead of
# blocking it.  Actual DuckDB work runs in the thread pool via
# asyncio.to_thread().  This replaces the old fire-and-forget scoring
# threads + non-blocking skip pattern, eliminating data loss.
#
# Keyed by corpus_key so concurrent sessions don't block each other.
# ---------------------------------------------------------------------------
_corpus_async_locks: dict[str, asyncio.Lock] = {}

# Fire-and-forget async tasks for background swarm cycles.  Tracked so
# we can log warnings if a previous iteration's task is still running.
_swarm_tasks: dict[str, "asyncio.Task[None]"] = {}


def queue_search_result(
    corpus_key: str, text: str, source_type: str, iteration: int,
) -> None:
    """Thread-safe enqueue of a search result for later corpus ingestion.

    Called from ``after_tool_callback`` which may fire from parallel
    tool threads.  The actual ``ingest_raw()`` happens in
    ``search_executor_callback`` on the main thread.
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
            admitted = _ingest_only(state, text, source_type)
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


def _ingest_only(
    state: dict,
    text: str,
    source_type: str,
) -> int:
    """Atomise *text* and ingest into the corpus — no scoring or battery.

    This is the lightweight ingestion path.  Scoring, dedup, and the
    full algorithm battery should be run **once** after all ingestion
    for the iteration is complete (see :func:`_score_and_battery`).

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

    user_query = state.get("user_query", "")
    ids = corpus.ingest_raw(
        raw_text=text,
        source_type=source_type,
        source_ref="",
        angle=f"iteration_{iteration}",
        iteration=iteration,
        user_query=user_query,
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

    return admitted_count


def _score_and_battery(state: dict) -> dict:
    """Run scoring, dedup, and the full algorithm battery once.

    Call this **after** all ingestion for the current iteration is
    complete.  Previously scoring ran inside the ingestion function
    which was called per-source, causing N redundant battery runs
    (each with hundreds of LLM calls) and making the pipeline appear
    to hang.

    Returns the battery results dict.
    """
    corpus = _get_corpus(state)
    iteration = state.get("_corpus_iteration", 0)
    user_query = state.get("user_query", "")
    _c = get_active_collector()

    # Score & dedup
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
            [t["strategy"] for t in expansion_targets],
        )

    return battery_results


# ---------------------------------------------------------------------------
# Search Executor callback
# ---------------------------------------------------------------------------

def _get_corpus_lock(corpus_key: str) -> asyncio.Lock:
    """Get or create the per-corpus asyncio.Lock.

    Uses the module-level ``_corpus_async_locks`` dict.  Creating a new
    Lock is cheap and idempotent — the first caller for a given key
    creates it, subsequent callers reuse it.
    """
    lock = _corpus_async_locks.get(corpus_key)
    if lock is None:
        lock = asyncio.Lock()
        _corpus_async_locks[corpus_key] = lock
    return lock


async def _safe_corpus_write(
    corpus_key: str,
    fn: "object",
    *args: "object",
) -> "object":
    """Execute a corpus-writing function with async mutual exclusion.

    Acquires the per-corpus ``asyncio.Lock`` (yields to event loop, never
    blocks), then dispatches the actual DuckDB work to the thread pool
    via ``asyncio.to_thread()``.

    This gives us:
      - No event loop blocking (asyncio.Lock yields)
      - No data loss (coroutines wait their turn instead of skipping)
      - Thread safety (only one thread touches DuckDB at a time)
    """
    lock = _get_corpus_lock(corpus_key)
    async with lock:
        return await asyncio.to_thread(fn, *args)  # type: ignore[arg-type]



async def search_executor_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Before-agent callback: run the automated search executor.

    Thin wrapper that delegates to ``SearchExecutorBlock`` via the block
    adapter.  All business logic lives in the block; cross-cutting
    concerns (timing, health, I/O validation, DuckDB safety, error
    escalation) are handled by aspects.

    ASYNC: ADK awaits this callback (``inspect.isawaitable`` check in
    ``base_agent.py``).
    """
    from callbacks.block_adapter import run_block_from_callback
    from models.pipeline_block import RoutingHint
    result = await run_block_from_callback("search_executor", callback_context)

    # Propagate ABORT routing — store in state so downstream callbacks
    # (maestro, swarm, synthesiser) can check and skip expensive work.
    if result.routing == RoutingHint.ABORT:
        callback_context.state["_pipeline_aborted"] = True
        callback_context.state["_abort_reason"] = result.diagnosis or "search_executor abort"
        logger.error("Pipeline ABORT signalled by search_executor: %s", result.diagnosis)

    return None


# ---------------------------------------------------------------------------
# Maestro callback (new maestro architecture)
# ---------------------------------------------------------------------------

async def maestro_condition_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """After-agent callback for the maestro: refresh corpus state.

    Thin wrapper that delegates to ``MaestroBlock`` via the block adapter.
    All business logic (thought preservation, scoring, state refresh,
    iteration advance) lives in the block; cross-cutting concerns are
    handled by aspects.

    ASYNC: ADK awaits async callbacks (``inspect.isawaitable`` check in
    ``base_agent.py``).
    """
    from callbacks.block_adapter import run_block_from_callback
    from models.pipeline_block import RoutingHint
    result = await run_block_from_callback("maestro", callback_context)

    # Propagate ABORT routing — escalate to break out of the LoopAgent
    # so the pipeline doesn't continue with corrupted/incomplete data.
    if result.routing == RoutingHint.ABORT:
        callback_context.state["_pipeline_aborted"] = True
        callback_context.state["_abort_reason"] = result.diagnosis or "maestro abort"
        callback_context.actions.escalate = True
        logger.error("Pipeline ABORT signalled by maestro — escalating: %s", result.diagnosis)

    return None  # preserve maestro output
# ---------------------------------------------------------------------------
# Serendipity: Devil's Advocate injection
# ---------------------------------------------------------------------------

def _maybe_inject_devils_advocate(
    briefing: str,
    corpus: "CorpusStore",
    iteration: int,
) -> str:
    """Inject a devil's advocate prompt when the corpus shows one-sided consensus.

    After 2+ iterations, if the corpus has many findings but zero
    contradictions detected, the research may be stuck in a confirmation
    bubble.  This injects a sceptic's prompt into the thinker's briefing
    to push toward unexplored counter-evidence.

    Zero LLM calls — pure SQL check + conditional string append.

    Returns the (possibly augmented) briefing string.
    """
    if not (os.environ.get("SERENDIPITY_ENABLED", "1") == "1"):
        return briefing
    if iteration < 2:
        return briefing

    try:
        # Check: are there any contradictions?
        contradiction_count = corpus.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE contradiction_flag = TRUE AND consider_for_use = TRUE"
        ).fetchone()[0]

        # Check: how many findings exist?
        finding_count = corpus.conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE row_type = 'finding' AND consider_for_use = TRUE"
        ).fetchone()[0]

        # Only inject if we have substantial findings but zero contradictions
        if contradiction_count > 0 or finding_count < 10:
            return briefing

        # Identify the dominant consensus by looking at the most common angle
        top_angle = corpus.conn.execute(
            "SELECT angle, COUNT(*) as cnt FROM conditions "
            "WHERE row_type = 'finding' AND consider_for_use = TRUE "
            "AND angle IS NOT NULL AND angle != '' "
            "GROUP BY angle ORDER BY cnt DESC LIMIT 1"
        ).fetchone()
        angle_hint = f" on '{top_angle[0]}'" if top_angle else ""

        devils_advocate = (
            "\n\n" + "=" * 60 + "\n"
            "DEVIL'S ADVOCATE (serendipity injection)\n"
            + "=" * 60 + "\n"
            f"After {iteration} iterations and {finding_count} findings, "
            f"the corpus shows ZERO contradictions{angle_hint}. "
            "This level of consensus is suspicious — real research topics "
            "almost always have dissenting views, methodological critiques, "
            "or competing explanations.\n\n"
            "Consider:\n"
            "- What counter-evidence SHOULD exist but hasn't been found?\n"
            "- What assumptions are ALL sources sharing uncritically?\n"
            "- What would a rigorous sceptic ask about these findings?\n"
            "- Are there industries, institutions, or ideologies that "
            "would challenge this consensus?\n"
            "- What was the PREVIOUS consensus before current understanding, "
            "and why did it change?\n\n"
            "Actively search for DISSENT, CRITICISM, and ALTERNATIVE "
            "EXPLANATIONS in your next strategy. The best research "
            "steelmans the opposition.\n"
            + "=" * 60 + "\n"
        )

        logger.info(
            "Serendipity: injected devil's advocate at iteration %d "
            "(%d findings, 0 contradictions)",
            iteration, finding_count,
        )
        return briefing + devils_advocate

    except Exception:
        logger.debug("Devil's advocate check failed (non-fatal)", exc_info=True)
        return briefing


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
        "corpus_summary_for_maestro": "(empty corpus — no conditions to organise)",
        # Iteration context for thinker (P1)
        "_prev_thinker_strategies": "(first iteration — no previous strategies)",
        "_last_thinker_strategy": "",
        # NOTE: _cumulative_api_cost is NOT included here.  It is
        # initialised by _init_pipeline_state (pipeline.py) which
        # uses a conditional guard to preserve accumulated cost on
        # continuation runs.  For the CLI run_pipeline() path,
        # it is set explicitly alongside user_query.
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

    THREAD-SAFETY: This is a synchronous function that cannot acquire
    the async lock.  The ``lock.locked()`` check is a best-effort
    TOCTOU guard.  This is safe because callers run synchronously on
    the event loop thread, blocking any async task from acquiring the
    lock between the check and the DuckDB read.  Do NOT call this
    function from an async context with intervening ``await`` points.
    """
    corpus_key = state.get("_corpus_key", "default")
    lock = _get_corpus_lock(corpus_key)
    if lock.locked():
        logger.warning("get_corpus_text: async lock held — returning cached state")
        return state.get("corpus_for_synthesis", "")
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

    THREAD-SAFETY: This is a synchronous function that cannot acquire
    the async lock.  The ``lock.locked()`` check is a best-effort
    TOCTOU guard.  This is safe because callers run synchronously on
    the event loop thread, blocking any async task from acquiring the
    lock between the check and the DuckDB access.  Do NOT call this
    function from an async context with intervening ``await`` points.
    """
    key = state.get("_corpus_key")
    if not key or key not in _corpus_stores:
        logger.warning("run_swarm_synthesis: no corpus found for key=%s", key)
        return ""

    corpus = _corpus_stores[key]
    user_query = state.get("user_query", "")

    # Non-blocking check: is the async lock held (DuckDB in use)?
    corpus_key = key  # reuse the key we already validated above
    lock = _get_corpus_lock(corpus_key)
    if lock.locked():
        logger.warning(
            "run_swarm_synthesis: async lock held — "
            "returning cached corpus_for_synthesis to avoid concurrent DuckDB access",
        )
        return state.get("corpus_for_synthesis", "")

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


def cleanup_corpus(state: dict) -> None:
    """Close the CorpusStore connection for a completed pipeline run.

    The DuckDB file is preserved on disk so the corpus can be reopened
    later for session continuation or post-hoc analysis.  Only the
    in-process connection and module-level reference are released.

    Also removes corpus-related keys from *state* so that
    :func:`_init_pipeline_state` properly re-initialises on the next
    pipeline run within the same session.

    **Architectural guardrail: mandatory scoring gate.**  If the pipeline
    stalled or exited before the maestro could score, this function
    forces scoring so the corpus is never left with all findings at
    -1.0 (the "dumpster fire" state the diagnostic tool couldn't catch).
    """
    # Best-effort wait for background swarm task before closing the DB.
    # The pipeline is finishing — no more SSE events will be sent.
    corpus_key = state.get("_corpus_key", "default")
    swarm_task = _swarm_tasks.pop(corpus_key, None)
    task_done = True
    if swarm_task is not None and not swarm_task.done():
        logger.info("cleanup_corpus: cancelling pending swarm task (key=%s)", corpus_key)
        swarm_task.cancel()
        # Note: we cannot await here (sync function) and must NOT use
        # time.sleep() — that would block the event loop, preventing
        # asyncio from delivering the CancelledError to the task.
        # Just check done() immediately; cancel() is synchronous and
        # if the task was waiting on the async lock it will already be
        # marked done.  If not, we accept the limitation and abandon
        # the CorpusStore below (connection will GC after task completes).
        task_done = swarm_task.done()
    # Clean up the per-corpus async lock entry
    _corpus_async_locks.pop(corpus_key, None)
    key = state.get("_corpus_key")
    if key and key in _corpus_stores:
        corpus = _corpus_stores[key]

        # ── Corpus continuity: preserve the DB path for the next run ──
        # Uses the typed PipelineStateRegistry (Phase 1) instead of ad-hoc
        # module-level variables.  The registry is thread-safe, TTL-aware,
        # and indexed by query fingerprint.
        try:
            from models.pipeline_state import PipelineStateRegistry
            db_path_str = str(corpus.db_path)
            state["_prev_corpus_db_path"] = db_path_str
            user_query = state.get("user_query", "")
            iteration = state.get("_corpus_iteration", 0)
            registry = PipelineStateRegistry.instance()
            registry.register(
                query=user_query,
                db_path=db_path_str,
                iteration_count=iteration,
            )
            logger.info(
                "cleanup_corpus: saved corpus path for continuity: %s "
                "(registry + state, iter=%d)",
                db_path_str, iteration,
            )
        except Exception:
            logger.warning("cleanup_corpus: failed to save continuity path", exc_info=True)

        # ── P1: Save previous synthesiser report for next iteration ──
        # The synthesiser can build on the previous report rather than
        # starting fresh each time.
        try:
            synth = state.get("corpus_for_synthesis", "")
            if synth and synth.strip() and synth != "(no findings)":
                state["_prev_synthesiser_report"] = synth
        except Exception:
            pass

        # ── P3: Accumulate cost from this run ──
        try:
            from tools.cost_tracker import get_session_cost
            run_cost = get_session_cost()
            prev = state.get("_cumulative_api_cost", 0.0)
            state["_cumulative_api_cost"] = prev + run_cost
            logger.info(
                "cleanup_corpus: cumulative cost %.4f (this run %.4f)",
                prev + run_cost, run_cost,
            )
        except Exception:
            pass

        if not task_done:
            # Swarm task still running — do NOT close the connection
            # while it's mid-query.  Drop our reference and let the
            # task finish naturally; the connection will be
            # garbage-collected once the task completes.
            logger.warning(
                "cleanup_corpus: swarm task still running — "
                "abandoning CorpusStore for key=%s (will GC after task completes)",
                key,
            )
            logger.warning(
                "Mandatory scoring gate SKIPPED — scoring thread still "
                "alive, cannot safely access DuckDB concurrently",
            )
            del _corpus_stores[key]
        else:
            # ── Mandatory scoring gate (architectural guardrail) ──
            # If the pipeline stalled before maestro ran, findings sit at
            # composite_quality=-1.0 (unscored).  Force scoring now so
            # the corpus is never left in a dumpster-fire state.
            # IMPORTANT: This MUST only run when scoring_done=True to
            # avoid concurrent DuckDB access (not thread-safe).
            try:
                unscored = corpus.conn.execute(
                    "SELECT COUNT(*) FROM conditions "
                    "WHERE row_type = 'finding' AND score_version = 0"
                ).fetchone()[0]
                total_findings = corpus.conn.execute(
                    "SELECT COUNT(*) FROM conditions "
                    "WHERE row_type = 'finding'"
                ).fetchone()[0]
                if unscored > 0 and total_findings > 0:
                    unscored_pct = unscored / total_findings * 100
                    logger.error(
                        "MANDATORY SCORING GATE: %d/%d findings "
                        "(%.0f%%) are unscored (score_version=0) "
                        "— forcing scoring now",
                        unscored, total_findings, unscored_pct,
                    )
                    user_query = state.get("user_query", "")
                    scored = corpus.score_new_conditions(user_query)
                    if scored:
                        logger.info(
                            "Mandatory scoring gate: scored %d "
                            "conditions", scored,
                        )
                    deduped = corpus.compute_duplications()
                    if deduped:
                        logger.info(
                            "Mandatory scoring gate: deduped %d "
                            "pairs", deduped,
                        )
                    try:
                        corpus.compute_composite_quality()
                    except Exception:
                        pass
                    logger.info("Mandatory scoring gate complete")
            except Exception:
                logger.warning(
                    "Mandatory scoring gate failed (non-fatal)",
                    exc_info=True,
                )

            try:
                logger.info(
                    "Closing CorpusStore for key=%s  db=%s  (%d conditions)",
                    key, corpus.db_path, corpus.count(),
                )
            except Exception:
                logger.warning("Could not log corpus stats before close", exc_info=True)
            corpus.close()
            del _corpus_stores[key]
    # No state cleaning — corpus keys (_corpus_key, _corpus_db_path,
    # _corpus_iteration, etc.) persist in session state so follow-up
    # messages reopen the same DuckDB file.  _get_corpus() lazily
    # recreates the CorpusStore from the surviving _corpus_db_path.
    # "Runs are episodes, the swarm is eternal." (PR #54 architecture)

    # Reset the swarm router so the next pipeline run starts fresh.
    try:
        from tools.swarm_thinkers import reset_swarm_router
        reset_swarm_router()
    except Exception:
        pass
