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
# The AG-UI adapter may overwrite session state between requests (the
# frontend sends ``"state": {}`` which can wipe backend-only keys like
# ``_prev_corpus_db_path``).  This module-level variable is the robust
# fallback: it tracks the last closed corpus DB path so that the next
# pipeline run in the same server process can reopen it.
#
# For multi-tenant safety we key by a stable identifier — the user query
# fingerprint — so different conversations don't cross-contaminate.
# ---------------------------------------------------------------------------
_last_corpus_db_path: str = ""
_corpus_continuity_map: dict[str, str] = {}  # query_fingerprint → db_path

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
            [t["strategy"] for t in expansion_targets[:5]],
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

    Reads expansion targets from the corpus table and the thinker's
    research strategy, programmatically fires search APIs (no LLM),
    and ingests results into the corpus.

    This replaces the researcher's search responsibility.  Runs as a
    before_agent_callback on the maestro so searches complete before
    the maestro starts organising.

    ASYNC: ADK awaits this callback (``inspect.isawaitable`` check in
    ``base_agent.py``).  Using ``await`` instead of blocking
    ``future.result()`` keeps the event loop responsive so SSE
    keepalive comments can fire during long search operations.
    """
    import asyncio

    state = callback_context.state
    corpus = _get_corpus(state)
    _c = get_active_collector()

    # Acquire the per-corpus async lock before touching DuckDB.
    # All DuckDB operations (safety-net scoring, swarm cycle, thinker
    # thought admission) go through this lock.  The async lock yields
    # to the event loop, so SSE keepalives still fire.
    corpus_key = state.get("_corpus_key", "default")
    lock = _get_corpus_lock(corpus_key)
    await lock.acquire()
    try:
        pass  # lock acquired — DuckDB is ours
    finally:
        lock.release()

    # Set tracing context
    iteration = state.get("_corpus_iteration", 0)
    if _c:
        corpus.set_trace_context(
            session_id=_c.session_id,
            iteration=iteration,
        )

    # Drain any queued search results from previous tool callbacks
    _drain_search_queue(state)

    # ── THOUGHT LINEAGE (legacy fallback) ────────────────────────────
    # thinker_escalate_callback now writes thoughts directly under the
    # async lock.  This fallback handles any residual pending thoughts
    # from before the migration (or if the async write failed).
    pending_thought = state.pop("_pending_thinker_thought", None)
    if pending_thought:
        try:
            corpus.admit_thought(**pending_thought)
            logger.info(
                "Legacy deferred thought admitted: %d chars at iteration %d",
                len(pending_thought.get("reasoning", "")),
                pending_thought.get("iteration", 0),
            )
        except Exception:
            logger.warning(
                "Failed to admit deferred thinker thought (non-fatal)",
                exc_info=True,
            )

    # Run the automated search executor
    try:
        import threading
        from tools.search_executor import run_search_executor

        cancel_event = threading.Event()

        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            timed_out = False
            try:
                future = pool.submit(
                    asyncio.run,
                    run_search_executor(state, cancel=cancel_event),
                )
                # NON-BLOCKING: wrap the concurrent.futures.Future as
                # an asyncio.Future and await it.  This keeps the event
                # loop responsive so SSE keepalive comments fire during
                # long search operations.  Previously future.result()
                # blocked the event loop for up to 120s.
                _se_timeout = int(os.environ.get("SEARCH_EXECUTOR_TIMEOUT", "300"))
                wrapped = asyncio.wrap_future(future)
                stats = await asyncio.wait_for(wrapped, timeout=_se_timeout)
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning("Search executor timed out after %ds", _se_timeout)
                # Signal the worker to stop touching DuckDB.
                cancel_event.set()
                # Non-blocking wait for the worker to honour the cancel
                # flag — same pattern as the main await above.
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=5)
                except (asyncio.TimeoutError, Exception):
                    pass
                stats = {"timed_out": True}
            finally:
                if not timed_out:
                    cancel_event.set()
                pool.shutdown(wait=False, cancel_futures=True)
        else:
            stats = loop.run_until_complete(
                run_search_executor(state, cancel=cancel_event),
            )

        logger.info("Search executor stats: %s", stats)

        # Emit search executor stats to dashboard
        if _c and isinstance(stats, dict):
            _c.emit_event("search_executor", data=stats)

    except Exception as exc:
        logger.warning("Search executor failed (non-fatal): %s", exc, exc_info=True)

    # Update state with views for each downstream consumer.
    # The maestro gets a compact structural summary (counts, status,
    # expansion targets) — it has SQL access for full detail.
    # The thinker gets the full corpus briefing (set by maestro_condition_callback).
    # The synthesiser gets the chunk-grouped format (also set by maestro_condition_callback).
    state["corpus_summary_for_maestro"] = corpus.format_summary_for_maestro()

    return None


# ---------------------------------------------------------------------------
# Maestro callback (new maestro architecture)
# ---------------------------------------------------------------------------

async def maestro_condition_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """After-agent callback for the maestro: refresh corpus state.

    The maestro has already modified the corpus directly via
    ``execute_flock_sql()``.  This callback:

    1. Drains any remaining search results from tool callbacks
    2. Runs safety-net scoring under the per-corpus ``asyncio.Lock``
       (DuckDB work dispatched to thread pool, event loop stays responsive)
    3. Refreshes state with the current corpus for the thinker
    4. Advances the iteration counter

    ASYNC: Uses ``_safe_corpus_write()`` for DuckDB operations, replacing
    the old fire-and-forget background thread pattern.  ADK awaits async
    callbacks (``inspect.isawaitable`` check in ``base_agent.py``).
    """
    state = callback_context.state
    corpus = _get_corpus(state)
    _c = get_active_collector()

    # Drain any remaining queued search results
    _drain_search_queue(state)

    # ── THOUGHT LINEAGE: Preserve maestro's reasoning ─────────────────
    # The maestro's text output (its reasoning about what SQL operations
    # to perform and why) would otherwise be overwritten when we refresh
    # state["research_findings"] below.  Store it as an immutable thought
    # row so the full reasoning chain is preserved in the Flock table.
    maestro_output = state.get("research_findings", "")
    corpus_key = state.get("_corpus_key", "default")
    if maestro_output and maestro_output.strip():
        iteration = state.get("_corpus_iteration", 0)
        try:
            await _safe_corpus_write(
                corpus_key,
                lambda: corpus.admit_thought(
                    reasoning=maestro_output,
                    angle="maestro_reasoning",
                    strategy=f"maestro_iteration_{iteration}",
                    iteration=iteration,
                ),
            )
            logger.info(
                "Maestro reasoning preserved: %d chars at iteration %d",
                len(maestro_output), iteration,
            )
        except Exception:
            logger.warning(
                "Failed to preserve maestro reasoning (non-fatal)",
                exc_info=True,
            )

    # Scoring safety net: ensure all conditions ingested during the
    # search executor + maestro phase are scored and deduped.  The
    # maestro may have created new rows via execute_flock_sql() that
    # bypassed the normal ingestion scoring path.
    user_query = state.get("user_query", "")

    # Snapshot corpus state BEFORE dispatching the scoring work to the
    # thread pool.  The thinker (next iteration) sees this pre-scoring
    # snapshot.  Unscored conditions appear with composite_quality=0
    # ("WEAK FINDINGS" tier).  This is acceptable because:
    #  1. Safety-net scoring is a backup — the maestro handles most
    #     scoring itself via execute_flock_sql().
    #  2. search_executor_callback waits for scoring and refreshes state
    #     before the maestro runs, so the maestro always sees scored data.
    iteration = state.get("_corpus_iteration", 0)
    thinker_briefing = corpus.format_for_thinker(current_iteration=iteration)

    # ── Serendipity: Devil’s Advocate injection ───────────────────────────
    # When the corpus has been building for 2+ iterations and shows
    # strong consensus with no contradictions, inject a sceptic’s prompt
    # to push the thinker toward unexplored territory.  This costs zero
    # LLM calls — just a conditional string append.
    thinker_briefing = _maybe_inject_devils_advocate(
        thinker_briefing, corpus, iteration,
    )

    state["research_findings"] = thinker_briefing
    state["corpus_for_synthesis"] = corpus.format_for_synthesiser()
    state["corpus_summary_for_maestro"] = corpus.format_summary_for_maestro()

    # Emit corpus stats to dashboard (also before thread start)
    if _c:
        try:
            total = corpus.count()
            iteration = state.get("_corpus_iteration", 0)
            # Pass 0 for admitted — the maestro organises existing
            # conditions, it doesn't ingest new ones.
            _c.corpus_update(0, total, iteration)
            _c.emit_event("maestro_complete", data={
                "total_conditions": total,
                "iteration": iteration,
            })
        except Exception:
            pass

    def _safety_net_scoring() -> None:
        """Run scoring + dedup (called via asyncio.to_thread under the lock).

        Skips conditions already scored by the maestro via
        execute_flock_sql().  ``score_new_conditions`` checks
        ``score_version = 0`` so it naturally skips maestro-scored rows.
        """
        try:
            scored = corpus.score_new_conditions(user_query)
            if scored:
                logger.info("Maestro safety-net: scored %d conditions", scored)

            deduped = corpus.compute_duplications()
            if deduped:
                logger.info("Maestro safety-net: deduped %d pairs", deduped)

            if scored:
                corpus.compute_composite_quality()
        except Exception:
            logger.warning("Maestro safety-net scoring failed", exc_info=True)

    # Run safety-net scoring under the per-corpus async lock.
    # This replaces the old fire-and-forget thread pattern that could
    # skip scoring entirely if the previous thread was still alive.
    # The async lock yields to the event loop (SSE keepalives fire),
    # and asyncio.to_thread() dispatches the actual DuckDB work to the
    # thread pool so the event loop is never blocked.
    try:
        await _safe_corpus_write(corpus_key, _safety_net_scoring)
    except Exception:
        logger.warning("Async safety-net scoring failed", exc_info=True)

    # Advance iteration at the loop boundary — the maestro is the last
    # agent in each LoopAgent iteration.
    iteration = state.get("_corpus_iteration", 0)

    # ── Periodic synthesis ────────────────────────────────────────────
    # After refreshing corpus state and before advancing the iteration
    # counter, run a swarm synthesis and persist it as a thought row so
    # the thinker can integrate cross-angle insights.
    synthesis_interval = int(os.environ.get("SYNTHESIS_INTERVAL", "1"))
    if iteration > 0 and iteration % synthesis_interval == 0:
        try:
            swarm_report = run_swarm_synthesis(state)
            if swarm_report and swarm_report.strip():
                corpus = _get_corpus(state)
                await _safe_corpus_write(
                    corpus_key,
                    lambda: corpus.admit_thought(
                        reasoning=swarm_report,
                        angle="periodic_synthesis",
                        strategy="periodic_synthesis_report",
                        iteration=iteration,
                    ),
                )
        except Exception:
            logger.warning("Periodic synthesis failed (non-fatal)", exc_info=True)

    # ── Thought swarm cycle (async offload) ─────────────────────────────
    # Spawn parallel specialist thinkers for angles identified by the
    # main thinker, then arbitrate competing conclusions and split broad
    # thoughts into focused sub-claims.
    #
    # Runs as a fire-and-forget ``asyncio.Task`` that acquires the
    # per-corpus async lock before dispatching ``run_swarm_cycle`` to the
    # thread pool via ``_safe_corpus_write``.  This replaces the old bare
    # ``threading.Thread`` pattern which bypassed the async lock protocol
    # entirely, creating a race between the swarm thread and any
    # ``asyncio.to_thread()`` work dispatched by ``_safe_corpus_write``
    # in ``thinker_escalate_callback`` or ``search_executor_callback``.
    #
    # The async lock ensures only one coroutine's DuckDB work runs at a
    # time.  If the swarm is still running when the next iteration's
    # ``thinker_escalate_callback`` fires, the callback simply waits for
    # the lock (yields to event loop, SSE keepalives still fire) instead
    # of racing.  No data loss, no DuckDB connection corruption.
    #
    # Results are visible to the thinker on the NEXT iteration (eventual
    # consistency, not blocking the callback return).
    try:
        from tools.swarm_thinkers import run_swarm_cycle
        corpus = _get_corpus(state)

        # Capture state values the swarm cycle needs (avoid holding a
        # reference to the full mutable state dict across async tasks).
        swarm_state_snapshot = {
            "_corpus_iteration": iteration,
            "user_query": state.get("user_query", ""),
            "research_strategy": state.get("research_strategy", ""),
        }

        async def _async_bg_swarm() -> None:
            """Background swarm — runs under the per-corpus async lock.

            ``_safe_corpus_write`` acquires the ``asyncio.Lock`` then
            dispatches the blocking ``run_swarm_cycle`` call to the
            thread pool via ``asyncio.to_thread()``.  This gives us:
              - No event loop blocking (async lock yields)
              - No DuckDB races (lock serialises all corpus access)
              - Thread safety for internal specialist parallelism
                (``CorpusStore._write_lock`` still serialises writes
                within the swarm's own ``ThreadPoolExecutor``)
            """
            try:
                swarm_ids = await _safe_corpus_write(
                    corpus_key,
                    lambda: run_swarm_cycle(swarm_state_snapshot, corpus),
                )
                if swarm_ids:
                    logger.info(
                        "Background swarm produced %d new thoughts at iter %d",
                        len(swarm_ids), iteration,
                    )
            except Exception:
                logger.warning("Background swarm cycle failed (non-fatal)", exc_info=True)

        # Warn if the previous iteration's swarm task is still running.
        prev_task = _swarm_tasks.get(corpus_key)
        if prev_task is not None and not prev_task.done():
            logger.warning(
                "Previous swarm task (key=%s) still running — new task "
                "will queue behind it on the async lock",
                corpus_key,
            )

        task = asyncio.create_task(
            _async_bg_swarm(),
            name=f"swarm-iter-{iteration}",
        )
        _swarm_tasks[corpus_key] = task
        logger.debug("Swarm cycle dispatched as async task (iter=%d)", iteration)
    except Exception:
        logger.warning("Swarm dispatch failed (non-fatal)", exc_info=True)

    state["_corpus_iteration"] = iteration + 1

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
        # Cumulative cost tracking (P3) — only set if not already present
        # (preserved across runs by cleanup_corpus)
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
    corpus_key = state.get("_corpus_key", "default")
    if not _wait_for_pending_scoring(corpus_key):
        logger.warning("get_corpus_text: scoring thread still alive — returning cached state")
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
    """
    key = state.get("_corpus_key")
    if not key or key not in _corpus_stores:
        logger.warning("run_swarm_synthesis: no corpus found for key=%s", key)
        return ""

    corpus = _corpus_stores[key]
    user_query = state.get("user_query", "")

    # Non-blocking check: is scoring still running?
    corpus_key = key  # reuse the key we already validated above
    if not _wait_for_pending_scoring(corpus_key):
        logger.warning(
            "run_swarm_synthesis: scoring thread still alive — "
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
    """
    # Best-effort wait for background scoring before closing the DB.
    # This is the ONE place where a blocking join is acceptable —
    # the pipeline is finishing and no more SSE events will be sent.
    corpus_key = state.get("_corpus_key", "default")
    with _scoring_lock:
        _cleanup_t = _scoring_threads.get(corpus_key)
    if _cleanup_t is not None and _cleanup_t.is_alive():
        logger.info("cleanup_corpus: waiting up to 30s for scoring thread (key=%s)", corpus_key)
        _cleanup_t.join(timeout=30)
    scoring_done = _cleanup_t is None or not _cleanup_t.is_alive()
    with _scoring_lock:
        if _scoring_threads.get(corpus_key) is _cleanup_t:
            _scoring_threads.pop(corpus_key, None)
    key = state.get("_corpus_key")
    if key and key in _corpus_stores:
        corpus = _corpus_stores[key]
        # ── Corpus continuity: preserve the DB path for the next run ──
        # Save to BOTH state AND module-level variables.  The AG-UI
        # adapter may overwrite session state between requests, so the
        # module-level fallback ensures continuity even when state is lost.
        global _last_corpus_db_path
        try:
            db_path_str = str(corpus.db_path)
            state["_prev_corpus_db_path"] = db_path_str
            _last_corpus_db_path = db_path_str

            # Also save to per-query map for multi-tenant safety
            query_fp = state.get("user_query", "")[:200].strip().lower()
            if query_fp:
                _corpus_continuity_map[query_fp] = db_path_str

            logger.info(
                "cleanup_corpus: saved corpus path for continuity: %s "
                "(module-level + state + query-map)",
                db_path_str,
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

        if not scoring_done:
            # Scoring thread still alive — do NOT close the connection
            # while it's mid-query.  Drop our reference and let the
            # daemon thread finish naturally; the connection will be
            # garbage-collected once the thread exits.
            logger.warning(
                "cleanup_corpus: scoring thread still alive after 30s — "
                "abandoning CorpusStore for key=%s (will GC after thread exits)",
                key,
            )
            del _corpus_stores[key]
        else:
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
