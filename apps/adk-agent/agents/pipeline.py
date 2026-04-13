# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Blackboard research pipeline: LoopAgent(Thinker → Maestro) + Synthesiser.

Uses ADK's ``LoopAgent`` to implement an iterative research cycle where
context grows richer on every iteration, then a ``SequentialAgent`` to
run the final synthesiser after the loop exits.

Architecture::

    SequentialAgent("mirothinker_pipeline")
    └── LoopAgent("research_loop", max_iterations=3)
    │     ├── Agent("thinker")           # pure reasoning, no tools
    │     └── Agent("maestro")           # free-form Flock conductor
    │           ├── before_agent_callback: search_executor_callback
    │           │     (automated API calls — no LLM, reads expansion
    │           │      targets + strategy queries, fires search APIs)
    │           └── after_agent_callback: maestro_condition_callback
    └── Agent("synthesiser")                # final report, uncensored

The loop implements a 3-phase cycle per iteration:

  1. **Strategy** (thinker): reads the enriched corpus (verbal prose
     briefing with tiered findings, contradictions, and under-explored
     areas), reasons deeply about gaps and emerging narratives, and
     plans the next expansion round.

  2. **Search Executor** (automated, runs as maestro's before_agent):
     reads expansion_tool + expansion_hint from corpus table AND
     extracts queries from the thinker's strategy text.  Fires APIs
     programmatically (no LLM involved).  Results are ingested into
     the corpus via ``ingest_raw()``.

  3. **Maestro** (free-form Flock conductor): has a single tool
     ``execute_flock_sql(query)`` for unrestricted SQL/Flock access.
     Can invent new columns, create new rows, update flags, run Flock
     LLM functions — whatever operations the corpus needs based on
     its current state.  Replaces the fixed 11-step algorithm battery.

Data flows via session state (blackboard):

  - **Thinker** reads ``{research_findings}`` — a verbal prose briefing
    of findings organised by quality tier (strong/moderate/weak),
    with contradictions, under-explored areas, and corpus health.
    The thinker outputs strategy to ``state["research_strategy"]``.
    When evidence is sufficient it outputs ``EVIDENCE_SUFFICIENT`` and an
    ``after_agent_callback`` sets ``escalate=True`` to break the loop.

  - **Search Executor** reads expansion targets from the corpus table
    and extracts queries from ``state["research_strategy"]``.  Fires
    APIs and ingests results.  No LLM — purely programmatic.

  - **Maestro** reads ``{research_findings}`` + ``{research_strategy}``,
    uses ``execute_flock_sql()`` to run arbitrary Flock/SQL operations
    on the corpus.  Its ``after_agent_callback`` refreshes state with
    the updated corpus for the thinker's next iteration.

  - **Final synthesiser** (runs once after the loop) reads
    ``{corpus_for_synthesis}`` — the swarm-synthesised report — and
    writes the polished final report.

The thinker and synthesiser NEVER touch tool format — they are pure
text-in/text-out LLM agents using the uncensored model.  Only the
maestro has tool-calling capability (execute_flock_sql).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.loop_agent import LoopAgent
from google.genai import types as genai_types

from agents.thinker import thinker_agent
from agents.maestro import maestro_agent
from agents.synthesiser import synthesiser_agent
from callbacks.condition_manager import (
    build_corpus_state,
    cleanup_corpus,
    init_corpus,
    run_swarm_synthesis,
    search_executor_callback,
)

logger = logging.getLogger(__name__)


def _init_pipeline_state(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Ensure session state has corpus keys before the pipeline starts.

    AG-UI creates sessions without initial state, so the first time the
    pipeline runs we inject the keys that thinker/researcher/synthesiser
    read via ``{research_findings}`` / ``{corpus_for_synthesis}`` template
    variables.  Also registers the CorpusStore singleton for DuckDB.
    """
    state = callback_context.state
    if "_corpus_key" not in state:
        # ── Corpus continuity: reopen the previous run's corpus ──
        # Try state first, then module-level fallbacks (query map, last path).
        # The AG-UI adapter may overwrite session state between requests,
        # so the module-level variables are the robust fallback.
        from callbacks.condition_manager import (
            _corpus_continuity_map,
            _last_corpus_db_path,
        )

        prev_db = state.get("_prev_corpus_db_path", "")
        if not prev_db:
            # Fallback 1: query-fingerprint map (multi-tenant safe)
            query_fp = state.get("user_query", "")[:200].strip().lower()
            if query_fp and query_fp in _corpus_continuity_map:
                prev_db = _corpus_continuity_map[query_fp]
                logger.info(
                    "Corpus continuity (query-map fallback): %s",
                    prev_db,
                )
        if not prev_db and _last_corpus_db_path:
            # Fallback 2: last corpus in this process (single-tenant)
            prev_db = _last_corpus_db_path
            logger.info(
                "Corpus continuity (module-level fallback): %s",
                prev_db,
            )

        # Verify the file actually exists before trying to reopen
        import os as _os
        if prev_db and not _os.path.exists(prev_db):
            logger.warning(
                "Corpus continuity: previous DB file not found: %s",
                prev_db,
            )
            prev_db = ""

        is_continuation = bool(prev_db)
        if prev_db:
            logger.info(
                "Corpus continuity: reopening previous corpus at %s",
                prev_db,
            )
        for k, v in build_corpus_state(db_path=prev_db).items():
            state[k] = v
        init_corpus(state)
        logger.info("Pipeline state initialised: corpus_key=%s", state["_corpus_key"])

        # Track cumulative cost across expansion iterations
        if "_cumulative_api_cost" not in state:
            state["_cumulative_api_cost"] = 0.0

        # Reset per-session cost tracker so each pipeline run starts fresh
        try:
            from tools.cost_tracker import reset_session_tracker
            reset_session_tracker()
        except Exception:
            pass

        # ── Architectural guardrail: reset circuit breakers ──
        # Each pipeline run starts with a clean slate — APIs that were
        # tripped in a previous run get a fresh chance.
        try:
            from tools.search_executor import reset_circuit_breakers
            reset_circuit_breakers()
        except Exception:
            pass

        # ── Pipeline health model ──
        # Create a fresh PipelineHealth tracker for this run.
        # Every phase reports structured health data here; the health
        # gate at each boundary validates phase contracts.
        try:
            from models.pipeline_health import PipelineHealth
            health = PipelineHealth()
            health.save(state)
        except Exception:
            logger.warning("Failed to initialise PipelineHealth", exc_info=True)

        # ── P0: Skip scout on corpus re-open ──
        # The scout decomposes the query and probes with cheap searches.
        # On continuation runs the thinker already has the full corpus,
        # so re-running the scout wastes 30-60s and may overwrite the
        # thinker's accumulated context with a fresh landscape assessment.
        if is_continuation:
            logger.info(
                "Skipping scout — corpus continuity re-open "
                "(previous corpus has existing findings)"
            )
        else:
            # Run Phase 0 scout only on fresh pipelines
            query = state.get("user_query", "")
            if query:
                from tools.scout import run_scout_phase
                import threading as _threading

                # Cancel event prevents a timed-out scout from writing to
                # state after the research loop has started.
                cancel_event = _threading.Event()

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        try:
                            future = pool.submit(
                                asyncio.run,
                                run_scout_phase(query, state, _cancel=cancel_event),
                            )
                            future.result(timeout=90)
                        finally:
                            cancel_event.set()
                            pool.shutdown(wait=False, cancel_futures=True)
                    else:
                        loop.run_until_complete(
                            run_scout_phase(query, state, _cancel=cancel_event)
                        )
                except Exception as exc:
                    cancel_event.set()
                    logger.warning("Phase 0 scout failed (non-fatal): %s", exc)

                # ── Pipeline health gate: scout ──
                try:
                    from models.pipeline_health import PipelineHealth, check_scout
                    health = PipelineHealth.from_state(state)
                    phase = health.begin_phase("scout")
                    phase.metrics["sub_queries"] = len(
                        state.get("_scout_sub_queries", [])
                    )
                    phase.metrics["initial_findings"] = len(
                        state.get("_scout_findings", [])
                    )
                    check_scout(phase, state)
                    health.evaluate_gate(phase)
                    health.save(state)
                except Exception:
                    pass  # health tracking is best-effort

    return None


def _pre_synthesiser_swarm(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Run Flock gossip swarm synthesis before the final synthesiser.

    Replaces the raw corpus of atomic conditions in
    ``state["corpus_for_synthesis"]`` with the swarm-synthesised report.
    The swarm uses a 3-phase gossip protocol for large corpora:
    per-angle workers → peer refinement → queen merge.

    The final synthesiser agent then polishes and restructures this
    swarm output — it works on a pre-synthesised narrative, not terse
    atomic facts.
    """
    state = callback_context.state
    swarm_report = run_swarm_synthesis(state)
    if swarm_report and swarm_report.strip():
        state["corpus_for_synthesis"] = swarm_report
        logger.info(
            "Swarm synthesis injected into corpus_for_synthesis "
            "(%d chars)",
            len(swarm_report),
        )
    else:
        logger.warning(
            "Swarm synthesis returned empty — final synthesiser "
            "will read the raw corpus format instead"
        )
    return None


def _cleanup_pipeline_state(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Release the CorpusStore after the pipeline finishes.

    Mirrors :func:`_init_pipeline_state` — closes the DuckDB connection
    and removes the store from the module-level ``_corpus_stores`` dict
    so memory and connections are not leaked across runs.

    Also records the final pipeline health verdict.
    """
    state = callback_context.state

    # ── Pipeline health gate: synthesiser (final) ──
    # The synthesiser just ran — check if it produced a useful report.
    try:
        from models.pipeline_health import PipelineHealth, check_synthesiser
        health = PipelineHealth.from_state(state)
        phase = health.begin_phase("synthesiser")
        # The synthesiser writes to corpus_for_synthesis / the final output
        synth_output = state.get("corpus_for_synthesis", "")
        phase.metrics["report_length"] = len(synth_output) if synth_output else 0
        # Count corpus findings for the health check
        try:
            from callbacks.condition_manager import _corpus_stores
            corpus_key = state.get("_corpus_key")
            if corpus_key and corpus_key in _corpus_stores:
                corpus = _corpus_stores[corpus_key]
                phase.metrics["corpus_findings"] = corpus.conn.execute(
                    "SELECT COUNT(*) FROM conditions WHERE row_type = 'finding'"
                ).fetchone()[0]
        except Exception:
            phase.metrics["corpus_findings"] = 0
        check_synthesiser(phase, state)
        health.evaluate_gate(phase)
        health.save(state)
        logger.info("Final pipeline health: %s", health.summary())
    except Exception:
        logger.warning("Pipeline health (final) failed (non-fatal)", exc_info=True)

    cleanup_corpus(state)
    return None


# ---------------------------------------------------------------------------
# Inner loop: thinker → maestro (with search executor) → repeat
# The thinker escalates when it judges evidence is sufficient.
# Each iteration:
#   1. Strategy (thinker) — pure reasoning
#   2. Search Executor (maestro's before_agent_callback) — automated APIs
#   3. Maestro — free-form Flock conductor
# ---------------------------------------------------------------------------
# Wrap the maestro with a before_agent_callback that runs the search
# executor.  ADK Agent() doesn't accept before_agent_callback directly,
# so we wrap it in a SequentialAgent of one.
_maestro_with_search = SequentialAgent(
    name="search_then_maestro",
    description=(
        "Runs the automated search executor (no LLM — reads expansion "
        "targets and strategy queries, fires APIs), then the maestro "
        "agent organises the corpus via free-form Flock SQL."
    ),
    sub_agents=[maestro_agent],
    before_agent_callback=search_executor_callback,
)

research_loop = LoopAgent(
    name="research_loop",
    max_iterations=5,
    sub_agents=[thinker_agent, _maestro_with_search],
)

# ---------------------------------------------------------------------------
# Outer pipeline: run the research loop, then synthesise once.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Wrap the final synthesiser with a before_agent_callback that runs the
# Flock gossip swarm.  ADK Agent() doesn't accept before_agent_callback
# directly, so we wrap it in a SequentialAgent of one.
# ---------------------------------------------------------------------------
_synthesiser_with_swarm = SequentialAgent(
    name="swarm_then_synthesise",
    description=(
        "Runs Flock gossip swarm synthesis on the corpus, then the "
        "final synthesiser agent polishes the swarm output into a "
        "coherent, readable report."
    ),
    sub_agents=[synthesiser_agent],
    before_agent_callback=_pre_synthesiser_swarm,
)

pipeline_agent = SequentialAgent(
    name="mirothinker_pipeline",
    description=(
        "Blackboard research pipeline: LoopAgent(thinker → maestro) "
        "runs iteratively with ever-expanding context. Each round: "
        "strategy (thinker) → automated search (executor) → free-form "
        "Flock organisation (maestro). Flock gossip swarm synthesises "
        "the corpus, then the final synthesiser writes the definitive report."
    ),
    sub_agents=[research_loop, _synthesiser_with_swarm],
    before_agent_callback=_init_pipeline_state,
    after_agent_callback=_cleanup_pipeline_state,
)
