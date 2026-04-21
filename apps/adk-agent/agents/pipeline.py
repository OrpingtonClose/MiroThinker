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
import os
from typing import Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from models.pipeline_block import PipelineRunner

logger = logging.getLogger(__name__)


def _preflight_api_check() -> dict[str, dict]:
    """Lightweight health check for all configured search APIs.

    Returns a dict of ``{api_name: {"reachable": bool, "error": str}}``.
    Each check is a simple HEAD/GET request with a short timeout — no
    actual search queries are fired.  Failed APIs are recorded as
    degradations, not pipeline aborts (graceful degradation).
    """
    import os as _os

    results: dict[str, dict] = {}

    # Map of API name → (env var for key, health-check URL)
    api_checks = {
        "exa": ("EXA_API_KEY", "https://api.exa.ai"),
        "tavily": ("TAVILY_API_KEY", "https://api.tavily.com"),
        "serper": ("SERPER_API_KEY", "https://google.serper.dev"),
        "jina": ("JINA_API_KEY", "https://r.jina.ai"),
        "brave": ("BRAVE_API_KEY", "https://api.search.brave.com"),
    }

    for name, (env_var, url) in api_checks.items():
        key = _os.environ.get(env_var, "")
        if not key:
            results[name] = {"reachable": False, "error": "API key not configured"}
            continue
        try:
            import urllib.request
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("Authorization", f"Bearer {key}")
            urllib.request.urlopen(req, timeout=5)
            results[name] = {"reachable": True}
        except Exception as exc:
            # A 4xx from HEAD is still "reachable" — the server responded
            err_str = str(exc)
            if "HTTP Error 4" in err_str or "HTTP Error 405" in err_str:
                results[name] = {"reachable": True}
            else:
                results[name] = {"reachable": False, "error": err_str}

    return results


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
        # Try state first, then PipelineStateRegistry (Phase 1).
        # The registry survives across AG-UI HTTP requests without
        # relying on fragile module-level mutable state.
        prev_db = state.get("_prev_corpus_db_path", "")
        if not prev_db:
            try:
                from models.pipeline_state import PipelineStateRegistry
                registry = PipelineStateRegistry.instance()
                user_query = state.get("user_query", "")
                # Primary: query-fingerprint lookup
                prev_db = registry.lookup(user_query) or ""
                if prev_db:
                    logger.info(
                        "Corpus continuity (registry lookup): %s", prev_db,
                    )
                else:
                    # Fallback: last registered path (any query)
                    prev_db = registry.lookup_fallback() or ""
                    if prev_db:
                        logger.info(
                            "Corpus continuity (registry fallback): %s", prev_db,
                        )
            except Exception:
                prev_db = ""

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

        # ── Phase 5: Pre-flight API validation ──
        # Validate all configured search APIs are reachable before the
        # pipeline wastes time on a run that will produce zero results.
        # Pipeline does NOT abort for API failures (graceful degradation).
        # Results are stored in state for the QualityManifest (Phase 4).
        try:
            _api_health = _preflight_api_check()
            state["_api_health"] = _api_health
            failed = [k for k, v in _api_health.items() if not v.get("reachable")]
            if failed:
                logger.warning(
                    "Pre-flight API check: %d/%d APIs unreachable: %s",
                    len(failed), len(_api_health), ", ".join(failed),
                )
                # Store as degradations for the quality manifest
                degradations = state.get("_pipeline_degradations", [])
                for name in failed:
                    degradations.append({
                        "source": name,
                        "error": _api_health[name].get("error", "unreachable"),
                        "category": "api_preflight_failure",
                    })
                state["_pipeline_degradations"] = degradations
            else:
                logger.info(
                    "Pre-flight API check: all %d APIs reachable",
                    len(_api_health),
                )
        except Exception:
            logger.debug("Pre-flight API check failed (non-fatal)", exc_info=True)

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
                    # The scout writes landscape text to research_findings.
                    # Measure whether it produced content (non-default value).
                    findings_text = state.get("research_findings", "")
                    has_landscape = (
                        bool(findings_text)
                        and findings_text != "(no findings yet)"
                    )
                    phase.metrics["has_landscape"] = has_landscape
                    phase.metrics["landscape_length"] = len(findings_text) if has_landscape else 0
                    check_scout(phase, state)
                    health.evaluate_gate(phase)
                    health.save(state)
                except Exception:
                    pass  # health tracking is best-effort

    return None


async def _pre_synthesiser_swarm(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Run Flock gossip swarm synthesis before the final synthesiser.

    Thin wrapper that delegates to ``SwarmSynthesisBlock`` via the block
    adapter.  All cross-cutting concerns are handled by aspects.
    """
    # Skip expensive swarm synthesis if the pipeline was aborted
    if callback_context.state.get("_pipeline_aborted"):
        logger.warning(
            "Skipping swarm synthesis — pipeline aborted: %s",
            callback_context.state.get("_abort_reason", "unknown"),
        )
        return None

    from callbacks.block_adapter import run_block_from_callback
    await run_block_from_callback("swarm_synthesis", callback_context)
    return None


async def _cleanup_pipeline_state(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Release the CorpusStore after the pipeline finishes.

    Delegates the synthesiser health check to ``SynthesiserBlock`` via
    the block adapter (aspects handle health tracking), then performs
    cleanup of the corpus store.
    """
    from callbacks.block_adapter import run_block_from_callback

    # Run synthesiser block for health tracking and metrics
    try:
        await run_block_from_callback("synthesiser", callback_context)
    except Exception:
        logger.warning("Synthesiser block failed (non-fatal)", exc_info=True)

    cleanup_corpus(callback_context.state)
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


# ---------------------------------------------------------------------------
# Aspect-oriented block pipeline
# ---------------------------------------------------------------------------
# ADK callbacks delegate to PipelineRunner which applies aspects
# uniformly around each block's execution.  The ADK agent graph is
# unchanged — blocks wrap the same logic, but with clean I/O contracts
# and cross-cutting concerns handled by aspects.
# ---------------------------------------------------------------------------


def build_pipeline_runner() -> "PipelineRunner":
    """Construct a PipelineRunner with all blocks and aspects wired up.

    Called once per pipeline session.  The runner is stateful — it tracks
    health, costs, and error counts across all blocks in a session.
    """
    from models.pipeline_block import PipelineRunner

    from aspects.timing import TimingAspect
    from aspects.heartbeat import HeartbeatAspect
    from aspects.io_validation import InputOutputValidationAspect
    from aspects.duckdb_safety import DuckDBSafetyAspect
    from aspects.health_gate import HealthGateAspect
    from aspects.cost_tracking import CostTrackingAspect
    from aspects.corpus_refresh import CorpusRefreshAspect
    from aspects.error_escalation import ErrorEscalationAspect

    from blocks.scout_block import ScoutBlock
    from blocks.thinker_block import ThinkerBlock
    from blocks.search_executor_block import SearchExecutorBlock
    from blocks.maestro_block import MaestroBlock
    from blocks.swarm_block import SwarmSynthesisBlock
    from blocks.synthesiser_block import SynthesiserBlock

    # Aspect order matters:
    # 1. Timing — outermost, measures total wall-clock
    # 2. Heartbeat — dashboard events at boundaries
    # 3. I/O Validation — validate inputs before execution
    # 4. DuckDB Safety — thread-safety gate for corpus blocks
    # 5. Health Gate — cumulative health tracking
    # 6. Cost Tracking — API cost deltas
    # 7. Corpus Refresh — state refresh after corpus mutators
    # 8. Error Escalation — LAST, decides absorb vs propagate
    aspects = [
        TimingAspect(),
        HeartbeatAspect(),
        InputOutputValidationAspect(),
        DuckDBSafetyAspect(),
        HealthGateAspect(),
        CostTrackingAspect(),
        CorpusRefreshAspect(),
        ErrorEscalationAspect(),
    ]

    blocks = [
        ScoutBlock(),
        ThinkerBlock(),
        SearchExecutorBlock(),
        MaestroBlock(),
        SwarmSynthesisBlock(),
        SynthesiserBlock(),
    ]

    return PipelineRunner(blocks=blocks, aspects=aspects)


# Module-level runner instance (lazy, created on first use in blocks mode)
_runner: Optional["PipelineRunner"] = None


def get_pipeline_runner() -> "PipelineRunner":
    """Get or create the module-level PipelineRunner singleton."""
    global _runner
    if _runner is None:
        _runner = build_pipeline_runner()
        logger.info("PipelineRunner created with %d blocks, %d aspects",
                     len(_runner.blocks), len(_runner.aspects))
    return _runner


def reset_pipeline_runner() -> None:
    """Reset the runner (e.g. between pipeline sessions)."""
    global _runner
    _runner = None
