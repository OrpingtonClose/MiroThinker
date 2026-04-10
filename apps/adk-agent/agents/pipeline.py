# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Blackboard research pipeline: LoopAgent(Thinker → Researcher) + Synthesiser.

Uses ADK's ``LoopAgent`` to implement an iterative 2-phase research
cycle where context grows richer on every iteration, then a
``SequentialAgent`` to run the final synthesiser after the loop exits.

Architecture::

    SequentialAgent("mirothinker_pipeline")
    └── LoopAgent("research_loop", max_iterations=3)
    │     ├── Agent("thinker")           # pure reasoning, no tools
    │     └── Agent("researcher")        # direct tool access, parallel_tool_calls=True
    │           └── after_agent_callback: researcher_condition_callback
    └── Agent("synthesiser")                # final report, uncensored

The loop implements a 2-phase cycle per iteration:

  1. **Strategy** (thinker): reads the enriched corpus (verbal prose
     briefing with tiered findings, contradictions, and under-explored
     areas), reasons deeply about gaps and emerging narratives, and
     plans the next expansion round.

  2. **Expansion** (researcher): brute-force tool execution — Flock's
     algorithm battery (scoring, dedup, contradiction detection,
     clustering, redundancy compression) runs in the after-agent
     callback, replacing the old loop_synthesiser's responsibilities.

Data flows via session state (blackboard):

  - **Thinker** reads ``{research_findings}`` — a verbal prose briefing
    of findings organised by quality tier (strong/moderate/weak),
    with contradictions, under-explored areas, and corpus health.
    The thinker outputs strategy to ``state["research_strategy"]``.
    When evidence is sufficient it outputs ``EVIDENCE_SUFFICIENT`` and an
    ``after_agent_callback`` sets ``escalate=True`` to break the loop.

  - **Researcher** reads ``{research_strategy}`` + ``{research_findings}``
    + ``{_expansion_targets}``, executes searches, then its
    ``after_agent_callback`` (condition_manager) decomposes the output
    into AtomicConditions, runs the full algorithm battery, and
    updates state with verbal prose for the thinker.

  - **Final synthesiser** (runs once after the loop) reads
    ``{corpus_for_synthesis}`` — the swarm-synthesised report — and
    writes the polished final report.

The thinker and synthesiser NEVER touch tool format — they are pure
text-in/text-out LLM agents using the uncensored model.  Only the
researcher needs tool-calling capability.
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
from agents.researcher import researcher_agent
from agents.synthesiser import synthesiser_agent
from callbacks.condition_manager import (
    build_corpus_state,
    cleanup_corpus,
    init_corpus,
    run_swarm_synthesis,
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
        for k, v in build_corpus_state().items():
            state[k] = v
        init_corpus(state)
        logger.info("Pipeline state initialised: corpus_key=%s", state["_corpus_key"])

        # Reset per-session cost tracker so each pipeline run starts fresh
        try:
            from tools.cost_tracker import reset_session_tracker
            reset_session_tracker()
        except Exception:
            pass

        # Run Phase 0 scout if this is a fresh pipeline
        query = state.get("user_query", "")
        if query:
            from tools.scout import run_scout_phase

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # ADK's event loop is already running — run in a
                    # separate thread with its own event loop.
                    # Do NOT use ThreadPoolExecutor as a context manager:
                    # __exit__ calls shutdown(wait=True), which blocks
                    # until the task finishes even after timeout.
                    import concurrent.futures
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = pool.submit(
                            asyncio.run, run_scout_phase(query, state)
                        )
                        future.result(timeout=90)
                    finally:
                        pool.shutdown(wait=False, cancel_futures=True)
                else:
                    loop.run_until_complete(run_scout_phase(query, state))
            except Exception as exc:
                logger.warning("Phase 0 scout failed (non-fatal): %s", exc)

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
    """
    cleanup_corpus(callback_context.state)
    return None


# ---------------------------------------------------------------------------
# Inner loop: thinker → researcher → repeat
# The thinker escalates when it judges evidence is sufficient.
# Each iteration: strategy (thinker) → expansion (researcher).
# Flock's algorithm battery runs inside researcher_condition_callback,
# replacing the old loop_synthesiser's responsibilities.
# ---------------------------------------------------------------------------
research_loop = LoopAgent(
    name="research_loop",
    max_iterations=3,
    sub_agents=[thinker_agent, researcher_agent],
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
        "Blackboard research pipeline: LoopAgent(thinker → researcher) "
        "runs iteratively with ever-expanding context. Each round: "
        "strategy → expansion (with algorithm battery). Flock gossip "
        "swarm synthesises the corpus, then the final synthesiser "
        "writes the definitive report."
    ),
    sub_agents=[research_loop, _synthesiser_with_swarm],
    before_agent_callback=_init_pipeline_state,
    after_agent_callback=_cleanup_pipeline_state,
)
