# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Blackboard research pipeline: LoopAgent(Thinker → Researcher → Ferment) + Synthesiser.

Uses ADK's ``LoopAgent`` to implement an iterative research cycle where
context grows richer on every iteration, then a ``SequentialAgent`` to
run the final synthesiser after the loop exits.

Architecture::

    SequentialAgent("mirothinker_pipeline")
    └── LoopAgent("research_loop", max_iterations=3)
    │     ├── Agent("thinker")           # uncensored, no tools
    │     ├── Agent("researcher")        # tool-capable, calls executor
    │     │     └── after_agent_callback: researcher_condition_callback
    │     └── Agent("loop_synthesiser")  # fermentation step
    │           └── after_agent_callback: synthesis_condition_callback
    └── Agent("synthesiser")                # final report, uncensored

The loop implements a 3-phase cycle per iteration:

  1. **Expansion** (researcher): brute-force tool execution — algorithmic
     factory line of enrichment.  Flock atomises, scores, and deduplicates
     the raw findings.

  2. **Fermentation** (loop_synthesiser): quasi-human analysis that
     connects themes, surfaces contradictions, and builds causal chains.
     Its output is re-ingested into the CorpusStore as just another input
     — no special status, scored and deduplicated equally with raw findings.

  3. **Strategy** (thinker): reads the enriched corpus (now containing
     both raw findings AND synthesised insights), reasons about gaps,
     and plans the next expansion round.

Data flows via session state (blackboard):

  - **Thinker** reads ``{research_findings}`` — a structured corpus of
    AtomicConditions stored in DuckDB.  Each condition carries confidence,
    verification_status, source_url, angle, and expansion_depth.
    The thinker reasons about gaps, duplicates, and contradictions
    holistically, then outputs strategy to ``state["research_strategy"]``.
    When evidence is sufficient it outputs ``EVIDENCE_SUFFICIENT`` and an
    ``after_agent_callback`` sets ``escalate=True`` to break the loop.

  - **Researcher** reads ``{research_strategy}`` + ``{research_findings}``,
    executes via the executor ``AgentTool``, then outputs accumulated
    findings to ``state["research_findings"]``.  Its ``after_agent_callback``
    (condition_manager) decomposes the free-text output into AtomicConditions,
    stores them in the DuckDB corpus, and overwrites the state keys with
    structured formatted text for the thinker and synthesiser.

  - **Loop synthesiser** reads ``{corpus_for_synthesis}`` and produces
    structured analysis.  Its ``after_agent_callback`` re-ingests the
    output into the CorpusStore — the synthesis becomes fertiliser for
    the next expansion round.

  - **Final synthesiser** (runs once after the loop) reads
    ``{corpus_for_synthesis}`` — all conditions formatted with confidence
    and verification metadata — and writes the polished report.

The thinker and synthesisers NEVER touch tool format — they are pure
text-in/text-out LLM agents using the uncensored model.  Only the
researcher and executor need tool-calling capability.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.loop_agent import LoopAgent
from google.genai import types as genai_types

from agents.thinker import thinker_agent
from agents.researcher import researcher_agent
from agents.synthesiser import synthesiser_agent
from agents.loop_synthesiser import loop_synthesiser_agent
from callbacks.condition_manager import build_corpus_state, cleanup_corpus, init_corpus

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
# Inner loop: thinker → researcher → fermentation → repeat
# The thinker escalates when it judges evidence is sufficient.
# Each iteration: expansion (researcher) → fermentation (loop_synthesiser)
# → strategy (thinker).  The loop_synthesiser's output is re-ingested
# into the corpus so synthesised insights are treated equally to raw
# findings by Flock's scoring, dedup, and contradiction detection.
# ---------------------------------------------------------------------------
research_loop = LoopAgent(
    name="research_loop",
    max_iterations=3,
    sub_agents=[thinker_agent, researcher_agent, loop_synthesiser_agent],
)

# ---------------------------------------------------------------------------
# Outer pipeline: run the research loop, then synthesise once.
# ---------------------------------------------------------------------------
pipeline_agent = SequentialAgent(
    name="mirothinker_pipeline",
    description=(
        "Blackboard research pipeline: LoopAgent(thinker → researcher → "
        "loop_synthesiser) runs iteratively with ever-expanding context. "
        "Each round: brute expansion → intelligent fermentation → strategy. "
        "The final synthesiser writes the definitive report after the loop."
    ),
    sub_agents=[research_loop, synthesiser_agent],
    before_agent_callback=_init_pipeline_state,
    after_agent_callback=_cleanup_pipeline_state,
)
