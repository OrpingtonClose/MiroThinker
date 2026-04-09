# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Blackboard research pipeline: LoopAgent(Thinker → Researcher) + Synthesiser.

Uses ADK's ``LoopAgent`` to implement an iterative research cycle where
context grows richer on every iteration, then a ``SequentialAgent`` to
run the synthesiser after the loop exits.

Architecture::

    SequentialAgent("mirothinker_pipeline")
    └── LoopAgent("research_loop", max_iterations=5)
    │     ├── Agent("thinker")       # uncensored, no tools
    │     └── Agent("researcher")    # tool-capable, calls executor
    │           └── after_agent_callback: condition_manager
    └── Agent("synthesiser")            # uncensored, no tools

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

  - **Synthesiser** (runs once after the loop) reads
    ``{corpus_for_synthesis}`` — all conditions formatted with confidence
    and verification metadata — and writes the polished report.

The thinker and synthesiser NEVER touch tool format — they are pure
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
from callbacks.condition_manager import build_corpus_state, init_corpus

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

# ---------------------------------------------------------------------------
# Inner loop: thinker reasons → researcher executes → repeat
# The thinker escalates when it judges evidence is sufficient.
# ---------------------------------------------------------------------------
research_loop = LoopAgent(
    name="research_loop",
    max_iterations=3,
    sub_agents=[thinker_agent, researcher_agent],
)

# ---------------------------------------------------------------------------
# Outer pipeline: run the research loop, then synthesise once.
# ---------------------------------------------------------------------------
pipeline_agent = SequentialAgent(
    name="mirothinker_pipeline",
    description=(
        "Blackboard research pipeline: LoopAgent(thinker → researcher) runs "
        "iteratively with ever-expanding context until the thinker signals "
        "EVIDENCE_SUFFICIENT, then the synthesiser writes the final report."
    ),
    sub_agents=[research_loop, synthesiser_agent],
    before_agent_callback=_init_pipeline_state,
)
