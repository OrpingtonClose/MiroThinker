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
    └── Agent("synthesiser")            # uncensored, no tools

Data flows via session state (blackboard):

  - **Thinker** reads ``{research_findings}`` (empty on first pass),
    reasons about gaps, outputs strategy to ``state["research_strategy"]``.
    When evidence is sufficient it outputs ``EVIDENCE_SUFFICIENT`` and an
    ``after_agent_callback`` sets ``escalate=True`` to break the loop.

  - **Researcher** reads ``{research_strategy}`` + ``{research_findings}``,
    executes via the executor ``AgentTool``, then outputs ALL accumulated
    findings (old + new) to ``state["research_findings"]``.

  - **Synthesiser** (runs once after the loop) reads the final
    ``{research_findings}`` and writes the polished report.

The thinker and synthesiser NEVER touch tool format — they are pure
text-in/text-out LLM agents using the uncensored model.  Only the
researcher and executor need tool-calling capability.
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent
from google.adk.agents.loop_agent import LoopAgent

from agents.thinker import thinker_agent
from agents.researcher import researcher_agent
from agents.synthesiser import synthesiser_agent

# ---------------------------------------------------------------------------
# Inner loop: thinker reasons → researcher executes → repeat
# The thinker escalates when it judges evidence is sufficient.
# ---------------------------------------------------------------------------
research_loop = LoopAgent(
    name="research_loop",
    max_iterations=5,
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
)
