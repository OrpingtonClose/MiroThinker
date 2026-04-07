# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Four-agent sequential pipeline: Thinker -> Researcher -> Synthesiser.

Uses ADK's ``SequentialAgent`` to run agents in fixed order with no
tool-calling overhead between stages.  Data flows via session state:

  1. **Thinker** (uncensored, no tools)
     Reads user query, reasons freely, outputs research strategy.
     -> state["research_strategy"]

  2. **Researcher** (tool-capable, calls executor via AgentTool)
     Reads {research_strategy}, plans searches, calls executor
     iteratively, reviews results, plans follow-ups.
     The **Executor** (tool-capable, owns all MCP tools) is wrapped
     as an AgentTool inside the researcher — it mechanically runs
     Brave/Kagi/Exa/Firecrawl/TranscriptAPI calls.
     -> state["research_findings"]

  3. **Synthesiser** (uncensored, no tools)
     Reads {research_findings}, writes the final report.

The thinker and synthesiser NEVER touch tool format — they are pure
text-in/text-out LLM agents using the uncensored model.  Only the
researcher and executor need tool-calling capability.
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from agents.thinker import thinker_agent
from agents.researcher import researcher_agent
from agents.synthesiser import synthesiser_agent

pipeline_agent = SequentialAgent(
    name="mirothinker_pipeline",
    description=(
        "Four-stage research pipeline: thinker (strategy) -> researcher "
        "(tool planning + executor) -> synthesiser (final report). "
        "Thinker and synthesiser are uncensored with no tools. "
        "Researcher calls the executor agent for mechanical tool execution."
    ),
    sub_agents=[thinker_agent, researcher_agent, synthesiser_agent],
)
