# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Main research agent definition (Tier 3 — fully agentic).

The research_agent delegates ALL web data retrieval to ``web_agent``
(a specialist sub-agent that owns Brave, Firecrawl, Exa, and Kagi MCP
toolsets).  It also has direct access to the Qualitative Research MCP
for structuring research into knowledge graphs.  This reduces context
burn from ~15 tool descriptions to just a few (web_agent + tool-python
+ qualitative-research).

Callbacks wired here:
  - before_model: Algorithm 5 (context window), Algorithm 7 (intermediate answers)
  - after_model:  token counting, trace recording
  - before_tool / after_tool on research_agent itself: only for tool-python
  - web_agent has its own before/after_tool for Algorithms 2, 4, 8, 9
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model
from agents.web_agent import web_agent
from callbacks.after_model import after_model_callback
from callbacks.after_tool import after_tool_callback
from callbacks.before_model import before_model_callback
from callbacks.before_tool import before_tool_callback
from prompts.templates import MAIN_AGENT_INSTRUCTION
from tools.mcp_tools import get_tools
from tools.discovery_store import DISCOVERY_TOOLS
from tools.knowledge_graph import KNOWLEDGE_GRAPH_TOOLS
from tools.research_tools import RESEARCH_TOOLS

research_agent = Agent(
    name="research_agent",
    model=build_model(),
    description="Deep research agent that uses tools and sub-agents to answer questions.",
    instruction=MAIN_AGENT_INSTRUCTION,
    tools=get_tools(["tool-python", "qualitative-research"]) + RESEARCH_TOOLS + KNOWLEDGE_GRAPH_TOOLS + DISCOVERY_TOOLS,
    sub_agents=[web_agent],  # Tier 3: web_agent owns all web tools
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
