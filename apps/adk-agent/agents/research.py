# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Main research agent definition.

Wires together all four callbacks (Algorithms 2, 4, 5, 7, 8) and
delegates to the browsing sub-agent via ADK's native ``sub_agents``
mechanism — no fake tool interception needed.
"""

from __future__ import annotations

import os

from google.adk import Agent

from agents.browsing import browsing_agent
from callbacks.after_model import after_model_callback
from callbacks.after_tool import after_tool_callback
from callbacks.before_model import before_model_callback
from callbacks.before_tool import before_tool_callback
from prompts.templates import MAIN_AGENT_INSTRUCTION
from tools.mcp_tools import get_tools

_MODEL = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")

research_agent = Agent(
    name="research_agent",
    model=_MODEL,
    description="Deep research agent that uses tools and sub-agents to answer questions.",
    instruction=MAIN_AGENT_INSTRUCTION,
    tools=get_tools(["search_and_scrape_webpage", "jina_scrape_llm_summary", "tool-python"]),
    sub_agents=[browsing_agent],  # ADK handles delegation natively
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
)
