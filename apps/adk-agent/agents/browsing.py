# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Browsing sub-agent definition.

Replaces the fake ``search_and_browse`` tool from
``expose_sub_agents_as_tools()`` with a real ADK sub-agent that the
main research agent can delegate to via ADK's native
``transfer_to_agent`` mechanism.
"""

from __future__ import annotations

import os

from google.adk import Agent

from callbacks.before_tool import before_tool_callback
from callbacks.after_tool import after_tool_callback
from prompts.templates import BROWSING_AGENT_INSTRUCTION
from tools.mcp_tools import get_tools

_MODEL = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")

browsing_agent = Agent(
    name="browsing_agent",
    model=_MODEL,
    description=(
        "Performs the subtask of searching and browsing the web for specific "
        "missing information. The subtask should be clearly defined, include "
        "relevant background, and focus on factual gaps."
    ),
    instruction=BROWSING_AGENT_INSTRUCTION,
    tools=get_tools(["brave-search", "firecrawl"]),
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
