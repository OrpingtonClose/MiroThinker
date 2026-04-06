# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Browsing sub-agent definition (Tier 3 — uses web_agent).

In the Tier 3 architecture, browsing_agent no longer owns web tools
directly.  It delegates to ``web_agent`` for all web data retrieval,
keeping its own context focused on task reasoning.

Note: with the Tier 3 pattern, the browsing_agent is now optional —
research_agent can delegate directly to web_agent.  We keep it for
backward compatibility with the orchestrator's transfer_to_agent
logic.
"""

from __future__ import annotations

import os

from google.adk import Agent

from agents.web_agent import web_agent
from callbacks.after_tool import after_tool_callback
from callbacks.before_tool import before_tool_callback
from prompts.templates import BROWSING_AGENT_INSTRUCTION

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
    sub_agents=[web_agent],  # Tier 3: delegates to web_agent
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
