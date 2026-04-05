# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
adk web-compatible agent module for MiroThinker.

Makes the research_agent loadable by ``adk web``::

    cd apps/adk-agent && adk web adk_web_agents/

The Trace -> Graph tab shows the Graphviz execution graph generated
by ``google.adk.cli.agent_graph.get_agent_graph()`` with highlight
pairs derived from each event's function_calls / function_responses.

Both UIs coexist:
- ``adk web`` (port 8000) -- ADK native Trace/Graph for execution flow
- ``python main.py`` (port 8080) -- Custom dashboard with algorithm observability
"""

from __future__ import annotations

import os
import sys

# Add apps/adk-agent/ to sys.path so existing imports
# (agents.research, callbacks.*, tools.*, prompts.*) resolve correctly.
# AgentLoader only adds adk_web_agents/ to sys.path.
_ADK_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ADK_AGENT_DIR not in sys.path:
    sys.path.insert(0, _ADK_AGENT_DIR)

from agents.research import research_agent  # noqa: E402

root_agent = research_agent
