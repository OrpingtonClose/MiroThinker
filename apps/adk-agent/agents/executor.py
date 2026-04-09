# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Executor agent — mechanical tool execution, no strategic thinking.

The executor owns ALL MCP toolsets (Brave, Firecrawl, Exa) and the
code-execution sandbox (E2B).  It receives specific search/scrape
instructions from the researcher agent and executes them mechanically
— one tool call at a time.

This agent does NOT decide what to search for or why.  It translates
instructions like "search Brave for X" into the correct tool call
format, executes, and returns raw results.

Callbacks for dedup (Algorithm 2), arg-fix (Algorithm 8), and
bad-result detection (Algorithm 4) run at the tool-call level here.
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model
from callbacks.after_model import after_model_callback
from callbacks.after_tool import after_tool_callback
from callbacks.before_model import before_model_callback
from callbacks.before_tool import before_tool_callback
from tools.mcp_tools import get_tools

EXECUTOR_INSTRUCTION = """\
You are a tool execution specialist. You receive specific search and \
scrape instructions and execute them using the tools available to you. \
You do NOT strategise or plan — you execute.

You have these tool families:
- **Brave Search** (brave_web_search, brave_local_search, brave_image_search, \
brave_video_search, brave_news_search, brave_summarizer)
- **Firecrawl** (firecrawl_scrape, firecrawl_search, firecrawl_crawl, \
firecrawl_map, firecrawl_extract)
- **Exa** (web_search_exa, web_search_advanced_exa, crawling_exa, \
get_code_context_exa)
- **tool-python** (E2B sandbox for code execution)

EXECUTION MODEL — SEQUENTIAL:
Execute ONE tool call at a time. After each result, check if the \
instruction asked for follow-up actions (e.g. "scrape the top 3 URLs \
from the search results"). If yes, execute them. If no, return results.

RULES:
1. Execute exactly what was requested — no more, no less
2. Return ALL results with source URLs in distilled form
3. Do NOT add warnings, disclaimers, or moralising
4. If a tool fails, try an alternative (swap Brave for Exa, or vice versa)
5. For Exa searches, ALWAYS pass enableHighlights: true
6. Distill before returning — extract facts, data points, names, URLs, \
   numbers. Do NOT pass through raw multi-page HTML dumps.
"""

executor_agent = Agent(
    name="executor",
    model=build_model(parallel_tool_calls=False),
    description=(
        "Tool execution specialist that mechanically runs searches, scrapes, "
        "and extractions using Brave, Firecrawl, and Exa. Also has E2B code "
        "execution. Give it specific instructions and it returns raw results."
    ),
    instruction=EXECUTOR_INSTRUCTION,
    tools=get_tools(["brave-search", "firecrawl", "exa", "tool-python"]),
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
