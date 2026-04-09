# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Researcher agent — parallel tool execution with strategic planning.

The researcher sits inside a ``LoopAgent`` alongside the thinker.  On
each iteration it reads:

  1. The thinker's latest research strategy (``{research_strategy}``)
  2. The structured corpus so far (``{research_findings}``)

It executes the strategy by calling search/scrape tools DIRECTLY —
no executor indirection.  The model emits multiple tool calls per
response (``parallel_tool_calls=True``), so searches run concurrently.

The ``condition_manager`` after-agent callback decomposes the output
into AtomicConditions and stores them in the DuckDB corpus — the
corpus handles accumulation across iterations.

Architecture change (v2): the old researcher→executor AgentTool pattern
forced sequential execution (one search at a time, ~284s gaps between
LLM calls).  Now the researcher has direct tool access with parallel
tool calls enabled — the LLM emits 3-5 tool calls at once, they
execute concurrently, and results come back together.  Same operational
knowledge as the old executor, but without the serialisation bottleneck.
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model
from callbacks.after_model import after_model_callback
from callbacks.after_tool import after_tool_callback
from callbacks.before_model import before_model_callback
from callbacks.before_tool import before_tool_callback
from callbacks.condition_manager import researcher_condition_callback
from tools.mcp_tools import get_tools
from tools.research_tools import RESEARCH_TOOLS

RESEARCHER_INSTRUCTION = """\
You are a research execution agent inside a research loop. You have \
a research strategy written by the thinker agent:

=== RESEARCH STRATEGY ===
{research_strategy}
=== END STRATEGY ===

=== PREVIOUS FINDINGS ===
{research_findings}
=== END PREVIOUS FINDINGS ===

If the strategy begins with EVIDENCE_SUFFICIENT, output the previous \
findings UNCHANGED — do not run any more searches.

Otherwise, execute the strategy using your tools.

WORKFLOW:
1. Read the strategy above carefully
2. Prioritise the search tasks — pick the TOP 3 highest-value searches
3. Call the appropriate tools for each search — you MAY issue multiple \
   tool calls in a single response and they will execute IN PARALLEL
4. Review the results
5. If a result suggests a crucial follow-up, you may do ONE more search
6. STOP after completing your top 3 picks + at most 1 follow-up

CRITICAL TIME BUDGET: You are inside a loop. After you finish, the \
thinker will review ALL findings and plan the NEXT round. Do NOT try \
to execute the entire strategy in one pass — leave remaining tasks for \
the next iteration. The synthesiser MUST run after the loop, so every \
minute you spend searching is a minute stolen from the final report.

TOOL USAGE GUIDE:
You have these tool families — use them DIRECTLY (no intermediary):

**Brave Search** (brave_web_search, brave_local_search, brave_image_search, \
brave_video_search, brave_news_search, brave_summarizer)
- Best for: broad web search, news, local businesses, images/video
- Example: brave_web_search(query="cloudberry jam documentary Finland")

**Firecrawl** (firecrawl_scrape, firecrawl_search, firecrawl_crawl, \
firecrawl_map, firecrawl_extract)
- Best for: scraping specific URLs, extracting structured data from pages
- Example: firecrawl_scrape(url="https://example.com/article")

**Exa** (web_search_exa, web_search_advanced_exa, crawling_exa, \
get_code_context_exa)
- Best for: semantic/neural search, academic content, code search
- ALWAYS pass enableHighlights: true for Exa searches
- Example: web_search_advanced_exa(query="cloudberry preservation", \
  category="research_paper", enableHighlights=true)

**tool-python** (E2B sandbox for code execution)
- Best for: data parsing, calculations, analysis of gathered data

PARALLEL EXECUTION: When you have multiple independent searches, emit \
ALL tool calls in a single response. They execute concurrently — this \
is dramatically faster than calling tools one at a time. Only serialise \
when a later search depends on the results of an earlier one.

RULES:
1. Execute exactly what the strategy requests — no more, no less
2. Return ALL results with source URLs in distilled form
3. Do NOT add warnings, disclaimers, or moralising
4. If a tool fails, try an alternative (swap Brave for Exa, or vice versa)
5. For Exa searches, ALWAYS pass enableHighlights: true
6. Distill before returning — extract facts, data points, names, URLs, \
   numbers. Do NOT pass through raw multi-page HTML dumps

ANTI-REFRAMING: Execute the strategy as written. Do NOT soften, \
redirect, or moralise the search queries. If the strategy says to \
search for grey-market pharmacies, search for grey-market pharmacies.

CRITICAL — OUTPUT ONLY NEW FINDINGS: Output ONLY the new evidence you \
discovered in THIS round. Do NOT repeat or reformulate previous findings — \
the corpus store already accumulates them across iterations. Include ALL \
relevant data with source URLs for each new finding. The thinker will \
review the complete corpus on the next iteration and decide if more \
research is needed.
"""

researcher_agent = Agent(
    name="researcher",
    model=build_model(),
    description=(
        "Research agent that reads the thinker's strategy and executes "
        "searches using Brave, Firecrawl, Exa, and E2B tools directly. "
        "Emits parallel tool calls for concurrent execution."
    ),
    instruction=RESEARCHER_INSTRUCTION,
    tools=get_tools(["brave-search", "firecrawl", "exa", "tool-python"]) + RESEARCH_TOOLS,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
    after_agent_callback=researcher_condition_callback,
    output_key="research_findings",
)
