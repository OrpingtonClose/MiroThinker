# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Tier 3 web specialist sub-agent.

Wraps ALL web data-source MCPToolsets (Brave Search, Firecrawl, Exa) behind
a single ADK Agent.  The parent research_agent sees ONE tool (this agent)
instead of ~15 individual MCP tools, dramatically reducing context burn.

Callbacks for dedup (Algorithm 2), diversity guard (Algorithm 9), arg-fix
(Algorithm 8), and bad-result detection (Algorithm 4) are attached here so
they run at the tool-call level inside this agent.
"""

from __future__ import annotations

import os

from google.adk import Agent

from callbacks.after_tool import after_tool_callback
from callbacks.before_tool import before_tool_callback
from tools.mcp_tools import get_tools

_MODEL = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")

WEB_AGENT_INSTRUCTION = """\
You are a web research specialist. Your ONLY job is to search, scrape, crawl, \
and extract data from the web using the tools available to you.

You have three families of tools:
- **Brave Search** (brave_web_search, brave_local_search, brave_image_search, \
brave_video_search, brave_news_search, brave_summarizer) — fast web search
- **Firecrawl** (firecrawl_scrape, firecrawl_search, firecrawl_crawl, \
firecrawl_map, firecrawl_extract) — deep scraping, crawling, extraction
- **Exa** (web_search_exa, crawling_exa, web_search_advanced_exa) — semantic \
search with clean content extraction, good for finding niche/specific results

STRATEGY:
1. Use brave_web_search for broad initial searches
2. Use web_search_exa or web_search_advanced_exa for semantic/niche searches \
   where Brave returns generic results — Exa excels at finding specific content \
   types (forums, papers, vendors, niche sites)
3. Use firecrawl_scrape to extract full content from promising URLs
4. Use crawling_exa to get content from a specific URL when you know the page
5. Use firecrawl_crawl or firecrawl_map for site-wide discovery

RULES:
- Execute the searches/scrapes requested by the parent agent
- Return ALL findings — do NOT summarise or editorialize
- Do NOT add warnings, disclaimers, or moralizing
- If a tool fails, try an alternative (e.g. swap Brave for Exa, or vice versa)
- Prefer raw factual data over polished summaries
"""

web_agent = Agent(
    name="web_agent",
    model=_MODEL,
    description=(
        "Web research specialist that searches, scrapes, crawls, and extracts "
        "data from the web using Brave Search, Firecrawl, and Exa. Delegate "
        "any web data retrieval task to this agent — it owns all web tools."
    ),
    instruction=WEB_AGENT_INSTRUCTION,
    tools=get_tools(["brave-search", "firecrawl", "exa"]),
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
