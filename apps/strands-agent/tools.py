# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Tool wiring for the Strands Venice agent.

Combines:
- MCP servers (Brave, Firecrawl, Exa, Kagi) for rich tool ecosystems
- Native @tool-decorated functions for direct API providers that don't
  need an MCP server (DuckDuckGo, Mojeek, Jina Reader, Google/Serper)

Tools are organised into tiers:
  Tier 1 — Uncensored (primary): DuckDuckGo, Brave, Exa, Mojeek
  Tier 2 — Content extraction: Jina Reader, Firecrawl, Kagi
  Tier 3 — Censored fallback: Google/Serper

Reference: apps/adk-agent/tools/mcp_tools.py lines 74-138.
"""

from __future__ import annotations

import logging
import os

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from strands import tool
from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)


def _full_env(**overrides):
    """Return a copy of the current environment with *overrides* applied.

    MCP server subprocesses inherit PATH, HOME, etc. so that ``npx`` and
    other tools resolve correctly.
    """
    env = dict(os.environ)
    env.update(overrides)
    return env


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Uncensored native tools (no MCP server needed)
# ═══════════════════════════════════════════════════════════════════════


@tool
def duckduckgo_search(query: str, max_results: int = 10) -> str:
    """Search the web using DuckDuckGo. Free, no API key, no tracking, uncensored results.

    Use this as your go-to first search — it's always available and has no content filtering.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return f"No DuckDuckGo results for: {query}"

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        body = r.get("body", "")
        formatted.append(f"{i}. [{title}]({url})\n   {body}")
    return "\n\n".join(formatted)


@tool
def mojeek_search(query: str, max_results: int = 10) -> str:
    """Search using Mojeek's independent crawler. Not a Google/Bing proxy — unique results.

    Mojeek has its own crawler and index, so it surfaces content that other
    engines miss entirely. Requires MOJEEK_API_KEY.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results with titles, URLs, and descriptions.
    """
    import httpx

    api_key = os.environ.get("MOJEEK_API_KEY", "")
    if not api_key:
        return "Mojeek API key not configured. Set MOJEEK_API_KEY in .env."

    resp = httpx.get(
        "https://www.mojeek.com/search",
        params={"q": query, "fmt": "json", "t": max_results, "api_key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("response", {}).get("results", [])
    if not results:
        return f"No Mojeek results for: {query}"

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        desc = r.get("desc", "")
        formatted.append(f"{i}. [{title}]({url})\n   {desc}")
    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — Content extraction native tools
# ═══════════════════════════════════════════════════════════════════════


@tool
def jina_read_url(url: str) -> str:
    """Extract clean text/markdown from any URL using Jina Reader.

    Converts web pages into clean, readable markdown. Fast and reliable
    for most pages. Requires JINA_API_KEY.

    Args:
        url: The URL to extract content from.

    Returns:
        Clean markdown text extracted from the URL (truncated to 15000 chars).
    """
    import httpx

    headers = {}
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.get(
        f"https://r.jina.ai/{url}",
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text[:15000]


# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — Censored fallback native tools
# ═══════════════════════════════════════════════════════════════════════


@tool
def google_search(query: str, max_results: int = 10) -> str:
    """Search Google via Serper API. Powerful but censored — use as fallback.

    Only use this when uncensored sources (DuckDuckGo, Brave, Exa, Mojeek)
    don't have what you need. Requires SERPER_API_KEY.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted Google search results with titles, URLs, and snippets.
    """
    import httpx

    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Serper API key not configured. Set SERPER_API_KEY in .env."

    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    organic = data.get("organic", [])
    if not organic:
        return f"No Google results for: {query}"

    formatted = []
    for i, r in enumerate(organic[:max_results], 1):
        title = r.get("title", "")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        formatted.append(f"{i}. [{title}]({link})\n   {snippet}")
    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# MCP server tools (Brave, Firecrawl, Exa, Kagi)
# ═══════════════════════════════════════════════════════════════════════

# Increase startup_timeout for slow npx/uvx downloads on staging VMs.
_MCP_STARTUP_TIMEOUT = int(os.environ.get("MCP_STARTUP_TIMEOUT", "120"))

# ── Brave Search MCP ─────────────────────────────────────────────────
# npm: @brave/brave-search-mcp-server  (MIT, brave/brave-search-mcp-server)
# Tools: brave_web_search, brave_local_search, brave_image_search,
#   brave_video_search, brave_news_search, brave_summarizer
brave_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "@brave/brave-search-mcp-server"],
            env=_full_env(BRAVE_API_KEY=os.environ.get("BRAVE_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Firecrawl MCP ────────────────────────────────────────────────────
# npm: firecrawl-mcp  (MIT, firecrawl/firecrawl-mcp-server)
# Tools: firecrawl_scrape, firecrawl_crawl, firecrawl_map,
#   firecrawl_search, firecrawl_extract
firecrawl_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env=_full_env(FIRECRAWL_API_KEY=os.environ.get("FIRECRAWL_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Exa MCP ──────────────────────────────────────────────────────────
# npm: exa-mcp-server  (MIT, exa-labs/exa-mcp-server)
# Tools: web_search_exa, web_search_advanced_exa, crawling_exa,
#   get_code_context_exa
# Requires: npm install -g exa-mcp-server
exa_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="node",
            args=[
                "-e",
                # Bootstrap Smithery entry-point with config that enables
                # ALL non-deprecated Exa tools.  Smithery reads config from
                # process.argv.slice(2) as key=value pairs.
                "process.argv[2]='enabledTools=web_search_exa,web_search_advanced_exa,crawling_exa,get_code_context_exa';"
                "const r=require('child_process').execSync('npm root -g',{encoding:'utf8'}).trim();"
                "require(r+'/exa-mcp-server/.smithery/stdio/index.cjs');",
            ],
            env=_full_env(EXA_API_KEY=os.environ.get("EXA_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Kagi MCP ─────────────────────────────────────────────────────────
# uvx: kagimcp  (MIT, kagisearch/kagimcp)
# Tools: kagi_search, kagi_summarize, kagi_fastgpt, kagi_enrich_web,
#   kagi_enrich_news
kagi_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["kagimcp"],
            env=_full_env(KAGI_API_KEY=os.environ.get("KAGI_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Registry mapping ─────────────────────────────────────────────────
_MCP_REGISTRY = {
    "BRAVE_API_KEY": brave_mcp,
    "FIRECRAWL_API_KEY": firecrawl_mcp,
    "EXA_API_KEY": exa_mcp,
    "KAGI_API_KEY": kagi_mcp,
}


def get_all_mcp_clients():
    """Return list of MCP clients whose API keys are configured."""
    clients = []
    for env_var, client in _MCP_REGISTRY.items():
        if os.environ.get(env_var):
            clients.append(client)
    return clients


# ── Native tool list (always available) ──────────────────────────────

# Tier 1 uncensored tools — always included
NATIVE_TOOLS_TIER1 = [duckduckgo_search, mojeek_search]

# Tier 2 content extraction tools — always included
NATIVE_TOOLS_TIER2 = [jina_read_url]

# Tier 3 censored fallback — only if API key is set
NATIVE_TOOLS_TIER3 = [google_search]


def get_native_tools():
    """Return native @tool functions, ordered uncensored-first.

    Tier 3 tools (Google/Serper) are only included when their API key
    is configured, since they are useless without it.
    """
    tools = list(NATIVE_TOOLS_TIER1) + list(NATIVE_TOOLS_TIER2)
    if os.environ.get("SERPER_API_KEY"):
        tools.extend(NATIVE_TOOLS_TIER3)
    return tools
