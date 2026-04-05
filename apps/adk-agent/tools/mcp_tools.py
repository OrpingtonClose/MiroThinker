# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
MCP tool factory for the ADK agent.

Creates MCPToolset instances for each MCP server, referencing the server
configurations from apps/miroflow-agent/src/config/settings.py.
Uses google.adk.tools.mcp_tool.MCPToolset with StdioServerParameters.

For Playwright browser: wraps it as a FunctionTool since ADK's MCPToolset
doesn't support persistent sessions.
"""

import json
import logging
import os
import sys
from typing import List

import httpx
from dotenv import load_dotenv
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import MCPToolset
from mcp import StdioServerParameters

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://r.jina.ai")
E2B_API_KEY = os.environ.get("E2B_API_KEY", "")
SUMMARY_LLM_API_KEY = os.environ.get("SUMMARY_LLM_API_KEY", "")
SUMMARY_LLM_BASE_URL = os.environ.get("SUMMARY_LLM_BASE_URL", "")
SUMMARY_LLM_MODEL_NAME = os.environ.get("SUMMARY_LLM_MODEL_NAME", "")
TENCENTCLOUD_SECRET_ID = os.environ.get("TENCENTCLOUD_SECRET_ID", "")
TENCENTCLOUD_SECRET_KEY = os.environ.get("TENCENTCLOUD_SECRET_KEY", "")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")

# ---------------------------------------------------------------------------
# MCP server configs — mirrors settings.py from miroflow-agent
# ---------------------------------------------------------------------------
_TOOL_CONFIGS = {
    "tool-google-search": lambda: StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.mcp_servers.searching_google_mcp_server"],
        env={
            "SERPER_API_KEY": SERPER_API_KEY,
            "SERPER_BASE_URL": SERPER_BASE_URL,
            "JINA_API_KEY": JINA_API_KEY,
            "JINA_BASE_URL": JINA_BASE_URL,
        },
    ),
    "tool-python": lambda: StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.mcp_servers.python_mcp_server"],
        env={"E2B_API_KEY": E2B_API_KEY},
    ),
    "search_and_scrape_webpage": lambda: StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.dev_mcp_servers.search_and_scrape_webpage"],
        env={
            "SERPER_API_KEY": SERPER_API_KEY,
            "SERPER_BASE_URL": SERPER_BASE_URL,
            "TENCENTCLOUD_SECRET_ID": TENCENTCLOUD_SECRET_ID,
            "TENCENTCLOUD_SECRET_KEY": TENCENTCLOUD_SECRET_KEY,
        },
    ),
    "jina_scrape_llm_summary": lambda: StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.dev_mcp_servers.jina_scrape_llm_summary"],
        env={
            "JINA_API_KEY": JINA_API_KEY,
            "JINA_BASE_URL": JINA_BASE_URL,
            "SUMMARY_LLM_BASE_URL": SUMMARY_LLM_BASE_URL,
            "SUMMARY_LLM_MODEL_NAME": SUMMARY_LLM_MODEL_NAME,
            "SUMMARY_LLM_API_KEY": SUMMARY_LLM_API_KEY,
        },
    ),
    "tool-sogou-search": lambda: StdioServerParameters(
        command=sys.executable,
        args=["-m", "miroflow_tools.mcp_servers.searching_sogou_mcp_server"],
        env={
            "TENCENTCLOUD_SECRET_ID": TENCENTCLOUD_SECRET_ID,
            "TENCENTCLOUD_SECRET_KEY": TENCENTCLOUD_SECRET_KEY,
            "JINA_API_KEY": JINA_API_KEY,
            "JINA_BASE_URL": JINA_BASE_URL,
        },
    ),
}


# ---------------------------------------------------------------------------
# Brave Search + Firecrawl FunctionTools (alternative to Serper/Jina)
# ---------------------------------------------------------------------------


async def brave_web_search(
    q: str,
    count: int = 10,
    country: str = "",
    search_lang: str = "en",
    freshness: str = "",
) -> str:
    """Search the web using the Brave Search API.

    Args:
        q: Search query string.
        count: Number of results to return (default: 10, max: 20).
        country: Country code for regional results (e.g., 'US', 'PL').
        search_lang: Language code (default: 'en').
        freshness: Time filter — 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year), or empty for any time.

    Returns:
        JSON string with search results including title, url, and description for each result.
    """
    if not BRAVE_API_KEY:
        return json.dumps({"error": "BRAVE_API_KEY not set", "organic": []})

    params = {"q": q, "count": min(count, 20), "search_lang": search_lang}
    if country:
        params["country"] = country
    if freshness:
        params["freshness"] = freshness

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params=params,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("url", ""),
                "snippet": item.get("description", ""),
            })

        return json.dumps({"organic": results, "searchParameters": {"q": q}}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Brave search failed: {exc}", "organic": []})


async def firecrawl_scrape(url: str, only_main_content: bool = True) -> str:
    """Scrape a webpage and return its content as clean markdown using the Firecrawl API.

    Args:
        url: The URL of the webpage to scrape.
        only_main_content: If True, return only the main content (skip navbars, footers, etc.). Default True.

    Returns:
        The scraped page content as markdown text, or an error message.
    """
    if not FIRECRAWL_API_KEY:
        return "[ERROR]: FIRECRAWL_API_KEY not set, scraping is unavailable."

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                json={"url": url, "formats": ["markdown"], "onlyMainContent": only_main_content},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("success"):
            md = data.get("data", {}).get("markdown", "")
            title = data.get("data", {}).get("metadata", {}).get("title", "")
            # Truncate to 30K chars to avoid overwhelming context
            if len(md) > 30000:
                md = md[:30000] + "\n\n[... content truncated at 30K characters ...]"
            header = f"# {title}\nSource: {url}\n\n" if title else f"Source: {url}\n\n"
            return header + md
        else:
            return f"[ERROR]: Firecrawl scrape failed: {data}"
    except Exception as exc:
        return f"[ERROR]: Firecrawl scrape error: {exc}"


_BRAVE_TOOLS = [
    FunctionTool(brave_web_search),
    FunctionTool(firecrawl_scrape),
]


# ---------------------------------------------------------------------------
# Playwright browser wrapper (persistent session via FunctionTool)
# ---------------------------------------------------------------------------
_browser_session = None


async def _get_browser_session():
    """Lazily initialise and return the shared PlaywrightSession."""
    global _browser_session
    if _browser_session is None:
        from miroflow_tools.mcp_servers.browser_session import PlaywrightSession

        params = StdioServerParameters(
            command="npx",
            args=["@playwright/mcp@latest"],
        )
        _browser_session = PlaywrightSession(params)
        await _browser_session.connect()
    return _browser_session


async def browser_navigate(url: str) -> str:
    """Navigate the browser to a URL."""
    session = await _get_browser_session()
    return await session.call_tool("browser_navigate", {"url": url})


async def browser_snapshot() -> str:
    """Take a snapshot of the current browser page."""
    session = await _get_browser_session()
    return await session.call_tool("browser_snapshot", {})


async def browser_click(element: str, ref: str) -> str:
    """Click an element on the page."""
    session = await _get_browser_session()
    return await session.call_tool("browser_click", {"element": element, "ref": ref})


async def browser_type(element: str, ref: str, text: str) -> str:
    """Type text into a page element."""
    session = await _get_browser_session()
    return await session.call_tool(
        "browser_type", {"element": element, "ref": ref, "text": text}
    )


_BROWSER_TOOLS = [
    FunctionTool(browser_navigate),
    FunctionTool(browser_snapshot),
    FunctionTool(browser_click),
    FunctionTool(browser_type),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tools(tool_names: List[str]):
    """
    Return a list of ADK tool instances for the requested tool config names.

    Args:
        tool_names: List of config names such as ``"tool-google-search"``,
            ``"search_and_scrape_webpage"``, ``"jina_scrape_llm_summary"``,
            ``"tool-python"``, ``"browser"``, or ``"brave-search"``.

    Returns:
        A list of MCPToolset / FunctionTool instances ready for an ADK Agent.
    """
    tools = []
    for name in tool_names:
        if name == "browser":
            tools.extend(_BROWSER_TOOLS)
        elif name == "brave-search":
            tools.extend(_BRAVE_TOOLS)
        elif name in _TOOL_CONFIGS:
            server_params = _TOOL_CONFIGS[name]()
            toolset = MCPToolset(
                connection_params=server_params,
            )
            tools.append(toolset)
        else:
            raise ValueError(
                f"Unknown tool config name: {name!r}. "
                f"Available: {sorted(list(_TOOL_CONFIGS.keys()) + ['browser', 'brave-search'])}"
            )
    return tools
