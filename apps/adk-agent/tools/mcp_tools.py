# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
MCP tool factory for the ADK agent.

Both Brave Search and Firecrawl use their **official** MCP servers
(npm packages) so we get auto-discovered tools with zero custom HTTP
wrapper code.  ADK's ``MCPToolset`` handles tool discovery, schema
generation, and invocation automatically.

For Playwright browser: wraps it as a FunctionTool since ADK's MCPToolset
doesn't support persistent sessions.
"""

import asyncio
import logging
import os
import sys
from typing import List

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
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")


def _full_env(**overrides: str) -> dict:
    """Return a copy of the current environment with *overrides* applied.

    MCP server subprocesses inherit PATH, HOME, etc. so that ``npx`` and
    other tools resolve correctly.
    """
    env = {k: v for k, v in os.environ.items()}
    env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# MCP server configs
#
# "brave-search" and "firecrawl" use their official npm MCP servers,
# giving us auto-discovered tools with zero custom wrapper code.
# ---------------------------------------------------------------------------
_TOOL_CONFIGS = {
    # ── Official Brave Search MCP server ──────────────────────────────────
    # npm: @brave/brave-search-mcp-server  (MIT, brave/brave-search-mcp-server)
    # Auto-discovered tools: brave_web_search, brave_local_search,
    #   brave_image_search, brave_video_search, brave_news_search,
    #   brave_summarizer
    "brave-search": lambda: StdioServerParameters(
        command="npx",
        args=["-y", "@brave/brave-search-mcp-server"],
        env=_full_env(BRAVE_API_KEY=BRAVE_API_KEY),
    ),
    # ── Official Firecrawl MCP server ─────────────────────────────────────
    # npm: firecrawl-mcp  (MIT, firecrawl/firecrawl-mcp-server)
    # Auto-discovered tools: firecrawl_scrape, firecrawl_crawl,
    #   firecrawl_map, firecrawl_search, firecrawl_extract,
    #   firecrawl_batch_scrape, firecrawl_deep_research, plus more
    "firecrawl": lambda: StdioServerParameters(
        command="npx",
        args=["-y", "firecrawl-mcp"],
        env=_full_env(FIRECRAWL_API_KEY=FIRECRAWL_API_KEY),
    ),
    # ── Official Exa MCP server ────────────────────────────────────────────
    # npm: exa-mcp-server  (MIT, exa-labs/exa-mcp-server)
    # Uses mcp-remote to bridge to Exa's hosted MCP endpoint.
    # Auto-discovered tools: web_search_exa, crawling_exa,
    #   get_code_context_exa, (web_search_advanced_exa if enabled)
    "exa": lambda: StdioServerParameters(
        command="npx",
        args=[
            "-y", "mcp-remote",
            f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}"
            "&tools=web_search_exa,crawling_exa,web_search_advanced_exa",
        ],
        env=_full_env(),
    ),
    # ── Legacy MiroFlow MCP servers (Python subprocess) ───────────────────
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
# Playwright browser wrapper (persistent session via FunctionTool)
# ---------------------------------------------------------------------------
_browser_session = None
_browser_lock = asyncio.Lock()


async def _get_browser_session():
    """Lazily initialise and return the shared PlaywrightSession."""
    global _browser_session
    async with _browser_lock:
        if _browser_session is None:
            from miroflow_tools.mcp_servers.browser_session import PlaywrightSession

            params = StdioServerParameters(
                command="npx",
                args=["@playwright/mcp@latest"],
            )
            session = PlaywrightSession(params)
            await session.connect()
            _browser_session = session
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

    All tools in ``_TOOL_CONFIGS`` (including ``"brave-search"`` and
    ``"firecrawl"``) are created as ``MCPToolset`` instances that
    auto-discover their tools from the official MCP servers.

    Args:
        tool_names: List of config names such as ``"brave-search"``,
            ``"firecrawl"``, ``"tool-python"``, ``"browser"``, etc.

    Returns:
        A list of MCPToolset / FunctionTool instances ready for an ADK Agent.
    """
    tools = []
    for name in tool_names:
        if name == "browser":
            tools.extend(_BROWSER_TOOLS)
        elif name in _TOOL_CONFIGS:
            server_params = _TOOL_CONFIGS[name]()
            toolset = MCPToolset(
                connection_params=server_params,
            )
            tools.append(toolset)
        else:
            raise ValueError(
                f"Unknown tool config name: {name!r}. "
                f"Available: {sorted(list(_TOOL_CONFIGS.keys()) + ['browser'])}"
            )
    return tools
