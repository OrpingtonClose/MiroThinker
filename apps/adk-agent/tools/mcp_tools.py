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
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StreamableHTTPConnectionParams,
)
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
KAGI_API_KEY = os.environ.get("KAGI_API_KEY", "")
TRANSCRIPTAPI_KEY = os.environ.get("TRANSCRIPTAPI_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
BRIGHT_DATA_API_KEY = os.environ.get("BRIGHT_DATA_API_KEY", "")


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
    "brave-search": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@brave/brave-search-mcp-server"],
            env=_full_env(BRAVE_API_KEY=BRAVE_API_KEY),
        ),
        timeout=30.0,
    ),
    # ── Official Firecrawl MCP server ─────────────────────────────────────
    # npm: firecrawl-mcp  (MIT, firecrawl/firecrawl-mcp-server)
    # Auto-discovered tools: firecrawl_scrape, firecrawl_crawl,
    #   firecrawl_map, firecrawl_search, firecrawl_extract,
    #   firecrawl_batch_scrape, firecrawl_deep_research, plus more
    "firecrawl": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env=_full_env(FIRECRAWL_API_KEY=FIRECRAWL_API_KEY),
        ),
        timeout=120.0,
    ),
    # ── Official Exa MCP server ────────────────────────────────────────────
    # npm: exa-mcp-server  (MIT, exa-labs/exa-mcp-server)
    # Runs the local Smithery stdio entry-point so the API key stays in
    # an env-var (not leaked on the command line like mcp-remote would).
    # Requires: npm install -g exa-mcp-server
    # All non-deprecated tools enabled: web_search_exa, crawling_exa,
    #   web_search_advanced_exa, get_code_context_exa
    "exa": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=[
                "-e",
                # Bootstrap Smithery entry-point with config that enables
                # ALL non-deprecated Exa tools (web_search_advanced_exa is
                # disabled by default but is the most powerful tool).
                # Smithery reads config from process.argv.slice(2) as
                # key=value pairs; with node -e argv has only 1 element,
                # so we inject at index 2+.
                "process.argv[2]='enabledTools=web_search_exa,web_search_advanced_exa,crawling_exa,get_code_context_exa';"
                "const r=require('child_process').execSync('npm root -g',{encoding:'utf8'}).trim();"
                "require(r+'/exa-mcp-server/.smithery/stdio/index.cjs');",
            ],
            env=_full_env(EXA_API_KEY=EXA_API_KEY),
        ),
        timeout=120.0,
    ),
    # ── Official Kagi MCP server ──────────────────────────────────────────
    # npm: kagimcp  (MIT, kagisearch/kagimcp)
    # Auto-discovered tools: kagi_search, kagi_summarize, kagi_fastgpt,
    #   kagi_enrich_web, kagi_enrich_news
    "kagi": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=["kagimcp"],
            env=_full_env(KAGI_API_KEY=KAGI_API_KEY),
        ),
        timeout=120.0,
    ),
    # ── TranscriptAPI MCP server (remote Streamable HTTP) ────────────────────
    # https://transcriptapi.com  (YouTube transcripts, search, channels, playlists)
    # Auto-discovered tools: get_youtube_transcript, search_youtube,
    #   get_channel_latest_videos, search_channel_videos,
    #   list_channel_videos, list_playlist_videos
    "transcriptapi": lambda: StreamableHTTPConnectionParams(
        url="https://transcriptapi.com/mcp",
        headers={"Authorization": f"Bearer {TRANSCRIPTAPI_KEY}"},
        timeout=30.0,
        sse_read_timeout=120.0,
    ),
    # ── Qualitative Research MCP server ─────────────────────────────────────
    # GitHub: tejpalvirk/qualitativeresearch  (TypeScript, knowledge-graph MCP)
    # Tools: startsession, loadcontext, endsession, buildcontext,
    #   deletecontext, advancedcontext + domain functions
    # (getProjectOverview, getThematicAnalysis, getCodedData, etc.)
    "qualitative-research": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="node",
            args=["/home/ubuntu/qualitativeresearch/index.js"],
            env=_full_env(),
        ),
        timeout=30.0,
    ),
    # ── DuckDB MCP server ──────────────────────────────────────────────────
    # npm: @seed-ship/duckdb-mcp-native  (MIT, theseedship/duckdb_mcp_node)
    # 32+ tools: SQL queries, schema inspection, CSV/Parquet loading,
    #   federation via mcp:// URIs, graph algorithms (PageRank, community
    #   detection), process mining, DuckPGQ property graphs.
    # Auto-discovered tools: query, describe_table, list_tables, load_csv,
    #   load_parquet, create_table, insert_data, pagerank, community_detection,
    #   eigenvector_centrality, weighted_shortest_path, etc.
    "duckdb": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@seed-ship/duckdb-mcp-native"],
            env=_full_env(
                MCP_SECURITY_MODE="development",
                DUCKDB_MEMORY="2GB",
                DUCKDB_THREADS="4",
            ),
        ),
        timeout=60.0,
    ),
    # ── Semantic Scholar MCP server ────────────────────────────────────────
    # npm: @xbghc/semanticscholar-mcp  (MIT, xbghc/semanticscholar-mcp)
    # 200M+ academic papers — paper search, citation graphs, author profiles.
    # Auto-discovered tools: search_papers, get_paper, get_paper_citations,
    #   get_paper_references, batch_get_papers, search_authors, get_author,
    #   get_author_papers, get_recommendations
    # API key optional (higher rate limits with key).
    "semantic-scholar": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@xbghc/semanticscholar-mcp"],
            env=_full_env(
                SEMANTIC_SCHOLAR_API_KEY=SEMANTIC_SCHOLAR_API_KEY,
            ),
        ),
        timeout=60.0,
    ),
    # ── arXiv MCP server ───────────────────────────────────────────────────
    # npm: arxiv-mcp-server  (madi/arxiv-mcp-server)
    # Free, no API key needed.  Searches arXiv preprints.
    # Auto-discovered tools: search_papers, get_paper, search_by_category
    "arxiv": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "arxiv-mcp-server"],
            env=_full_env(),
        ),
        timeout=60.0,
    ),
    # ── Wikipedia MCP server ───────────────────────────────────────────────
    # npm: wikipedia-mcp  (MIT, timjuenemann/wikipedia-mcp)
    # Free, no API key needed.  1.2K weekly downloads.
    # Auto-discovered tools: search (Wikipedia search), read (full article
    #   content as markdown)
    "wikipedia": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "wikipedia-mcp"],
            env=_full_env(),
        ),
        timeout=30.0,
    ),
    # ── Bright Data Web MCP server ─────────────────────────────────────────
    # npm: @brightdata/mcp  (MIT, brightdata/brightdata-mcp)  2.3K+ stars
    # Anti-block web scraping — bypasses CAPTCHAs, geo-restrictions, rate
    # limits.  5,000 free requests/month.
    # Auto-discovered tools: search_engine, scrape_as_markdown,
    #   scrape_as_html, session_stats, web_data_amazon_product, etc.
    # GROUPS env var controls which tool groups to load (default: all).
    "brightdata": lambda: StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@brightdata/mcp"],
            env=_full_env(
                API_TOKEN=BRIGHT_DATA_API_KEY,
                GROUPS="search,scraping",
            ),
        ),
        timeout=60.0,
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

# Track all created MCPToolset instances for graceful teardown.
_active_toolsets: List[MCPToolset] = []


# Prefixes to namespace tools from MCP servers whose tool names collide.
# e.g. Semantic Scholar and arXiv both expose "search_papers" and "get_paper".
# With prefixes, they become "ss_search_papers" and "arxiv_search_papers".
_TOOL_NAME_PREFIXES: dict[str, str] = {
    "semantic-scholar": "ss_",
    "arxiv": "arxiv_",
}


def get_tools(tool_names: List[str]):
    """
    Return a list of ADK tool instances for the requested tool config names.

    All tools in ``_TOOL_CONFIGS`` (including ``"brave-search"`` and
    ``"firecrawl"``) are created as ``MCPToolset`` instances that
    auto-discover their tools from the official MCP servers.

    Servers listed in ``_TOOL_NAME_PREFIXES`` get their tool names
    prefixed to avoid collisions (e.g. Semantic Scholar and arXiv both
    expose ``search_papers`` — prefixes make them ``ss_search_papers``
    and ``arxiv_search_papers``).

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
            prefix = _TOOL_NAME_PREFIXES.get(name)
            toolset = MCPToolset(
                connection_params=server_params,
                **({
                    "tool_name_prefix": prefix,
                } if prefix else {}),
            )
            _active_toolsets.append(toolset)
            tools.append(toolset)
        else:
            raise ValueError(
                f"Unknown tool config name: {name!r}. "
                f"Available: {sorted(list(_TOOL_CONFIGS.keys()) + ['browser'])}"
            )
    return tools


async def close_all_mcp_toolsets() -> None:
    """Gracefully close all MCP toolset connections.

    Call this before the event loop shuts down to prevent the
    'loop is closed, resources may be leaked' warnings from
    ``MCPSessionManager._cleanup_session``.

    This is the **Chainlit pattern**: explicitly tear down MCP
    subprocess connections (npx Brave, Firecrawl, Exa, etc.)
    instead of letting them crash when ``asyncio.run()`` exits.
    """
    global _browser_session

    for toolset in _active_toolsets:
        try:
            await toolset.close()
        except Exception as exc:
            logger.warning("Error closing MCP toolset: %s", exc)
    _active_toolsets.clear()

    # Also close the Playwright browser session if it was started.
    if _browser_session is not None:
        try:
            await _browser_session.close()
        except Exception as exc:
            logger.warning("Error closing Playwright session: %s", exc)
        _browser_session = None

    logger.info("All MCP toolsets closed")
