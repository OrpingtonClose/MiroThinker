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

import asyncio
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
# Brave Search FunctionTool
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


_BRAVE_SEARCH_TOOL = FunctionTool(brave_web_search)


# ---------------------------------------------------------------------------
# Firecrawl FunctionTools
# ---------------------------------------------------------------------------

_FC_BASE = "https://api.firecrawl.dev/v1"


def _fc_auth_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
    }


def _truncate_md(md: str, limit: int = 30000) -> str:
    if len(md) > limit:
        return md[:limit] + "\n\n[... content truncated ...]"
    return md


async def firecrawl_scrape(url: str, only_main_content: bool = True) -> str:
    """Scrape a single webpage and return its content as clean markdown.

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
                f"{_FC_BASE}/scrape",
                json={"url": url, "formats": ["markdown"], "onlyMainContent": only_main_content},
                headers=_fc_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("success"):
            md = data.get("data", {}).get("markdown", "")
            title = data.get("data", {}).get("metadata", {}).get("title", "")
            md = _truncate_md(md)
            header = f"# {title}\nSource: {url}\n\n" if title else f"Source: {url}\n\n"
            return header + md
        else:
            return f"[ERROR]: Firecrawl scrape failed: {data}"
    except Exception as exc:
        return f"[ERROR]: Firecrawl scrape error: {exc}"


async def firecrawl_search(
    query: str,
    limit: int = 5,
    lang: str = "en",
    location: str = "",
    scrape_options: bool = True,
) -> str:
    """Search the web and return full page content for each result using Firecrawl.

    Combines web search with scraping — returns markdown content for each hit,
    not just titles/snippets. More thorough than brave_web_search but slower.

    Args:
        query: Search query string. Supports operators: "quotes", -exclude, site:, inurl:, intitle:.
        limit: Max number of results (default: 5).
        lang: Language code (default: 'en').
        location: Geo-target results (e.g., 'Poland', 'Germany').
        scrape_options: If True, return full markdown content per result. If False, only metadata.

    Returns:
        JSON string with search results including full markdown content per page.
    """
    if not FIRECRAWL_API_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not set"})

    body: dict = {"query": query, "limit": limit, "lang": lang}
    if location:
        body["location"] = location
    if scrape_options:
        body["scrapeOptions"] = {"formats": ["markdown"]}

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_FC_BASE}/search",
                json=body,
                headers=_fc_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data.get("data", []):
            md = item.get("markdown", "")
            results.append({
                "title": item.get("metadata", {}).get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("metadata", {}).get("description", ""),
                "markdown": _truncate_md(md, 8000),
            })
        return json.dumps({"results": results, "query": query}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Firecrawl search failed: {exc}"})


async def firecrawl_map(url: str, search: str = "", limit: int = 100) -> str:
    """Discover all URLs on a website using Firecrawl's map endpoint.

    Returns a list of all pages/links found on a site — useful for understanding
    site structure before scraping specific pages.

    Args:
        url: The base URL to map (e.g., 'https://example.com').
        search: Optional search query to filter discovered URLs.
        limit: Max number of URLs to return (default: 100, max: 5000).

    Returns:
        JSON string with a list of discovered URLs.
    """
    if not FIRECRAWL_API_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not set"})

    body: dict = {"url": url, "limit": min(limit, 5000)}
    if search:
        body["search"] = search

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_FC_BASE}/map",
                json=body,
                headers=_fc_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        links = data.get("links", [])
        return json.dumps({"url": url, "links_found": len(links), "links": links[:limit]}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Firecrawl map failed: {exc}"})


async def firecrawl_crawl(
    url: str,
    max_depth: int = 2,
    limit: int = 10,
    include_paths: str = "",
    exclude_paths: str = "",
) -> str:
    """Recursively crawl a website and return markdown content for all discovered pages.

    Starts at the given URL and follows links up to max_depth levels deep.
    This is an async operation — starts the crawl, polls for completion, and
    returns all results.

    Args:
        url: The starting URL to crawl from.
        max_depth: How many levels deep to follow links (default: 2).
        limit: Max number of pages to crawl (default: 10, max: 50).
        include_paths: Comma-separated glob patterns to include (e.g., '/blog/*,/docs/*').
        exclude_paths: Comma-separated glob patterns to exclude (e.g., '/admin/*,/login').

    Returns:
        JSON string with crawled pages including URL and markdown content for each.
    """
    if not FIRECRAWL_API_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not set"})

    body: dict = {
        "url": url,
        "maxDepth": max_depth,
        "limit": min(limit, 50),
        "scrapeOptions": {"formats": ["markdown"]},
    }
    if include_paths:
        body["includePaths"] = [p.strip() for p in include_paths.split(",")]
    if exclude_paths:
        body["excludePaths"] = [p.strip() for p in exclude_paths.split(",")]

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Start the crawl
            resp = await client.post(
                f"{_FC_BASE}/crawl",
                json=body,
                headers=_fc_auth_headers(),
            )
            resp.raise_for_status()
            start_data = resp.json()
            crawl_id = start_data.get("id", "")

            if not crawl_id:
                return json.dumps({"error": "No crawl ID returned", "response": start_data})

            # Poll for completion (max 90 seconds)
            for _ in range(30):
                await asyncio.sleep(3)
                status_resp = await client.get(
                    f"{_FC_BASE}/crawl/{crawl_id}",
                    headers=_fc_auth_headers(),
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()

                if status_data.get("status") == "completed":
                    pages = []
                    for item in status_data.get("data", []):
                        md = item.get("markdown", "")
                        pages.append({
                            "url": item.get("metadata", {}).get("sourceURL", ""),
                            "title": item.get("metadata", {}).get("title", ""),
                            "markdown": _truncate_md(md, 5000),
                        })
                    return json.dumps({
                        "crawl_id": crawl_id,
                        "pages_crawled": len(pages),
                        "pages": pages,
                    }, ensure_ascii=False)

                if status_data.get("status") == "failed":
                    return json.dumps({"error": "Crawl failed", "details": status_data})

            return json.dumps({"error": "Crawl timed out after 90s", "crawl_id": crawl_id})
    except Exception as exc:
        return json.dumps({"error": f"Firecrawl crawl failed: {exc}"})


async def firecrawl_extract(
    urls: str,
    prompt: str,
    schema: str = "",
) -> str:
    """Extract structured data from one or more URLs using Firecrawl's LLM-powered extraction.

    Scrapes the given URL(s) and uses an LLM to extract specific information
    based on your prompt. Useful for pulling structured data like prices, names,
    contact info, etc.

    Args:
        urls: Comma-separated list of URLs to extract from (e.g., 'https://example.com,https://other.com').
        prompt: Natural language description of what to extract (e.g., 'Extract all product names and prices').
        schema: Optional JSON schema string defining the expected output structure.

    Returns:
        JSON string with extracted data.
    """
    if not FIRECRAWL_API_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not set"})

    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    body: dict = {"urls": url_list, "prompt": prompt}
    if schema:
        try:
            body["schema"] = json.loads(schema)
        except json.JSONDecodeError:
            pass  # ignore invalid schema, let the API handle it

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_FC_BASE}/extract",
                json=body,
                headers=_fc_auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        return json.dumps(data, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"Firecrawl extract failed: {exc}"})


async def firecrawl_batch_scrape(urls: str, only_main_content: bool = True) -> str:
    """Scrape multiple URLs at once using Firecrawl's batch endpoint.

    More efficient than calling firecrawl_scrape repeatedly — submits all URLs
    in one request and polls for results.

    Args:
        urls: Comma-separated list of URLs to scrape (e.g., 'https://a.com,https://b.com').
        only_main_content: If True, return only main content (skip navbars, footers). Default True.

    Returns:
        JSON string with scraped content for each URL.
    """
    if not FIRECRAWL_API_KEY:
        return json.dumps({"error": "FIRECRAWL_API_KEY not set"})

    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    body: dict = {
        "urls": url_list,
        "formats": ["markdown"],
        "onlyMainContent": only_main_content,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{_FC_BASE}/batch/scrape",
                json=body,
                headers=_fc_auth_headers(),
            )
            resp.raise_for_status()
            start_data = resp.json()
            batch_id = start_data.get("id", "")

            if not batch_id:
                return json.dumps({"error": "No batch ID returned", "response": start_data})

            # Poll for completion (max 90 seconds)
            for _ in range(30):
                await asyncio.sleep(3)
                status_resp = await client.get(
                    f"{_FC_BASE}/batch/scrape/{batch_id}",
                    headers=_fc_auth_headers(),
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()

                if status_data.get("status") == "completed":
                    pages = []
                    for item in status_data.get("data", []):
                        md = item.get("markdown", "")
                        title = item.get("metadata", {}).get("title", "")
                        pages.append({
                            "url": item.get("metadata", {}).get("sourceURL", ""),
                            "title": title,
                            "markdown": _truncate_md(md, 5000),
                        })
                    return json.dumps({
                        "batch_id": batch_id,
                        "pages_scraped": len(pages),
                        "pages": pages,
                    }, ensure_ascii=False)

                if status_data.get("status") == "failed":
                    return json.dumps({"error": "Batch scrape failed", "details": status_data})

            return json.dumps({"error": "Batch scrape timed out after 90s", "batch_id": batch_id})
    except Exception as exc:
        return json.dumps({"error": f"Firecrawl batch scrape failed: {exc}"})


_FIRECRAWL_TOOLS = [
    FunctionTool(firecrawl_scrape),
    FunctionTool(firecrawl_search),
    FunctionTool(firecrawl_map),
    FunctionTool(firecrawl_crawl),
    FunctionTool(firecrawl_extract),
    FunctionTool(firecrawl_batch_scrape),
]


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

    Args:
        tool_names: List of config names such as ``"tool-google-search"``,
            ``"search_and_scrape_webpage"``, ``"jina_scrape_llm_summary"``,
            ``"tool-python"``, ``"browser"``, ``"brave-search"``,
            or ``"firecrawl-scrape"``.

    Returns:
        A list of MCPToolset / FunctionTool instances ready for an ADK Agent.
    """
    tools = []
    for name in tool_names:
        if name == "browser":
            tools.extend(_BROWSER_TOOLS)
        elif name == "brave-search":
            tools.append(_BRAVE_SEARCH_TOOL)
        elif name == "firecrawl-scrape":
            tools.append(_FIRECRAWL_TOOLS[0])  # firecrawl_scrape only
        elif name == "firecrawl":
            tools.extend(_FIRECRAWL_TOOLS)  # all 6 firecrawl tools
        elif name in _TOOL_CONFIGS:
            server_params = _TOOL_CONFIGS[name]()
            toolset = MCPToolset(
                connection_params=server_params,
            )
            tools.append(toolset)
        else:
            raise ValueError(
                f"Unknown tool config name: {name!r}. "
                f"Available: {sorted(list(_TOOL_CONFIGS.keys()) + ['browser', 'brave-search', 'firecrawl', 'firecrawl-scrape'])}"
            )
    return tools
