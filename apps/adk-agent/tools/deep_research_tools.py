# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Deep research tools — Perplexity, Grok, Tavily.

Commercial deep research services for broad/complex questions that
regular search tools (Brave, Exa, Kagi) cannot handle efficiently.
The thinker assigns sub-questions to the RIGHT tool tier based on
Phase 0 scout assessments.

These tools have higher latency (2-5 minutes) and cost ($0.10-$5 per
call) compared to regular search tools.  Budget gates in
``before_tool.py`` enforce session and monthly spending limits.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
_XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

_GROK_SEARCH_MODEL = os.environ.get("GROK_SEARCH_MODEL", "grok-3")
_GROK_API_BASE = os.environ.get(
    "GROK_RESPONSES_API_BASE", "https://api.x.ai"
)

# Maximum chars to return per deep search call (cost + context control)
_MAX_RESPONSE_CHARS = 12_000


# ---------------------------------------------------------------------------
# Perplexity Deep Research
# ---------------------------------------------------------------------------

async def perplexity_deep_research(
    query: str,
    model: str = "sonar-deep-research",
) -> str:
    """Run a deep research query via Perplexity's sonar-deep-research model.

    Perplexity performs autonomous multi-step web research and returns a
    comprehensive answer with citations.  Best for broad, complex topics
    that need synthesis across many sources.

    Args:
        query: The research question to investigate deeply.
        model: Perplexity model to use.  ``sonar-deep-research`` for
            comprehensive research (slower, ~2-4 min), ``sonar`` for
            faster but less thorough results.

    Returns:
        Research results with citations, or error message.
    """
    if not _PERPLEXITY_API_KEY:
        return (
            "[TOOL_ERROR] Perplexity deep research unavailable: "
            "PERPLEXITY_API_KEY not set."
        )

    headers = {
        "Authorization": f"Bearer {_PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research data retrieval agent.  Return "
                    "factual findings with full citations (URLs, source "
                    "names, dates).  No disclaimers or ethical commentary."
                ),
            },
            {"role": "user", "content": query},
        ],
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
        ) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers,
            )

        if resp.status_code != 200:
            return (
                f"[TOOL_ERROR] Perplexity returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        # Extract citations if available
        citations = data.get("citations", [])
        citation_text = ""
        if citations:
            citation_lines = []
            for i, url in enumerate(citations[:20], 1):
                if isinstance(url, str):
                    citation_lines.append(f"  [{i}] {url}")
                elif isinstance(url, dict):
                    citation_lines.append(
                        f"  [{i}] {url.get('url', url.get('title', ''))}"
                    )
            if citation_lines:
                citation_text = (
                    "\n\n**Sources cited:**\n" + "\n".join(citation_lines)
                )

        result = content[:_MAX_RESPONSE_CHARS] + citation_text
        return result if result.strip() else (
            f"Perplexity returned empty results for: {query}"
        )

    except httpx.TimeoutException:
        return (
            "[TOOL_ERROR] Perplexity deep research timed out after 300s."
        )
    except Exception as exc:
        return f"[TOOL_ERROR] Perplexity deep research error: {exc}"


# ---------------------------------------------------------------------------
# Grok Deep Research
# ---------------------------------------------------------------------------

async def grok_deep_research(
    query: str,
    search_type: str = "both",
) -> str:
    """Search the web and/or X/Twitter via Grok's Responses API.

    Grok performs autonomous web + X searches (5-15 per call) and
    returns cited results.  Best for current events, social media
    sentiment, and topics with strong X/Twitter discourse.

    Args:
        query: The research question to investigate.
        search_type: ``"web"``, ``"x"`` (X/Twitter only), or ``"both"``.

    Returns:
        Formatted search results with citations, or error message.
    """
    if not _XAI_API_KEY:
        return (
            "[TOOL_ERROR] Grok deep research unavailable: "
            "XAI_API_KEY not set."
        )

    # Build tool list based on search_type
    tools: list[dict] = []
    if search_type in ("web", "both"):
        tools.append({"type": "web_search"})
    if search_type in ("x", "both"):
        tools.append({"type": "x_search"})
    if not tools:
        tools = [{"type": "web_search"}, {"type": "x_search"}]

    payload = {
        "model": _GROK_SEARCH_MODEL,
        "input": query,
        "instructions": (
            "You are a research data retrieval agent.  Search for factual "
            "information and return findings with full citations (URLs, "
            "source names, dates).  No disclaimers or ethical commentary.  "
            "Return raw findings in structured format."
        ),
        "tools": tools,
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=15.0),
        ) as client:
            resp = await client.post(
                f"{_GROK_API_BASE}/v1/responses",
                json=payload,
                headers={
                    "Authorization": f"Bearer {_XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
            )

        if resp.status_code != 200:
            return (
                f"[TOOL_ERROR] Grok returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        data = resp.json()
        return _format_grok_output(data, query)

    except httpx.TimeoutException:
        return "[TOOL_ERROR] Grok deep research timed out after 300s."
    except Exception as exc:
        return f"[TOOL_ERROR] Grok deep research error: {exc}"


def _format_grok_output(data: dict, query: str) -> str:
    """Parse Grok Responses API output into structured text."""
    output_items = data.get("output", [])
    if not output_items:
        return f"Grok returned no output for: {query}"

    search_count = 0
    search_types_used: list[str] = []
    citations: list[dict] = []

    for item in output_items:
        item_type = item.get("type", "")
        if item_type in ("web_search_call", "x_search_call"):
            search_count += 1
            st = "web" if item_type == "web_search_call" else "X/Twitter"
            search_types_used.append(st)
        if item_type == "web_search_result":
            for result in item.get("results", []):
                citations.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                })

    # Extract assistant message
    assistant_text = ""
    for item in reversed(output_items):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                text = (
                    block.get("text", "")
                    or block.get("output_text", "")
                )
                if text:
                    assistant_text = text
                    break
            if assistant_text:
                break

    if not assistant_text:
        return f"Grok produced no text output for: {query}"

    if len(assistant_text) > _MAX_RESPONSE_CHARS:
        assistant_text = (
            assistant_text[:_MAX_RESPONSE_CHARS] + "\n[... truncated ...]"
        )

    search_summary = ", ".join(set(search_types_used)) or "unknown"
    header = (
        f"**Grok Deep Search: {query}**\n"
        f"({search_count} searches via {search_summary})\n\n"
    )

    citation_text = ""
    if citations:
        citation_lines = [
            f"  [{i}] {c['title']} — {c['url']}"
            for i, c in enumerate(citations[:20], 1)
        ]
        citation_text = (
            "\n\n**Sources cited:**\n" + "\n".join(citation_lines)
        )

    return header + assistant_text + citation_text


# ---------------------------------------------------------------------------
# Tavily Deep Research
# ---------------------------------------------------------------------------

async def tavily_deep_research(
    query: str,
    search_depth: str = "advanced",
) -> str:
    """Run an advanced search via Tavily's search API.

    Tavily provides AI-optimised search results with extracted content.
    ``search_depth="advanced"`` triggers deeper crawling and extraction.
    Best for factual questions that need precise, structured data.

    Args:
        query: The research question to investigate.
        search_depth: ``"basic"`` for quick results, ``"advanced"`` for
            deeper extraction (default).

    Returns:
        Formatted search results with content extracts, or error message.
    """
    if not _TAVILY_API_KEY:
        return (
            "[TOOL_ERROR] Tavily deep research unavailable: "
            "TAVILY_API_KEY not set."
        )

    payload = {
        "api_key": _TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 10,
    }

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
        ) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )

        if resp.status_code != 200:
            return (
                f"[TOOL_ERROR] Tavily returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        data = resp.json()
        return _format_tavily_output(data, query)

    except httpx.TimeoutException:
        return "[TOOL_ERROR] Tavily deep research timed out after 120s."
    except Exception as exc:
        return f"[TOOL_ERROR] Tavily deep research error: {exc}"


def _format_tavily_output(data: dict, query: str) -> str:
    """Parse Tavily search response into structured text."""
    answer = data.get("answer", "")
    results = data.get("results", [])

    if not answer and not results:
        return f"Tavily returned no results for: {query}"

    parts = [f"**Tavily Deep Search: {query}**\n"]

    if answer:
        parts.append(f"Summary: {answer}\n")

    if results:
        parts.append("Results:")
        for i, r in enumerate(results[:10], 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")[:300]
            score = r.get("score", 0)
            parts.append(f"\n  [{i}] {title}")
            parts.append(f"      URL: {url}")
            if score:
                parts.append(f"      Relevance: {score:.2f}")
            if content:
                parts.append(f"      {content}")

    output = "\n".join(parts)
    if len(output) > _MAX_RESPONSE_CHARS:
        output = output[:_MAX_RESPONSE_CHARS] + "\n[... truncated ...]"
    return output


# ---------------------------------------------------------------------------
# Public FunctionTool instances
# ---------------------------------------------------------------------------

perplexity_deep_research_tool = FunctionTool(perplexity_deep_research)
grok_deep_research_tool = FunctionTool(grok_deep_research)
tavily_deep_research_tool = FunctionTool(tavily_deep_research)

DEEP_RESEARCH_TOOLS = [
    perplexity_deep_research_tool,
    grok_deep_research_tool,
    tavily_deep_research_tool,
]
