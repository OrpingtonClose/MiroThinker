# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Automated search executor — reads expansion targets, fires APIs programmatically.

This module replaces the researcher's search responsibility.  It reads
``expansion_tool`` + ``expansion_hint`` from the corpus table and
programmatically fires the specified API — no LLM involved.

The search executor runs between the thinker and maestro in the loop:

  Thinker (strategy) → Search Executor (automated) → Maestro (organise)

It also executes searches from the thinker's research strategy by
extracting search queries from the strategy text.

Results are ingested into the corpus via ``ingest_raw()`` and the
expansion targets are marked as fulfilled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API keys (read once at module load)
# ---------------------------------------------------------------------------
_BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
_EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
_KAGI_API_KEY = os.environ.get("KAGI_API_KEY", "")
_FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
_PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
_JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
_MOJEEK_API_KEY = os.environ.get("MOJEEK_API_KEY", "")
_APIFY_API_KEY = os.environ.get("APIFY_API_KEY", "")

# Maximum concurrent searches to prevent API rate limiting
_MAX_CONCURRENT = int(os.environ.get("SEARCH_EXECUTOR_CONCURRENCY", "4"))


# ---------------------------------------------------------------------------
# Individual search API implementations
# ---------------------------------------------------------------------------

async def _search_brave(query: str, num_results: int = 5) -> str:
    """Execute a Brave web search and return formatted results."""
    if not _BRAVE_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": _BRAVE_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("web", {}).get("results", [])
            if not results:
                return ""
            lines = [f"Brave search: {query}"]
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("description", "")
                lines.append(f"- {title} [{url}]: {desc}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Brave search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_exa(query: str, num_results: int = 5) -> str:
    """Execute an Exa semantic search and return formatted results."""
    if not _EXA_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "numResults": num_results,
                    "type": "auto",
                    "contents": {
                        "text": {"maxCharacters": 3000},
                        "highlights": {"query": query},
                    },
                },
                headers={
                    "x-api-key": _EXA_API_KEY,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return ""
            lines = [f"Exa search: {query}"]
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                text = r.get("text", "")[:500]
                highlights = " ".join(r.get("highlights", []))[:300]
                content = highlights or text
                lines.append(f"- {title} [{url}]: {content}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Exa search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_kagi(query: str) -> str:
    """Execute a Kagi fastgpt search and return formatted results."""
    if not _KAGI_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://kagi.com/api/v0/fastgpt",
                params={"query": query},
                headers={"Authorization": f"Bot {_KAGI_API_KEY}"},
            )
            resp.raise_for_status()
            data = resp.json()
            output = data.get("data", {}).get("output", "")
            refs = data.get("data", {}).get("references", [])
            if not output and not refs:
                return ""
            lines = [f"Kagi search: {query}"]
            if output:
                lines.append(output[:2000])
            for r in refs[:5]:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")[:200]
                lines.append(f"- {title} [{url}]: {snippet}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Kagi search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_tavily(query: str, num_results: int = 5) -> str:
    """Execute a Tavily search and return formatted results."""
    if not _TAVILY_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": _TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": num_results,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            results = data.get("results", [])
            if not answer and not results:
                return ""
            lines = [f"Tavily search: {query}"]
            if answer:
                lines.append(answer[:2000])
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:300]
                lines.append(f"- {title} [{url}]: {content}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Tavily search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_perplexity(query: str) -> str:
    """Execute a Perplexity sonar search (lighter than deep research)."""
    if not _PERPLEXITY_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return factual findings with citations. "
                                "No disclaimers."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                },
                headers={
                    "Authorization": f"Bearer {_PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            citations = data.get("citations", [])
            if not content:
                return ""
            lines = [f"Perplexity search: {query}", content[:3000]]
            for i, url in enumerate(citations[:10], 1):
                if isinstance(url, str):
                    lines.append(f"  [{i}] {url}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Perplexity search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_jina(query: str) -> str:
    """Execute a Jina reader search and return formatted results."""
    if not _JINA_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"https://s.jina.ai/{quote(query, safe='')}",
                headers={
                    "Authorization": f"Bearer {_JINA_API_KEY}",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", [])
            if not results:
                return ""
            lines = [f"Jina search: {query}"]
            for r in results[:5]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:500]
                lines.append(f"- {title} [{url}]: {content}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Jina search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_mojeek(query: str, num_results: int = 5) -> str:
    """Execute a Mojeek search and return formatted results."""
    if not _MOJEEK_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.mojeek.com/search",
                params={
                    "q": query,
                    "fmt": "json",
                    "t": num_results,
                    "api_key": _MOJEEK_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("response", {}).get("results", [])
            if not results:
                return ""
            lines = [f"Mojeek search: {query}"]
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("desc", "")[:200]
                lines.append(f"- {title} [{url}]: {desc}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Mojeek search failed for '%s': %s", query[:60], exc)
        return ""


# ---------------------------------------------------------------------------
# Tool routing — maps expansion_tool values to API functions
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    # Exa tools
    "web_search_exa": _search_exa,
    "web_search_advanced_exa": _search_exa,
    "crawling_exa": _search_exa,
    # Brave tools
    "brave_web_search": _search_brave,
    "brave_news_search": _search_brave,
    # Kagi tools
    "kagi_search": _search_kagi,
    "kagi_enrich_web": _search_kagi,
    "kagi_enrich_news": _search_kagi,
    # Firecrawl — use Exa as fallback (firecrawl needs MCP)
    "firecrawl_search": _search_exa,
    "firecrawl_scrape": _search_exa,
    # Tavily
    "tavily_deep_research": _search_tavily,
    # Perplexity
    "perplexity_deep_research": _search_perplexity,
    # Mojeek
    "mojeek_search": _search_mojeek,
    # Jina
    "jina_search": _search_jina,
    "jina_reader": _search_jina,
}

# Fallback search order when the specified tool isn't available
_FALLBACK_ORDER = [
    _search_exa,
    _search_brave,
    _search_tavily,
    _search_jina,
    _search_mojeek,
    _search_perplexity,
    _search_kagi,
]


async def _execute_single_search(
    tool_name: str, query: str,
) -> str:
    """Execute a single search using the specified tool or fallback."""
    # Try the specified tool first
    fn = _TOOL_DISPATCH.get(tool_name)
    if fn:
        result = await fn(query)
        if result:
            return result

    # Fallback: try each search API in order until one works
    for fallback_fn in _FALLBACK_ORDER:
        if fallback_fn == fn:
            continue  # Already tried this one
        result = await fallback_fn(query)
        if result:
            return result

    return ""


# ---------------------------------------------------------------------------
# Strategy query extraction
# ---------------------------------------------------------------------------

# Patterns that extract explicit search queries from the thinker's
# strategy.  ORDER MATTERS — explicit "SEARCH_QUERY:" markers are
# checked first, then verb-based patterns, then numbered lists last
# (numbered lists are noisy and often pick up meta-instructions).
_QUERY_PATTERNS = [
    # Explicit marker: "SEARCH_QUERY: ..." or "QUERY: ..."
    re.compile(r"(?:SEARCH_)?QUERY:\s*['\"]?(.+?)['\"]?\s*$", re.M),
    # Quoted queries: "search for 'X'" / search for "X"
    re.compile(r"[Ss]earch\s+(?:for\s+)?['\"](.+?)['\"]\s*(?:\.|$)", re.M),
    # "Search for X" (unquoted)
    re.compile(r"[Ss]earch\s+(?:for\s+)?(.+?)\s*(?:\.|$)", re.M),
    # "Look up X" / "look into X"
    re.compile(r"[Ll]ook\s+(?:up|into)\s+['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    # "Find X" / "find information about X"
    re.compile(
        r"[Ff]ind\s+(?:information\s+(?:about|on)\s+)?['\"]?(.+?)['\"]?\s*(?:\.|$)",
        re.M,
    ),
    # "Investigate X"
    re.compile(r"[Ii]nvestigate\s+['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    # "Research X"
    re.compile(r"[Rr]esearch\s+['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    # Numbered list items: "1. Query text" — ONLY if short and looks
    # like a search query (no verbs like "should", "must", "consider")
    re.compile(r"^\s*(?:\d+[.)]\s*|-\s+)(.{15,120})\s*$", re.M),
]


# Words that indicate a line is a meta-instruction, not a search query.
# NOTE: words that are also common research topics (strategy, approach,
# methodology, framework) have a trailing space to avoid matching inside
# compound terms like "AI strategy developments".
_SKIP_WORDS = [
    "evidence_sufficient", "stop searching", "do not", "don't",
    "the researcher", "the thinker", "the maestro",
    "should ", "must ", "consider ", "ensure ", "focus on",
    "prioritize", "prioritise", "important to", "note that",
    "strategy ", "approach ", "methodology ", "framework ",
    "we need to", "we should", "next step", "in order to",
    "this will", "this would", "hypothesis", "assess ",
    "evaluate ", "determine ", "analyze ", "analyse ",
]


def extract_search_queries(strategy_text: str) -> list[str]:
    """Extract search queries from the thinker's research strategy.

    Uses pattern matching to find explicit search instructions.
    Heavily filters meta-instructions that the thinker produces
    (e.g. "Consider investigating…", "We should focus on…").
    Returns deduplicated queries, capped at 8.
    """
    if not strategy_text or not strategy_text.strip():
        return []

    queries: list[str] = []
    seen: set[str] = set()

    # The last pattern in _QUERY_PATTERNS is the numbered-list pattern
    # which is noisy and captures meta-instructions.  Verb-prefixed
    # patterns ("search for X", "find X") already have strong signal
    # so we trust those extractions without skip-word filtering.
    noisy_pattern = _QUERY_PATTERNS[-1]

    for pattern in _QUERY_PATTERNS:
        for match in pattern.finditer(strategy_text):
            q = match.group(1).strip()
            # Clean up: remove trailing punctuation, quotes
            q = q.rstrip(".,;:!?\"')")
            # Skip too short or too long
            if len(q) < 10 or len(q) > 200:
                continue
            # Only apply meta-instruction filter to the noisy
            # numbered-list pattern — high-signal verb patterns
            # ("search for X", "find X") are trusted as-is.
            lower = q.lower()
            if pattern is noisy_pattern:
                if any(skip in lower for skip in _SKIP_WORDS):
                    continue
            normalised = lower.strip()
            if normalised not in seen:
                seen.add(normalised)
                queries.append(q)

    return queries[:8]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_search_executor(
    state: dict,
    cancel: threading.Event | None = None,
) -> dict[str, int]:
    """Run the automated search executor.

    Reads expansion targets from the corpus and the thinker's strategy,
    fires the appropriate APIs, and ingests results into the corpus.

    Args:
        state: Pipeline session state dict.
        cancel: Optional threading.Event.  When set, the executor stops
            ingesting results and returns early.  Used by
            ``search_executor_callback`` to prevent an orphaned worker
            thread from accessing DuckDB after timeout.

    Returns a dict with execution stats.
    """
    from callbacks.condition_manager import _get_corpus

    corpus = _get_corpus(state)
    iteration = state.get("_corpus_iteration", 0)
    strategy = state.get("research_strategy", "")

    # Check for EVIDENCE_SUFFICIENT — don't search if thinker says stop
    if strategy and "EVIDENCE_SUFFICIENT" in strategy:
        logger.info("Search executor: thinker says EVIDENCE_SUFFICIENT, skipping")
        return {"skipped": True, "reason": "EVIDENCE_SUFFICIENT"}

    stats: dict[str, int] = {
        "expansion_searches": 0,
        "strategy_searches": 0,
        "total_results": 0,
        "total_ingested": 0,
    }

    # Collect all search tasks
    # (tool, query, source_type, target_id_or_None)
    search_tasks: list[tuple[str, str, str, int | None]] = []

    # 1. Expansion targets from the corpus (quality/specificity gates)
    expansion_targets = corpus.get_expansion_targets()
    for target in expansion_targets[:6]:
        tool = target.get("strategy", "brave_web_search")
        hint = target.get("hint", "")
        if hint:
            search_tasks.append((tool, hint, f"expansion_{tool}", target["id"]))
            stats["expansion_searches"] += 1

    # 2. Strategy queries from the thinker
    strategy_queries = extract_search_queries(strategy)
    for query in strategy_queries[:6]:
        # Distribute across ALL available search engines for diversity
        idx = len(search_tasks)
        tools_cycle = [
            "web_search_advanced_exa",
            "brave_web_search",
            "tavily_deep_research",
            "perplexity_deep_research",
            "mojeek_search",
            "kagi_search",
        ]
        tool = tools_cycle[idx % len(tools_cycle)]
        search_tasks.append((tool, query, "strategy_search", None))
        stats["strategy_searches"] += 1
        logger.info("Strategy query [%s]: %s", tool, query[:80])

    if not search_tasks:
        logger.info("Search executor: no searches to execute")
        return stats

    logger.info(
        "Search executor: %d searches to execute "
        "(%d expansion, %d strategy)",
        len(search_tasks),
        stats["expansion_searches"],
        stats["strategy_searches"],
    )

    # Execute searches with concurrency limit
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    async def _bounded_search(
        tool: str, query: str, source_type: str,
        target_id: int | None,
    ) -> tuple[str, str, int | None]:
        async with semaphore:
            result = await _execute_single_search(tool, query)
            return result, source_type, target_id

    tasks = [
        _bounded_search(tool, query, source_type, target_id)
        for tool, query, source_type, target_id in search_tasks
    ]

    start = time.monotonic()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.monotonic() - start

    # Track which expansion target IDs had successful searches
    fulfilled_target_ids: set[int] = set()

    # Ingest results into corpus
    for i, result in enumerate(results):
        # Check cancellation before each corpus write
        if cancel and cancel.is_set():
            logger.info("Search executor: cancelled, stopping ingestion")
            break

        if isinstance(result, Exception):
            logger.warning("Search task %d failed: %s", i, result)
            continue
        text, source_type, target_id = result
        if not text or not text.strip():
            continue

        stats["total_results"] += 1
        try:
            ids = corpus.ingest_raw(
                raw_text=text,
                source_type=source_type,
                source_ref="search_executor",
                angle=f"iteration_{iteration}",
                iteration=iteration,
            )
            stats["total_ingested"] += len(ids)
            # Only mark as fulfilled if ingestion succeeded
            if target_id is not None and ids:
                fulfilled_target_ids.add(target_id)
        except Exception:
            logger.warning(
                "Failed to ingest search result %d", i, exc_info=True,
            )

    # Mark only successfully searched expansion targets as fulfilled
    if not (cancel and cancel.is_set()):
        for target_id in fulfilled_target_ids:
            try:
                corpus.conn.execute(
                    "UPDATE conditions SET expansion_fulfilled = TRUE "
                    "WHERE id = ? AND expansion_tool != 'none'",
                    [target_id],
                )
            except Exception:
                pass

    logger.info(
        "Search executor complete: %d searches, %d results, "
        "%d atoms ingested (%.1fs)",
        len(search_tasks),
        stats["total_results"],
        stats["total_ingested"],
        elapsed,
    )
    return stats
