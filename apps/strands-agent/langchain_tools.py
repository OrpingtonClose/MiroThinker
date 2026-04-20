"""Research tools for the LangGraph researcher agent.

Re-implements the native tools from tools.py and forum_tools.py using
LangChain's @tool decorator instead of Strands. Same function bodies,
different decorator — this keeps the researcher on the same async
runtime as the orchestrator (ChatOpenAI), avoiding the dual event loop
deadlock that caused the Strands researcher to hang.

All tools are sync functions. LangGraph runs them in a thread pool,
which is fine since they're short-lived HTTP requests with explicit
timeouts.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── Forum helpers (imported from forum_tools.py internals) ────────────

_ENGLISH_FORUMS = [
    ("meso-rx.org", "en", "MesoRx — harm reduction, protocols, bloodwork"),
    ("elitefitness.com", "en", "EliteFitness — large community, vendor reviews"),
    ("professionalmuscle.com", "en", "Professional Muscle — advanced users"),
    ("anabolicminds.com", "en", "AnabolicMinds — supplements + PEDs"),
    ("forums.t-nation.com", "en", "T-Nation — training + pharma"),
    ("thinksteroids.com", "en", "ThinkSteroids — evidence-based PED"),
    ("uk-muscle.co.uk", "en", "UK-Muscle — UK community"),
    ("evolutionary.org", "en", "Evolutionary — protocols + stacking"),
]

_INTERNATIONAL_FORUMS = [
    ("extrem-bodybuilding.de", "de", "Extrem-Bodybuilding (DE) — Team-Andro successor"),
    ("sfd.pl", "pl", "SFD (PL) — largest Polish fitness forum"),
    ("hipertrofia.org", "es", "Hipertrofia (ES) — Spanish bodybuilding"),
    ("musculacion.net", "es", "Musculacion (ES) — Spanish training + PED"),
    ("superphysique.org", "fr", "Superphysique (FR) — French bodybuilding"),
    ("ironpharm.org", "ru", "IronPharm (RU) — Russian PED community"),
]

_ALL_FORUMS = _ENGLISH_FORUMS + _INTERNATIONAL_FORUMS
_FORUM_BY_DOMAIN = {f[0]: f for f in _ALL_FORUMS}

_DDG_TIMEOUT = 30
_FORUM_SEARCH_TIMEOUT = 45
_PER_FORUM_HARD_TIMEOUT = 60  # Hard timeout per forum search (covers DDGS init hang)


def _ddg_search_in_process(args: tuple) -> tuple:
    """Run a single DuckDuckGo search in an isolated process.

    The ddgs library uses curl_cffi which deadlocks when multiple threads
    try to initialize it simultaneously. Running each search in its own
    process avoids the GIL + native dlopen contention entirely.

    This function is the multiprocessing.Pool worker target.
    """
    domain, query, max_results = args
    try:
        from ddgs import DDGS

        site_query = f"site:{domain} {query}" if domain else query
        with DDGS(timeout=30) as ddgs:
            raw = list(ddgs.text(site_query, max_results=max_results))
        results = [
            {
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", ""),
            }
            for r in raw
        ]
        return (domain, results, None)
    except Exception as exc:
        return (domain, [], str(exc))


def _ddg_site_search(query: str, domain: str, max_results: int = 10) -> list[dict]:
    """Run a DuckDuckGo search scoped to a specific domain.

    Delegates to a child process to avoid curl_cffi thread deadlock.
    """
    try:
        with mp.Pool(1) as pool:
            async_result = pool.map_async(
                _ddg_search_in_process,
                [(domain, query, max_results)],
            )
            results = async_result.get(timeout=_DDG_TIMEOUT + 10)
            _, hits, err = results[0]
            if err:
                logger.debug("ddg search error for %s: %s", domain or "web", err)
            return hits
    except mp.TimeoutError:
        logger.warning("ddg search timed out for %s", domain or "web")
        return []
    except Exception as exc:
        logger.warning("ddg search failed for %s: %s", domain or "web", exc)
        return []


def _jina_extract(url: str) -> str:
    """Extract clean text from a URL via Jina Reader."""
    import httpx

    headers = {}
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text[:30000]
    except Exception as exc:
        return f"[extraction failed: {exc}]"


def _forum_search_impl(
    query: str,
    forums: str = "all",
    max_results_per_forum: int = 5,
) -> str:
    """Core forum search logic (shared by forum_search and forum_deep_dive)."""
    max_results_per_forum = min(max_results_per_forum, 10)

    if forums == "all":
        forum_list = _ALL_FORUMS
    elif forums == "english":
        forum_list = _ENGLISH_FORUMS
    elif forums == "international":
        forum_list = _INTERNATIONAL_FORUMS
    else:
        domains = [d.strip() for d in forums.split(",")]
        forum_list = [_FORUM_BY_DOMAIN[d] for d in domains if d in _FORUM_BY_DOMAIN]
        if not forum_list:
            return json.dumps({
                "error": f"No matching forums for: {forums}",
                "available": [f[0] for f in _ALL_FORUMS],
            })

    # Build args for multiprocessing pool: (domain, query, max_results)
    mp_args = [(f[0], query, max_results_per_forum) for f in forum_list]
    forum_meta = {f[0]: (f[1], f[2]) for f in forum_list}  # domain -> (lang, desc)

    # Use multiprocessing.Pool instead of ThreadPoolExecutor.
    # The ddgs library uses curl_cffi which deadlocks when multiple threads
    # try to initialize it simultaneously. Each process gets its own
    # curl_cffi instance with no contention.
    all_results = []
    try:
        with mp.Pool(min(len(forum_list), 8)) as pool:
            async_result = pool.map_async(_ddg_search_in_process, mp_args)
            try:
                raw_results = async_result.get(timeout=_FORUM_SEARCH_TIMEOUT * 2)
            except mp.TimeoutError:
                logger.warning(
                    "forum search overall timeout — terminating pool"
                )
                pool.terminate()
                raw_results = []

        for domain, hits, err in raw_results:
            lang, desc = forum_meta.get(domain, ("?", "?"))
            entry = {
                "forum": domain,
                "language": lang,
                "description": desc,
                "count": len(hits),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in hits
                ],
            }
            if err:
                entry["error"] = err
            all_results.append(entry)

    except Exception as exc:
        logger.warning("forum search pool failed: %s", exc)

    all_results.sort(key=lambda r: r["count"], reverse=True)

    total = sum(r["count"] for r in all_results)
    forums_with_results = sum(1 for r in all_results if r["count"] > 0)

    output = {
        "query": query,
        "forums_searched": len(forum_list),
        "forums_with_results": forums_with_results,
        "total_results": total,
        "per_forum": all_results,
    }
    return json.dumps(output, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════
# TIER 1 — Uncensored search tools
# ═════════════════════════════════════════════════════════════════════


@tool
def duckduckgo_search(query: str, max_results: int = 10) -> str:
    """Search the web using DuckDuckGo. Free, no API key, uncensored.

    Use this as your go-to first search — always available, no content filtering.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    results = _ddg_site_search(query, "", max_results)

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
    """Search using Mojeek's independent crawler. Not a Google/Bing proxy.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results.
    """
    import httpx

    api_key = os.environ.get("MOJEEK_API_KEY", "")
    if not api_key:
        return "Mojeek API key not configured."

    try:
        resp = httpx.get(
            "https://api.mojeek.com/search",
            params={"q": query, "fmt": "json", "t": max_results, "api_key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Mojeek search failed: {exc}"

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


@tool
def stract_search(query: str, max_results: int = 10) -> str:
    """Search using Stract — independent, open-source web search engine.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results.
    """
    import httpx

    try:
        resp = httpx.post(
            "https://stract.com/beta/api/search",
            json={"query": query, "numResults": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Stract search failed: {exc}"

    results = data.get("webpages", [])
    if not results:
        return f"No Stract results for: {query}"

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", r.get("body", ""))
        formatted.append(f"{i}. [{title}]({url})\n   {snippet}")
    return "\n\n".join(formatted)


# ═════════════════════════════════════════════════════════════════════
# TIER 2 — Content extraction tools
# ═════════════════════════════════════════════════════════════════════


@tool
def jina_read_url(url: str) -> str:
    """Extract clean text/markdown from any URL using Jina Reader.

    Args:
        url: The URL to extract content from.

    Returns:
        Clean markdown text (truncated to 15000 chars).
    """
    import httpx

    headers = {}
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text[:15000]
    except Exception as exc:
        return f"[TOOL_ERROR] Jina read failed: {exc}"


@tool
def wayback_search(url: str, limit: int = 5) -> str:
    """Search the Wayback Machine for archived snapshots of a URL.

    Args:
        url: The URL or domain to search for.
        limit: Maximum number of snapshots (default 5).

    Returns:
        List of archived snapshots with timestamps and archive URLs.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://web.archive.org/cdx/search/cdx",
            params={
                "url": url,
                "output": "json",
                "limit": limit,
                "fl": "timestamp,original,statuscode,mimetype",
                "filter": "statuscode:200",
                "collapse": "timestamp:8",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Wayback Machine search failed: {exc}"

    if not data or len(data) <= 1:
        return f"No Wayback Machine snapshots for: {url}"

    header = data[0]
    rows = data[1:]

    formatted = []
    for i, row in enumerate(rows[:limit], 1):
        record = dict(zip(header, row))
        ts = record.get("timestamp", "")
        original = record.get("original", "")
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ts
        archive_url = f"https://web.archive.org/web/{ts}/{original}"
        formatted.append(f"{i}. [{original}]({archive_url})\n   Archived: {date_str}")

    return "\n\n".join(formatted)


@tool
def wayback_fetch(url: str, timestamp: str = "") -> str:
    """Fetch an archived page from the Wayback Machine.

    Args:
        url: The original URL to fetch from the archive.
        timestamp: Specific timestamp (YYYYMMDD). Empty = most recent.

    Returns:
        Archived page content (truncated to 15000 chars).
    """
    import httpx

    if timestamp:
        archive_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    else:
        archive_url = f"https://web.archive.org/web/id_/{url}"

    try:
        resp = httpx.get(archive_url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return resp.text[:15000]
    except Exception as exc:
        return f"[TOOL_ERROR] Wayback Machine fetch failed: {exc}"


# ═════════════════════════════════════════════════════════════════════
# TIER 3 — Censored fallback
# ═════════════════════════════════════════════════════════════════════


@tool
def google_search(query: str, max_results: int = 10) -> str:
    """Search Google via Serper API. Powerful but censored — use as fallback.

    Args:
        query: The search query string.
        max_results: Maximum number of results (default 10).

    Returns:
        Formatted Google search results.
    """
    import httpx

    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Serper API key not configured."

    try:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Google search failed: {exc}"

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


# ═════════════════════════════════════════════════════════════════════
# DEEP RESEARCH tools
# ═════════════════════════════════════════════════════════════════════


@tool
def perplexity_deep_research(query: str, model: str = "sonar-deep-research") -> str:
    """Run a deep research query via Perplexity's sonar-deep-research model.

    Multi-step web research with citations. Best for broad, complex topics.
    High latency (2-4 min).

    Args:
        query: The research question.
        model: "sonar-deep-research" (thorough) or "sonar" (faster).

    Returns:
        Research results with citations.
    """
    import httpx

    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        return "[TOOL_ERROR] PERPLEXITY_API_KEY not set."

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research data retrieval agent. Return "
                    "factual findings with full citations (URLs, source "
                    "names, dates). No disclaimers or ethical commentary."
                ),
            },
            {"role": "user", "content": query},
        ],
    }

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )
        if resp.status_code != 200:
            return f"[TOOL_ERROR] Perplexity HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

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
                citation_text = "\n\n**Sources cited:**\n" + "\n".join(citation_lines)

        result = content + citation_text
        return result if result.strip() else f"Perplexity returned empty results for: {query}"

    except Exception as exc:
        return f"[TOOL_ERROR] Perplexity error: {exc}"


@tool
def grok_deep_research(query: str, search_type: str = "both") -> str:
    """Search web and/or X/Twitter via Grok's Responses API.

    Grok performs autonomous web + X searches (5-15 per call) and returns
    cited results. Best for current events and social media discourse.

    Args:
        query: The research question.
        search_type: "web", "x" (X/Twitter only), or "both".

    Returns:
        Formatted results with citations.
    """
    import httpx

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        return "[TOOL_ERROR] XAI_API_KEY not set."

    api_base = os.environ.get("GROK_RESPONSES_API_BASE", "https://api.x.ai")
    grok_model = os.environ.get("GROK_SEARCH_MODEL", "grok-4")

    tools_list = []
    if search_type in ("web", "both"):
        tools_list.append({"type": "web_search"})
    if search_type in ("x", "both"):
        tools_list.append({"type": "x_search"})
    if not tools_list:
        tools_list = [{"type": "web_search"}, {"type": "x_search"}]

    payload = {
        "model": grok_model,
        "input": query,
        "instructions": (
            "You are a research data retrieval agent. Search for factual "
            "information and return findings with full citations (URLs, "
            "source names, dates). No disclaimers or ethical commentary."
        ),
        "tools": tools_list,
    }

    try:
        resp = httpx.post(
            f"{api_base}/v1/responses",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )
        if resp.status_code != 200:
            return f"[TOOL_ERROR] Grok HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
        output_items = data.get("output", [])
        if not output_items:
            return f"Grok returned no output for: {query}"

        search_count = 0
        search_types_used = []
        citations = []

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

        assistant_text = ""
        for item in reversed(output_items):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []):
                    text = block.get("text", "") or block.get("output_text", "")
                    if text:
                        assistant_text = text
                        break
                if assistant_text:
                    break

        if not assistant_text:
            return f"Grok produced no text output for: {query}"

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
            citation_text = "\n\n**Sources cited:**\n" + "\n".join(citation_lines)

        return header + assistant_text + citation_text

    except Exception as exc:
        return f"[TOOL_ERROR] Grok error: {exc}"


@tool
def tavily_deep_research(query: str, search_depth: str = "advanced") -> str:
    """Run an advanced search via Tavily's API.

    AI-optimised search with extracted content. "advanced" triggers deeper
    crawling and extraction.

    Args:
        query: The research question.
        search_depth: "basic" or "advanced".

    Returns:
        Formatted search results with content extracts.
    """
    import httpx

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "[TOOL_ERROR] TAVILY_API_KEY not set."

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 10,
    }

    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=120.0,
        )
        if resp.status_code != 200:
            return f"[TOOL_ERROR] Tavily HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
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
                content = r.get("content", "")
                score = r.get("score", 0)
                parts.append(f"\n  [{i}] {title}")
                parts.append(f"      URL: {url}")
                if score:
                    parts.append(f"      Relevance: {score:.2f}")
                if content:
                    parts.append(f"      {content}")

        return "\n".join(parts)

    except Exception as exc:
        return f"[TOOL_ERROR] Tavily error: {exc}"


@tool
def exa_multi_search(queries: str, num_results_per_query: int = 5) -> str:
    """Run multiple Exa searches in parallel.

    CENSORSHIP WARNING: Exa rejects health/PED queries. Prefer DuckDuckGo
    or Perplexity for sensitive topics. Use Exa for neutral subjects.

    Args:
        queries: JSON array of search query strings (max 10).
        num_results_per_query: Results per query (default 5, max 8).

    Returns:
        JSON with per-query results and unified source list.
    """
    import httpx

    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "EXA_API_KEY not set"})

    try:
        query_list = json.loads(queries)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": f"Invalid JSON queries: {queries[:200]}"})

    if not isinstance(query_list, list):
        query_list = [str(query_list)]
    query_list = query_list[:10]
    if not query_list:
        return json.dumps({"error": "No queries provided."})
    num_results_per_query = min(num_results_per_query, 8)

    def _search_one(q: str) -> dict:
        try:
            resp = httpx.post(
                "https://api.exa.ai/search",
                json={
                    "query": q,
                    "numResults": num_results_per_query,
                    "type": "auto",
                    "contents": {
                        "text": {"maxCharacters": 5000},
                        "highlights": {"query": q},
                    },
                },
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return {"query": q, "count": len(results), "results": results}
        except Exception as exc:
            return {"query": q, "count": 0, "results": [], "error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(len(query_list), 5)) as pool:
        futures = {pool.submit(_search_one, q): q for q in query_list}
        raw_results = []
        try:
            for future in as_completed(futures, timeout=120):
                try:
                    raw_results.append(future.result(timeout=60))
                except Exception as exc:
                    query_name = futures.get(future, "unknown")
                    raw_results.append({
                        "query": query_name, "count": 0,
                        "results": [], "error": f"timeout: {exc}",
                    })
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.warning("exa batch timeout — %d partial results", len(raw_results))

    order = {q: i for i, q in enumerate(query_list)}
    raw_results.sort(key=lambda b: order.get(b["query"], 999))

    total_results = sum(b["count"] for b in raw_results)
    all_sources = []
    for batch in raw_results:
        for r in batch.get("results", []):
            all_sources.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "query": batch["query"],
            })

    output = {
        "queries_executed": len(query_list),
        "total_results": total_results,
        "per_query": [
            {
                "query": b["query"],
                "count": b["count"],
                "error": b.get("error"),
                "top_results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": " ".join(r.get("highlights", []))[:200]
                        or r.get("text", "")[:200],
                    }
                    for r in b.get("results", [])[:5]
                ],
            }
            for b in raw_results
        ],
        "all_sources": all_sources[:30],
    }
    return json.dumps(output, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════
# FORUM tools
# ═════════════════════════════════════════════════════════════════════


@tool
def forum_search(query: str, forums: str = "all", max_results_per_forum: int = 5) -> str:
    """Search bodybuilding & PED forums for practitioner knowledge.

    Searches across multiple bodybuilding forums simultaneously using
    site-scoped DuckDuckGo. Returns results from real users discussing
    protocols, bloodwork, dosing, side effects.

    WHEN TO USE: Always use this for PED protocols, cycle planning,
    hormone stacking, training under gear, insulin/GH protocols.

    Args:
        query: Search query (e.g. "trenbolone insulin timing protocol").
        forums: "all", "english", "international", or comma-separated domains.
        max_results_per_forum: Results per forum (default 5, max 10).

    Returns:
        JSON with per-forum results including titles, URLs, and snippets.
    """
    return _forum_search_impl(query, forums, max_results_per_forum)


@tool
def forum_read_thread(url: str) -> str:
    """Extract full text from a forum thread URL.

    Use after forum_search to read complete content of a promising thread.
    Works with any forum URL.

    Args:
        url: Full URL of the forum thread.

    Returns:
        Clean text content (up to 30000 chars).
    """
    return _jina_extract(url)


@tool
def forum_deep_dive(
    query: str,
    forums: str = "all",
    max_threads: int = 3,
    max_results_per_forum: int = 5,
) -> str:
    """Search forums AND extract full text from top threads in one call.

    Combines forum_search + thread extraction. Use when you want deep
    forum knowledge in a single tool call.

    WARNING: Makes many HTTP requests, can take 30-60s.

    Args:
        query: Search query.
        forums: Forum selection (same as forum_search).
        max_threads: Number of top threads to extract (default 3, max 5).
        max_results_per_forum: Results per forum for initial search.

    Returns:
        JSON with search results + full extracted text for top threads.
    """
    max_threads = min(max_threads, 5)

    search_raw = _forum_search_impl(query, forums, max_results_per_forum)
    search_data = json.loads(search_raw)

    all_urls = []
    seen: set[str] = set()
    for forum_result in search_data.get("per_forum", []):
        for r in forum_result.get("results", []):
            url = r.get("url", "")
            if url and url not in seen:
                all_urls.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "forum": forum_result.get("forum", ""),
                    "snippet": r.get("snippet", ""),
                })
                seen.add(url)

    threads_to_extract = all_urls[:max_threads]

    def _extract_one(entry: dict) -> dict:
        text = _jina_extract(entry["url"])
        return {**entry, "full_text": text}

    extracted = []
    if threads_to_extract:
        pool = ThreadPoolExecutor(max_workers=min(len(threads_to_extract), 3))
        futures = {pool.submit(_extract_one, e): e for e in threads_to_extract}
        try:
            for future in as_completed(futures, timeout=120):
                try:
                    extracted.append(future.result(timeout=60))
                except Exception as exc:
                    logger.warning("thread extraction timed out: %s", exc)
        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.warning(
                "thread extraction timeout — %d partial results",
                len(extracted),
            )
        finally:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

    output = {
        "query": query,
        "search_summary": {
            "forums_searched": search_data.get("forums_searched", 0),
            "forums_with_results": search_data.get("forums_with_results", 0),
            "total_results": search_data.get("total_results", 0),
        },
        "extracted_threads": extracted,
        "remaining_urls": [u["url"] for u in all_urls[max_threads:]],
    }
    return json.dumps(output, ensure_ascii=False)


@tool
def forum_list() -> str:
    """List all registered bodybuilding & PED forums.

    Returns:
        JSON array of forum entries with domain, language, description.
    """
    return json.dumps(
        [
            {"domain": domain, "language": lang, "description": desc}
            for domain, lang, desc in _ALL_FORUMS
        ],
        ensure_ascii=False,
    )


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════


def get_all_research_tools() -> list:
    """Return all research tools as LangChain StructuredTool instances."""
    return [
        # Tier 1 — Uncensored
        duckduckgo_search,
        mojeek_search,
        stract_search,
        # Tier 2 — Extraction
        jina_read_url,
        wayback_search,
        wayback_fetch,
        # Tier 3 — Censored fallback
        google_search,
        # Deep research
        perplexity_deep_research,
        grok_deep_research,
        tavily_deep_research,
        exa_multi_search,
        # Forums
        forum_search,
        forum_read_thread,
        forum_deep_dive,
        forum_list,
    ]
