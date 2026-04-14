# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Maximally rich search executor -- multi-API fan-out, content extraction,
academic search, citation following.

This module implements a 4-phase search strategy:

  Phase A: Multi-API Surface Search
    Send each query to 3+ APIs simultaneously for diverse results.

  Phase B: Content Extraction
    Top URLs from Phase A -> Jina Reader / Apify for full article text.

  Phase C: Academic Search (conditional)
    Semantic Scholar + arXiv for research papers when thinker flags
    academic sub-questions.

  Phase D: Citation Following
    Parse URLs from extracted content; follow the most promising ones
    one hop deep.

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
from urllib.parse import quote as url_quote, urlparse

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
_MARGINALIA_API_KEY = os.environ.get("MARGINALIA_API_KEY", "")
_SCITE_CLIENT_ID = os.environ.get("SCITE_CLIENT_ID", "")
_SCITE_REFRESH_TOKEN = os.environ.get("SCITE_REFRESH_TOKEN", "")
_SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# Maximum concurrent searches to prevent API rate limiting
_MAX_CONCURRENT = int(os.environ.get("SEARCH_EXECUTOR_CONCURRENCY", "4"))

# ---------------------------------------------------------------------------
# Circuit breaker: track consecutive failures per API.  After
# _CIRCUIT_BREAKER_THRESHOLD consecutive failures, the API is "tripped"
# and skipped for the rest of the pipeline run.  This prevents the
# pipeline from wasting time retrying broken APIs (e.g. arXiv returning
# 301, Semantic Scholar choking on Unicode, scite.ai 404 on OAuth).
# ---------------------------------------------------------------------------
_CIRCUIT_BREAKER_THRESHOLD = int(
    os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "2"),
)
_circuit_breaker: dict[str, int] = {}  # fn_name → consecutive failures


def _circuit_is_open(fn_name: str) -> bool:
    """Return True if the circuit breaker is tripped for this API."""
    return _circuit_breaker.get(fn_name, 0) >= _CIRCUIT_BREAKER_THRESHOLD


def _circuit_record_success(fn_name: str) -> None:
    """Reset the circuit breaker on success."""
    _circuit_breaker.pop(fn_name, None)


def _circuit_record_failure(fn_name: str) -> None:
    """Increment the consecutive failure count."""
    _circuit_breaker[fn_name] = _circuit_breaker.get(fn_name, 0) + 1
    if _circuit_is_open(fn_name):
        logger.error(
            "CIRCUIT BREAKER TRIPPED for %s after %d consecutive failures "
            "— API will be skipped for the rest of this run",
            fn_name, _circuit_breaker[fn_name],
        )


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (call between pipeline runs)."""
    _circuit_breaker.clear()

# Serendipity: fraction of strategy queries that get contrarian variants
_SERENDIPITY_RATE = float(os.environ.get("SERENDIPITY_QUERY_RATE", "0.3"))
_SERENDIPITY_ENABLED = os.environ.get("SERENDIPITY_ENABLED", "1") == "1"

# Content extraction budget per iteration — Jina Reader / Apify are
# expensive (~$0.01-0.05 each).  Keep low to stay under ~$2/run.
_MAX_CONTENT_EXTRACTIONS = int(
    os.environ.get("MAX_CONTENT_EXTRACTIONS", "3"),
)

# Citation following budget per iteration — uses Jina/Apify too.
_MAX_CITATION_FOLLOWS = int(
    os.environ.get("MAX_CITATION_FOLLOWS", "2"),
)


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
                text = r.get("text", "")
                highlights = " ".join(r.get("highlights", []))
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
            resp = await client.post(
                "https://kagi.com/api/v0/fastgpt",
                json={"query": query},
                headers={
                    "Authorization": f"Bot {_KAGI_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            output = data.get("data", {}).get("output", "")
            refs = data.get("data", {}).get("references", [])
            if not output and not refs:
                return ""
            lines = [f"Kagi search: {query}"]
            if output:
                lines.append(output)
            for r in refs[:5]:
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("snippet", "")
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
                lines.append(answer)
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
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
            lines = [f"Perplexity search: {query}", content]
            for i, url in enumerate(citations[:10], 1):
                if isinstance(url, str):
                    lines.append(f"  [{i}] {url}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Perplexity search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_jina(query: str, num_results: int = 5) -> str:
    """Execute a Jina search and return formatted results."""
    if not _JINA_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://s.jina.ai/" + url_quote(query),
                headers={
                    "Authorization": f"Bearer {_JINA_API_KEY}",
                    "Accept": "application/json",
                    "X-Retain-Images": "none",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("data", [])
            if not results:
                return ""
            lines = [f"Jina search: {query}"]
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")
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
                desc = r.get("desc", "")
                lines.append(f"- {title} [{url}]: {desc}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Mojeek search failed for '%s': %s", query[:60], exc)
        return ""


async def _search_marginalia(query: str, num_results: int = 5) -> str:
    """Execute a Marginalia search (independent index, non-commercial web)."""
    if not _MARGINALIA_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.marginalia.nu/public/search/"
                + url_quote(query),
                params={"count": num_results, "index": 0},
                headers={"X-Api-Key": _MARGINALIA_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                return ""
            lines = [f"Marginalia search: {query}"]
            for r in results[:num_results]:
                title = r.get("title", "")
                url = r.get("url", "")
                desc = r.get("description", "")
                lines.append(f"- {title} [{url}]: {desc}")
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Marginalia search failed for '%s': %s", query[:60], exc)
        return ""


# ---------------------------------------------------------------------------
# Content extraction APIs (Phase B)
# ---------------------------------------------------------------------------

async def _jina_reader(url: str) -> str:
    """Extract full markdown content from a URL via Jina Reader.

    Returns the COMPLETE markdown text — no truncation.  The full content
    flows into ``ingest_raw()`` which stores it verbatim before chunking.
    """
    if not _JINA_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "Authorization": f"Bearer {_JINA_API_KEY}",
                    "Accept": "application/json",
                    "X-Retain-Images": "none",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("data", {}).get("content", "")
            title = data.get("data", {}).get("title", "")
            if not content:
                return ""
            lines = [f"Content extracted from: {title} [{url}]"]
            lines.append(content)
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Jina reader failed for '%s': %s", url[:80], exc)
        return ""


async def _apify_extract(url: str) -> str:
    """Extract content via Apify web scraper (JS-heavy sites fallback)."""
    if not _APIFY_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.apify.com/v2/acts/apify~website-content-crawler"
                "/run-sync-get-dataset-items",
                params={
                    "token": _APIFY_API_KEY,
                    "timeout": 30,
                    "memory": 256,
                },
                json={
                    "startUrls": [{"url": url}],
                    "maxCrawlPages": 1,
                    "crawlerType": "cheerio",
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return ""
            item = data[0] if isinstance(data, list) else data
            text = item.get("text", "") or item.get("markdown", "")
            title = (
                item.get("metadata", {}).get("title", "")
                or item.get("title", "")
            )
            if not text:
                return ""
            lines = [f"Content extracted (Apify) from: {title} [{url}]"]
            lines.append(text)
            return "\n".join(lines)
    except Exception as exc:
        logger.warning("Apify extract failed for '%s': %s", url[:80], exc)
        return ""


# ---------------------------------------------------------------------------
# Academic search APIs (Phase C)
# ---------------------------------------------------------------------------

def _sanitise_query_for_api(query: str) -> str:
    """Sanitise a query string for ASCII-only APIs.

    Replaces Unicode characters that break APIs like Semantic Scholar
    (which choke on em-dashes, curly quotes, etc.) with ASCII equivalents.
    This is an architectural guardrail — not a one-off fix for a single
    character, but a systematic defence against the entire class of
    encoding failures.
    """
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00e9": "e",    # e-acute
        "\u00e8": "e",    # e-grave
        "\u00fc": "u",    # u-umlaut
        "\u00f6": "o",    # o-umlaut
        "\u00e4": "a",    # a-umlaut
    }
    for char, replacement in replacements.items():
        query = query.replace(char, replacement)
    # Final safety net: strip any remaining non-ASCII
    return query.encode("ascii", errors="replace").decode("ascii")


async def _search_semantic_scholar(
    query: str, num_results: int = 5,
) -> str:
    """Search Semantic Scholar for academic papers."""
    fn_name = "_search_semantic_scholar"
    if _circuit_is_open(fn_name):
        return ""
    # Sanitise query to prevent ASCII encoding failures
    query = _sanitise_query_for_api(query)
    try:
        headers: dict[str, str] = {}
        if _SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = _SEMANTIC_SCHOLAR_API_KEY
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": num_results,
                    "fields": (
                        "title,abstract,url,year,citationCount,"
                        "authors.name,externalIds"
                    ),
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            if not papers:
                return ""
            _circuit_record_success(fn_name)
            lines = [f"Semantic Scholar: {query}"]
            for p in papers[:num_results]:
                title = p.get("title", "")
                year = p.get("year", "")
                citations = p.get("citationCount", 0)
                abstract = p.get("abstract") or ""
                url = p.get("url", "")
                authors = ", ".join(
                    a.get("name", "")
                    for a in (p.get("authors") or [])[:3]
                )
                doi = (p.get("externalIds") or {}).get("DOI", "")
                ref = f"[{url}]" if url else ""
                if doi:
                    ref = f"[https://doi.org/{doi}]"
                lines.append(
                    f"- {title} ({year}, {citations} citations, "
                    f"{authors}) {ref}: {abstract}"
                )
            return "\n".join(lines)
    except Exception as exc:
        _circuit_record_failure(fn_name)
        logger.error(
            "Semantic Scholar search FAILED for '%s': %s", query[:60], exc,
        )
        return ""


async def _search_arxiv(query: str, num_results: int = 3) -> str:
    """Search arXiv for preprints via the Atom feed API."""
    fn_name = "_search_arxiv"
    if _circuit_is_open(fn_name):
        return ""
    # Sanitise query for ASCII safety
    query = _sanitise_query_for_api(query)
    try:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True,
        ) as client:
            resp = await client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": num_results,
                    "sortBy": "relevance",
                },
            )
            resp.raise_for_status()
            text = resp.text
            entries = re.findall(
                r"<entry>(.*?)</entry>", text, re.DOTALL,
            )
            if not entries:
                return ""
            _circuit_record_success(fn_name)
            lines = [f"arXiv search: {query}"]
            for entry in entries[:num_results]:
                title_m = re.search(
                    r"<title>(.*?)</title>", entry, re.DOTALL,
                )
                title = (title_m.group(1).strip() if title_m else "")
                title = re.sub(r"\s+", " ", title)
                summary_m = re.search(
                    r"<summary>(.*?)</summary>", entry, re.DOTALL,
                )
                summary = (
                    summary_m.group(1).strip() if summary_m else ""
                )
                summary = re.sub(r"\s+", " ", summary)
                id_m = re.search(r"<id>(.*?)</id>", entry)
                entry_url = id_m.group(1).strip() if id_m else ""
                authors = re.findall(r"<name>(.*?)</name>", entry)
                author_str = ", ".join(authors[:3])
                lines.append(
                    f"- {title} ({author_str}) [{entry_url}]: {summary}"
                )
            return "\n".join(lines)
    except Exception as exc:
        _circuit_record_failure(fn_name)
        logger.error("arXiv search FAILED for '%s': %s", query[:60], exc)
        return ""


async def _search_scite(query: str, num_results: int = 5) -> str:
    """Search scite.ai for smart citations (support/contradict analysis)."""
    fn_name = "_search_scite"
    if _circuit_is_open(fn_name):
        return ""
    if not _SCITE_REFRESH_TOKEN or not _SCITE_CLIENT_ID:
        return ""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            token_resp = await client.post(
                "https://api.scite.ai/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": _SCITE_REFRESH_TOKEN,
                    "client_id": _SCITE_CLIENT_ID,
                },
            )
            token_resp.raise_for_status()
            access_token = token_resp.json().get("access_token", "")
            if not access_token:
                return ""

            resp = await client.get(
                "https://api.scite.ai/search",
                params={"q": query, "limit": num_results},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", data.get("hits", []))
            if not results:
                return ""
            lines = [f"scite.ai search: {query}"]
            for r in results[:num_results]:
                title = r.get("title", "")
                doi = r.get("doi", "")
                supporting = r.get("supporting", 0)
                contradicting = r.get("contradicting", 0)
                mentioning = r.get("mentioning", 0)
                ref_url = f"https://doi.org/{doi}" if doi else ""
                lines.append(
                    f"- {title} [{ref_url}]: "
                    f"{supporting} supporting, {contradicting} contradicting, "
                    f"{mentioning} mentioning citations"
                )
            _circuit_record_success(fn_name)
            return "\n".join(lines)
    except Exception as exc:
        _circuit_record_failure(fn_name)
        logger.error("scite.ai search FAILED for '%s': %s", query[:60], exc)
        return ""


# ---------------------------------------------------------------------------
# Multi-API fan-out
# ---------------------------------------------------------------------------


def _available_search_fns() -> list[Any]:
    """Return all search functions whose API keys are configured."""
    fns: list[Any] = []
    if _EXA_API_KEY:
        fns.append(_search_exa)
    if _BRAVE_API_KEY:
        fns.append(_search_brave)
    if _TAVILY_API_KEY:
        fns.append(_search_tavily)
    if _PERPLEXITY_API_KEY:
        fns.append(_search_perplexity)
    if _JINA_API_KEY:
        fns.append(_search_jina)
    if _MOJEEK_API_KEY:
        fns.append(_search_mojeek)
    if _MARGINALIA_API_KEY:
        fns.append(_search_marginalia)
    if _KAGI_API_KEY:
        fns.append(_search_kagi)
    return fns


_FAN_OUT_WIDTH = int(os.environ.get("FAN_OUT_WIDTH", "2"))

# ---------------------------------------------------------------------------
# P3: Adaptive API selection — track success/failure per API and weight
# selection toward APIs that return useful results.
# ---------------------------------------------------------------------------
_api_stats: dict[str, dict[str, int]] = {}  # fn_name → {success, failure, total_chars}


def _record_api_result(fn_name: str, success: bool, chars: int = 0) -> None:
    """Record a search API call result for adaptive selection."""
    if fn_name not in _api_stats:
        _api_stats[fn_name] = {"success": 0, "failure": 0, "total_chars": 0}
    if success:
        _api_stats[fn_name]["success"] += 1
        _api_stats[fn_name]["total_chars"] += chars
    else:
        _api_stats[fn_name]["failure"] += 1


def _api_quality_score(fn_name: str) -> float:
    """Compute a quality score (0-1) for an API based on its track record.

    APIs with no history get a neutral score of 0.5.
    Score combines success rate (70% weight) and average content richness (30%).
    """
    stats = _api_stats.get(fn_name)
    if not stats:
        return 0.5  # neutral for unknown APIs

    total = stats["success"] + stats["failure"]
    if total == 0:
        return 0.5

    success_rate = stats["success"] / total
    # Average chars per successful call (normalised to 0-1 range)
    avg_chars = (
        stats["total_chars"] / max(stats["success"], 1)
    )
    richness = min(avg_chars / 5000.0, 1.0)  # 5000+ chars = max score

    return 0.7 * success_rate + 0.3 * richness


def get_api_stats() -> dict[str, Any]:
    """Return current API quality stats (for /corpus/stats endpoint)."""
    result = {}
    for fn_name, stats in _api_stats.items():
        total = stats["success"] + stats["failure"]
        result[fn_name] = {
            **stats,
            "quality_score": round(_api_quality_score(fn_name), 3),
            "success_rate": round(stats["success"] / max(total, 1), 3),
        }
    return result


def _pick_fan_out_apis(query_index: int) -> list[Any]:
    """Pick APIs for a query using quality-weighted selection.

    Width is controlled by ``FAN_OUT_WIDTH`` (default 2) to keep
    costs under ~$2/run.  APIs are sorted by quality score and
    selected with a rotating window that biases toward higher-quality
    APIs while still ensuring coverage across all APIs.
    """
    available = _available_search_fns()
    width = min(_FAN_OUT_WIDTH, len(available))
    if len(available) <= width:
        return available

    # Sort by quality score (highest first), with rotation offset
    scored = sorted(
        available,
        key=lambda fn: _api_quality_score(fn.__name__),
        reverse=True,
    )

    # Use a rotating window that starts from query_index but biases
    # toward the top-quality APIs:
    # - First slot always goes to a top-quality API (rotating among top half)
    # - Remaining slots rotate through the rest for diversity
    top_half = scored[:max(len(scored) // 2, 1)]

    picked: list[Any] = []
    # First pick: rotate among top-quality APIs
    picked.append(top_half[query_index % len(top_half)])

    # Remaining picks: rotate among the rest (including top for coverage)
    rest = [fn for fn in scored if fn not in picked]
    for i in range(width - 1):
        if rest:
            picked.append(rest[(query_index + i) % len(rest)])

    return picked[:width]


async def _execute_fan_out_search(
    query: str, query_index: int,
) -> list[str]:
    """Execute a single query against multiple APIs simultaneously.

    Returns a list of result strings (one per successful API).
    Records success/failure per API for adaptive selection.
    """
    apis = _pick_fan_out_apis(query_index)
    if not apis:
        return []

    tasks = [api(query) for api in apis]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Track results per API for adaptive selection
    good: list[str] = []
    for api_fn, r in zip(apis, results):
        if isinstance(r, str) and r and r.strip():
            _record_api_result(api_fn.__name__, success=True, chars=len(r))
            good.append(r)
        else:
            _record_api_result(api_fn.__name__, success=False)

    return good


# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------

_TOOL_DISPATCH: dict[str, Any] = {
    "web_search_exa": _search_exa,
    "web_search_advanced_exa": _search_exa,
    "crawling_exa": _search_exa,
    "brave_web_search": _search_brave,
    "brave_news_search": _search_brave,
    "kagi_search": _search_kagi,
    "kagi_enrich_web": _search_kagi,
    "kagi_enrich_news": _search_kagi,
    "firecrawl_search": _search_exa,
    "firecrawl_scrape": _search_exa,
    "tavily_deep_research": _search_tavily,
    "perplexity_deep_research": _search_perplexity,
    "mojeek_search": _search_mojeek,
    "jina_search": _search_jina,
    "jina_reader": _jina_reader,
    "marginalia_search": _search_marginalia,
    "semantic_scholar": _search_semantic_scholar,
    "arxiv_search": _search_arxiv,
    "scite_search": _search_scite,
}

_FALLBACK_ORDER = [
    _search_exa,
    _search_brave,
    _search_tavily,
    _search_jina,
    _search_mojeek,
    _search_perplexity,
    _search_marginalia,
    _search_kagi,
]


async def _execute_single_search(
    tool_name: str, query: str,
) -> str:
    """Execute a single search using the specified tool or fallback."""
    fn = _TOOL_DISPATCH.get(tool_name)
    if fn:
        result = await fn(query)
        if result:
            return result

    for fallback_fn in _FALLBACK_ORDER:
        if fallback_fn == fn:
            continue
        result = await fallback_fn(query)
        if result:
            return result

    return ""


# ---------------------------------------------------------------------------
# URL extraction from search results
# ---------------------------------------------------------------------------

_BARE_URL_RE = re.compile(r"(https?://[^\s)\]\"'>]+)")

_SKIP_DOMAINS = {
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "linkedin.com", "pinterest.com",
    "amazon.com", "reddit.com",
}


def _extract_urls_from_text(text: str) -> list[str]:
    """Extract unique URLs from search result text."""
    urls: list[str] = []
    seen: set[str] = set()
    for match in _BARE_URL_RE.finditer(text):
        url = match.group(1).rstrip(".,;:")
        normalised = url.lower()
        if normalised in seen:
            continue
        try:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            if domain in _SKIP_DOMAINS or any(
                domain.endswith("." + skip) for skip in _SKIP_DOMAINS
            ):
                continue
        except Exception:
            continue
        seen.add(normalised)
        urls.append(url)
    return urls


def _select_diverse_urls(
    urls: list[str], max_count: int,
) -> list[str]:
    """Select URLs prioritising domain diversity and authority."""
    by_domain: dict[str, list[str]] = {}
    for url in urls:
        try:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
        except Exception:
            domain = "unknown"
        by_domain.setdefault(domain, []).append(url)

    priority_suffixes = (".edu", ".gov", ".org", ".ac.uk")
    selected: list[str] = []
    seen_domains: set[str] = set()

    for domain, domain_urls in sorted(by_domain.items()):
        if len(selected) >= max_count:
            break
        if any(domain.endswith(s) for s in priority_suffixes):
            selected.append(domain_urls[0])
            seen_domains.add(domain)

    for domain, domain_urls in by_domain.items():
        if len(selected) >= max_count:
            break
        if domain not in seen_domains:
            selected.append(domain_urls[0])
            seen_domains.add(domain)

    return selected


# ---------------------------------------------------------------------------
# Unicode sanitisation for ASCII-sensitive APIs
# ---------------------------------------------------------------------------

def _sanitise_query_for_api(query: str) -> str:
    """Replace Unicode characters that break search APIs with ASCII equivalents.

    Smart quotes, em-dashes, and other typographic characters cause 422
    errors on Jina, Semantic Scholar, and arXiv.  This is an architectural
    guardrail — not a one-off fix for a single character.
    """
    replacements = {
        "\u2014": "--",   # em-dash
        "\u2013": "-",    # en-dash
        "\u2018": "'",    # left single quote
        "\u2019": "'",    # right single quote
        "\u201c": '"',    # left double quote
        "\u201d": '"',    # right double quote
        "\u2026": "...",  # ellipsis
        "\u00e9": "e",    # e-acute
        "\u00e8": "e",    # e-grave
        "\u00fc": "u",    # u-umlaut
        "\u00f6": "o",    # o-umlaut
        "\u00e4": "a",    # a-umlaut
    }
    for char, replacement in replacements.items():
        query = query.replace(char, replacement)
    return query.encode("ascii", errors="replace").decode("ascii")


# ---------------------------------------------------------------------------
# Strategy query extraction — LLM-powered deep dissolution
#
# The old approach used 7 regex patterns to shred the thinker's prose
# into fragments.  This produced shallow, generic search terms like
# "research strategy" or "exact match" that pulled in Elasticsearch
# docs and HBR strategy articles instead of neuropsychoanalysis papers.
#
# The new approach uses an LLM call to deeply deconstruct the user's
# query AND the thinker's strategy into conceptually rich, specific
# search queries that encode the full interdisciplinary depth.
# ---------------------------------------------------------------------------

# Regex fallback patterns (used ONLY when LLM dissolution fails)
_QUERY_PATTERNS = [
    re.compile(r"(?:SEARCH_)?QUERY:\s*['\"]?(.+?)['\"]?\s*$", re.M),
    re.compile(r"[Ss]earch\s+(?:for\s+)?['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    re.compile(r"[Ll]ook\s+(?:up|into)\s+['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    re.compile(
        r"[Ff]ind\s+(?:information\s+(?:about|on)\s+)?['\"]?(.+?)['\"]?\s*(?:\.|$)",
        re.M,
    ),
    re.compile(r"[Ii]nvestigate\s+['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    re.compile(r"[Rr]esearch\s+['\"]?(.+?)['\"]?\s*(?:\.|$)", re.M),
    re.compile(r"^\s*(?:\d+[.)]\s*|-\s+)(.{15,120})\s*$", re.M),
]


# ---------------------------------------------------------------------------
# Query relevance validator — architectural guardrail
#
# Rejects queries that have zero topical overlap with the user's original
# question.  This prevents the thinker's strategy decomposition from
# producing generic queries like "research strategy" or "exact match"
# that pull in Elasticsearch docs, HBR strategy articles, etc.
#
# The check is deterministic (no LLM) — it computes token overlap between
# the extracted query and the user query.  Queries with zero overlap are
# rejected.  This catches the entire class of "generic strategy noise"
# that polluted 60% of the Lacan-metabolism corpus.
# ---------------------------------------------------------------------------

_QUERY_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "about", "against", "up", "down", "what", "which",
    "who", "whom", "this", "that", "these", "those", "it", "its",
    "research", "strategy", "find", "search", "look", "investigate",
    "information", "data", "results", "analysis", "evidence", "study",
    "question", "answer", "topic", "subject", "key", "main", "sub",
    "explore", "examine", "understand", "determine", "identify",
}


def _query_relevance_score(query: str, user_query: str) -> float:
    """Compute topical overlap between an extracted query and the user query.

    Returns a score in [0, 1].  Queries with score 0 have zero topical
    overlap and should be rejected.
    """
    def _content_tokens(text: str) -> set[str]:
        tokens = set(re.findall(r'\w+', text.lower()))
        return tokens - _QUERY_STOPWORDS

    q_tokens = _content_tokens(query)
    u_tokens = _content_tokens(user_query)

    if not q_tokens:
        return 0.0
    if not u_tokens:
        # User query reduced to empty after stopword removal — we can't
        # compute meaningful overlap.  Allow the query through rather
        # than rejecting everything.
        return 1.0

    overlap = q_tokens & u_tokens
    # Jaccard-like: overlap / smaller set
    return len(overlap) / min(len(q_tokens), len(u_tokens))


def _validate_query_relevance(
    queries: list[str],
    user_query: str,
    min_score: float = 0.0,
) -> tuple[list[str], list[str]]:
    """Split queries into relevant and rejected based on topical overlap.

    Args:
        queries: Extracted search queries.
        user_query: The user's original question.
        min_score: Minimum relevance score (0 = at least 1 content token overlap).

    Returns:
        (relevant, rejected) — two lists.
    """
    if not user_query:
        return queries, []

    relevant: list[str] = []
    rejected: list[str] = []

    for q in queries:
        score = _query_relevance_score(q, user_query)
        if score > min_score:
            relevant.append(q)
        else:
            rejected.append(q)
            logger.warning(
                "QUERY REJECTED (zero topical overlap with user query): %r",
                q[:120],
            )

    if rejected:
        logger.error(
            "Query validator rejected %d/%d queries as off-topic "
            "(zero overlap with user query)",
            len(rejected), len(queries),
        )

    return relevant, rejected


_DISSOLUTION_PROMPT = """\
You are a research librarian with deep expertise across every academic field.
Your task: dissolve a research question and strategy into precise, specific
search queries that will retrieve the exact scholarly and specialist sources
needed.

RULES:
- Each query must be CONCEPTUALLY RICH: use the actual technical terms,
  named theorists, specific mechanisms, journal-quality language.
- NEVER produce generic queries like "research strategy" or "energy allocation".
  These retrieve business/tech content, not the interdisciplinary scholarship
  the user needs.
- Each query should target a DIFFERENT facet or discipline of the question.
- Include the proper nouns, named theories, specific biological pathways,
  philosophical concepts, and cross-disciplinary bridges that a domain expert
  would search for.
- Aim for 6-10 queries, each 8-25 words.
- For interdisciplinary questions, make sure queries span ALL relevant fields
  (e.g. neuroscience AND psychoanalysis AND philosophy AND biology).
- Output ONLY the queries, one per line, prefixed with "QUERY: ".
  No commentary, no numbering, no explanation.

EXAMPLE -- for a question about Lacan's psychic economy and metabolic theory:
QUERY: Lacanian Real as autonomic dysregulation dorsal vagal shutdown polyvagal theory
QUERY: jouissance allostatic overload metabolic cost psychoanalytic neuropsychoanalysis
QUERY: "subject supposed to know" transferential safety signaling ventral vagal engagement
QUERY: mTOR pathway growth signaling polyvagal safe-enough-to-grow neurobiological
QUERY: Porges polyvagal theory Lacan Symbolic order cortical narrative regulation
QUERY: imaginary register mirror stage interoception body schema neuroscience
QUERY: psychoanalytic economy libido as metabolic energy Freud beyond pleasure principle
QUERY: dorsal vagal conservation trauma freeze response dissociation neurobiology

Now dissolve this:

USER QUERY:
{user_query}

THINKER REASONING:
{strategy}
"""


def _dissolve_via_llm(
    strategy_text: str,
    user_query: str,
) -> list[str]:
    """Use LLM to dissolve strategy into conceptually rich search queries.

    Calls the Flock proxy (same LiteLLM backend used for scoring) to
    deeply deconstruct the user's query and thinker strategy into
    search queries that encode the full conceptual depth.
    """
    from utils.flock_proxy import get_flock_proxy_url
    import json as _json
    import urllib.request

    proxy_url = get_flock_proxy_url()
    if not proxy_url:
        return []

    prompt = _DISSOLUTION_PROMPT.format(
        user_query=user_query[:2000],
        strategy=strategy_text[:3000],
    )

    body = _json.dumps({
        "model": "flock-model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.4,
    }).encode()
    req = urllib.request.Request(
        f"{proxy_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        choices = data.get("choices", [])
        raw = choices[0]["message"]["content"] if choices else ""
    except Exception as exc:
        logger.warning("LLM query dissolution failed: %s", exc)
        return []

    # Parse QUERY: lines from the response
    queries: list[str] = []
    seen: set[str] = set()
    for line in raw.split("\n"):
        line = line.strip()
        if line.upper().startswith("QUERY:"):
            q = line[6:].strip().strip('"\'')
        elif line and not line.startswith("#") and len(line) > 15:
            # Accept bare lines too (LLM may skip the prefix)
            q = line.strip().strip('"\'')
        else:
            continue

        q = _sanitise_query_for_api(q)
        if len(q) < 10 or len(q) > 250:
            continue
        normalised = q.lower()
        if normalised not in seen:
            seen.add(normalised)
            queries.append(q)

    return queries[:12]


def _regex_extract_queries(strategy_text: str) -> list[str]:
    """Regex fallback -- extract queries from thinker prose.

    Used only when LLM dissolution is unavailable.
    """
    if not strategy_text or not strategy_text.strip():
        return []

    queries: list[str] = []
    seen: set[str] = set()

    for pattern in _QUERY_PATTERNS:
        for match in pattern.finditer(strategy_text):
            q = match.group(1).strip()
            q = q.rstrip(".,;:!?\"')")
            if len(q) < 10 or len(q) > 200:
                continue
            lower = q.lower()
            if any(
                skip in lower
                for skip in [
                    "evidence_sufficient",
                    "stop searching",
                    "do not",
                    "the researcher",
                    "the thinker",
                    "the maestro",
                ]
            ):
                continue
            normalised = lower.strip()
            if normalised not in seen:
                seen.add(normalised)
                queries.append(q)

    raw_queries = queries[:10]

    # ── Architectural guardrail: query relevance validation ──
    if user_query:
        relevant, rejected = _validate_query_relevance(raw_queries, user_query)
        return relevant

    return raw_queries


def extract_search_queries(
    strategy_text: str,
    user_query: str = "",
) -> list[str]:
    """Extract search queries from the thinker's research strategy.

    Primary path: LLM-powered deep dissolution that deconstructs the
    query's full conceptual architecture into specific, intellectually
    rich search queries.  Falls back to regex extraction if the LLM
    call fails.

    Returns deduplicated queries, capped at 12.
    """
    if not strategy_text or not strategy_text.strip():
        return []

    # Primary: LLM dissolution (when user_query is available)
    if user_query:
        try:
            dissolved = _dissolve_via_llm(strategy_text, user_query)
        except Exception:
            dissolved = []
        if dissolved:
            logger.info(
                "LLM query dissolution produced %d conceptually rich queries",
                len(dissolved),
            )
            # Apply relevance validation (architectural guardrail)
            relevant, rejected = _validate_query_relevance(dissolved, user_query)
            return relevant if relevant else dissolved
        logger.warning(
            "LLM query dissolution failed or empty -- falling back to regex"
        )

    # Fallback: regex extraction
    queries = _regex_extract_queries(strategy_text)

    # Apply relevance validation when user_query is available
    if user_query and queries:
        relevant, rejected = _validate_query_relevance(queries, user_query)
        return relevant if relevant else queries

    return queries


def _get_existing_corpus_urls(corpus: Any) -> set[str]:
    """Query the corpus for all source URLs already ingested.

    Returns a set of normalised URLs (lowercase, stripped of trailing slash)
    that the search executor can check against to avoid re-scraping pages
    that are already in the corpus from previous pipeline runs.
    """
    try:
        rows = corpus.conn.execute(
            "SELECT DISTINCT source_url FROM conditions "
            "WHERE source_url IS NOT NULL AND source_url != ''"
        ).fetchall()
        urls: set[str] = set()
        for (url,) in rows:
            normalised = url.strip().lower().rstrip("/")
            if normalised:
                urls.add(normalised)
        return urls
    except Exception:
        logger.debug("Could not query existing corpus URLs", exc_info=True)
        return set()


def _get_executed_query_fingerprints(state: dict) -> set[str]:
    """Return fingerprints of queries already executed in previous iterations.

    Tracks queries across iterations via session state so the search
    executor skips queries that were already run.  This prevents the
    search executor from repeating the same queries when the thinker
    generates similar strategies across expansion iterations.
    """
    executed: set[str] = set()
    raw = state.get("_executed_query_fingerprints", "")
    if raw:
        for line in raw.strip().split("\n"):
            fp = line.strip().lower()
            if fp:
                executed.add(fp)
    return executed


def _record_executed_queries(state: dict, queries: list[str]) -> None:
    """Append newly executed query fingerprints to session state."""
    existing = state.get("_executed_query_fingerprints", "")
    new_fps = "\n".join(q.strip().lower() for q in queries if q.strip())
    if existing:
        state["_executed_query_fingerprints"] = existing + "\n" + new_fps
    else:
        state["_executed_query_fingerprints"] = new_fps


_SERENDIPITY_DISSOLUTION_PROMPT = """\
You are a contrarian intellectual provocateur.
Given these research queries, generate {n} unexpected but deeply relevant
alternative queries that the researcher would NEVER think to search for.

These should be:
- From adjacent or opposing fields (e.g. if the queries are about
  psychoanalysis, try neurobiology, comparative religion, ethology)
- Historically deep (precursor theories, forgotten debates)
- Methodologically surprising (opposite paradigm, different species,
  different culture)
- Genuinely likely to surface serendipitous connections

Do NOT just add "criticism of" or "debate about" — that's lazy.
Instead, find the ADJACENT POSSIBLE: what related concept from a
different field would illuminate this question unexpectedly?

Original user question: {user_query}

Current queries:
{queries}

Output ONLY the contrarian queries, one per line, prefixed with "QUERY: ".
No commentary.
"""


def _generate_serendipitous_queries(
    queries: list[str],
    user_query: str,
    corpus_summary: str,
) -> list[str]:
    """Generate contrarian/unexpected query variants for serendipity.

    Primary path: LLM-powered serendipity that finds the adjacent
    possible — queries from neighbouring fields that illuminate the
    research question from unexpected angles.

    Fallback: deterministic templates (better than nothing).
    """
    if not _SERENDIPITY_ENABLED or not queries:
        return []

    n_variants = max(2, int(len(queries) * _SERENDIPITY_RATE))

    # Try LLM-powered serendipity first
    if user_query:
        try:
            from utils.flock_proxy import get_flock_proxy_url
            import json as _json
            import urllib.request

            proxy_url = get_flock_proxy_url()
            if proxy_url:
                prompt = _SERENDIPITY_DISSOLUTION_PROMPT.format(
                    n=n_variants,
                    user_query=user_query[:1000],
                    queries="\n".join(f"- {q}" for q in queries[:8]),
                )
                body = _json.dumps({
                    "model": "flock-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                }).encode()
                req = urllib.request.Request(
                    f"{proxy_url}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = _json.loads(resp.read())
                choices = data.get("choices", [])
                raw = choices[0]["message"]["content"] if choices else ""
                variants: list[str] = []
                for line in raw.split("\n"):
                    line = line.strip()
                    if line.upper().startswith("QUERY:"):
                        q = line[6:].strip().strip('"\'')
                    elif line and not line.startswith("#") and len(line) > 15:
                        q = line.strip().strip('"\'')
                    else:
                        continue
                    q = _sanitise_query_for_api(q)
                    if 10 <= len(q) <= 200:
                        variants.append(q)
                if variants:
                    logger.info(
                        "Serendipity (LLM): generated %d contrarian queries",
                        len(variants),
                    )
                    return variants[:n_variants + 1]
        except Exception as exc:
            logger.warning("LLM serendipity failed: %s", exc)

    # Fallback: deterministic templates
    import random
    rng = random.Random(hash(user_query) & 0xFFFFFFFF)

    _TEMPLATES = [
        "criticisms of {q}",
        "evidence against {q}",
        "{q} controversy OR debate OR disputed",
        "{q} unexpected findings OR surprising results",
        "alternative explanation for {q}",
        "historical evolution of understanding {q}",
        "{q} cross-disciplinary perspectives",
        "what was wrong about early research on {q}",
        "{q} minority viewpoint OR dissenting opinion",
        "{q} unintended consequences OR side effects",
    ]

    selected = rng.sample(queries, min(n_variants, len(queries)))
    variants = []
    for q in selected:
        short_q = q[:80].strip()
        template = rng.choice(_TEMPLATES)
        variant = template.replace("{q}", short_q)
        if len(variant) >= 10:
            variants.append(variant)

    if variants:
        logger.info(
            "Serendipity (template fallback): generated %d variants from %d queries",
            len(variants), len(selected),
        )

    return variants


def _detect_academic_need(strategy_text: str) -> bool:
    """Detect if the thinker's strategy flags academic research needs."""
    if not strategy_text:
        return False
    lower = strategy_text.lower()
    academic_signals = [
        "academic", "research paper", "peer-reviewed", "journal",
        "pubmed", "clinical trial", "meta-analysis", "systematic review",
        "scientific", "study", "mechanism", "pathophysiology",
        "molecular", "in vivo", "in vitro", "randomized controlled",
        "literature review", "evidence-based",
        "arxiv", "semantic scholar", "preprint",
    ]
    return sum(1 for s in academic_signals if s in lower) >= 2


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_search_executor(
    state: dict,
    cancel: threading.Event | None = None,
) -> dict[str, int]:
    """Run the maximally rich search executor.

    4-phase search strategy:
      A. Multi-API Surface Search -- each query to 3 APIs simultaneously
      B. Content Extraction -- top URLs via Jina Reader / Apify
      C. Academic Search -- Semantic Scholar + arXiv (when flagged)
      D. Citation Following -- parse extracted content for more URLs

    Args:
        state: Pipeline session state dict.
        cancel: Optional threading.Event for early stop.

    Returns a dict with execution stats.
    """
    from callbacks.condition_manager import _get_corpus

    corpus = _get_corpus(state)
    iteration = state.get("_corpus_iteration", 0)
    strategy = state.get("research_strategy", "")
    user_query = state.get("user_query", "")

    if strategy and "EVIDENCE_SUFFICIENT" in strategy:
        logger.info("Search executor: thinker says EVIDENCE_SUFFICIENT, skipping")
        return {"skipped": True, "reason": "EVIDENCE_SUFFICIENT"}

    stats: dict[str, int] = {
        "expansion_searches": 0,
        "strategy_searches": 0,
        "fan_out_searches": 0,
        "content_extractions": 0,
        "academic_searches": 0,
        "citation_follows": 0,
        "total_results": 0,
        "total_ingested": 0,
        "queries_deduped": 0,
        "urls_deduped": 0,
        "queries_rejected_noise": 0,
        "circuit_breakers_tripped": 0,
    }

    # ── Architectural guardrail: heartbeat emitter ──
    # Emits progress events so the SSE stream stays alive and the stall
    # detector doesn't kill productive search work.  Each phase boundary
    # emits a heartbeat with the current stats.
    try:
        from dashboard import get_active_collector
        _heartbeat_collector = get_active_collector()
    except Exception:
        _heartbeat_collector = None

    def _emit_heartbeat(phase: str) -> None:
        """Emit a search-executor heartbeat event for stall prevention."""
        if _heartbeat_collector:
            _heartbeat_collector.emit_event(
                "search_executor_heartbeat",
                data={"phase": phase, **stats},
            )
        logger.info("Search executor heartbeat: %s", phase)

    _emit_heartbeat("init")

    # ── P1: Cross-run dedup — collect URLs and query fingerprints ──
    existing_urls = _get_existing_corpus_urls(corpus)
    if existing_urls:
        logger.info(
            "Cross-run dedup: %d URLs already in corpus from previous runs",
            len(existing_urls),
        )
    executed_fps = _get_executed_query_fingerprints(state)
    if executed_fps:
        logger.info(
            "Cross-run dedup: %d query fingerprints from previous iterations",
            len(executed_fps),
        )

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    # -------------------------------------------------------------------
    # Phase A: Multi-API Surface Search
    # -------------------------------------------------------------------

    search_tasks: list[tuple[str, str, str, int | None]] = []

    # A1. Expansion targets from the corpus
    expansion_targets = corpus.get_expansion_targets()
    for target in expansion_targets[:6]:
        tool = target.get("strategy", "brave_web_search")
        hint = target.get("hint", "")
        if hint:
            # Skip expansion hints that match previously executed queries
            if hint.strip().lower() in executed_fps:
                stats["queries_deduped"] += 1
                logger.debug("Skipping duplicate expansion query: %s", hint[:80])
                continue
            search_tasks.append(
                (tool, hint, f"expansion_{tool}", target["id"]),
            )
            stats["expansion_searches"] += 1

    # A2. Strategy queries -- fan-out to multiple APIs
    # Pass user_query for both LLM dissolution and the query relevance
    # validator (architectural guardrail that rejects off-topic queries
    # before they hit any API).
    strategy_queries = extract_search_queries(strategy, user_query=user_query)

    # ── Serendipity: inject contrarian query variants ──
    # Generate unexpected-but-relevant query variants that push the
    # search toward directions the thinker wouldn't have chosen.
    # Serendipitous queries are interleaved with regular queries so
    # they share the fan-out budget rather than being silently dropped
    # when the thinker produces 6+ strategy queries.
    serendipitous_queries = _generate_serendipitous_queries(
        strategy_queries, user_query, "",
    )
    if serendipitous_queries:
        # Interleave: after every 2 regular queries, insert 1 serendipitous
        merged: list[str] = []
        s_idx = 0
        for r_idx, q in enumerate(strategy_queries):
            merged.append(q)
            if (r_idx + 1) % 2 == 0 and s_idx < len(serendipitous_queries):
                merged.append(serendipitous_queries[s_idx])
                s_idx += 1
        # Append any remaining serendipitous queries
        while s_idx < len(serendipitous_queries):
            merged.append(serendipitous_queries[s_idx])
            s_idx += 1
        strategy_queries = merged
        stats["serendipitous_queries"] = len(serendipitous_queries)

    # Budget: 6 regular + however many serendipitous queries were injected
    fan_out_cap = min(6 + len(serendipitous_queries), len(strategy_queries))
    fan_out_tasks: list[tuple[str, int]] = []
    new_queries: list[str] = []
    for i, query in enumerate(strategy_queries[:fan_out_cap]):
        # Skip queries already executed in previous iterations
        if query.strip().lower() in executed_fps:
            stats["queries_deduped"] += 1
            logger.debug("Skipping duplicate strategy query: %s", query[:80])
            continue
        fan_out_tasks.append((query, i))
        new_queries.append(query)
        stats["strategy_searches"] += 1

    # Record newly executed queries for future dedup
    all_executed = [t[1] for t in search_tasks] + new_queries
    if all_executed:
        _record_executed_queries(state, all_executed)

    if stats["queries_deduped"]:
        logger.info(
            "Cross-run query dedup: skipped %d duplicate queries",
            stats["queries_deduped"],
        )

    # Track rejected noise queries in stats BEFORE the early return so
    # that even when all queries are rejected, stats capture the root cause.
    # Uses _query_relevance_score directly to avoid double-logging.
    if user_query:
        raw_all = extract_search_queries(strategy)
        rejected_count = sum(
            1 for q in raw_all
            if _query_relevance_score(q, user_query) <= 0.0
        )
        stats["queries_rejected_noise"] = rejected_count
        stats["total_raw_queries"] = len(raw_all)

    # Track tripped circuit breakers (also before early return)
    stats["circuit_breakers_tripped"] = sum(
        1 for fn_name in _circuit_breaker
        if _circuit_is_open(fn_name)
    )

    if not search_tasks and not fan_out_tasks:
        logger.info("Search executor: no searches to execute")
        return stats

    logger.info(
        "Search executor Phase A: %d expansion + %d strategy (fan-out)",
        stats["expansion_searches"],
        stats["strategy_searches"],
    )
    _emit_heartbeat("phase_a_start")

    async def _bounded_search(
        tool: str, query: str, source_type: str,
        target_id: int | None,
    ) -> tuple[str, str, int | None]:
        async with semaphore:
            result = await _execute_single_search(tool, query)
            return result, source_type, target_id

    expansion_coros = [
        _bounded_search(tool, query, source_type, target_id)
        for tool, query, source_type, target_id in search_tasks
    ]

    async def _bounded_fan_out(
        query: str, idx: int,
    ) -> list[tuple[str, str, None]]:
        async with semaphore:
            results = await _execute_fan_out_search(query, idx)
            return [(r, "strategy_fan_out", None) for r in results]

    fan_out_coros = [
        _bounded_fan_out(query, idx) for query, idx in fan_out_tasks
    ]

    start = time.monotonic()

    # Run both expansion and fan-out in parallel
    gather_tasks: list[Any] = []
    if expansion_coros:
        gather_tasks.append(asyncio.gather(*expansion_coros, return_exceptions=True))
    if fan_out_coros:
        gather_tasks.append(asyncio.gather(*fan_out_coros, return_exceptions=True))

    all_results = await asyncio.gather(*gather_tasks, return_exceptions=True)

    flat_results: list[tuple[str, str, int | None]] = []

    result_idx = 0
    if expansion_coros:
        expansion_results = (
            all_results[result_idx]
            if not isinstance(all_results[result_idx], Exception) else []
        )
        for result in expansion_results:
            if isinstance(result, Exception):
                continue
            flat_results.append(result)
        result_idx += 1

    if fan_out_coros:
        fan_out_results = (
            all_results[result_idx]
            if not isinstance(all_results[result_idx], Exception) else []
        )
        for result in fan_out_results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                flat_results.extend(result)
                stats["fan_out_searches"] += len(result)

    elapsed_a = time.monotonic() - start

    fulfilled_target_ids: set[int] = set()
    all_result_text: list[str] = []

    for text, source_type, target_id in flat_results:
        if cancel and cancel.is_set():
            logger.info("Search executor: cancelled, stopping")
            break
        if not text or not text.strip():
            continue

        stats["total_results"] += 1
        all_result_text.append(text)
        try:
            ids = corpus.ingest_raw(
                raw_text=text,
                source_type=source_type,
                source_ref="search_executor",
                angle=f"iteration_{iteration}",
                iteration=iteration,
                user_query=user_query,
            )
            stats["total_ingested"] += len(ids)
            if target_id is not None and ids:
                fulfilled_target_ids.add(target_id)
        except Exception:
            logger.warning(
                "Failed to ingest search result", exc_info=True,
            )

    logger.info(
        "Phase A complete: %d results, %d ingested (%.1fs)",
        stats["total_results"], stats["total_ingested"], elapsed_a,
    )
    _emit_heartbeat("phase_a_complete")

    # -------------------------------------------------------------------
    # Phase B: Content Extraction (Jina Reader + Apify fallback)
    # -------------------------------------------------------------------
    extracted_text: list[str] = []
    diverse_urls: list[str] = []

    if not (cancel and cancel.is_set()):
        combined_text = "\n".join(all_result_text)
        all_urls = _extract_urls_from_text(combined_text)
        # ── Cross-run URL dedup: skip URLs already in corpus ──
        if existing_urls:
            pre_dedup = len(all_urls)
            all_urls = [
                u for u in all_urls
                if u.strip().lower().rstrip("/") not in existing_urls
            ]
            deduped = pre_dedup - len(all_urls)
            if deduped:
                stats["urls_deduped"] += deduped
                logger.info(
                    "Phase B URL dedup: skipped %d URLs already in corpus",
                    deduped,
                )
        diverse_urls = _select_diverse_urls(
            all_urls, _MAX_CONTENT_EXTRACTIONS,
        )

        if diverse_urls:
            logger.info(
                "Phase B: extracting content from %d URLs (of %d found)",
                len(diverse_urls), len(all_urls),
            )

            async def _bounded_extract(
                ext_url: str,
            ) -> tuple[str, str]:
                async with semaphore:
                    content = await _jina_reader(ext_url)
                    if not content:
                        content = await _apify_extract(ext_url)
                    return content, ext_url

            extract_coros = [_bounded_extract(u) for u in diverse_urls]
            extract_results = await asyncio.gather(
                *extract_coros, return_exceptions=True,
            )

            for result in extract_results:
                if cancel and cancel.is_set():
                    break
                if isinstance(result, Exception):
                    continue
                content, ext_url = result
                if not content or not content.strip():
                    continue

                stats["content_extractions"] += 1
                extracted_text.append(content)
                try:
                    ids = corpus.ingest_raw(
                        raw_text=content,
                        source_type="content_extraction",
                        source_ref=ext_url,
                        angle=f"iteration_{iteration}",
                        iteration=iteration,
                        user_query=user_query,
                    )
                    stats["total_ingested"] += len(ids)
                except Exception:
                    logger.warning(
                        "Failed to ingest extracted content from %s",
                        ext_url[:80], exc_info=True,
                    )

            logger.info(
                "Phase B complete: %d content extractions",
                stats["content_extractions"],
            )
            _emit_heartbeat("phase_b_complete")

    # -------------------------------------------------------------------
    # Phase C: Academic Search (conditional)
    # -------------------------------------------------------------------
    if not (cancel and cancel.is_set()) and _detect_academic_need(strategy):
        logger.info(
            "Phase C: academic search triggered by thinker strategy",
        )

        academic_queries = strategy_queries[:3]

        async def _bounded_academic(coro: Any) -> str:
            async with semaphore:
                return await coro

        academic_coros: list[Any] = []
        for q in academic_queries:
            academic_coros.append(_bounded_academic(_search_semantic_scholar(q)))
            academic_coros.append(_bounded_academic(_search_arxiv(q)))

        if _SCITE_REFRESH_TOKEN and _SCITE_CLIENT_ID:
            for q in academic_queries[:2]:
                academic_coros.append(_bounded_academic(_search_scite(q)))

        academic_results = await asyncio.gather(
            *academic_coros, return_exceptions=True,
        )

        for result in academic_results:
            if cancel and cancel.is_set():
                break
            if isinstance(result, Exception) or not result:
                continue
            if not isinstance(result, str) or not result.strip():
                continue

            stats["academic_searches"] += 1
            try:
                ids = corpus.ingest_raw(
                    raw_text=result,
                    source_type="academic_search",
                    source_ref="search_executor",
                    angle=f"iteration_{iteration}_academic",
                    iteration=iteration,
                    user_query=user_query,
                )
                stats["total_ingested"] += len(ids)
            except Exception:
                logger.warning(
                    "Failed to ingest academic result", exc_info=True,
                )

        # Report academic search failures transparently
        if stats["academic_searches"] == 0:
            logger.error(
                "Phase C: ZERO academic results ingested — all academic "
                "APIs failed or returned empty (tripped breakers: %s)",
                [
                    fn for fn in _circuit_breaker
                    if _circuit_is_open(fn)
                ],
            )
        logger.info(
            "Phase C complete: %d academic results ingested",
            stats["academic_searches"],
        )
        _emit_heartbeat("phase_c_complete")

    # -------------------------------------------------------------------
    # Phase D: Citation Following (one hop)
    # -------------------------------------------------------------------
    if (
        not (cancel and cancel.is_set())
        and extracted_text
        and stats.get("content_extractions", 0) > 0
    ):
        citation_text = "\n".join(extracted_text)
        citation_urls = _extract_urls_from_text(citation_text)
        already_extracted = set(diverse_urls)
        new_urls = [
            u for u in citation_urls
            if u not in already_extracted
            and u.strip().lower().rstrip("/") not in existing_urls
        ]
        follow_urls = _select_diverse_urls(
            new_urls, _MAX_CITATION_FOLLOWS,
        )

        if follow_urls:
            logger.info(
                "Phase D: following %d citations (of %d new URLs)",
                len(follow_urls), len(new_urls),
            )

            async def _bounded_follow(
                follow_url: str,
            ) -> tuple[str, str]:
                async with semaphore:
                    content = await _jina_reader(follow_url)
                    if not content:
                        content = await _apify_extract(follow_url)
                    return content, follow_url

            follow_coros = [_bounded_follow(u) for u in follow_urls]
            follow_results = await asyncio.gather(
                *follow_coros, return_exceptions=True,
            )

            for result in follow_results:
                if cancel and cancel.is_set():
                    break
                if isinstance(result, Exception):
                    continue
                content, follow_url = result
                if not content or not content.strip():
                    continue

                stats["citation_follows"] += 1
                try:
                    ids = corpus.ingest_raw(
                        raw_text=content,
                        source_type="citation_follow",
                        source_ref=follow_url,
                        angle=f"iteration_{iteration}_citations",
                        iteration=iteration,
                        user_query=user_query,
                    )
                    stats["total_ingested"] += len(ids)
                except Exception:
                    logger.warning(
                        "Failed to ingest citation from %s",
                        follow_url[:80], exc_info=True,
                    )

            logger.info(
                "Phase D complete: %d citations followed",
                stats["citation_follows"],
            )

    # -------------------------------------------------------------------
    # Mark expansion targets as fulfilled
    # -------------------------------------------------------------------
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

    total_elapsed = time.monotonic() - start
    stats["total_elapsed_s"] = int(total_elapsed)

    # ── Architectural guardrail: transparent failure summary ──
    # Log ERROR (not warning) for systemic failures so the diagnostic
    # tool and operators can see what went wrong.
    failure_lines: list[str] = []
    if stats.get("queries_rejected_noise", 0) > 0:
        failure_lines.append(
            f"  - {stats['queries_rejected_noise']} queries rejected as "
            f"off-topic noise"
        )
    tripped = [
        fn for fn in _circuit_breaker if _circuit_is_open(fn)
    ]
    if tripped:
        failure_lines.append(
            f"  - Circuit breakers tripped: {', '.join(tripped)}"
        )
    if stats["total_ingested"] == 0 and stats["total_results"] == 0:
        failure_lines.append(
            "  - ZERO results from ALL search APIs — complete search failure"
        )
    if failure_lines:
        logger.error(
            "Search executor FAILURE SUMMARY:\n%s",
            "\n".join(failure_lines),
        )

    logger.info(
        "Search executor complete: %d surface, %d extractions, "
        "%d academic, %d citations, %d total ingested (%.1fs)",
        stats["total_results"],
        stats["content_extractions"],
        stats["academic_searches"],
        stats["citation_follows"],
        stats["total_ingested"],
        total_elapsed,
    )
    _emit_heartbeat("complete")
    return stats
