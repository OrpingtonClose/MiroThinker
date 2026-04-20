# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Bodybuilding & PED forum scraping tools.

Provides site-scoped search and thread extraction for bodybuilding forums
where practitioner knowledge lives — protocols, bloodwork, dosing
adjustments, side effect management. These forums are the primary source
of real-world experience reports that academic literature does not cover.

Forums covered:
  English:
    MesoRx (meso-rx.org) — gold standard for harm reduction + protocols
    EliteFitness (elitefitness.com) — large community, vendor reviews
    Professional Muscle (professionalmuscle.com) — advanced users
    AnabolicMinds (anabolicminds.com) — supplement + PED discussion
    T-Nation (forums.t-nation.com) — training + pharma subforum
    ThinkSteroids (thinksteroids.com) — evidence-based PED discussion
    UK-Muscle (uk-muscle.co.uk) — UK community
    Evolutionary.org (evolutionary.org) — protocols + stacking

  International:
    extrem-bodybuilding.de (DE) — successor to Team-Andro
    sfd.pl (PL) — largest Polish fitness forum
    hipertrofia.org (ES) — Spanish bodybuilding
    musculacion.net (ES) — Spanish training + PED
    superphysique.org (FR) — French bodybuilding
    ironpharm.org (RU) — Russian PED community

Strategy: DuckDuckGo site-scoped search (uncensored, free, no key) +
Jina Reader for full thread text extraction. No per-forum API keys needed.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from strands import tool

logger = logging.getLogger(__name__)


# ── Forum registry ───────────────────────────────────────────────────

# Each entry: (domain, language, description)
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

_FORUM_DOMAINS = {f[0] for f in _ALL_FORUMS}
_FORUM_BY_DOMAIN = {f[0]: f for f in _ALL_FORUMS}


# Per-search timeout (seconds) — prevents DDGS from blocking indefinitely
_DDG_TIMEOUT = 30
# Per-forum timeout in ThreadPoolExecutor (seconds)
_FORUM_SEARCH_TIMEOUT = 45


def _ddg_site_search(query: str, domain: str, max_results: int = 10) -> list[dict]:
    """Run a DuckDuckGo search scoped to a specific domain."""
    from ddgs import DDGS

    site_query = f"site:{domain} {query}"
    with DDGS(timeout=_DDG_TIMEOUT) as ddgs:
        return list(ddgs.text(site_query, max_results=max_results))


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


# ═══════════════════════════════════════════════════════════════════════
# Public tools
# ═══════════════════════════════════════════════════════════════════════


@tool
def forum_search(query: str, forums: str = "all", max_results_per_forum: int = 5) -> str:
    """Search bodybuilding & PED forums for practitioner knowledge.

    Searches across multiple bodybuilding forums simultaneously using
    site-scoped DuckDuckGo. Returns results from real users discussing
    protocols, bloodwork, dosing, side effects — knowledge that academic
    literature does not cover.

    WHEN TO USE: Always use this for PED protocols, cycle planning,
    hormone stacking, training under gear, insulin/GH protocols, and
    any topic where practitioner experience matters more than theory.

    Args:
        query: Search query (e.g. "trenbolone insulin timing protocol").
        forums: Which forums to search. Options:
            "all" — all English + international forums (default)
            "english" — English forums only
            "international" — international forums only
            Comma-separated domains for specific forums
            (e.g. "meso-rx.org,elitefitness.com")
        max_results_per_forum: Results per forum (default 5, max 10).

    Returns:
        JSON with per-forum results including titles, URLs, and snippets.
    """
    max_results_per_forum = min(max_results_per_forum, 10)

    # Resolve forum list
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

    def _search_forum(forum_entry: tuple) -> dict:
        domain, lang, desc = forum_entry
        try:
            results = _ddg_site_search(query, domain, max_results_per_forum)
            return {
                "forum": domain,
                "language": lang,
                "description": desc,
                "count": len(results),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in results
                ],
            }
        except Exception as exc:
            return {
                "forum": domain,
                "language": lang,
                "description": desc,
                "count": 0,
                "results": [],
                "error": str(exc),
            }

    # Search all forums in parallel with per-future timeout
    with ThreadPoolExecutor(max_workers=min(len(forum_list), 8)) as pool:
        futures = {pool.submit(_search_forum, f): f for f in forum_list}
        all_results = []
        for future in as_completed(futures, timeout=_FORUM_SEARCH_TIMEOUT * 2):
            try:
                all_results.append(future.result(timeout=_FORUM_SEARCH_TIMEOUT))
            except Exception as exc:
                domain = futures[future][0] if future in futures else "unknown"
                logger.warning("forum search timed out for %s: %s", domain, exc)
                all_results.append({
                    "forum": domain,
                    "count": 0,
                    "results": [],
                    "error": f"timeout: {exc}",
                })

    # Sort by result count (most results first)
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


@tool
def forum_read_thread(url: str) -> str:
    """Extract full text from a forum thread URL.

    Use this after forum_search to read the complete content of a
    promising thread. Extracts clean text via Jina Reader, preserving
    post structure, quotes, and user information.

    Works with any forum URL — not limited to the registered forums.

    Args:
        url: Full URL of the forum thread to extract.

    Returns:
        Clean text content of the thread (up to 30000 chars).
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

    Combines forum_search + forum_read_thread for the most relevant
    threads. Use this when you want deep forum knowledge in a single
    tool call — searches all forums, then extracts the top N threads.

    WARNING: This tool makes many HTTP requests and can take 30-60s.
    Use forum_search first if you just need to scan what's available.

    Args:
        query: Search query (e.g. "GH insulin timing pre workout").
        forums: Forum selection (same as forum_search).
        max_threads: Number of top threads to extract (default 3, max 5).
        max_results_per_forum: Results per forum for initial search.

    Returns:
        JSON with search results + full extracted text for top threads.
    """
    max_threads = min(max_threads, 5)

    # First: search
    search_raw = forum_search(
        query=query,
        forums=forums,
        max_results_per_forum=max_results_per_forum,
    )
    search_data = json.loads(search_raw)

    # Collect all result URLs, ranked by appearance order
    all_urls = []
    seen = set()
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

    # Extract top threads in parallel
    threads_to_extract = all_urls[:max_threads]

    def _extract_one(entry: dict) -> dict:
        text = _jina_extract(entry["url"])
        return {**entry, "full_text": text}

    extracted = []
    if threads_to_extract:
        with ThreadPoolExecutor(max_workers=min(len(threads_to_extract), 3)) as pool:
            futures = {pool.submit(_extract_one, e): e for e in threads_to_extract}
            for future in as_completed(futures, timeout=120):
                try:
                    extracted.append(future.result(timeout=60))
                except Exception as exc:
                    logger.warning("thread extraction timed out: %s", exc)

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

    Returns the full registry of forums that forum_search can query,
    with domain, language, and description for each.

    Returns:
        JSON array of forum entries.
    """
    return json.dumps(
        [
            {"domain": domain, "language": lang, "description": desc}
            for domain, lang, desc in _ALL_FORUMS
        ],
        ensure_ascii=False,
    )


# ── Export list ───────────────────────────────────────────────────────

FORUM_TOOLS = [forum_search, forum_read_thread, forum_deep_dive, forum_list]
