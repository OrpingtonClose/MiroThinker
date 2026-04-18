# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Preprint server tools for the Strands research agent.

Bypasses journal gatekeeping by accessing pre-peer-review research directly
from preprint servers. These contain studies that journals may reject for
political, commercial, or methodological reasons — including negative results,
replication failures, and controversial findings.

Sources:
  1. bioRxiv  — biology preprints (Cold Spring Harbor Laboratory)
  2. medRxiv  — medical/health preprints (Cold Spring Harbor Laboratory)
  3. ChemRxiv — chemistry preprints (American Chemical Society)
  4. SSRN     — social science, economics, law preprints (Elsevier)
  5. OSF Preprints — umbrella for 30+ community preprint servers
     (SocArXiv, EarthArXiv, engrXiv, LawArXiv, MarXiv, PaleorXiv, etc.)

All free APIs — no keys required for basic use.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

from strands import tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# bioRxiv / medRxiv — Cold Spring Harbor Laboratory
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_biorxiv(
    query: str,
    server: str = "biorxiv",
    max_results: int = 10,
    date_from: str = "",
    date_to: str = "",
) -> str:
    """Search bioRxiv or medRxiv for preprints. Free, no API key needed.

    These servers contain pre-peer-review research — studies that may never
    be published because they're controversial, show negative results, or
    challenge established narratives. Contains COVID-origin research,
    gain-of-function studies, and drug efficacy data that was politically
    suppressed during review.

    Args:
        query: Search query (title, keywords, author names).
        server: "biorxiv" or "medrxiv" (default: "biorxiv").
        max_results: Maximum results to return (default 10, max 75).
        date_from: Start date YYYY-MM-DD (default: 30 days ago).
        date_to: End date YYYY-MM-DD (default: today).

    Returns:
        Formatted list of preprints with metadata and PDF links.
    """
    import httpx

    if server not in ("biorxiv", "medrxiv"):
        return "[TOOL_ERROR] server must be 'biorxiv' or 'medrxiv'"

    if not date_from:
        date_from = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not date_to:
        date_to = datetime.now().strftime("%Y-%m-%d")

    max_results = min(max_results, 75)

    try:
        # Use the content detail endpoint for richer metadata
        resp = httpx.get(
            f"https://api.biorxiv.org/details/{server}/{date_from}/{date_to}/0/{max_results}",
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] {server} API failed: {exc}"

    collection = data.get("collection", [])
    if not collection:
        # Fallback: try the search via jats endpoint
        return f"No {server} results for date range {date_from} to {date_to}. Try broader dates."

    # Filter by query terms (API returns all papers in date range)
    query_lower = query.lower()
    query_terms = query_lower.split()
    filtered = []
    for paper in collection:
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        authors = (paper.get("authors") or "").lower()
        text = f"{title} {abstract} {authors}"
        if all(term in text for term in query_terms):
            filtered.append(paper)

    if not filtered:
        # Return all papers if no matches (user may want to browse)
        filtered = collection[:max_results]

    formatted = [f"**{server} preprints for: {query}** ({len(filtered)} results)\n"]
    for i, paper in enumerate(filtered[:max_results], 1):
        doi = paper.get("doi", "")
        title = paper.get("title", "Unknown")
        authors = paper.get("authors", "")
        # Truncate long author lists
        if len(authors) > 150:
            authors = authors[:150] + "..."
        date = paper.get("date", "")
        category = paper.get("category", "")
        abstract = (paper.get("abstract") or "")[:300]
        version = paper.get("version", "1")
        pdf_url = f"https://www.{server}.org/content/{doi}v{version}.full.pdf" if doi else ""

        formatted.append(
            f"  {i}. **{title}**\n"
            f"     Authors: {authors}\n"
            f"     Date: {date} | Category: {category} | Version: {version}\n"
            f"     DOI: {doi}\n"
            f"     PDF: {pdf_url}\n"
            f"     {abstract}..."
        )

    return "\n\n".join(formatted)


@tool
def search_biorxiv_by_doi(doi: str, server: str = "biorxiv") -> str:
    """Get full details for a specific bioRxiv/medRxiv preprint by DOI.

    Args:
        doi: The DOI of the preprint (e.g. "10.1101/2024.01.01.123456").
        server: "biorxiv" or "medrxiv" (default: "biorxiv").

    Returns:
        Full metadata including title, authors, abstract, dates, and PDF link.
    """
    import httpx

    try:
        resp = httpx.get(
            f"https://api.biorxiv.org/details/{server}/{doi}",
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] {server} DOI lookup failed: {exc}"

    collection = data.get("collection", [])
    if not collection:
        return f"No {server} preprint found for DOI: {doi}"

    paper = collection[0]
    version = paper.get("version", "1")
    pdf_url = f"https://www.{server}.org/content/{doi}v{version}.full.pdf"

    return (
        f"**{paper.get('title', 'Unknown')}**\n"
        f"Authors: {paper.get('authors', '')}\n"
        f"Date: {paper.get('date', '')} | Category: {paper.get('category', '')}\n"
        f"Version: {version} | Type: {paper.get('type', '')}\n"
        f"DOI: {doi}\n"
        f"PDF: {pdf_url}\n\n"
        f"**Abstract:**\n{paper.get('abstract', 'No abstract available.')}"
    )


# ═══════════════════════════════════════════════════════════════════════
# ChemRxiv — American Chemical Society / Cambridge Open Engage
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_chemrxiv(query: str, max_results: int = 10) -> str:
    """Search ChemRxiv for chemistry preprints. Free, no API key needed.

    Contains research on banned substance synthesis, environmental contaminant
    studies, drug precursor chemistry, and other topics that published journals
    may self-censor.

    Args:
        query: Search query (title, keywords, author names).
        max_results: Maximum results to return (default 10).

    Returns:
        Formatted list of chemistry preprints with metadata.
    """
    import httpx

    try:
        # ChemRxiv uses Cambridge Open Engage API
        resp = httpx.get(
            "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items",
            params={
                "term": query,
                "limit": min(max_results, 50),
                "sort": "RELEVANT_DESC",
            },
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] ChemRxiv search failed: {exc}"

    items = data.get("itemHits", [])
    if not items:
        return f"No ChemRxiv results for: {query}"

    formatted = [f"**ChemRxiv preprints for: {query}** ({len(items)} results)\n"]
    for i, hit in enumerate(items[:max_results], 1):
        item = hit.get("item", {})
        title = item.get("title", "Unknown")
        authors = ", ".join(
            f"{a.get('firstName', '')} {a.get('lastName', '')}".strip()
            for a in item.get("authors", [])[:5]
        )
        if len(item.get("authors", [])) > 5:
            authors += f" +{len(item['authors']) - 5} more"
        date = item.get("statusDate", "")[:10]
        doi = item.get("doi", "")
        abstract = (item.get("abstract") or "")[:300]
        # Remove HTML tags
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()
        categories = ", ".join(
            cat.get("name", "") for cat in item.get("categories", [])[:3]
        )
        pdf_url = ""
        for asset in item.get("asset", {}).get("original", {}).get("url", ""), :
            if asset:
                pdf_url = asset

        formatted.append(
            f"  {i}. **{title}**\n"
            f"     Authors: {authors}\n"
            f"     Date: {date} | Categories: {categories}\n"
            f"     DOI: {doi}\n"
            + (f"     PDF: {pdf_url}\n" if pdf_url else "")
            + (f"     {abstract}..." if abstract else "")
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# SSRN — Social Science Research Network (Elsevier)
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_ssrn(query: str, max_results: int = 10) -> str:
    """Search SSRN for social science, economics, and law preprints.

    1M+ papers covering politically sensitive economics papers, studies on
    censorship itself, classified policy analysis, and legal scholarship
    that challenges establishment positions.

    Args:
        query: Search query (title, keywords, author names).
        max_results: Maximum results to return (default 10).

    Returns:
        Formatted list of SSRN papers with metadata and download links.
    """
    import httpx

    try:
        # SSRN search via their public API endpoint
        resp = httpx.get(
            "https://api.ssrn.com/content/v1/papers",
            params={
                "query": query,
                "limit": min(max_results, 25),
            },
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        if resp.status_code == 200:
            data = resp.json()
            papers = data.get("papers", data.get("results", []))
        else:
            # Fallback: scrape SSRN search results
            resp2 = httpx.get(
                "https://papers.ssrn.com/sol3/results.cfm",
                params={"txtKey_Words": query, "npage": 1},
                timeout=30,
                headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
                follow_redirects=True,
            )
            resp2.raise_for_status()
            # Parse basic info from HTML
            import re
            titles = re.findall(r'class="title"[^>]*>([^<]+)</a>', resp2.text)
            authors_list = re.findall(r'class="authors"[^>]*>([^<]+)', resp2.text)
            abstracts = re.findall(r'class="abstract-text"[^>]*>([^<]+)', resp2.text)
            links = re.findall(r'href="(https://ssrn\.com/abstract=\d+)"', resp2.text)

            if not titles:
                return f"No SSRN results for: {query}"

            formatted = [f"**SSRN papers for: {query}** ({len(titles)} results)\n"]
            for i, title in enumerate(titles[:max_results], 1):
                author = authors_list[i - 1].strip() if i <= len(authors_list) else ""
                abstract = abstracts[i - 1].strip()[:200] if i <= len(abstracts) else ""
                link = links[i - 1] if i <= len(links) else ""
                formatted.append(
                    f"  {i}. **{title.strip()}**\n"
                    f"     Authors: {author}\n"
                    f"     URL: {link}\n"
                    + (f"     {abstract}..." if abstract else "")
                )
            return "\n\n".join(formatted)

    except Exception as exc:
        return f"[TOOL_ERROR] SSRN search failed: {exc}"

    if not papers:
        return f"No SSRN results for: {query}"

    formatted = [f"**SSRN papers for: {query}** ({len(papers)} results)\n"]
    for i, paper in enumerate(papers[:max_results], 1):
        title = paper.get("title", "Unknown")
        authors = paper.get("authors", "")
        if isinstance(authors, list):
            authors = ", ".join(str(a) for a in authors[:5])
        abstract = (paper.get("abstract") or "")[:300]
        ssrn_id = paper.get("id", paper.get("ssrn_id", ""))
        url = f"https://ssrn.com/abstract={ssrn_id}" if ssrn_id else ""
        date = paper.get("create_date", paper.get("date", ""))

        formatted.append(
            f"  {i}. **{title}**\n"
            f"     Authors: {authors}\n"
            f"     Date: {date}\n"
            f"     URL: {url}\n"
            + (f"     {abstract}..." if abstract else "")
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# OSF Preprints — Open Science Framework (ONE API for 30+ servers)
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_osf_preprints(
    query: str,
    provider: str = "",
    max_results: int = 10,
) -> str:
    """Search OSF Preprints — one API for 30+ preprint communities.

    Covers SocArXiv, EarthArXiv, engrXiv, LawArXiv, MarXiv, PaleorXiv,
    AgriXiv, Frenxiv, INA-Rxiv, and many more. Includes negative results
    and replication failures that journals won't publish.

    Args:
        query: Search query (title, keywords, author names).
        provider: Filter by provider (e.g. "socarxiv", "eartharxiv", "lawarxiv").
                  Leave empty to search all providers.
        max_results: Maximum results to return (default 10).

    Returns:
        Formatted list of preprints from multiple communities.
    """
    import httpx

    params: dict = {
        "filter[title,description]": query,
        "page[size]": min(max_results, 25),
    }
    if provider:
        params["filter[provider]"] = provider

    try:
        resp = httpx.get(
            "https://api.osf.io/v2/preprints/",
            params=params,
            timeout=30,
            headers={
                "User-Agent": "MiroThinker/1.0 (research agent)",
                "Accept": "application/vnd.api+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] OSF Preprints search failed: {exc}"

    preprints = data.get("data", [])
    if not preprints:
        return f"No OSF Preprints results for: {query}" + (
            f" (provider: {provider})" if provider else ""
        )

    formatted = [
        f"**OSF Preprints for: {query}**"
        + (f" (provider: {provider})" if provider else "")
        + f" ({len(preprints)} results)\n"
    ]

    for i, pp in enumerate(preprints[:max_results], 1):
        attrs = pp.get("attributes", {})
        title = attrs.get("title", "Unknown")
        date = (attrs.get("date_published") or attrs.get("date_created") or "")[:10]
        doi = attrs.get("doi", "")
        abstract = (attrs.get("description") or "")[:300]
        # Remove HTML tags
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()

        # Get provider name from relationships
        provider_data = pp.get("relationships", {}).get("provider", {}).get("data", {})
        provider_name = provider_data.get("id", "") if isinstance(provider_data, dict) else ""

        # Links
        links = pp.get("links", {})
        html_url = links.get("html", links.get("self", ""))
        preprint_doi_url = f"https://doi.org/{doi}" if doi else ""

        formatted.append(
            f"  {i}. **{title}**\n"
            f"     Provider: {provider_name} | Date: {date}\n"
            + (f"     DOI: {doi}\n" if doi else "")
            + (f"     URL: {html_url}\n" if html_url else "")
            + (f"     {abstract}..." if abstract else "")
        )

    return "\n\n".join(formatted)


@tool
def list_osf_providers() -> str:
    """List all available OSF preprint providers (30+ communities).

    Returns the full list of preprint communities accessible via the
    OSF Preprints API — use provider IDs with search_osf_preprints().

    Returns:
        List of provider names and IDs.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://api.osf.io/v2/preprint_providers/",
            params={"page[size]": 50},
            timeout=30,
            headers={
                "User-Agent": "MiroThinker/1.0 (research agent)",
                "Accept": "application/vnd.api+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Failed to list OSF providers: {exc}"

    providers = data.get("data", [])
    if not providers:
        return "No OSF preprint providers found."

    formatted = [f"**OSF Preprint Providers** ({len(providers)} communities)\n"]
    for p in providers:
        pid = p.get("id", "")
        attrs = p.get("attributes", {})
        name = attrs.get("name", pid)
        desc = (attrs.get("description") or "")[:100]
        import re
        desc = re.sub(r"<[^>]+>", "", desc).strip()
        count = attrs.get("preprint_count", "?")
        formatted.append(f"  - **{name}** (id: `{pid}`) — {count} preprints")
        if desc:
            formatted.append(f"    {desc}")

    return "\n\n".join(formatted)


# ── Tool registry ─────────────────────────────────────────────────────

PREPRINT_TOOLS = [
    search_biorxiv,
    search_biorxiv_by_doi,
    search_chemrxiv,
    search_ssrn,
    search_osf_preprints,
    list_osf_providers,
]
