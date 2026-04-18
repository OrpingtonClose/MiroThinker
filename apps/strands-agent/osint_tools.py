# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
OSINT & censorship-resistant tools for the Strands research agent.

Provides access to deleted/censored web content, dark web intelligence,
leaked corporate data, and infrastructure for bypassing content removal.

Sources:
  1. Wayback Machine (enhanced) — recover deleted/censored web pages
  2. IPFS gateway — access DMCA-removed content on decentralized storage
  3. Beacon Censorship Database — global censorship tracking
  4. Common Crawl — 300B+ historical web pages including removed content

Note: OnionClaw, Onion Search, OSINT MCP, OSINT Intelligence Platform,
and OSINT Tools are configured as MCP servers in mcp_configs.py.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote, quote_plus

from strands import tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Wayback Machine (enhanced) — censored content recovery
# ═══════════════════════════════════════════════════════════════════════


@tool
def wayback_cdx_search(
    url: str,
    match_type: str = "prefix",
    from_date: str = "",
    to_date: str = "",
    max_results: int = 20,
    filter_status: str = "200",
) -> str:
    """Search the Wayback Machine CDX API for archived snapshots. Free.

    The CDX API provides detailed access to all archived snapshots of a URL,
    allowing recovery of deleted pages, censored content, and historical
    versions of any website. More powerful than the standard Wayback API.

    Args:
        url: URL to search for archived snapshots (domain or full URL).
        match_type: "exact" for exact URL, "prefix" for all pages under URL,
                    "host" for all pages on domain, "domain" for domain + subdomains.
        from_date: Start date YYYYMMDD (e.g. "20200101"). Optional.
        to_date: End date YYYYMMDD. Optional.
        max_results: Maximum snapshots to return (default 20).
        filter_status: HTTP status filter (default "200" for successful captures).

    Returns:
        List of archived snapshots with timestamps and access URLs.
    """
    import httpx

    params: dict = {
        "url": url,
        "matchType": match_type,
        "output": "json",
        "limit": min(max_results, 100),
        "fl": "timestamp,original,mimetype,statuscode,digest,length",
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    if filter_status:
        params["filter"] = f"statuscode:{filter_status}"

    try:
        resp = httpx.get(
            "https://web.archive.org/cdx/search/cdx",
            params=params,
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Wayback CDX search failed: {exc}"

    if not data or len(data) <= 1:
        return f"No Wayback Machine snapshots found for: {url}"

    # First row is headers
    headers_row = data[0] if data else []
    rows = data[1:] if len(data) > 1 else []

    formatted = [f"**Wayback Machine snapshots for: {url}** ({len(rows)} captures)\n"]
    for i, row in enumerate(rows[:max_results], 1):
        timestamp = row[0] if len(row) > 0 else ""
        original = row[1] if len(row) > 1 else ""
        mimetype = row[2] if len(row) > 2 else ""
        status = row[3] if len(row) > 3 else ""
        size = row[5] if len(row) > 5 else ""

        # Format timestamp
        display_date = timestamp
        if len(timestamp) >= 8:
            display_date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
            if len(timestamp) >= 14:
                display_date += f" {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"

        archive_url = f"https://web.archive.org/web/{timestamp}/{original}"

        formatted.append(
            f"  {i}. [{display_date}] {original}\n"
            f"     Status: {status} | Type: {mimetype} | Size: {size}\n"
            f"     Archive: {archive_url}"
        )

    return "\n\n".join(formatted)


@tool
def wayback_diff(url: str, timestamp1: str, timestamp2: str) -> str:
    """Compare two Wayback Machine snapshots of the same URL.

    Useful for detecting what was changed/censored between two dates.

    Args:
        url: The original URL to compare.
        timestamp1: First snapshot timestamp (YYYYMMDDHHMMSS).
        timestamp2: Second snapshot timestamp (YYYYMMDDHHMMSS).

    Returns:
        Comparison URL and instructions for viewing the diff.
    """
    diff_url = (
        f"https://web.archive.org/web/diff/{timestamp1}/{timestamp2}/{url}"
    )
    view1 = f"https://web.archive.org/web/{timestamp1}/{url}"
    view2 = f"https://web.archive.org/web/{timestamp2}/{url}"

    return (
        f"**Wayback Machine diff for: {url}**\n\n"
        f"Snapshot 1: {view1}\n"
        f"Snapshot 2: {view2}\n"
        f"Diff view: {diff_url}\n\n"
        f"Use the diff view to see what was added, removed, or modified between "
        f"the two snapshots — this reveals censorship and content changes."
    )


# ═══════════════════════════════════════════════════════════════════════
# IPFS Gateway — Decentralized content access
# ═══════════════════════════════════════════════════════════════════════


@tool
def ipfs_fetch(cid: str, path: str = "") -> str:
    """Fetch content from IPFS via public gateway. Free.

    IPFS content is censorship-resistant — once pinned, it can't be
    DMCA-removed. Useful for accessing papers and data that have been
    removed from centralized services.

    Args:
        cid: IPFS Content ID (CID), e.g. "QmXoypiz..."
        path: Optional path within the CID (e.g. "/paper.pdf").

    Returns:
        Content fetched from IPFS (text preview for text, info for binary).
    """
    import httpx

    # Try multiple gateways
    gateways = [
        f"https://ipfs.io/ipfs/{cid}{path}",
        f"https://dweb.link/ipfs/{cid}{path}",
        f"https://cloudflare-ipfs.com/ipfs/{cid}{path}",
        f"https://gateway.pinata.cloud/ipfs/{cid}{path}",
    ]

    for gateway_url in gateways:
        try:
            resp = httpx.head(gateway_url, timeout=15, follow_redirects=True)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                content_length = resp.headers.get("content-length", "unknown")

                if "text" in content_type or "json" in content_type:
                    # Fetch text content
                    text_resp = httpx.get(gateway_url, timeout=30, follow_redirects=True)
                    return (
                        f"**IPFS content: {cid}{path}**\n"
                        f"Gateway: {gateway_url}\n"
                        f"Type: {content_type} | Size: {content_length}\n\n"
                        f"{text_resp.text[:5000]}"
                    )
                else:
                    return (
                        f"**IPFS content: {cid}{path}**\n"
                        f"Gateway: {gateway_url}\n"
                        f"Type: {content_type} | Size: {content_length}\n\n"
                        f"Binary content — download from: {gateway_url}"
                    )
        except Exception:
            continue

    return f"[TOOL_ERROR] Could not fetch IPFS content {cid}{path} from any gateway."


# ═══════════════════════════════════════════════════════════════════════
# Common Crawl — 300B+ historical web pages
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_common_crawl(
    url: str,
    max_results: int = 10,
    index: str = "",
) -> str:
    """Search Common Crawl for archived web pages. Free, no key needed.

    300B+ pages from historical web crawls. Contains content that has been
    removed from the live web, including censored articles, deleted corporate
    pages, and political content removed under government pressure.

    Args:
        url: URL pattern to search for (supports wildcards: "*.example.com/*").
        max_results: Maximum results (default 10).
        index: Common Crawl index (e.g. "CC-MAIN-2024-10"). Leave empty for latest.

    Returns:
        List of matching pages from Common Crawl archive.
    """
    import httpx

    # Get latest index if not specified
    if not index:
        try:
            idx_resp = httpx.get(
                "https://index.commoncrawl.org/collinfo.json",
                timeout=15,
            )
            if idx_resp.status_code == 200:
                indices = idx_resp.json()
                if indices:
                    index = indices[0].get("id", "CC-MAIN-2024-10")
        except Exception:
            index = "CC-MAIN-2024-10"

    try:
        resp = httpx.get(
            f"https://index.commoncrawl.org/{index}-index",
            params={
                "url": url,
                "output": "json",
                "limit": min(max_results, 50),
            },
            timeout=30,
        )
        resp.raise_for_status()
        # Response is newline-delimited JSON
        lines = resp.text.strip().split("\n")
        results = [json.loads(line) for line in lines if line.strip()]
    except Exception as exc:
        return f"[TOOL_ERROR] Common Crawl search failed: {exc}"

    if not results:
        return f"No Common Crawl results for: {url}"

    formatted = [f"**Common Crawl results for: {url}** ({len(results)} pages)\n"]
    for i, page in enumerate(results[:max_results], 1):
        page_url = page.get("url", "")
        timestamp = page.get("timestamp", "")
        mime = page.get("mime", "")
        status = page.get("status", "")
        length = page.get("length", "")
        filename = page.get("filename", "")

        display_date = timestamp
        if len(timestamp) >= 8:
            display_date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"

        formatted.append(
            f"  {i}. [{display_date}] {page_url}\n"
            f"     Status: {status} | Type: {mime} | Size: {length}\n"
            f"     WARC: {filename}"
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# Beacon Censorship Database
# ═══════════════════════════════════════════════════════════════════════


@tool
def beacon_censorship_info() -> str:
    """Information about the Beacon Censorship Database (Norwegian National Library).

    Tracks global censorship events — which books, articles, and media have
    been banned, where, when, and why. No public API, but provides bulk data
    access and web search.

    Returns:
        Access URLs and usage instructions.
    """
    return (
        "**Beacon for Freedom of Expression — Global Censorship Database**\n\n"
        "Maintained by the Norwegian National Library.\n"
        "Tracks 100,000+ censorship events globally since antiquity.\n\n"
        "Web search: https://www.beaconforfreedom.org/search.html\n"
        "About: https://www.beaconforfreedom.org/about.html\n\n"
        "Data includes:\n"
        "  - Books, periodicals, films, music censored by governments\n"
        "  - Country, year, type of censorship (ban, seizure, prosecution)\n"
        "  - Legal basis for censorship\n"
        "  - Status (still banned, lifted, etc.)\n\n"
        "Also see: PEN America Banned Books Index\n"
        "  URL: https://pen.org/banned-books-database/\n"
        "  10,000+ book bans in US schools/libraries with reasons and locations."
    )


# ═══════════════════════════════════════════════════════════════════════
# IACR ePrint — Cryptography & privacy research
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_iacr_eprint(query: str, max_results: int = 10) -> str:
    """Search IACR ePrint Archive for cryptography research. Free, no key.

    Contains research on encryption, privacy, surveillance countermeasures,
    zero-knowledge proofs, and anonymous communication — topics that some
    governments actively suppress.

    Args:
        query: Search query (title, keywords, author names).
        max_results: Maximum results (default 10).

    Returns:
        Formatted list of cryptography papers with PDF links.
    """
    import httpx

    try:
        # IACR ePrint has a simple search endpoint
        resp = httpx.get(
            "https://eprint.iacr.org/search",
            params={
                "q": query,
            },
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
            follow_redirects=True,
        )
        resp.raise_for_status()

        # Parse results from HTML
        import re

        # Find paper entries
        papers = re.findall(
            r'<a href="/(\d{4}/\d+)"[^>]*>([^<]+)</a>.*?'
            r'(?:<em>([^<]*)</em>)?',
            resp.text,
            re.DOTALL,
        )

        if not papers:
            # Try alternative pattern
            paper_blocks = re.findall(
                r'class="paper-title"[^>]*>.*?<a href="/(\d{4}/\d+)"[^>]*>(.*?)</a>',
                resp.text,
                re.DOTALL,
            )
            if paper_blocks:
                formatted = [f"**IACR ePrint: {query}** ({len(paper_blocks)} results)\n"]
                for i, (paper_id, title) in enumerate(paper_blocks[:max_results], 1):
                    title = re.sub(r"<[^>]+>", "", title).strip()
                    formatted.append(
                        f"  {i}. **{title}**\n"
                        f"     ID: {paper_id}\n"
                        f"     PDF: https://eprint.iacr.org/{paper_id}.pdf\n"
                        f"     URL: https://eprint.iacr.org/{paper_id}"
                    )
                return "\n\n".join(formatted)

            return (
                f"**IACR ePrint Archive — Search for: {query}**\n\n"
                f"Direct search: https://eprint.iacr.org/search?q={quote_plus(query)}\n"
                f"Browse: https://eprint.iacr.org/\n\n"
                f"All papers are free PDFs. Categories include:\n"
                f"  - Public-key cryptography\n"
                f"  - Secret-key cryptography\n"
                f"  - Privacy and anonymity\n"
                f"  - Zero-knowledge proofs\n"
                f"  - Surveillance countermeasures"
            )

        formatted = [f"**IACR ePrint: {query}** ({len(papers)} results)\n"]
        for i, (paper_id, title, authors) in enumerate(papers[:max_results], 1):
            title = re.sub(r"<[^>]+>", "", title).strip()
            authors = re.sub(r"<[^>]+>", "", authors).strip() if authors else ""
            formatted.append(
                f"  {i}. **{title}**\n"
                + (f"     Authors: {authors}\n" if authors else "")
                + f"     ID: {paper_id}\n"
                f"     PDF: https://eprint.iacr.org/{paper_id}.pdf\n"
                f"     URL: https://eprint.iacr.org/{paper_id}"
            )
        return "\n\n".join(formatted)

    except Exception as exc:
        return (
            f"**IACR ePrint Archive — Search for: {query}**\n\n"
            f"Direct search: https://eprint.iacr.org/search?q={quote_plus(query)}\n"
            f"Error: {exc}"
        )


# ── Tool registry ─────────────────────────────────────────────────────

OSINT_TOOLS = [
    wayback_cdx_search,
    wayback_diff,
    ipfs_fetch,
    search_common_crawl,
    beacon_censorship_info,
    search_iacr_eprint,
]
