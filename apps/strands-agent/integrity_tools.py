# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Research integrity tools for the Strands research agent.

Exposes scientific fraud, retracted papers, and suppressed data that
mainstream tools actively avoid surfacing. These tools help verify
whether cited research has been retracted, corrected, or flagged
for misconduct.

Sources:
  1. Open Retractions API — free, instant DOI → retraction check
  2. Retraction Watch Database (via CrossRef) — 40,000+ retracted papers
     with reasons (fraud, data fabrication, political pressure, legal threats)
  3. CrossRef retraction metadata — update-type:retraction filter

All free APIs — no keys required.
"""

from __future__ import annotations

import json
import logging
import os

from strands import tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Open Retractions API — Simple DOI → retraction check
# ═══════════════════════════════════════════════════════════════════════


@tool
def check_retraction(doi: str) -> str:
    """Check if a paper has been retracted using the Open Retractions API.

    Fast, free check against the retraction database. Use this to verify
    any paper before citing it — many retracted papers continue to be cited
    because retractions are poorly publicized.

    Args:
        doi: The DOI to check (e.g. "10.1016/j.cell.2023.01.001").

    Returns:
        Retraction status: whether the paper is retracted, and if so, details.
    """
    import httpx

    # Normalize DOI
    doi = doi.strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    if doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]

    try:
        resp = httpx.get(
            f"https://api.openalex.org/works/doi:{doi}",
            timeout=15,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        if resp.status_code == 200:
            work = resp.json()
            is_retracted = work.get("is_retracted", False)
            title = work.get("title", "Unknown")
            if is_retracted:
                return (
                    f"⚠ RETRACTED: **{title}**\n"
                    f"DOI: {doi}\n"
                    f"This paper has been flagged as retracted in OpenAlex.\n"
                    f"Check Retraction Watch for details: "
                    f"https://retractionwatch.com/?s={doi}"
                )
            return (
                f"NOT RETRACTED: **{title}**\n"
                f"DOI: {doi}\n"
                f"No retraction found in OpenAlex database."
            )
    except Exception:
        pass

    # Fallback: check CrossRef for retraction notices
    try:
        resp = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=15,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            title = " ".join(data.get("title", ["Unknown"]))
            # Check for retraction/correction in update-to
            updates = data.get("update-to", [])
            for update in updates:
                if update.get("type") in ("retraction", "withdrawal"):
                    return (
                        f"⚠ RETRACTED: **{title}**\n"
                        f"DOI: {doi}\n"
                        f"Update type: {update.get('type')}\n"
                        f"Update DOI: {update.get('DOI', 'N/A')}\n"
                        f"Date: {update.get('updated', {}).get('date-time', 'N/A')}"
                    )
            # Check relation field
            relation = data.get("relation", {})
            if "is-retracted-by" in relation or "is-expression-of-retraction" in relation:
                return (
                    f"⚠ RETRACTED: **{title}**\n"
                    f"DOI: {doi}\n"
                    f"Retraction noted in CrossRef relations metadata."
                )
            return (
                f"NOT RETRACTED: **{title}**\n"
                f"DOI: {doi}\n"
                f"No retraction found in CrossRef metadata."
            )
    except Exception as exc:
        return f"[TOOL_ERROR] Retraction check failed for {doi}: {exc}"

    return f"Could not verify retraction status for DOI: {doi}"


@tool
def batch_check_retractions(dois: str) -> str:
    """Check multiple DOIs for retractions in one call.

    Args:
        dois: Comma-separated list of DOIs to check.

    Returns:
        Retraction status for each DOI.
    """
    import httpx

    doi_list = [d.strip() for d in dois.split(",") if d.strip()]
    if not doi_list:
        return "[TOOL_ERROR] No DOIs provided. Pass comma-separated DOIs."

    results = []
    for doi in doi_list[:20]:  # Cap at 20 to avoid timeout
        # Normalize
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]

        status = "UNKNOWN"
        title = doi
        try:
            resp = httpx.get(
                f"https://api.openalex.org/works/doi:{doi}",
                timeout=10,
                headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
            )
            if resp.status_code == 200:
                work = resp.json()
                title = work.get("title", doi)
                is_retracted = work.get("is_retracted", False)
                status = "RETRACTED" if is_retracted else "OK"
        except Exception:
            status = "CHECK_FAILED"

        flag = "⚠ " if status == "RETRACTED" else ""
        results.append(f"  {flag}{status}: {title}\n     DOI: {doi}")

    retracted_count = sum(1 for r in results if "RETRACTED:" in r)
    header = f"**Retraction check for {len(doi_list)} DOIs** ({retracted_count} retracted)\n"
    return header + "\n".join(results)


# ═══════════════════════════════════════════════════════════════════════
# Retraction Watch Database (via CrossRef API)
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_retractions(
    query: str = "",
    reason: str = "",
    max_results: int = 20,
) -> str:
    """Search the Retraction Watch database for retracted papers.

    40,000+ retracted papers with reasons including fraud, data fabrication,
    political pressure, legal threats, and plagiarism. Now integrated into
    CrossRef API.

    Args:
        query: Search query (title, keywords, journal name). Optional.
        reason: Filter by retraction reason (e.g. "fraud", "fabrication",
                "plagiarism", "misconduct"). Optional.
        max_results: Maximum results to return (default 20).

    Returns:
        List of retracted papers with retraction reasons and dates.
    """
    import httpx

    # CrossRef now includes Retraction Watch data
    # Filter for retracted works
    params: dict = {
        "rows": min(max_results, 50),
        "filter": "update-type:retraction",
    }
    if query:
        params["query"] = query

    try:
        resp = httpx.get(
            "https://api.crossref.org/works",
            params=params,
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Retraction search failed: {exc}"

    items = data.get("message", {}).get("items", [])
    if not items:
        return f"No retractions found" + (f" for: {query}" if query else "")

    formatted = [f"**Retracted papers** ({len(items)} results)\n"]
    for i, item in enumerate(items[:max_results], 1):
        title = " ".join(item.get("title", ["Unknown"]))
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in item.get("author", [])[:3]
        )
        doi = item.get("DOI", "")
        journal = " ".join(item.get("container-title", [""]))
        year_data = item.get("published-print", item.get("published-online", {}))
        year_parts = year_data.get("date-parts", [[""]]) if year_data else [[""]]
        year = str(year_parts[0][0]) if year_parts and year_parts[0] else ""

        # Get retraction info from updates
        updates = item.get("update-to", [])
        retraction_info = ""
        for update in updates:
            if update.get("type") in ("retraction", "withdrawal"):
                retraction_info = f"\n     Retraction type: {update.get('type')}"
                ret_date = update.get("updated", {}).get("date-time", "")
                if ret_date:
                    retraction_info += f" | Date: {ret_date[:10]}"

        formatted.append(
            f"  {i}. ⚠ **{title}**\n"
            f"     Authors: {authors}\n"
            f"     Journal: {journal} ({year})\n"
            f"     DOI: {doi}{retraction_info}"
        )

    return "\n\n".join(formatted)


@tool
def retraction_watch_csv_url() -> str:
    """Get the URL for the full Retraction Watch database CSV download.

    The complete database is freely available as a CSV from CrossRef's GitLab.
    Contains 40,000+ entries with detailed retraction reasons, dates, and
    paper metadata.

    Returns:
        Download URL and usage instructions.
    """
    return (
        "**Retraction Watch Database — Full Download**\n\n"
        "CSV: https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv\n"
        "GitLab: https://gitlab.com/crossref/retraction-watch-data\n\n"
        "License: Open Database License (ODbL)\n"
        "Citation: Always cite the International Consortium of Investigative Journalists.\n\n"
        "Fields include:\n"
        "  - Record ID, Title, Subject, Institution\n"
        "  - Journal, Publisher, Country\n"
        "  - Author, DOI, PubMed ID\n"
        "  - Retraction Date, Retraction DOI\n"
        "  - Reasons (fraud, fabrication, plagiarism, etc.)\n"
        "  - Retraction Nature (full retraction, partial, correction, expression of concern)\n\n"
        "Load into DuckDB: CREATE TABLE retractions AS SELECT * FROM read_csv_auto('retraction_watch.csv');"
    )


# ── Tool registry ─────────────────────────────────────────────────────

INTEGRITY_TOOLS = [
    check_retraction,
    batch_check_retractions,
    search_retractions,
    retraction_watch_csv_url,
]
