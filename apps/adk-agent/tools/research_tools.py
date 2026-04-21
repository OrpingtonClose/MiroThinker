# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Research orchestration tools — store_finding, read_findings.

JSONL-based external accumulator so the LLM can persist findings
*outside* its context window.  Older findings can be trimmed by
Keep-K-Recent without data loss.

Note: For link extraction, use Firecrawl's ``firecrawl_map`` MCP tool
instead of a custom implementation — it does the same thing natively.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import httpx
from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# ── Configurable paths ──────────────────────────────────────────────
FINDINGS_DIR = Path(os.environ.get(
    "FINDINGS_DIR", os.path.join(os.path.expanduser("~"), ".mirothinker")
))
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Current session findings file — set by batch orchestrator or defaults
_findings_file: Path = FINDINGS_DIR / "findings.jsonl"


def set_findings_file(name: str) -> Path:
    """Set the active findings file (called by batch orchestrator)."""
    global _findings_file
    _findings_file = FINDINGS_DIR / name
    return _findings_file


def get_findings_file() -> Path:
    """Return the current findings file path."""
    return _findings_file


def clear_findings() -> None:
    """Clear the current findings file (called between batch runs)."""
    if _findings_file.exists():
        _findings_file.unlink()


# ── store_finding ───────────────────────────────────────────────────


async def store_finding(
    name: str,
    url: str,
    category: str,
    summary: str,
    rating: int = 0,
) -> str:
    """Store an evaluated finding to persistent JSONL storage.

    Findings persist outside the LLM context window so Keep-K-Recent
    can trim old tool results without losing accumulated data.

    Args:
        name: Short name / title of the finding.
        url: Source URL.
        category: Category (e.g. "vendor", "forum", "news", "academic").
        summary: One-paragraph evaluation summary.
        rating: Quality rating 1-10 (0 = unrated).

    Returns:
        Confirmation message.
    """
    finding = {
        "name": name,
        "url": url,
        "category": category,
        "summary": summary,
        "rating": rating,
        "ts": time.time(),
    }
    with open(_findings_file, "a") as f:
        f.write(json.dumps(finding, ensure_ascii=False) + "\n")

    logger.info("Stored finding: %s (%s)", name, category)
    return f"Stored: {name} [{category}] (rating={rating})"


# ── read_findings ───────────────────────────────────────────────────


async def read_findings(category: str = "") -> str:
    """Read back all stored findings, optionally filtered by category.

    Args:
        category: If non-empty, only return findings matching this category.

    Returns:
        JSON array of finding objects.
    """
    if not _findings_file.exists():
        return json.dumps([])

    findings = []
    for line in _findings_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if category and obj.get("category", "") != category:
            continue
        findings.append(obj)

    return json.dumps(findings, ensure_ascii=False)


# ── exa_multi_search ───────────────────────────────────────────────

_EXA_API_BASE = "https://api.exa.ai"
_EXA_API_KEY = os.environ.get("EXA_API_KEY", "")


async def _exa_search_one(
    client: httpx.AsyncClient,
    query: str,
    num_results: int,
    text_max_chars: int,
) -> dict:
    """Execute a single Exa search and return the parsed response."""
    payload = {
        "query": query,
        "numResults": num_results,
        "type": "auto",
        "contents": {
            "text": {"maxCharacters": text_max_chars},
            "highlights": {"query": query},
        },
    }
    try:
        resp = await client.post(
            f"{_EXA_API_BASE}/search",
            json=payload,
            headers={"x-api-key": _EXA_API_KEY, "Content-Type": "application/json"},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        return {"query": query, "count": len(results), "results": results}
    except Exception as exc:
        logger.warning("exa_multi_search query failed: %s — %s", query, exc)
        return {"query": query, "count": 0, "results": [], "error": str(exc)}


async def exa_multi_search(
    queries: List[str],
    num_results_per_query: int = 5,
    text_max_chars: int = 5000,
) -> str:
    """Run multiple Exa searches in parallel and return a single unified result.

    Use this when you need to compare multiple topics simultaneously
    (e.g. "compare these 6 companies") or gather data on several
    entities at once.  All queries run in parallel internally but
    return one combined result to keep the agent loop sequential.

    Args:
        queries: List of search queries to run (max 10).
        num_results_per_query: Number of results per query (default 5, max 8).
        text_max_chars: Max characters of text per result (default 5000).

    Returns:
        JSON object with per-query results and a unified summary.
    """
    if not _EXA_API_KEY:
        return json.dumps({"error": "EXA_API_KEY not set"})

    # Safety bounds — API cost protection, not data truncation
    queries = queries[:10]
    num_results_per_query = min(num_results_per_query, 8)
    text_max_chars = min(text_max_chars, 10_000)

    async with httpx.AsyncClient() as client:
        tasks = [
            _exa_search_one(client, q, num_results_per_query, text_max_chars)
            for q in queries
        ]
        raw_results = await asyncio.gather(*tasks)

    # Build unified compressed output
    all_sources: list = []
    total_results = 0

    for batch in raw_results:
        query = batch["query"]
        total_results += batch["count"]
        for r in batch.get("results", []):
            url = r.get("url", "")
            title = r.get("title", "")
            all_sources.append({"url": url, "title": title, "query": query})

    output = {
        "queries_executed": len(queries),
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
                        "snippet": " ".join(r.get("highlights", []))
                        or r.get("text", ""),
                    }
                    for r in b.get("results", [])
                ],
            }
            for b in raw_results
        ],
        "all_sources": all_sources,
    }

    return json.dumps(output, ensure_ascii=False)


# ── Public FunctionTool instances ───────────────────────────────────

store_finding_tool = FunctionTool(store_finding)
read_findings_tool = FunctionTool(read_findings)
exa_multi_search_tool = FunctionTool(exa_multi_search)

RESEARCH_TOOLS = [store_finding_tool, read_findings_tool, exa_multi_search_tool]
