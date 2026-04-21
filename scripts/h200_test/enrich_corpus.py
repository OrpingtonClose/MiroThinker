#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Corpus enrichment pipeline — gather external web data for the swarm.

Runs enrichment queries across multiple search backends and forum tools,
storing all results as AtomicConditions in the ConditionStore. No YouTube.

Usage:
    python enrich_corpus.py [--db PATH] [--max-per-query N] [--dry-run]

The pipeline uses the existing search tools from strands-agent:
  - DuckDuckGo (uncensored, free, no key)
  - Forum tools (MesoRx, EliteFitness, Professional Muscle, etc.)
  - Jina Reader (URL to markdown extraction)

Optional (if API keys are set):
  - Brave Search
  - Exa (neural search — may censor PED queries)
  - PubMed / Semantic Scholar (academic)

All discovered data lands in the ConditionStore immediately (zero trimming
on intake). A post-enrichment dedup pass groups items and flags duplicates.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_STRANDS_AGENT = str(Path(__file__).resolve().parents[2] / "apps" / "strands-agent")
if _STRANDS_AGENT not in sys.path:
    sys.path.insert(0, _STRANDS_AGENT)

from corpus import AtomicCondition, ConditionStore
from angles import ALL_ANGLES, AngleDefinition
from swarm_log import log_enrichment_result

logger = logging.getLogger(__name__)


# ── Search backends ───────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 10) -> list[dict]:
    """DuckDuckGo search — uncensored, free, no API key."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in results
            ]
    except Exception as exc:
        logger.warning("query=<%s>, error=<%s> | DDG search failed", query, exc)
        return []


def _brave_search(query: str, max_results: int = 10) -> list[dict]:
    """Brave Search — independent index, requires BRAVE_API_KEY."""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return []
    try:
        import httpx
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": api_key},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("description", ""),
            }
            for r in results
        ]
    except Exception as exc:
        logger.warning("query=<%s>, error=<%s> | Brave search failed", query, exc)
        return []


def _pubmed_search(query: str, max_results: int = 5) -> list[dict]:
    """PubMed search — academic papers, free API."""
    try:
        import httpx
        pubmed_query = query
        # Search for PMIDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": pubmed_query,
            "retmax": max_results,
            "retmode": "json",
        }
        api_key = os.environ.get("NCBI_API_KEY", "")
        if api_key:
            params["api_key"] = api_key

        resp = httpx.get(search_url, params=params, timeout=30.0)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Fetch summaries
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "json",
        }
        if api_key:
            params["api_key"] = api_key
        resp = httpx.get(summary_url, params=params, timeout=30.0)
        resp.raise_for_status()
        result_data = resp.json().get("result", {})

        results = []
        for pmid in ids:
            entry = result_data.get(pmid, {})
            if isinstance(entry, dict):
                results.append({
                    "title": entry.get("title", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "snippet": entry.get("title", ""),
                    "source": "pubmed",
                })
        return results
    except Exception as exc:
        logger.warning("query=<%s>, error=<%s> | PubMed search failed", query, exc)
        return []


def _forum_search(query: str, max_results: int = 5) -> list[dict]:
    """Search bodybuilding forums via DuckDuckGo site-scoped queries."""
    forums = [
        "meso-rx.org",
        "elitefitness.com",
        "professionalmuscle.com",
        "thinksteroids.com",
        "evolutionary.org",
    ]
    results = []
    for domain in forums:
        site_query = f"site:{domain} {query}"
        hits = _ddg_search(site_query, max_results=max_results)
        for hit in hits:
            hit["source"] = f"forum:{domain}"
        results.extend(hits)
    return results


def _jina_extract(url: str) -> str:
    """Extract full text from URL via Jina Reader."""
    jina_key = os.environ.get("JINA_API_KEY", "")
    try:
        import httpx
        headers = {"Accept": "text/markdown"}
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"
        resp = httpx.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.text[:50000]  # Cap per-page extraction
    except Exception as exc:
        logger.debug("url=<%s>, error=<%s> | Jina extraction failed", url, exc)
    return ""


# ── Enrichment pipeline ──────────────────────────────────────────────

# ── LLM-based relevance gate ─────────────────────────────────────────
# Instead of a static blacklist, we use a loose LLM check to filter
# results that have zero pharmacological or physiological connection
# to any compound in the research query.  Animal studies (cattle, etc.)
# are explicitly welcome — they contain dose-response and tissue
# composition data that directly informs human use.

_RELEVANCE_PROMPT = """You are a relevance filter for a pharmacology research corpus.
The research topic is: {query_summary}

Below are search results (title + snippet). For EACH result, output Y if it
contains ANY of the following — even indirectly, even in animal models:
- How a compound works (mechanism, receptor binding, pharmacokinetics)
- What it does to tissue (muscle, fat, organ, blood composition)
- Dosing, timing, cycling, or stacking information
- Side effects, health markers, or physiological measurements
- Nutrient interactions, mineral/vitamin effects on compounds

Output N ONLY if the result is completely unrelated (e.g. broken page,
unrelated disease epidemiology with no mechanism, pure economics with
no pharmacology, marketing/spam).

Bias: when in doubt, output Y.  Animal studies are RELEVANT.

Output one letter per line (Y or N), in order. Nothing else.

Results:
{results_block}
"""


async def _llm_relevance_filter(
    results: list[dict],
    query_summary: str,
    api_base: str,
    model: str,
) -> list[dict]:
    """Filter results using a loose LLM relevance check.

    Batches results into a single LLM call. Bias: admit by default.
    Only rejects results that are clearly zero pharmacological connection.
    Falls back to admitting everything if the LLM call fails.
    """
    if not results:
        return results

    # Build the results block
    lines = []
    for i, r in enumerate(results):
        title = r.get("title", "")[:200]
        snippet = r.get("snippet", "")[:300]
        lines.append(f"{i+1}. {title} — {snippet}")
    results_block = "\n".join(lines)

    prompt = _RELEVANCE_PROMPT.format(
        query_summary=query_summary,
        results_block=results_block,
    )

    try:
        import httpx
        url = f"{api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Output Y or N for each result, one per line."},
            ],
            "max_tokens": len(results) * 4,  # ~2 chars per result + newlines
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]

            # Strip think blocks if present
            import re
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Parse Y/N lines
            verdicts = []
            for line in raw.strip().splitlines():
                line = line.strip().upper()
                if line.startswith("Y"):
                    verdicts.append(True)
                elif line.startswith("N"):
                    verdicts.append(False)
                else:
                    verdicts.append(True)  # Bias: admit if unclear

            # If we got fewer verdicts than results, admit the rest
            while len(verdicts) < len(results):
                verdicts.append(True)

            filtered = [r for r, keep in zip(results, verdicts) if keep]
            rejected = len(results) - len(filtered)
            if rejected > 0:
                logger.info(
                    "relevance_filter=<%d/%d rejected> | LLM gate applied",
                    rejected, len(results),
                )
            return filtered

    except Exception as exc:
        logger.warning("error=<%s> | relevance filter failed, admitting all", exc)
        return results  # Fail open — admit everything


def _search_all_backends(
    query: str,
    max_per_backend: int = 10,
) -> list[dict]:
    """Run a query across all available search backends."""
    results = []

    # Tier 1: uncensored
    results.extend(_ddg_search(query, max_per_backend))

    # Tier 1: forums
    results.extend(_forum_search(query, max_results=3))

    # Tier 2: Brave (if key available)
    results.extend(_brave_search(query, max_per_backend))

    # Academic
    results.extend(_pubmed_search(query, max_results=5))

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

    return unique


def enrich_angle(
    angle: AngleDefinition,
    store: ConditionStore,
    max_per_query: int = 10,
    extract_full_text: bool = False,
    dry_run: bool = False,
    llm_filter: bool = False,
    api_base: str = "",
    model: str = "",
) -> int:
    """Run all enrichment queries for a single angle.

    Args:
        angle: The angle definition with label and enrichment queries.
        store: The ConditionStore to admit results into.
        max_per_query: Max results per backend per query.
        extract_full_text: Whether to extract full page text via Jina.
        dry_run: If True, log what would be admitted but don't write.
        llm_filter: If True, run LLM relevance gate on results.
        api_base: vLLM API base URL (required if llm_filter=True).
        model: Model name for relevance gate (required if llm_filter=True).

    Returns the number of conditions admitted to the store.
    """
    admitted = 0

    for query in angle.enrichment_queries:
        logger.info(
            "angle=<%s>, query=<%s> | searching",
            angle.label, query,
        )
        results = _search_all_backends(query, max_per_backend=max_per_query)
        logger.info(
            "angle=<%s>, query=<%s>, results=<%d> | search complete",
            angle.label, query, len(results),
        )

        # LLM relevance gate — angle-aware, very loose
        if llm_filter and results and api_base and model:
            query_summary = (
                f"{angle.label}: {angle.description}"
                if hasattr(angle, "description") and angle.description
                else angle.label
            )
            import asyncio
            results = asyncio.get_event_loop().run_until_complete(
                _llm_relevance_filter(results, query_summary, api_base, model)
            )

        for result in results:
            title = result.get("title", "")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            source = result.get("source", "web")

            if not snippet and not title:
                log_enrichment_result(
                    angle=angle.label, query=query, backend=source,
                    title=title, url=url, snippet=snippet,
                    admitted=False, reject_reason="empty title and snippet",
                )
                continue

            # Build the fact from title + snippet
            fact = f"{title}\n{snippet}" if title and snippet else (title or snippet)

            # Optionally extract full text via Jina
            full_text = ""
            if extract_full_text and url:
                full_text = _jina_extract(url)
                if full_text:
                    fact = f"{title}\n\nSource: {url}\n\n{full_text}"

            if dry_run:
                logger.info(
                    "DRY RUN | would admit: %.100s... (url=%s)", fact, url,
                )
                log_enrichment_result(
                    angle=angle.label, query=query, backend=source,
                    title=title, url=url, snippet=snippet,
                    admitted=True, reject_reason="dry_run",
                )
                admitted += 1
                continue

            cid = store.admit(
                fact=fact,
                source_url=url,
                source_type=f"enrichment:{source}",
                source_ref=query,
                row_type="finding",
                angle=angle.label,
                confidence=0.4,  # Lower confidence for unverified web data
                strategy="web_enrichment",
            )
            was_admitted = cid is not None
            log_enrichment_result(
                angle=angle.label, query=query, backend=source,
                title=title, url=url, snippet=snippet,
                admitted=was_admitted,
                reject_reason="" if was_admitted else "store_dedup",
            )
            if was_admitted:
                admitted += 1

    return admitted


def enrich_all(
    store: ConditionStore,
    max_per_query: int = 10,
    extract_full_text: bool = False,
    dry_run: bool = False,
    llm_filter: bool = False,
    api_base: str = "",
    model: str = "",
) -> dict[str, int]:
    """Run enrichment for all angles.

    Args:
        store: The ConditionStore to admit results into.
        max_per_query: Max results per backend per query.
        extract_full_text: Whether to extract full page text via Jina.
        dry_run: If True, log what would be admitted but don't write.
        llm_filter: If True, run LLM relevance gate on each angle's results.
        api_base: vLLM API base URL (required if llm_filter=True).
        model: Model name for relevance gate (required if llm_filter=True).

    Returns a dict mapping angle labels to number of conditions admitted.
    """
    results: dict[str, int] = {}

    for angle in ALL_ANGLES:
        t0 = time.monotonic()
        count = enrich_angle(
            angle, store,
            max_per_query=max_per_query,
            extract_full_text=extract_full_text,
            dry_run=dry_run,
            llm_filter=llm_filter,
            api_base=api_base,
            model=model,
        )
        elapsed = time.monotonic() - t0
        results[angle.label] = count
        logger.info(
            "angle=<%s>, admitted=<%d>, elapsed_s=<%.1f> | angle enrichment complete",
            angle.label, count, elapsed,
        )

    total = sum(results.values())
    logger.info(
        "total_admitted=<%d>, angles=<%d> | full enrichment complete",
        total, len(results),
    )
    return results


# ── CLI entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich the MiroThinker corpus with external web data",
    )
    parser.add_argument(
        "--db", default="",
        help="DuckDB database path (empty = in-memory)",
    )
    parser.add_argument(
        "--max-per-query", type=int, default=10,
        help="Maximum results per search query per backend",
    )
    parser.add_argument(
        "--extract-full-text", action="store_true",
        help="Extract full page text via Jina Reader (slower, richer)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search but don't write to store — log what would be admitted",
    )
    parser.add_argument(
        "--export", default="",
        help="Export enriched corpus as JSON to this path after enrichment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    store = ConditionStore(db_path=args.db)
    logger.info("db_path=<%s> | store initialized", args.db or "in-memory")

    results = enrich_all(
        store,
        max_per_query=args.max_per_query,
        extract_full_text=args.extract_full_text,
        dry_run=args.dry_run,
    )

    print("\n═══ Enrichment Summary ═══")
    for angle_label, count in results.items():
        print(f"  {angle_label}: {count} conditions")
    print(f"  TOTAL: {sum(results.values())} conditions")

    if args.export and not args.dry_run:
        findings = store.get_findings(limit=100000)
        with open(args.export, "w") as f:
            json.dump(
                [
                    {
                        "id": row["id"],
                        "fact": row["fact"],
                        "source_url": row.get("source_url", ""),
                        "angle": row.get("angle", ""),
                        "confidence": row.get("confidence", 0.5),
                    }
                    for row in findings
                ],
                f, indent=2,
            )
        logger.info("export_path=<%s> | corpus exported", args.export)


if __name__ == "__main__":
    main()
