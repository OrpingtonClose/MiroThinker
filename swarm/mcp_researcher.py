# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MCP Researcher — external data acquisition driven by ConditionStore gaps.

Researcher agents pull concrete data from external MCP APIs based on
what the ConditionStore NEEDS, not random searching.  The gradient flags
tell us exactly what's missing:

    low specificity → search for specific data points, studies, numbers
    high fabrication_risk → verify claims against authoritative sources
    expansion_gap set → fill the specific gap identified by workers
    low trust_score → find higher-quality sources for the same claim

Each research result goes into the ConditionStore as a scored finding,
making it immediately available for Flock evaluation in the next round.

Supported MCP API sources:
    - Brave Search (uncensored web search)
    - Exa (semantic search, academic focus)
    - Tavily (research-focused search)
    - Perplexity (synthesized answers with citations)
    - Semantic Scholar (academic papers)
    - arXiv (preprints)
    - PubMed (biomedical literature)

Architecture:

    ConditionStore (gap analysis via flags)
        → MCP Researcher selects research targets
        → fans out across available APIs
        → extracts concrete findings with sources
        → writes back to ConditionStore as 'mcp_finding' rows
        → Flock scores them in the next evaluation round
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)


def _get_store_lock(store: "ConditionStore") -> Any:
    """Return the store's write lock, supporting both CorpusStore and ConditionStore."""
    return getattr(store, "_write_lock", getattr(store, "_lock", None))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCPResearcherConfig:
    """Configuration for MCP researcher agents.

    Attributes:
        max_targets: Maximum conditions to research per round.
        max_queries_per_target: Maximum search queries per condition.
        max_concurrent: Maximum parallel API calls.
        search_timeout_s: Timeout per API call.
        min_fabrication_risk: Minimum fabrication_risk to trigger verification research.
        min_expansion_priority: Minimum expansion_priority for gap-filling research.
        max_specificity_for_enrichment: Maximum specificity for enrichment research.
    """

    max_targets: int = 30
    max_queries_per_target: int = 3
    max_concurrent: int = 6
    search_timeout_s: float = 20.0
    min_fabrication_risk: float = 0.5
    min_expansion_priority: float = 0.3
    max_specificity_for_enrichment: float = 0.4


@dataclass
class ResearchTarget:
    """A condition identified as needing external data.

    Attributes:
        condition_id: The ConditionStore row to research.
        fact: The claim text.
        angle: The research angle.
        reason: Why this condition needs research.
        search_queries: Generated search queries.
        priority: How urgently this needs research.
    """

    condition_id: int
    fact: str
    angle: str
    reason: str
    search_queries: list[str] = field(default_factory=list)
    priority: float = 0.5


@dataclass
class ResearchResult:
    """Result from an MCP API research call.

    Attributes:
        source_api: Which API returned this result.
        fact: The concrete finding extracted.
        source_url: URL of the source.
        confidence: Estimated confidence of this finding.
        target_condition_id: Which condition this was researched for.
        raw_snippet: The raw text from the API.
    """

    source_api: str
    fact: str
    source_url: str
    confidence: float = 0.5
    target_condition_id: int = 0
    raw_snippet: str = ""


@dataclass
class MCPResearchRoundMetrics:
    """Metrics from one research round.

    Attributes:
        targets_researched: How many conditions were researched.
        api_calls_made: Total API calls across all sources.
        findings_stored: New findings written to ConditionStore.
        apis_used: Which APIs were called.
        elapsed_s: Wall-clock time.
    """

    targets_researched: int = 0
    api_calls_made: int = 0
    findings_stored: int = 0
    apis_used: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Target selection — which conditions need external data
# ---------------------------------------------------------------------------

def select_research_targets(
    store: "ConditionStore",
    config: MCPResearcherConfig,
    complete: Callable[[str], Awaitable[str]] | None = None,
) -> list[ResearchTarget]:
    """Select conditions that would benefit most from external data.

    Uses gradient flags to identify the highest-value research targets:
    1. High fabrication_risk → need authoritative verification
    2. Unfulfilled expansion gaps → workers explicitly asked for data
    3. Low specificity + high relevance → need concrete enrichment
    4. Low trust_score + high actionability → need better sources

    Args:
        store: The ConditionStore.
        config: Researcher configuration.
        complete: Optional LLM function for query generation.

    Returns:
        Sorted list of ResearchTargets, highest priority first.
    """
    targets: list[ResearchTarget] = []
    max_per_type = config.max_targets // 4
    lock = _get_store_lock(store)

    # --- Type 1: High fabrication risk → verify against authoritative sources ---
    try:
        with lock:
            fab_rows = store.conn.execute(
                "SELECT id, fact, angle, fabrication_risk "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND fabrication_risk > ? "
                "AND score_version > 0 "
                "ORDER BY fabrication_risk DESC "
                "LIMIT ?",
                [config.min_fabrication_risk, max_per_type],
            ).fetchall()
        for cid, fact, angle, fab_risk in fab_rows:
            targets.append(ResearchTarget(
                condition_id=cid,
                fact=fact,
                angle=angle,
                reason="high_fabrication_risk",
                search_queries=_generate_verification_queries(fact),
                priority=fab_risk,
            ))
    except Exception as exc:
        logger.warning("error=<%s> | fabrication risk target selection failed", exc)

    # --- Type 2: Unfulfilled expansion gaps ---
    try:
        with lock:
            gap_rows = store.conn.execute(
                "SELECT id, fact, angle, expansion_gap, expansion_priority "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND expansion_gap != '' "
                "AND expansion_fulfilled = FALSE "
                "AND expansion_priority > ? "
                "ORDER BY expansion_priority DESC "
                "LIMIT ?",
                [config.min_expansion_priority, max_per_type],
            ).fetchall()
        for cid, fact, angle, gap, priority in gap_rows:
            targets.append(ResearchTarget(
                condition_id=cid,
                fact=f"{fact} [GAP: {gap}]",
                angle=angle,
                reason="expansion_gap",
                search_queries=_generate_gap_queries(fact, gap),
                priority=priority,
            ))
    except Exception as exc:
        logger.warning("error=<%s> | expansion gap target selection failed", exc)

    # --- Type 3: Low specificity + high relevance ---
    try:
        with lock:
            spec_rows = store.conn.execute(
                "SELECT id, fact, angle, specificity_score, relevance_score "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND specificity_score < ? "
                "AND relevance_score > 0.5 "
                "AND score_version > 0 "
                "ORDER BY (relevance_score - specificity_score) DESC "
                "LIMIT ?",
                [config.max_specificity_for_enrichment, max_per_type],
            ).fetchall()
        for cid, fact, angle, spec, rel in spec_rows:
            targets.append(ResearchTarget(
                condition_id=cid,
                fact=fact,
                angle=angle,
                reason="low_specificity",
                search_queries=_generate_enrichment_queries(fact, angle),
                priority=(rel - spec) * 0.8,
            ))
    except Exception as exc:
        logger.warning("error=<%s> | specificity target selection failed", exc)

    # --- Type 4: Low trust + high actionability ---
    try:
        with lock:
            trust_rows = store.conn.execute(
                "SELECT id, fact, angle, trust_score, actionability_score "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND trust_score < 0.4 "
                "AND actionability_score > 0.6 "
                "AND score_version > 0 "
                "ORDER BY (actionability_score - trust_score) DESC "
                "LIMIT ?",
                [max_per_type],
            ).fetchall()
        for cid, fact, angle, trust, action in trust_rows:
            targets.append(ResearchTarget(
                condition_id=cid,
                fact=fact,
                angle=angle,
                reason="low_trust_high_action",
                search_queries=_generate_source_upgrade_queries(fact, angle),
                priority=(action - trust) * 0.7,
            ))
    except Exception as exc:
        logger.warning("error=<%s> | trust/action target selection failed", exc)

    # Sort by priority and cap
    targets.sort(key=lambda t: t.priority, reverse=True)
    return targets[:config.max_targets]


# ---------------------------------------------------------------------------
# Query generation — turn conditions into search queries
# ---------------------------------------------------------------------------

def _generate_verification_queries(fact: str) -> list[str]:
    """Generate search queries to verify a potentially fabricated claim."""
    # Extract key terms for focused searching
    words = fact.split()
    key_phrase = " ".join(words[:15])  # first 15 words as core query
    return [
        f'"{key_phrase}" site:pubmed.ncbi.nlm.nih.gov OR site:scholar.google.com',
        f"{key_phrase} systematic review meta-analysis",
        f"{key_phrase} evidence clinical trial",
    ]


def _generate_gap_queries(fact: str, gap: str) -> list[str]:
    """Generate search queries to fill a specific research gap."""
    return [
        gap,
        f"{gap} research evidence",
        f"{gap} mechanism data",
    ]


def _generate_enrichment_queries(fact: str, angle: str) -> list[str]:
    """Generate search queries to enrich a vague claim with specifics."""
    words = fact.split()
    key_phrase = " ".join(words[:12])
    return [
        f"{key_phrase} dosage protocol study",
        f"{key_phrase} specific mechanism pathway",
        f"{angle} {key_phrase} quantitative data",
    ]


def _generate_source_upgrade_queries(fact: str, angle: str) -> list[str]:
    """Generate queries to find more authoritative sources for a claim."""
    words = fact.split()
    key_phrase = " ".join(words[:12])
    return [
        f"{key_phrase} peer-reviewed journal",
        f"{key_phrase} site:ncbi.nlm.nih.gov",
        f"{key_phrase} clinical guidelines recommendation",
    ]


# ---------------------------------------------------------------------------
# API execution — fan out across available MCP sources
# ---------------------------------------------------------------------------

async def _search_brave(query: str, timeout: float = 20.0) -> list[dict[str, str]]:
    """Search via Brave Search API."""
    import httpx

    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key},
                params={"q": query, "count": 5},
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("web", {}).get("results", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })
            return results
    except Exception as exc:
        logger.debug("query=<%s>, error=<%s> | brave search failed", query[:50], exc)
        return []


async def _search_exa(query: str, timeout: float = 20.0) -> list[dict[str, str]]:
    """Search via Exa semantic search API."""
    import httpx

    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": api_key},
                json={
                    "query": query,
                    "numResults": 5,
                    "useAutoprompt": True,
                    "type": "auto",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("text", "")[:500],
                })
            return results
    except Exception as exc:
        logger.debug("query=<%s>, error=<%s> | exa search failed", query[:50], exc)
        return []


async def _search_tavily(query: str, timeout: float = 20.0) -> list[dict[str, str]]:
    """Search via Tavily research API."""
    import httpx

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": 5,
                    "search_depth": "advanced",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[:500],
                })
            return results
    except Exception as exc:
        logger.debug("query=<%s>, error=<%s> | tavily search failed", query[:50], exc)
        return []


async def _search_semantic_scholar(query: str, timeout: float = 20.0) -> list[dict[str, str]]:
    """Search via Semantic Scholar API for academic papers."""
    import httpx

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params={
                    "query": query,
                    "limit": 5,
                    "fields": "title,url,abstract,year,citationCount",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("data", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", "") or f"https://api.semanticscholar.org/paper/{item.get('paperId', '')}",
                    "snippet": (item.get("abstract", "") or "")[:500],
                })
            return results
    except Exception as exc:
        logger.debug("query=<%s>, error=<%s> | semantic scholar search failed", query[:50], exc)
        return []


async def execute_research(
    targets: list[ResearchTarget],
    config: MCPResearcherConfig,
) -> tuple[list[ResearchResult], int]:
    """Execute research across all available MCP APIs.

    Fans out each target's search queries across available APIs in
    parallel, respecting concurrency limits.

    Args:
        targets: Research targets with pre-generated queries.
        config: Researcher configuration.

    Returns:
        Tuple of (results, api_calls_made) where api_calls_made is the
        total number of API calls dispatched (including failures).
    """
    search_fns = {
        "brave": _search_brave,
        "exa": _search_exa,
        "tavily": _search_tavily,
        "semantic_scholar": _search_semantic_scholar,
    }

    # Filter to available APIs (have keys configured)
    available: dict[str, Any] = {}
    if os.environ.get("BRAVE_API_KEY"):
        available["brave"] = _search_brave
    if os.environ.get("EXA_API_KEY"):
        available["exa"] = _search_exa
    if os.environ.get("TAVILY_API_KEY"):
        available["tavily"] = _search_tavily
    # Semantic Scholar works without API key (rate-limited)
    available["semantic_scholar"] = _search_semantic_scholar

    if not available:
        logger.warning("no search APIs available for MCP researcher")
        return [], 0

    semaphore = asyncio.Semaphore(config.max_concurrent)
    all_results: list[ResearchResult] = []

    async def _search_with_limit(
        api_name: str,
        search_fn: Any,
        query: str,
        target: ResearchTarget,
    ) -> list[ResearchResult]:
        async with semaphore:
            raw_results = await search_fn(query, config.search_timeout_s)
            findings: list[ResearchResult] = []
            for item in raw_results:
                snippet = item.get("snippet", "").strip()
                if not snippet or len(snippet) < 30:
                    continue
                findings.append(ResearchResult(
                    source_api=api_name,
                    fact=snippet,
                    source_url=item.get("url", ""),
                    confidence=0.5,  # will be scored by Flock
                    target_condition_id=target.condition_id,
                    raw_snippet=snippet,
                ))
            return findings

    # Fan out: each target × each query × each API
    tasks = []
    for target in targets:
        for query in target.search_queries[:config.max_queries_per_target]:
            for api_name, search_fn in available.items():
                tasks.append(
                    _search_with_limit(api_name, search_fn, query, target)
                )

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_results.extend(r)
        elif isinstance(r, Exception):
            logger.debug("error=<%s> | research task failed", r)

    return all_results, len(tasks)


# ---------------------------------------------------------------------------
# Store integration — write research results as mcp_finding rows
# ---------------------------------------------------------------------------

def store_research_results(
    store: "ConditionStore",
    results: list[ResearchResult],
    run_id: str,
) -> int:
    """Write research results into the ConditionStore.

    Each result becomes an 'mcp_finding' row linked to the condition
    it was researched for.  Flock will score these in the next round.

    Args:
        store: The ConditionStore.
        results: Research results to store.
        run_id: Current run identifier.

    Returns:
        Number of findings stored.
    """
    stored = 0
    for result in results:
        fact = result.fact.strip()
        if not fact or len(fact) < 30:
            continue

        with _get_store_lock(store):
            cid = store._next_id
            store._next_id += 1
            store.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_url, source_type, source_ref, row_type,
                    consider_for_use, confidence,
                    created_at, parent_id, phase)
                   VALUES (?, ?, ?, 'mcp_research', ?, 'mcp_finding', TRUE,
                           ?, ?, ?, 'mcp_research')""",
                [
                    cid,
                    fact,
                    result.source_url,
                    f"{result.source_api}_{run_id}",
                    result.confidence,
                    datetime.now(timezone.utc).isoformat(),
                    result.target_condition_id,
                ],
            )
            stored += 1

    logger.info(
        "results_received=<%d>, stored=<%d> | MCP research results stored",
        len(results), stored,
    )
    return stored


# ---------------------------------------------------------------------------
# Main entry point — run a research round
# ---------------------------------------------------------------------------

async def run_mcp_research_round(
    store: "ConditionStore",
    run_id: str,
    config: MCPResearcherConfig | None = None,
    complete: Callable[[str], Awaitable[str]] | None = None,
) -> MCPResearchRoundMetrics:
    """Run one round of MCP-powered external data acquisition.

    1. Select conditions that need external data (flag-driven)
    2. Generate search queries per target
    3. Fan out across available MCP APIs
    4. Store results as mcp_finding rows
    5. Return metrics

    Args:
        store: The ConditionStore.
        run_id: Current run identifier.
        config: Researcher configuration.
        complete: Optional LLM function for query generation.

    Returns:
        MCPResearchRoundMetrics with execution stats.
    """
    config = config or MCPResearcherConfig()
    t0 = time.monotonic()

    # Bootstrap: promote unscored findings so score_version > 0 filters
    # in select_research_targets can match.  Same chicken-and-egg issue
    # as FlockQueryManager — score_version starts at 0 and is only
    # incremented by _apply_score_delta which requires queries first.
    try:
        lock = _get_store_lock(store)
        with lock:
            bootstrapped = store.conn.execute(
                "UPDATE conditions "
                "SET score_version = 1 "
                "WHERE score_version = 0 "
                "AND row_type = 'finding' "
                "AND consider_for_use = TRUE"
            ).rowcount
        if bootstrapped:
            logger.info(
                "bootstrapped=<%d> | promoted unscored findings to score_version=1",
                bootstrapped,
            )
    except Exception as exc:
        logger.warning("error=<%s> | bootstrap scoring failed", exc)

    # 1. Select targets
    targets = select_research_targets(store, config, complete)
    if not targets:
        logger.info("no research targets found — ConditionStore well-populated")
        return MCPResearchRoundMetrics(elapsed_s=time.monotonic() - t0)

    logger.info(
        "targets=<%d>, reasons=<%s> | research targets selected",
        len(targets),
        [t.reason for t in targets[:10]],
    )

    # 2. Execute research
    results, total_api_calls = await execute_research(targets, config)

    # 3. Store results
    stored = store_research_results(store, results, run_id)

    elapsed = time.monotonic() - t0

    # Collect which APIs returned successful results
    apis_used = list({r.source_api for r in results})

    metrics = MCPResearchRoundMetrics(
        targets_researched=len(targets),
        api_calls_made=total_api_calls,
        findings_stored=stored,
        apis_used=apis_used,
        elapsed_s=elapsed,
    )

    logger.info(
        "targets=<%d>, api_calls=<%d>, stored=<%d>, apis=<%s>, "
        "elapsed_s=<%.1f> | MCP research round complete",
        metrics.targets_researched, metrics.api_calls_made,
        metrics.findings_stored, metrics.apis_used, metrics.elapsed_s,
    )

    return metrics
