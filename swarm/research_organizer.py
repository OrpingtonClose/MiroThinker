# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Research Organizer — between-wave external data acquisition via clones.

After each wave, tool-free workers produce transcripts containing
accumulated expertise and — critically — doubts, uncertainties, and
"I wish I could verify..." moments.  The Research Organizer:

1. Reads ALL worker transcripts from the completed wave
2. Extracts specific research needs (doubts, unverified claims, gaps)
3. Plans strategically which gaps to fill (not random searching)
4. Spawns tool-armed clones of specific workers to resolve gaps

Each clone carries the worker's full conversation context (system
prompt + data package + reasoning output) but is additionally equipped
with search tools.  The clone searches with domain expertise — it knows
what "Milos protocol" means, what dosage ranges are plausible, what to
look for.  A generic searcher doesn't.

Clone findings flow back into the ConditionStore as
``source_type='clone_research'`` and appear in the next wave's data
packages as §8 FRESH EVIDENCE.

Architecture reference: docs/SWARM_WAVE_ARCHITECTURE.md
    § "Unleashed Clones: Tool-Armed Expert Researchers"

Context isolation (invariant #5):
    The clone is a SEPARATE agent instance.  It shares the worker's
    conversation as initialization but runs independently.  Its search
    results and intermediate reasoning do NOT flow into the worker's
    context.  Only final findings (stored in ConditionStore) reach the
    worker, mediated by the data package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from swarm.corpus_builder import (
    CorpusBuilderConfig,
    _get_available_search_fns,
    _search_arxiv,
    _search_semantic_scholar,
    extract_url_content,
)

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ResearchNeed:
    """A specific doubt or gap extracted from a worker's transcript.

    Attributes:
        angle: The worker's research angle.
        doubt: The specific uncertainty or gap.
        data_needed: What data would resolve this doubt.
        priority: Estimated information gain (higher = more impactful).
        source_worker_id: The worker that expressed this doubt.
    """

    angle: str
    doubt: str
    data_needed: str
    priority: float = 0.5
    source_worker_id: str = ""


@dataclass
class CloneResearchResult:
    """Result from a tool-armed clone's research.

    Attributes:
        angle: The research angle this clone investigated.
        doubt: The original doubt being resolved.
        findings: List of findings with evidence.
        search_queries_used: Queries the clone executed.
        sources_found: URLs and references discovered.
        worker_id: The clone's worker identifier.
    """

    angle: str
    doubt: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    search_queries_used: list[str] = field(default_factory=list)
    sources_found: list[str] = field(default_factory=list)
    worker_id: str = ""


@dataclass
class ResearchOrganizerConfig:
    """Configuration for the Research Organizer.

    Attributes:
        max_doubts_per_worker: Maximum doubts to extract per worker.
        max_clones: Maximum simultaneous clones to spawn.
        max_searches_per_clone: Budget: max search calls per clone.
        max_extractions_per_clone: Budget: max content extractions
            per clone.
        clone_timeout_s: Timeout per clone research session.
        trigger_uncertainty_threshold: Minimum uncertainty signals in
            a transcript to trigger clone research.
        trigger_every_n_waves: Spawn clones every N waves regardless.
    """

    max_doubts_per_worker: int = 3
    max_clones: int = 4
    max_searches_per_clone: int = 5
    max_extractions_per_clone: int = 2
    clone_timeout_s: float = 120.0
    trigger_uncertainty_threshold: int = 3
    trigger_every_n_waves: int = 2


# ---------------------------------------------------------------------------
# Doubt extraction
# ---------------------------------------------------------------------------

_DOUBT_EXTRACTION_PROMPT = """\
Review this worker's reasoning transcript and list every uncertainty, \
doubt, unverified claim, and "I wish I knew..." moment.

For each doubt, state:
1. The specific uncertainty
2. Exactly what data would resolve it (be specific — not "more research \
needed" but "bloodwork data showing hematocrit at 300mg/week trenbolone \
enanthate over 8+ weeks")
3. Priority (HIGH / MEDIUM / LOW) based on how much resolving this doubt \
would improve the overall analysis

Worker angle: {angle}
Worker transcript:
{transcript}

Output ONLY valid JSON — an array of objects:
[
  {{
    "doubt": "the specific uncertainty",
    "data_needed": "exactly what data would resolve it",
    "priority": "HIGH|MEDIUM|LOW"
  }}
]"""


_UNCERTAINTY_SIGNALS = [
    "uncertain", "unclear", "unknown", "unverified", "insufficient",
    "need data", "need evidence", "need research", "need to verify",
    "I wish", "would need", "lacking data", "no evidence",
    "insufficient evidence", "conflicting", "contradictory",
    "might be", "possibly", "perhaps", "presumably",
    "not enough", "limited data", "sparse", "anecdotal",
    "requires further", "remains to be seen",
]


def count_uncertainty_signals(transcript: str) -> int:
    """Count uncertainty signals in a worker transcript.

    Args:
        transcript: The worker's reasoning output text.

    Returns:
        Count of uncertainty signal phrases found.
    """
    lower = transcript.lower()
    return sum(1 for signal in _UNCERTAINTY_SIGNALS if signal in lower)


async def extract_doubts(
    transcript: str,
    angle: str,
    worker_id: str,
    complete: Callable[[str], Awaitable[str]],
    *,
    max_doubts: int = 3,
) -> list[ResearchNeed]:
    """Extract specific research needs from a worker's transcript.

    Uses LLM to identify doubts, uncertainties, and gaps in the
    worker's reasoning.  Falls back to heuristic extraction if LLM
    call fails.

    Args:
        transcript: The worker's reasoning output text.
        angle: The worker's research angle.
        worker_id: The worker's identifier.
        complete: Async LLM completion function.
        max_doubts: Maximum doubts to extract.

    Returns:
        List of ResearchNeed objects, sorted by priority.
    """
    prompt = _DOUBT_EXTRACTION_PROMPT.replace("{angle}", angle)
    prompt = prompt.replace("{transcript}", transcript[:15000])

    try:
        content = await complete(prompt)
    except Exception as exc:
        logger.warning(
            "angle=<%s>, error=<%s> | doubt extraction LLM call failed",
            angle, exc,
        )
        return _extract_doubts_heuristic(transcript, angle, worker_id, max_doubts)

    if not content:
        return _extract_doubts_heuristic(transcript, angle, worker_id, max_doubts)

    # Strip markdown fences
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        items = json.loads(content)
        if not isinstance(items, list):
            return _extract_doubts_heuristic(transcript, angle, worker_id, max_doubts)
    except json.JSONDecodeError:
        return _extract_doubts_heuristic(transcript, angle, worker_id, max_doubts)

    priority_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    needs: list[ResearchNeed] = []
    for item in items[:max_doubts]:
        needs.append(ResearchNeed(
            angle=angle,
            doubt=item.get("doubt", ""),
            data_needed=item.get("data_needed", ""),
            priority=priority_map.get(
                item.get("priority", "MEDIUM").upper(), 0.5,
            ),
            source_worker_id=worker_id,
        ))

    needs.sort(key=lambda n: n.priority, reverse=True)
    return needs[:max_doubts]


def _extract_doubts_heuristic(
    transcript: str,
    angle: str,
    worker_id: str,
    max_doubts: int,
) -> list[ResearchNeed]:
    """Heuristic fallback for doubt extraction.

    Scans for uncertainty signal phrases and extracts surrounding
    context as research needs.

    Args:
        transcript: The worker's reasoning output text.
        angle: The worker's research angle.
        worker_id: The worker's identifier.
        max_doubts: Maximum doubts to extract.

    Returns:
        List of ResearchNeed objects.
    """
    needs: list[ResearchNeed] = []
    sentences = re.split(r"[.!?]+", transcript)

    for sentence in sentences:
        if len(needs) >= max_doubts:
            break
        lower = sentence.lower().strip()
        if not lower or len(lower) < 20:
            continue
        signal_count = sum(1 for s in _UNCERTAINTY_SIGNALS if s in lower)
        if signal_count > 0:
            needs.append(ResearchNeed(
                angle=angle,
                doubt=sentence.strip(),
                data_needed=f"Research: {sentence.strip()[:200]}",
                priority=min(0.3 + signal_count * 0.2, 0.9),
                source_worker_id=worker_id,
            ))

    return needs[:max_doubts]


# ---------------------------------------------------------------------------
# Clone research — tool-armed expert search
# ---------------------------------------------------------------------------

_CLONE_SEARCH_PROMPT = """\
You are a {angle} specialist. You have been reasoning deeply about \
{angle} and have specific doubts and uncertainties.

Your original analysis identified this gap:
DOUBT: {doubt}
DATA NEEDED: {data_needed}

Generate 3-5 specific, expert-informed search queries that would find \
the exact data needed to resolve this doubt. Be precise — use domain \
terminology, specific compound names, dosage ranges, and mechanism \
keywords that a specialist would use.

Output ONLY valid JSON — an array of strings:
["query 1", "query 2", "query 3"]"""


async def _generate_clone_queries(
    need: ResearchNeed,
    complete: Callable[[str], Awaitable[str]],
) -> list[str]:
    """Generate expert-informed search queries for a research need.

    The clone uses its domain understanding to formulate precise queries
    that a generic searcher wouldn't know to ask.

    Args:
        need: The research need to generate queries for.
        complete: Async LLM completion function.

    Returns:
        List of search query strings.
    """
    prompt = _CLONE_SEARCH_PROMPT.replace("{angle}", need.angle)
    prompt = prompt.replace("{doubt}", need.doubt)
    prompt = prompt.replace("{data_needed}", need.data_needed)

    try:
        content = await complete(prompt)
    except Exception as exc:
        logger.warning(
            "angle=<%s>, error=<%s> | clone query generation failed",
            need.angle, exc,
        )
        return [need.data_needed, f"{need.angle} {need.doubt[:100]}"]

    if not content:
        return [need.data_needed, f"{need.angle} {need.doubt[:100]}"]

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        queries = json.loads(content)
        if isinstance(queries, list):
            return [q for q in queries if isinstance(q, str)][:5]
    except json.JSONDecodeError:
        pass

    return [need.data_needed, f"{need.angle} {need.doubt[:100]}"]


async def run_clone_research(
    need: ResearchNeed,
    complete: Callable[[str], Awaitable[str]],
    config: ResearchOrganizerConfig,
) -> CloneResearchResult:
    """Run a single tool-armed clone to resolve a specific doubt.

    The clone generates expert-informed search queries, fans out across
    available APIs, extracts content from top URLs, and returns
    structured findings.

    Args:
        need: The research need to investigate.
        complete: Async LLM completion function.
        config: Research organizer configuration.

    Returns:
        CloneResearchResult with findings and sources.
    """
    result = CloneResearchResult(
        angle=need.angle,
        doubt=need.doubt,
        worker_id=f"clone_{need.source_worker_id}",
    )

    # Step 1: Generate expert search queries
    queries = await _generate_clone_queries(need, complete)
    result.search_queries_used = queries

    logger.info(
        "angle=<%s>, doubt=<%s>, queries=<%d> | clone generating search queries",
        need.angle, need.doubt[:60], len(queries),
    )

    # Step 2: Fan-out search across available APIs
    search_fns = _get_available_search_fns()
    all_search_results: list[dict[str, str]] = []
    searches_done = 0

    for q in queries:
        if searches_done >= config.max_searches_per_clone:
            break
        fn = search_fns[searches_done % len(search_fns)]
        try:
            results = await fn(q)
            all_search_results.extend(results)
            searches_done += 1
        except Exception as exc:
            logger.warning(
                "angle=<%s>, query=<%s>, error=<%s> | clone search failed",
                need.angle, q[:50], exc,
            )

    # Step 3: Extract content from top URLs
    urls_seen: set[str] = set()
    top_urls: list[str] = []
    for r in all_search_results:
        url = r.get("url", "")
        if url and url not in urls_seen:
            urls_seen.add(url)
            top_urls.append(url)
            if len(top_urls) >= config.max_extractions_per_clone:
                break

    extracted_texts: list[str] = []
    for url in top_urls:
        content = await extract_url_content(url)
        if content:
            extracted_texts.append(content)
            result.sources_found.append(url)

    # Step 4: Synthesize findings from gathered material
    all_material = "\n\n".join(extracted_texts)
    snippet_material = "\n".join(
        f"- {r.get('title', '')}: {r.get('snippet', '')}"
        for r in all_search_results
        if r.get("snippet")
    )

    if all_material or snippet_material:
        findings = await _synthesize_clone_findings(
            need=need,
            full_articles=all_material,
            snippets=snippet_material,
            complete=complete,
        )
        result.findings = findings

    logger.info(
        "angle=<%s>, doubt=<%s>, findings=<%d>, sources=<%d> | "
        "clone research complete",
        need.angle, need.doubt[:60], len(result.findings),
        len(result.sources_found),
    )

    return result


# ---------------------------------------------------------------------------
# Finding synthesis from clone research
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a {angle} specialist investigating a specific doubt.

DOUBT: {doubt}
DATA NEEDED: {data_needed}

You searched and found the following material:

FULL ARTICLES:
{articles}

SEARCH SNIPPETS:
{snippets}

Extract specific findings that resolve (or partially resolve) the doubt. \
For each finding:
1. State the factual claim with exact numbers/dosages/mechanisms
2. Cite the source (URL or article title)
3. Rate confidence: HIGH (direct evidence) / MEDIUM (indirect/inferred) / LOW (anecdotal)

Output ONLY valid JSON — an array of objects:
[
  {{
    "fact": "specific factual claim with exact numbers",
    "source": "URL or article title",
    "confidence": 0.8,
    "resolves_doubt": true/false
  }}
]"""


async def _synthesize_clone_findings(
    need: ResearchNeed,
    full_articles: str,
    snippets: str,
    complete: Callable[[str], Awaitable[str]],
) -> list[dict[str, Any]]:
    """Synthesize structured findings from clone's gathered material.

    Args:
        need: The research need being investigated.
        full_articles: Full article texts from content extraction.
        snippets: Search result snippets.
        complete: Async LLM completion function.

    Returns:
        List of finding dicts with fact, source, confidence.
    """
    prompt = _SYNTHESIS_PROMPT.replace("{angle}", need.angle)
    prompt = prompt.replace("{doubt}", need.doubt)
    prompt = prompt.replace("{data_needed}", need.data_needed)
    prompt = prompt.replace("{articles}", full_articles[:20000] if full_articles else "(none)")
    prompt = prompt.replace("{snippets}", snippets[:5000] if snippets else "(none)")

    try:
        content = await complete(prompt)
    except Exception as exc:
        logger.warning(
            "angle=<%s>, error=<%s> | clone synthesis failed",
            need.angle, exc,
        )
        return []

    if not content:
        return []

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        items = json.loads(content)
        if not isinstance(items, list):
            return []
        findings = []
        for item in items:
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            findings.append({
                "fact": item.get("fact", ""),
                "source": item.get("source", ""),
                "confidence": conf,
                "resolves_doubt": bool(item.get("resolves_doubt", False)),
            })
        return findings
    except json.JSONDecodeError:
        return []


# ---------------------------------------------------------------------------
# Store integration — persist clone findings
# ---------------------------------------------------------------------------

def store_clone_findings(
    store: "ConditionStore",
    clone_result: CloneResearchResult,
    wave: int,
    run_id: str,
) -> int:
    """Store clone research findings in the ConditionStore.

    Findings are stored with ``source_type='clone_research'`` so the
    data package builder can include them as §8 FRESH EVIDENCE.

    Args:
        store: The shared ConditionStore.
        clone_result: Results from a clone's research.
        wave: Current wave number.
        run_id: Run identifier for provenance.

    Returns:
        Number of findings stored.
    """
    stored = 0
    for finding in clone_result.findings:
        fact = finding.get("fact", "")
        if not fact:
            continue
        try:
            confidence = float(finding.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        source_ref = finding.get("source", "clone_research")
        store.admit(
            fact=fact,
            angle=clone_result.angle,
            confidence=confidence,
            source_type="clone_research",
            source_ref=source_ref,
            source_model=f"clone_{clone_result.angle}",
            source_run=run_id,
            iteration=wave,
            strategy=clone_result.doubt,
        )
        stored += 1

    if stored:
        logger.info(
            "angle=<%s>, doubt=<%s>, stored=<%d> | clone findings persisted",
            clone_result.angle, clone_result.doubt[:60], stored,
        )

    return stored


# ---------------------------------------------------------------------------
# Fresh evidence section builder (for data package §8)
# ---------------------------------------------------------------------------

def build_fresh_evidence_section(
    store: "ConditionStore",
    angle: str,
    wave: int,
) -> str:
    """Build the §8 FRESH EVIDENCE section from clone research findings.

    Retrieves clone_research findings for this angle from the previous
    wave and formats them as doubt-resolution pairs.

    Args:
        store: The shared ConditionStore.
        angle: The worker's research angle.
        wave: Current wave number.

    Returns:
        Formatted fresh evidence string, or empty string if no
        clone research exists.
    """
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact, confidence, strategy as doubt, source_ref
                   FROM conditions
                   WHERE source_type = 'clone_research'
                     AND angle = ?
                     AND iteration = ?
                   ORDER BY confidence DESC
                   LIMIT 10""",
                [angle, wave - 1],
            ).fetchall()
    except Exception as exc:
        logger.warning(
            "angle=<%s>, wave=<%d>, error=<%s> | "
            "failed to retrieve clone research findings",
            angle, wave, exc,
        )
        return ""

    if not rows:
        return ""

    lines = [
        "Your clone-researcher investigated specific doubts from your "
        "previous analysis. Here's what it found:\n"
    ]

    for row in rows:
        fact = row[0]
        confidence = row[1]
        doubt = row[2] or "unspecified doubt"
        source = row[3] or "clone research"
        conf_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
        lines.append(f"DOUBT: {doubt}")
        lines.append(f"EVIDENCE ({conf_label} confidence): {fact}")
        lines.append(f"SOURCE: {source}")
        lines.append("")

    lines.append(
        "Integrate this evidence into your analysis. Where it confirms "
        "your prior reasoning, strengthen your claims. Where it "
        "contradicts, revise."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point — the Research Organizer
# ---------------------------------------------------------------------------

async def run_research_organizer(
    store: "ConditionStore",
    worker_results: list[dict[str, Any]],
    wave: int,
    run_id: str,
    complete: Callable[[str], Awaitable[str]],
    config: ResearchOrganizerConfig | None = None,
) -> list[CloneResearchResult]:
    """Run the Research Organizer after a wave completes.

    Reads all worker transcripts, extracts research needs, decides
    which workers need clone assistance, and spawns tool-armed clones
    to resolve gaps.

    The Research Organizer is the strategic brain — it doesn't search
    randomly.  It reads what workers are uncertain about and dispatches
    clones that carry the worker's domain expertise to find exactly
    the right data.

    Args:
        store: The shared ConditionStore.
        worker_results: List of worker result dicts from the wave.
        wave: Current wave number.
        run_id: Run identifier for provenance.
        complete: Async LLM completion function.
        config: Optional configuration overrides.

    Returns:
        List of CloneResearchResult objects.
    """
    config = config or ResearchOrganizerConfig()

    # ── Step 1: Decide whether to trigger clone research ─────────
    should_trigger = (wave % config.trigger_every_n_waves == 0)

    if not should_trigger:
        # Check uncertainty levels across all workers
        high_uncertainty_workers = []
        for wr in worker_results:
            transcript = wr.get("response", "")
            if not transcript:
                continue
            signals = count_uncertainty_signals(transcript)
            if signals >= config.trigger_uncertainty_threshold:
                high_uncertainty_workers.append(wr)

        if high_uncertainty_workers:
            should_trigger = True
            logger.info(
                "wave=<%d>, uncertain_workers=<%d> | "
                "uncertainty threshold triggered clone research",
                wave, len(high_uncertainty_workers),
            )

    if not should_trigger:
        logger.info(
            "wave=<%d> | research organizer skipping — no trigger conditions met",
            wave,
        )
        return []

    # ── Step 2: Extract doubts from all worker transcripts ───────
    all_needs: list[ResearchNeed] = []
    extraction_tasks = []

    for wr in worker_results:
        transcript = wr.get("response", "")
        if not transcript or wr.get("status") != "success":
            continue
        extraction_tasks.append(
            extract_doubts(
                transcript=transcript,
                angle=wr.get("angle", ""),
                worker_id=wr.get("worker_id", ""),
                complete=complete,
                max_doubts=config.max_doubts_per_worker,
            )
        )

    if extraction_tasks:
        needs_lists = await asyncio.gather(*extraction_tasks)
        for needs in needs_lists:
            all_needs.extend(needs)

    if not all_needs:
        logger.info(
            "wave=<%d> | no research needs extracted from worker transcripts",
            wave,
        )
        return []

    # Sort by priority and take top N
    all_needs.sort(key=lambda n: n.priority, reverse=True)
    selected_needs = all_needs[:config.max_clones]

    logger.info(
        "wave=<%d>, total_doubts=<%d>, selected=<%d> | "
        "research organizer dispatching clones",
        wave, len(all_needs), len(selected_needs),
    )

    # ── Step 3: Spawn tool-armed clones in parallel ──────────────
    clone_tasks = [
        asyncio.wait_for(
            run_clone_research(need, complete, config),
            timeout=config.clone_timeout_s,
        )
        for need in selected_needs
    ]

    clone_results_raw = await asyncio.gather(
        *clone_tasks, return_exceptions=True,
    )

    clone_results: list[CloneResearchResult] = []
    total_findings = 0

    for i, result in enumerate(clone_results_raw):
        if isinstance(result, Exception):
            logger.warning(
                "angle=<%s>, error=<%s> | clone research failed or timed out",
                selected_needs[i].angle, result,
            )
            continue
        if isinstance(result, CloneResearchResult):
            clone_results.append(result)
            # Step 4: Store findings in ConditionStore
            stored = store_clone_findings(store, result, wave, run_id)
            total_findings += stored

    logger.info(
        "wave=<%d>, clones_succeeded=<%d>, total_findings=<%d> | "
        "research organizer complete",
        wave, len(clone_results), total_findings,
    )

    return clone_results
