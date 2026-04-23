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
import string
import time
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


def _safe_substitute(template: str, **kwargs: str) -> str:
    """Single-pass template substitution immune to cross-contamination.

    Unlike sequential ``.replace()`` calls, this uses
    ``string.Template.safe_substitute`` so a replacement value that
    happens to contain a later placeholder token (e.g. ``{articles}``)
    is NOT re-processed.

    Placeholders use ``$name`` syntax internally; the caller's
    ``{name}`` braces are converted automatically.  Doubled braces
    ``{{`` / ``}}`` are treated as literal brace escapes and rendered
    as single ``{`` / ``}`` in the output (matching f-string semantics).

    Args:
        template: The prompt template with ``{name}`` placeholders.
        **kwargs: Name→value mapping for substitution.

    Returns:
        The populated prompt string.
    """
    # 1. Protect doubled braces from the regex by replacing with sentinels
    protected = template.replace("{{", "\x00LBRACE\x00").replace("}}", "\x00RBRACE\x00")
    # 2. Convert {name} placeholders to $name for string.Template
    converted = re.sub(r"\{(\w+)\}", r"$\1", protected)
    # 3. Substitute
    result = string.Template(converted).safe_substitute(**kwargs)
    # 4. Restore doubled braces as single braces (f-string semantics)
    return result.replace("\x00LBRACE\x00", "{").replace("\x00RBRACE\x00", "}")


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
        iterations_run: Planning loop iterations completed.
        retirement_reason: Why the clone stopped.
    """

    angle: str
    doubt: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    search_queries_used: list[str] = field(default_factory=list)
    sources_found: list[str] = field(default_factory=list)
    worker_id: str = ""
    iterations_run: int = 0
    retirement_reason: str = ""


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
        max_planning_iterations: Max plan-act-evaluate loops per clone.
        max_empty_searches: Retire clone after this many fruitless searches.
        max_clone_api_calls_per_run: Global ceiling on search API calls
            across all clones in a single run.
    """

    max_doubts_per_worker: int = 3
    max_clones: int = 4
    max_searches_per_clone: int = 8
    max_extractions_per_clone: int = 3
    clone_timeout_s: float = 180.0
    trigger_uncertainty_threshold: int = 3
    trigger_every_n_waves: int = 2
    max_planning_iterations: int = 6
    max_empty_searches: int = 3
    max_clone_api_calls_per_run: int = 100


@dataclass
class RetirementSignal:
    """Signal from orchestrator to a running clone.

    Attributes:
        should_retire: Whether the clone should stop.
        reason: Human-readable reason for retirement.
        urgency: 'immediate' (stop now) or 'graceful' (finish current step).
    """

    should_retire: bool
    reason: str = ""
    urgency: str = ""  # "immediate" | "graceful"


class RetirementChecker:
    """Evaluates retirement conditions for active clones.

    The orchestrator creates one RetirementChecker per run and passes
    it to all clones.  The checker reads from the ConditionStore and
    from shared orchestrator state.
    """

    def __init__(
        self,
        store: "ConditionStore",
        config: ResearchOrganizerConfig,
    ) -> None:
        self.store = store
        self.config = config
        self._global_api_calls = 0
        self._wave_advancing = False
        self._convergence_achieved = False
        self._lock = asyncio.Lock()

    async def record_api_call(self) -> None:
        """Track a search API call against the global budget."""
        async with self._lock:
            self._global_api_calls += 1

    def signal_wave_advancing(self) -> None:
        """Notify all clones that the next wave is starting."""
        self._wave_advancing = True

    def signal_convergence(self) -> None:
        """Notify all clones that global convergence was reached."""
        self._convergence_achieved = True

    async def check(
        self,
        clone_id: str,
        doubt: str,
        angle: str,
        tool_calls_used: int,
        searches_used: int,
        findings_stored: int,
        elapsed_s: float,
        consecutive_empty_searches: int = 0,
    ) -> RetirementSignal:
        """Evaluate whether a clone should retire.

        Args:
            clone_id: The clone's identifier.
            doubt: The doubt being investigated.
            angle: The research angle.
            tool_calls_used: Total tool invocations so far.
            searches_used: Search API calls so far.
            findings_stored: Findings stored so far.
            elapsed_s: Wall-clock time since clone start.
            consecutive_empty_searches: Searches returning 0 findings in a row.

        Returns:
            RetirementSignal indicating whether to stop.
        """
        # 1. Immediate kills
        if self._convergence_achieved:
            return RetirementSignal(True, "convergence_achieved", "immediate")

        if elapsed_s > self.config.clone_timeout_s:
            return RetirementSignal(True, "timeout", "immediate")

        async with self._lock:
            if self._global_api_calls >= self.config.max_clone_api_calls_per_run:
                return RetirementSignal(True, "global_budget_exhausted", "immediate")

        # 2. Graceful stops
        if self._wave_advancing:
            return RetirementSignal(True, "wave_advancing", "graceful")

        if searches_used >= self.config.max_searches_per_clone:
            return RetirementSignal(True, "clone_budget_exhausted", "graceful")

        # 3. Doubt resolved externally?
        if await self._doubt_resolved(doubt, angle):
            return RetirementSignal(True, "doubt_resolved_externally", "graceful")

        # 4. Diminishing returns
        if consecutive_empty_searches >= self.config.max_empty_searches:
            return RetirementSignal(True, "diminishing_returns", "graceful")

        return RetirementSignal(False)

    async def _doubt_resolved(
        self,
        doubt: str,
        angle: str,
    ) -> bool:
        """Check if another clone already resolved this specific doubt.

        Clone findings store the doubt text in the ``strategy`` column.
        We match on the first 50 chars of the doubt (case-insensitive)
        to tolerate minor LLM wording variation while still being
        doubt-specific rather than angle-level.
        """
        try:
            doubt_prefix = doubt.strip()[:50].lower()
            with self.store._lock:
                count = self.store.conn.execute(
                    """SELECT COUNT(*) FROM conditions
                       WHERE source_type = 'clone_research'
                         AND angle = ?
                         AND confidence >= 0.8
                         AND lower(left(strategy, 50)) = ?""",
                    [angle, doubt_prefix],
                ).fetchone()[0]
            return count >= 2
        except Exception:
            return False


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
  {
    "doubt": "the specific uncertainty",
    "data_needed": "exactly what data would resolve it",
    "priority": "HIGH|MEDIUM|LOW"
  }
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
    prompt = _safe_substitute(
        _DOUBT_EXTRACTION_PROMPT,
        angle=angle,
        transcript=transcript[:15000],
    )

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
    prompt = _safe_substitute(
        _CLONE_SEARCH_PROMPT,
        angle=need.angle,
        doubt=need.doubt,
        data_needed=need.data_needed,
    )

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


_PLAN_PROMPT = """\
You are a {angle} specialist investigating a specific doubt.

DOUBT: {doubt}
DATA NEEDED: {data_needed}

MATERIAL GATHERED SO FAR:
{gathered_so_far}

Based on what you have (and what is still missing), plan your NEXT \
research action. Choose ONE:
- SEARCH: generate 2-3 search queries to find missing data
- EXTRACT: list 1-2 URLs from search results to read in full
- DONE: the doubt is sufficiently resolved

Output ONLY valid JSON:
{{
  "action": "SEARCH" or "EXTRACT" or "DONE",
  "queries": ["q1", "q2"],
  "urls": ["url1"],
  "reasoning": "why this action"
}}"""

_EVALUATE_PROMPT = """\
You are evaluating whether a research doubt has been resolved.

ORIGINAL DOUBT: {doubt}
DATA NEEDED: {data_needed}

FINDINGS SO FAR:
{findings_summary}

Has the doubt been resolved? What specific gaps remain?

Output ONLY valid JSON:
{{
  "resolved": true/false,
  "confidence": 0.0-1.0,
  "remaining_gaps": ["gap1", "gap2"],
  "reasoning": "assessment"
}}"""


async def run_clone_research(
    need: ResearchNeed,
    complete: Callable[[str], Awaitable[str]],
    config: ResearchOrganizerConfig,
    retirement_checker: RetirementChecker | None = None,
) -> CloneResearchResult:
    """Run a clone with a plan-act-evaluate loop to resolve a doubt.

    Instead of a flat fan-out, the clone iterates: it plans what to
    search next, acts (search or extract), evaluates whether the doubt
    is resolved, and checks retirement conditions.  Exits early when
    the doubt is resolved or a retirement rule fires.

    Architecture reference: docs/CLONE_DEEP_AGENT_ARCHITECTURE.md §3.3

    Args:
        need: The research need to investigate.
        complete: Async LLM completion function.
        config: Research organizer configuration.
        retirement_checker: Optional retirement checker for mid-flight
            budget and convergence checks.

    Returns:
        CloneResearchResult with findings, iteration count, and
        retirement reason.
    """
    clone_id = f"clone_{need.source_worker_id}"
    result = CloneResearchResult(
        angle=need.angle,
        doubt=need.doubt,
        worker_id=clone_id,
    )
    t0 = time.monotonic()

    search_fns = _get_available_search_fns()
    all_search_results: list[dict[str, str]] = []
    extracted_texts: list[str] = []
    searches_done = 0
    extractions_done = 0
    consecutive_empty = 0

    logger.info(
        "clone=<%s>, angle=<%s>, doubt=<%s> | starting planning loop",
        clone_id, need.angle, need.doubt[:60],
    )

    for iteration in range(1, config.max_planning_iterations + 1):
        elapsed = time.monotonic() - t0

        # ── Mid-flight retirement check ──────────────────────────
        if retirement_checker is not None:
            signal = await retirement_checker.check(
                clone_id=clone_id,
                doubt=need.doubt,
                angle=need.angle,
                tool_calls_used=searches_done + extractions_done,
                searches_used=searches_done,
                findings_stored=len(result.findings),
                elapsed_s=elapsed,
                consecutive_empty_searches=consecutive_empty,
            )
            if signal.should_retire:
                result.retirement_reason = signal.reason
                result.iterations_run = iteration - 1
                logger.info(
                    "clone=<%s>, iteration=<%d>, reason=<%s> | "
                    "retiring mid-flight",
                    clone_id, iteration, signal.reason,
                )
                break

        # ── PLAN: ask LLM what to do next ────────────────────────
        gathered = _summarize_gathered(
            all_search_results, extracted_texts, result.findings,
        )
        plan_prompt = _safe_substitute(
            _PLAN_PROMPT,
            angle=need.angle,
            doubt=need.doubt,
            data_needed=need.data_needed,
            gathered_so_far=gathered or "(nothing yet — first iteration)",
        )

        try:
            plan_raw = await complete(plan_prompt)
            plan = _parse_plan(plan_raw)
        except Exception as exc:
            logger.warning(
                "clone=<%s>, iteration=<%d>, error=<%s> | plan failed",
                clone_id, iteration, exc,
            )
            # Fall back to initial query generation on first iteration
            if iteration == 1:
                queries = await _generate_clone_queries(need, complete)
                plan = {"action": "SEARCH", "queries": queries}
            else:
                break

        action = plan.get("action", "DONE").upper()

        # ── ACT: execute the planned action ──────────────────────
        if action == "DONE":
            result.retirement_reason = "doubt_resolved"
            result.iterations_run = iteration
            logger.info(
                "clone=<%s>, iteration=<%d> | plan says DONE",
                clone_id, iteration,
            )
            break

        if action == "SEARCH":
            queries = plan.get("queries", [])
            if not queries:
                queries = [f"{need.angle} {need.doubt[:100]}"]
            result.search_queries_used.extend(queries)

            new_results = 0
            for q in queries:
                if searches_done >= config.max_searches_per_clone:
                    break
                fn = search_fns[searches_done % len(search_fns)]
                try:
                    hits = await fn(q)
                    all_search_results.extend(hits)
                    new_results += len(hits)
                    searches_done += 1
                    if retirement_checker is not None:
                        await retirement_checker.record_api_call()
                except Exception as exc:
                    logger.warning(
                        "clone=<%s>, query=<%s>, error=<%s> | search failed",
                        clone_id, q[:50], exc,
                    )

            if new_results == 0:
                consecutive_empty += 1
            else:
                consecutive_empty = 0

        elif action == "EXTRACT":
            urls = plan.get("urls", [])
            if not urls:
                # Pick from search results
                for r in all_search_results:
                    url = r.get("url", "")
                    if url and url not in {s for s in result.sources_found}:
                        urls.append(url)
                        if len(urls) >= 2:
                            break

            for url in urls:
                if extractions_done >= config.max_extractions_per_clone:
                    break
                try:
                    content = await extract_url_content(url)
                    if content:
                        extracted_texts.append(content)
                        result.sources_found.append(url)
                    extractions_done += 1
                    if retirement_checker is not None:
                        await retirement_checker.record_api_call()
                except Exception as exc:
                    logger.warning(
                        "clone=<%s>, url=<%s>, error=<%s> | extraction failed",
                        clone_id, url[:80], exc,
                    )

        # ── Synthesize after every iteration that gathered material ──
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
            result.findings = findings if findings else result.findings

        # ── EVALUATE: did this resolve the doubt? ────────────────
        if result.findings:
            eval_result = await _evaluate_doubt_resolution(
                need=need,
                findings=result.findings,
                complete=complete,
            )
            if eval_result.get("resolved", False):
                result.retirement_reason = "doubt_resolved"
                result.iterations_run = iteration
                logger.info(
                    "clone=<%s>, iteration=<%d>, findings=<%d> | "
                    "doubt resolved by evaluation",
                    clone_id, iteration, len(result.findings),
                )
                break

        result.iterations_run = iteration

    else:
        # Exhausted all iterations
        result.retirement_reason = "max_iterations"

    logger.info(
        "clone=<%s>, iterations=<%d>, findings=<%d>, sources=<%d>, "
        "reason=<%s>, elapsed_s=<%.1f> | clone planning loop complete",
        clone_id, result.iterations_run, len(result.findings),
        len(result.sources_found), result.retirement_reason,
        time.monotonic() - t0,
    )

    return result


def _summarize_gathered(
    search_results: list[dict[str, str]],
    extracted_texts: list[str],
    findings: list[dict[str, Any]],
) -> str:
    """Build a summary of what the clone has gathered so far."""
    parts: list[str] = []

    if search_results:
        parts.append(f"Search results: {len(search_results)} hits")
        for r in search_results[:5]:
            parts.append(f"  - {r.get('title', 'untitled')}: {r.get('snippet', '')[:100]}")

    if extracted_texts:
        parts.append(f"Full articles extracted: {len(extracted_texts)}")
        for t in extracted_texts:
            parts.append(f"  - ({len(t)} chars) {t[:100]}...")

    if findings:
        parts.append(f"Findings synthesized: {len(findings)}")
        for f in findings[:3]:
            parts.append(f"  - {f.get('fact', '')[:120]}")

    return "\n".join(parts) if parts else ""


def _parse_plan(raw: str) -> dict[str, Any]:
    """Parse the LLM's plan response into a dict."""
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        plan = json.loads(raw)
        if isinstance(plan, dict):
            return plan
    except json.JSONDecodeError:
        pass

    # Heuristic fallback
    upper = raw.upper()
    if "DONE" in upper:
        return {"action": "DONE"}
    if "EXTRACT" in upper:
        return {"action": "EXTRACT", "urls": []}
    return {"action": "SEARCH", "queries": []}


async def _evaluate_doubt_resolution(
    need: ResearchNeed,
    findings: list[dict[str, Any]],
    complete: Callable[[str], Awaitable[str]],
) -> dict[str, Any]:
    """Ask LLM whether the gathered findings resolve the original doubt."""
    findings_summary = "\n".join(
        f"- {f.get('fact', '')[:200]} (confidence: {f.get('confidence', '?')})"
        for f in findings[:10]
    )

    prompt = _safe_substitute(
        _EVALUATE_PROMPT,
        doubt=need.doubt,
        data_needed=need.data_needed,
        findings_summary=findings_summary or "(no findings yet)",
    )

    try:
        raw = await complete(prompt)
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass

    return {"resolved": False, "confidence": 0.0}


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
  {
    "fact": "specific factual claim with exact numbers",
    "source": "URL or article title",
    "confidence": 0.8,
    "resolves_doubt": true/false
  }
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
    prompt = _safe_substitute(
        _SYNTHESIS_PROMPT,
        angle=need.angle,
        doubt=need.doubt,
        data_needed=need.data_needed,
        articles=full_articles[:20000] if full_articles else "(none)",
        snippets=snippets[:5000] if snippets else "(none)",
    )

    try:
        content = await complete(prompt)
    except Exception as exc:
        logger.warning(
            "angle=<%s>, error=<%s> | clone synthesis LLM call failed",
            need.angle, exc,
        )
        return _heuristic_clone_findings(snippets, need)

    if not content:
        logger.warning(
            "angle=<%s> | clone synthesis returned empty content",
            need.angle,
        )
        return _heuristic_clone_findings(snippets, need)

    logger.debug(
        "angle=<%s>, content_len=<%d>, content_preview=<%s> | "
        "clone synthesis raw response",
        need.angle, len(content), content[:200],
    )

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    # Detect LLM refusal (censorship)
    refusal_phrases = [
        "i can't", "i cannot", "i'm unable", "i am unable",
        "i'm not able", "i won't", "i will not",
        "not able to provide", "not able to assist",
        "is there anything else",
    ]
    content_lower = content.lower()
    if any(phrase in content_lower for phrase in refusal_phrases):
        logger.warning(
            "angle=<%s>, content_preview=<%s> | clone synthesis "
            "LLM refused, falling back to heuristic",
            need.angle, content[:100],
        )
        return _heuristic_clone_findings(snippets, need)

    # Try to parse JSON — first directly, then by extracting an embedded
    # JSON array from preamble text (LLMs often wrap valid JSON in prose
    # like "Here is the analysis:\n\n[{...}]").
    parsed: list | None = None
    try:
        raw = json.loads(content)
        if isinstance(raw, list):
            parsed = raw
    except json.JSONDecodeError:
        pass

    if parsed is None:
        # Look for the first '[' ... last ']' span in the content
        bracket_start = content.find("[")
        bracket_end = content.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            candidate = content[bracket_start:bracket_end + 1]
            try:
                raw = json.loads(candidate)
                if isinstance(raw, list):
                    parsed = raw
                    logger.debug(
                        "angle=<%s> | clone synthesis JSON extracted "
                        "from preamble text",
                        need.angle,
                    )
            except json.JSONDecodeError:
                pass

    if parsed is None:
        logger.warning(
            "angle=<%s>, content_preview=<%s> | clone synthesis "
            "JSON parse failed, falling back to heuristic",
            need.angle, content[:200],
        )
        return _heuristic_clone_findings(snippets, need)

    findings = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
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
    if not findings:
        logger.info(
            "angle=<%s> | clone synthesis JSON parsed but empty, "
            "falling back to heuristic",
            need.angle,
        )
        return _heuristic_clone_findings(snippets, need)
    return findings


def _heuristic_clone_findings(
    snippets: str,
    need: "ResearchNeed",
) -> list[dict[str, Any]]:
    """Extract findings from search snippets when LLM synthesis fails.

    Uses pattern matching to identify sentences with specific claims
    (numbers, dosages, mechanisms) from raw search snippets.  This is
    the clone equivalent of ``finding_extractor.extract_findings_heuristic``.

    Args:
        snippets: Raw search result snippets (one per line).
        need: The research need being investigated.

    Returns:
        List of finding dicts with fact, source, confidence.
    """
    if not snippets or len(snippets.strip()) < 30:
        return []

    findings: list[dict[str, Any]] = []

    # Patterns indicating factual claims worth extracting
    claim_patterns = [
        r"\d+\s*(?:mg|mcg|iu|ml|µg|ng|g/dl|mmol|%)",
        r"\d+\s*(?:minutes?|hours?|days?|weeks?)",
        r"(?:increases?|decreases?|inhibits?|activates?|binds?)\s",
        r"(?:receptor|pathway|enzyme|hormone|protein)\s",
        r"(?:dosage|dose|protocol|cycle|stack)\s",
    ]

    for line in snippets.split("\n"):
        line = line.strip()
        if not line or len(line) < 40:
            continue

        # Strip leading "- Title: " prefix from snippet lines
        if line.startswith("- "):
            colon_idx = line.find(":", 2)
            if colon_idx > 0 and colon_idx < 80:
                snippet_text = line[colon_idx + 1:].strip()
            else:
                snippet_text = line[2:].strip()
        else:
            snippet_text = line

        if not snippet_text or len(snippet_text) < 30:
            continue

        # Check if the snippet contains a factual claim
        for pattern in claim_patterns:
            if re.search(pattern, snippet_text, re.IGNORECASE):
                findings.append({
                    "fact": snippet_text[:500],
                    "source": "search_snippet",
                    "confidence": 0.5,
                    "resolves_doubt": False,
                })
                break

        if len(findings) >= 15:
            break

    logger.info(
        "angle=<%s>, findings=<%d> | heuristic clone finding extraction",
        need.angle, len(findings),
    )

    return findings


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
    should_trigger = (config.trigger_every_n_waves > 0 and wave % config.trigger_every_n_waves == 0)

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

    # ── Step 3: Pre-spawn retirement checks ────────────────────────
    retirement_checker = RetirementChecker(store, config)
    spawn_needs: list[ResearchNeed] = []

    for need in selected_needs:
        # Pre-spawn rule: doubt already resolved?
        if await retirement_checker._doubt_resolved(need.doubt, need.angle):
            logger.info(
                "angle=<%s>, doubt=<%s> | pre-spawn skip: doubt already resolved",
                need.angle, need.doubt[:60],
            )
            continue

        # Pre-spawn rule: duplicate doubt (same angle, same core question)?
        is_dup = False
        for existing in spawn_needs:
            if (existing.angle == need.angle
                    and existing.doubt.lower()[:50] == need.doubt.lower()[:50]):
                is_dup = True
                break
        if is_dup:
            logger.info(
                "angle=<%s>, doubt=<%s> | pre-spawn skip: duplicate doubt",
                need.angle, need.doubt[:60],
            )
            continue

        spawn_needs.append(need)

    if not spawn_needs:
        logger.info(
            "wave=<%d> | all doubts filtered by pre-spawn checks",
            wave,
        )
        return []

    logger.info(
        "wave=<%d>, pre_spawn_selected=<%d>, filtered=<%d> | "
        "spawning clones with planning loop",
        wave, len(spawn_needs), len(selected_needs) - len(spawn_needs),
    )

    # ── Step 4: Spawn tool-armed clones in parallel ──────────────
    clone_tasks = [
        asyncio.wait_for(
            run_clone_research(need, complete, config, retirement_checker),
            timeout=config.clone_timeout_s,
        )
        for need in spawn_needs
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
                spawn_needs[i].angle, result,
            )
            continue
        if isinstance(result, CloneResearchResult):
            clone_results.append(result)
            # Step 5: Store findings in ConditionStore
            stored = store_clone_findings(store, result, wave, run_id)
            total_findings += stored

    logger.info(
        "wave=<%d>, clones_succeeded=<%d>, total_findings=<%d> | "
        "research organizer complete",
        wave, len(clone_results), total_findings,
    )

    return clone_results
