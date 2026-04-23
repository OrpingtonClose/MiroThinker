# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Orchestrator-side finding extraction from worker reasoning text.

Workers are tool-free — they produce unstructured reasoning text.  The
orchestrator extracts structured findings from that text and stores them
in the ConditionStore.  This is the bridge between free-form reasoning
and the structured audit trail.

Two extraction strategies:
    1. LLM-based: Ask the LLM to parse the worker's output into structured
       claims.  Higher quality, costs one LLM call per worker per wave.
    2. Heuristic: Pattern-match sentences that look like findings (contain
       numbers, dosages, mechanisms).  Free, lower quality, good fallback.

Architecture reference: docs/SWARM_WAVE_ARCHITECTURE.md § Phase 1E
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)

# Markers that indicate web scraping garbage, forum UI elements, or
# non-informative content that should never be stored as findings.
_GARBAGE_MARKERS = [
    "Logged", "getbig.com", "Getbig", "star.gif", "star01.gif",
    "blocked by an extension", "Competitors II", "ip.gif",
    "post/xx.gif", "Themes/default", ".gif)", ".png)",
    "shopify.com/s/files", "cdn.shopify.com",
]


def _is_garbage_finding(fact: str) -> bool:
    """Check if a finding is web scraping garbage or too low quality.

    Filters out HTML/markdown artifacts, forum UI elements, separators,
    and overly short or non-alphabetic content.

    Args:
        fact: The finding text to evaluate.

    Returns:
        True if the finding should be rejected.
    """
    stripped = fact.strip()

    if len(stripped) < 30:
        return True

    # Markdown image syntax (scraped web artifacts)
    if "![" in stripped:
        return True

    lower = stripped.lower()
    if any(marker.lower() in lower for marker in _GARBAGE_MARKERS):
        return True

    # Pure separators
    if stripped in ("---", "* * *", "***", "===", "—", "–"):
        return True

    # Mostly non-alphabetic (raw URLs, markdown, encoded content)
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 0 and alpha_chars / len(stripped) < 0.3:
        return True

    return False


@dataclass
class ExtractedFinding:
    """A structured finding extracted from worker reasoning text.

    Attributes:
        fact: The factual claim.
        confidence: Estimated confidence (0.0-1.0).
        angle: The research angle this finding belongs to.
        source_type: Always 'worker_analysis' for tool-free workers.
        source_url: URL reference extracted from the worker's reasoning.
        tags: Optional categorization tags.
    """

    fact: str
    confidence: float = 0.7
    angle: str = ""
    source_type: str = "worker_analysis"
    source_url: str = ""
    tags: list[str] = field(default_factory=list)


async def extract_findings_llm(
    worker_output: str,
    angle: str,
    query: str,
    complete: Callable[[str], Awaitable[str]],
    *,
    max_findings: int = 30,
) -> list[ExtractedFinding]:
    """Extract structured findings from worker output using an LLM.

    Asks the orchestrator's LLM to parse the worker's reasoning into
    discrete factual claims with confidence scores.

    Args:
        worker_output: The worker's full reasoning text.
        angle: Research angle the worker was analyzing.
        query: The user's research query.
        complete: LLM completion function.
        max_findings: Maximum number of findings to extract.

    Returns:
        List of ExtractedFinding objects.
    """
    if not worker_output or len(worker_output.strip()) < 50:
        logger.debug(
            "angle=<%s>, output_chars=<%d> | worker output too short for extraction",
            angle, len(worker_output),
        )
        return []

    prompt = (
        f"Extract discrete factual claims from this research analysis.\n\n"
        f"RESEARCH ANGLE: {angle}\n"
        f"RESEARCH QUERY: {query}\n\n"
        f"WORKER ANALYSIS:\n{worker_output}\n\n"
        f"Extract up to {max_findings} specific, evidence-backed claims. "
        f"For each claim, provide:\n"
        f"- fact: the specific claim (preserve exact numbers, dosages, citations)\n"
        f"- confidence: 0.0-1.0 (how well-supported is this claim?)\n"
        f"- source: if the analysis references a URL or [Source: ...] marker, "
        f"include the URL here. Otherwise omit or leave empty.\n"
        f"- tags: categorization (e.g. 'mechanism', 'dosage', 'timing', "
        f"'interaction', 'gap', 'contradiction')\n\n"
        f"Return ONLY a JSON array of objects. No markdown, no explanation.\n"
        f"Example: [{{"
        f'"fact": "Insulin sensitivity peaks 45 min post-workout", '
        f'"confidence": 0.85, '
        f'"source": "https://example.com/study", '
        f'"tags": ["timing", "mechanism"]'
        f"}}]\n\n"
        f"JSON array:"
    )

    try:
        response = await complete(prompt)
        findings = _parse_findings_json(response, angle)

        if not findings:
            logger.info(
                "angle=<%s> | LLM extraction returned 0 findings, falling back to heuristic",
                angle,
            )
            findings = extract_findings_heuristic(worker_output, angle, max_findings=max_findings)

        logger.info(
            "angle=<%s>, findings_extracted=<%d> | LLM finding extraction complete",
            angle, len(findings),
        )

        return findings[:max_findings]
    except Exception as exc:
        logger.warning(
            "angle=<%s>, error=<%s> | LLM finding extraction failed, falling back to heuristic",
            angle, exc,
        )
        return extract_findings_heuristic(worker_output, angle, max_findings=max_findings)


def extract_findings_heuristic(
    worker_output: str,
    angle: str,
    *,
    max_findings: int = 30,
) -> list[ExtractedFinding]:
    """Extract findings from worker output using pattern matching.

    Identifies sentences that contain specific claims: numbers, dosages,
    mechanisms, timing information, and other domain-specific patterns.

    Args:
        worker_output: The worker's full reasoning text.
        angle: Research angle the worker was analyzing.
        max_findings: Maximum number of findings to extract.

    Returns:
        List of ExtractedFinding objects.
    """
    if not worker_output or len(worker_output.strip()) < 50:
        return []

    findings: list[ExtractedFinding] = []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', worker_output)

    # Patterns that indicate a factual claim worth extracting
    claim_patterns = [
        # Numbers and measurements
        (r'\d+\s*(?:mg|mcg|iu|ml|µg|ng|g/dl|mmol|%)', 0.8, ["dosage"]),
        # Timing patterns
        (r'\d+\s*(?:min|hour|h|day|week|month|minute)', 0.75, ["timing"]),
        # Mechanism keywords
        (r'(?:inhibit|stimulat|upregulat|downregulat|activat|suppress|induc|block|bind)', 0.7, ["mechanism"]),
        # Causal language
        (r'(?:because|therefore|consequently|leads to|results in|causes|driven by)', 0.65, ["causal"]),
        # Interaction patterns
        (r'(?:interact|compound|synerg|antagoni|potentiat|amplif)', 0.75, ["interaction"]),
        # Contradiction markers
        (r'(?:contradict|conflict|however|despite|although|inconsistent)', 0.6, ["contradiction"]),
        # Gap markers
        (r'(?:unknown|unclear|insufficient|no evidence|need.{0,20}data|gap)', 0.5, ["gap"]),
    ]

    seen_facts: set[str] = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 30 or len(sentence) > 500:
            continue

        # Check against claim patterns
        best_conf = 0.0
        best_tags: list[str] = []

        for pattern, conf, tags in claim_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                if conf > best_conf:
                    best_conf = conf
                    best_tags = tags

        if best_conf > 0.0:
            # Deduplicate by normalized content
            normalized = sentence.lower().strip()
            if normalized not in seen_facts:
                seen_facts.add(normalized)
                findings.append(ExtractedFinding(
                    fact=sentence,
                    confidence=best_conf,
                    angle=angle,
                    tags=best_tags,
                ))

        if len(findings) >= max_findings:
            break

    logger.info(
        "angle=<%s>, findings_extracted=<%d> | heuristic finding extraction complete",
        angle, len(findings),
    )

    return findings


def store_extracted_findings(
    store: "ConditionStore",
    findings: list[ExtractedFinding],
    *,
    source_model: str = "",
    source_run: str = "",
    iteration: int = 0,
) -> int:
    """Store extracted findings in the ConditionStore.

    Args:
        store: The shared ConditionStore.
        findings: List of extracted findings to store.
        source_model: Model name for provenance.
        source_run: Run identifier for provenance.
        iteration: Wave number.

    Returns:
        Number of findings successfully stored.
    """
    stored = 0
    skipped_garbage = 0

    for f in findings:
        # Filter garbage at ingestion — HTML artifacts, forum UI, separators
        if _is_garbage_finding(f.fact):
            skipped_garbage += 1
            continue

        try:
            store.admit(
                fact=f.fact,
                row_type="finding",
                source_type=f.source_type,
                source_url=f.source_url or None,
                angle=f.angle,
                confidence=f.confidence,
                iteration=iteration,
                source_model=source_model,
                source_run=source_run,
            )
            stored += 1
        except Exception as exc:
            logger.debug(
                "angle=<%s>, error=<%s> | failed to store finding",
                f.angle, exc,
            )

    if skipped_garbage:
        logger.info(
            "skipped_garbage=<%d> | filtered garbage findings at ingestion",
            skipped_garbage,
        )

    logger.info(
        "stored=<%d>, total=<%d>, source_run=<%s> | findings stored in ConditionStore",
        stored, len(findings), source_run,
    )

    return stored


def store_worker_transcript(
    store: "ConditionStore",
    worker_id: str,
    angle: str,
    transcript: str,
    *,
    source_model: str = "",
    source_run: str = "",
    iteration: int = 0,
) -> None:
    """Store the full worker transcript for audit trail.

    The transcript is stored as a 'worker_transcript' row in the
    ConditionStore.  This enables clone reconstruction in later stages
    and provides full lineage from raw reasoning to extracted findings.

    Args:
        store: The shared ConditionStore.
        worker_id: Worker identifier.
        angle: Research angle.
        transcript: The worker's full reasoning output.
        source_model: Model name for provenance.
        source_run: Run identifier for provenance.
        iteration: Wave number.
    """
    try:
        store.admit(
            fact=transcript,
            row_type="worker_transcript",
            source_type="worker_reasoning",
            angle=angle,
            confidence=1.0,
            iteration=iteration,
            source_model=source_model,
            source_run=source_run,
            consider_for_use=False,
        )

        logger.info(
            "worker_id=<%s>, angle=<%s>, transcript_chars=<%d> | transcript stored",
            worker_id, angle, len(transcript),
        )
    except Exception as exc:
        logger.warning(
            "worker_id=<%s>, angle=<%s>, error=<%s> | failed to store transcript",
            worker_id, angle, exc,
        )


def _parse_findings_json(
    response: str,
    angle: str,
) -> list[ExtractedFinding]:
    """Parse LLM response into ExtractedFinding objects.

    Handles common LLM response quirks: markdown code blocks,
    trailing commas, extra text around the JSON array.

    Args:
        response: Raw LLM response text.
        angle: Research angle for the findings.

    Returns:
        List of ExtractedFinding objects.
    """
    # Strip markdown code blocks
    text = response.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    # Try to find a JSON array in the response
    array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if not array_match:
        logger.debug(
            "angle=<%s> | no JSON array found in LLM response",
            angle,
        )
        return []

    json_text = array_match.group()

    # Remove trailing commas before ] (common LLM mistake)
    json_text = re.sub(r',\s*\]', ']', json_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | JSON parse failed",
            angle, exc,
        )
        return []

    if not isinstance(data, list):
        return []

    findings: list[ExtractedFinding] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        fact = item.get("fact", "")
        if not fact or len(fact) < 10:
            continue

        try:
            confidence = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        tags = item.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        source_url = item.get("source", "") or ""
        # Reject non-URL values the LLM may hallucinate
        if source_url and not source_url.startswith(("http://", "https://")):
            source_url = ""

        findings.append(ExtractedFinding(
            fact=fact,
            confidence=confidence,
            angle=angle,
            source_url=source_url,
            tags=tags,
        ))

    return findings
