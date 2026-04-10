# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Phase 0 Scout — pre-research landscape probes.

Before the main research loop, run cheap probes to assess which
sub-questions are broad/complex (warrant deep research) vs. targeted
(regular tools suffice).  The results are injected into session state
so the thinker sees them on iteration 1 instead of "(no findings yet)".

5-step process:
  1. Comprehend — decompose query into entities, domains, implicit
     questions, adjacent territories, and sub-questions.
  2. Decompose — extract 3-6 concrete sub-questions.
  3. Probe — run ONE cheap search per sub-question (Brave or Kagi).
  4. Assess — classify each sub-question as SHALLOW / MODERATE / DEEP.
  5. Inject — format as verbal prose and write to state.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
_KAGI_API_KEY = os.environ.get("KAGI_API_KEY", "")

# LLM endpoint — reuse the Flock proxy or fall back to OpenAI-compatible
# endpoint configured for the main model.
_LLM_API_BASE = os.environ.get(
    "SCOUT_LLM_API_BASE",
    os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
)
_LLM_API_KEY = os.environ.get(
    "SCOUT_LLM_API_KEY",
    os.environ.get("OPENAI_API_KEY", ""),
)
_LLM_MODEL = os.environ.get(
    "SCOUT_LLM_MODEL",
    os.environ.get("ADK_MODEL", "gpt-4o"),
)
# Strip the litellm/ prefix if present
if _LLM_MODEL.startswith("litellm/"):
    _LLM_MODEL = _LLM_MODEL[len("litellm/"):]


# ---------------------------------------------------------------------------
# Step 1: Query Comprehension
# ---------------------------------------------------------------------------

@dataclass
class QueryComprehension:
    """Deep semantic understanding of a research query.

    Ported from deep-search-portal's pipeline.py — maps the full
    knowledge territory: entities, domains, implicit questions,
    adjacent territories, and sub-questions.
    """

    entities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    implicit_questions: list[str] = field(default_factory=list)
    adjacent_territories: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    core_need: str = ""


_COMPREHENSION_PROMPT = """\
You are a research analyst.  Deeply understand what this query is about \
— not just the surface words, but the full knowledge territory.

Research query: {query}

Analyze this query and output ONLY valid JSON:
{
  "entities": ["every entity, person, substance, organization, concept mentioned or implied"],
  "domains": ["every knowledge domain this touches — be expansive"],
  "implicit_questions": ["what is the user REALLY trying to accomplish? list 5-8 implicit questions"],
  "adjacent_territories": ["topics NOT in the query but likely to contain relevant deep knowledge"],
  "sub_questions": ["3-6 concrete sub-questions that research needs to answer"],
  "semantic_summary": "one paragraph explaining what this query is really about",
  "core_need": "one sentence describing what the user ultimately needs"
}"""


async def _llm_complete(prompt: str, max_tokens: int = 2048) -> str:
    """Call the configured LLM endpoint (OpenAI-compatible)."""
    if not _LLM_API_KEY:
        return ""

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_LLM_API_KEY}",
    }
    body = {
        "model": _LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{_LLM_API_BASE}/chat/completions",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
    except Exception as exc:
        logger.warning("Scout LLM call failed: %s", exc)
        return ""


async def _comprehend_query(query: str) -> QueryComprehension:
    """Step 1: produce a deep semantic understanding of the query."""
    prompt = _COMPREHENSION_PROMPT.replace("{query}", query[:2000])
    content = await _llm_complete(prompt)

    if not content:
        # Fallback: minimal comprehension from query words
        words = [w for w in re.split(r"\W+", query.lower()) if len(w) > 3]
        return QueryComprehension(
            entities=words[:10],
            sub_questions=[query],
            semantic_summary=query,
        )

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        words = [w for w in re.split(r"\W+", query.lower()) if len(w) > 3]
        return QueryComprehension(
            entities=words[:10],
            sub_questions=[query],
            semantic_summary=query,
        )

    return QueryComprehension(
        entities=data.get("entities", [])[:20],
        domains=data.get("domains", [])[:15],
        implicit_questions=data.get("implicit_questions", [])[:10],
        adjacent_territories=data.get("adjacent_territories", [])[:10],
        sub_questions=data.get("sub_questions", [])[:6],
        semantic_summary=data.get("semantic_summary", ""),
        core_need=data.get("core_need", "")[:500],
    )


# ---------------------------------------------------------------------------
# Step 2: Decompose (extract sub-questions from comprehension)
# ---------------------------------------------------------------------------

def _extract_sub_questions(comp: QueryComprehension) -> list[str]:
    """Step 2: return 3-6 concrete sub-questions from the comprehension."""
    questions = list(comp.sub_questions)
    # If comprehension didn't produce enough, pull from implicit_questions
    if len(questions) < 3:
        for iq in comp.implicit_questions:
            if iq not in questions:
                questions.append(iq)
            if len(questions) >= 6:
                break
    return questions[:6]


# ---------------------------------------------------------------------------
# Step 3: Probe — cheap search per sub-question
# ---------------------------------------------------------------------------

async def _probe_brave(query: str) -> dict[str, Any]:
    """Run a cheap Brave search probe (3 results)."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 3},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": _BRAVE_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("web", {}).get("results", [])
            return {
                "query": query,
                "count": len(results),
                "snippets": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("description", "")[:200],
                    }
                    for r in results[:3]
                ],
                "error": None,
            }
    except Exception as exc:
        return {"query": query, "count": 0, "snippets": [], "error": str(exc)}


async def _probe_kagi(query: str) -> dict[str, Any]:
    """Run a cheap Kagi fastgpt probe."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://kagi.com/api/v0/fastgpt",
                params={"query": query},
                headers={"Authorization": f"Bot {_KAGI_API_KEY}"},
            )
            resp.raise_for_status()
            data = resp.json()
            output = data.get("data", {}).get("output", "")
            refs = data.get("data", {}).get("references", [])
            return {
                "query": query,
                "count": len(refs),
                "snippets": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", "")[:200],
                    }
                    for r in refs[:3]
                ],
                "summary": output[:500] if output else "",
                "error": None,
            }
    except Exception as exc:
        return {"query": query, "count": 0, "snippets": [], "error": str(exc)}


async def _probe_sub_question(query: str) -> dict[str, Any]:
    """Step 3: probe a single sub-question using the best available API."""
    if _BRAVE_API_KEY:
        return await _probe_brave(query)
    if _KAGI_API_KEY:
        return await _probe_kagi(query)
    # No API key — return a stub indicating we couldn't probe
    return {
        "query": query,
        "count": -1,
        "snippets": [],
        "error": "No search API key configured (BRAVE_API_KEY or KAGI_API_KEY)",
    }


# ---------------------------------------------------------------------------
# Step 4: Assess — classify each sub-question tier
# ---------------------------------------------------------------------------

_ASSESS_PROMPT = """\
Given these scout probe results for each sub-question, assess the \
research landscape.

For each sub-question, classify as:
- SHALLOW: comprehensive information readily available, regular search tools sufficient
- MODERATE: partial information available, needs multiple targeted searches
- DEEP: sparse/complex/broad topic, would benefit from a deep research service

Probe results:
{probes}

Output ONLY valid JSON — an array of objects:
[
  {"sub_question": "...", "tier": "SHALLOW|MODERATE|DEEP", "rationale": "...", "recommended_approach": "..."}
]"""


@dataclass
class SubQuestionAssessment:
    """Assessment of a single sub-question's research difficulty."""

    sub_question: str = ""
    tier: str = "MODERATE"  # SHALLOW | MODERATE | DEEP
    rationale: str = ""
    recommended_approach: str = ""


async def _assess_landscape(
    probes: list[dict[str, Any]],
) -> list[SubQuestionAssessment]:
    """Step 4: use LLM to classify each sub-question's tier."""
    probes_text = json.dumps(probes, indent=2, ensure_ascii=False)
    prompt = _ASSESS_PROMPT.replace("{probes}", probes_text[:6000])

    content = await _llm_complete(prompt)
    if not content:
        # Fallback: heuristic-based assessment
        return _heuristic_assess(probes)

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    try:
        items = json.loads(content)
        if not isinstance(items, list):
            return _heuristic_assess(probes)
        return [
            SubQuestionAssessment(
                sub_question=item.get("sub_question", ""),
                tier=item.get("tier", "MODERATE").upper(),
                rationale=item.get("rationale", ""),
                recommended_approach=item.get("recommended_approach", ""),
            )
            for item in items
        ]
    except json.JSONDecodeError:
        return _heuristic_assess(probes)


def _heuristic_assess(probes: list[dict[str, Any]]) -> list[SubQuestionAssessment]:
    """Fallback heuristic when LLM assessment fails."""
    results = []
    for p in probes:
        count = p.get("count", 0)
        if count >= 3:
            tier = "SHALLOW"
            rationale = f"Found {count} results readily available."
            approach = "Regular search tools (Brave/Exa) will suffice."
        elif count >= 1:
            tier = "MODERATE"
            rationale = f"Only {count} result(s) found — partial coverage."
            approach = "Multiple targeted searches needed."
        else:
            tier = "DEEP"
            rationale = "No results or error — sparse information landscape."
            approach = "Deep research service recommended (Perplexity/Grok)."
        results.append(SubQuestionAssessment(
            sub_question=p.get("query", ""),
            tier=tier,
            rationale=rationale,
            recommended_approach=approach,
        ))
    return results


# ---------------------------------------------------------------------------
# Step 5: Inject — format as verbal prose and write to state
# ---------------------------------------------------------------------------

def _format_landscape_assessment(
    assessments: list[SubQuestionAssessment],
    comprehension: QueryComprehension,
) -> str:
    """Step 5: format the landscape assessment as verbal prose."""
    lines = ["LANDSCAPE ASSESSMENT (from Phase 0 scout probes):", ""]

    if comprehension.semantic_summary:
        lines.append(f"Query understanding: {comprehension.semantic_summary}")
        lines.append("")

    for a in assessments:
        lines.append(f"Sub-question: \"{a.sub_question}\"")
        lines.append(f"Assessment: {a.rationale}")
        lines.append(f"Recommended approach: {a.recommended_approach}")
        lines.append(f"Tier: {a.tier}")
        lines.append("")

    # Summary counts
    shallow = sum(1 for a in assessments if a.tier == "SHALLOW")
    moderate = sum(1 for a in assessments if a.tier == "MODERATE")
    deep = sum(1 for a in assessments if a.tier == "DEEP")
    lines.append(
        f"Summary: {shallow} shallow, {moderate} moderate, {deep} deep "
        f"sub-questions identified."
    )

    if deep > 0:
        lines.append(
            "Deep sub-questions should be delegated to deep research "
            "services (Perplexity, Grok, or Tavily) for comprehensive "
            "coverage."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_scout_phase(query: str, state: dict[str, Any]) -> None:
    """Run the full 5-step Phase 0 scout and inject results into state.

    This is called once at pipeline initialization, before the first
    thinker iteration.  It replaces "(no findings yet)" with a
    landscape assessment so the thinker has empirical data to plan with.

    Args:
        query: The user's research query.
        state: The session state dict (modified in place).
    """
    logger.info("Phase 0 scout: starting for query: %s", query[:100])

    # Step 1: Comprehend
    comprehension = await _comprehend_query(query)
    logger.info(
        "Scout step 1 (comprehend): %d entities, %d domains, %d sub-questions",
        len(comprehension.entities),
        len(comprehension.domains),
        len(comprehension.sub_questions),
    )

    # Step 2: Decompose
    sub_questions = _extract_sub_questions(comprehension)
    if not sub_questions:
        logger.warning("Scout: no sub-questions extracted, skipping probes")
        return
    logger.info("Scout step 2 (decompose): %d sub-questions", len(sub_questions))

    # Step 3: Probe
    probes: list[dict[str, Any]] = []
    for sq in sub_questions:
        probe = await _probe_sub_question(sq)
        probes.append(probe)
    logger.info("Scout step 3 (probe): %d probes completed", len(probes))

    # Step 4: Assess
    assessments = await _assess_landscape(probes)
    logger.info("Scout step 4 (assess): %d assessments", len(assessments))

    # Step 5: Inject
    landscape_text = _format_landscape_assessment(assessments, comprehension)
    current_findings = state.get("research_findings", "(no findings yet)")
    if current_findings == "(no findings yet)":
        state["research_findings"] = landscape_text
    else:
        state["research_findings"] = landscape_text + "\n\n" + current_findings

    logger.info(
        "Scout step 5 (inject): landscape assessment injected (%d chars)",
        len(landscape_text),
    )
