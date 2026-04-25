# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Queen merge — stitches worker analyses into a single coherent document.

The queen is an EDITOR, not an author.  Workers have done the deep
reasoning, found cross-domain connections, and traced causal chains
through multiple gossip rounds.  The queen's job is to:

1. ORDER worker sections into a logical narrative flow
2. SMOOTH transitions between sections (connecting causal chains)
3. REMOVE redundancy where workers cover the same ground
4. PRESERVE worker voice, evidence, exact numbers, and connections
5. CHECK for cross-section contradictions that workers didn't resolve

Key principles:

- **Never rewrite**: Workers wrote connected analyses grounded in
  evidence.  The queen arranges and smooths, not rewrites.

- **No meta-commentary**: Never write about the document itself.
  No "How to Read This", no "This chapter covers", no "The reader
  will find".  Write the content directly.

- **Numerical integrity**: Workers have preserved exact numbers from
  the corpus.  The queen carries these through without rounding,
  converting, or paraphrasing.

- **Contradiction handling**: Workers have already reasoned through
  contradictions via gossip.  The queen respects their verdicts.
  If cross-section contradictions remain, flag them with
  [UNRESOLVED: conflicting evidence].

- **Serendipity integration**: Serendipity insights are woven into
  the causal flow at the points where they illuminate connections
  between specialist domains — never appended as a separate section.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Type alias for the async LLM completion function used throughout.
CompleteFn = Callable[[str], Awaitable[str]]

# Pattern to strip raw model reasoning tokens that should never appear in output
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _sanitize_output(text: str) -> str:
    """Strip raw model artifacts from queen output.

    Removes `<think>` reasoning tokens and leading/trailing whitespace.
    These tags indicate raw model internals leaking into the output.
    """
    cleaned = _THINK_TAG_RE.sub("", text)
    return cleaned.strip()


def _build_queen_prompt(
    date: str,
    n_workers: int,
    query: str,
    summaries_text: str,
    serendipity_block: str,
) -> str:
    """Build the queen merge prompt via concatenation (not .replace()).

    Concatenation prevents template injection: if worker summaries or the
    user query contain literal ``{placeholder}`` strings, they won't be
    re-scanned by a subsequent replacement pass.
    """
    serendipity_rule = ""
    if serendipity_block:
        serendipity_rule = (
            f"RULE 6 — SERENDIPITY INTEGRATION:\n"
            f"Weave serendipity insights at the causal points where they "
            f"illuminate cross-domain connections. Do NOT append them as a "
            f"separate section.\n\n"
        )

    return (
        f"You are the QUEEN EDITOR. Your job is to STITCH worker analyses "
        f"into a single coherent document. Workers have done the deep "
        f"reasoning and found the connections. You ORDER their sections, "
        f"SMOOTH transitions between them, REMOVE redundancy where workers "
        f"cover the same ground, and ensure the narrative flows as one "
        f"continuous piece. Today is: {date}\n\n"
        f"{n_workers} specialist workers have independently processed "
        f"different sections of a research corpus, then refined their "
        f"analyses through multiple gossip rounds where they found "
        f"cross-domain connections, traced causal chains, and resolved "
        f"contradictions with evidence.\n\n"
        f"USER QUERY: {query}\n\n"
        f"WORKER ANALYSES (post-gossip, with connections and reasoning):\n"
        f"{summaries_text}\n\n"
        f"{serendipity_block}"
        f"═══ EDITORIAL PROTOCOL ═══\n\n"
        f"RULE 0 — NEVER REWRITE WORKER CONTENT:\n"
        f"Workers wrote connected analyses grounded in evidence. Your job "
        f"is to ARRANGE and SMOOTH, not rewrite. Preserve their voice, "
        f"their specific examples, their exact numbers, their causal chains. "
        f"You are an EDITOR, not an author.\n\n"
        f"RULE 1 — NEVER WRITE ABOUT THE DOCUMENT ITSELF:\n"
        f"No 'How to Read This Book'. No 'This chapter covers'. No 'The "
        f"reader will find'. No meta-commentary about the text structure. "
        f"Write the CONTENT directly.\n\n"
        f"RULE 2 — NUMERICAL INTEGRITY:\n"
        f"Workers have preserved exact numbers from the corpus. Carry them "
        f"through EXACTLY. Never round, convert, or paraphrase. If workers "
        f"state different values for the same thing and provided verdicts, "
        f"use the verdict. If still conflicting, present both with evidence.\n\n"
        f"RULE 3 — REMOVE REDUNDANCY:\n"
        f"Where multiple workers cover the same ground, keep the version "
        f"with the strongest evidence chain. Don't repeat the same finding "
        f"from different workers.\n\n"
        f"RULE 4 — SMOOTH TRANSITIONS:\n"
        f"Add brief transition sentences between worker sections so the "
        f"document flows as one continuous narrative. These transitions "
        f"should connect the causal chains — 'This molecular mechanism "
        f"manifests in practice as...' — not just signal topic changes.\n\n"
        f"RULE 5 — INTERNAL CONSISTENCY:\n"
        f"Scan the stitched output for contradictions across worker "
        f"sections. If Worker A's section says X and Worker B's section "
        f"says not-X, and neither resolved it in gossip, flag it with "
        f"[UNRESOLVED: conflicting evidence].\n\n"
        f"{serendipity_rule}"
        f"RULE 7 — HIGHLIGHT CROSS-DOMAIN INSIGHTS:\n"
        f"The most valuable content in this document is where one domain "
        f"illuminated another — connections that no single specialist could "
        f"have found alone. When you encounter these (from worker analyses "
        f"or serendipity insights), give them PROMINENCE: place them at "
        f"transition points between sections, use them to open or close "
        f"major segments, and frame them as the key takeaways. The reader "
        f"should walk away remembering the 2-3 most surprising cross-domain "
        f"connections, not just a list of facts organized by topic.\n\n"
        f"Produce the stitched, edited document:"
    )


async def queen_merge(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn,
    serendipity_insights: str = "",
    max_summary_chars: int = 100000,
) -> str:
    """Merge all worker summaries + serendipity insights into final answer.

    Args:
        worker_summaries: Mapping of angle name -> refined summary text.
        query: The user's original research query.
        complete_fn: Async LLM completion callable.
        serendipity_insights: Cross-angle insights from the serendipity bridge.
        max_summary_chars: Maximum chars per worker summary in the prompt.

    Returns:
        Final synthesized answer string. Falls back to concatenation if
        the LLM call fails.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    summaries_text = ""
    for angle, summary in worker_summaries.items():
        summaries_text += (
            f"\n### Worker: {angle}\n"
            f"{summary[:max_summary_chars]}\n"
        )

    if serendipity_insights:
        serendipity_block = (
            "CROSS-ANGLE SERENDIPITY INSIGHTS (unexpected connections "
            "between specialist domains — integrate these into the "
            "appropriate sections):\n"
            f"{serendipity_insights}\n\n"
        )
    else:
        serendipity_block = ""

    prompt = _build_queen_prompt(
        date=date,
        n_workers=len(worker_summaries),
        query=query,
        summaries_text=summaries_text,
        serendipity_block=serendipity_block,
    )

    # Attempt queen merge with retry on suspiciously fast completion
    for attempt in range(2):
        t0 = time.monotonic()
        try:
            result = await complete_fn(prompt)
            elapsed = time.monotonic() - t0

            if elapsed < 2.0:
                logger.error(
                    "attempt=<%d>, elapsed_s=<%.3f> | queen merge completed suspiciously fast — "
                    "likely a crash or empty response, retrying",
                    attempt + 1, elapsed,
                )
                if attempt == 0:
                    continue
                # Second attempt also fast — fall through to fallback

            if result:
                result = _sanitize_output(result)

            if result and len(result) > 100:
                if "<think>" in result.lower():
                    logger.warning(
                        "queen output contains residual <think> tags after sanitization"
                    )
                logger.info(
                    "attempt=<%d>, elapsed_s=<%.1f>, output_chars=<%d> | queen merge succeeded",
                    attempt + 1, elapsed, len(result),
                )
                return result

            logger.warning(
                "attempt=<%d>, output_chars=<%d> | queen merge returned short/empty response",
                attempt + 1, len(result) if result else 0,
            )
            if attempt == 0:
                continue

        except Exception as exc:
            logger.error(
                "attempt=<%d>, error=<%s> | queen merge LLM call failed",
                attempt + 1, exc,
            )
            if attempt == 0:
                continue

    # Fallback: concatenate worker summaries with warning header
    logger.error(
        "queen merge failed after 2 attempts — falling back to concatenated worker summaries"
    )
    parts = [
        "# ⚠️ QUEEN MERGE FAILED — RAW WORKER OUTPUT\n\n"
        "The queen synthesis failed after 2 attempts. Below are the raw worker "
        "analyses concatenated without editorial integration. This output lacks "
        "cross-section stitching and narrative flow.\n\n---\n"
    ]
    for angle, summary in worker_summaries.items():
        parts.append(f"## {angle}\n{summary}")
    if serendipity_insights:
        parts.append(f"## Cross-Angle Insights\n{serendipity_insights}")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  DIFFUSION QUEEN — iterative report manifestation via swarm confrontation
# ═══════════════════════════════════════════════════════════════════════
#
# Instead of a single-shot queen merge, the report is built iteratively:
#
#   Step 0  Scaffold    — queen produces structural skeleton only
#   Step 1  Manifest    — each WORKER drafts their report section (parallel)
#   Step 2  Confront    — workers from OTHER angles critique sections (parallel)
#   Step 3  Correct     — section writers revise based on critique (parallel)
#   Step 4  Converge?   — check if sections stabilized (reuses gossip convergence)
#           (repeat Steps 2-4 until stable)
#   Step 5  Stitch      — queen does ONLY transitions + dedup (light pass)
#
# The swarm is the prime implementor: the same confrontation mechanism
# that produces depth in the analysis phase now produces depth in the
# report phase.


def _build_scaffold_prompt(
    date: str,
    query: str,
    angle_summaries: dict[str, str],
    serendipity_types: list[str],
) -> str:
    """Build the scaffold prompt — structural skeleton only.

    The scaffold defines section ordering, narrative arc, and which
    worker material maps where.  This is the coarsest representation
    of the report — analogous to the noisiest state in diffusion.

    Args:
        date: Current date string.
        query: The user's research query.
        angle_summaries: Mapping of angle → condensed summary (200-400 chars).
        serendipity_types: Labels of cross-domain connection types discovered.
    """
    angle_block = "\n".join(
        f"- **{angle}**: {summary}" for angle, summary in angle_summaries.items()
    )

    serendipity_block = ""
    if serendipity_types:
        serendipity_block = (
            f"\nCROSS-DOMAIN CONNECTIONS DISCOVERED:\n"
            + "\n".join(f"- {t}" for t in serendipity_types)
            + "\n"
        )

    return (
        f"You are a REPORT ARCHITECT. Your job is to design the STRUCTURE "
        f"of a research report — NOT to write the content. Workers will "
        f"write their own sections based on your scaffold. Today: {date}\n\n"
        f"USER QUERY: {query}\n\n"
        f"SPECIALIST ANGLES AND THEIR FINDINGS:\n{angle_block}\n"
        f"{serendipity_block}\n"
        f"Produce a SCAFFOLD — an ordered list of report sections. For each:\n"
        f"1. Section title\n"
        f"2. Which angle(s) contribute to this section\n"
        f"3. One sentence: what this section should accomplish in the narrative\n"
        f"4. Key cross-domain connections to weave in (if any)\n\n"
        f"RULES:\n"
        f"- Order sections for NARRATIVE FLOW, not alphabetical or by angle\n"
        f"- Group related angles into single sections where they share causal chains\n"
        f"- Place cross-domain surprises at transition points between sections\n"
        f"- The scaffold should read as a table of contents with intent annotations\n"
        f"- Do NOT write any actual content — only structure and intent\n\n"
        f"Produce the scaffold:"
    )


def _build_section_manifest_prompt(
    date: str,
    angle: str,
    worker_analysis: str,
    scaffold: str,
    adjacent_sections: str,
    query: str,
    pass_number: int,
) -> str:
    """Build the section manifestation prompt — worker drafts their section.

    Each worker writes their own report section, grounded in their
    gossip-refined analysis.  On pass 1, they have no adjacent sections.
    On subsequent passes, they see the surrounding context.

    Args:
        date: Current date string.
        angle: The worker's specialist angle.
        worker_analysis: Full gossip-refined analysis from this worker.
        scaffold: Structural skeleton from Step 0.
        adjacent_sections: Drafts from neighboring sections (empty on pass 1).
        query: The user's research query.
        pass_number: Which diffusion pass (1-indexed).
    """
    context_block = ""
    if adjacent_sections:
        context_block = (
            f"\nADJACENT SECTIONS (for context — connect your section to these):\n"
            f"{adjacent_sections}\n"
        )

    revision_note = ""
    if pass_number > 1:
        revision_note = (
            f"\nThis is revision pass {pass_number}. You have received confrontation "
            f"feedback from other specialists. Your corrections have been applied — "
            f"now re-draft your section incorporating those corrections while "
            f"maintaining narrative flow with adjacent sections.\n"
        )

    return (
        f"You are a specialist writer for the **{angle}** section of a "
        f"research report. Your job is to transform your analysis into "
        f"a polished report section. Today: {date}\n\n"
        f"USER QUERY: {query}\n\n"
        f"REPORT SCAFFOLD (your section's place in the narrative):\n"
        f"{scaffold}\n\n"
        f"YOUR FULL ANALYSIS (gossip-refined, evidence-grounded):\n"
        f"{worker_analysis}\n"
        f"{context_block}"
        f"{revision_note}\n"
        f"RULES:\n"
        f"1. Write ONLY your section — do not attempt the full report\n"
        f"2. Preserve YOUR voice, YOUR evidence chains, YOUR exact numbers\n"
        f"3. Trace causal chains: A causes B because C (mechanism), which predicts D\n"
        f"4. Include specific values: doses, concentrations, timelines, study details\n"
        f"5. Where your evidence connects to other domains, make the connection "
        f"explicit with mechanism\n"
        f"6. Flag uncertainty: mark claims as [ESTABLISHED], [PROBABLE], "
        f"[SPECULATIVE], or [UNKNOWN]\n"
        f"7. No meta-commentary about the document. Write content directly.\n"
        f"8. Depth over brevity — every evidence chain matters\n\n"
        f"Write your section:"
    )


def _build_confrontation_prompt(
    date: str,
    reviewer_angle: str,
    reviewer_analysis: str,
    section_draft: str,
    section_angle: str,
    scaffold: str,
    query: str,
) -> str:
    """Build the confrontation prompt — reviewer critiques a section.

    Workers from OTHER angles review each section draft against their
    own evidence base.  This is the diffusion "score function" — it
    identifies what's wrong with the current state.

    Args:
        date: Current date string.
        reviewer_angle: The reviewing worker's specialist angle.
        reviewer_analysis: The reviewer's own gossip-refined analysis.
        section_draft: The section being reviewed.
        section_angle: Which angle the section covers.
        scaffold: Full report structure for context.
        query: The user's research query.
    """
    return (
        f"You are a **{reviewer_angle}** specialist reviewing the "
        f"**{section_angle}** section of a research report. Your job is "
        f"to confront this section with YOUR evidence. Today: {date}\n\n"
        f"USER QUERY: {query}\n\n"
        f"REPORT SCAFFOLD:\n{scaffold}\n\n"
        f"SECTION UNDER REVIEW ({section_angle}):\n{section_draft}\n\n"
        f"YOUR OWN ANALYSIS ({reviewer_angle} — your evidence base):\n"
        f"{reviewer_analysis}\n\n"
        f"REVIEW THIS SECTION. For each issue found, output one of these tags:\n\n"
        f"CONTRADICTION: [section claims X] vs [your evidence shows Y] — "
        f"[why this matters]\n"
        f"MISSING_CONNECTION: [your analysis found A↔B which this section "
        f"should reference] — [the mechanism]\n"
        f"NUMERICAL_ERROR: [section says X] vs [your source says Y] — [source]\n"
        f"DEPTH_GAP: [section mentions Z superficially] — [your data has the "
        f"full mechanism: ...]\n"
        f"REDUNDANCY: [this repeats what section N covers more thoroughly]\n"
        f"UNSUPPORTED: [section claims X without evidence] — [your assessment "
        f"of whether this is supportable]\n\n"
        f"RULES:\n"
        f"- Only flag issues you can SUPPORT from your own evidence\n"
        f"- Be specific: cite the exact claim and your exact counter-evidence\n"
        f"- If the section is accurate from your perspective, say CONFIRMED: "
        f"[what you verified and why]\n"
        f"- Do not critique writing style — only factual accuracy, depth, "
        f"and cross-domain connections\n\n"
        f"Your review:"
    )


def _build_correction_prompt(
    date: str,
    angle: str,
    current_draft: str,
    critiques: list[str],
    adjacent_sections: str,
    worker_analysis: str,
    query: str,
    pass_number: int,
) -> str:
    """Build the correction prompt — writer revises based on confrontation.

    This is the diffusion "update step" — the section writer incorporates
    feedback from reviewers to fix errors and deepen connections.

    Args:
        date: Current date string.
        angle: The section writer's specialist angle.
        current_draft: The writer's current section draft.
        critiques: All confrontation feedback for this section.
        adjacent_sections: Surrounding sections for context.
        worker_analysis: Original evidence base.
        query: The user's research query.
        pass_number: Which diffusion pass.
    """
    critique_block = "\n\n".join(
        f"--- Reviewer {i + 1} ---\n{c}" for i, c in enumerate(critiques)
    )

    return (
        f"You are revising the **{angle}** section based on confrontation "
        f"feedback from other specialists. Pass {pass_number}. Today: {date}\n\n"
        f"USER QUERY: {query}\n\n"
        f"YOUR CURRENT DRAFT:\n{current_draft}\n\n"
        f"CONFRONTATION FEEDBACK FROM OTHER SPECIALISTS:\n{critique_block}\n\n"
        f"YOUR ORIGINAL ANALYSIS (evidence base):\n{worker_analysis}\n\n"
        f"ADJACENT SECTIONS (for context):\n{adjacent_sections}\n\n"
        f"REVISION RULES:\n"
        f"1. Address every CONTRADICTION by checking your evidence — if the "
        f"reviewer is right, correct your section. If they're wrong, add a "
        f"[DISPUTED: ...] note with your counter-evidence.\n"
        f"2. Incorporate MISSING_CONNECTIONs where they add genuine depth — "
        f"don't add connections that are tangential.\n"
        f"3. Fix all NUMERICAL_ERRORs immediately.\n"
        f"4. Expand DEPTH_GAPs with the full mechanism from your analysis.\n"
        f"5. Remove REDUNDANCY if another section covers it better.\n"
        f"6. For UNSUPPORTED claims: either add evidence or remove the claim.\n"
        f"7. Preserve everything that reviewers CONFIRMED.\n"
        f"8. Maintain your specialist voice and evidence chains.\n\n"
        f"Produce your revised section:"
    )


def _build_light_stitch_prompt(
    date: str,
    query: str,
    sections: dict[str, str],
    scaffold: str,
    serendipity_insights: str,
) -> str:
    """Build the final stitch prompt — transitions and dedup only.

    The heavy lifting (depth, accuracy, cross-referencing) is done by
    the confrontation loop.  The queen just polishes transitions.

    Args:
        date: Current date string.
        query: The user's research query.
        sections: Mapping of angle → converged section text.
        scaffold: Original structural skeleton.
        serendipity_insights: Cross-angle insights from serendipity bridge.
    """
    sections_text = ""
    for angle, text in sections.items():
        sections_text += f"\n### {angle}\n{text}\n"

    serendipity_block = ""
    if serendipity_insights:
        serendipity_block = (
            f"\nCROSS-ANGLE SERENDIPITY INSIGHTS (weave these into transitions "
            f"where they illuminate connections between sections):\n"
            f"{serendipity_insights}\n"
        )

    return (
        f"You are the FINAL EDITOR. The sections below have been written by "
        f"specialist workers and refined through multiple rounds of "
        f"cross-specialist confrontation. Every claim has been reviewed by "
        f"experts from other domains. Your job is LIGHT: Today: {date}\n\n"
        f"USER QUERY: {query}\n\n"
        f"REPORT SCAFFOLD (intended structure):\n{scaffold}\n\n"
        f"CONVERGED SECTIONS (confrontation-verified):\n{sections_text}\n"
        f"{serendipity_block}\n"
        f"YOUR JOB (editorial only — do NOT rewrite content):\n"
        f"1. ORDER sections according to the scaffold's narrative flow\n"
        f"2. ADD brief transition sentences between sections that connect "
        f"causal chains across domains\n"
        f"3. REMOVE any remaining redundancy between sections\n"
        f"4. WEAVE serendipity insights at causal transition points\n"
        f"5. Ensure [ESTABLISHED]/[PROBABLE]/[SPECULATIVE]/[UNKNOWN] tags "
        f"are consistent across sections\n\n"
        f"RULES:\n"
        f"- NEVER rewrite worker content — their voice, numbers, and evidence "
        f"chains are the product of multi-round confrontation\n"
        f"- NO meta-commentary about the document itself\n"
        f"- Transitions should connect mechanisms across domains, not just "
        f"signal topic changes\n"
        f"- The reader should walk away remembering the 2-3 most surprising "
        f"cross-domain connections\n\n"
        f"Produce the final stitched report:"
    )


def _condense_summary(summary: str, max_chars: int = 400) -> str:
    """Condense a worker summary to a short overview for scaffold generation.

    Extracts the first few sentences up to max_chars, preferring sentence
    boundaries over hard truncation.
    """
    if len(summary) <= max_chars:
        return summary

    # Try to cut at a sentence boundary
    truncated = summary[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        return truncated[: last_period + 1]
    return truncated + "..."


def _get_adjacent_sections(
    angle: str,
    sections: dict[str, str],
    max_chars: int = 8000,
) -> str:
    """Get adjacent section drafts for context during manifestation/correction.

    Returns the sections immediately before and after the given angle
    in the section ordering, truncated to fit context.

    Args:
        angle: The current section's angle.
        sections: All current section drafts.
        max_chars: Maximum chars for adjacent context.
    """
    angles = list(sections.keys())
    if angle not in angles:
        return ""

    idx = angles.index(angle)
    adjacent_parts = []

    # Previous section
    if idx > 0:
        prev_angle = angles[idx - 1]
        prev_text = sections[prev_angle][:max_chars // 2]
        adjacent_parts.append(f"[PRECEDING: {prev_angle}]\n{prev_text}")

    # Next section
    if idx < len(angles) - 1:
        next_angle = angles[idx + 1]
        next_text = sections[next_angle][:max_chars // 2]
        adjacent_parts.append(f"[FOLLOWING: {next_angle}]\n{next_text}")

    return "\n\n".join(adjacent_parts)


async def generate_scaffold(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn: CompleteFn,
    serendipity_insights: str = "",
) -> str:
    """Step 0: Generate structural skeleton for the report.

    Produces a scaffold defining section ordering, narrative arc, and
    which worker material maps where.

    Args:
        worker_summaries: Mapping of angle → full gossip-refined analysis.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        serendipity_insights: Cross-angle insights for structure planning.

    Returns:
        Scaffold string (ordered section list with intent annotations).
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Condense summaries for the scaffold prompt (400 chars each)
    angle_summaries = {
        angle: _condense_summary(summary)
        for angle, summary in worker_summaries.items()
    }

    # Extract serendipity connection types (first line of each insight)
    serendipity_types: list[str] = []
    if serendipity_insights:
        for line in serendipity_insights.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("---"):
                serendipity_types.append(stripped[:200])
                if len(serendipity_types) >= 10:
                    break

    prompt = _build_scaffold_prompt(
        date=date,
        query=query,
        angle_summaries=angle_summaries,
        serendipity_types=serendipity_types,
    )

    t0 = time.monotonic()
    scaffold = await complete_fn(prompt)
    elapsed = time.monotonic() - t0

    if scaffold:
        scaffold = _sanitize_output(scaffold)

    logger.info(
        "scaffold_chars=<%d>, elapsed_s=<%.1f> | scaffold generated",
        len(scaffold) if scaffold else 0, elapsed,
    )

    return scaffold or ""


async def manifest_sections(
    worker_summaries: dict[str, str],
    scaffold: str,
    query: str,
    complete_fn: CompleteFn,
    current_sections: dict[str, str] | None = None,
    pass_number: int = 1,
) -> dict[str, str]:
    """Step 1: Workers draft their report sections (parallel).

    Each worker writes their own section based on their gossip-refined
    analysis, the scaffold, and adjacent sections (if available).

    Args:
        worker_summaries: Mapping of angle → full gossip-refined analysis.
        scaffold: Structural skeleton from Step 0.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        current_sections: Previous section drafts (None on first pass).
        pass_number: Which diffusion pass (1-indexed).

    Returns:
        Mapping of angle → section draft text.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _manifest_one(angle: str, analysis: str) -> tuple[str, str]:
        adjacent = ""
        if current_sections:
            adjacent = _get_adjacent_sections(angle, current_sections)

        prompt = _build_section_manifest_prompt(
            date=date,
            angle=angle,
            worker_analysis=analysis,
            scaffold=scaffold,
            adjacent_sections=adjacent,
            query=query,
            pass_number=pass_number,
        )

        result = await complete_fn(prompt)
        if result:
            result = _sanitize_output(result)
        return angle, result or ""

    # Build ordered task list so we can map exceptions back to angles
    angle_order = list(worker_summaries.keys())
    tasks = [
        _manifest_one(angle, worker_summaries[angle])
        for angle in angle_order
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    sections: dict[str, str] = {}
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            fallback_angle = angle_order[i]
            logger.error(
                "angle=<%s> | section manifestation failed: %s — using raw analysis as fallback",
                fallback_angle, r,
            )
            # Preserve the section using the worker's raw analysis
            sections[fallback_angle] = worker_summaries[fallback_angle]
            continue
        angle, text = r
        sections[angle] = text

    logger.info(
        "pass=<%d>, sections=<%d>, total_chars=<%d> | sections manifested",
        pass_number, len(sections), sum(len(t) for t in sections.values()),
    )

    return sections


async def confront_sections(
    sections: dict[str, str],
    worker_summaries: dict[str, str],
    scaffold: str,
    query: str,
    complete_fn: CompleteFn,
    reviewers_per_section: int = 3,
) -> dict[str, list[str]]:
    """Step 2: Cross-section confrontation (parallel).

    Each section draft is reviewed by K workers from OTHER angles.
    Uses diversity-aware selection to pick maximally disagreeing reviewers.

    Args:
        sections: Current section drafts (angle → text).
        worker_summaries: Full worker analyses (angle → text).
        scaffold: Report scaffold for context.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        reviewers_per_section: Number of cross-angle reviewers per section.

    Returns:
        Mapping of angle → list of critique strings.
    """
    from swarm.convergence import select_diverse_peers

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_angles = list(sections.keys())

    async def _confront_one(
        section_angle: str,
        reviewer_angle: str,
    ) -> tuple[str, str]:
        prompt = _build_confrontation_prompt(
            date=date,
            reviewer_angle=reviewer_angle,
            reviewer_analysis=worker_summaries.get(reviewer_angle, ""),
            section_draft=sections[section_angle],
            section_angle=section_angle,
            scaffold=scaffold,
            query=query,
        )

        result = await complete_fn(prompt)
        if result:
            result = _sanitize_output(result)
        return section_angle, result or ""

    # Build review assignments: for each section, pick K diverse reviewers
    tasks = []
    for section_angle in all_angles:
        # Other angles that can review this section
        other_angles = [a for a in all_angles if a != section_angle]
        if not other_angles:
            continue

        # Use diversity-aware selection: pick reviewers whose analyses
        # diverge most from the section draft
        section_text = sections[section_angle]
        other_analyses = [worker_summaries.get(a, "") for a in other_angles]

        diverse_analyses = select_diverse_peers(
            section_text, other_analyses, top_k=reviewers_per_section,
        )

        # Map back to angle names
        reviewer_angles = []
        for analysis in diverse_analyses:
            for a in other_angles:
                if worker_summaries.get(a, "") == analysis and a not in reviewer_angles:
                    reviewer_angles.append(a)
                    break

        for reviewer_angle in reviewer_angles:
            tasks.append(_confront_one(section_angle, reviewer_angle))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    critiques: dict[str, list[str]] = {angle: [] for angle in all_angles}
    for r in results:
        if isinstance(r, Exception):
            logger.error("confrontation failed: %s", r)
            continue
        section_angle, critique_text = r
        if critique_text:
            critiques[section_angle].append(critique_text)

    total_critiques = sum(len(c) for c in critiques.values())
    logger.info(
        "sections=<%d>, total_critiques=<%d> | confrontation complete",
        len(sections), total_critiques,
    )

    return critiques


async def correct_sections(
    sections: dict[str, str],
    critiques: dict[str, list[str]],
    worker_summaries: dict[str, str],
    scaffold: str,
    query: str,
    complete_fn: CompleteFn,
    pass_number: int,
) -> dict[str, str]:
    """Step 3: Section writers revise based on confrontation (parallel).

    Each section writer incorporates feedback from reviewers to fix
    errors and deepen cross-domain connections.

    Args:
        sections: Current section drafts (angle → text).
        critiques: Confrontation feedback per section (angle → list of critiques).
        worker_summaries: Full worker analyses (angle → text).
        scaffold: Report scaffold for context.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        pass_number: Which diffusion pass.

    Returns:
        Mapping of angle → revised section text.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _correct_one(angle: str) -> tuple[str, str]:
        section_critiques = critiques.get(angle, [])

        # If no critiques, section is confirmed — keep as-is
        if not section_critiques:
            logger.info(
                "angle=<%s>, pass=<%d> | no critiques — section confirmed",
                angle, pass_number,
            )
            return angle, sections[angle]

        adjacent = _get_adjacent_sections(angle, sections)

        prompt = _build_correction_prompt(
            date=date,
            angle=angle,
            current_draft=sections[angle],
            critiques=section_critiques,
            adjacent_sections=adjacent,
            worker_analysis=worker_summaries.get(angle, ""),
            query=query,
            pass_number=pass_number,
        )

        result = await complete_fn(prompt)
        if result:
            result = _sanitize_output(result)
        return angle, result or sections[angle]

    # Build ordered task list so we can map exceptions back to angles
    angle_order = list(sections.keys())
    tasks = [_correct_one(angle) for angle in angle_order]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    corrected: dict[str, str] = {}
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            fallback_angle = angle_order[i]
            logger.error(
                "angle=<%s> | correction failed: %s — preserving previous draft",
                fallback_angle, r,
            )
            # Preserve the previous draft so the section isn't lost
            corrected[fallback_angle] = sections[fallback_angle]
            continue
        angle, text = r
        corrected[angle] = text

    logger.info(
        "pass=<%d>, sections=<%d>, total_chars=<%d> | sections corrected",
        pass_number, len(corrected), sum(len(t) for t in corrected.values()),
    )

    return corrected


async def light_stitch(
    sections: dict[str, str],
    scaffold: str,
    query: str,
    complete_fn: CompleteFn,
    serendipity_insights: str = "",
) -> str:
    """Step 5: Final editorial stitch — transitions and dedup only.

    All the hard work (depth, accuracy, cross-referencing) is done by
    the confrontation loop.  The queen just polishes transitions.

    Args:
        sections: Converged section drafts (angle → text).
        scaffold: Report scaffold for section ordering.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        serendipity_insights: Cross-angle insights to weave in.

    Returns:
        Final stitched report string.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    prompt = _build_light_stitch_prompt(
        date=date,
        query=query,
        sections=sections,
        scaffold=scaffold,
        serendipity_insights=serendipity_insights,
    )

    t0 = time.monotonic()
    result = await complete_fn(prompt)
    elapsed = time.monotonic() - t0

    if result:
        result = _sanitize_output(result)

    if not result or len(result) < 100:
        logger.warning(
            "stitch failed or too short (%d chars) — concatenating sections as fallback",
            len(result) if result else 0,
        )
        parts = []
        for angle, text in sections.items():
            parts.append(f"## {angle}\n\n{text}")
        return "\n\n---\n\n".join(parts)

    logger.info(
        "elapsed_s=<%.1f>, output_chars=<%d> | final stitch complete",
        elapsed, len(result),
    )

    return result


async def diffusion_queen_merge(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn: CompleteFn,
    serendipity_insights: str = "",
    max_passes: int = 3,
    convergence_threshold: float = 0.85,
    reviewers_per_section: int = 3,
) -> tuple[str, int]:
    """Full diffusion queen pipeline — iterative report manifestation.

    Replaces the single-shot queen_merge() with an iterative process:
    workers draft sections, confront each other's drafts, correct,
    and converge — applying the same gossip mechanism to the report
    itself.

    Args:
        worker_summaries: Mapping of angle → gossip-refined analysis.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        serendipity_insights: Cross-angle insights from serendipity bridge.
        max_passes: Maximum confrontation passes before forced convergence.
        convergence_threshold: Jaccard threshold for section stability.
        reviewers_per_section: Number of cross-angle reviewers per section.

    Returns:
        Tuple of (final report string, actual LLM calls made).
    """
    from swarm.convergence import check_convergence

    t0 = time.monotonic()
    llm_calls = 0

    # Step 0: Scaffold
    logger.info("diffusion_pass=<scaffold> | generating report structure")
    scaffold = await generate_scaffold(
        worker_summaries, query, complete_fn, serendipity_insights,
    )
    llm_calls += 1  # scaffold generation

    if not scaffold:
        logger.warning(
            "scaffold generation failed — falling back to single-shot queen_merge"
        )
        fallback = await queen_merge(
            worker_summaries, query, complete_fn, serendipity_insights,
        )
        return fallback, llm_calls + 1

    sections: dict[str, str] | None = None
    converged = False

    for pass_num in range(1, max_passes + 1):
        prev_sections = sections

        # Step 1: Manifest sections
        logger.info(
            "diffusion_pass=<%d>/%d, phase=<manifest> | workers drafting sections",
            pass_num, max_passes,
        )
        sections = await manifest_sections(
            worker_summaries, scaffold, query, complete_fn,
            current_sections=prev_sections,
            pass_number=pass_num,
        )
        llm_calls += len(worker_summaries)  # one call per section

        if not sections:
            logger.error(
                "diffusion_pass=<%d> | section manifestation produced no sections",
                pass_num,
            )
            break

        # Step 2: Confront
        logger.info(
            "diffusion_pass=<%d>/%d, phase=<confront> | cross-section confrontation",
            pass_num, max_passes,
        )
        critiques = await confront_sections(
            sections, worker_summaries, scaffold, query, complete_fn,
            reviewers_per_section=reviewers_per_section,
        )
        # Count actual confrontation calls (reviewers assigned, not worst-case)
        llm_calls += sum(len(c) for c in critiques.values())

        # Step 3: Correct
        logger.info(
            "diffusion_pass=<%d>/%d, phase=<correct> | applying confrontation feedback",
            pass_num, max_passes,
        )
        # Count sections that actually need correction (have critiques)
        sections_needing_correction = sum(
            1 for c in critiques.values() if c
        )
        sections = await correct_sections(
            sections, critiques, worker_summaries, scaffold, query,
            complete_fn, pass_number=pass_num,
        )
        llm_calls += sections_needing_correction

        # Step 4: Convergence check
        if prev_sections is not None and sections:
            current_texts = list(sections.values())
            previous_texts = list(prev_sections.values())

            if check_convergence(
                current_texts, previous_texts, convergence_threshold,
            ):
                logger.info(
                    "diffusion_pass=<%d>/%d | CONVERGED — sections stabilized",
                    pass_num, max_passes,
                )
                converged = True
                break

            logger.info(
                "diffusion_pass=<%d>/%d | not converged — continuing refinement",
                pass_num, max_passes,
            )

    elapsed = time.monotonic() - t0

    if not sections:
        logger.error(
            "diffusion queen produced no sections — falling back to single-shot"
        )
        fallback = await queen_merge(
            worker_summaries, query, complete_fn, serendipity_insights,
        )
        return fallback, llm_calls + 1

    # Step 5: Final stitch
    logger.info(
        "diffusion_pass=<stitch>, converged=<%s>, passes=<%d>, elapsed_s=<%.1f> | "
        "final editorial stitching",
        converged, max_passes, elapsed,
    )
    report = await light_stitch(
        sections, scaffold, query, complete_fn, serendipity_insights,
    )
    llm_calls += 1  # stitch

    total_elapsed = time.monotonic() - t0
    logger.info(
        "total_elapsed_s=<%.1f>, report_chars=<%d>, passes_used=<%d>, "
        "converged=<%s>, llm_calls=<%d> | diffusion queen merge complete",
        total_elapsed, len(report), max_passes, converged, llm_calls,
    )

    return report, llm_calls


def _build_knowledge_exec_summary_prompt(
    date: str,
    n_workers: int,
    query: str,
    summaries_text: str,
    serendipity_block: str,
) -> str:
    """Build the prompt for the knowledge report executive summary."""
    return (
        f"You are a senior research editor preparing the executive summary for a "
        f"comprehensive knowledge report. Today is: {date}\n\n"
        f"A research swarm of {n_workers} specialist workers has produced detailed "
        f"findings organized by angle. The full findings will follow this summary "
        f"in the final document. Your job is to write ONLY the executive summary "
        f"and cross-reference matrix that will sit at the TOP of the report.\n\n"
        f"USER QUERY: {query}\n\n"
        f"WORKER FINDINGS (condensed for overview):\n"
        f"{summaries_text}\n\n"
        f"{serendipity_block}"
        f"PRODUCE EXACTLY:\n\n"
        f"1. An EXECUTIVE SUMMARY (800-1500 words) that:\n"
        f"   - Opens with the single most important finding or resolution\n"
        f"   - States the core findings ranked by EVIDENCE STRENGTH:\n"
        f"     • CONSENSUS findings (3+ workers agree) first\n"
        f"     • CORROBORATED findings (2 workers agree) second\n"
        f"     • NOVEL single-source findings third\n"
        f"   - Identifies the most important cross-angle connections\n"
        f"   - Highlights key contradictions WITH assessment of which "
        f"position has stronger evidence (do not just list them neutrally)\n"
        f"   - Does NOT repeat specific data points — those are in the full "
        f"sections below\n"
        f"   - Reads as a standalone overview for someone who may not read "
        f"the full report\n\n"
        f"2. A CROSS-REFERENCE MATRIX (markdown table) showing which angles "
        f"share findings, contradict each other, or have complementary insights. "
        f"Use symbols: ✓ (agreement), ✗ (contradiction), ↔ (complementary), "
        f"— (no overlap).\n\n"
        f"3. A KEY FINDINGS list (5-10 items) — the most important individual "
        f"discoveries across all angles. Each must include:\n"
        f"   - The finding itself\n"
        f"   - Evidence strength indicator (consensus/corroborated/single-source)\n"
        f"   - Which angle(s) contributed it\n\n"
        f"Do NOT include disclaimers, safety warnings, or moral commentary. "
        f"Do NOT summarize the full worker findings — those will appear in full "
        f"after your summary. Focus on the META-VIEW: what patterns emerge when "
        f"you look across all angles together."
    )


async def build_knowledge_report(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn,
    serendipity_insights: str = "",
    corpus_chars: int = 0,
    gossip_rounds: int = 0,
    converged_early: bool = False,
) -> str:
    """Build the full knowledge report — arbitrary length, preserves all findings.

    The knowledge report has three parts:
    1. LLM-generated executive summary + cross-reference matrix + key findings
    2. Full worker findings by angle (no truncation — every detail preserved)
    3. Serendipity insights (if available)
    4. Methodology appendix

    Args:
        worker_summaries: Mapping of angle name -> full gossip-refined summary.
        query: The user's original research query.
        complete_fn: Async LLM completion callable.
        serendipity_insights: Cross-angle insights from the serendipity bridge.
        corpus_chars: Total source corpus size for metadata.
        gossip_rounds: Number of gossip rounds executed.
        converged_early: Whether gossip converged early.

    Returns:
        Full knowledge report as markdown string.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build condensed summaries text for the exec summary prompt
    condensed = ""
    for angle, summary in worker_summaries.items():
        condensed += f"\n### {angle}\n{summary[:4000]}\n"

    serendipity_block = ""
    if serendipity_insights:
        serendipity_block = (
            "CROSS-ANGLE SERENDIPITY INSIGHTS:\n"
            f"{serendipity_insights}\n\n"
        )

    # Generate executive summary via LLM
    prompt = _build_knowledge_exec_summary_prompt(
        date=date,
        n_workers=len(worker_summaries),
        query=query,
        summaries_text=condensed,
        serendipity_block=serendipity_block,
    )

    fallback_summary = (
        "## Executive Summary\n\n"
        "*Executive summary generation failed. "
        "See full findings below for complete analysis.*"
    )
    try:
        exec_summary = await complete_fn(prompt)
        if exec_summary:
            exec_summary = _sanitize_output(exec_summary)
        if not exec_summary or len(exec_summary) < 50:
            logger.warning(
                "knowledge report exec summary returned short/empty response (%d chars), using fallback",
                len(exec_summary) if exec_summary else 0,
            )
            exec_summary = fallback_summary
    except Exception as exc:
        logger.warning("knowledge report exec summary failed: %s", exc)
        exec_summary = fallback_summary

    # Derive a clean title from the query (first sentence)
    title_text = query.split(".")[0].split("?")[0].split("\n")[0].strip()

    # Assemble the full knowledge report
    parts = [
        f"# Knowledge Report: {title_text}",
        f"*Generated: {date} | {len(worker_summaries)} specialist angles | "
        f"{corpus_chars:,} chars analyzed | {gossip_rounds} gossip round(s)"
        f"{' (converged early)' if converged_early else ''}*",
        "",
        "---",
        "",
        exec_summary,
        "",
        "---",
        "",
        "# Detailed Findings by Angle",
        "",
    ]

    # Full worker findings — NO truncation
    for angle, summary in worker_summaries.items():
        parts.append(f"## {angle}")
        parts.append("")
        parts.append(summary)
        parts.append("")

    # Serendipity section
    if serendipity_insights:
        parts.append("---")
        parts.append("")
        parts.append("# Cross-Angle Serendipity Insights")
        parts.append("")
        parts.append(
            "*These connections were identified by a polymath connector that "
            "read all specialist summaries and looked for unexpected patterns "
            "between domains.*"
        )
        parts.append("")
        parts.append(serendipity_insights)
        parts.append("")

    # Methodology appendix
    parts.append("---")
    parts.append("")
    parts.append("# Methodology")
    parts.append("")
    parts.append(
        f"This report was generated by a gossip swarm of "
        f"{len(worker_summaries)} specialist workers. Each worker was assigned "
        f"a topical angle of the source corpus ({corpus_chars:,} chars total). "
        f"Workers first synthesized their section independently, then participated "
        f"in {gossip_rounds} round(s) of peer gossip where each worker read all "
        f"peers' summaries and refined their own analysis with cross-references. "
    )
    if serendipity_insights:
        parts.append(
            "A polymath serendipity bridge then scanned all refined summaries "
            "for unexpected cross-angle connections."
        )
    parts.append("")
    parts.append("**Angles analyzed:**")
    for angle in worker_summaries:
        parts.append(f"- {angle}")

    return "\n".join(parts)
