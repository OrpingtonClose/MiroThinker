# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Queen merge — combines all worker analyses into the final synthesis.

The queen is a senior synthesis editor whose primary obligation is
factual integrity.  It receives worker ANALYSES (not summaries) that
contain reasoned evaluations, source quotes, and contradiction verdicts.
The queen's job is to produce a narrative that preserves the workers'
verified facts while being engaging and deeply informative.

Key principles:

- **Mixture-of-Agents** (ICLR 2025, arXiv 2406.04692): The queen acts
  as the final aggregation layer reading all previous-layer outputs.
  It must produce a response strictly better than any individual input.

- **Evidence hierarchy**: The queen evaluates claims by source quality,
  cross-worker agreement, and specificity.  Consensus across 3+ workers
  is near-certain; single-worker claims are flagged for confidence.

- **Numerical integrity**: Workers have preserved exact numbers from
  the corpus.  The queen MUST carry these numbers through to the final
  output without rounding, converting, or paraphrasing.  Daily totals
  must not become per-meal values.  Per-dose values must not become
  daily totals.

- **Contradiction resolution protocol**: Workers have already reasoned
  through contradictions and provided verdicts.  The queen respects
  these verdicts unless it can identify a clear error in the worker's
  reasoning.  It does not re-introduce contradictions that workers
  resolved.

- **Internal consistency**: Before finalizing, the queen checks that
  no two paragraphs in its output contradict each other.

- **Narrative integration**: Serendipity insights are not appended but
  woven into the causal flow at the points where they illuminate
  connections between specialist domains.

The queen sees ~6-7K tokens instead of the full corpus (which may be 20K+).
This is a 3x context reduction while retaining all cross-referenced
analytical depth from the gossip rounds.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _build_queen_prompt(
    date: str,
    n_workers: int,
    query: str,
    summaries_text: str,
    serendipity_block: str,
    readability_frame: str = "",
) -> str:
    """Build the queen merge prompt via concatenation (not .replace()).

    Concatenation prevents template injection: if worker summaries or the
    user query contain literal ``{placeholder}`` strings, they won't be
    re-scanned by a subsequent replacement pass.
    """
    return (
        f"You are the QUEEN SYNTHESIZER — a senior research editor whose output "
        f"must be strictly superior to any individual worker's analysis, both in "
        f"narrative quality AND factual integrity. Today is: {date}\n\n"
        f"{n_workers} specialist workers have independently processed different "
        f"sections of a research corpus, then refined their analyses through "
        f"multiple rounds of peer gossip where each worker cross-examined "
        f"findings, resolved contradictions with evidence-based verdicts, and "
        f"verified numerical claims against source text.\n\n"
        f"USER QUERY: {query}\n\n"
        f"WORKER ANALYSES (post-gossip, with reasoning and verdicts):\n"
        f"{summaries_text}\n\n"
        f"{serendipity_block}"
        f"═══ SYNTHESIS PROTOCOL ═══\n\n"
        f"PHASE 0 — NUMERICAL INTEGRITY (do this BEFORE writing):\n"
        f"Scan all worker analyses for specific numbers (dosages, timings, "
        f"ratios, macro values, frequencies). For each number:\n"
        f"  • Note the exact value as the worker stated it\n"
        f"  • Check if any other worker states a DIFFERENT value for the same "
        f"thing\n"
        f"  • If workers resolved the conflict with a verdict, use the verdict\n"
        f"  • If still conflicting, present BOTH values with their evidence\n"
        f"  • NEVER invent, round, or convert numbers. If a worker says "
        f"'400g protein daily across 7-8 meals', you write exactly that — "
        f"do NOT write '400g protein per meal'\n"
        f"  • ALWAYS distinguish daily totals from per-dose/per-meal values\n\n"
        f"PHASE 1 — EVIDENCE HIERARCHY:\n"
        f"Classify every claim by confidence:\n"
        f"  • CONSENSUS (3+ workers agree): Near-certain. State directly.\n"
        f"  • CORROBORATED (2 workers agree): High confidence. Note the agreement.\n"
        f"  • SINGLE-SOURCE (1 worker only): Flag confidence level explicitly.\n"
        f"  • CONTRADICTED (workers disagree): Use worker verdicts or apply "
        f"contradiction protocol below.\n\n"
        f"PHASE 2 — CONTRADICTION RESOLUTION:\n"
        f"Workers have already reasoned through most contradictions. Respect "
        f"their verdicts unless you can identify a clear error in their reasoning. "
        f"For any remaining conflicts:\n"
        f"  a) Identify WHAT exactly they disagree about\n"
        f"  b) Evaluate the evidence QUALITY behind each position\n"
        f"  c) Determine if the disagreement is real (different conclusions from "
        f"same data) or apparent (different aspects of the same phenomenon)\n"
        f"  d) Either RESOLVE with reasoning, or PRESERVE both positions with "
        f"explicit evidence assessment for each\n\n"
        f"PHASE 3 — INTERNAL CONSISTENCY CHECK:\n"
        f"Before finalizing your output, scan it for self-contradictions. If "
        f"paragraph A says 'administer GH and insulin simultaneously' and "
        f"paragraph B says 'wait 60-90 minutes between GH and insulin', you "
        f"have an internal contradiction. Fix it. If a case study section says "
        f"'activating all three arms' and then says 'missing two arms', that is "
        f"an internal contradiction. Fix it. Your output must be internally "
        f"consistent at every point.\n\n"
        f"PHASE 4 — NARRATIVE SYNTHESIS:\n"
        f"  1. Structure with clear headings building a causal argument.\n"
        f"  2. WEAVE findings into connected explanations — show how mechanism A "
        f"leads to consequence B which interacts with pathway C.\n"
        f"  3. Cite sources with URLs where available.\n"
        f"  4. Every sentence must deliver information. No filler, no disclaimers.\n"
        f"  5. Keep speculative findings, clearly labeled as hypotheses.\n"
        f"  6. Do NOT add unsolicited warnings or ethical disclaimers.\n"
        f"  7. Mark weak evidence with [NEEDS VERIFICATION].\n"
        f"  8. Do NOT produce bullet-point lists or data dumps.\n"
        f"  9. If serendipity insights are provided, integrate them at the "
        f"causal points where they illuminate cross-domain connections — do NOT "
        f"append them as a separate section.\n"
        f"  10. Identify at least one EMERGENT INSIGHT that exists in the "
        f"combined evidence but was not explicitly stated by any individual worker.\n\n"
        f"{readability_frame + chr(10) * 2 if readability_frame else ''}"
        f"AIM FOR 3000-6000 WORDS. Be comprehensive but not redundant. "
        f"Merge overlapping findings — do not repeat the same point from different "
        f"workers. Your synthesis must be worth more than the sum of its parts.\n\n"
        f"Produce the final comprehensive synthesis:"
    )


async def queen_merge(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn,
    serendipity_insights: str = "",
    max_summary_chars: int = 6000,
    readability_frame: str = "",
) -> str:
    """Merge all worker summaries + serendipity insights into final answer.

    Args:
        worker_summaries: Mapping of angle name -> refined summary text.
        query: The user's original research query.
        complete_fn: Async LLM completion callable.
        serendipity_insights: Cross-angle insights from the serendipity bridge.
        max_summary_chars: Maximum chars per worker summary in the prompt.
        readability_frame: Optional readability instructions for the queen.

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
        readability_frame=readability_frame,
    )

    try:
        result = await complete_fn(prompt)
        if result and len(result.strip()) > 100:
            return result
        logger.warning(
            "queen merge returned short/empty response (%d chars), falling back to concatenation",
            len(result.strip()) if result else 0,
        )
    except Exception as exc:
        logger.warning("queen merge LLM call failed: %s, falling back to concatenation", exc)

    # Fallback: concatenate worker summaries
    parts = []
    for angle, summary in worker_summaries.items():
        parts.append(f"## {angle}\n{summary}")
    if serendipity_insights:
        parts.append(f"## Cross-Angle Insights\n{serendipity_insights}")
    return "\n\n".join(parts)


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
        if not exec_summary or len(exec_summary.strip()) < 50:
            logger.warning(
                "knowledge report exec summary returned short/empty response (%d chars), using fallback",
                len(exec_summary.strip()) if exec_summary else 0,
            )
            exec_summary = fallback_summary
    except Exception as exc:
        logger.warning("knowledge report exec summary failed: %s", exc)
        exec_summary = fallback_summary

    # Derive a clean title from the query (first sentence, capped at 100 chars on word boundary)
    title_text = query.split(".")[0].split("?")[0].split("\n")[0].strip()
    if len(title_text) > 100:
        # Cut on word boundary
        title_text = title_text[:100].rsplit(" ", 1)[0] + "..."

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
