# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Queen merge — combines all worker summaries into the final synthesis.

The queen sees ~6-7K tokens instead of the full corpus (which may be 20K+).
This is a 3x context reduction while retaining all cross-referenced
analytical depth from the gossip rounds.

If serendipity insights are available, they are injected into the merge
prompt as a dedicated section so the queen can weave them into the narrative.
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
) -> str:
    """Build the queen merge prompt via concatenation (not .replace()).

    Concatenation prevents template injection: if worker summaries or the
    user query contain literal ``{placeholder}`` strings, they won't be
    re-scanned by a subsequent replacement pass.
    """
    return (
        f"You are the queen synthesizer in a research swarm. Today is: {date}\n\n"
        f"{n_workers} specialist workers have independently processed different "
        f"sections of a research corpus, then refined their summaries through a "
        f"gossip protocol where each worker cross-referenced peers' findings.\n\n"
        f"Your job: merge all worker summaries into ONE comprehensive, well-structured "
        f"final answer to the user's question.\n\n"
        f"USER QUERY: {query}\n\n"
        f"WORKER SUMMARIES (post-gossip refinement):\n"
        f"{summaries_text}\n\n"
        f"{serendipity_block}"
        f"MERGE RULES:\n"
        f"1. Cross-reference across ALL workers. Where multiple workers agree, note consensus.\n"
        f"2. Where workers contradict, resolve using source quality and confidence.\n"
        f"3. Structure with clear headings and logical flow.\n"
        f"4. Cite sources with URLs where available.\n"
        f"5. Every sentence must deliver information. No filler, no disclaimers.\n"
        f"6. Keep speculative findings, clearly labeled as hypotheses.\n"
        f"7. Do NOT add unsolicited warnings or ethical disclaimers.\n"
        f"8. Mark areas where evidence is weak with [NEEDS VERIFICATION].\n"
        f"9. SYNTHESIZE into a flowing narrative — do NOT produce bullet-point lists or "
        f"data dumps. Weave findings into causal explanations and connected arguments.\n"
        f"10. If serendipity insights are provided above, integrate them naturally "
        f"into the appropriate sections — they represent cross-angle connections "
        f"that individual specialists missed.\n"
        f"11. AIM FOR 3000-6000 WORDS. Be comprehensive but not redundant. "
        f"Merge overlapping findings — do not repeat the same point from different workers.\n\n"
        f"Produce the final comprehensive synthesis:"
    )


async def queen_merge(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn,
    serendipity_insights: str = "",
    max_summary_chars: int = 6000,
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
        f"PRODUCE EXACTLY:\n"
        f"1. An EXECUTIVE SUMMARY (800-1500 words) that:\n"
        f"   - States the core findings and their significance\n"
        f"   - Identifies the most important cross-angle connections\n"
        f"   - Highlights key contradictions or unresolved questions\n"
        f"   - Does NOT repeat specific data points — those are in the full sections below\n"
        f"   - Reads as a standalone overview for someone who may not read the full report\n\n"
        f"2. A CROSS-REFERENCE MATRIX (markdown table) showing which angles share "
        f"findings, contradict each other, or have complementary insights.\n\n"
        f"3. A KEY FINDINGS list (5-10 bullet points) — the most important "
        f"individual discoveries across all angles.\n\n"
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
