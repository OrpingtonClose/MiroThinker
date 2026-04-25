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

import logging
import re
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Pattern to strip raw model reasoning tokens that should never appear in output
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


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
