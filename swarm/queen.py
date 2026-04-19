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
