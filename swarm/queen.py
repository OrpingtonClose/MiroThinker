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

from datetime import datetime, timezone


QUEEN_MERGE_PROMPT = """\
You are the queen synthesizer in a research swarm. Today is: {date}

{n_workers} specialist workers have independently processed different \
sections of a research corpus, then refined their summaries through a \
gossip protocol where each worker cross-referenced peers' findings.

Your job: merge all worker summaries into ONE comprehensive, well-structured \
final answer to the user's question.

USER QUERY: {query}

WORKER SUMMARIES (post-gossip refinement):
{worker_summaries}

{serendipity_block}

MERGE RULES:
1. Cross-reference across ALL workers. Where multiple workers agree, note consensus.
2. Where workers contradict, resolve using source quality and confidence.
3. Structure with clear headings and logical flow.
4. Cite sources with URLs where available.
5. Every sentence must deliver information. No filler, no disclaimers.
6. Keep speculative findings, clearly labeled as hypotheses.
7. Do NOT add unsolicited warnings or ethical disclaimers.
8. Mark areas where evidence is weak with [NEEDS VERIFICATION].
9. SYNTHESIZE into a flowing narrative — do NOT produce bullet-point lists or \
data dumps. Weave findings into causal explanations and connected arguments.
10. If serendipity insights are provided below, integrate them naturally \
into the appropriate sections — they represent cross-angle connections \
that individual specialists missed.

Produce the final comprehensive synthesis:"""


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
            f"{serendipity_insights}\n"
        )
    else:
        serendipity_block = ""

    prompt = QUEEN_MERGE_PROMPT \
        .replace("{date}", date) \
        .replace("{n_workers}", str(len(worker_summaries))) \
        .replace("{query}", query) \
        .replace("{worker_summaries}", summaries_text) \
        .replace("{serendipity_block}", serendipity_block)

    try:
        result = await complete_fn(prompt)
        if result and len(result.strip()) > 100:
            return result
    except Exception:
        pass

    # Fallback: concatenate worker summaries
    parts = []
    for angle, summary in worker_summaries.items():
        parts.append(f"## {angle}\n{summary}")
    if serendipity_insights:
        parts.append(f"## Cross-Angle Insights\n{serendipity_insights}")
    return "\n\n".join(parts)
