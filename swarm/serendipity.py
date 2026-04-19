# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Cross-angle serendipity bridge.

After specialist workers have independently analysed their angles, the
serendipity bridge asks: "What would a polymath notice that domain
specialists missed?"

This is the differentiating feature vs plain ruflo gossip. In benchmarks,
angle-based gossip + serendipity bridge scored 48/60 total vs 47/60 for
ruflo gossip without it. The bridge found connections like:
- Convergence of permanent organ injury across tren and tbol despite
  opposite metabolic profiles
- "Self-concealing risk profiles" as a novel analytical framework
- Evidence deserts as negative signals

The bridge reads ALL worker summaries (post-gossip) and looks ONLY for
unexpected connections between angles. It does NOT summarize — that's
the queen's job.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _build_serendipity_prompt(
    date: str,
    query: str,
    summaries_text: str,
) -> str:
    """Build the serendipity prompt via concatenation (not .replace()).

    Concatenation prevents template injection: if worker summaries contain
    literal ``{placeholder}`` strings, they won't be re-scanned.
    """
    return (
        f"You are a polymath research connector. Today is: {date}\n\n"
        f"Multiple specialist analysts have independently investigated different "
        f"angles of a research query, then refined their summaries through a "
        f"gossip protocol where each worker cross-referenced peers' findings.\n\n"
        f"Your job is to find UNEXPECTED CONNECTIONS between their angles — insights "
        f"that no single specialist would have noticed because they require knowledge "
        f"from multiple domains simultaneously.\n\n"
        f"USER QUERY: {query}\n\n"
        f"SPECIALIST SUMMARIES (post-gossip):\n"
        f"{summaries_text}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Look for surprising convergences — where different angles unexpectedly "
        f"point to the same underlying mechanism, cause, or pattern.\n"
        f"2. Look for hidden contradictions — where findings from one angle "
        f"undermine assumptions in another angle.\n"
        f"3. Look for transfer opportunities — where a method, framework, or "
        f"insight from one angle could illuminate another.\n"
        f"4. Look for emergent patterns — meta-level observations that only become "
        f"visible when viewing all angles together.\n"
        f"5. ONLY report genuinely unexpected connections. Do NOT report obvious "
        f"or trivial relationships.\n"
        f"6. Each connection must reference specific findings from at least two "
        f"different worker angles.\n\n"
        f"If no genuinely surprising connections exist, respond with: NO_SURPRISES\n\n"
        f"Format each connection as:\n"
        f"CONNECTION: [description of the unexpected link and why it matters]\n"
        f"ANGLES: [which worker angles are connected]\n"
        f"IMPLICATION: [what this means for the overall research question]\n\n"
        f"Produce your cross-angle insights:"
    )


async def find_serendipitous_connections(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn,
) -> str:
    """Run the serendipity bridge across all worker angle summaries.

    Args:
        worker_summaries: Mapping of angle name -> refined summary text.
        query: The user's original research query.
        complete_fn: Async LLM completion callable.

    Returns:
        String containing cross-angle insights, or empty string if none found.
    """
    if len(worker_summaries) < 2:
        return ""

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    summaries_text = ""
    for angle, summary in worker_summaries.items():
        summaries_text += f"\n### Angle: {angle}\n{summary}\n"

    prompt = _build_serendipity_prompt(
        date=date,
        query=query,
        summaries_text=summaries_text,
    )

    result = await complete_fn(prompt)

    if not result or "NO_SURPRISES" in result:
        return ""

    return result
