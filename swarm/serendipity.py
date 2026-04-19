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


SERENDIPITY_PROMPT = """\
You are a polymath research connector. Today is: {date}

Multiple specialist analysts have independently investigated different \
angles of a research query, then refined their summaries through a gossip \
protocol where each worker cross-referenced peers' findings.

Your job is to find UNEXPECTED CONNECTIONS between their angles — insights \
that no single specialist would have noticed because they require knowledge \
from multiple domains simultaneously.

USER QUERY: {query}

SPECIALIST SUMMARIES (post-gossip):
{worker_summaries}

INSTRUCTIONS:
1. Look for surprising convergences — where different angles unexpectedly \
point to the same underlying mechanism, cause, or pattern.
2. Look for hidden contradictions — where findings from one angle \
undermine assumptions in another angle.
3. Look for transfer opportunities — where a method, framework, or \
insight from one angle could illuminate another.
4. Look for emergent patterns — meta-level observations that only become \
visible when viewing all angles together.
5. ONLY report genuinely unexpected connections. Do NOT report obvious \
or trivial relationships.
6. Each connection must reference specific findings from at least two \
different worker angles.

If no genuinely surprising connections exist, respond with: NO_SURPRISES

Format each connection as:
CONNECTION: [description of the unexpected link and why it matters]
ANGLES: [which worker angles are connected]
IMPLICATION: [what this means for the overall research question]

Produce your cross-angle insights:"""


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

    prompt = SERENDIPITY_PROMPT \
        .replace("{date}", date) \
        .replace("{query}", query) \
        .replace("{worker_summaries}", summaries_text)

    result = await complete_fn(prompt)

    if not result or "NO_SURPRISES" in result:
        return ""

    return result
