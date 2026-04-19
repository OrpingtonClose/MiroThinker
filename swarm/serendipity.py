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
        f"You are a polymath research connector — your unique value is seeing "
        f"what domain specialists cannot. Today is: {date}\n\n"
        f"Multiple specialist analysts have independently investigated different "
        f"angles of a research query, then refined their summaries through "
        f"multiple rounds of peer gossip where each worker cross-referenced "
        f"peers' findings.\n\n"
        f"Your job is to find UNEXPECTED CONNECTIONS between their angles — "
        f"insights that no single specialist would have noticed because they "
        f"require knowledge from multiple domains simultaneously.\n\n"
        f"USER QUERY: {query}\n\n"
        f"SPECIALIST SUMMARIES (post-gossip):\n"
        f"{summaries_text}\n\n"
        f"═══ CONNECTION TAXONOMY ═══\n\n"
        f"Search for these specific types of cross-angle insight:\n\n"
        f"1. MECHANISTIC CONVERGENCE — Different angles point to the same "
        f"underlying mechanism, cause, or pathway through different routes. "
        f"The specialists described the same elephant from different angles.\n\n"
        f"2. HIDDEN CONTRADICTION — Findings from one angle undermine a key "
        f"assumption in another angle. Neither specialist noticed because "
        f"the contradiction spans their domain boundary.\n\n"
        f"3. COMPOUNDING EFFECTS — When findings from angle A and angle B "
        f"are combined, the effect is not A+B but A×B (synergistic or "
        f"catastrophic). The interaction is more than the sum of parts.\n\n"
        f"4. EVIDENCE DESERT — A topic that SHOULD appear in multiple angles "
        f"based on the query, but is conspicuously absent. Absence of "
        f"evidence is sometimes the most important signal.\n\n"
        f"5. FRAMEWORK TRANSFER — A method, model, or analytical framework "
        f"used in one angle that would resolve an open question in another.\n\n"
        f"6. META-PATTERN — A pattern that only becomes visible when viewing "
        f"all angles together (e.g., 'practice has converged on X while "
        f"science remains conflicted on the mechanism').\n\n"
        f"RULES:\n"
        f"- ONLY report genuinely unexpected connections. Obvious or trivial "
        f"relationships add no value.\n"
        f"- Each connection must reference specific findings from at least "
        f"two different worker angles.\n"
        f"- Prioritize connections where the IMPLICATION changes how you "
        f"would answer the user's query.\n\n"
        f"If no genuinely surprising connections exist, respond with: NO_SURPRISES\n\n"
        f"Format each connection as:\n"
        f"TYPE: [one of the 6 types above]\n"
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
