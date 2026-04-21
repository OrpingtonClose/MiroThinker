# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Multi-agent serendipity panel — concurrent polymaths with different lenses.

Instead of one polymath, run 3-5 parallel serendipity agents, each with
a different analytical lens:

1. Mechanistic Convergence — finds where different domains describe the
   same underlying mechanism from different angles
2. Hidden Contradictions — spots where one angle's findings undermine
   a key assumption in another
3. Compounding Effects — identifies synergistic or catastrophic
   interactions that are more than the sum of parts
4. Framework Transfer — sees analytical methods from one domain that
   could resolve open questions in another
5. Evidence Desert — notices conspicuous absences (what SHOULD be
   there but isn't)

Each panel member sees only:
- Condensed angle summaries (from coordinators)
- Top collision statements (from bridge workers)

Never raw corpus data.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from swarm.topologies.collision import CollisionStatement

logger = logging.getLogger(__name__)


# Panel member lenses — each focuses on a different type of insight
PANEL_LENSES: dict[str, str] = {
    "mechanistic_convergence": (
        "You hunt for MECHANISTIC CONVERGENCE — places where different "
        "specialist domains are describing the SAME underlying mechanism, "
        "pathway, or cause from different vantage points. The specialists "
        "used different terminology and different evidence, but they're "
        "describing the same elephant. Your value: you see the unifying "
        "mechanism that connects seemingly unrelated specialist findings."
    ),
    "hidden_contradictions": (
        "You hunt for HIDDEN CONTRADICTIONS — places where one angle's "
        "findings quietly undermine a key assumption in another angle. "
        "Neither specialist noticed because the contradiction spans their "
        "domain boundary. These contradictions often hide the most "
        "important insights — they signal either (a) a genuine conflict "
        "that changes conclusions, or (b) an apparent conflict that "
        "reveals a deeper unifying explanation."
    ),
    "compounding_effects": (
        "You hunt for COMPOUNDING EFFECTS — places where combining "
        "findings from angle A and angle B produces an effect that is "
        "A×B, not A+B. Synergistic amplification or catastrophic "
        "interaction. The specialists each described their piece, but "
        "the COMBINATION creates something neither predicted. Your value: "
        "you see the emergent danger or opportunity that is invisible "
        "from any single angle."
    ),
    "framework_transfer": (
        "You hunt for FRAMEWORK TRANSFER — analytical methods, models, "
        "or reasoning frameworks used in one angle that could resolve "
        "an open question in another. Specialist A has a tool that "
        "Specialist B needs but doesn't know exists. Your value: you "
        "cross-pollinate methods, not just facts."
    ),
    "evidence_desert": (
        "You hunt for EVIDENCE DESERTS — topics that SHOULD appear in "
        "multiple angles based on the query, but are conspicuously "
        "absent. Absence of evidence is sometimes the most important "
        "signal. Also look for META-PATTERNS: regularities that only "
        "become visible when viewing all angles together (e.g. 'practice "
        "has converged on X while science remains conflicted on the "
        "mechanism')."
    ),
}


def _build_panel_prompt(
    date: str,
    lens_name: str,
    lens_instruction: str,
    query: str,
    angle_summaries_text: str,
    collision_text: str,
) -> str:
    """Build the prompt for one panel member."""
    return (
        f"You are a polymath on a Serendipity Panel. Your specific lens is "
        f"**{lens_name.replace('_', ' ').title()}**. Today is: {date}\n\n"
        f"{lens_instruction}\n\n"
        f"USER QUERY: {query}\n\n"
        f"ANGLE SUMMARIES (condensed from coordinators — each represents "
        f"the consolidated findings of one research angle):\n"
        f"{angle_summaries_text}\n\n"
        f"TOP COLLISION STATEMENTS (cross-domain insights discovered by "
        f"bridge workers during mesh conversation):\n"
        f"{collision_text}\n\n"
        f"RULES:\n"
        f"- ONLY report genuinely unexpected insights through YOUR specific "
        f"lens ({lens_name.replace('_', ' ')}).\n"
        f"- Each insight must reference specific findings from at least "
        f"two different angles.\n"
        f"- Prioritize insights that CHANGE how you would answer the query.\n"
        f"- If no genuinely surprising insights exist through your lens, "
        f"respond with: NO_INSIGHTS\n\n"
        f"Format each insight as:\n"
        f"LENS: {lens_name.replace('_', ' ').title()}\n"
        f"INSIGHT: [the unexpected finding and why it matters]\n"
        f"ANGLES: [which specialist angles are connected]\n"
        f"IMPLICATION: [what this means for the research question]\n"
        f"CONFIDENCE: [high/medium/low based on evidence strength]\n\n"
        f"Produce your insights:"
    )


async def run_serendipity_panel(
    angle_summaries: dict[str, str],
    collisions: list[CollisionStatement],
    query: str,
    complete_fn,
    max_concurrency: int = 5,
    max_panel_size: int = 5,
) -> str:
    """Run the multi-agent serendipity panel.

    Args:
        angle_summaries: Condensed summaries from angle coordinators.
        collisions: Top collision statements from bridge workers.
        query: The user's research query.
        complete_fn: Async LLM completion callable.
        max_concurrency: Max parallel LLM calls.
        max_panel_size: Max number of panel members (1-5).

    Returns:
        Combined serendipity insights from all panel members.
    """
    if len(angle_summaries) < 2:
        return ""

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build angle summaries text
    summaries_text = ""
    for angle, summary in angle_summaries.items():
        summaries_text += f"\n### {angle}\n{summary[:3000]}\n"

    # Build collision text (top 20 by serendipity score)
    top_collisions = sorted(
        collisions, key=lambda c: c.serendipity_score, reverse=True,
    )[:20]
    if top_collisions:
        collision_text = "\n".join(c.to_prompt_line() for c in top_collisions)
    else:
        collision_text = "(No collision statements available yet)"

    # Select panel members (up to max_panel_size)
    lenses = list(PANEL_LENSES.items())[:max_panel_size]

    sem = asyncio.Semaphore(max_concurrency)
    results: dict[str, str] = {}

    async def _bounded(lens_name: str, lens_instruction: str) -> None:
        async with sem:
            prompt = _build_panel_prompt(
                date=date,
                lens_name=lens_name,
                lens_instruction=lens_instruction,
                query=query,
                angle_summaries_text=summaries_text,
                collision_text=collision_text,
            )
            try:
                result = await complete_fn(prompt)
                if result and "NO_INSIGHTS" not in result and len(result.strip()) > 50:
                    results[lens_name] = result
            except Exception:
                logger.warning(
                    "lens=<%s> | serendipity panel member failed", lens_name,
                )

    await asyncio.gather(*[
        _bounded(name, instruction) for name, instruction in lenses
    ])

    if not results:
        return ""

    # Combine all panel insights
    parts = []
    for lens_name, insights in results.items():
        parts.append(
            f"═══ {lens_name.replace('_', ' ').upper()} ═══\n{insights}"
        )

    combined = "\n\n".join(parts)

    logger.info(
        "panel_members=<%d>, responding=<%d>, total_chars=<%d> | "
        "serendipity panel complete",
        len(lenses), len(results), len(combined),
    )

    return combined
