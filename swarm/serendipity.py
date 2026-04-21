# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Cross-angle serendipity bridge — two-pass focused discovery.

After specialist workers have independently analysed their angles, the
serendipity bridge asks: "What would a polymath notice that domain
specialists missed?"

Two focused passes instead of one broad sweep:
  Pass 1 (CONVERGENCE): Find where domains amplify each other —
    mechanistic convergences, compounding effects, framework transfers.
  Pass 2 (CONTRADICTION): Find where domains conflict or are
    suspiciously silent — hidden contradictions, evidence deserts,
    meta-patterns.

Each pass focuses on fewer connection types, producing deeper analysis
per type than a single pass covering all six types.

The bridge reads ALL worker summaries (post-gossip) and looks ONLY for
unexpected connections between angles. It does NOT summarize — that's
the queen's job.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _build_convergence_prompt(
    date: str,
    query: str,
    summaries_text: str,
) -> str:
    """Build Pass 1: find where domains AMPLIFY each other."""
    return (
        f"You are a polymath research connector — your unique value is seeing "
        f"what domain specialists cannot. Today is: {date}\n\n"
        f"Multiple specialist analysts have independently investigated different "
        f"angles of a research query, then refined their analyses through "
        f"multiple rounds of peer conversation where they traced cross-domain "
        f"connections, made predictions, and tested them against each other's "
        f"evidence.\n\n"
        f"USER QUERY: {query}\n\n"
        f"SPECIALIST ANALYSES (post-conversation):\n"
        f"{summaries_text}\n\n"
        f"═══ PASS 1: WHERE DOMAINS AMPLIFY EACH OTHER ═══\n\n"
        f"Search ONLY for these three types of cross-angle insight:\n\n"
        f"1. MECHANISTIC CONVERGENCE — Different angles point to the same "
        f"underlying mechanism through different routes. The specialists "
        f"described the same elephant from different angles but didn't realize "
        f"it. Trace the COMPLETE chain showing how finding A from Angle X "
        f"and finding B from Angle Y converge on mechanism C.\n\n"
        f"2. COMPOUNDING EFFECTS — When findings from angle A and angle B "
        f"are combined, the effect is not A+B but A*B (synergistic or "
        f"catastrophic). Show the specific evidence from each angle and "
        f"explain WHY the interaction is multiplicative, not additive.\n\n"
        f"3. FRAMEWORK TRANSFER — A method, model, or analytical framework "
        f"used in one angle that would resolve an open question in another. "
        f"State the specific question AND how the framework resolves it.\n\n"
        f"DEPTH RULES:\n"
        f"- For each connection, trace the FULL evidence chain with sources\n"
        f"- State what this connection PREDICTS that could be tested\n"
        f"- Reference specific numbers, sources, and claims from the workers\n"
        f"- If no genuinely surprising connections of these types exist, "
        f"respond with: NO_CONVERGENCES\n\n"
        f"Format each connection as:\n"
        f"TYPE: [MECHANISTIC_CONVERGENCE / COMPOUNDING / FRAMEWORK_TRANSFER]\n"
        f"EVIDENCE: [specific findings from each angle with sources]\n"
        f"CONNECTION: [the insight that emerges from combining them]\n"
        f"PREDICTION: [what this predicts that could be tested]\n"
        f"ANGLES: [which worker angles are connected]\n\n"
        f"Produce your convergence insights:"
    )


def _build_contradiction_prompt(
    date: str,
    query: str,
    summaries_text: str,
    convergence_insights: str,
) -> str:
    """Build Pass 2: find where domains CONFLICT or are suspiciously SILENT."""
    convergence_block = ""
    if convergence_insights:
        convergence_block = (
            f"CONVERGENCES ALREADY FOUND (Pass 1):\n"
            f"{convergence_insights}\n\n"
            f"Do NOT repeat these. Look for the OPPOSITE: where domains "
            f"conflict or where important topics are suspiciously absent.\n\n"
        )
    return (
        f"You are a polymath research connector — your unique value is seeing "
        f"what domain specialists cannot. Today is: {date}\n\n"
        f"Multiple specialist analysts have independently investigated different "
        f"angles of a research query, then refined their analyses through "
        f"multiple rounds of peer conversation.\n\n"
        f"USER QUERY: {query}\n\n"
        f"SPECIALIST ANALYSES (post-conversation):\n"
        f"{summaries_text}\n\n"
        f"{convergence_block}"
        f"═══ PASS 2: WHERE DOMAINS CONFLICT OR GO SILENT ═══\n\n"
        f"Search ONLY for these three types of cross-angle insight:\n\n"
        f"1. HIDDEN CONTRADICTION — Findings from one angle undermine a key "
        f"assumption in another angle. Neither specialist noticed because "
        f"the contradiction spans their domain boundary. Trace BOTH sides "
        f"of the contradiction with exact evidence and reason about which "
        f"side has stronger support — or whether the contradiction reveals "
        f"something deeper.\n\n"
        f"2. EVIDENCE DESERT — A topic that SHOULD appear in multiple angles "
        f"based on the query, but is conspicuously absent. Absence of "
        f"evidence is sometimes the most important signal. State what's "
        f"missing, WHY you'd expect it to be present, and what its absence "
        f"implies.\n\n"
        f"3. META-PATTERN — A pattern that only becomes visible when viewing "
        f"all angles together (e.g., 'practice has converged on X while "
        f"science remains conflicted on the mechanism' or 'every angle "
        f"shows the same time-dependent pattern'). The pattern itself is "
        f"the insight.\n\n"
        f"DEPTH RULES:\n"
        f"- For each finding, cite specific evidence from multiple angles\n"
        f"- State what this means for the overall research question\n"
        f"- Reference specific numbers, sources, and claims from the workers\n"
        f"- If no genuinely surprising findings of these types exist, "
        f"respond with: NO_CONTRADICTIONS\n\n"
        f"Format each finding as:\n"
        f"TYPE: [HIDDEN_CONTRADICTION / EVIDENCE_DESERT / META_PATTERN]\n"
        f"EVIDENCE: [specific findings or absences from each angle]\n"
        f"INSIGHT: [what this reveals about the research question]\n"
        f"IMPLICATION: [how this changes understanding of the query]\n"
        f"ANGLES: [which worker angles are involved]\n\n"
        f"Produce your contradiction/absence insights:"
    )


async def find_serendipitous_connections(
    worker_summaries: dict[str, str],
    query: str,
    complete_fn,
) -> tuple[str, int]:
    """Run the two-pass serendipity bridge across all worker angle summaries.

    Pass 1 (parallel-safe): Find convergences, compounding effects, transfers.
    Pass 2 (sequential): Find contradictions, deserts, meta-patterns.
    Pass 2 receives Pass 1 results to avoid duplication.

    Args:
        worker_summaries: Mapping of angle name -> refined summary text.
        query: The user's original research query.
        complete_fn: Async LLM completion callable.

    Returns:
        Tuple of (combined insights string, number of successful LLM calls).
        Insights string is empty if neither pass found anything.
    """
    if len(worker_summaries) < 2:
        return "", 0

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    summaries_text = ""
    for angle, summary in worker_summaries.items():
        summaries_text += f"\n### Angle: {angle}\n{summary}\n"

    # Pass 1: Convergences
    convergence_prompt = _build_convergence_prompt(
        date=date,
        query=query,
        summaries_text=summaries_text,
    )

    llm_calls = 0
    try:
        convergence_result = await complete_fn(convergence_prompt)
        llm_calls += 1
    except Exception:
        logger.warning("serendipity pass 1 (convergence) failed")
        convergence_result = ""

    if convergence_result and "NO_CONVERGENCES" in convergence_result:
        convergence_result = ""

    # Pass 2: Contradictions (receives Pass 1 to avoid duplication)
    contradiction_prompt = _build_contradiction_prompt(
        date=date,
        query=query,
        summaries_text=summaries_text,
        convergence_insights=convergence_result,
    )

    try:
        contradiction_result = await complete_fn(contradiction_prompt)
        llm_calls += 1
    except Exception:
        logger.warning("serendipity pass 2 (contradiction) failed")
        contradiction_result = ""

    if contradiction_result and "NO_CONTRADICTIONS" in contradiction_result:
        contradiction_result = ""

    # Combine results
    parts = []
    if convergence_result:
        parts.append(
            "═══ CONVERGENCES (where domains amplify each other) ═══\n"
            + convergence_result
        )
    if contradiction_result:
        parts.append(
            "═══ CONTRADICTIONS & SILENCES (where domains conflict or go quiet) ═══\n"
            + contradiction_result
        )

    combined = "\n\n".join(parts)

    logger.info(
        "convergence_chars=<%d>, contradiction_chars=<%d> | two-pass serendipity complete",
        len(convergence_result) if convergence_result else 0,
        len(contradiction_result) if contradiction_result else 0,
    )

    return combined, llm_calls
