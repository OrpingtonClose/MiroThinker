# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm worker — synthesizes a corpus section and refines via gossip.

Each worker:
1. Phase 1 (Map): Receives a corpus section, produces a focused summary
2. Phase 2 (Gossip): Reads all peer summaries + optionally its own raw
   corpus section, then refines its summary with cross-references

The key innovation from benchmark findings: during gossip, workers retain
access to their FULL original corpus section (not just their compressed
summary). When Worker 1 reads Worker 5's forum findings, it can go back
to its own raw data and pull out details that become relevant in light
of the peer's insight. This produced 9/10 cross-referencing vs 8/10
for summary-only gossip.

Multi-round gossip with round-specific prompts:
  Round 1: Connection Discovery — find where peers' findings explain your evidence
  Round 2: Causal Depth — trace second-order connections and predictions
  Round 3: Gaps and Final Synthesis — identify unknowns, produce connected narrative
"""

from __future__ import annotations

from datetime import datetime, timezone


def _build_synth_prompt(
    date: str,
    angle: str,
    char_count: int,
    section_content: str,
    query: str,
    max_chars: int,
) -> str:
    """Build the worker synthesis prompt via concatenation (not .replace()).

    Concatenation prevents template injection: if corpus text contains
    literal ``{query}`` or ``{max_chars}``, it won't be re-scanned.
    """
    return (
        f"You are a specialist research analyst assigned to one slice of a larger "
        f"corpus. Other specialists are processing OTHER slices — you will "
        f"exchange analyses with them in subsequent rounds. Today is: {date}\n\n"
        f"YOUR ASSIGNED ANGLE: {angle}\n"
        f"YOUR SECTION ({char_count} chars):\n"
        f"{section_content}\n\n"
        f"USER QUERY: {query}\n\n"
        f"ANALYSIS RULES — REASON, DON'T JUST LIST:\n"
        f"1. For each key finding, state what it IMPLIES beyond the obvious. "
        f"Don't just say 'ferritin elevated 340%' — say what that elevation "
        f"would PREDICT for other biological systems, what it EXPLAINS about "
        f"related observations, what it makes SURPRISING if true.\n"
        f"2. Preserve exact numbers, sources, and citations inline — but embed "
        f"them WITHIN reasoning, not as a separate list.\n"
        f"3. When you see a pattern, trace the CAUSAL CHAIN: A causes B "
        f"because C, which predicts D. The chain is the insight.\n"
        f"4. Mark contradictions and reason about WHY they exist — different "
        f"populations? Different dosing? Measurement error? The resolution "
        f"often contains the deepest insight.\n"
        f"5. Do NOT add disclaimers, moral commentary, or safety warnings.\n"
        f"6. Do NOT pad with generic context — only specific findings and "
        f"your reasoning about them.\n"
        f"7. Keep speculative connections, clearly labeled as hypotheses.\n\n"
        f"MANDATORY FINAL SECTION — CROSS-DOMAIN IMPLICATIONS:\n"
        f"End your analysis with a section titled 'IMPLICATIONS FOR OTHER "
        f"DOMAINS' where you reason about what your findings MEAN for "
        f"specialists in angles you DON'T cover. What would a molecular "
        f"biologist need from your practitioner data? What would a safety "
        f"specialist want from your pharmacokinetics data? What PREDICTIONS "
        f"does your data generate that other specialists could confirm or "
        f"refute with THEIR data? This section is how cross-domain "
        f"connections get discovered during conversation.\n\n"
        f"Stay under {max_chars} characters.\n\n"
        f"Produce your reasoned analysis:"
    )


def _build_gossip_prompt(
    date: str,
    angle: str,
    raw_section_block: str,
    own_summary: str,
    n_peers: int,
    peers_text: str,
    max_chars: int,
    round_prompt: str = "",
    delta_text: str = "",
) -> str:
    """Build the gossip refinement prompt via concatenation (not .replace()).

    Args:
        round_prompt: Optional round-specific focus instructions injected
            before the refinement rules (e.g. "ROUND 2 FOCUS — CONTRADICTION
            RESOLUTION: ...").
        delta_text: New findings that arrived between gossip rounds
            (corpus delta injection).  Prepended as supplementary evidence.
    """
    round_block = f"{round_prompt}\n\n" if round_prompt else ""
    delta_block = ""
    if delta_text:
        delta_block = (
            f"═══ NEW EVIDENCE (arrived during your deliberation) ═══\n"
            f"{delta_text}\n\n"
            f"Process this new evidence with the SAME depth as peer "
            f"findings. What connections emerge between this new data and "
            f"what you and your peers have already established?\n\n"
        )
    return (
        f"You are a specialist analyst in a peer-to-peer research gossip protocol. "
        f"Your job is to find where your peers' findings CONNECT with your own "
        f"evidence — not just verify facts, but discover where domains collide "
        f"to produce new understanding. Today is: {date}\n\n"
        f"In the previous round, you produced an analysis from your section.\n"
        f"Now you have received analyses from your PEER WORKERS who processed "
        f"other sections of the same corpus.\n\n"
        f"YOUR ASSIGNED ANGLE: {angle}\n\n"
        f"{raw_section_block}"
        f"YOUR PREVIOUS SUMMARY:\n"
        f"{own_summary}\n\n"
        f"PEER SUMMARIES (from {n_peers} other workers):\n"
        f"{peers_text}\n\n"
        f"{delta_block}"
        f"{round_block}"
        f"CONVERSATION RULES — DEPTH OVER BREADTH:\n"
        f"1. Pick the TOP 2-3 connections between your data and peers' — the "
        f"ones that CHANGE UNDERSTANDING the most. Go DEEP on those rather "
        f"than shallow on many. A single deeply-traced connection is worth "
        f"more than ten surface-level overlaps.\n"
        f"2. For each connection, provide ALL THREE:\n"
        f"   (a) EVIDENCE CHAIN: A (your data, with source) + B (peer data, "
        f"with source) → C (the insight neither stated alone)\n"
        f"   (b) PREDICTION: If this connection is real, what ELSE should be "
        f"true? What would you expect to find in data you haven't seen?\n"
        f"   (c) FALSIFICATION: What specific evidence would DISPROVE this "
        f"connection? If nothing could disprove it, it's not a real insight.\n"
        f"3. PRESERVE EXACT NUMBERS. Never paraphrase numerical values. When "
        f"peers state different numbers for the same thing, quote BOTH with "
        f"sources and reason about which is correct and WHY.\n"
        f"4. CONTRADICTIONS are OPPORTUNITIES. If a peer's claim conflicts "
        f"with your evidence, this is where the deepest insight often hides. "
        f"Go back to your raw section, find the exact source, and reason: "
        f"is this a real conflict (different data) or apparent (same "
        f"phenomenon from different angles)? The resolution IS the insight.\n"
        f"5. RE-MINE YOUR RAW DATA. Now that you've read peers' analyses, go "
        f"back to your original section and pull out details you OVERLOOKED "
        f"in your initial analysis — details that become relevant because of "
        f"what peers found. This is the most valuable thing you can do.\n"
        f"6. Your output is a REASONED ANALYSIS, not a fact list. Each "
        f"finding connects to others through causal chains and predictions. "
        f"The connections ARE the output.\n"
        f"7. Maintain source citations and evidence quality assessments.\n"
        f"8. Stay under {max_chars} characters.\n\n"
        f"Produce your DEEPLY CONNECTED analysis:"
    )


async def worker_synthesize(
    angle: str,
    section_content: str,
    query: str,
    max_chars: int,
    complete_fn,
) -> str:
    """Phase 1: Worker synthesizes its assigned corpus section."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = _build_synth_prompt(
        date=date,
        angle=angle,
        char_count=len(section_content),
        section_content=section_content,
        query=query,
        max_chars=max_chars,
    )
    return await complete_fn(prompt)


async def worker_gossip_refine(
    angle: str,
    own_summary: str,
    peer_summaries: list[str],
    raw_section: str | None,
    query: str,
    max_chars: int,
    complete_fn,
    round_prompt: str = "",
    delta_text: str = "",
) -> str:
    """Phase 2: Worker refines its summary using peer gossip.

    If raw_section is provided (full-corpus gossip mode), the worker
    retains access to its original corpus section during refinement.
    This allows it to pull out specific details that become relevant
    only after reading peer findings.

    Args:
        angle: The worker's assigned research angle.
        own_summary: The worker's current summary (from previous round).
        peer_summaries: Summaries from all other workers.
        raw_section: Original corpus section (if full-corpus gossip enabled).
        query: The user's research query.
        max_chars: Maximum summary length in characters.
        complete_fn: Async LLM completion callable.
        round_prompt: Optional round-specific focus instructions.
        delta_text: New findings that arrived between gossip rounds
            (corpus delta injection).

    Returns:
        Refined summary string.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build raw section block
    if raw_section:
        raw_section_block = (
            f"YOUR ORIGINAL RAW SECTION (full text — refer back to pull out "
            f"details relevant to peer findings):\n"
            f"{raw_section}\n\n"
        )
    else:
        raw_section_block = ""

    # Build peer summaries text
    peers_text = ""
    for i, ps in enumerate(peer_summaries):
        peers_text += f"\n--- Peer Specialist {i + 1} ---\n{ps[:max_chars]}\n"

    prompt = _build_gossip_prompt(
        date=date,
        angle=angle,
        raw_section_block=raw_section_block,
        own_summary=own_summary,
        n_peers=len(peer_summaries),
        peers_text=peers_text,
        max_chars=max_chars,
        round_prompt=round_prompt,
        delta_text=delta_text,
    )

    return await complete_fn(prompt)
