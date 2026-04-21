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

_OFF_ANGLE_NOTE = (
    "NOTE: Your section contains MOSTLY data from your domain, but also "
    "some RAW DATA FROM OTHER DOMAINS. This off-angle data is the most "
    "valuable part of your section — it contains details that specialists "
    "in those domains may have overlooked but that YOUR expertise can "
    "illuminate. When you encounter data outside your field, EXPLAIN it "
    "through your {angle} lens. What mechanisms from your field explain "
    "what you're seeing? What would your domain predict about it?\n\n"
)


def _build_synth_prompt(
    date: str,
    angle: str,
    char_count: int,
    section_content: str,
    query: str,
    max_chars: int,
    has_off_angle_data: bool = False,
) -> str:
    """Build the worker synthesis prompt via concatenation (not .replace()).

    Concatenation prevents template injection: if corpus text contains
    literal ``{query}`` or ``{max_chars}``, it won't be re-scanned.
    """
    return (
        f"You are a {angle} specialist. Everything you encounter, you interpret "
        f"through the lens of {angle}. When you read data from outside your "
        f"domain, your job is not to summarize it — it is to EXPLAIN it through "
        f"your domain's mechanisms, frameworks, and first principles. You see "
        f"what others miss because you bring {angle} expertise to data that "
        f"wasn't collected with your domain in mind.\n\n"
        f"Other specialists are processing OTHER slices of the same corpus — "
        f"you will exchange analyses with them in subsequent rounds. "
        f"Today is: {date}\n\n"
        f"YOUR CORE IDENTITY: {angle}\n"
        f"YOUR SECTION ({char_count} chars):\n"
        f"{section_content}\n\n"
        f"USER QUERY: {query}\n\n"
        f"{_OFF_ANGLE_NOTE.format(angle=angle) if has_off_angle_data else ''}"
        f"ANALYSIS RULES — REASON THROUGH YOUR LENS:\n"
        f"1. For each key finding, state what it IMPLIES through the lens of "
        f"{angle}. Don't just say 'ferritin elevated 340%' — explain what "
        f"that elevation MEANS in your domain, what it PREDICTS for other "
        f"biological systems, what it EXPLAINS about related observations.\n"
        f"2. Preserve exact numbers, sources, and citations inline — embed "
        f"them WITHIN reasoning, not as a separate list.\n"
        f"3. When you see a pattern, trace the CAUSAL CHAIN through your "
        f"domain: A causes B because C (mechanism from {angle}), which "
        f"predicts D. The chain is the insight.\n"
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
        f"specialists in angles you DON'T cover. What PREDICTIONS does your "
        f"{angle} analysis generate that other specialists could confirm or "
        f"refute with THEIR data? State specific, testable claims. "
        f"This section is how cross-domain connections get discovered "
        f"during conversation.\n\n"
        f"Produce a THOROUGH analysis — preserve all reasoning chains, "
        f"evidence, and implications. Depth over brevity.\n\n"
        f"Produce your reasoned analysis through the {angle} lens:"
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
    hive_memory: str = "",
) -> str:
    """Build the gossip refinement prompt via concatenation (not .replace()).

    Args:
        round_prompt: Optional round-specific focus instructions injected
            before the refinement rules (e.g. "ROUND 2 FOCUS — CONTRADICTION
            RESOLUTION: ...").
        delta_text: New findings that arrived between gossip rounds
            (corpus delta injection).  Prepended as supplementary evidence.
        hive_memory: Targeted findings from the persistent store (RAG)
            relevant to this bee's current analysis.  Injected as
            "FROM THE HIVE" block.
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
    hive_block = ""
    if hive_memory:
        hive_block = (
            f"═══ FROM THE HIVE (targeted findings from other bees' "
            f"previous work that relate to YOUR current analysis) ═══\n"
            f"{hive_memory}\n\n"
            f"These findings were retrieved because they match concepts "
            f"in your current analysis. Interpret them through your "
            f"{angle} lens — what do these cross-domain findings MEAN "
            f"in your domain? What mechanisms from {angle} explain or "
            f"predict what other bees observed?\n\n"
        )
    return (
        f"You are a {angle} specialist in a peer-to-peer research "
        f"conversation. Everything you encounter, you interpret through "
        f"the lens of {angle}. Your job is to find where your peers' "
        f"findings CONNECT with your own evidence — not just verify facts, "
        f"but discover where domains collide to produce new understanding "
        f"that YOUR {angle} expertise uniquely reveals. Today is: {date}\n\n"
        f"In the previous round, you produced an analysis from your section.\n"
        f"Now you have received analyses from your PEER WORKERS who processed "
        f"other sections of the same corpus.\n\n"
        f"YOUR CORE IDENTITY: {angle}\n\n"
        f"{raw_section_block}"
        f"YOUR PREVIOUS ANALYSIS:\n"
        f"{own_summary}\n\n"
        f"PEER ANALYSES (from {n_peers} other specialists):\n"
        f"{peers_text}\n\n"
        f"{hive_block}"
        f"{delta_block}"
        f"{round_block}"
        f"CONVERSATION RULES — DEPTH OVER BREADTH:\n"
        f"1. Pick the TOP 2-3 connections between your data and peers' — the "
        f"ones that CHANGE UNDERSTANDING the most when viewed through {angle}. "
        f"Go DEEP on those rather than shallow on many. A single deeply-traced "
        f"connection is worth more than ten surface-level overlaps.\n"
        f"2. For each connection, provide ALL THREE:\n"
        f"   (a) EVIDENCE CHAIN: A (your data, with source) + B (peer data, "
        f"with source) → C (the insight neither stated alone, explained "
        f"through {angle} mechanisms)\n"
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
        f"5. RE-MINE YOUR RAW DATA through your {angle} lens. Now that "
        f"you've read peers' analyses, go back to your original section "
        f"and pull out details you OVERLOOKED in your initial analysis — "
        f"details that become relevant because of what peers found. "
        f"This is the most valuable thing you can do.\n"
        f"6. Your output is a REASONED ANALYSIS through {angle}, not a "
        f"fact list. Each finding connects to others through causal chains "
        f"and predictions grounded in your domain. The connections ARE "
        f"the output.\n"
        f"7. Maintain source citations and evidence quality assessments.\n"
        f"8. Produce a THOROUGH analysis — do not artificially compress your "
        f"reasoning. Every evidence chain, prediction, and falsification "
        f"condition matters. Depth over brevity.\n\n"
        f"Produce your DEEPLY CONNECTED {angle} analysis:"
    )


async def worker_synthesize(
    angle: str,
    section_content: str,
    query: str,
    max_chars: int,
    complete_fn,
    has_off_angle_data: bool = False,
) -> str:
    """Phase 1: Worker synthesizes its assigned corpus section.

    Args:
        angle: The worker's assigned research angle.
        section_content: The raw corpus section to analyze.
        query: The user's research query.
        max_chars: Maximum summary length in characters.
        complete_fn: Async LLM completion callable.
        has_off_angle_data: Whether this worker's slice contains
            deliberately injected off-angle data from misassignment.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = _build_synth_prompt(
        date=date,
        angle=angle,
        char_count=len(section_content),
        section_content=section_content,
        query=query,
        max_chars=max_chars,
        has_off_angle_data=has_off_angle_data,
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
    hive_memory: str = "",
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
        hive_memory: Targeted findings from the persistent store (RAG)
            relevant to this worker's current analysis.

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
        peers_text += f"\n--- Peer Specialist {i + 1} ---\n{ps}\n"

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
        hive_memory=hive_memory,
    )

    return await complete_fn(prompt)
