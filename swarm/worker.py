# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm worker — reasons deeply over a corpus section and refines via gossip.

Each worker:
1. Phase 1 (Map): Receives a corpus section, performs deep analytical
   reasoning — evaluates evidence quality, identifies contradictions
   within its section, and preserves exact numbers/dosages/timings
   verbatim from the source material.
2. Phase 2 (Gossip): Reads all peer analyses + its own raw corpus
   section, reasons about how peer findings interact with its own
   evidence, and resolves cross-worker contradictions by going back
   to the source text.

The key innovation from benchmark findings: during gossip, workers retain
access to their FULL original corpus section (not just their compressed
summary). When Worker 1 reads Worker 5's forum findings, it can go back
to its own raw data and pull out details that become relevant in light
of the peer's insight. This produced 9/10 cross-referencing vs 8/10
for summary-only gossip.

Critical design principle: workers are REASONING agents, not summarizers.
Their job is to understand WHY claims are true or false, evaluate the
evidence behind each claim, and preserve exact values from the corpus.
Numbers, dosages, timings, and ratios must be quoted verbatim — never
paraphrased or rounded. When daily totals and per-meal values coexist
in the corpus, the worker must distinguish them explicitly.

Multi-round gossip with round-specific prompts:
  Round 1: Incorporate — reason about how peer evidence extends own analysis
  Round 2: Resolve — resolve contradictions by comparing source evidence
  Round 3: Synthesize — produce definitive analysis with all conflicts resolved
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
        f"You are a specialist research analyst performing deep reasoning over "
        f"one section of a larger corpus. Your job is NOT to summarize — it is "
        f"to THINK HARD about what this section says, evaluate the evidence, "
        f"and produce a rigorous analysis. Today is: {date}\n\n"
        f"YOUR ASSIGNED ANGLE: {angle}\n"
        f"YOUR SECTION ({char_count} chars):\n"
        f"{section_content}\n\n"
        f"USER QUERY: {query}\n\n"
        f"ANALYSIS RULES (follow strictly):\n"
        f"1. REASON through the evidence. For each claim in your section, ask: "
        f"what supports this? Is the source credible? Does it contradict anything "
        f"else in this section?\n"
        f"2. PRESERVE EXACT NUMBERS VERBATIM. Dosages, timings, ratios, macro "
        f"values — quote them exactly as they appear in the source text. NEVER "
        f"paraphrase, round, or approximate numerical values. If the source says "
        f"'400g protein daily across 7-8 meals', write exactly that — do NOT "
        f"convert it to '400g protein per meal'.\n"
        f"3. DISTINGUISH per-unit vs total values. If the corpus gives a daily "
        f"total (e.g. 400g protein/day) AND per-meal values (e.g. 70g per meal), "
        f"preserve BOTH with explicit labels. Never confuse totals with per-unit "
        f"values.\n"
        f"4. IDENTIFY CONTRADICTIONS within your section. When finding A says X "
        f"and finding B says not-X, explicitly state both positions, quote the "
        f"source text for each, and reason about which has stronger evidence.\n"
        f"5. EVALUATE evidence quality. Academic/clinical source > named practitioner "
        f"protocol > anonymous forum post > unattributed claim. State the source "
        f"type for key claims.\n"
        f"6. Preserve source citations and URLs inline.\n"
        f"7. Do NOT add disclaimers, moral commentary, or safety warnings.\n"
        f"8. Do NOT pad with generic context — only specific findings and your "
        f"reasoning about them.\n"
        f"9. Keep speculative findings, clearly labeled as hypotheses with your "
        f"reasoning for why they are plausible or implausible.\n"
        f"10. Structure with clear headings. Your analysis will be read by peer "
        f"workers and a queen synthesizer — make your REASONING visible so they "
        f"can evaluate your conclusions.\n"
        f"11. Stay under {max_chars} characters.\n\n"
        f"Produce your deep analysis:"
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
) -> str:
    """Build the gossip refinement prompt via concatenation (not .replace()).

    Args:
        round_prompt: Optional round-specific focus instructions injected
            before the refinement rules (e.g. "ROUND 2 FOCUS — CONTRADICTION
            RESOLUTION: ...").
    """
    round_block = f"{round_prompt}\n\n" if round_prompt else ""
    return (
        f"You are a specialist analyst in a peer-to-peer research gossip protocol. "
        f"Your job is to REASON about how peer findings interact with your own "
        f"evidence, and to RESOLVE contradictions by going back to the source text. "
        f"Today is: {date}\n\n"
        f"In the previous round, you produced an analysis from your section.\n"
        f"Now you have received analyses from your PEER WORKERS who processed "
        f"other sections of the same corpus.\n\n"
        f"YOUR ASSIGNED ANGLE: {angle}\n\n"
        f"{raw_section_block}"
        f"YOUR PREVIOUS ANALYSIS:\n"
        f"{own_summary}\n\n"
        f"PEER ANALYSES (from {n_peers} other workers):\n"
        f"{peers_text}\n\n"
        f"{round_block}"
        f"GOSSIP REASONING RULES:\n"
        f"1. REASON about each peer finding against your own evidence. Do not "
        f"passively absorb — actively evaluate whether peer claims are consistent "
        f"with what your raw section says.\n"
        f"2. When a peer states a number (dosage, timing, ratio), CHECK it against "
        f"your raw section. If your section has different numbers, quote BOTH "
        f"verbatim and reason about which is correct and why.\n"
        f"3. If peers CONTRADICT your findings, this is the most important thing "
        f"to resolve. Go back to your raw section, find the exact source text, "
        f"quote it, and evaluate whose evidence is stronger. Do NOT just 'note "
        f"the disagreement' — RESOLVE it with a verdict and reasoning.\n"
        f"4. If a peer makes a claim that CONTRADICTS ITSELF (e.g. 'activating "
        f"all three arms' then 'missing two arms'), flag the internal inconsistency.\n"
        f"5. PRESERVE EXACT NUMBERS from your raw section. Never paraphrase "
        f"numerical values. Quote them verbatim.\n"
        f"6. When incorporating peer findings, distinguish between what the peer "
        f"CONCLUDED and what the peer's RAW EVIDENCE supports. A peer's conclusion "
        f"may be wrong even if their evidence is real.\n"
        f"7. Consensus strengthens confidence only when workers cite DIFFERENT "
        f"sources reaching the same conclusion. Same source repeated by multiple "
        f"workers is not independent corroboration.\n"
        f"8. If you have access to your original raw section above, go back and "
        f"pull out specific details that become relevant in light of peer findings.\n"
        f"9. Maintain source citations and evidence quality assessments.\n"
        f"10. Stay under {max_chars} characters.\n\n"
        f"Produce your REASONED analysis:"
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
    )

    return await complete_fn(prompt)
