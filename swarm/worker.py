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
  Round 1: Incorporate — absorb peer findings into own analysis
  Round 2: Resolve — identify and resolve contradictions with peers
  Round 3: Synthesize — produce definitive refined analysis
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
        f"You are a specialist research analyst assigned to one section of a larger "
        f"corpus. Today is: {date}\n\n"
        f"YOUR ASSIGNED ANGLE: {angle}\n"
        f"YOUR SECTION ({char_count} chars):\n"
        f"{section_content}\n\n"
        f"USER QUERY: {query}\n\n"
        f"SYNTHESIS RULES:\n"
        f"1. Extract every concrete fact, name, data point, URL, source reference.\n"
        f"2. Preserve source citations inline.\n"
        f"3. Note confidence levels and verification status where apparent.\n"
        f"4. Mark contradictions between findings.\n"
        f"5. Do NOT add disclaimers, moral commentary, or safety warnings.\n"
        f"6. Do NOT pad with generic context — only specific findings.\n"
        f"7. Keep speculative findings, clearly labeled as hypotheses.\n"
        f"8. Structure with clear headings for different topics/angles.\n"
        f"9. Your summary will be merged with other workers' summaries, so focus "
        f"on what is UNIQUE and IMPORTANT in your section.\n"
        f"10. Stay under {max_chars} characters.\n\n"
        f"Produce your focused synthesis:"
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
        f"Today is: {date}\n\n"
        f"In the previous round, you produced a summary from your section.\n"
        f"Now you have received summaries from your PEER WORKERS who processed "
        f"other sections of the same corpus.\n\n"
        f"YOUR ASSIGNED ANGLE: {angle}\n\n"
        f"{raw_section_block}"
        f"YOUR PREVIOUS SUMMARY:\n"
        f"{own_summary}\n\n"
        f"PEER SUMMARIES (from {n_peers} other workers):\n"
        f"{peers_text}\n\n"
        f"{round_block}"
        f"GOSSIP REFINEMENT RULES:\n"
        f"1. Cross-reference your findings with peers'. Note agreements and contradictions.\n"
        f"2. If peers found information that COMPLEMENTS yours, incorporate key points.\n"
        f"3. If peers found the SAME information, note the consensus (strengthens confidence).\n"
        f"4. If peers CONTRADICT your findings, note the disagreement with both sources.\n"
        f"5. Do NOT simply concatenate — SYNTHESIZE and cross-reference.\n"
        f"6. Remove redundancy between your summary and peers'.\n"
        f"7. Preserve all unique findings from your original section.\n"
        f"8. If you have access to your original raw section above, go back and pull "
        f"out specific details that become relevant in light of peer findings.\n"
        f"9. Maintain source citations and confidence levels.\n"
        f"10. Stay under {max_chars} characters.\n\n"
        f"Produce your REFINED summary:"
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
