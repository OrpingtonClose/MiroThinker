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
"""

from __future__ import annotations

from datetime import datetime, timezone


WORKER_SYNTH_PROMPT = """\
You are a specialist research analyst assigned to one section of a larger \
corpus. Today is: {date}

YOUR ASSIGNED ANGLE: {angle}
YOUR SECTION ({char_count} chars):
{section_content}

USER QUERY: {query}

SYNTHESIS RULES:
1. Extract every concrete fact, name, data point, URL, source reference.
2. Preserve source citations inline.
3. Note confidence levels and verification status where apparent.
4. Mark contradictions between findings.
5. Do NOT add disclaimers, moral commentary, or safety warnings.
6. Do NOT pad with generic context — only specific findings.
7. Keep speculative findings, clearly labeled as hypotheses.
8. Structure with clear headings for different topics/angles.
9. Your summary will be merged with other workers' summaries, so focus \
on what is UNIQUE and IMPORTANT in your section.
10. Stay under {max_chars} characters.

Produce your focused synthesis:"""


GOSSIP_REFINE_PROMPT = """\
You are a specialist analyst in a peer-to-peer research gossip protocol. \
Today is: {date}

In the previous round, you produced a summary from your section.
Now you have received summaries from your PEER WORKERS who processed \
other sections of the same corpus.

YOUR ASSIGNED ANGLE: {angle}

{raw_section_block}

YOUR PREVIOUS SUMMARY:
{own_summary}

PEER SUMMARIES (from {n_peers} other workers):
{peer_summaries}

GOSSIP REFINEMENT RULES:
1. Cross-reference your findings with peers'. Note agreements and contradictions.
2. If peers found information that COMPLEMENTS yours, incorporate key points.
3. If peers found the SAME information, note the consensus (strengthens confidence).
4. If peers CONTRADICT your findings, note the disagreement with both sources.
5. Do NOT simply concatenate — SYNTHESIZE and cross-reference.
6. Remove redundancy between your summary and peers'.
7. Preserve all unique findings from your original section.
8. If you have access to your original raw section above, go back and pull \
out specific details that become relevant in light of peer findings.
9. Maintain source citations and confidence levels.
10. Stay under {max_chars} characters.

Produce your REFINED summary:"""


async def worker_synthesize(
    angle: str,
    section_content: str,
    query: str,
    max_chars: int,
    complete_fn,
) -> str:
    """Phase 1: Worker synthesizes its assigned corpus section."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = WORKER_SYNTH_PROMPT.replace("{date}", date) \
        .replace("{angle}", angle) \
        .replace("{char_count}", str(len(section_content))) \
        .replace("{section_content}", section_content) \
        .replace("{query}", query) \
        .replace("{max_chars}", str(max_chars))

    return await complete_fn(prompt)


async def worker_gossip_refine(
    angle: str,
    own_summary: str,
    peer_summaries: list[str],
    raw_section: str | None,
    query: str,
    max_chars: int,
    complete_fn,
) -> str:
    """Phase 2: Worker refines its summary using peer gossip.

    If raw_section is provided (full-corpus gossip mode), the worker
    can reference its original raw data while incorporating peer insights.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build raw section block — only included in full-corpus gossip mode
    if raw_section:
        raw_section_block = (
            f"YOUR ORIGINAL RAW SECTION (reference for detail lookup):\n"
            f"{raw_section}\n"
        )
    else:
        raw_section_block = ""

    peers_text = ""
    for i, ps in enumerate(peer_summaries):
        peers_text += f"\n--- Worker {i + 1} ---\n{ps[:max_chars]}\n"

    prompt = GOSSIP_REFINE_PROMPT.replace("{date}", date) \
        .replace("{angle}", angle) \
        .replace("{raw_section_block}", raw_section_block) \
        .replace("{own_summary}", own_summary[:max_chars]) \
        .replace("{n_peers}", str(len(peer_summaries))) \
        .replace("{peer_summaries}", peers_text) \
        .replace("{max_chars}", str(max_chars))

    return await complete_fn(prompt)
