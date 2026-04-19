# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Adapter to integrate the new GossipSwarm engine with the existing proxy layer.

The existing proxies use ``ruflo_gossip_synthesize()`` from
``proxies/tools/ruflo_synthesis.py``. This adapter provides a drop-in
replacement that routes through the new swarm engine instead.

Usage in ``proxies/tools/synthesis.py``:

    # Toggle via env var: SWARM_ENGINE=v2 (default: v1 for backward compat)
    if os.getenv("SWARM_ENGINE", "v1") == "v2":
        from swarm.adapter import gossip_synthesize_v2
        result = await gossip_synthesize_v2(user_query, subagent_results, req_id)
    else:
        result = await ruflo_gossip_synthesize(user_query, subagent_results, req_id)

The adapter converts between the proxy layer's SubagentResult/AtomicCondition
types and the swarm engine's plain-text corpus interface.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from swarm.config import SwarmConfig
from swarm.engine import GossipSwarm, SwarmResult

if TYPE_CHECKING:
    from proxies.tools.models import SubagentResult


async def _build_complete_fn():
    """Build an LLM completion callable using the proxy layer's call_llm.

    Reuses the existing proxy infrastructure (model selection, API keys,
    rate limiting, langfuse tracing) so the swarm engine benefits from
    the same LLM backend without reimplementing any of it.
    """
    from proxies.tools.config import UPSTREAM_MODEL
    from proxies.tools.llm import call_llm

    async def complete(prompt: str) -> str:
        result = await call_llm(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Produce your analysis."},
            ],
            req_id="swarm-v2",
            model=UPSTREAM_MODEL,
            max_tokens=4096,
            temperature=0.3,
        )
        if "error" in result:
            return ""
        return result.get("content", "")

    return complete


def _subagent_results_to_corpus(
    subagent_results: list[SubagentResult],
) -> str:
    """Convert SubagentResult list to a plain-text corpus for the swarm engine.

    Each SubagentResult contains conditions grouped by research angle.
    We reconstruct a structured corpus with angle headers.
    """
    sections: list[str] = []
    for sr in subagent_results:
        if not sr.conditions:
            continue
        angle_text = f"## {sr.angle}\n\n"
        for c in sr.conditions:
            angle_text += f"{c.to_text()}\n\n"
        sections.append(angle_text)

    return "\n".join(sections)


async def gossip_synthesize_v2(
    user_query: str,
    subagent_results: list[SubagentResult],
    req_id: str,
    prior_text: str = "",
) -> str:
    """Drop-in replacement for ruflo_gossip_synthesize using the v2 engine.

    Args:
        user_query: The user's research query.
        subagent_results: List of SubagentResult from the research net.
        req_id: Request ID for tracing.
        prior_text: Optional prior context (conversation history, etc.)

    Returns:
        Final synthesized answer string, or empty string if below threshold.
    """
    # Convert proxy types to plain corpus text
    corpus = _subagent_results_to_corpus(subagent_results)
    if not corpus.strip():
        return ""

    # Prepend prior context if available
    if prior_text:
        corpus = f"## Prior Context\n{prior_text}\n\n{corpus}"

    # Build completion function from proxy infrastructure
    complete_fn = await _build_complete_fn()

    # Run the swarm
    config = SwarmConfig()
    swarm = GossipSwarm(complete=complete_fn, config=config)
    result = await swarm.synthesize(corpus=corpus, query=user_query)

    return result.synthesis
