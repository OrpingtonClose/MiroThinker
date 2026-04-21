# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm engine configuration.

Supports per-phase model selection so workers, queen, and serendipity
bridge can each use the optimal model for their role:

- Workers: fast, uncensored, parallel (e.g. local Gemma-4 on Ollama)
- Queen: best available writer (e.g. Qwen-Claude-Opus or remote Claude)
- Serendipity: best cross-domain reasoner

An uncensored model with equivalent capability is strictly better than
its censored counterpart — it is a superset.  The model hierarchy
defaults to uncensored-first, with censored fallback only when the
capability gap justifies it for non-sensitive topics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from swarm.lineage import LineageStore


# Type alias for the LLM completion callable.
# Accepts a prompt string, returns the completion string.
# This makes the swarm engine model-agnostic — any backend works.
CompleteFn = Callable[[str], Awaitable[str]]


# Round-specific gossip prompt modifiers.
# Each entry is injected as an extra instruction block into the gossip prompt
# for that round number.  Missing entries → no extra instruction.
DEFAULT_ROUND_PROMPTS: dict[int, str] = {
    1: (
        "ROUND 1 FOCUS — CONNECTION DISCOVERY:\n"
        "Your primary task is to find where your peers' findings EXPLAIN "
        "something in your own evidence. For each peer's key finding, ask: "
        "does this illuminate a pattern, anomaly, or mechanism in my raw "
        "section? Where do our domains COLLIDE in a way that produces new "
        "understanding? Trace the causal chain — if a peer's molecular "
        "mechanism could explain the practitioner result in your section, "
        "show the COMPLETE path from mechanism to outcome. Don't just note "
        "overlaps — find the moments where combining two domains reveals "
        "something neither stated alone. Preserve exact numbers verbatim."
    ),
    2: (
        "ROUND 2 FOCUS — CAUSAL DEPTH:\n"
        "Go deeper into the connections you and your peers found in Round 1. "
        "For each connection: (a) trace it back to specific evidence in your "
        "raw section AND the peer's cited sources, (b) ask 'what does this "
        "connection PREDICT?' — if mechanism X explains outcome Y, what else "
        "should be true? (c) look for SECOND-ORDER connections — does your "
        "connection with Peer A, combined with Peer B's findings, reveal "
        "something none of you stated? Resolve numerical contradictions by "
        "comparing source evidence quality (academic > named practitioner > "
        "anonymous forum post). The depth here creates the surface area for "
        "serendipity."
    ),
    3: (
        "ROUND 3 FOCUS — GAPS AND FINAL SYNTHESIS:\n"
        "What's still unexplained? What connections did you find that you "
        "couldn't fully resolve because you're missing specific data? State "
        "these gaps explicitly as research questions (e.g. 'Need bloodwork "
        "data showing mTOR activation despite short GH-insulin window'). "
        "Then produce your FINAL analysis: a single coherent narrative that "
        "connects everything you know across all peer domains, at maximum "
        "depth. Every claim must trace to evidence. Every connection must "
        "be grounded. The connections — moments where one domain illuminates "
        "another — are the primary output. Numbers must be exact and sourced."
    ),
}


@dataclass
class SwarmConfig:
    """Configuration for the gossip swarm engine.

    All values have sensible defaults and can be overridden via environment
    variables (prefixed with ``SWARM_``).

    Per-phase model selection:
        Pass ``worker_complete``, ``queen_complete``, and/or
        ``serendipity_complete`` to ``GossipSwarm`` to use different
        models for each phase.  If not provided, the single ``complete``
        function is used for all phases.

    Attributes:
        max_workers: Maximum parallel specialist workers per phase.
        max_concurrency: Maximum concurrent LLM calls within a phase.
            Defaults to ``max_workers`` (fully parallel).  Set lower
            for rate-limited APIs.
        gossip_rounds: Number of gossip refinement rounds (0 = map-only).
        min_gossip_rounds: Minimum rounds before adaptive stopping is allowed.
            Ensures workers get at least N rounds of cross-referencing even
            if Jaccard similarity is high early on.
        max_summary_chars: Maximum chars per worker summary.
        max_section_chars: Maximum chars per corpus section assigned to a worker.
        convergence_threshold: Jaccard similarity threshold for adaptive stopping.
        context_budget: Maximum tokens the queen merge prompt should target.
        enable_serendipity: Whether to run the cross-angle serendipity bridge.
        enable_full_corpus_gossip: If True, workers retain their original raw
            corpus section during gossip rounds (not just compressed summaries).
        enable_adaptive_rounds: If True, stop gossip early when convergence
            is detected across workers.
        round_prompts: Per-round prompt modifiers for gossip.  Keys are
            1-indexed round numbers, values are instruction blocks injected
            into the gossip prompt.
        worker_temperature: Temperature for worker LLM calls.
        queen_temperature: Temperature for the queen merge call.
        worker_max_tokens: Max output tokens per worker call.
        queen_max_tokens: Max output tokens for the queen merge.
        enable_semantic_assignment: If True, use LLM-scored semantic matching
            to optimally assign sections to angles (Hungarian algorithm).
            Costs 1 extra LLM call but ensures each section goes to the
            specialist who would extract the most value.
        enable_diversity_aware_gossip: If True, apply Diversity-Aware
            Retention (DAR) during gossip — select the ``dar_top_k`` most
            disagreeing peer summaries instead of all peers.  Reduces
            echo-chamber effects (arXiv 2603.20640v1).
        dar_top_k: Number of most-diverse peers to retain per worker
            during DAR gossip filtering.
    """

    max_workers: int = int(os.getenv("SWARM_MAX_WORKERS", "6"))
    max_concurrency: int = int(os.getenv("SWARM_MAX_CONCURRENCY", "0"))  # 0 = max_workers
    gossip_rounds: int = int(os.getenv("SWARM_GOSSIP_ROUNDS", "3"))
    min_gossip_rounds: int = int(os.getenv("SWARM_MIN_GOSSIP_ROUNDS", "2"))
    max_summary_chars: int = int(os.getenv("SWARM_MAX_SUMMARY_CHARS", "6000"))
    max_section_chars: int = int(os.getenv("SWARM_MAX_SECTION_CHARS", "30000"))
    convergence_threshold: float = float(os.getenv("SWARM_CONVERGENCE_THRESHOLD", "0.85"))
    context_budget: int = int(os.getenv("SWARM_CONTEXT_BUDGET", "100000"))
    enable_serendipity: bool = os.getenv("SWARM_SERENDIPITY", "1") == "1"
    enable_full_corpus_gossip: bool = os.getenv("SWARM_FULL_CORPUS_GOSSIP", "1") == "1"
    enable_adaptive_rounds: bool = os.getenv("SWARM_ADAPTIVE_ROUNDS", "1") == "1"
    round_prompts: dict[int, str] = field(default_factory=lambda: dict(DEFAULT_ROUND_PROMPTS))
    worker_temperature: float = 0.3
    queen_temperature: float = 0.3
    worker_max_tokens: int = 4096
    queen_max_tokens: int = 8192
    enable_semantic_assignment: bool = os.getenv("SWARM_SEMANTIC_ASSIGNMENT", "1") == "1"
    enable_diversity_aware_gossip: bool = os.getenv("SWARM_DAR_GOSSIP", "1") == "1"
    dar_top_k: int = int(os.getenv("SWARM_DAR_TOP_K", "3"))
    lineage_store: LineageStore | None = None
    enable_quality_manifest: bool = True
    corpus_delta_fn: "Callable[[], Awaitable[str]] | None" = None
    max_gossip_rounds: int = int(os.getenv("SWARM_MAX_GOSSIP_ROUNDS", "10"))

    # ── Three-tier topology settings ──────────────────────────────
    enable_hierarchy: bool = os.getenv("SWARM_HIERARCHY", "1") == "1"
    max_leaf_size: int = int(os.getenv("SWARM_MAX_LEAF_SIZE", "25"))
    min_leaf_size: int = int(os.getenv("SWARM_MIN_LEAF_SIZE", "5"))
    bridge_rounds: int = int(os.getenv("SWARM_BRIDGE_ROUNDS", "3"))
    serendipity_panel_size: int = int(os.getenv("SWARM_PANEL_SIZE", "5"))
    coordinator_max_chars: int = int(os.getenv("SWARM_COORD_MAX_CHARS", "4000"))
    enable_serendipity_panel: bool = os.getenv("SWARM_PANEL", "1") == "1"
    enable_queen_lucidity_pass: bool = os.getenv("SWARM_LUCIDITY", "1") == "1"

    def __post_init__(self) -> None:
        """Resolve defaults that depend on other fields."""
        if self.max_concurrency <= 0:
            self.max_concurrency = self.max_workers
        # min_gossip_rounds must not exceed gossip_rounds
        if self.min_gossip_rounds > self.gossip_rounds:
            self.min_gossip_rounds = self.gossip_rounds
        # max_gossip_rounds must be >= gossip_rounds
        if self.max_gossip_rounds < self.gossip_rounds:
            self.max_gossip_rounds = self.gossip_rounds
