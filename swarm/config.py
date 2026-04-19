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
        "ROUND 1 FOCUS — INCORPORATION:\n"
        "Your primary task this round is to incorporate relevant findings "
        "from your peers into your analysis. Look for data points, sources, "
        "and mechanisms in peer summaries that complement or extend your own "
        "findings. Pull out specific details from your raw section that "
        "become relevant in light of what peers found."
    ),
    2: (
        "ROUND 2 FOCUS — CONTRADICTION RESOLUTION:\n"
        "This is your second gossip round. You have already incorporated "
        "peer findings. Now focus on identifying and RESOLVING contradictions "
        "between your analysis and your peers'. For each contradiction, "
        "evaluate source quality and evidence strength to determine which "
        "position is better supported. Note unresolvable disagreements "
        "explicitly with both positions and evidence for each."
    ),
    3: (
        "ROUND 3 FOCUS — FINAL SYNTHESIS:\n"
        "This is your final gossip round. You have incorporated peer findings "
        "and resolved contradictions. Now produce your DEFINITIVE refined "
        "analysis. Ensure every claim is cross-referenced against peer "
        "evidence. Remove any remaining redundancy. Your output will be "
        "the version read by the queen synthesizer, so make it as "
        "information-dense and well-structured as possible."
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
    lineage_store: LineageStore | None = None
    enable_quality_manifest: bool = True

    def __post_init__(self) -> None:
        """Resolve defaults that depend on other fields."""
        if self.max_concurrency <= 0:
            self.max_concurrency = self.max_workers
        # min_gossip_rounds must not exceed gossip_rounds
        if self.min_gossip_rounds > self.gossip_rounds:
            self.min_gossip_rounds = self.gossip_rounds
