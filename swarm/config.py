# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm engine configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Awaitable


# Type alias for the LLM completion callable.
# Accepts a prompt string, returns the completion string.
# This makes the swarm engine model-agnostic — any backend works.
CompleteFn = Callable[[str], Awaitable[str]]


@dataclass
class SwarmConfig:
    """Configuration for the gossip swarm engine.

    All values have sensible defaults and can be overridden via environment
    variables (prefixed with ``SWARM_``).

    Attributes:
        max_workers: Maximum parallel specialist workers per phase.
        gossip_rounds: Number of gossip refinement rounds (0 = map-only).
        max_summary_chars: Maximum chars per worker summary.
        max_section_chars: Maximum chars per corpus section assigned to a worker.
        convergence_threshold: Cosine similarity threshold for adaptive stopping.
        context_budget: Maximum tokens the queen merge prompt should target.
        enable_serendipity: Whether to run the cross-angle serendipity bridge.
        enable_full_corpus_gossip: If True, workers retain their original raw
            corpus section during gossip rounds (not just compressed summaries).
        enable_adaptive_rounds: If True, stop gossip early when convergence
            is detected across workers.
        worker_temperature: Temperature for worker LLM calls.
        queen_temperature: Temperature for the queen merge call.
        worker_max_tokens: Max output tokens per worker call.
        queen_max_tokens: Max output tokens for the queen merge.
    """

    max_workers: int = int(os.getenv("SWARM_MAX_WORKERS", "6"))
    gossip_rounds: int = int(os.getenv("SWARM_GOSSIP_ROUNDS", "1"))
    max_summary_chars: int = int(os.getenv("SWARM_MAX_SUMMARY_CHARS", "6000"))
    max_section_chars: int = int(os.getenv("SWARM_MAX_SECTION_CHARS", "30000"))
    convergence_threshold: float = float(os.getenv("SWARM_CONVERGENCE_THRESHOLD", "0.85"))
    context_budget: int = int(os.getenv("SWARM_CONTEXT_BUDGET", "100000"))
    enable_serendipity: bool = os.getenv("SWARM_SERENDIPITY", "1") == "1"
    enable_full_corpus_gossip: bool = os.getenv("SWARM_FULL_CORPUS_GOSSIP", "1") == "1"
    enable_adaptive_rounds: bool = os.getenv("SWARM_ADAPTIVE_ROUNDS", "1") == "1"
    worker_temperature: float = 0.3
    queen_temperature: float = 0.3
    worker_max_tokens: int = 4096
    queen_max_tokens: int = 8192
