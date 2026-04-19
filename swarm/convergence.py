# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Adaptive convergence detection for gossip rounds.

Instead of running a fixed number of gossip rounds, detect when workers
have converged (their summaries stop changing meaningfully) and stop early.

Two strategies:
1. **Trigram Jaccard** (fast, zero LLM calls): Compare word trigram overlap
   between consecutive rounds. When similarity exceeds threshold, stop.
2. **Length-based** (fallback): If summaries stop growing or shrinking
   significantly between rounds, convergence is assumed.

From benchmark findings: the gossip round is the transformative feature —
both swarm approaches jumped from 6→8-9 on synthesis quality just by
having workers read each other's summaries. But more than 1-2 rounds
typically shows diminishing returns.
"""

from __future__ import annotations


def _trigrams(text: str) -> set[str]:
    """Extract word trigrams from text."""
    words = text.lower().split()
    if len(words) < 3:
        return set(words)
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts using word trigrams.

    Returns 0.0 (completely different) to 1.0 (identical).
    """
    if not text_a or not text_b:
        return 0.0

    tri_a = _trigrams(text_a)
    tri_b = _trigrams(text_b)

    if not tri_a or not tri_b:
        return 0.0

    intersection = len(tri_a & tri_b)
    union = len(tri_a | tri_b)

    return intersection / union if union > 0 else 0.0


def check_convergence(
    current_summaries: list[str],
    previous_summaries: list[str],
    threshold: float = 0.85,
) -> bool:
    """Check if worker summaries have converged between gossip rounds.

    Convergence is declared when the AVERAGE Jaccard similarity across
    all workers exceeds the threshold. This means most workers are no
    longer producing meaningfully different content.

    Args:
        current_summaries: Worker summaries from the current round.
        previous_summaries: Worker summaries from the previous round.
        threshold: Similarity threshold for convergence (0.0-1.0).

    Returns:
        True if workers have converged (stop gossip early).
    """
    if len(current_summaries) != len(previous_summaries):
        return False

    if not current_summaries:
        return True

    similarities = []
    for curr, prev in zip(current_summaries, previous_summaries):
        sim = jaccard_similarity(curr, prev)
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity >= threshold
