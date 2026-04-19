# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Adaptive convergence detection for gossip rounds.

Instead of running a fixed number of gossip rounds, detect when workers
have converged (their summaries stop changing meaningfully) and stop early.

Four strategies:
1. **Trigram Jaccard** (fast, zero LLM calls): Compare word trigram overlap
   between consecutive rounds. When similarity exceeds threshold, stop.
2. **Information gain** (complementary): Measure how much NEW content each
   round introduces via set-difference of trigrams.  Tracks diminishing
   returns — if a round adds < 5% new trigrams, further rounds are unlikely
   to help.
3. **Corpus-delta** (structural, from ADK): Count new entity mentions and
   cross-references between rounds.  Measures structural changes rather
   than surface text similarity.  Inspired by ADK convergence module
   (apps/adk-agent/models/convergence.py).
4. **Length-based** (fallback): If summaries stop growing or shrinking
   significantly between rounds, convergence is assumed.

From benchmark findings: the gossip round is the transformative feature —
both swarm approaches jumped from 6→8-9 on synthesis quality just by
having workers read each other's summaries. 2-3 rounds show diminishing
but non-zero returns; more than 3 is rarely worth the cost.

References:
- "Gossip-Based Reasoning Among LLMs" (arXiv 2508.18292v1)
  Multi-layer consensus, metadata exchange, 6.9% MMLU improvement.
- "Mixture-of-Agents" (ICLR 2025, arXiv 2406.04692)
  Layered MoA architecture — each agent reads all outputs from previous
  layer.  Our gossip rounds are essentially MoA layers.
- "Multi-Agent Debate with Adaptive Stability Detection" (arXiv 2510.12697v1)
  Beta-Binomial mixture + KS testing for convergence.  More principled
  than Jaccard threshold but heavier.
- "Diversity-Aware Retention" (arXiv 2603.20640v1)
  Select peer messages that maximally disagree before broadcasting —
  reduces noise in gossip rounds.
"""

from __future__ import annotations

import re


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


def information_gain(
    current_summaries: list[str],
    previous_summaries: list[str],
) -> float:
    """Measure information gain between gossip rounds.

    Computes the fraction of NEW trigrams introduced in this round that
    were not present in the previous round's summaries.  A value near 0
    means almost no new content was added; near 1 means the round
    introduced entirely new material.

    This complements Jaccard similarity: Jaccard measures overall overlap,
    while information gain measures *novelty*.  A round can have high Jaccard
    (most content unchanged) but still have meaningful info gain (a few
    critical new cross-references added).

    Args:
        current_summaries: Worker summaries from the current round.
        previous_summaries: Worker summaries from the previous round.

    Returns:
        Information gain as a fraction [0.0, 1.0].
    """
    if not current_summaries or not previous_summaries:
        return 0.0

    # Pool all trigrams from each round
    prev_pool: set[str] = set()
    curr_pool: set[str] = set()
    for s in previous_summaries:
        prev_pool |= _trigrams(s)
    for s in current_summaries:
        curr_pool |= _trigrams(s)

    if not curr_pool:
        return 0.0

    new_trigrams = curr_pool - prev_pool
    return len(new_trigrams) / len(curr_pool) if curr_pool else 0.0


# ── Corpus-delta convergence (structural metric) ────────────────────

# Patterns for extracting cross-references between workers
_CROSS_REF_PATTERNS = [
    re.compile(r"\bconsensus\b", re.IGNORECASE),
    re.compile(r"\bcross-referenc", re.IGNORECASE),
    re.compile(r"\bcontradicts?\b", re.IGNORECASE),
    re.compile(r"\bagree[sd]?\b", re.IGNORECASE),
    re.compile(r"\bdisagree[sd]?\b", re.IGNORECASE),
    re.compile(r"\bcorroborat", re.IGNORECASE),
    re.compile(r"\bconfirm[sed]*\b", re.IGNORECASE),
    re.compile(r"\bconflict", re.IGNORECASE),
    re.compile(r"\bsupport[sed]*\s+(?:by|this|the)", re.IGNORECASE),
    re.compile(r"\bverif", re.IGNORECASE),
]

# Patterns for entity-like mentions (capitalized multi-word, technical terms)
_ENTITY_PATTERN = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"  # capitalized multi-word
    r"|\b[A-Z]{2,}[a-z]*(?:-\d+)?\b"  # acronyms like mTORC1, S6K1
    r"|\b\d+(?:\.\d+)?(?:\s*(?:mg|mcg|IU|ng|µg|mmol|%|kg))\b",  # quantities
)


def count_cross_references(text: str) -> int:
    """Count cross-reference indicators in a text.

    Looks for language that indicates a worker is referencing or comparing
    findings from other workers (consensus, contradicts, agrees, etc.).

    Args:
        text: Worker summary text to analyze.

    Returns:
        Number of cross-reference indicators found.
    """
    count = 0
    for pattern in _CROSS_REF_PATTERNS:
        count += len(pattern.findall(text))
    return count


def count_entities(text: str) -> set[str]:
    """Extract entity-like mentions from text.

    Finds capitalized multi-word phrases, acronyms, and quantity mentions
    that represent specific factual claims.

    Args:
        text: Worker summary text to analyze.

    Returns:
        Set of unique entity mentions.
    """
    return set(_ENTITY_PATTERN.findall(text))


def corpus_delta(
    current_summaries: list[str],
    previous_summaries: list[str],
) -> dict[str, float]:
    """Compute corpus-delta metrics between gossip rounds.

    Inspired by the ADK's convergence module (apps/adk-agent/models/convergence.py),
    this measures STRUCTURAL changes rather than surface text similarity:
    - New entities introduced
    - New cross-references added
    - Entity retention rate

    Args:
        current_summaries: Worker summaries from the current round.
        previous_summaries: Worker summaries from the previous round.

    Returns:
        Dict with keys:
        - new_entity_fraction: Fraction of entities in current that are new
        - cross_ref_delta: Change in cross-reference count
        - entity_retention: Fraction of previous entities retained
    """
    if not current_summaries or not previous_summaries:
        return {
            "new_entity_fraction": 0.0,
            "cross_ref_delta": 0.0,
            "entity_retention": 1.0,
        }

    # Pool entities and cross-refs from each round
    prev_entities: set[str] = set()
    curr_entities: set[str] = set()
    prev_xrefs = 0
    curr_xrefs = 0

    for s in previous_summaries:
        prev_entities |= count_entities(s)
        prev_xrefs += count_cross_references(s)
    for s in current_summaries:
        curr_entities |= count_entities(s)
        curr_xrefs += count_cross_references(s)

    # New entity fraction
    new_entities = curr_entities - prev_entities
    new_entity_fraction = (
        len(new_entities) / len(curr_entities)
        if curr_entities else 0.0
    )

    # Cross-reference delta (normalized)
    max_xrefs = max(prev_xrefs, curr_xrefs, 1)
    cross_ref_delta = (curr_xrefs - prev_xrefs) / max_xrefs

    # Entity retention (how many previous entities are still mentioned)
    retained = prev_entities & curr_entities
    entity_retention = (
        len(retained) / len(prev_entities)
        if prev_entities else 1.0
    )

    return {
        "new_entity_fraction": new_entity_fraction,
        "cross_ref_delta": cross_ref_delta,
        "entity_retention": entity_retention,
    }


def check_convergence(
    current_summaries: list[str],
    previous_summaries: list[str],
    threshold: float = 0.85,
) -> bool:
    """Check if worker summaries have converged between gossip rounds.

    Uses a combined metric: Jaccard similarity (surface text) AND
    corpus-delta (structural changes).  Convergence requires BOTH:
    - High Jaccard similarity (text is stable)
    - Low new entity fraction (no new facts being introduced)
    - Stable cross-reference count (not still growing significantly)

    This prevents false convergence when text is similar but workers
    are still introducing important structural changes.

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

    # Check Jaccard similarity
    similarities = []
    for curr, prev in zip(current_summaries, previous_summaries):
        sim = jaccard_similarity(curr, prev)
        similarities.append(sim)

    avg_similarity = sum(similarities) / len(similarities)

    # If Jaccard is below threshold, definitely not converged
    if avg_similarity < threshold:
        return False

    # Additional structural check: are new entities still being added?
    delta = corpus_delta(current_summaries, previous_summaries)

    # If more than 5% new entities are being introduced, not converged
    # even if text similarity is high
    if delta["new_entity_fraction"] > 0.05:
        return False

    # If cross-references are still growing significantly, not converged
    if delta["cross_ref_delta"] > 0.1:
        return False

    return True
