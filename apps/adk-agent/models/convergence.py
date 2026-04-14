# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Corpus-delta convergence metric (Phase 6).

Replaces the text-similarity heuristic in ThinkerBlock with a
ground-truth metric based on actual corpus changes:

    Converged = last N iterations added <3 findings,
                0 contradictions, quality didn't improve.

This is the structural signal that research has saturated —
not an LLM judgement about whether strategies "look similar."
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Minimum findings per iteration to consider research productive.
_MIN_NEW_FINDINGS = 3

# Number of recent iterations to check for convergence.
_LOOKBACK_ITERATIONS = 2


def check_convergence(
    corpus: object,
    current_iteration: int,
    current_strategy: str,
    previous_strategy: str,
) -> bool:
    """Check if the research pipeline has converged.

    Uses corpus-delta metric: compares actual corpus changes across
    the last N iterations instead of text similarity between strategies.

    Args:
        corpus: The CorpusStore instance (must have a .conn attribute).
        current_iteration: The current iteration number.
        current_strategy: Current thinker strategy (for fallback).
        previous_strategy: Previous thinker strategy (for fallback).

    Returns:
        True if the pipeline has converged (should exit).
    """
    if current_iteration < _LOOKBACK_ITERATIONS:
        return False

    conn = getattr(corpus, "conn", None)
    if conn is None:
        return False

    try:
        return _check_corpus_delta(conn, current_iteration)
    except Exception:
        logger.debug("Corpus-delta convergence check failed, falling back to keyword overlap")
        return _keyword_fallback(current_strategy, previous_strategy)


def _check_corpus_delta(conn: object, current_iteration: int) -> bool:
    """Check convergence via actual corpus changes."""

    # Count findings added in the last N iterations
    lookback_start = max(0, current_iteration - _LOOKBACK_ITERATIONS)
    try:
        recent_findings = conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE row_type = 'finding' AND iteration >= ?",
            [lookback_start],
        ).fetchone()[0]
    except Exception:
        return False

    if recent_findings >= _MIN_NEW_FINDINGS * _LOOKBACK_ITERATIONS:
        # Still finding enough new things — not converged
        return False

    # Check if any contradictions were found recently
    try:
        recent_contradictions = conn.execute(
            "SELECT COUNT(*) FROM conditions "
            "WHERE contradiction_flag = TRUE AND iteration >= ?",
            [lookback_start],
        ).fetchone()[0]
    except Exception:
        recent_contradictions = 0

    if recent_contradictions > 0:
        # Finding contradictions = research is still productive
        return False

    # Check quality improvement: compare average quality of recent vs older findings
    try:
        recent_avg = conn.execute(
            "SELECT AVG(composite_quality) FROM conditions "
            "WHERE row_type = 'finding' AND iteration >= ? "
            "AND composite_quality > 0",
            [lookback_start],
        ).fetchone()[0]

        older_avg = conn.execute(
            "SELECT AVG(composite_quality) FROM conditions "
            "WHERE row_type = 'finding' AND iteration < ? "
            "AND composite_quality > 0",
            [lookback_start],
        ).fetchone()[0]

        if recent_avg is not None and older_avg is not None:
            # Quality improved by >5% — still making progress
            if recent_avg > older_avg * 1.05:
                return False
    except Exception:
        pass

    logger.info(
        "Corpus-delta convergence: last %d iterations added only %d "
        "findings (threshold: %d), 0 contradictions, no quality improvement",
        _LOOKBACK_ITERATIONS, recent_findings,
        _MIN_NEW_FINDINGS * _LOOKBACK_ITERATIONS,
    )
    return True


def _keyword_fallback(current: str, previous: str) -> bool:
    """Fallback convergence check using keyword overlap."""
    if not current or not previous:
        return False

    import re

    def _keywords(text: str) -> set[str]:
        words = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
        stops = {
            "this", "that", "with", "from", "have", "been", "will",
            "would", "could", "should", "about", "which", "their",
            "there", "these", "those", "then", "than", "when", "what",
            "search", "find", "look", "also", "more", "most", "some",
            "into", "each", "such", "much", "very", "just", "only",
        }
        return words - stops

    curr_kw = _keywords(current)
    prev_kw = _keywords(previous)
    if not curr_kw or len(curr_kw) < 5:
        return False
    overlap = len(curr_kw & prev_kw) / len(curr_kw)
    return overlap > 0.80
