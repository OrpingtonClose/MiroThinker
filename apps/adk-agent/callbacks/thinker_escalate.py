# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-agent callback for the thinker inside a LoopAgent.

When the thinker outputs text containing the sentinel ``EVIDENCE_SUFFICIENT``,
this callback sets ``escalate=True`` on the event actions so that the
enclosing ``LoopAgent`` breaks out of the research loop and hands off to
the synthesiser.

This keeps the thinker 100 % tool-free: it signals "done" via plain text,
and the callback translates that into an ADK-native escalation event.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from dashboard import get_active_collector

logger = logging.getLogger(__name__)

_SENTINEL = "EVIDENCE_SUFFICIENT"


def thinker_escalate_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Check if the thinker signalled that enough evidence has been gathered.

    Reads the thinker's ``output_key`` value (``research_strategy``) from
    session state.  If the text contains the ``EVIDENCE_SUFFICIENT`` sentinel
    the callback sets ``escalate = True`` so the ``LoopAgent`` exits.

    Also tracks thinker strategies across iterations for context injection
    and convergence detection.

    Returns ``None`` so the thinker's original output is preserved.
    """
    state = callback_context.state
    strategy = state.get("research_strategy", "")

    # ── P1: Track strategies for iteration context injection ──
    # Save a condensed summary of this strategy for the next iteration's
    # thinker prompt so it knows what was tried before.
    if strategy and strategy.strip():
        iteration = state.get("_corpus_iteration", 0)
        # Keep last ~500 chars per iteration to avoid prompt bloat
        summary = strategy[:500]
        prev = state.get("_prev_thinker_strategies", "")
        separator = f"\n--- Iteration {iteration} strategy ---\n"
        # Cap total history at ~2000 chars to prevent prompt overflow
        new_history = prev + separator + summary
        if len(new_history) > 2000:
            new_history = new_history[-2000:]
        state["_prev_thinker_strategies"] = new_history

    # ── P1: Convergence detection ──
    # If the thinker's strategy is very similar to the previous one,
    # the pipeline has converged and should stop early.
    if strategy and not (strategy and _SENTINEL in strategy):
        prev_strategy = state.get("_last_thinker_strategy", "")
        if prev_strategy and _strategies_converged(strategy, prev_strategy):
            logger.info(
                "Convergence detected — thinker strategy is repeating "
                "previous iteration's queries. Escalating."
            )
            callback_context.actions.escalate = True
            _c = get_active_collector()
            if _c:
                _c.emit_event("convergence_detected", data={
                    "iteration": state.get("_corpus_iteration", 0),
                })
        state["_last_thinker_strategy"] = strategy

    if _SENTINEL in strategy:
        logger.info("Thinker signalled EVIDENCE_SUFFICIENT — escalating out of research loop")
        callback_context.actions.escalate = True
        _c = get_active_collector()
        if _c:
            _c.thinker_escalate()

    # ── Pipeline health gate: thinker ─────────────────────────────
    try:
        from models.pipeline_health import PipelineHealth, check_thinker
        health = PipelineHealth.from_state(state)
        iteration = state.get("_corpus_iteration", 0)
        phase = health.begin_phase(f"thinker_iter{iteration}")
        phase.metrics["strategy_length"] = len(strategy)
        # Count extractable queries for health check
        try:
            from tools.search_executor import extract_search_queries
            queries = extract_search_queries(strategy)
            phase.metrics["extractable_queries"] = len(queries)
        except Exception:
            phase.metrics["extractable_queries"] = 0
        check_thinker(phase, state)
        health.evaluate_gate(phase)
        health.save(state)
    except Exception:
        pass  # health tracking is best-effort

    return None


def _strategies_converged(current: str, previous: str) -> bool:
    """Detect if two strategies are substantively the same.

    Uses keyword overlap as a lightweight convergence signal.
    If >80% of the meaningful words in the current strategy appeared
    in the previous one, the thinker is repeating itself.
    """
    import re as _re

    def _keywords(text: str) -> set[str]:
        words = set(_re.findall(r'\b[a-z]{4,}\b', text.lower()))
        # Remove common stop words
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
