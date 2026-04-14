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

ASYNC: This callback is async so it can write the thinker's thought
directly to DuckDB under the per-corpus ``asyncio.Lock``, eliminating
the old deferred-admission pattern (``_pending_thinker_thought``).
ADK supports async callbacks (``inspect.isawaitable`` check in
``base_agent.py``).
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from dashboard import get_active_collector

logger = logging.getLogger(__name__)

_SENTINEL = "EVIDENCE_SUFFICIENT"


async def thinker_escalate_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Check if the thinker signalled that enough evidence has been gathered.

    Reads the thinker's ``output_key`` value (``research_strategy``) from
    session state.  If the text contains the ``EVIDENCE_SUFFICIENT`` sentinel
    the callback sets ``escalate = True`` so the ``LoopAgent`` exits.

    Also tracks thinker reasoning across iterations for context injection
    and convergence detection.

    ASYNC: Writes the thinker's thought directly to DuckDB under the
    per-corpus ``asyncio.Lock``, eliminating the old deferred-admission
    pattern.  ADK awaits async callbacks (``inspect.isawaitable`` check).

    Returns ``None`` so the thinker's original output is preserved.
    """
    state = callback_context.state
    strategy = state.get("research_strategy", "")

    # ── THOUGHT LINEAGE: Write thinker thought directly under async lock ──
    # Previously this was deferred to search_executor_callback because the
    # background scoring thread might still hold DuckDB.  With the async
    # lock pattern, we can safely write here — the lock serialises all
    # DuckDB access without blocking the event loop.
    if strategy and strategy.strip():
        iteration = state.get("_corpus_iteration", 0)
        corpus_key = state.get("_corpus_key", "default")

        try:
            from callbacks.condition_manager import _safe_corpus_write, _get_corpus

            corpus = _get_corpus(state)

            await _safe_corpus_write(
                corpus_key,
                corpus.admit_thought,
                strategy,                              # reasoning
                "thinker_reasoning",                    # angle
                f"thinker_iteration_{iteration}",       # strategy
                iteration,                              # iteration
            )
            logger.info(
                "Thinker thought admitted directly: %d chars at iteration %d",
                len(strategy), iteration,
            )
        except Exception:
            logger.warning(
                "Failed to admit thinker thought (non-fatal)",
                exc_info=True,
            )

        # Keep a condensed version in state for the prompt context
        # (thought rows are in the corpus briefing, but this provides
        # a quick summary the thinker prompt can reference directly).
        prev = state.get("_prev_thinker_strategies", "")
        separator = f"\n--- Iteration {iteration} reasoning ---\n"
        new_history = prev + separator + strategy[:500]
        if len(new_history) > 2000:
            new_history = new_history[-2000:]
        state["_prev_thinker_strategies"] = new_history

    # ── P1: Convergence detection ──
    # If the thinker's reasoning is very similar to the previous one,
    # the pipeline has converged and should stop early.
    if strategy and not (strategy and _SENTINEL in strategy):
        prev_strategy = state.get("_last_thinker_strategy", "")
        if prev_strategy and _strategies_converged(strategy, prev_strategy):
            logger.info(
                "Convergence detected — thinker reasoning is repeating "
                "previous iteration. Escalating."
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
    return None


def _strategies_converged(current: str, previous: str) -> bool:
    """Detect if two strategies are substantively the same.

    Primary path: LLM-powered semantic comparison that understands
    whether the thinker is making genuine intellectual progress or
    repeating the same ideas with different words.

    Falls back to keyword overlap if the LLM is unavailable.
    """
    if not current or not previous:
        return False

    # Primary: LLM-powered convergence check
    try:
        from utils.flock_proxy import get_flock_proxy_url
        import json as _json
        import urllib.request

        proxy_url = get_flock_proxy_url()
        if proxy_url:
            prompt = (
                "Compare these two research strategies from consecutive "
                "iterations.  Is the second one making GENUINE NEW "
                "INTELLECTUAL PROGRESS, or is it repeating the same ideas "
                "(possibly with different vocabulary)?\n\n"
                "Signs of real progress: new research angles, deeper "
                "questions, new connections, refined hypotheses.\n"
                "Signs of repetition: same questions rephrased, same "
                "angles with minor rewording, no new intellectual content.\n\n"
                "Return ONLY one word: PROGRESSING or CONVERGED\n\n"
                f"PREVIOUS STRATEGY:\n{previous[:1500]}\n\n"
                f"CURRENT STRATEGY:\n{current[:1500]}"
            )
            body = _json.dumps({
                "model": "flock-model",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0.1,
            }).encode()
            req = urllib.request.Request(
                f"{proxy_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            choices = data.get("choices", [])
            answer = (choices[0]["message"]["content"] if choices else "").strip().strip(".,!?\n").upper()
            if answer == "CONVERGED":
                logger.info("LLM convergence check: CONVERGED")
                return True
            if answer in ("PROGRESSING", "PROGRESS"):
                logger.info("LLM convergence check: PROGRESSING")
                return False
    except Exception:
        pass  # fall through to keyword heuristic

    # Fallback: keyword overlap
    import re as _re

    def _keywords(text: str) -> set[str]:
        words = set(_re.findall(r'\b[a-z]{4,}\b', text.lower()))
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
