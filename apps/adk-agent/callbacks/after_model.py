# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-model callback implementing Algorithm 7 (Intermediate Boxed Extraction).

After every model response, scan the text for ``\\boxed{...}`` and, if found,
append the extracted answer to ``state["intermediate_boxed_answers"]``.

These intermediate answers serve as a fallback if the final summary agent
fails to produce a valid boxed answer.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext

from callbacks.before_model import release_llm_semaphore_if_held
from dashboard import get_active_collector
from utils.boxed import extract_boxed_content

# Lazy import to avoid circular dependency at module load time.
# _get_corpus lives in callbacks.condition_manager which imports models
# that may reference this module's logger.
_get_corpus = None


def _lazy_get_corpus():
    global _get_corpus
    if _get_corpus is None:
        from callbacks.condition_manager import _get_corpus as _gc
        _get_corpus = _gc
    return _get_corpus

logger = logging.getLogger(__name__)


def after_model_callback(
    callback_context: CallbackContext, llm_response: Any
) -> Optional[Any]:
    """
    ADK after_model_callback.

    1. Releases the LLM concurrency semaphore acquired in
       ``before_model_callback`` so the next worker can proceed.
    2. Scans the model response for \\boxed{} content and stores it in
       session state as a fallback answer.

    Returns None to keep the original response unchanged.
    """
    state = callback_context.state

    # Release LLM concurrency semaphore (acquired in before_model_callback)
    release_llm_semaphore_if_held(state)

    if "intermediate_boxed_answers" not in state:
        state["intermediate_boxed_answers"] = []

    # Extract text and reasoning from LlmResponse.content
    response_text = ""
    reasoning_text = ""
    if llm_response and getattr(llm_response, "content", None):
        content = llm_response.content
        if hasattr(content, "parts") and content.parts:
            for part in content.parts:
                # Capture reasoning/thinking content from reasoning models.
                # When part.thought is True, the part's .text contains
                # reasoning content — capture it separately so it doesn't
                # bloat the conversation context.
                if hasattr(part, "thought") and part.thought:
                    if hasattr(part, "text") and part.text:
                        reasoning_text += part.text
                elif hasattr(part, "thinking") and part.thinking:
                    reasoning_text += str(part.thinking)
                elif hasattr(part, "text") and part.text:
                    response_text += part.text

    if not response_text and not reasoning_text:
        return None

    # ── Reasoning content capture ────────────────────────────────────
    # Persist full reasoning to DuckDB as a row_type='thought' row via
    # CorpusStore.admit_thought().  Only a lightweight reference is kept
    # in session state (agent name, char counts, thought row ID) to
    # avoid state bloat.
    if reasoning_text:
        if "reasoning_traces" not in state:
            state["reasoning_traces"] = []
        agent_name = getattr(callback_context, "agent_name", "unknown")
        thought_row_id = None
        try:
            get_corpus = _lazy_get_corpus()
            corpus = get_corpus(state)
            thought_row_id = corpus.admit_thought(
                reasoning=reasoning_text,
                angle=agent_name,
                strategy="reasoning_capture",
                iteration=state.get("_corpus_iteration", 0),
            )
        except Exception:
            logger.debug(
                "Could not persist reasoning to DuckDB (non-fatal)",
                exc_info=True,
            )
        state["reasoning_traces"].append({
            "agent": agent_name,
            "reasoning_chars": len(reasoning_text),
            "response_preview_chars": len(response_text),
            "thought_row_id": thought_row_id,
        })
        logger.info(
            "Reasoning content captured from %s: %d chars (thought_row_id=%s)",
            agent_name, len(reasoning_text), thought_row_id,
        )

    # ── Boxed answer extraction ──────────────────────────────────────
    if response_text:
        boxed = extract_boxed_content(response_text)
        if boxed:
            state["intermediate_boxed_answers"].append(boxed)
            logger.info("Intermediate boxed answer captured: %s", boxed[:200])

    # Record LLM end in dashboard — but ONLY for actual completions,
    # not partial streaming chunks.  ADK fires after_model_callback on
    # both token-by-token streaming events AND final completions.
    # Guard: only count as a completion if we got substantive text
    # (>20 chars rules out single-token streaming fragments).
    _c = get_active_collector()
    if _c and len(response_text) > 20:
        agent_name = getattr(callback_context, "agent_name", "")
        _c.llm_end(agent_name, 0.0, len(response_text) // 4)

    return None  # keep the original response
