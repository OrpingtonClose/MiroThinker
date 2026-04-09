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
                if hasattr(part, "text") and part.text:
                    response_text += part.text
                # Capture reasoning/thinking content from reasoning models.
                # Different providers expose this differently:
                #   - thought/thinking attribute on the part
                #   - reasoning_content in the part metadata
                if hasattr(part, "thought") and part.thought:
                    reasoning_text += str(part.thought)
                elif hasattr(part, "thinking") and part.thinking:
                    reasoning_text += str(part.thinking)

    if not response_text and not reasoning_text:
        return None

    # ── Reasoning content capture ────────────────────────────────────
    # Save reasoning traces to state for observability, but do NOT feed
    # them back into the conversation context (that wastes tokens).
    if reasoning_text:
        if "reasoning_traces" not in state:
            state["reasoning_traces"] = []
        agent_name = getattr(callback_context, "agent_name", "unknown")
        state["reasoning_traces"].append({
            "agent": agent_name,
            "reasoning": reasoning_text[:5000],  # cap to avoid state bloat
            "response_preview": response_text[:500],
        })
        logger.info(
            "Reasoning content captured from %s: %d chars",
            agent_name, len(reasoning_text),
        )

    # ── Boxed answer extraction ──────────────────────────────────────
    if response_text:
        boxed = extract_boxed_content(response_text)
        if boxed:
            state["intermediate_boxed_answers"].append(boxed)
            logger.info("Intermediate boxed answer captured: %s", boxed[:200])

    # Record LLM end in dashboard
    _c = get_active_collector()
    if _c:
        agent_name = getattr(callback_context, "agent_name", "")
        _c.llm_end(agent_name, 0.0, len(response_text) // 4)  # rough token estimate

    return None  # keep the original response
