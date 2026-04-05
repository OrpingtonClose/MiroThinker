# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-model callback implementing Algorithm 5 (Keep-K-Recent).

Trims the conversation history so that only the initial user message and
the last *K* tool-result messages are fully retained.  Older tool results
are replaced with a short placeholder to save tokens.

Also performs a rough context-length check.  If the estimated token count
exceeds a configurable threshold the ``force_end`` flag is set in session
state so the agent instruction can trigger a final answer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# Default number of recent tool results to keep (matches keep_tool_result=5)
DEFAULT_KEEP_K = int(os.environ.get("KEEP_TOOL_RESULT", "5"))

# Rough chars-per-token ratio for estimating context size
_CHARS_PER_TOKEN = 4

# Maximum estimated tokens before forcing a final answer
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "128000"))

_PLACEHOLDER = "Tool result is omitted to save tokens."


def _estimate_tokens(contents: List[genai_types.Content]) -> int:
    """Rough token estimate based on total character count."""
    total_chars = 0
    for content in contents:
        if content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    total_chars += len(part.text)
    return total_chars // _CHARS_PER_TOKEN


def before_model_callback(
    callback_context: CallbackContext, llm_request: Any
) -> Optional[genai_types.Content]:
    """
    ADK before_model_callback.

    Modifies the LLM request in-place to trim old tool results and
    optionally signals the agent to produce a final answer when context
    is too large.

    Returns None (ADK proceeds with the — possibly modified — request).
    """
    state = callback_context.state
    keep_k = state.get("keep_k", DEFAULT_KEEP_K)

    # Access the contents list from the LLM request
    contents: Optional[List[genai_types.Content]] = getattr(
        llm_request, "contents", None
    )
    if not contents:
        return None

    # ── Algorithm 5: Keep-K-Recent ──────────────────────────────────────
    # Identify indices of tool-role messages (function responses)
    tool_indices: List[int] = []
    first_user_idx: Optional[int] = None

    for idx, content in enumerate(contents):
        role = getattr(content, "role", None)
        if role == "user" and first_user_idx is None:
            first_user_idx = idx
        # In ADK, tool results come back as role="tool" or with
        # function_response parts
        if role == "tool" or (
            role == "user"
            and idx != first_user_idx
            and content.parts
            and any(hasattr(p, "function_response") for p in content.parts)
        ):
            tool_indices.append(idx)

    collector = state.get("_dashboard_collector")

    if keep_k >= 0 and len(tool_indices) > keep_k:
        # Replace all but the last K tool results with placeholder
        indices_to_trim = tool_indices[: len(tool_indices) - keep_k]
        for idx in indices_to_trim:
            contents[idx] = genai_types.Content(
                role=contents[idx].role,
                parts=[genai_types.Part(text=_PLACEHOLDER)],
            )
        omitted_count = len(indices_to_trim)
        logger.info(
            "Keep-K-Recent: trimmed %d tool results, kept last %d",
            omitted_count,
            keep_k,
        )
        if collector:
            from dashboard.models import DashboardEvent, EventType

            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    collector.emit(
                        DashboardEvent(
                            event_type=EventType.CONTEXT_TRIMMED,
                            turn=collector.current_turn,
                            data={
                                "total_tool_results": len(tool_indices),
                                "kept_results": keep_k,
                                "omitted_count": omitted_count,
                                "total_messages": len(contents),
                            },
                        )
                    )
                ),
            )

    # ── Context length check ────────────────────────────────────────────
    estimated = _estimate_tokens(contents)

    # Emit LLM_CALL_START with token estimate
    if collector:
        from dashboard.models import DashboardEvent, EventType

        asyncio.get_event_loop().call_soon(
            lambda: asyncio.ensure_future(
                collector.emit(
                    DashboardEvent(
                        event_type=EventType.LLM_CALL_START,
                        turn=collector.current_turn,
                        data={
                            "estimated_prompt_tokens": estimated,
                            "max_context_tokens": MAX_CONTEXT_TOKENS,
                        },
                    )
                )
            ),
        )

    if estimated > MAX_CONTEXT_TOKENS:
        state["force_end"] = True
        logger.warning(
            "Context length estimate (%d tokens) exceeds threshold (%d). "
            "Setting force_end=True and injecting wrap-up instruction.",
            estimated,
            MAX_CONTEXT_TOKENS,
        )
        if collector:
            asyncio.get_event_loop().call_soon(
                lambda: asyncio.ensure_future(
                    collector.emit(
                        DashboardEvent(
                            event_type=EventType.FORCE_END_TRIGGERED,
                            turn=collector.current_turn,
                            data={
                                "estimated_tokens": estimated,
                                "threshold": MAX_CONTEXT_TOKENS,
                            },
                        )
                    )
                ),
            )
        # Inject a system-level message telling the model to wrap up now
        force_end_msg = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=(
                "[SYSTEM] Your context window is nearly full. You MUST produce "
                "your final \\boxed{} answer NOW without making any more tool "
                "calls. Summarize what you have found and give your best answer."
            ))],
        )
        contents.append(force_end_msg)

    return None  # proceed with the (modified) request
