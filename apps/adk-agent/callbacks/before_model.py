# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-model callback — LLM concurrency gate + context-length safety net.

Keep-K-Recent (Algorithm 5), Adaptive K, and dynamic compression have been
removed — they are now handled by ``ContextFilterPlugin`` which trims at the
invocation level (simpler, framework-native, and sufficient because
MiroThinker's important data flows through session *state*, not raw
conversation context).

What remains:
1. **LLM concurrency semaphore** — limits parallel LLM calls to
   ``MAX_CONCURRENT_LLM`` so providers (Venice, RunPod) aren't rate-limited
   when running batch workers.
2. **Context-length safety net** — if the estimated token count exceeds
   ``MAX_CONTEXT_TOKENS``, sets ``force_end=True`` in session state and
   injects a wrap-up instruction so the agent produces a final answer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from dashboard import get_active_collector

logger = logging.getLogger(__name__)

# ── LLM concurrency gate ────────────────────────────────────────────
# Limits how many concurrent LLM calls can be in-flight at once.
# Prevents Venice/provider rate limiting when running parallel workers.
# ADK-native: this is just an asyncio.Semaphore inside a callback.
_MAX_CONCURRENT_LLM = int(os.environ.get("MAX_CONCURRENT_LLM", "2"))
_llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)


def release_llm_semaphore_if_held(state: dict) -> None:
    """Safety release — call from after_model or any error-cleanup path."""
    if state.get("_llm_sem_held"):
        _llm_semaphore.release()
        state["_llm_sem_held"] = False


# Rough chars-per-token ratio for estimating context size
_CHARS_PER_TOKEN = 4

# Maximum estimated tokens before forcing a final answer
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "128000"))


def _estimate_tokens(contents: List[genai_types.Content]) -> int:
    """Rough token estimate based on total character count.

    Counts text, function_call (name + args), and function_response
    (name + response) parts so that tool-heavy conversations are not
    under-counted.
    """
    total_chars = 0
    for content in contents:
        if content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    total_chars += len(part.text)
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    total_chars += len(getattr(fc, "name", "") or "")
                    args = getattr(fc, "args", None)
                    if args:
                        total_chars += len(str(args))
                if hasattr(part, "function_response") and part.function_response:
                    fr = part.function_response
                    total_chars += len(getattr(fr, "name", "") or "")
                    resp = getattr(fr, "response", None)
                    if resp:
                        total_chars += len(str(resp))
    return total_chars // _CHARS_PER_TOKEN


async def before_model_callback(
    callback_context: CallbackContext, llm_request: Any
) -> Optional[genai_types.Content]:
    """
    ADK before_model_callback (async).

    1. Acquires the LLM concurrency semaphore — limits parallel LLM
       calls to ``MAX_CONCURRENT_LLM`` (default 2) so providers like
       Venice don't get rate-limited when running batch workers.
    2. Checks context length and signals force_end if too large.

    Returns None (ADK proceeds with the — possibly modified — request).
    """
    # Acquire semaphore — blocks if too many LLM calls in flight.
    await _llm_semaphore.acquire()
    callback_context.state["_llm_sem_held"] = True
    logger.debug(
        "LLM semaphore acquired (%d/%d slots used)",
        _MAX_CONCURRENT_LLM - _llm_semaphore._value,
        _MAX_CONCURRENT_LLM,
    )

    # Access the contents list from the LLM request
    contents: Optional[List[genai_types.Content]] = getattr(
        llm_request, "contents", None
    )
    if not contents:
        return None

    state = callback_context.state

    # ── Context length check ────────────────────────────────────────────
    estimated = _estimate_tokens(contents)

    if estimated > MAX_CONTEXT_TOKENS:
        if not state.get("force_end"):
            state["force_end"] = True
            logger.warning(
                "Context length estimate (%d tokens) exceeds threshold (%d). "
                "Setting force_end=True and injecting wrap-up instruction.",
                estimated,
                MAX_CONTEXT_TOKENS,
            )
            _c = get_active_collector()
            if _c:
                _c.force_end(estimated)
        # Re-inject on every call since contents are rebuilt from session
        # history (the injected message is not persisted to the session)
        report_mode = state.get("report_mode", False)
        if report_mode:
            wrap_up_text = (
                "[SYSTEM] Your context window is nearly full. You MUST produce "
                "your final answer NOW without making any more tool calls. "
                "Summarize what you have found into a comprehensive report."
            )
        else:
            wrap_up_text = (
                "[SYSTEM] Your context window is nearly full. You MUST produce "
                "your final \\boxed{} answer NOW without making any more tool "
                "calls. Summarize what you have found and give your best answer."
            )
        force_end_msg = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=wrap_up_text)],
        )
        contents.append(force_end_msg)

    # Record LLM start in dashboard
    _c = get_active_collector()
    if _c:
        agent_name = getattr(callback_context, "agent_name", "")
        _c.llm_start(agent_name, estimated)

    return None  # proceed with the (modified) request
