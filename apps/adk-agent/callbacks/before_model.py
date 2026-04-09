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
3. **Hard context truncation** — if the estimated token count exceeds
   ``HARD_CONTEXT_LIMIT``, truncates old function_response parts to stay
   within the model's actual context window.  This prevents 400 errors
   from providers when the model ignores the soft wrap-up instruction.
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


# Rough chars-per-token ratio for estimating context size.
# GLM / Qwen / most modern tokenizers average ~2.8 chars/token for mixed
# content (English prose + JSON + tool schemas).  The old value of 4 was
# too generous and led to 400 errors when actual token counts exceeded
# our estimate by ~1.4×.
_CHARS_PER_TOKEN = 2.8

# Fixed overhead for system prompt, tool definitions, and message framing
# that isn't captured in the contents list.  ~15K tokens is conservative
# for MiroThinker's multi-tool setup.
_OVERHEAD_TOKENS = int(os.environ.get("CONTEXT_OVERHEAD_TOKENS", "15000"))

# Maximum estimated tokens before forcing a final answer
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "128000"))

# Hard limit: truncate old function responses to stay within the model's
# actual context window.  Set well below the provider's hard limit to
# leave headroom for system prompt, tool definitions, and model response.
# GLM-5.1 has a 202K hard limit; 120K target keeps us safe.
HARD_CONTEXT_LIMIT = int(os.environ.get("HARD_CONTEXT_LIMIT", "120000"))


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
    return int(total_chars / _CHARS_PER_TOKEN) + _OVERHEAD_TOKENS


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

    # ── Hard context truncation ────────────────────────────────────────
    # If the model ignored the soft wrap-up and context is still growing,
    # truncate old function_response parts so the LLM call doesn't 400.
    # Strategy: walk backwards through contents, find function_response
    # parts, and replace their response text with a short summary marker.
    # We keep the most recent responses intact and only trim older ones.
    if estimated > HARD_CONTEXT_LIMIT:
        _truncate_old_responses(contents, HARD_CONTEXT_LIMIT)
        new_est = _estimate_tokens(contents)
        logger.warning(
            "Hard context truncation: %d -> %d estimated tokens "
            "(limit %d)",
            estimated, new_est, HARD_CONTEXT_LIMIT,
        )
        estimated = new_est

    # Record LLM start in dashboard
    _c = get_active_collector()
    if _c:
        agent_name = getattr(callback_context, "agent_name", "")
        _c.llm_start(agent_name, estimated)

    return None  # proceed with the (modified) request


def _truncate_old_responses(
    contents: List[genai_types.Content],
    target_tokens: int,
) -> None:
    """Truncate old function_response parts to fit within *target_tokens*.

    Walks backwards through the contents list, collecting indices of
    function_response parts.  Skips the most recent 4 responses (so the
    model has immediate context), then truncates older ones from oldest
    to newest until the estimated token count is below the target.
    """
    # Collect (content_idx, part_idx, char_count) for all function_response parts
    fr_parts: list[tuple[int, int, int]] = []
    for ci, content in enumerate(contents):
        if not content.parts:
            continue
        for pi, part in enumerate(content.parts):
            if hasattr(part, "function_response") and part.function_response:
                resp = getattr(part.function_response, "response", None)
                char_count = len(str(resp)) if resp else 0
                fr_parts.append((ci, pi, char_count))

    if len(fr_parts) <= 4:
        # Too few responses to truncate — keep them all
        return

    # Truncate from oldest, skipping the 4 most recent
    truncatable = fr_parts[:-4]
    current_est = _estimate_tokens(contents)

    _TRUNCATION_MARKER = "[content truncated to fit context window]"

    for ci, pi, char_count in truncatable:
        if current_est <= target_tokens:
            break
        part = contents[ci].parts[pi]
        fr = part.function_response
        if fr and getattr(fr, "response", None):
            saved_chars = char_count - len(_TRUNCATION_MARKER)
            if saved_chars > 100:  # only truncate if meaningful savings
                fr.response = {"result": _TRUNCATION_MARKER}
                current_est -= saved_chars // _CHARS_PER_TOKEN
                logger.debug(
                    "Truncated function_response at content[%d].parts[%d] "
                    "(saved ~%d tokens)",
                    ci, pi, saved_chars // _CHARS_PER_TOKEN,
                )
