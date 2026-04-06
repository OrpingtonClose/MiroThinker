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

# ── LLM concurrency gate ────────────────────────────────────────────
# Limits how many concurrent LLM calls can be in-flight at once.
# Prevents Venice/provider rate limiting when running parallel workers.
# ADK-native: this is just an asyncio.Semaphore inside a callback.
_MAX_CONCURRENT_LLM = int(os.environ.get("MAX_CONCURRENT_LLM", "2"))
_llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)

# Default number of recent tool results to keep (matches keep_tool_result=5)
DEFAULT_KEEP_K = int(os.environ.get("KEEP_TOOL_RESULT", "5"))

# Rough chars-per-token ratio for estimating context size
_CHARS_PER_TOKEN = 4

# Maximum estimated tokens before forcing a final answer
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "128000"))

_PLACEHOLDER = "Tool result is omitted to save tokens."


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
    2. Trims old tool results via Keep-K-Recent (Algorithm 5).
    3. Checks context length and signals force_end if too large.

    Returns None (ADK proceeds with the — possibly modified — request).
    """
    # Acquire semaphore — blocks if too many LLM calls in flight.
    # ADK will await this coroutine, so other workers can proceed
    # while we wait for a slot.
    await _llm_semaphore.acquire()
    # Release happens in after_model_callback via _llm_semaphore.release().
    # Store on callback_context state so after_model can find it.
    callback_context.state["_llm_sem_held"] = True
    logger.debug("LLM semaphore acquired (%d/%d slots used)",
                 _MAX_CONCURRENT_LLM - _llm_semaphore._value,
                 _MAX_CONCURRENT_LLM)
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
            and any(hasattr(p, "function_response") and p.function_response for p in content.parts)
        ):
            tool_indices.append(idx)

    if keep_k >= 0 and len(tool_indices) > keep_k:
        # Replace all but the last K tool results with placeholder.
        # IMPORTANT: Also replace the corresponding FunctionCall parts in
        # the preceding model message to maintain the FunctionCall →
        # FunctionResponse pairing required by both Gemini and OpenAI APIs.
        #
        # To avoid corrupting parallel tool call groups (where a single
        # model message contains multiple FunctionCalls with the same name),
        # we first group tool responses by their originating model message.
        # If the trim boundary falls inside a group, we adjust it so the
        # entire group is either trimmed or kept.
        raw_split = len(tool_indices) - keep_k

        # Map each tool_index to the model message that issued its FunctionCall
        def _find_model_msg(resp_idx: int) -> int:
            """Find the model message index that issued the FunctionCall for resp_idx."""
            for pi in range(resp_idx - 1, -1, -1):
                prev = contents[pi]
                if getattr(prev, "role", None) == "model" and prev.parts:
                    if any(hasattr(p, "function_call") and p.function_call for p in prev.parts):
                        return pi
            return -1

        # Check if the split point falls inside a parallel tool call group
        # (i.e., the last trimmed and first kept share the same model message)
        if raw_split > 0 and raw_split < len(tool_indices):
            last_trimmed_model = _find_model_msg(tool_indices[raw_split - 1])
            first_kept_model = _find_model_msg(tool_indices[raw_split])
            if last_trimmed_model == first_kept_model and last_trimmed_model != -1:
                # Adjust split to keep the entire group — move split point
                # back to before this group starts
                while raw_split > 0 and _find_model_msg(tool_indices[raw_split - 1]) == last_trimmed_model:
                    raw_split -= 1

        indices_to_trim = tool_indices[:raw_split]

        for idx in indices_to_trim:
            # Replace the tool response content with a placeholder.
            # IMPORTANT: Leave the corresponding FunctionCall parts in
            # model messages UNTOUCHED to maintain the FunctionCall →
            # FunctionResponse structural pairing required by both
            # Gemini and OpenAI APIs. Only the bulky response content
            # is replaced — this mirrors the original MiroThinker approach.
            # IMPORTANT: role="tool" messages MUST contain function_response
            # parts (not plain text) to satisfy the API schema. For role="user"
            # messages that happen to carry function_response parts, plain text
            # is acceptable.
            orig_role = getattr(contents[idx], "role", "user")
            if orig_role == "tool":
                # Build proper FunctionResponse placeholders preserving names
                placeholder_parts = []
                for part in (contents[idx].parts or []):
                    if hasattr(part, "function_response") and part.function_response:
                        fr_name = getattr(part.function_response, "name", "unknown")
                        placeholder_parts.append(genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=fr_name,
                                response={"result": _PLACEHOLDER},
                            )
                        ))
                if not placeholder_parts:
                    # Fallback: shouldn't happen, but use text under "user" role
                    placeholder_parts = [genai_types.Part(text=_PLACEHOLDER)]
                    orig_role = "user"
                contents[idx] = genai_types.Content(
                    role=orig_role, parts=placeholder_parts
                )
            else:
                contents[idx] = genai_types.Content(
                    role=orig_role,
                    parts=[genai_types.Part(text=_PLACEHOLDER)],
                )
        omitted_count = len(indices_to_trim)
        logger.info(
            "Keep-K-Recent: trimmed %d tool results, kept last %d",
            omitted_count,
            len(tool_indices) - omitted_count,
        )
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

    return None  # proceed with the (modified) request
