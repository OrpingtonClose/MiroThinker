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

        # Track which FunctionCalls have already been replaced to handle
        # parallel calls correctly (use positional index within model message)
        replaced_fc_positions: dict[int, set[int]] = {}  # model_idx -> set of part positions

        for idx in indices_to_trim:
            # Find the function name(s) in this tool response
            response_names: list[str] = []
            for part in (contents[idx].parts or []):
                if hasattr(part, "function_response") and part.function_response:
                    name = getattr(part.function_response, "name", None)
                    if name:
                        response_names.append(name)

            # Search backwards for the nearest model message containing
            # matching FunctionCall parts.
            model_idx = _find_model_msg(idx)
            if model_idx >= 0:
                prev = contents[model_idx]
                if model_idx not in replaced_fc_positions:
                    replaced_fc_positions[model_idx] = set()

                new_parts = []
                names_to_replace = list(response_names)  # consume one per match
                for part_pos, part in enumerate(prev.parts):
                    if (
                        hasattr(part, "function_call") and part.function_call
                        and part_pos not in replaced_fc_positions[model_idx]
                    ):
                        fc_name = getattr(part.function_call, "name", "")
                        if fc_name in names_to_replace:
                            # Replace this specific FunctionCall (by position)
                            names_to_replace.remove(fc_name)
                            replaced_fc_positions[model_idx].add(part_pos)
                            new_parts.append(genai_types.Part(
                                text=f"[Called {fc_name} — result omitted to save tokens]"
                            ))
                        else:
                            new_parts.append(part)
                    else:
                        new_parts.append(part)
                contents[model_idx] = genai_types.Content(
                    role="model", parts=new_parts
                )

            # Replace the tool response with a text placeholder
            contents[idx] = genai_types.Content(
                role=contents[idx].role,
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

    if estimated > MAX_CONTEXT_TOKENS and not state.get("force_end"):
        state["force_end"] = True
        logger.warning(
            "Context length estimate (%d tokens) exceeds threshold (%d). "
            "Setting force_end=True and injecting wrap-up instruction.",
            estimated,
            MAX_CONTEXT_TOKENS,
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
