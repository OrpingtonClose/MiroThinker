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
from google.genai import types as genai_types

from callbacks.before_model import release_llm_semaphore_if_held
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

    # Extract text from LlmResponse.content (not .candidates)
    response_text = ""
    if llm_response and getattr(llm_response, "content", None):
        content = llm_response.content
        if hasattr(content, "parts") and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

    if not response_text:
        return None

    boxed = extract_boxed_content(response_text)
    if boxed:
        state["intermediate_boxed_answers"].append(boxed)
        logger.info("Intermediate boxed answer captured: %s", boxed[:200])

    return None  # keep the original response
