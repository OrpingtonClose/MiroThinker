# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Summary / final-answer agent definition.

Receives the conversation context and produces the final ``\\boxed{}``
answer.  Has no tools — summary only.  Replaces
``AnswerGenerator.generate_final_answer_with_retries()``.
"""

from __future__ import annotations

import os

from google.adk import Agent

from prompts.templates import build_main_summary_prompt

_MODEL = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")

# The instruction is generic; the actual task-specific summary prompt
# is injected as the user message at runtime (see main.py).
summary_agent = Agent(
    name="summary_agent",
    model=_MODEL,
    description="Summarizes research findings and produces a final boxed answer.",
    instruction=(
        "You are a summarization agent. You receive a conversation history and "
        "produce a concise, well-formatted final answer wrapped in \\boxed{}. "
        "Do NOT call any tools."
    ),
    tools=[],  # no tools — summary only
)
