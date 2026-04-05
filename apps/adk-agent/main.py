# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Entry point for the MiroThinker ADK agent.

Implements Algorithm 6 (Context Compression Retry): an outer retry loop
that creates a *fresh* ADK session on each attempt, prepending failure
experience summaries from prior attempts to the user message.

Each retry attempt:
  1. Creates a new InMemorySession (empty history).
  2. Prepends any accumulated failure experiences.
  3. Runs the research agent.
  4. Checks for a valid \\boxed{} answer.
  5. If no valid answer and retries remain, generates a failure summary
     and loops.
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.research import research_agent
from agents.summary import summary_agent
from prompts.templates import (
    FAILURE_EXPERIENCE_FOOTER,
    FAILURE_EXPERIENCE_HEADER,
    FAILURE_EXPERIENCE_ITEM,
    FAILURE_SUMMARY_PROMPT,
    build_main_summary_prompt,
)
from utils.boxed import extract_boxed_content

logger = logging.getLogger(__name__)

APP_NAME = "mirothinker-adk"
USER_ID = "user"
CONTEXT_COMPRESS_LIMIT = int(os.environ.get("CONTEXT_COMPRESS_LIMIT", "5"))


async def _collect_response_text(runner: Runner, user_id: str, session_id: str, message: str) -> str:
    """Run an agent and collect the full response text."""
    collected = ""
    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=message)]
    )
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    collected += part.text
    return collected


async def main(task: str | None = None) -> str:
    """
    Run the MiroThinker ADK agent with context-compression retry.

    Args:
        task: The research question / task description.  If ``None`` a
              default demo question is used.

    Returns:
        The final extracted answer string.
    """
    if task is None:
        task = "What is the title of today's arxiv paper in computer science?"

    session_service = InMemorySessionService()
    failure_experiences: list[str] = []
    final_answer: str | None = None

    for attempt in range(1, CONTEXT_COMPRESS_LIMIT + 1):
        is_final = attempt == CONTEXT_COMPRESS_LIMIT
        logger.info("=== Attempt %d / %d ===", attempt, CONTEXT_COMPRESS_LIMIT)

        # Build user message with failure experiences from prior attempts
        user_message = task
        if failure_experiences:
            prefix = FAILURE_EXPERIENCE_HEADER
            for i, fe in enumerate(failure_experiences):
                prefix += FAILURE_EXPERIENCE_ITEM.format(
                    attempt_number=i + 1, failure_summary=fe
                )
            prefix += FAILURE_EXPERIENCE_FOOTER
            user_message = prefix + "\n\n" + task

        # Create fresh session (empty history) for each attempt
        session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID
        )

        # Run the research agent
        runner = Runner(
            agent=research_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        research_result = await _collect_response_text(
            runner, USER_ID, session.id, user_message
        )

        # Check for valid \boxed{} answer
        boxed = extract_boxed_content(research_result)

        # Also check intermediate answers captured by after_model_callback
        intermediate = session.state.get("intermediate_boxed_answers", [])

        if boxed and boxed not in ("?", "unknown"):
            final_answer = boxed
            logger.info("Valid answer found on attempt %d: %s", attempt, boxed[:200])
            break

        if is_final and intermediate:
            # Last chance: use the most recent intermediate answer
            final_answer = intermediate[-1]
            logger.info(
                "Using intermediate answer on final attempt: %s",
                final_answer[:200],
            )
            break

        if not is_final:
            # Generate a failure summary for the next attempt
            summary_session = await session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID
            )
            summary_runner = Runner(
                agent=summary_agent,
                app_name=APP_NAME,
                session_service=session_service,
            )

            failure_text = await _collect_response_text(
                summary_runner, USER_ID, summary_session.id, FAILURE_SUMMARY_PROMPT
            )
            failure_experiences.append(failure_text)
            logger.info(
                "Failure summary for attempt %d: %s", attempt, failure_text[:300]
            )

    if final_answer is None:
        final_answer = "(No answer could be determined)"

    print(f"\nFinal answer: {final_answer}\n")
    return final_answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
