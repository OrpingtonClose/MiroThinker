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
import threading
import uuid

from dotenv import load_dotenv

load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agents.research import research_agent
from agents.summary import summary_agent
from dashboard.collector import DashboardCollector
from dashboard.models import DashboardEvent, EventType
from dashboard.registry import clear_collector, set_collector
from dashboard.server import app as dashboard_app
from dashboard.server import collectors as dashboard_collectors
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
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))
DASHBOARD_ENABLED = os.environ.get("DASHBOARD_ENABLED", "1") != "0"


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


def _start_dashboard_server() -> None:
    """Start the FastAPI dashboard server in a background thread."""
    import uvicorn

    def _run():
        uvicorn.run(
            dashboard_app,
            host="0.0.0.0",
            port=DASHBOARD_PORT,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Dashboard server started on http://localhost:%d", DASHBOARD_PORT)


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

    # ── Dashboard setup ──────────────────────────────────────────────────
    collector: DashboardCollector | None = None
    session_id = str(uuid.uuid4())[:8]

    if DASHBOARD_ENABLED:
        _start_dashboard_server()
        collector = DashboardCollector(session_id=session_id, query=task)
        dashboard_collectors[session_id] = collector
        set_collector(collector)  # module-level global for callbacks
        print(f"\n  Dashboard: http://localhost:{DASHBOARD_PORT}/?session={session_id}\n")

        await collector.emit(
            DashboardEvent(
                event_type=EventType.SESSION_START,
                data={"query": task, "max_attempts": CONTEXT_COMPRESS_LIMIT},
            )
        )

    session_service = InMemorySessionService()
    failure_experiences: list[str] = []
    final_answer: str | None = None
    attempt = 0  # initialise so it's defined even if CONTEXT_COMPRESS_LIMIT is 0

    for attempt in range(1, CONTEXT_COMPRESS_LIMIT + 1):
        is_final = attempt == CONTEXT_COMPRESS_LIMIT
        logger.info("=== Attempt %d / %d ===", attempt, CONTEXT_COMPRESS_LIMIT)

        # Emit retry-attempt event
        if collector:
            collector.start_retry(attempt, CONTEXT_COMPRESS_LIMIT)
            await collector.emit(
                DashboardEvent(
                    event_type=EventType.RETRY_ATTEMPT,
                    attempt=attempt,
                    data={
                        "attempt_number": attempt,
                        "max_attempts": CONTEXT_COMPRESS_LIMIT,
                        "is_final": is_final,
                    },
                )
            )

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

        # Collector is accessible via dashboard.registry (module-level global)
        # ADK's InMemorySessionService doesn't persist arbitrary objects in state

        # Run the research agent
        runner = Runner(
            agent=research_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        research_result = await _collect_response_text(
            runner, USER_ID, session.id, user_message
        )

        # Check for valid \boxed{} answer in the research result
        boxed = extract_boxed_content(research_result)

        # Also check intermediate answers captured by after_model_callback
        intermediate = session.state.get("intermediate_boxed_answers", [])

        if boxed and boxed not in ("?", "unknown"):
            final_answer = boxed
            logger.info("Valid answer found on attempt %d: %s", attempt, boxed[:200])
            if collector and collector.retry_attempts:
                collector.retry_attempts[-1].answer_found = True
                collector.retry_attempts[-1].answer_source = "research"
                collector.retry_attempts[-1].finish()
            break

        # No boxed answer from research — run explicit summarization step
        summary_prompt = build_main_summary_prompt(task)
        summary_session = await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID
        )
        summary_runner = Runner(
            agent=summary_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        # Feed the research result as context followed by the summary prompt
        summary_input = research_result + "\n\n" + summary_prompt
        summary_result = await _collect_response_text(
            summary_runner, USER_ID, summary_session.id, summary_input
        )
        boxed = extract_boxed_content(summary_result)
        if boxed and boxed not in ("?", "unknown"):
            final_answer = boxed
            logger.info(
                "Answer found via summarization on attempt %d: %s",
                attempt,
                boxed[:200],
            )
            if collector and collector.retry_attempts:
                collector.retry_attempts[-1].answer_found = True
                collector.retry_attempts[-1].answer_source = "summarization"
                collector.retry_attempts[-1].finish()
            break

        if is_final and intermediate:
            # Last chance: use the most recent intermediate answer
            final_answer = intermediate[-1]
            logger.info(
                "Using intermediate answer on final attempt: %s",
                final_answer[:200],
            )
            if collector and collector.retry_attempts:
                collector.retry_attempts[-1].answer_found = True
                collector.retry_attempts[-1].answer_source = "intermediate"
                collector.retry_attempts[-1].finish()
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

            failure_input = research_result + "\n\n" + FAILURE_SUMMARY_PROMPT
            failure_text = await _collect_response_text(
                summary_runner, USER_ID, summary_session.id, failure_input
            )
            failure_experiences.append(failure_text)
            logger.info(
                "Failure summary for attempt %d: %s", attempt, failure_text[:300]
            )

            # Emit failure-summary event
            if collector:
                if collector.retry_attempts:
                    collector.retry_attempts[-1].failure_summary = failure_text[:500]
                    collector.retry_attempts[-1].finish()
                await collector.emit(
                    DashboardEvent(
                        event_type=EventType.FAILURE_SUMMARY,
                        attempt=attempt,
                        data={
                            "attempt_number": attempt,
                            "summary_preview": failure_text[:500],
                        },
                    )
                )

    if final_answer is None:
        final_answer = "(No answer could be determined)"

    # ── Dashboard teardown ───────────────────────────────────────────────
    if collector:
        # end_session() emits SESSION_END internally — no separate emit needed.
        # Must call end_session() BEFORE save() so the SESSION_END event and
        # final_answer data are included in the persisted JSON report.
        await collector.end_session(
            final_answer=str(final_answer)[:500],
            attempts_used=min(attempt, CONTEXT_COMPRESS_LIMIT),
        )
        report_path = collector.save()
        clear_collector()
        logger.info("Dashboard metrics saved to %s", report_path)

    print(f"\nFinal answer: {final_answer}\n")
    return final_answer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
