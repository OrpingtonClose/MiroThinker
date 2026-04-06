# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Entry point for the MiroThinker ADK agent.

Supports three execution modes (``--mode``):

* **factoid** (default): Algorithm 6 context-compression retry loop
  that looks for ``\\boxed{}`` answers.
* **report**: Single deep-research pass that returns full prose output.
  Skips ``\\boxed{}`` extraction entirely.
* **batch**: Multi-phase orchestration for exhaustive crawl tasks.
  Phase 1 → discover items, Phase 2 → parallel batch evaluation
  (findings persisted to JSONL), Phase 3 → synthesise report.

Batch mode uses ADK's native ``ResumabilityConfig`` with a
``DatabaseSessionService`` (SQLite) so that stalled workers can be
resumed from the last event rather than restarted from scratch.

Parallel workers (``--workers N``) run N batch sessions concurrently
via ``asyncio.gather`` on separate ADK ``Runner`` instances.

Keep-K (``--keep-k K``) overrides KEEP_TOOL_RESULT for bulk mode,
reducing context burn per batch session.

Health monitoring uses an **event-stream stall detector**: each ADK
event from ``runner.run_async()`` acts as a natural heartbeat.  Workers
that stop producing events for ``--stall-timeout`` seconds (default 45)
are cancelled — but workers actively making tool calls can run
indefinitely.  This is fundamentally different from a hard timeout:
"run as long as you keep making progress; die if you go silent."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv(override=True)

# ── Phoenix/Arize observability (zero custom code) ─────────────────────
# Uses phoenix.otel + GoogleADKInstrumentor for rich agent graph visualization.
# Start the local Phoenix server with:  python -m phoenix.server.main serve
# Then open http://localhost:6006 to view traces + Agent Graph & Path.
from phoenix.otel import register
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

_phoenix_endpoint = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
)
_tracer_provider = register(
    project_name="mirothinker-adk",
    endpoint=_phoenix_endpoint,
)
GoogleADKInstrumentor().instrument(tracer_provider=_tracer_provider)

from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
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
from tools.research_tools import clear_findings, read_findings, set_findings_file
from utils.boxed import extract_boxed_content

logger = logging.getLogger(__name__)

APP_NAME = "mirothinker_adk"
USER_ID = "user"
CONTEXT_COMPRESS_LIMIT = int(os.environ.get("CONTEXT_COMPRESS_LIMIT", "5"))


# ── Shared helpers ──────────────────────────────────────────────────

# Default stall timeout: cancel a worker if no ADK event arrives for
# this many seconds.  Each event (LLM chunk, tool call, tool result,
# delegation) resets the timer, so active workers run indefinitely.
DEFAULT_STALL_TIMEOUT = float(os.environ.get("STALL_TIMEOUT", "45"))


async def _collect_response_text(
    runner: Runner, user_id: str, session_id: str, message: str
) -> str:
    """Run an agent and collect the full response text (no stall detection)."""
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


async def _collect_with_heartbeat(
    runner: Runner,
    user_id: str,
    session_id: str,
    message: str,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    worker_id: int | None = None,
) -> str:
    """Run an agent with event-stream stall detection.

    Each event yielded by ``runner.run_async()`` acts as a heartbeat.
    If no event arrives for *stall_timeout* seconds the worker is
    considered stalled and ``asyncio.TimeoutError`` is raised.

    A worker actively making tool calls can run for hours — it stays
    alive as long as events keep flowing.
    """
    collected = ""
    content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=message)]
    )
    tag = f"worker {worker_id}" if worker_id is not None else "agent"
    event_count = 0

    aiter = runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ).__aiter__()

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    aiter.__anext__(), timeout=stall_timeout
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                logger.warning(
                    "%s stalled — no event for %.0fs after %d events",
                    tag, stall_timeout, event_count,
                )
                raise

            event_count += 1
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        collected += part.text
    finally:
        # Always close the async generator to free MCP connections
        # and Runner resources — critical on stall/cancel paths.
        await aiter.aclose()

    logger.info("%s completed normally (%d events)", tag, event_count)
    return collected


# Directory for persistent session DBs (batch resume).
_FINDINGS_DIR = os.environ.get("FINDINGS_DIR", "/tmp/mirothinker")


def _batch_db_url() -> str:
    """Return the SQLite URL for the persistent batch session store."""
    os.makedirs(_FINDINGS_DIR, exist_ok=True)
    return f"sqlite+aiosqlite:///{_FINDINGS_DIR}/batch_sessions.db"


async def _new_session(
    session_service: InMemorySessionService | DatabaseSessionService,
    keep_k: int | None = None,
    report_mode: bool = False,
) -> object:
    """Create a fresh session, optionally overriding Keep-K and report mode."""
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID
    )
    if keep_k is not None:
        session.state["keep_k"] = keep_k
    if report_mode:
        session.state["report_mode"] = True
    return session


# ═════════════════════════════════════════════════════════════════════
# Mode: factoid (default) — Algorithm 6 context-compression retry
# ═════════════════════════════════════════════════════════════════════


async def run_factoid(task: str) -> str:
    """Original retry loop that expects a ``\\boxed{}`` answer."""
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

        session = await _new_session(session_service)

        runner = Runner(
            agent=research_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        research_result = await _collect_response_text(
            runner, USER_ID, session.id, user_message
        )

        boxed = extract_boxed_content(research_result)
        intermediate = session.state.get("intermediate_boxed_answers", [])

        if boxed and boxed not in ("?", "unknown"):
            final_answer = boxed
            logger.info("Valid answer found on attempt %d: %s", attempt, boxed[:200])
            break

        # No boxed answer — run explicit summarisation step
        summary_prompt = build_main_summary_prompt(task)
        summary_session = await _new_session(session_service)
        summary_runner = Runner(
            agent=summary_agent,
            app_name=APP_NAME,
            session_service=session_service,
        )
        summary_input = research_result + "\n\n" + summary_prompt
        summary_result = await _collect_response_text(
            summary_runner, USER_ID, summary_session.id, summary_input
        )
        boxed = extract_boxed_content(summary_result)
        if boxed and boxed not in ("?", "unknown"):
            final_answer = boxed
            logger.info(
                "Answer found via summarisation on attempt %d: %s",
                attempt, boxed[:200],
            )
            break

        if is_final and intermediate:
            final_answer = intermediate[-1]
            logger.info(
                "Using intermediate answer on final attempt: %s",
                final_answer[:200],
            )
            break

        if not is_final:
            summary_session = await _new_session(session_service)
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

    if final_answer is None:
        final_answer = "(No answer could be determined)"

    return final_answer


# ═════════════════════════════════════════════════════════════════════
# Mode: report — single deep-research pass, no \\boxed{}
# ═════════════════════════════════════════════════════════════════════


async def run_report(task: str) -> str:
    """Single research pass that returns full prose (no \\boxed{} check)."""
    session_service = InMemorySessionService()
    session = await _new_session(session_service, report_mode=True)

    runner = Runner(
        agent=research_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    result = await _collect_response_text(runner, USER_ID, session.id, task)
    return result


# ═════════════════════════════════════════════════════════════════════
# Mode: batch — multi-phase orchestration with parallel workers
# ═════════════════════════════════════════════════════════════════════


async def _run_batch_worker(
    app: App,
    session_service: DatabaseSessionService,
    batch_prompt: str,
    batch_id: int,
    keep_k: int,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
) -> str:
    """Run a single batch evaluation in its own ADK session.

    Uses ADK ``App`` with ``ResumabilityConfig`` so that stalled
    workers can be resumed from the last persisted event.  The
    ``DatabaseSessionService`` (SQLite) persists session state across
    crashes — on restart the same ``invocation_id`` replays completed
    events and continues from where it stopped.

    Health monitoring uses event-stream heartbeat detection: the worker
    can run indefinitely as long as ADK events keep flowing.  If no
    event arrives for *stall_timeout* seconds it is cancelled.
    """
    session = await _new_session(session_service, keep_k=keep_k, report_mode=True)

    runner = Runner(
        app=app,
        session_service=session_service,
    )

    logger.info("Batch worker %d starting (session %s)", batch_id, session.id)
    result = await _collect_with_heartbeat(
        runner, USER_ID, session.id, batch_prompt,
        stall_timeout=stall_timeout,
        worker_id=batch_id,
    )
    logger.info("Batch worker %d finished (%d chars)", batch_id, len(result))
    return result


async def run_batch(
    task: str,
    workers: int = 3,
    batch_size: int = 5,
    keep_k: int = 2,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    resume: bool = True,
) -> str:
    """Multi-phase batch orchestration with heartbeat-based health monitoring.

    Phase 1: Research agent discovers items/URLs to evaluate.
    Phase 2: Items split into batches, run in parallel via
             ``asyncio.gather`` on separate ADK ``Runner`` instances.
             Each batch worker stores findings via ``store_finding``.
             Workers are monitored via event-stream heartbeats —
             they can run indefinitely while making progress, but are
             cancelled after *stall_timeout* seconds of silence.
    Phase 3: Summary agent synthesises all accumulated findings into
             a final report — always runs, even if some workers stalled.

    Checkpoint / resume uses ADK-native ``ResumabilityConfig`` backed
    by ``DatabaseSessionService`` (SQLite).  On restart, stalled
    workers are replayed from the last persisted event — no custom
    JSONL checkpoint scanning required.
    """
    # ── ADK-native resumable App ──────────────────────────────────────
    batch_app = App(
        name=APP_NAME,
        root_agent=research_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    session_service = DatabaseSessionService(db_url=_batch_db_url())
    set_findings_file("batch_findings.jsonl")

    if not resume:
        clear_findings()

    # ── Phase 1: Discovery ──────────────────────────────────────────
    logger.info("=== Phase 1: Discovery ===")
    discovery_prompt = (
        f"{task}\n\n"
        "PHASE 1 INSTRUCTIONS: Your goal is to DISCOVER all relevant items, "
        "URLs, sources, or entities that need evaluation. Use firecrawl_map "
        "to discover links on index pages. Return a JSON array of items to "
        "evaluate, each with 'name' and 'url' fields. Format your response "
        "as ```json\\n[...]\\n```."
    )

    session = await _new_session(session_service, report_mode=True)
    runner = Runner(
        app=batch_app,
        session_service=session_service,
    )

    try:
        discovery_result = await _collect_with_heartbeat(
            runner, USER_ID, session.id, discovery_prompt,
            stall_timeout=stall_timeout,
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning(
            "Phase 1 discovery failed (%s: %s); falling back to report mode",
            type(exc).__name__, exc,
        )
        return await run_report(task)

    # Parse discovered items from the agent's response
    items: list[dict] = []
    try:
        # Try to extract JSON from ```json ... ``` block
        json_match = re.search(r"```json\s*\n(.*?)\n```", discovery_result, re.DOTALL)
        if json_match:
            items = json.loads(json_match.group(1))
        else:
            # Try to parse the whole response as JSON
            items = json.loads(discovery_result)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse discovery items as JSON; treating full output as single-item batch")

    if not items:
        # Fallback: run as a single report-mode research pass
        logger.info("No structured items discovered; falling back to report mode")
        return await run_report(task)

    logger.info("Phase 1 discovered %d items", len(items))

    # ── Phase 2: Parallel batch evaluation ──────────────────────────
    results: list[str | BaseException] = []

    if items:  # skip if checkpoint already covered everything
        logger.info("=== Phase 2: Parallel batch evaluation (%d workers) ===", workers)

        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        sem = asyncio.Semaphore(workers)

        async def _guarded_worker(batch: list[dict], bid: int) -> str:
            async with sem:
                batch_desc = "\n".join(
                    f"- {item.get('name', 'unknown')}: {item.get('url', 'N/A')}"
                    for item in batch
                )
                prompt = (
                    f"Original task: {task}\n\n"
                    f"PHASE 2 INSTRUCTIONS: Evaluate these {len(batch)} items. "
                    "For each item, scrape it and call store_finding with your "
                    "evaluation (name, url, category, summary, rating 1-10).\n\n"
                    f"Items to evaluate:\n{batch_desc}"
                )
                try:
                    return await _run_batch_worker(
                        batch_app, session_service, prompt, bid, keep_k,
                        stall_timeout=stall_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Batch worker %d stalled (no event for %.0fs) — moving on",
                        bid, stall_timeout,
                    )
                    return "(stalled)"
                except Exception as exc:
                    logger.warning(
                        "Batch worker %d failed (%s: %s) — moving on",
                        bid, type(exc).__name__, exc,
                    )
                    return "(error)"

        batch_tasks = [
            _guarded_worker(batch, i) for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*batch_tasks, return_exceptions=True)

    completed = sum(1 for r in results if isinstance(r, str) and r not in ("(stalled)", "(error)"))
    stalled = sum(1 for r in results if r == "(stalled)")
    errored = sum(1 for r in results if r == "(error)" or isinstance(r, BaseException))
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            logger.error("Batch worker %d raised %s: %s", i, type(r).__name__, r)

    logger.info(
        "Phase 2 complete: %d/%d batches OK, %d stalled, %d errored",
        completed, len(results) if results else 0, stalled, errored,
    )

    # ── Phase 3: Synthesis (always runs) ─────────────────────────────
    logger.info("=== Phase 3: Synthesis ===")
    all_findings = await read_findings()

    findings_empty = (
        not all_findings
        or all_findings.strip() in ("", "[]", "No findings recorded yet.")
    )
    if findings_empty:
        logger.warning("No findings accumulated; synthesising from worker prose")
        # Fall back to concatenating whatever prose the workers returned
        worker_prose = "\n\n---\n\n".join(
            r for r in results
            if isinstance(r, str) and r not in ("", "(stalled)", "(error)")
        )
        all_findings = worker_prose if worker_prose else "(no data collected)"

    synthesis_prompt = (
        f"Original task: {task}\n\n"
        f"PHASE 3 INSTRUCTIONS: Below are ALL findings accumulated from "
        f"evaluating {len(items)} discovered sources "
        f"({completed} batches completed, {stalled} stalled). "
        "Synthesise them into a comprehensive, well-structured final report. "
        "Include ratings, categories, and specific URLs.\n\n"
        f"Accumulated findings:\n{all_findings}"
    )

    synthesis_session = await _new_session(session_service)
    synthesis_runner = Runner(
        agent=summary_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    report = await _collect_response_text(
        synthesis_runner, USER_ID, synthesis_session.id, synthesis_prompt
    )

    return report


# ═════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════


async def main(
    task: str | None = None,
    mode: str = "report",
    workers: int = 3,
    batch_size: int = 5,
    keep_k: int | None = None,
    stall_timeout: float | None = None,
    no_resume: bool = False,
) -> str:
    """Run MiroThinker in the specified mode.

    Args:
        task: Research question.  Defaults to a demo question.
        mode: ``"factoid"`` | ``"report"`` | ``"batch"``.
        workers: Number of parallel batch workers (batch mode only).
        batch_size: Items per batch (batch mode only).
        keep_k: Override KEEP_TOOL_RESULT for this run.
        stall_timeout: Per-event stall timeout in seconds (batch mode).
        no_resume: If True, clear previous findings instead of resuming.
    """
    if task is None:
        task = "What is the title of today's arxiv paper in computer science?"

    if keep_k is not None:
        os.environ["KEEP_TOOL_RESULT"] = str(keep_k)

    logger.info("Mode: %s | Task: %s", mode, task[:200])

    if mode == "factoid":
        result = await run_factoid(task)
    elif mode == "report":
        result = await run_report(task)
    elif mode == "batch":
        result = await run_batch(
            task, workers=workers, batch_size=batch_size,
            keep_k=keep_k if keep_k is not None else 2,
            stall_timeout=stall_timeout or DEFAULT_STALL_TIMEOUT,
            resume=not no_resume,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'factoid', 'report', or 'batch'.")

    print(f"\n{'=' * 60}")
    print(f"Mode: {mode}")
    print(f"{'=' * 60}")
    print(result)
    print(f"{'=' * 60}\n")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiroThinker ADK Agent")
    parser.add_argument("task", nargs="?", default=None, help="Research question")
    parser.add_argument(
        "--mode", choices=["factoid", "report", "batch"], default="report",
        help="Execution mode (default: report)",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Number of parallel batch workers (batch mode only)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Items per batch (batch mode only)",
    )
    parser.add_argument(
        "--keep-k", type=int, default=None,
        help="Override KEEP_TOOL_RESULT for this run",
    )
    parser.add_argument(
        "--stall-timeout", type=float, default=None,
        help="Per-event stall timeout in seconds (batch mode, default: 45). "
             "Workers can run indefinitely as long as events keep flowing.",
    )
    parser.add_argument(
        "--no-resume", action="store_true", default=False,
        help="Clear previous findings instead of resuming from checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    asyncio.run(
        main(
            task=args.task,
            mode=args.mode,
            workers=args.workers,
            batch_size=args.batch_size,
            keep_k=args.keep_k,
            stall_timeout=args.stall_timeout,
            no_resume=args.no_resume,
        )
    )
