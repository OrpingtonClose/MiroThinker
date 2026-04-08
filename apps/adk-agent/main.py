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

from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.genai import types as genai_types

from agents.research import research_agent
from agents.summary import summary_agent
from agents.pipeline import pipeline_agent
from callbacks.condition_manager import build_corpus_state, cleanup_corpus, get_corpus_text, init_corpus
from tools.mcp_tools import close_all_mcp_toolsets
from prompts.templates import (
    FAILURE_EXPERIENCE_FOOTER,
    FAILURE_EXPERIENCE_HEADER,
    FAILURE_EXPERIENCE_ITEM,
    FAILURE_SUMMARY_PROMPT,
    build_main_summary_prompt,
)
from tools.knowledge_graph import clear_graph, load_graph, query_graph
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
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        collected += part.text
    finally:
        # Safety-release the LLM semaphore if held — prevents permanent
        # slot leak when an LLM call fails between before_model and
        # after_model callbacks (after_model never fires in that case).
        from callbacks.before_model import release_llm_semaphore_if_held
        try:
            sess = await runner._session_service.get_session(
                app_name=runner._app_name, user_id=user_id, session_id=session_id
            )
            if sess:
                release_llm_semaphore_if_held(sess.state)
        except Exception:
            pass  # best-effort cleanup
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
        try:
            await aiter.aclose()
        except Exception:
            logger.warning("Failed to close async iterator for %s", tag, exc_info=True)
        # Safety-release the LLM semaphore if held — prevents permanent
        # slot leak when a worker stalls mid-LLM-call (the after_model
        # callback never fires in that case).
        from callbacks.before_model import release_llm_semaphore_if_held
        try:
            sess = await runner._session_service.get_session(
                app_name=runner._app_name, user_id=user_id, session_id=session_id
            )
            if sess:
                release_llm_semaphore_if_held(sess.state)
        except Exception:
            pass  # best-effort cleanup

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
    initial_state: dict | None = None,
) -> object:
    """Create a fresh session, optionally overriding Keep-K and report mode.

    IMPORTANT: ``InMemorySessionService`` deep-copies the session on both
    ``create_session`` and ``get_session``, so state set *after* creation
    on the returned object is invisible to the Runner.  All initial state
    must be passed via the ``state`` parameter of ``create_session``.
    """
    state: dict = initial_state.copy() if initial_state else {}
    if keep_k is not None:
        state["keep_k"] = keep_k
    if report_mode:
        state["report_mode"] = True
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, state=state,
    )
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
# Mode: pipeline — 4-agent sequential: thinker → researcher → synthesiser
# ═════════════════════════════════════════════════════════════════════


async def run_pipeline(task: str) -> str:
    """Four-agent pipeline: thinker → researcher (+ executor) → synthesiser.

    The thinker and synthesiser use an uncensored model with NO tools.
    The researcher uses a tool-capable model and calls the executor
    (which owns all MCP tools) via AgentTool.

    Data flows via session state:
      thinker  -> state["research_strategy"]
      researcher -> state["research_findings"]
      synthesiser reads {research_findings} and writes the final report.
    """
    session_service = InMemorySessionService()

    # Build the initial state for the corpus *before* session creation.
    # InMemorySessionService deep-copies on create, so state set after
    # creation is invisible to the Runner's get_session() call.
    corpus_state = build_corpus_state()
    session = await _new_session(
        session_service, report_mode=True, initial_state=corpus_state,
    )
    # Register the corpus store singleton (keyed by the session's _corpus_key).
    init_corpus(session.state)

    runner = Runner(
        agent=pipeline_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    try:
        result = await _collect_with_heartbeat(
            runner, USER_ID, session.id, task,
            stall_timeout=1200.0,  # synthesis stage may take 10+ min on a single LLM call
        )
    except asyncio.TimeoutError:
        logger.warning("Pipeline stalled — no event for 1200s; returning partial corpus")
        # Dump the corpus as partial results so research data isn't lost.
        corpus_text = get_corpus_text(session.state)
        if corpus_text:
            result = (
                "## Partial Results (pipeline stalled during synthesis)\n\n"
                + corpus_text
            )
        else:
            result = "(Pipeline stalled before completion — no output produced)"
    finally:
        # Release DuckDB connection so it doesn't leak in long-running servers.
        cleanup_corpus(session.state)
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
    except Exception as exc:
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

    # Delegate to shared evaluation + synthesis pipeline
    return await _run_evaluation_and_synthesis(
        task, items, batch_app, session_service,
        workers=workers, batch_size=batch_size, keep_k=keep_k,
        stall_timeout=stall_timeout,
    )


async def _run_evaluation_and_synthesis(
    task: str,
    items: list[dict],
    batch_app: App,
    session_service: DatabaseSessionService,
    workers: int = 3,
    batch_size: int = 5,
    keep_k: int = 2,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
) -> str:
    """Shared Phase 2 (parallel evaluation) + Phase 3 (synthesis).

    Used by both ``run_batch`` and ``run_exhaustive``.
    """
    # ── Phase 2: Parallel batch evaluation ──────────────────────────
    results: list[str | BaseException] = []

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
                "evaluation (name, url, category, summary, rating 1-10). "
                "Also call add_entity and add_edge to build the knowledge graph.\n\n"
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

    # Include knowledge graph summary if available
    kg_summary = await query_graph()
    kg_section = ""
    try:
        kg_data = json.loads(kg_summary)
        if kg_data.get("entity_count", 0) > 0:
            kg_section = (
                f"\n\nKnowledge Graph ({kg_data['entity_count']} entities, "
                f"{kg_data['edge_count']} relationships):\n{kg_summary}"
            )
    except (json.JSONDecodeError, TypeError):
        pass

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
        "Include ratings, categories, and specific URLs."
        f"{kg_section}\n\n"
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
# Mode: exhaustive — iterative discovery + parallel evaluation
# ═════════════════════════════════════════════════════════════════════


async def run_exhaustive(
    task: str,
    workers: int = 3,
    batch_size: int = 5,
    keep_k: int = 2,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    resume: bool = True,
    crawl_depth: int = 3,
) -> str:
    """Enhanced batch mode with iterative multi-round discovery.

    Unlike standard batch mode (single discovery pass), exhaustive mode
    runs *crawl_depth* rounds of discovery.  Each round:

    1. Searches for new items using different angles / queries
    2. Uses ``firecrawl_map`` to discover links on index pages
    3. Deduplicates against previously discovered items
    4. Builds the knowledge graph with entities + relationships

    After discovery, runs the same Phase 2 (parallel evaluation) and
    Phase 3 (synthesis) as standard batch mode — but with a richer,
    more complete item list.
    """
    batch_app = App(
        name=APP_NAME,
        root_agent=research_agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )
    session_service = DatabaseSessionService(db_url=_batch_db_url())
    set_findings_file("exhaustive_findings.jsonl")

    if not resume:
        clear_findings()
        clear_graph()
    else:
        load_graph()

    # ── Multi-round iterative discovery ────────────────────────────────
    all_items: list[dict] = []
    seen_urls: set[str] = set()

    for round_num in range(1, crawl_depth + 1):
        logger.info("=== Exhaustive Discovery Round %d/%d ===", round_num, crawl_depth)

        if round_num == 1:
            discovery_prompt = (
                f"{task}\n\n"
                f"EXHAUSTIVE DISCOVERY — ROUND {round_num}/{crawl_depth}:\n"
                "Your goal is to discover ALL relevant items, URLs, sources, or "
                "entities. Cast the widest net possible:\n"
                "1. Use brave_web_search and web_search_advanced_exa for broad searches\n"
                "2. Use firecrawl_map on promising index/directory pages to discover all links\n"
                "3. Search from multiple angles: different keywords, categories, regions\n"
                "4. For each entity found, call add_entity to register it in the knowledge graph\n"
                "5. For relationships between entities, call add_edge\n\n"
                "Return a JSON array of items to evaluate, each with 'name' and 'url' fields. "
                "Format: ```json\n[...]\n```"
            )
        else:
            # Subsequent rounds: ask for items NOT already discovered
            known = json.dumps([{"name": i.get("name"), "url": i.get("url")} for i in all_items[:50]])
            discovery_prompt = (
                f"{task}\n\n"
                f"EXHAUSTIVE DISCOVERY — ROUND {round_num}/{crawl_depth}:\n"
                f"We already found {len(all_items)} items in previous rounds. "
                "Search for items NOT in this list:\n"
                f"{known}\n\n"
                "Try different search angles:\n"
                "- Different keywords and phrasings\n"
                "- Different source types (forums, news, academic, vendor sites)\n"
                "- Foreign-language sources\n"
                "- Use firecrawl_map on new index pages\n"
                "- Use find_gaps to identify poorly-connected knowledge graph nodes\n\n"
                "Return ONLY NEW items as ```json\n[...]\n```"
            )

        session = await _new_session(session_service, report_mode=True)
        runner = Runner(app=batch_app, session_service=session_service)

        try:
            result = await _collect_with_heartbeat(
                runner, USER_ID, session.id, discovery_prompt,
                stall_timeout=stall_timeout,
            )
        except Exception as exc:
            logger.warning(
                "Discovery round %d failed (%s: %s); continuing with items so far",
                round_num, type(exc).__name__, exc,
            )
            continue

        # Parse new items
        new_items: list[dict] = []
        try:
            json_match = re.search(r"```json\s*\n(.*?)\n```", result, re.DOTALL)
            if json_match:
                new_items = json.loads(json_match.group(1))
            else:
                new_items = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Round %d: could not parse items as JSON", round_num)

        # Deduplicate
        added = 0
        for item in new_items:
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_items.append(item)
                added += 1

        logger.info(
            "Round %d: found %d new items (%d duplicates skipped), total now %d",
            round_num, added, len(new_items) - added, len(all_items),
        )

        if added == 0 and round_num > 1:
            logger.info("No new items in round %d; stopping discovery early", round_num)
            break

    if not all_items:
        logger.info("No items discovered; falling back to report mode")
        return await run_report(task)

    logger.info("Exhaustive discovery complete: %d total items", len(all_items))

    # ── Phase 2 + 3: reuse batch mode's evaluation + synthesis ─────────
    # We reuse the batch mode infrastructure from here
    return await _run_evaluation_and_synthesis(
        task, all_items, batch_app, session_service,
        workers=workers, batch_size=batch_size, keep_k=keep_k,
        stall_timeout=stall_timeout,
    )


# ═════════════════════════════════════════════════════════════════════
# Mode: decompose — sub-query decomposition + parallel sub-reports
# ═════════════════════════════════════════════════════════════════════


async def run_decompose(
    task: str,
    workers: int = 3,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
) -> str:
    """Break a complex query into sub-questions and run each in parallel.

    Step 1: LLM decomposes the task into independent sub-questions.
    Step 2: Each sub-question runs as a separate report-mode pass.
    Step 3: All sub-reports are synthesised into a final answer.

    This is useful for complex multi-faceted questions like
    "Compare X, Y, Z across dimensions A, B, C" where each
    comparison can be researched independently.
    """
    session_service = InMemorySessionService()

    # ── Step 1: Decompose ─────────────────────────────────────────────
    logger.info("=== Step 1: Query Decomposition ===")
    decompose_prompt = (
        f"{task}\n\n"
        "DECOMPOSITION INSTRUCTIONS: Break this complex question into "
        "independent sub-questions that can be researched separately. "
        "Each sub-question should be self-contained and answerable with "
        "web research.\n\n"
        "Rules:\n"
        "- 3-8 sub-questions (no more than 8)\n"
        "- Each must be specific and searchable\n"
        "- Together they must cover the full scope of the original question\n"
        "- DO NOT use any tools — just think and decompose\n\n"
        "Return as ```json\n[{\"sub_query\": \"...\", \"aspect\": \"...\"}]\n```"
    )

    session = await _new_session(session_service, report_mode=True)
    runner = Runner(
        agent=summary_agent,  # Use summary agent (no tools) for decomposition
        app_name=APP_NAME,
        session_service=session_service,
    )

    decompose_result = await _collect_response_text(
        runner, USER_ID, session.id, decompose_prompt
    )

    # Parse sub-questions
    sub_queries: list[dict] = []
    try:
        json_match = re.search(r"```json\s*\n(.*?)\n```", decompose_result, re.DOTALL)
        if json_match:
            sub_queries = json.loads(json_match.group(1))
        else:
            sub_queries = json.loads(decompose_result)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Could not parse sub-queries; falling back to report mode")
        return await run_report(task)

    if not sub_queries:
        logger.info("No sub-queries generated; falling back to report mode")
        return await run_report(task)

    # Cap at 8
    sub_queries = sub_queries[:8]
    logger.info("Decomposed into %d sub-queries", len(sub_queries))
    for i, sq in enumerate(sub_queries):
        logger.info("  Sub-query %d: %s", i + 1, sq.get("sub_query", str(sq))[:100])

    # ── Step 2: Parallel sub-reports ──────────────────────────────────
    logger.info("=== Step 2: Parallel Sub-Reports (%d workers) ===", workers)
    sem = asyncio.Semaphore(workers)

    async def _sub_report(sq: dict, idx: int) -> str:
        async with sem:
            query_text = sq.get("sub_query", str(sq))
            aspect = sq.get("aspect", f"sub-query-{idx}")
            logger.info("Sub-report %d starting: %s", idx, aspect)

            sub_session_service = InMemorySessionService()
            sub_session = await _new_session(sub_session_service, report_mode=True)
            sub_runner = Runner(
                agent=research_agent,
                app_name=APP_NAME,
                session_service=sub_session_service,
            )

            prompt = (
                f"Original question context: {task}\n\n"
                f"YOUR SPECIFIC SUB-QUESTION: {query_text}\n\n"
                f"Research this specific aspect thoroughly. Focus on: {aspect}\n"
                "Provide a detailed, well-sourced answer with specific facts, "
                "URLs, and data points."
            )

            try:
                result = await _collect_with_heartbeat(
                    sub_runner, USER_ID, sub_session.id, prompt,
                    stall_timeout=stall_timeout,
                    worker_id=idx,
                )
                logger.info("Sub-report %d completed (%d chars)", idx, len(result))
                return f"## {aspect}\n\n{result}"
            except Exception as exc:
                logger.warning(
                    "Sub-report %d failed (%s: %s)",
                    idx, type(exc).__name__, exc,
                )
                return f"## {aspect}\n\n(Research failed: {exc})"

    sub_tasks = [_sub_report(sq, i) for i, sq in enumerate(sub_queries)]
    sub_results = await asyncio.gather(*sub_tasks, return_exceptions=True)

    # Collect sub-reports
    sub_reports: list[str] = []
    for i, r in enumerate(sub_results):
        if isinstance(r, str):
            sub_reports.append(r)
        elif isinstance(r, BaseException):
            aspect = sub_queries[i].get("aspect", f"sub-query-{i}")
            sub_reports.append(f"## {aspect}\n\n(Error: {r})")
            logger.error("Sub-report %d raised %s: %s", i, type(r).__name__, r)

    completed = sum(1 for r in sub_results if isinstance(r, str) and "(Research failed" not in r)
    logger.info("Sub-reports complete: %d/%d successful", completed, len(sub_queries))

    # ── Step 3: Synthesis ─────────────────────────────────────────────
    logger.info("=== Step 3: Synthesis ===")
    all_sub_reports = "\n\n---\n\n".join(sub_reports)

    synthesis_prompt = (
        f"Original question: {task}\n\n"
        f"Below are {len(sub_queries)} independent research sub-reports, "
        f"each covering a different aspect of the question "
        f"({completed}/{len(sub_queries)} completed successfully).\n\n"
        f"{all_sub_reports}\n\n"
        "SYNTHESIS INSTRUCTIONS: Combine these sub-reports into a single, "
        "comprehensive, well-structured final report that fully answers the "
        "original question. Cross-reference findings across sub-reports. "
        "Highlight agreements, contradictions, and knowledge gaps. "
        "Include all specific URLs, data points, and source citations."
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
    crawl_depth: int = 3,
) -> str:
    """Run MiroThinker in the specified mode.

    Args:
        task: Research question.  Defaults to a demo question.
        mode: ``"factoid"`` | ``"report"`` | ``"pipeline"`` | ``"batch"`` | ``"exhaustive"`` | ``"decompose"``.
        workers: Number of parallel batch workers (batch/exhaustive/decompose modes).
        batch_size: Items per batch (batch/exhaustive modes only).
        keep_k: Override KEEP_TOOL_RESULT for this run.
        stall_timeout: Per-event stall timeout in seconds (batch/exhaustive modes).
        no_resume: If True, clear previous findings instead of resuming.
        crawl_depth: Number of discovery rounds (exhaustive mode only, default 3).
    """
    if task is None:
        task = "What is the title of today's arxiv paper in computer science?"

    if keep_k is not None:
        os.environ["KEEP_TOOL_RESULT"] = str(keep_k)

    _effective_stall = stall_timeout if stall_timeout is not None else DEFAULT_STALL_TIMEOUT

    logger.info("Mode: %s | Task: %s", mode, task[:200])

    try:
        if mode == "factoid":
            result = await run_factoid(task)
        elif mode == "report":
            result = await run_report(task)
        elif mode == "pipeline":
            result = await run_pipeline(task)
        elif mode == "batch":
            result = await run_batch(
                task, workers=workers, batch_size=batch_size,
                keep_k=keep_k if keep_k is not None else 2,
                stall_timeout=_effective_stall,
                resume=not no_resume,
            )
        elif mode == "exhaustive":
            result = await run_exhaustive(
                task, workers=workers, batch_size=batch_size,
                keep_k=keep_k if keep_k is not None else 2,
                stall_timeout=_effective_stall,
                resume=not no_resume,
                crawl_depth=crawl_depth,
            )
        elif mode == "decompose":
            result = await run_decompose(
                task, workers=workers,
                stall_timeout=_effective_stall,
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. "
                "Use 'factoid', 'report', 'pipeline', 'batch', 'exhaustive', or 'decompose'."
            )
    finally:
        # Chainlit pattern: gracefully close all MCP subprocess connections
        # BEFORE the event loop shuts down.  Without this, npx processes
        # (Brave, Firecrawl, Exa) crash during teardown with
        # "loop is closed, resources may be leaked" warnings.
        await close_all_mcp_toolsets()

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
        "--mode",
        choices=["factoid", "report", "pipeline", "batch", "exhaustive", "decompose"],
        default="report",
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
    parser.add_argument(
        "--crawl-depth", type=int, default=3,
        help="Number of discovery rounds (exhaustive mode only, default: 3)",
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
            crawl_depth=args.crawl_depth,
        )
    )
