# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI server for Miro — deepagents orchestrator + continuous gossip swarm.

Exposes the research orchestrator as an HTTP API with:
- POST /query — single-turn query (single-agent mode, Strands)
- POST /query/multi — async research job (returns job_id immediately)
- GET  /query/multi/{job_id}/stream — SSE event stream (real-time progress)
- GET  /query/multi/{job_id}/status — polling status snapshot
- GET  /query/multi/{job_id}/result — final report (202 if still running)
- POST /query/multi/{job_id}/cancel — cancel a running job
- GET  /query/multi/jobs — list all jobs
- POST /v1/chat/completions — OpenAI-compatible (LibreChat integration)
- GET  /v1/models — OpenAI-compatible model list
- GET  /health — health check
- GET  /tools — list loaded tools
- GET  /logs/{request_id} — human-readable activity log for a request

Architecture:
- /query/multi runs TWO CONCURRENT TASKS sharing one ConditionStore:
  1. Research task (deepagents orchestrator):
     - Plans strategy via TodoListMiddleware
     - Loads OSINT methodology on-demand via SkillsMiddleware
     - Delegates to Strands researcher via run_research tool
     - Ingests findings into ConditionStore as they arrive
  2. Gossip task (_gossip_loop):
     - Polls ConditionStore continuously for new findings
     - Triggers gossip synthesis when 15+ findings accumulate (or 5min quiet)
     - Stores full synthesis reports back into the corpus
     - Orchestrator sees gossip reports via corpus tools (query_corpus, get_gap_analysis)
  The ConditionStore IS the feedback loop — no explicit handoffs.
- /query uses the Strands single agent (simple queries, all tools direct)
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

# ── Observability ─────────────────────────────────────────────────────
try:
    from strands_observability import (
        extract_usage,
        format_inline_log,
        get_request_log,
        setup_strands_sdk_logging,
        store_request_log,
        write_metrics_jsonl,
    )
    _HAS_OBSERVABILITY = True
except ImportError:
    _HAS_OBSERVABILITY = False

logger = logging.getLogger(__name__)

# ── Globals (initialised in lifespan) ────────────────────────────────

_single_agent = None          # Strands Agent for /query (simple single-turn)
_researcher_agent = None      # Strands Agent researcher (tools: MCP + native)
_orchestrator = None          # deepagents CompiledStateGraph (planning + coordination)
_mcp_clients: list = []
_search_tools: list = []      # Full tool list for researcher (uncensored-first)


# ── Observability wrappers ────────────────────────────────────────────

def _write_metrics_jsonl(req_id: str, model: str, query: str, elapsed: float,
                         metrics_summary: dict | None, tool_events: list[dict]) -> None:
    if _HAS_OBSERVABILITY:
        write_metrics_jsonl(req_id, model, query, elapsed, metrics_summary, tool_events)
    else:
        logger.info(
            "[metrics] %s model=%s elapsed=%.1fs tools=%d",
            req_id, model, elapsed, len(tool_events),
        )


def _format_inline_log(tool_events: list[dict], elapsed: float, query: str = "",
                       model: str = "", reasoning: str = "",
                       metrics: dict | None = None) -> str:
    if _HAS_OBSERVABILITY:
        return format_inline_log(
            tool_events, elapsed,
            query=query, model=model, reasoning=reasoning, metrics=metrics,
        )
    tool_names = [e.get("tool", "?") for e in tool_events]
    summary = ", ".join(tool_names) if tool_names else "(no tools)"
    return f"\n\n---\n*{len(tool_events)} tool calls in {elapsed:.1f}s: {summary}*"


def _store_log(req_id: str, entry: dict) -> None:
    if _HAS_OBSERVABILITY:
        store_request_log(req_id, entry)


def _get_log(req_id: str) -> dict | None:
    if _HAS_OBSERVABILITY:
        return get_request_log(req_id)
    return None


def _extract_usage(metrics_summary: dict | None) -> dict[str, int]:
    if _HAS_OBSERVABILITY:
        return extract_usage(metrics_summary)
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if metrics_summary and metrics_summary.get("accumulated_usage"):
        u = metrics_summary["accumulated_usage"]
        usage = {
            "prompt_tokens": u.get("inputTokens", 0),
            "completion_tokens": u.get("outputTokens", 0),
            "total_tokens": u.get("totalTokens", 0),
        }
    return usage


# ── Research delegation tool ──────────────────────────────────────────
# The orchestrator calls this to delegate research to the Strands agent.
# The Strands researcher keeps all its MCP + native tools.

_research_lock = threading.Lock()

# Per-job cancel event (threading.Event) bridged from asyncio cancel_event.
# Set via _set_job_cancel_event() in _run_job, read by run_research.
import contextvars
_job_cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "_job_cancel_event", default=None,
)


def _invoke_researcher(
    task: str,
    cancel_event: threading.Event | None = None,
) -> str:
    """Run the Strands researcher agent on a task (sync, holds lock).

    Args:
        task: Research task description.
        cancel_event: Threading event bridged from the job's asyncio
            cancel_event. When set, budget_callback raises
            JobCancelledError on the next tool call.

    Returns:
        Raw research text.

    Raises:
        jobs.JobCancelledError: If the cancel_event is set during research.
    """
    if _researcher_agent is None:
        return "(researcher agent not initialised)"

    from agent import reset_budget, set_cancel_flag, stream_capture
    from jobs import JobCancelledError

    with _research_lock:
        _researcher_agent.messages.clear()
        reset_budget()
        set_cancel_flag(cancel_event)
        stream_capture.activate()
        try:
            response = _researcher_agent(task)
            return str(response)
        except JobCancelledError:
            raise  # Propagate cancellation to caller
        except Exception as exc:
            logger.exception("researcher agent error")
            return f"(research failed: {exc})"
        finally:
            stream_capture.deactivate()
            set_cancel_flag(None)


def run_research(task: str) -> str:
    """Delegate a deep research task to the researcher agent.

    The researcher has all search, scrape, and transcript tools
    (TranscriptAPI, DuckDuckGo, Brave, Exa, Jina, Firecrawl,
    Semantic Scholar, arXiv, Perplexity, Grok, Reddit, etc.)
    ordered uncensored-first.

    Be SPECIFIC about what to search, what sources to prioritise,
    and what data to extract.

    After research completes, the raw findings are automatically
    ingested into the ConditionStore. Use query_corpus to see
    what was gathered.

    Args:
        task: Detailed description of what to research.

    Returns:
        Summary of research completed and findings ingested.
    """
    from corpus_tools import _get_store

    raw_text = _invoke_researcher(task, cancel_event=_job_cancel_event.get())

    # Ingest into ConditionStore (no truncation, no string concat)
    store = _get_store()
    ids = store.ingest_raw(
        raw_text,
        source_type="researcher",
        source_ref=task[:200],
    )

    return (
        f"Research complete. {len(ids)} findings ingested into corpus. "
        f"Use query_corpus to inspect what was gathered."
    )


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create agents + orchestrator. Shutdown: close connections."""
    global _single_agent, _researcher_agent, _orchestrator, _mcp_clients, _search_tools

    from agent import (
        _build_tool_list,
        _enter_mcp_clients,
        _setup_otel,
        create_single_agent,
    )
    from tools import get_all_mcp_clients

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if _HAS_OBSERVABILITY:
        setup_strands_sdk_logging()
    else:
        logger.info("strands_observability not available — SDK debug logging disabled")

    _setup_otel()

    # Enter MCP clients and build combined tool list (uncensored-first)
    try:
        _mcp_clients = get_all_mcp_clients()
        mcp_tools = _enter_mcp_clients(_mcp_clients)
        _search_tools = _build_tool_list(mcp_tools)
    except Exception:
        logger.exception("failed to initialise MCP tools")
        _search_tools = _build_tool_list([])
        _mcp_clients = []

    # Single agent (for /query endpoint — simple single-turn)
    try:
        _single_agent, _ = create_single_agent(
            tool_list=_search_tools, mcp_clients=_mcp_clients,
        )
        logger.info(
            "single agent ready — %d tools",
            len(_single_agent.tool_registry.get_all_tools_config()),
        )
    except Exception:
        logger.exception("failed to create single agent")

    # Researcher agent (for orchestrator delegation via run_research)
    try:
        from strands import Agent
        from strands.agent.conversation_manager import SlidingWindowConversationManager
        from agent import _build_callback_handler
        from prompts import SYSTEM_PROMPT

        _researcher_agent = Agent(
            model=__import__("config", fromlist=["build_model"]).build_model(),
            system_prompt=(
                "You are a research specialist. Execute the research task "
                "thoroughly and exhaustively. Use every available tool. "
                "Search in multiple languages if relevant.\n\n"
                "Tool priority (uncensored-first):\n"
                "1. TranscriptAPI: search_youtube, get_youtube_transcript, "
                "search_channel_videos\n"
                "2. Uncensored web: duckduckgo_search, brave_search\n"
                "3. Academic: semantic_scholar_search, arxiv_search\n"
                "4. Deep research: perplexity_search, grok_search\n"
                "5. Community: reddit_search\n"
                "6. General web: google_search (last resort)\n\n"
                "Return a comprehensive raw research report with ALL "
                "data gathered. Include specific numbers, protocols, "
                "dosages, bloodwork values, and source URLs."
            ),
            tools=_search_tools,
            conversation_manager=SlidingWindowConversationManager(
                window_size=15,
                should_truncate_results=True,
            ),
            callback_handler=_build_callback_handler(),
        )
        logger.info("researcher agent ready — %d tools", len(_search_tools))
    except Exception:
        logger.exception("failed to create researcher agent")

    # Deepagents orchestrator (for /query/multi — planning + coordination)
    try:
        from orchestrator import create_orchestrator
        from corpus_tools import (
            query_corpus,
            assess_coverage,
            get_gap_analysis,
            trigger_gossip,
            build_report,
        )

        _skills_dir = Path(__file__).parent / "skills"
        skills_paths = [str(_skills_dir)] if _skills_dir.is_dir() else None

        _orchestrator = create_orchestrator(
            research_fn=run_research,
            corpus_tools=[query_corpus, assess_coverage, get_gap_analysis],
            gossip_tools=[trigger_gossip, build_report],
            skills_paths=skills_paths,
        )
        logger.info("deepagents orchestrator ready")
    except Exception:
        logger.exception("failed to create orchestrator")

    # Start periodic job cleanup task
    async def _job_cleanup_loop():
        from jobs import job_store as _js
        while True:
            await asyncio.sleep(300)
            _js.cleanup_expired()

    cleanup_task = asyncio.create_task(_job_cleanup_loop())

    yield

    # Shutdown
    cleanup_task.cancel()
    from agent import _cleanup_mcp
    _cleanup_mcp(_mcp_clients)
    logger.info("MCP connections closed")


app = FastAPI(
    title="Miro Research Orchestrator API",
    description="Deepagents orchestrator + gossip swarm — uncensored research",
    version="0.3.0",
    lifespan=lifespan,
)


# ── Request / Response models ────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., description="The research query to send to the agent")


class QueryResponse(BaseModel):
    query: str
    response: str
    mode: str
    elapsed_seconds: float


class JobCreateResponse(BaseModel):
    job_id: str
    stream_url: str
    status_url: str
    result_url: str
    cancel_url: str


class ChatMessage(BaseModel):
    role: str
    content: str | list = ""


class ChatCompletionRequest(BaseModel):
    model: str = "strands-venice-single"
    messages: list[ChatMessage] = []
    stream: bool = False

    model_config = {"extra": "allow"}


# ── Helper: extract user message from ChatML messages ────────────────


def _extract_user_message(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            return str(content)
    return ""


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "single_agent": _single_agent is not None,
        "orchestrator": _orchestrator is not None,
        "researcher": _researcher_agent is not None,
    }


@app.get("/tools")
async def list_tools():
    if _single_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    tools = _single_agent.tool_registry.get_all_tools_config()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": name,
                "description": spec.get("description", "")
                if isinstance(spec, dict)
                else "",
            }
            for name, spec in tools.items()
        ],
    }


@app.post("/query", response_model=QueryResponse)
def query_single(req: QueryRequest):
    """Single-turn query via Strands single agent (all tools direct)."""
    if _single_agent is None:
        raise HTTPException(status_code=503, detail="Single agent not initialised")

    start = time.time()
    from agent import reset_budget

    with _research_lock:
        _single_agent.messages.clear()
        reset_budget()
        try:
            response = _single_agent(req.query)
        except Exception as exc:
            logger.exception("agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = round(time.time() - start, 2)

    return QueryResponse(
        query=req.query,
        response=str(response),
        mode="single",
        elapsed_seconds=elapsed,
    )


# ── Async research jobs (deepagents orchestrator) ────────────────────


@app.post("/query/multi", response_model=JobCreateResponse)
async def query_multi(req: QueryRequest):
    """Create an async research job with continuous gossip synthesis.

    Returns a job ID immediately. Stream real-time progress via
    ``/query/multi/{job_id}/stream``.

    Launches two concurrent tasks sharing one ConditionStore:
    1. Research: orchestrator plans and delegates to researcher
    2. Gossip: polls for new findings, synthesizes continuously

    Gossip reports feed back into the corpus — the orchestrator sees
    them via query_corpus/get_gap_analysis and adjusts research.
    """
    from jobs import job_store

    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")

    job = job_store.create(query=req.query, iterations=0)  # orchestrator decides iterations
    asyncio.create_task(_run_job(job))

    return JobCreateResponse(
        job_id=job.job_id,
        stream_url=f"/query/multi/{job.job_id}/stream",
        status_url=f"/query/multi/{job.job_id}/status",
        result_url=f"/query/multi/{job.job_id}/result",
        cancel_url=f"/query/multi/{job.job_id}/cancel",
    )


async def _gossip_loop(
    job: "jobs.JobState",
    store: "ConditionStore",
    cancel_event: "asyncio.Event",
) -> None:
    """Continuous gossip loop running alongside research.

    Polls the ConditionStore for new findings and triggers gossip
    synthesis rounds as material accumulates. Gossip reports are
    stored back into the corpus, where the orchestrator's tools
    (query_corpus, get_gap_analysis) surface them to the researcher
    on the next iteration — closing the feedback loop.

    Trigger logic (hybrid):
    - 15+ new findings since last round, OR
    - 5 minutes elapsed since last round (catch stale gaps), OR
    - research_complete flag set (one final comprehensive round)

    Args:
        job: The running job state (for SSE event emission).
        store: Shared ConditionStore (thread-safe via RLock).
        cancel_event: Checked every poll cycle; stops the loop.
    """
    from swarm_bridge import gossip_synthesize

    POLL_INTERVAL_S = 30  # Check every 30 seconds
    MIN_NEW_FINDINGS = 15  # Trigger threshold
    MAX_QUIET_S = 300  # 5 minutes without gossip → force a round
    gossip_round = 0
    last_gossip_max_id = 0
    last_gossip_time = time.time()

    logger.info("job_id=<%s> | gossip loop started", job.job_id)

    while not cancel_event.is_set():
        # Sleep in short increments so we can react to cancellation
        for _ in range(POLL_INTERVAL_S):
            if cancel_event.is_set() or store.research_complete:
                break
            await asyncio.sleep(1.0)

        if cancel_event.is_set():
            break

        # Check if enough new material has accumulated
        new_count = store.count_since(last_gossip_max_id)
        elapsed_since_gossip = time.time() - last_gossip_time
        is_final = store.research_complete

        should_trigger = (
            new_count >= MIN_NEW_FINDINGS
            or (elapsed_since_gossip >= MAX_QUIET_S and new_count > 0)
            or is_final
        )

        if not should_trigger:
            if is_final:
                break  # research done, no new material
            continue

        # Export corpus and run gossip
        corpus_text = store.export_for_swarm(min_confidence=0.0)
        if "(corpus is empty" in corpus_text:
            if is_final:
                break
            continue

        gossip_round += 1
        current_max_id = store.max_id()
        logger.info(
            "job_id=<%s>, gossip_round=<%d>, new_findings=<%d>, is_final=<%s> "
            "| triggering gossip synthesis",
            job.job_id, gossip_round, new_count, is_final,
        )

        job.current_phase = "gossip"
        job.emit({
            "type": "gossip_start",
            "gossip_round": gossip_round,
            "new_findings_since_last": new_count,
            "corpus_size": store.count(),
            "is_final_round": is_final,
        })

        async def _on_gossip_event(e: dict) -> None:
            job.event_queue.put_nowait(e)

        try:
            result = await gossip_synthesize(
                corpus=corpus_text,
                query=store.user_query,
                on_event=_on_gossip_event,
                cancel_event=cancel_event,
            )

            # Store full synthesis (no truncation)
            metrics_dict: dict = {}
            if hasattr(result, "metrics"):
                m = result.metrics
                metrics_dict = {
                    "info_gain": list(getattr(m, "gossip_info_gain", [])),
                    "llm_calls": getattr(m, "total_llm_calls", 0),
                    "elapsed_seconds": getattr(m, "total_elapsed_s", 0),
                }

            store.admit_synthesis(
                report=result.user_report,
                iteration=gossip_round,
                metrics=metrics_dict,
            )

            last_gossip_max_id = current_max_id
            last_gossip_time = time.time()

            job.emit({
                "type": "intermediate_report",
                "gossip_round": gossip_round,
                "report": result.user_report,
                "report_chars": len(result.user_report),
                "info_gain": metrics_dict.get("info_gain", []),
                "is_final_round": is_final,
            })
            job.emit({
                "type": "gossip_end",
                "gossip_round": gossip_round,
            })

            logger.info(
                "job_id=<%s>, gossip_round=<%d>, report_chars=<%d> "
                "| gossip synthesis complete",
                job.job_id, gossip_round, len(result.user_report),
            )

        except Exception:
            logger.exception(
                "job_id=<%s>, gossip_round=<%d> | gossip synthesis failed",
                job.job_id, gossip_round,
            )
            job.emit({
                "type": "gossip_error",
                "gossip_round": gossip_round,
                "error": "gossip synthesis failed (see server logs)",
            })
        finally:
            job.current_phase = "research"

        if is_final:
            break

    logger.info(
        "job_id=<%s>, total_gossip_rounds=<%d> | gossip loop ended",
        job.job_id, gossip_round,
    )


async def _run_job(job: "jobs.JobState") -> None:
    """Background task: research and gossip run concurrently.

    Two concurrent async tasks share one ConditionStore:
    - Research task: orchestrator streams events, researcher writes to store
    - Gossip task: polls store, synthesizes when material accumulates,
      writes reports back — which the orchestrator sees via corpus tools

    The store IS the feedback loop — no explicit handoffs.
    Events from both tasks stream to the same SSE endpoint.
    """
    from corpus import ConditionStore
    from corpus_tools import set_current_store
    from jobs import JobCancelledError
    from langchain_core.messages import HumanMessage

    job.status = "running"
    job.emit({
        "type": "job_started",
        "job_id": job.job_id,
        "query": job.query,
    })

    # Per-job ConditionStore (DuckDB, in-memory)
    store = ConditionStore()
    store.user_query = job.query
    set_current_store(store)

    # Cancel bridge: asyncio Event → threading Event for Strands budget_callback
    cancel_threading = threading.Event()
    _job_cancel_event.set(cancel_threading)

    async def _cancel_bridge() -> None:
        await job.cancel_event.wait()
        cancel_threading.set()

    cancel_bridge_task = asyncio.create_task(_cancel_bridge())

    # Launch continuous gossip loop alongside research
    gossip_task = asyncio.create_task(
        _gossip_loop(job, store, job.cancel_event)
    )

    try:
        # Stream events from the orchestrator (research side)
        event_count = 0
        final_content = ""
        job.current_phase = "research"

        async for event in _orchestrator.astream_events(
            {"messages": [HumanMessage(content=job.query)]},
            version="v2",
        ):
            event_count += 1

            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.finished_at = time.time()
                job.emit({"type": "job_cancelled", "reason": "user_requested"})
                return

            event_type = event.get("event", "")
            event_name = event.get("name", "")

            if event_type == "on_tool_start":
                job.tool_calls += 1
                tool_input = event.get("data", {}).get("input", "")
                if isinstance(tool_input, dict):
                    tool_input = json.dumps(tool_input)[:200]
                elif isinstance(tool_input, str):
                    tool_input = tool_input[:200]
                else:
                    tool_input = str(tool_input)[:200]

                job.emit({
                    "type": "tool_call",
                    "tool": event_name,
                    "input_summary": tool_input,
                    "tool_call_number": job.tool_calls,
                })

                if event_name == "run_research":
                    job.current_phase = "research"
                    job.emit({"type": "research_start"})

            elif event_type == "on_tool_end":
                if event_name == "run_research":
                    job.emit({
                        "type": "research_end",
                        "tool_calls": job.tool_calls,
                    })

            elif event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    final_content += chunk.content

            if event_type == "on_tool_start" and job.tool_calls > 0 and job.tool_calls % 10 == 0:
                job.emit({
                    "type": "budget_update",
                    "tool_calls": job.tool_calls,
                    "elapsed_s": round(time.time() - job.created_at, 1),
                })

        # ── Research complete — signal gossip for final round ─────
        store.research_complete = True
        logger.info("job_id=<%s> | research complete, waiting for final gossip round", job.job_id)

        # Wait for gossip loop to finish its final round (with timeout)
        try:
            await asyncio.wait_for(gossip_task, timeout=600)
        except asyncio.TimeoutError:
            logger.warning("job_id=<%s> | gossip final round timed out after 600s", job.job_id)
            gossip_task.cancel()

        # ── Build final report from ConditionStore ────────────────
        report = store.build_report(user_query=job.query)
        if not report.strip() or "(No gossip synthesis" in report:
            report = final_content or "(no report generated)"

        elapsed = round(time.time() - job.created_at, 2)
        _write_metrics_jsonl(job.job_id, _MODEL_MULTI, job.query, elapsed, None, [])

        job.result = {
            "query": job.query,
            "response": report,
            "mode": "multi",
            "elapsed_seconds": elapsed,
            "corpus_stats": store.count_by_type(),
        }
        job.status = "complete"
        job.finished_at = time.time()
        job.current_phase = "idle"
        job.emit({
            "type": "job_complete",
            "elapsed_s": elapsed,
            "report_chars": len(report),
            "corpus_stats": store.count_by_type(),
        })

    except JobCancelledError:
        logger.info("job_id=<%s> | job cancelled by user", job.job_id)
        job.status = "cancelled"
        job.finished_at = time.time()
        job.emit({"type": "job_cancelled", "reason": "user_requested"})
    except Exception as exc:
        logger.exception("job_id=<%s> | job failed", job.job_id)
        job.status = "failed"
        job.finished_at = time.time()
        job.error = str(exc)
        job.emit({
            "type": "job_failed",
            "error": str(exc),
        })
    finally:
        # Ensure gossip loop is cleaned up
        if not gossip_task.done():
            gossip_task.cancel()
            try:
                await gossip_task
            except asyncio.CancelledError:
                pass
        cancel_bridge_task.cancel()
        _job_cancel_event.set(None)
        store.close()


# ── Job lifecycle endpoints ──────────────────────────────────────────


@app.get("/query/multi/jobs")
async def list_jobs():
    from jobs import job_store
    return JSONResponse({"jobs": job_store.list_jobs()})


@app.get("/query/multi/{job_id}/stream")
async def query_multi_stream(job_id: str):
    """SSE stream of job progress events."""
    from jobs import job_store

    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    terminal_types = {"job_complete", "job_failed", "job_cancelled"}

    async def _generate():
        while True:
            try:
                event = await asyncio.wait_for(job.event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") in terminal_types:
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/query/multi/{job_id}/status")
async def query_multi_status(job_id: str):
    from jobs import job_store
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.snapshot())


@app.get("/query/multi/{job_id}/result")
async def query_multi_result(job_id: str):
    """Get final result. Returns 202 if job is still running."""
    from jobs import job_store

    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ("pending", "running"):
        return JSONResponse(
            status_code=202,
            content={
                "status": job.status,
                "message": "Job still running",
                **job.snapshot(),
            },
        )

    if job.status == "cancelled":
        return JSONResponse(
            status_code=200,
            content={"status": "cancelled", "message": "Job was cancelled"},
        )

    if job.status == "failed":
        return JSONResponse(
            status_code=500,
            content={"status": "failed", "error": job.error or "Unknown error"},
        )

    return JSONResponse(job.result or {"error": "No result available"})


@app.post("/query/multi/{job_id}/cancel")
async def query_multi_cancel(job_id: str):
    """Cancel a running job."""
    from jobs import job_store

    if not job_store.cancel(job_id):
        job = job_store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be cancelled (status: {job.status})",
        )
    return JSONResponse({"status": "cancellation_requested", "job_id": job_id})


# ── OpenAI-compatible endpoints (for LibreChat integration) ──────────

_MODEL_SINGLE = "strands-venice-single"
_MODEL_MULTI = "strands-venice-multi"


@app.get("/v1/models")
async def openai_models():
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": _MODEL_SINGLE,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "miro-orchestrator",
                },
                {
                    "id": _MODEL_MULTI,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "miro-orchestrator",
                },
            ],
        }
    )


def _openai_chunk(req_id: str, model: str, content: str, finish: bool = False) -> str:
    """Format a single SSE chunk in OpenAI streaming format."""
    chunk = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Routes to single or multi agent based on the ``model`` field.
    For multi model, creates a job and awaits completion.
    """
    from agent import stream_capture

    model = body.model
    stream = body.stream
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    start_time = time.time()

    user_message = _extract_user_message(body.messages)
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No user message found",
                    "type": "invalid_request_error",
                }
            },
        )

    logger.info(
        "[%s] model=%s messages=%d stream=%s query=%.100s",
        req_id, model, len(body.messages), stream, user_message,
    )

    if model == _MODEL_MULTI:
        # Multi model: create a job and await completion
        return await _openai_multi(req_id, model, user_message, stream, start_time)

    # Single model: direct Strands agent invocation
    if stream:
        result_holder: dict = {
            "text": None, "error": None,
            "tool_events": [], "streamed_text": "",
        }
        queue_holder: dict = {"q": None}
        queue_ready = threading.Event()

        def _agent_thread():
            from agent import reset_budget
            with _research_lock:
                token_q = stream_capture.activate()
                queue_holder["q"] = token_q
                queue_ready.set()
                try:
                    if _single_agent is None:
                        raise RuntimeError("Single agent not initialised")
                    _single_agent.messages.clear()
                    reset_budget()
                    agent_result = _single_agent(user_message)
                    result_holder["text"] = str(agent_result)
                    try:
                        result_holder["metrics"] = agent_result.metrics.get_summary()
                    except Exception:
                        pass
                except Exception as exc:
                    logger.exception("agent error in streaming [%s]", req_id)
                    result_holder["error"] = str(exc)
                finally:
                    result_holder["tool_events"] = list(stream_capture.tool_events)
                    result_holder["streamed_text"] = "".join(stream_capture.response_text)
                    result_holder["reasoning_text"] = "".join(stream_capture.reasoning_text)
                    stream_capture.deactivate()

        thread = threading.Thread(target=_agent_thread, daemon=True)
        thread.start()

        async def _generate_sse():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, queue_ready.wait)
            token_queue = queue_holder["q"]

            while True:
                try:
                    item = await loop.run_in_executor(
                        None, lambda: token_queue.get(timeout=5)
                    )
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    continue

                if item is None:
                    break

                event_type, data = item
                if event_type == "text":
                    yield _openai_chunk(req_id, model, data)
                elif event_type == "tool":
                    yield f": tool {data['tool']}\n\n"

            if result_holder["error"] and not result_holder["streamed_text"]:
                yield _openai_chunk(
                    req_id, model, f"\n\nError: {result_holder['error']}"
                )

            elapsed = round(time.time() - start_time, 2)
            inline_log = _format_inline_log(
                result_holder["tool_events"], elapsed,
                query=user_message, model=model,
                reasoning=result_holder.get("reasoning_text", ""),
                metrics=result_holder.get("metrics"),
            )
            yield _openai_chunk(req_id, model, inline_log)
            yield _openai_chunk(req_id, model, "", finish=True)
            yield "data: [DONE]\n\n"

            _store_log(req_id, {
                "query": user_message, "model": model,
                "response": result_holder.get("text", ""),
                "error": result_holder.get("error"),
                "tool_events": result_holder["tool_events"],
                "streamed_text": result_holder["streamed_text"],
                "elapsed": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            _write_metrics_jsonl(
                req_id, model, user_message, elapsed,
                result_holder.get("metrics"), result_holder["tool_events"],
            )

        return StreamingResponse(_generate_sse(), media_type="text/event-stream")

    # Non-streaming single
    def _sync_single():
        from agent import reset_budget
        with _research_lock:
            stream_capture.activate()
            try:
                if _single_agent is None:
                    raise RuntimeError("Single agent not initialised")
                _single_agent.messages.clear()
                reset_budget()
                agent_result = _single_agent(user_message)
                result = str(agent_result)
                metrics = None
                try:
                    metrics = agent_result.metrics.get_summary()
                except Exception:
                    pass
            finally:
                captured_tool_events = list(stream_capture.tool_events)
                captured_all_text = "".join(stream_capture.response_text)
                captured_reasoning = "".join(stream_capture.reasoning_text)
                stream_capture.deactivate()
        return result, metrics, captured_tool_events, captured_all_text, captured_reasoning

    try:
        answer, metrics, captured_tool_events, captured_all_text, captured_reasoning = (
            await asyncio.to_thread(_sync_single)
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "server_error"}},
        )

    elapsed = round(time.time() - start_time, 2)
    inline_log = _format_inline_log(
        captured_tool_events, elapsed,
        query=user_message, model=model,
        reasoning=captured_reasoning, metrics=metrics,
    )
    answer_with_log = f"{answer}{inline_log}"

    _store_log(req_id, {
        "query": user_message, "model": model,
        "response": answer, "error": None,
        "tool_events": captured_tool_events,
        "streamed_text": captured_all_text,
        "elapsed": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _write_metrics_jsonl(req_id, model, user_message, elapsed, metrics, captured_tool_events)

    return JSONResponse({
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer_with_log},
            "finish_reason": "stop",
        }],
        "usage": _extract_usage(metrics),
    })


async def _openai_multi(
    req_id: str, model: str, user_message: str,
    stream: bool, start_time: float,
) -> JSONResponse | StreamingResponse:
    """Handle multi-model requests via deepagents orchestrator.

    Creates a job, invokes the orchestrator, and returns either
    a streaming SSE response or a blocking JSON response.
    """
    from jobs import job_store

    if _orchestrator is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Orchestrator not initialised", "type": "server_error"}},
        )

    job = job_store.create(query=user_message, iterations=0)
    asyncio.create_task(_run_job(job))

    if stream:
        # Stream orchestrator events as OpenAI SSE chunks
        async def _generate():
            terminal = {"job_complete", "job_failed", "job_cancelled"}
            while True:
                try:
                    event = await asyncio.wait_for(job.event_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue

                etype = event.get("type", "")

                if etype == "tool_call":
                    yield f": tool {event.get('tool', 'unknown')}\n\n"
                elif etype == "intermediate_report":
                    report = event.get("report", "")
                    if report:
                        yield _openai_chunk(req_id, model, f"\n\n--- Intermediate Report ---\n{report}\n")
                elif etype == "job_complete":
                    if job.result:
                        yield _openai_chunk(req_id, model, job.result.get("response", ""))
                    yield _openai_chunk(req_id, model, "", finish=True)
                    yield "data: [DONE]\n\n"
                    break
                elif etype == "job_failed":
                    yield _openai_chunk(req_id, model, f"\n\nError: {event.get('error', 'unknown')}")
                    yield _openai_chunk(req_id, model, "", finish=True)
                    yield "data: [DONE]\n\n"
                    break
                elif etype in terminal:
                    yield _openai_chunk(req_id, model, "", finish=True)
                    yield "data: [DONE]\n\n"
                    break

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # Non-streaming: wait for job completion
    terminal = {"complete", "failed", "cancelled"}
    while job.status not in terminal:
        await asyncio.sleep(1.0)

    elapsed = round(time.time() - start_time, 2)
    response_text = ""
    if job.result:
        response_text = job.result.get("response", "")

    return JSONResponse({
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# ── Public activity log endpoint ─────────────────────────────────────


@app.get("/logs/{request_id}", response_class=HTMLResponse)
async def get_request_log_page(request_id: str):
    """Human-readable HTML page showing what the agent did during a request."""
    entry = _get_log(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Log not found")

    tool_rows = ""
    for i, ev in enumerate(entry.get("tool_events", []), 1):
        t = datetime.fromtimestamp(ev["time"], tz=timezone.utc).strftime("%H:%M:%S")
        tool_rows += (
            f"<tr><td>{i}</td><td><code>{html.escape(ev['tool'])}</code></td>"
            f"<td><pre>{html.escape(ev.get('input', ''))}</pre></td>"
            f"<td>{t}</td></tr>\n"
        )

    if not tool_rows:
        tool_rows = '<tr><td colspan="4">No tool calls recorded</td></tr>'

    escaped_query = html.escape(entry.get("query", ""))
    escaped_response = html.escape(entry.get("response", "") or "")
    escaped_streamed = html.escape(entry.get("streamed_text", "") or "")
    error_block = ""
    if entry.get("error"):
        error_block = (
            f'<div style="background:#fee;padding:12px;border-radius:6px;">'
            f"<strong>Error:</strong> {html.escape(entry['error'])}</div>"
        )

    page = f"""\
<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Agent Log — {html.escape(request_id)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; font-size: 1.4em; }}
  h2 {{ color: #8b949e; font-size: 1.1em; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
  th {{ background: #161b22; color: #8b949e; }}
  tr:nth-child(even) {{ background: #161b22; }}
  pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 0.85em; }}
  code {{ color: #79c0ff; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
  .response {{ background: #161b22; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; line-height: 1.6; }}
  .thinking {{ background: #1c2128; padding: 16px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; line-height: 1.5; color: #8b949e; font-size: 0.9em; max-height: 600px; overflow-y: auto; }}
</style>
</head><body>
<h1>Agent Activity Log</h1>
<p class="meta">
  <strong>Request:</strong> <code>{html.escape(request_id)}</code><br>
  <strong>Model:</strong> <code>{html.escape(entry.get('model', ''))}</code><br>
  <strong>Time:</strong> {html.escape(entry.get('timestamp', ''))}<br>
  <strong>Elapsed:</strong> {entry.get('elapsed', 0)}s
</p>
{error_block}
<h2>Query</h2>
<div class="response">{escaped_query}</div>

<h2>Tool Calls ({len(entry.get('tool_events', []))})</h2>
<table>
<tr><th>#</th><th>Tool</th><th>Input</th><th>Time</th></tr>
{tool_rows}
</table>

<h2>Agent Thinking (streamed tokens)</h2>
<details>
<summary>Click to expand ({len(entry.get('streamed_text', ''))} chars)</summary>
<div class="thinking">{escaped_streamed}</div>
</details>

<h2>Final Response</h2>
<div class="response">{escaped_response}</div>
</body></html>"""

    return HTMLResponse(page)
