# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI server for Miro — deepagents orchestrator + gossip swarm.

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
- /query/multi uses the deepagents orchestrator (create_deep_agent) which:
  - Plans research strategy via TodoListMiddleware
  - Loads OSINT methodology on-demand via SkillsMiddleware (no regex)
  - Compacts context via SummarizationMiddleware (no truncation)
  - Launches parallel research / harvest / gossip tasks on an
    AsyncTaskPool (task_tools.launch_*) — each task spawns a fresh
    Strands researcher with its own budget tracking
  - Ingests research into ConditionStore (DuckDB) — no string concatenation
  - Builds final report from corpus state via build_report tool
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
_orchestrator = None          # ResearchOrchestrator (deepagents + task pool)
_mcp_clients: list = []
_censored_mcp_clients: list = []
_search_tools: list = []      # Full tool list (uncensored-first)

# Single-agent lock — protects the shared ``_single_agent`` instance
# across concurrent /query and /v1/chat/completions requests. The
# research path no longer uses this lock: each task in the AsyncTaskPool
# spawns its own researcher via create_researcher_instance.
_single_agent_lock = threading.Lock()


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


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create single agent + orchestrator. Shutdown: close MCP."""
    global _single_agent, _orchestrator, _mcp_clients, _censored_mcp_clients, _search_tools

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

    # Enter MCP clients and build combined tool list (uncensored-first).
    # MCP clients are SHARED across all research tasks (thread-safe
    # connection pools); only agent instances are per-task.
    try:
        _mcp_clients, _censored_mcp_clients = get_all_mcp_clients()
        mcp_tools = _enter_mcp_clients(_mcp_clients)
        censored_mcp_tools = _enter_mcp_clients(_censored_mcp_clients)
        _search_tools = _build_tool_list(mcp_tools, censored_mcp_tools)
    except Exception:
        logger.exception("failed to initialise MCP tools")
        _search_tools = _build_tool_list([])
        _mcp_clients = []
        _censored_mcp_clients = []

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

    # Deepagents orchestrator (for /query/multi — planning + coordination).
    # Each job constructs its own AsyncTaskPool inside _run_job; the
    # orchestrator resolves the active pool via a contextvar set there.
    try:
        from orchestrator import create_orchestrator
        from corpus_tools import (
            assess_coverage,
            build_report,
            get_gap_analysis,
            query_corpus,
        )

        _skills_dir = Path(__file__).parent / "skills"
        skills_paths = [str(_skills_dir)] if _skills_dir.is_dir() else None

        _orchestrator = create_orchestrator(
            corpus_tools=[
                query_corpus,
                assess_coverage,
                get_gap_analysis,
                build_report,
            ],
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
    _cleanup_mcp(_censored_mcp_clients)
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

    with _single_agent_lock:
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
    """Create an async research job using the deepagents orchestrator.

    Returns a job ID immediately. Stream real-time progress via
    ``/query/multi/{job_id}/stream``.

    The orchestrator:
    1. Plans research strategy (TodoListMiddleware)
    2. Launches parallel research/harvest tasks (AsyncTaskPool)
    3. Auto-ingests findings into ConditionStore (DuckDB)
    4. Launches gossip synthesis when corpus is sufficient
    5. Reads gap analysis and iterates until coverage is adequate
    6. Builds final report from corpus state
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


async def _run_job(job: "jobs.JobState") -> None:
    """Background task: drive the orchestrator with a per-job AsyncTaskPool.

    The orchestrator consumes ``OrchestratorEvent`` values (protocol
    defined in ``orchestrator_protocol``). Task lifecycle events emitted
    by the ``AsyncTaskPool`` are forwarded directly into
    ``job.event_queue`` via ``event_emit``.
    """
    from corpus import ConditionStore
    from corpus_tools import set_current_store
    from jobs import JobCancelledError
    from task_pool import AsyncTaskPool
    from task_tools import set_current_task_pool

    job.status = "running"
    job.emit({
        "type": "job_started",
        "job_id": job.job_id,
        "query": job.query,
    })

    # Per-job ConditionStore (DuckDB, in-memory).
    store = ConditionStore()
    store.user_query = job.query
    set_current_store(store)

    # Cancel bridge: asyncio Event → threading Event for worker threads.
    cancel_threading = threading.Event()

    async def _cancel_bridge() -> None:
        await job.cancel_event.wait()
        cancel_threading.set()

    cancel_bridge_task = asyncio.create_task(_cancel_bridge())

    # Per-job AsyncTaskPool. Workers spawn fresh researcher agents and
    # auto-ingest results into ``store``. Task lifecycle events are
    # forwarded straight to the job event queue via ``event_emit``.
    pool = AsyncTaskPool(
        store=store,
        tools=_search_tools,
        job_cancel_event=cancel_threading,
        event_emit=job.emit,
        max_concurrent=int(os.environ.get("TASK_POOL_MAX_CONCURRENT", "4")),
    )
    set_current_task_pool(pool)

    try:
        final_content = ""

        if _orchestrator is None:
            raise RuntimeError("Orchestrator not initialised")

        async for event in _orchestrator.run(job.query):
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.finished_at = time.time()
                job.emit({"type": "job_cancelled", "reason": "user_requested"})
                return

            etype = event.type
            name = event.name
            data = event.data

            if etype == "tool_start":
                job.tool_calls += 1
                job.emit({
                    "type": "tool_call",
                    "tool": name,
                    "input_summary": data.get("input_summary", ""),
                    "tool_call_number": job.tool_calls,
                })
                if name == "launch_gossip":
                    job.current_phase = "gossip"
                    job.emit({"type": "gossip_start"})
                elif name == "launch_research":
                    job.current_phase = "research"
                    job.emit({"type": "research_start"})
                elif name == "launch_harvest":
                    job.current_phase = "harvest"
                    job.emit({"type": "harvest_start"})

                if job.tool_calls > 0 and job.tool_calls % 10 == 0:
                    job.emit({
                        "type": "budget_update",
                        "tool_calls": job.tool_calls,
                        "elapsed_s": round(time.time() - job.created_at, 1),
                    })

            elif etype == "tool_end":
                # ``launch_*`` tools return immediately with a task_id JSON
                # blob; actual task completion is surfaced as
                # ``task_completed`` / ``task_failed`` events emitted by
                # the ``AsyncTaskPool`` (forwarded straight to
                # ``job.event_queue``). We therefore do NOT treat
                # tool_end of a launch_* tool as an end-of-phase signal
                # — doing so would emit misleading ``*_end`` /
                # ``intermediate_report`` events with launch metadata
                # rather than real results.
                pass

            elif etype == "stream":
                chunk = data.get("chunk", "")
                if chunk:
                    final_content += chunk

            elif etype == "error":
                job.emit({
                    "type": "orchestrator_error",
                    "error": data.get("error", ""),
                })

        # ── Job complete ──────────────────────────────────────────
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
        cancel_bridge_task.cancel()
        try:
            pool.shutdown()
        except Exception:
            logger.exception("pool shutdown failed")
        set_current_task_pool(None)
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
            with _single_agent_lock:
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
        with _single_agent_lock:
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
