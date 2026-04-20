# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI server for the Venice uncensored research agent (Strands SDK).

Exposes the Strands agent as an HTTP API with:
- POST /query — single-turn query (single-agent mode)
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
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import queue
import re
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
# strands_observability lives in MiroThinker/proxies/ (consolidated from
# deep-search-portal).  PYTHONPATH must include the repo root or proxies/.
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

_single_agent = None
_multi_agent = None
_mcp_clients: list = []
_multi_researcher = None
_agent_lock = threading.Lock()

# ── Skill auto-activation ─────────────────────────────────────────────
# Weaker models don't reliably call the `skills` tool when instructed to.
# Auto-detect query intent and inject the full skill content directly into
# the system prompt so the agent follows the methodology without needing
# the meta-reasoning step of activating it.

_SKILLS_DIR = Path(__file__).parent / "skills"

_SKILL_TRIGGERS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"youtube|video.?channel|youtube.?transcript|harvest.?channel",
            re.IGNORECASE,
        ),
        "osint-censored-discovery",
    ),
]

_skill_cache: dict[str, str] = {}


def _load_skill_content(skill_name: str) -> str | None:
    """Read and cache a skill's SKILL.md content."""
    if skill_name in _skill_cache:
        return _skill_cache[skill_name]
    path = _SKILLS_DIR / skill_name / "SKILL.md"
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8")
    # Strip YAML frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].strip()
    _skill_cache[skill_name] = content
    return content


def _auto_activate_skills(query: str, agent: object) -> None:
    """Detect query intent and inject matching skill content into system prompt.

    Modifies agent.system_prompt in-place before invocation.  The injected
    block is wrapped in XML tags so the AgentSkills plugin's own injection
    doesn't conflict.
    """
    for pattern, skill_name in _SKILL_TRIGGERS:
        if pattern.search(query):
            content = _load_skill_content(skill_name)
            if content:
                marker = f"<!-- auto-skill:{skill_name} -->"
                current = getattr(agent, "system_prompt", "") or ""
                if marker in current:
                    return  # already injected
                injection = (
                    f"\n\n{marker}\n"
                    f"<active_skill name=\"{skill_name}\">\n"
                    f"{content}\n"
                    f"</active_skill>\n"
                )
                agent.system_prompt = current + injection  # type: ignore[attr-defined]
                logger.info(
                    "auto-activated skill=%s for query (pattern matched)",
                    skill_name,
                )
                return  # one skill per query is enough


# ── Observability wrappers ────────────────────────────────────────────
# When strands_observability (from deep-search-portal) is available,
# delegate to it.  Otherwise fall back to minimal local-only logging.

def _write_metrics_jsonl(req_id: str, model: str, query: str, elapsed: float,
                         metrics_summary: dict | None, tool_events: list[dict]) -> None:
    """Write per-request metrics to JSONL.  Delegates to deep-search-portal module."""
    if _HAS_OBSERVABILITY:
        write_metrics_jsonl(req_id, model, query, elapsed, metrics_summary, tool_events)
    else:
        # Minimal fallback: just log a summary line
        logger.info(
            "[metrics] %s model=%s elapsed=%.1fs tools=%d",
            req_id, model, elapsed, len(tool_events),
        )


def _format_inline_log(tool_events: list[dict], elapsed: float, query: str = "",
                       model: str = "", reasoning: str = "",
                       metrics: dict | None = None) -> str:
    """Format activity log for inline display.  Delegates to deep-search-portal module."""
    if _HAS_OBSERVABILITY:
        return format_inline_log(
            tool_events, elapsed,
            query=query, model=model, reasoning=reasoning, metrics=metrics,
        )
    # Minimal fallback: just a tool count summary
    tool_names = [e.get("tool", "?") for e in tool_events]
    summary = ", ".join(tool_names) if tool_names else "(no tools)"
    return f"\n\n---\n*{len(tool_events)} tool calls in {elapsed:.1f}s: {summary}*"


def _store_log(req_id: str, entry: dict) -> None:
    """Store per-request activity log.  Delegates to deep-search-portal module."""
    if _HAS_OBSERVABILITY:
        store_request_log(req_id, entry)


def _get_log(req_id: str) -> dict | None:
    """Retrieve a stored request log."""
    if _HAS_OBSERVABILITY:
        return get_request_log(req_id)
    return None


def _extract_usage(metrics_summary: dict | None) -> dict[str, int]:
    """Extract OpenAI-compatible usage dict from metrics."""
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create agents. Shutdown: close MCP connections."""
    global _single_agent, _multi_agent, _multi_researcher, _mcp_clients

    from agent import (
        _build_tool_list,
        _enter_mcp_clients,
        _setup_otel,
        create_multi_agent,
        create_single_agent,
    )
    from tools import get_all_mcp_clients

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Structured JSON logging for Strands SDK internals ──
    # Delegates to deep-search-portal's strands_observability module when available.
    if _HAS_OBSERVABILITY:
        setup_strands_sdk_logging()
    else:
        logger.info("strands_observability not available — SDK debug logging disabled")

    _setup_otel()

    # Enter MCP clients once and build the combined tool list (uncensored-first)
    try:
        _mcp_clients = get_all_mcp_clients()
        mcp_tools = _enter_mcp_clients(_mcp_clients)
        tool_list = _build_tool_list(mcp_tools)
    except Exception:
        logger.exception("Failed to initialise MCP tools")
        tool_list = _build_tool_list([])  # Still include native tools
        _mcp_clients = []

    try:
        _single_agent, _ = create_single_agent(
            tool_list=tool_list, mcp_clients=_mcp_clients
        )
        logger.info(
            "Single agent ready — %d tools",
            len(_single_agent.tool_registry.get_all_tools_config()),
        )
    except Exception:
        logger.exception("Failed to create single agent")

    try:
        _multi_agent, _multi_researcher, _ = create_multi_agent(
            tool_list=tool_list, mcp_clients=_mcp_clients
        )
        logger.info("Multi agent ready")
    except Exception:
        logger.exception("Failed to create multi agent")

    # Start periodic job cleanup task
    async def _job_cleanup_loop():
        from jobs import job_store as _js
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            _js.cleanup_expired()

    cleanup_task = asyncio.create_task(_job_cleanup_loop())

    yield

    # Shutdown: cancel cleanup task and close MCP connections
    cleanup_task.cancel()

    from agent import _cleanup_mcp

    _cleanup_mcp(_mcp_clients)
    logger.info("MCP connections closed")


app = FastAPI(
    title="Strands Venice Agent API",
    description="Venice uncensored research agent — Strands Agents SDK",
    version="0.2.0",
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
    """Health check."""
    return {
        "status": "ok",
        "single_agent": _single_agent is not None,
        "multi_agent": _multi_agent is not None,
    }


@app.get("/tools")
async def list_tools():
    """List all loaded tools."""
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
    """Send a query to the single-agent (all tools directly available)."""
    if _single_agent is None:
        raise HTTPException(status_code=503, detail="Single agent not initialised")

    start = time.time()
    req_id = f"query-{uuid.uuid4().hex[:12]}"
    with _agent_lock:
        from agent import reset_budget

        _single_agent.messages.clear()
        reset_budget()
        original_prompt = _single_agent.system_prompt
        _auto_activate_skills(req.query, _single_agent)
        try:
            response = _single_agent(req.query)
            metrics = None
            try:
                metrics = response.metrics.get_summary()
            except Exception:
                pass
        except Exception as exc:
            logger.exception("Agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _single_agent.system_prompt = original_prompt

    elapsed = round(time.time() - start, 2)
    _write_metrics_jsonl(req_id, _MODEL_SINGLE, req.query, elapsed, metrics, [])

    return QueryResponse(
        query=req.query,
        response=str(response),
        mode="single",
        elapsed_seconds=elapsed,
    )


def _run_research(query: str) -> tuple[str, dict | None]:
    """Phase 1: run planner+researcher to gather raw data (sync, holds lock)."""
    from agent import reset_budget

    with _agent_lock:
        _multi_agent.messages.clear()
        original_prompts: dict[str, str | None] = {}
        original_prompts["planner"] = _multi_agent.system_prompt
        _auto_activate_skills(query, _multi_agent)
        if _multi_researcher is not None:
            _multi_researcher.messages.clear()
            original_prompts["researcher"] = _multi_researcher.system_prompt
            _auto_activate_skills(query, _multi_researcher)
        reset_budget()
        try:
            response = _multi_agent(query)
            metrics = None
            try:
                metrics = response.metrics.get_summary()
            except Exception:
                pass
        except Exception as exc:
            logger.exception("Agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if "planner" in original_prompts:
                _multi_agent.system_prompt = original_prompts["planner"]
            if _multi_researcher is not None and "researcher" in original_prompts:
                _multi_researcher.system_prompt = original_prompts["researcher"]
    return str(response), metrics


_SWARM_ITERATIONS = int(os.environ.get("SWARM_ITERATIONS", "2"))


def _store_gossip_report(report: str, iteration: int, req_id: str) -> None:
    """Persist a gossip report so the researcher can read it on the next pass."""
    from tools import store_finding

    store_finding(
        name=f"gossip-synthesis-iter-{iteration}",
        url=f"internal://{req_id}/gossip/{iteration}",
        category="gossip_report",
        summary=report[:6000],
        rating=0,
    )


@app.post("/query/multi", response_model=JobCreateResponse)
async def query_multi(req: QueryRequest):
    """Create an async research job with research ↔ gossip feedback loop.

    Returns a job ID immediately.  Stream real-time progress via the
    ``/query/multi/{job_id}/stream`` SSE endpoint, or poll status via
    ``/query/multi/{job_id}/status``.

    Each iteration:
      1. Researcher gathers data (tools: TranscriptAPI, web, etc.)
      2. Gossip swarm produces reports from that data (stored + streamed)
      3. Next iteration: researcher reads gossip reports, fills gaps
      4. Gossip refines on expanded corpus
    """
    from jobs import job_store

    if _multi_agent is None:
        raise HTTPException(status_code=503, detail="Multi agent not initialised")

    job = job_store.create(query=req.query, iterations=_SWARM_ITERATIONS)
    asyncio.create_task(_run_job(job))

    return JobCreateResponse(
        job_id=job.job_id,
        stream_url=f"/query/multi/{job.job_id}/stream",
        status_url=f"/query/multi/{job.job_id}/status",
        result_url=f"/query/multi/{job.job_id}/result",
        cancel_url=f"/query/multi/{job.job_id}/cancel",
    )


async def _run_job(job: "jobs.JobState") -> None:
    """Background task: research ↔ gossip loop with event emission.

    Emits structured events to job.event_queue at every phase boundary.
    Checks job.cancel_event between phases for early termination.
    Stores intermediate gossip reports both as findings and as events.
    """
    import jobs as jobs_mod

    job.status = "running"
    job.emit({
        "type": "job_started",
        "job_id": job.job_id,
        "query": job.query,
        "iterations": job.total_iterations,
    })

    accumulated_corpus = ""
    final_response = ""
    metrics = None

    try:
        for iteration in range(1, job.total_iterations + 1):
            # ── Check cancellation ────────────────────────────────────
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.finished_at = time.time()
                job.emit({"type": "job_cancelled", "reason": "user_requested"})
                return

            job.current_iteration = iteration
            iter_start = time.time()

            job.emit({
                "type": "iteration_start",
                "iteration": iteration,
                "total": job.total_iterations,
            })

            # ── Research phase ────────────────────────────────────────
            if iteration == 1:
                research_query = job.query
            else:
                research_query = (
                    f"{job.query}\n\n"
                    f"--- PRIOR GOSSIP SYNTHESIS (iteration {iteration - 1}) ---\n"
                    f"The previous research round produced the analysis below. "
                    f"Read it carefully. Identify gaps, unverified claims, missing "
                    f"channels, and topics with shallow coverage. Then do targeted "
                    f"research to fill those gaps. Use read_findings(category='gossip_report') "
                    f"to see the full prior reports.\n\n"
                    f"{final_response[:4000]}"
                )

            job.current_phase = "research"
            job.emit({"type": "research_start", "iteration": iteration})

            # Bridge StreamCapture events to job event queue during research
            from agent import stream_capture, set_cancel_flag

            # Set cancel flag so budget_callback can check it
            cancel_threading_event = threading.Event()
            if job.cancel_event.is_set():
                cancel_threading_event.set()

            def _sync_research():
                set_cancel_flag(cancel_threading_event)
                # Activate stream_capture so tool events are recorded
                # during the research phase (StreamCapture.__call__
                # returns early when no queue is active).
                capture_queue = stream_capture.activate()
                try:
                    return _run_research(research_query)
                finally:
                    stream_capture.deactivate()
                    set_cancel_flag(None)

            # Start a drain task that forwards StreamCapture events to job
            drain_active = True

            async def _drain_capture():
                """Forward tool events from StreamCapture to job event queue."""
                loop = asyncio.get_event_loop()
                while drain_active:
                    try:
                        # Poll the stream capture for tool events
                        if (
                            hasattr(stream_capture, 'tool_events')
                            and len(stream_capture.tool_events) > job.tool_calls
                        ):
                            new_events = stream_capture.tool_events[job.tool_calls:]
                            for ev in new_events:
                                job.tool_calls += 1
                                job.emit({
                                    "type": "tool_call",
                                    "tool": ev.get("tool", "unknown"),
                                    "input_summary": ev.get("input", "")[:200],
                                    "tool_call_number": job.tool_calls,
                                })
                                # Periodic budget updates every 10 calls
                                if job.tool_calls % 10 == 0:
                                    job.emit({
                                        "type": "budget_update",
                                        "tool_calls": job.tool_calls,
                                        "elapsed_s": round(time.time() - job.created_at, 1),
                                    })
                    except Exception:
                        pass
                    await asyncio.sleep(2.0)

            drain_task = asyncio.create_task(_drain_capture())

            try:
                # Propagate cancel from asyncio Event to threading Event
                async def _cancel_bridge():
                    await job.cancel_event.wait()
                    cancel_threading_event.set()

                cancel_bridge_task = asyncio.create_task(_cancel_bridge())

                try:
                    raw_research, metrics = await asyncio.to_thread(_sync_research)
                except jobs_mod.JobCancelledError:
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    job.emit({"type": "job_cancelled", "reason": "user_requested"})
                    return
                finally:
                    cancel_bridge_task.cancel()
            finally:
                drain_active = False
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass

            research_elapsed = round(time.time() - iter_start, 2)
            logger.info(
                "job_id=<%s>, iteration=<%d>, research_chars=<%d>, elapsed=<%.1f> | research phase complete",
                job.job_id, iteration, len(raw_research), research_elapsed,
            )

            job.emit({
                "type": "research_end",
                "iteration": iteration,
                "chars": len(raw_research),
                "tool_calls": job.tool_calls,
                "elapsed_s": research_elapsed,
            })

            accumulated_corpus += f"\n\n--- RESEARCH ITERATION {iteration} ---\n{raw_research}"

            # ── Check cancellation before gossip ──────────────────────
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.finished_at = time.time()
                job.emit({"type": "job_cancelled", "reason": "user_requested"})
                return

            # ── Gossip synthesis ──────────────────────────────────────
            job.current_phase = "gossip"

            try:
                from swarm_bridge import gossip_synthesize

                async def _on_swarm_event(event: dict) -> None:
                    """Forward swarm engine events to job event queue."""
                    event["iteration"] = iteration
                    job.emit(event)

                job.emit({
                    "type": "gossip_start",
                    "iteration": iteration,
                    "workers": int(os.environ.get("SWARM_MAX_WORKERS", "6")),
                    "rounds": int(os.environ.get("SWARM_GOSSIP_ROUNDS", "3")),
                })

                swarm_result = await gossip_synthesize(
                    corpus=accumulated_corpus,
                    query=job.query,
                    on_event=_on_swarm_event,
                    cancel_event=job.cancel_event,
                )
                final_response = swarm_result.user_report
                logger.info(
                    "job_id=<%s>, iteration=<%d>, swarm_llm_calls=<%d>, swarm_elapsed=<%.1f>, "
                    "gossip_rounds=<%d>, workers=<%d> | gossip synthesis complete",
                    job.job_id, iteration,
                    swarm_result.metrics.total_llm_calls,
                    swarm_result.metrics.total_elapsed_s,
                    swarm_result.metrics.gossip_rounds_executed,
                    swarm_result.metrics.total_workers,
                )

                # Emit the full intermediate report — this is the key output
                job.emit({
                    "type": "intermediate_report",
                    "iteration": iteration,
                    "report": swarm_result.user_report,
                    "knowledge_report": swarm_result.knowledge_report,
                    "info_gain": list(swarm_result.metrics.gossip_info_gain),
                    "llm_calls": swarm_result.metrics.total_llm_calls,
                    "elapsed_s": round(swarm_result.metrics.total_elapsed_s, 1),
                })

                # Store report so researcher can read it next iteration
                _store_gossip_report(final_response, iteration, job.job_id)

                job.emit({
                    "type": "gossip_end",
                    "iteration": iteration,
                    "llm_calls": swarm_result.metrics.total_llm_calls,
                    "elapsed_s": round(swarm_result.metrics.total_elapsed_s, 1),
                    "info_gain": list(swarm_result.metrics.gossip_info_gain),
                })

            except Exception:
                logger.exception(
                    "job_id=<%s>, iteration=<%d> | gossip synthesis failed, using raw research",
                    job.job_id, iteration,
                )
                final_response = raw_research
                job.emit({
                    "type": "gossip_error",
                    "iteration": iteration,
                    "error": "gossip synthesis failed, using raw research",
                })

            job.emit({
                "type": "iteration_end",
                "iteration": iteration,
                "elapsed_s": round(time.time() - iter_start, 1),
            })

        # ── Job complete ──────────────────────────────────────────────
        elapsed = round(time.time() - job.created_at, 2)
        _write_metrics_jsonl(job.job_id, _MODEL_MULTI, job.query, elapsed, metrics, [])

        job.result = {
            "query": job.query,
            "response": final_response,
            "mode": "multi",
            "elapsed_seconds": elapsed,
        }
        job.status = "complete"
        job.finished_at = time.time()
        job.current_phase = "idle"
        job.emit({
            "type": "job_complete",
            "elapsed_s": elapsed,
            "report_chars": len(final_response),
        })

    except Exception as exc:
        logger.exception("job_id=<%s> | job failed with exception", job.job_id)
        job.status = "failed"
        job.finished_at = time.time()
        job.error = str(exc)
        job.emit({
            "type": "job_failed",
            "error": str(exc),
        })


# ── Job lifecycle endpoints ──────────────────────────────────────────


@app.get("/query/multi/jobs")
async def list_jobs():
    """List all research jobs."""
    from jobs import job_store
    return JSONResponse({"jobs": job_store.list_jobs()})


@app.get("/query/multi/{job_id}/stream")
async def query_multi_stream(job_id: str):
    """SSE stream of job progress events.

    Events are JSON objects with a ``type`` field.  The stream ends
    with a ``job_complete``, ``job_failed``, or ``job_cancelled`` event.
    Heartbeats sent every 5s during idle periods.
    """
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
    """Polling fallback: current job state snapshot."""
    from jobs import job_store

    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.snapshot())


@app.get("/query/multi/{job_id}/result")
async def query_multi_result(job_id: str):
    """Get final result.  Returns 202 Accepted if job is still running."""
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

    # Complete — return QueryResponse-compatible result
    return JSONResponse(job.result or {"error": "No result available"})


@app.post("/query/multi/{job_id}/cancel")
async def query_multi_cancel(job_id: str):
    """Cancel a running job.

    Research phase stops at the next tool call boundary.
    Gossip phase stops between gossip rounds.
    """
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
    """Return available models in OpenAI list format."""
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": _MODEL_SINGLE,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "strands-venice-agent",
                },
                {
                    "id": _MODEL_MULTI,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "strands-venice-agent",
                },
            ],
        }
    )


def _dispatch_agent(model: str, user_message: str) -> tuple[str, dict | None]:
    """Run the appropriate agent.  **Caller must already hold _agent_lock**.

    Returns (text_answer, metrics_summary).  If the agent result has no text
    content (e.g. it ended on a tool call), falls back to the captured
    streamed text via ``stream_capture.response_text``.
    """
    from agent import reset_budget, stream_capture

    metrics_summary = None

    if model == _MODEL_MULTI:
        if _multi_agent is None:
            raise RuntimeError("Multi agent not initialised")
        _multi_agent.messages.clear()
        original_prompts: dict[str, str | None] = {}
        # Inject skill into BOTH planner and researcher (same as query_multi)
        original_prompts["planner"] = _multi_agent.system_prompt
        _auto_activate_skills(user_message, _multi_agent)
        if _multi_researcher is not None:
            _multi_researcher.messages.clear()
            original_prompts["researcher"] = _multi_researcher.system_prompt
            _auto_activate_skills(user_message, _multi_researcher)
        reset_budget()
        try:
            agent_result = _multi_agent(user_message)
            result = str(agent_result)
            try:
                metrics_summary = agent_result.metrics.get_summary()
            except Exception:
                pass
        finally:
            if "planner" in original_prompts:
                _multi_agent.system_prompt = original_prompts["planner"]
            if _multi_researcher is not None and "researcher" in original_prompts:
                _multi_researcher.system_prompt = original_prompts["researcher"]

        # Note: gossip synthesis handled by caller for async paths
    elif _single_agent is not None:
        _single_agent.messages.clear()
        original_prompt = _single_agent.system_prompt
        _auto_activate_skills(user_message, _single_agent)
        reset_budget()
        try:
            agent_result = _single_agent(user_message)
            result = str(agent_result)
            try:
                metrics_summary = agent_result.metrics.get_summary()
            except Exception:
                pass
        finally:
            _single_agent.system_prompt = original_prompt
    else:
        raise RuntimeError("No agent initialised")

    # Fallback: if the agent result has no text (e.g. ended on a tool call),
    # use only the response text (not reasoning/thinking tokens).
    if not result.strip() and stream_capture.response_text:
        result = "".join(stream_capture.response_text)
    return result, metrics_summary


def _run_agent(model: str, user_message: str) -> tuple[str, dict | None]:
    """Dispatch to the correct agent under lock (convenience wrapper)."""
    from agent import stream_capture

    with _agent_lock:
        stream_capture.activate()
        try:
            return _dispatch_agent(model, user_message)
        finally:
            stream_capture.deactivate()


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

    Accepts standard OpenAI request format. Routes to single or multi
    agent based on the ``model`` field. Supports both streaming (SSE)
    and non-streaming responses.

    When streaming, tokens are emitted in real-time as the agent thinks
    and searches. A per-request activity log is stored and accessible
    at ``GET /logs/{request_id}``.
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
        req_id,
        model,
        len(body.messages),
        stream,
        user_message,
    )

    if stream:
        # ── Streaming mode ───────────────────────────────────────
        # activate() and deactivate() happen inside _agent_lock so
        # no concurrent request can clear the capture state.
        # A threading.Event signals the SSE generator once the queue
        # is ready (i.e. the lock has been acquired).
        result_holder: dict = {
            "text": None, "error": None,
            "tool_events": [], "streamed_text": "",
        }
        queue_holder: dict = {"q": None}
        queue_ready = threading.Event()

        def _agent_thread():
            with _agent_lock:
                token_q = stream_capture.activate()
                queue_holder["q"] = token_q
                queue_ready.set()
                try:
                    text, metrics = _dispatch_agent(model, user_message)
                    result_holder["text"] = text
                    result_holder["metrics"] = metrics
                except Exception as exc:
                    logger.exception("Agent error in streaming [%s]", req_id)
                    result_holder["error"] = str(exc)
                finally:
                    # Snapshot captured data while still under lock
                    result_holder["tool_events"] = list(stream_capture.tool_events)
                    result_holder["streamed_text"] = "".join(stream_capture.response_text)
                    result_holder["reasoning_text"] = "".join(stream_capture.reasoning_text)
                    stream_capture.deactivate()

        thread = threading.Thread(target=_agent_thread, daemon=True)
        thread.start()

        async def _generate_sse():
            loop = asyncio.get_event_loop()
            # Wait until agent thread has acquired the lock and created the queue
            await loop.run_in_executor(None, queue_ready.wait)
            token_queue = queue_holder["q"]

            while True:
                try:
                    item = await loop.run_in_executor(
                        None, lambda: token_queue.get(timeout=5)
                    )
                except queue.Empty:
                    # Keep connection alive during long tool executions
                    yield ": heartbeat\n\n"
                    continue

                if item is None:
                    # Agent finished
                    break

                event_type, data = item
                if event_type == "text":
                    yield _openai_chunk(req_id, model, data)
                elif event_type == "thinking":
                    # Skip reasoning tokens — not shown to user
                    pass
                elif event_type == "tool":
                    # Emit tool call as SSE comment (visible in logs)
                    yield f": tool {data['tool']}\n\n"

            # If agent errored and produced no streamed text, send error
            if result_holder["error"] and not result_holder["streamed_text"]:
                yield _openai_chunk(
                    req_id, model, f"\n\nError: {result_holder['error']}"
                )

            # Append inline activity log at end of response
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

            # Store activity log (reads from snapshot, not live capture)
            _store_log(
                req_id,
                {
                    "query": user_message,
                    "model": model,
                    "response": result_holder.get("text", ""),
                    "error": result_holder.get("error"),
                    "tool_events": result_holder["tool_events"],
                    "streamed_text": result_holder["streamed_text"],
                    "elapsed": round(time.time() - start_time, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Write metrics to JSONL log file
            _write_metrics_jsonl(
                req_id, model, user_message, elapsed,
                result_holder.get("metrics"),
                result_holder["tool_events"],
            )

        return StreamingResponse(_generate_sse(), media_type="text/event-stream")

    # ── Non-streaming mode ───────────────────────────────────────
    # Offload to a thread so the asyncio event loop stays responsive
    # for health checks, SSE heartbeats, and new request acceptance.
    def _sync_non_streaming():
        with _agent_lock:
            stream_capture.activate()
            try:
                answer, metrics = _dispatch_agent(model, user_message)
            except Exception as exc:
                logger.exception("Agent error in /v1/chat/completions [%s]", req_id)
                raise
            finally:
                # Snapshot captured data while still under lock
                captured_tool_events = list(stream_capture.tool_events)
                captured_all_text = "".join(stream_capture.response_text)
                captured_reasoning = "".join(stream_capture.reasoning_text)
                stream_capture.deactivate()
        return answer, metrics, captured_tool_events, captured_all_text, captured_reasoning

    try:
        answer, metrics, captured_tool_events, captured_all_text, captured_reasoning = await asyncio.to_thread(
            _sync_non_streaming
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
        reasoning=captured_reasoning,
        metrics=metrics,
    )
    answer_with_log = f"{answer}{inline_log}"

    _store_log(
        req_id,
        {
            "query": user_message,
            "model": model,
            "response": answer,
            "error": None,
            "tool_events": captured_tool_events,
            "streamed_text": captured_all_text,
            "elapsed": round(time.time() - start_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    # Write metrics to JSONL log file
    _write_metrics_jsonl(
        req_id, model, user_message, elapsed,
        metrics, captured_tool_events,
    )

    # Extract token usage from metrics if available
    usage_data = _extract_usage(metrics)

    return JSONResponse(
        {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_with_log},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_data,
        }
    )


# ── Public activity log endpoint ─────────────────────────────────────


@app.get("/logs/{request_id}", response_class=HTMLResponse)
async def get_request_log_page(request_id: str):
    """Human-readable HTML page showing what the agent did during a request."""
    entry = _get_log(request_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Log not found")

    # Build tool events table
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
