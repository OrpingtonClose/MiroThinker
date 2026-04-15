# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI server for the Venice uncensored research agent (Strands SDK).

Exposes the Strands agent as an HTTP API with:
- POST /query — single-turn query (single-agent mode)
- POST /query/multi — single-turn query (planner + researcher mode)
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
import queue
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# ── Globals (initialised in lifespan) ────────────────────────────────

_single_agent = None
_multi_agent = None
_mcp_clients: list = []
_multi_researcher = None
_agent_lock = threading.Lock()

# ── Per-request activity logs (ring buffer of last 200) ──────────────

_MAX_LOGS = 200
_request_logs: OrderedDict[str, dict] = OrderedDict()


def _store_log(req_id: str, entry: dict) -> None:
    _request_logs[req_id] = entry
    while len(_request_logs) > _MAX_LOGS:
        _request_logs.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create agents. Shutdown: close MCP connections."""
    global _single_agent, _multi_agent, _multi_researcher, _mcp_clients

    from agent import (
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
    _setup_otel()

    # Enter MCP clients once and share tools between both agents
    try:
        _mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(_mcp_clients)
    except Exception:
        logger.exception("Failed to initialise MCP tools")
        tool_list = []
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

    yield

    # Shutdown: close MCP connections (once)
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
    with _agent_lock:
        from agent import reset_budget

        _single_agent.messages.clear()
        reset_budget()
        try:
            response = _single_agent(req.query)
        except Exception as exc:
            logger.exception("Agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        query=req.query,
        response=str(response),
        mode="single",
        elapsed_seconds=round(time.time() - start, 2),
    )


@app.post("/query/multi", response_model=QueryResponse)
def query_multi(req: QueryRequest):
    """Send a query to the multi-agent (planner delegates to researcher)."""
    if _multi_agent is None:
        raise HTTPException(status_code=503, detail="Multi agent not initialised")

    start = time.time()
    with _agent_lock:
        from agent import reset_budget

        _multi_agent.messages.clear()
        if _multi_researcher is not None:
            _multi_researcher.messages.clear()
        reset_budget()
        try:
            response = _multi_agent(req.query)
        except Exception as exc:
            logger.exception("Agent error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        query=req.query,
        response=str(response),
        mode="multi",
        elapsed_seconds=round(time.time() - start, 2),
    )


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


def _dispatch_agent(model: str, user_message: str) -> str:
    """Run the appropriate agent.  **Caller must already hold _agent_lock**.

    Returns the final text answer.  If the agent result has no text
    content (e.g. it ended on a tool call), falls back to the captured
    streamed text via ``stream_capture.all_text``.
    """
    from agent import reset_budget, stream_capture

    if model == _MODEL_MULTI:
        if _multi_agent is None:
            raise RuntimeError("Multi agent not initialised")
        _multi_agent.messages.clear()
        if _multi_researcher is not None:
            _multi_researcher.messages.clear()
        reset_budget()
        result = str(_multi_agent(user_message))
    elif _single_agent is not None:
        _single_agent.messages.clear()
        reset_budget()
        result = str(_single_agent(user_message))
    else:
        raise RuntimeError("No agent initialised")

    # Fallback: if the agent result has no text (e.g. ended on a tool call),
    # use the captured streamed text which includes the full thinking output.
    if not result.strip() and stream_capture.all_text:
        result = "".join(stream_capture.all_text)
    return result


def _run_agent(model: str, user_message: str) -> str:
    """Dispatch to the correct agent under lock (convenience wrapper)."""
    with _agent_lock:
        return _dispatch_agent(model, user_message)


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
                    result_holder["text"] = _dispatch_agent(model, user_message)
                except Exception as exc:
                    logger.exception("Agent error in streaming [%s]", req_id)
                    result_holder["error"] = str(exc)
                finally:
                    # Snapshot captured data while still under lock
                    result_holder["tool_events"] = list(stream_capture.tool_events)
                    result_holder["streamed_text"] = "".join(stream_capture.all_text)
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
                elif event_type == "tool":
                    # Emit tool call as SSE comment (visible in logs)
                    yield f": tool {data['tool']}\n\n"

            # If agent errored and produced no streamed text, send error
            if result_holder["error"] and not result_holder["streamed_text"]:
                yield _openai_chunk(
                    req_id, model, f"\n\nError: {result_holder['error']}"
                )

            # Append log link at end of response
            log_url = f"/logs/{req_id}"
            yield _openai_chunk(
                req_id, model, f"\n\n---\n[Agent activity log]({log_url})"
            )

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

        return StreamingResponse(_generate_sse(), media_type="text/event-stream")

    # ── Non-streaming mode ───────────────────────────────────────
    # Offload to a thread so the asyncio event loop stays responsive
    # for health checks, SSE heartbeats, and new request acceptance.
    def _sync_non_streaming():
        with _agent_lock:
            stream_capture.activate()
            try:
                answer = _dispatch_agent(model, user_message)
            except Exception as exc:
                logger.exception("Agent error in /v1/chat/completions [%s]", req_id)
                stream_capture.deactivate()
                raise
            # Snapshot captured data while still under lock
            captured_tool_events = list(stream_capture.tool_events)
            captured_all_text = "".join(stream_capture.all_text)
            stream_capture.deactivate()
        return answer, captured_tool_events, captured_all_text

    try:
        answer, captured_tool_events, captured_all_text = await asyncio.to_thread(
            _sync_non_streaming
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "server_error"}},
        )

    log_url = f"/logs/{req_id}"
    answer_with_log = f"{answer}\n\n---\n[Agent activity log]({log_url})"

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
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


# ── Public activity log endpoint ─────────────────────────────────────


@app.get("/logs/{request_id}", response_class=HTMLResponse)
async def get_request_log(request_id: str):
    """Human-readable HTML page showing what the agent did during a request."""
    entry = _request_logs.get(request_id)
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
