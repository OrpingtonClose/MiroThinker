# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
FastAPI server for the Venice GLM-4.7 uncensored research agent.

Exposes the Strands agent as an HTTP API with:
- POST /query — single-turn query (single-agent mode)
- POST /query/multi — single-turn query (planner + researcher mode)
- GET /health — health check
- GET /tools — list loaded tools
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

# ── Globals (initialised in lifespan) ────────────────────────────────

_single_agent = None
_multi_agent = None
_mcp_clients_single: list = []
_mcp_clients_multi: list = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create agents. Shutdown: close MCP connections."""
    global _single_agent, _multi_agent, _mcp_clients_single, _mcp_clients_multi

    from agent import _setup_otel, create_multi_agent, create_single_agent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _setup_otel()

    try:
        _single_agent, _mcp_clients_single = create_single_agent()
        logger.info(
            "Single agent ready — %d tools",
            len(_single_agent.tool_registry.get_all_tools_config()),
        )
    except Exception:
        logger.exception("Failed to create single agent")

    try:
        _multi_agent, _mcp_clients_multi = create_multi_agent()
        logger.info("Multi agent ready")
    except Exception:
        logger.exception("Failed to create multi agent")

    yield

    # Shutdown: close MCP connections
    from agent import _cleanup_mcp

    _cleanup_mcp(_mcp_clients_single)
    _cleanup_mcp(_mcp_clients_multi)
    logger.info("MCP connections closed")


app = FastAPI(
    title="Strands Venice Agent API",
    description="Venice GLM-4.7 uncensored research agent — Strands Agents SDK",
    version="0.1.0",
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
            {"name": t.get("name", "unknown"), "description": t.get("description", "")}
            for t in tools
        ],
    }


@app.post("/query", response_model=QueryResponse)
def query_single(req: QueryRequest):
    """Send a query to the single-agent (all tools directly available).

    Uses plain ``def`` so FastAPI runs it in a threadpool, avoiding
    event-loop blocking from synchronous agent / MCP calls.
    Conversation history is cleared before each call so requests
    are truly single-turn and never leak context between callers.
    """
    if _single_agent is None:
        raise HTTPException(status_code=503, detail="Single agent not initialised")

    # Reset conversation so each HTTP request is independent
    _single_agent.messages.clear()

    start = time.time()
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
    """Send a query to the multi-agent (planner delegates to researcher).

    Uses plain ``def`` so FastAPI runs it in a threadpool, avoiding
    event-loop blocking from synchronous agent / MCP calls.
    Conversation history is cleared before each call so requests
    are truly single-turn and never leak context between callers.
    """
    if _multi_agent is None:
        raise HTTPException(status_code=503, detail="Multi agent not initialised")

    # Reset conversation so each HTTP request is independent
    _multi_agent.messages.clear()

    start = time.time()
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
