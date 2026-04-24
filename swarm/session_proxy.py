"""Session proxy — maps model names to cloned worker conversations.

This is the core of the clone pattern (Stage 2). It sits between DuckDB/Flock
and the actual LLM backend (vLLM, OpenRouter, etc.). When a request arrives
with model_name='clone_{angle}', the proxy prepends the stored conversation
history before forwarding to the backend. For non-clone model names, requests
pass through unmodified.

Architecture ref: docs/SWARM_WAVE_ARCHITECTURE.md § Session Proxy

Port: 18199 (configurable via SESSION_PROXY_PORT env var)

API:
    POST /v1/chat/completions   — proxied chat completion (clone context injected)
    POST /sessions/{clone_id}   — register/update a cloned conversation
    GET  /sessions               — list all registered clones
    DELETE /sessions/{clone_id} — remove a clone
    GET  /health                 — health check
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Session Proxy", version="0.1.0")


# ═══════════════════════════════════════════════════════════════════════
# Clone store — in-memory map of clone_id → conversation messages
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CloneSession:
    """A stored worker conversation that can be prepended to new queries."""

    clone_id: str
    messages: list[dict[str, str]]
    angle: str = ""
    wave: int = 0
    registered_at: float = field(default_factory=time.time)
    query_count: int = 0


# Global clone registry (single-process, in-memory for PoC)
_clones: dict[str, CloneSession] = {}


def register_clone(
    clone_id: str,
    messages: list[dict[str, str]],
    angle: str = "",
    wave: int = 0,
) -> CloneSession:
    """Register or update a cloned conversation.

    Args:
        clone_id: Unique identifier (typically 'clone_{angle}').
        messages: Full conversation history to prepend.
        angle: The research angle this clone specializes in.
        wave: The wave number when this clone was captured.

    Returns:
        The registered CloneSession.
    """
    session = CloneSession(
        clone_id=clone_id,
        messages=messages,
        angle=angle,
        wave=wave,
    )
    _clones[clone_id] = session
    logger.info(
        "clone_id=<%s>, angle=<%s>, wave=<%d>, messages=<%d> | clone registered",
        clone_id, angle, wave, len(messages),
    )
    return session


def get_clone(clone_id: str) -> CloneSession | None:
    """Look up a registered clone by ID."""
    return _clones.get(clone_id)


def list_clones() -> list[dict[str, Any]]:
    """List all registered clones with metadata."""
    return [
        {
            "clone_id": s.clone_id,
            "angle": s.angle,
            "wave": s.wave,
            "message_count": len(s.messages),
            "query_count": s.query_count,
            "registered_at": s.registered_at,
        }
        for s in _clones.values()
    ]


def remove_clone(clone_id: str) -> bool:
    """Remove a clone from the registry."""
    return _clones.pop(clone_id, None) is not None


# ═══════════════════════════════════════════════════════════════════════
# Backend configuration
# ═══════════════════════════════════════════════════════════════════════

_BACKEND_URL = os.environ.get("SESSION_PROXY_BACKEND", "http://localhost:8000/v1")
_BACKEND_KEY = os.environ.get("SESSION_PROXY_BACKEND_KEY", "not-needed")


def get_backend_url() -> str:
    """Return the current backend URL."""
    return _BACKEND_URL


def set_backend(url: str, api_key: str = "not-needed") -> None:
    """Configure the backend LLM endpoint."""
    global _BACKEND_URL, _BACKEND_KEY
    _BACKEND_URL = url
    _BACKEND_KEY = api_key
    logger.info("backend_url=<%s> | backend configured", url)


# ═══════════════════════════════════════════════════════════════════════
# Core proxy logic — prepend clone context if model is a clone
# ═══════════════════════════════════════════════════════════════════════


async def proxy_chat_completion(request_body: dict[str, Any]) -> dict[str, Any]:
    """Proxy a chat completion request, injecting clone context if applicable.

    If the model name matches a registered clone (e.g. 'clone_insulin_timing'),
    the clone's conversation history is prepended to the messages array before
    forwarding to the backend. The actual model sent to the backend is the
    default model (not the clone name).

    Args:
        request_body: OpenAI-compatible chat completion request.

    Returns:
        The backend's response, unmodified.
    """
    model = request_body.get("model", "")
    messages = list(request_body.get("messages", []))

    # Check if this is a clone-routed request
    clone = get_clone(model)
    if clone is not None:
        # Prepend the clone's conversation history
        clone_messages = list(clone.messages)
        combined = clone_messages + messages
        clone.query_count += 1

        logger.debug(
            "clone_id=<%s>, prepended=<%d>, new=<%d>, total=<%d> | clone context injected",
            model, len(clone_messages), len(messages), len(combined),
        )

        # Replace model with the actual backend model
        # The clone name is just a routing key, not a real model
        backend_model = request_body.get("_backend_model", "default")
        forwarded = {**request_body, "model": backend_model, "messages": combined}
        forwarded.pop("_backend_model", None)
    else:
        # Strip internal fields (prefixed with '_') so strict OpenAI-compatible
        # backends don't reject unknown keys.
        forwarded = {k: v for k, v in request_body.items() if not k.startswith("_")}

    # Forward to backend
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{_BACKEND_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {_BACKEND_KEY}",
                "Content-Type": "application/json",
            },
            json=forwarded,
        )
        resp.raise_for_status()
        return resp.json()


# ═══════════════════════════════════════════════════════════════════════
# FastAPI endpoints
# ═══════════════════════════════════════════════════════════════════════


@app.post("/v1/chat/completions")
async def chat_completions(request_body: dict[str, Any]) -> JSONResponse:
    """Proxy chat completion with clone context injection."""
    try:
        result = await proxy_chat_completion(request_body)
        return JSONResponse(content=result)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Backend error: {exc.response.text[:500]}",
        ) from exc
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot reach backend at {_BACKEND_URL}: {exc}",
        ) from exc


@app.post("/sessions/{clone_id}")
async def register_session(clone_id: str, body: dict[str, Any]) -> JSONResponse:
    """Register or update a cloned conversation.

    Body:
        messages: list of {role, content} dicts
        angle: (optional) research angle
        wave: (optional) wave number
    """
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages array required")

    session = register_clone(
        clone_id=clone_id,
        messages=messages,
        angle=body.get("angle", ""),
        wave=body.get("wave", 0),
    )
    return JSONResponse(content={
        "clone_id": session.clone_id,
        "message_count": len(session.messages),
        "status": "registered",
    })


@app.get("/sessions")
async def list_sessions() -> JSONResponse:
    """List all registered clone sessions."""
    return JSONResponse(content={"clones": list_clones()})


@app.delete("/sessions/{clone_id}")
async def delete_session(clone_id: str) -> JSONResponse:
    """Remove a clone session."""
    if remove_clone(clone_id):
        return JSONResponse(content={"status": "removed", "clone_id": clone_id})
    raise HTTPException(status_code=404, detail=f"Clone {clone_id} not found")


@app.get("/health")
async def health() -> JSONResponse:
    """Health check."""
    return JSONResponse(content={
        "status": "ok",
        "backend_url": _BACKEND_URL,
        "clone_count": len(_clones),
    })


# ═══════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SESSION_PROXY_PORT", "18199"))
    logger.info("port=<%d>, backend=<%s> | starting session proxy", port, _BACKEND_URL)
    uvicorn.run(app, host="0.0.0.0", port=port)
