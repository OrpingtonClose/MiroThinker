# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
LiteLLM-backed proxy for Flock <-> local-model compatibility.

Flock (the DuckDB community extension) makes OpenAI-compatible HTTP requests
but always sends ``response_format: {type: json_schema, ...}`` which many
providers (Venice, Ollama, vLLM, etc.) reject.

This module runs a tiny aiohttp server on localhost that:

1. Accepts Flock's HTTP requests (``/v1/chat/completions``)
2. Strips the ``response_format`` field
3. Routes the call through **LiteLLM** (``acompletion``), which supports
   dozens of providers out of the box -- Ollama, vLLM, OpenAI, Anthropic,
   Venice, etc.
4. Wraps the response content in JSON when Flock expects ``json_schema``
   output (so Flock's C++ ``ExtractCompletionOutput`` parser succeeds).

Configuration (env vars):
    ``FLOCK_MODEL``     LiteLLM model string, e.g. ``ollama/qwen3-80b``,
                        ``openai/zai-org-glm-5-1``.  Defaults to
                        ``ADK_MODEL`` with ``litellm/`` stripped.
    ``FLOCK_API_KEY``   API key for the model provider (optional for
                        local providers like Ollama).
    ``FLOCK_API_BASE``  Base URL for the model provider (optional --
                        LiteLLM infers it from the model prefix).
    ``FLOCK_PROXY_PORT`` Localhost port (default 18199).

Why LiteLLM instead of raw HTTP forwarding?
    LiteLLM knows the request/response quirks of each provider, handles
    auth, retries, and streaming translation.  We get Ollama + vLLM +
    Venice + OpenAI support for free, instead of maintaining per-provider
    hacks in a custom proxy.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

from aiohttp import web

logger = logging.getLogger(__name__)

_FLOCK_PROXY_PORT = int(os.environ.get("FLOCK_PROXY_PORT", "18199"))
_proxy_started = False
_proxy_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Multi-instance round-robin load balancing
# ---------------------------------------------------------------------------

@dataclass
class _ModelInstance:
    """A single LLM backend that the proxy can route requests to."""
    model: str
    api_key: str
    api_base: str


# Parsed from FLOCK_INSTANCES env var at startup.  Each entry is
# "model@base_url" or just "model" (api_base inferred by LiteLLM).
# Falls back to the single default instance when empty.
_instances: list[_ModelInstance] = []
_instance_cycle: itertools.cycle | None = None  # type: ignore[type-arg]
_instance_lock = threading.Lock()


def _parse_instances(raw: str, default_key: str) -> list[_ModelInstance]:
    """Parse ``FLOCK_INSTANCES`` env var into a list of backends.

    Format: comma-separated entries, each is ``model@base_url`` or ``model``.
    The api_key from the default FLOCK_API_KEY is shared across all instances.
    """
    instances: list[_ModelInstance] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "@" in entry:
            model, base = entry.split("@", 1)
            instances.append(_ModelInstance(model=model.strip(), api_key=default_key, api_base=base.strip()))
        else:
            instances.append(_ModelInstance(model=entry.strip(), api_key=default_key, api_base=""))
    return instances


def _pick_instance(app_default: _ModelInstance) -> _ModelInstance:
    """Return the next instance via round-robin, or the default."""
    global _instance_cycle
    with _instance_lock:
        if not _instances:
            return app_default
        if _instance_cycle is None:
            _instance_cycle = itertools.cycle(range(len(_instances)))
        idx = next(_instance_cycle)
        return _instances[idx]


# ---------------------------------------------------------------------------
# JSON wrapping for Flock compatibility
# ---------------------------------------------------------------------------

def _wrap_content_as_json(content: str, had_schema: bool) -> str:
    """If Flock expected ``json_schema`` output, wrap plain-text into it.

    Flock's C++ ``ExtractCompletionOutput`` does
    ``nlohmann::json::parse(content)`` and then indexes into the result
    as an object.  Without ``response_format`` the model returns plain
    text (or a bare number), which Flock can't parse.  We wrap it here.
    """
    if not had_schema:
        return content

    # Already a JSON *object*?  Return as-is.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return content
    except (json.JSONDecodeError, TypeError):
        pass

    # Wrap in the expected schema: {"items": ["<content>"]}
    return json.dumps({"items": [content]})


# ---------------------------------------------------------------------------
# aiohttp handler -- the only HTTP endpoint Flock talks to
# ---------------------------------------------------------------------------

async def _proxy_handler(request: web.Request) -> web.Response:
    """Route Flock's chat-completion request through LiteLLM."""
    import litellm  # lazy import -- avoid top-level slowdown

    path = request.path.rstrip("/")

    # -- /v1/chat/completions (or /chat/completions) --------------------
    if path.endswith("/chat/completions"):
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"message": "invalid JSON body", "type": "proxy_error"}},
                status=400,
            )

        logger.debug("Flock request: %s", json.dumps(data)[:500])

        # Strip response_format -- Flock always sends it, many providers
        # reject it.  Remember whether it was present so we can wrap the
        # response content later.
        had_schema = "response_format" in data
        data.pop("response_format", None)

        # Build LiteLLM kwargs.  Pick the next backend instance via
        # round-robin if FLOCK_INSTANCES is configured, otherwise use
        # the single default.
        default_inst = _ModelInstance(
            model=request.app["litellm_model"],
            api_key=request.app["litellm_api_key"],
            api_base=request.app["litellm_api_base"],
        )
        inst = _pick_instance(default_inst)

        kwargs: dict = {
            "model": inst.model,
            "messages": data.get("messages", []),
        }
        if inst.api_key:
            kwargs["api_key"] = inst.api_key
        if inst.api_base:
            kwargs["api_base"] = inst.api_base

        # Forward safe optional params
        for param in ("temperature", "max_tokens", "top_p", "stop", "seed"):
            if param in data:
                kwargs[param] = data[param]

        try:
            response = await litellm.acompletion(**kwargs)
            resp_dict = response.model_dump()

            # Wrap content for Flock's json_schema expectation
            for choice in resp_dict.get("choices", []):
                msg = choice.get("message", {})
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = _wrap_content_as_json(
                        msg["content"], had_schema,
                    )

            logger.debug("Proxy response OK (model=%s)", inst.model)
            return web.json_response(resp_dict)

        except Exception as exc:
            logger.error("LiteLLM completion failed: %s", exc, exc_info=True)
            return web.json_response(
                {"error": {"message": str(exc), "type": "litellm_error"}},
                status=502,
            )

    # -- /v1/models (Flock may probe this) ------------------------------
    if path.endswith("/models"):
        model_id = request.app.get("litellm_model", "flock-model")
        return web.json_response({
            "data": [{"id": model_id, "object": "model"}],
            "object": "list",
        })

    return web.json_response(
        {"error": {"message": "not found", "type": "proxy_error"}},
        status=404,
    )


# ---------------------------------------------------------------------------
# Background thread lifecycle
# ---------------------------------------------------------------------------

def _run_proxy(
    litellm_model: str,
    litellm_api_key: str,
    litellm_api_base: str,
    port: int,
    ready_event: threading.Event,
) -> None:
    """Run the proxy server in a background thread."""
    import asyncio

    async def _start() -> None:
        app = web.Application()
        app["litellm_model"] = litellm_model
        app["litellm_api_key"] = litellm_api_key
        app["litellm_api_base"] = litellm_api_base
        app.router.add_route("*", "/{path_info:.*}", _proxy_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        logger.info(
            "Flock LiteLLM proxy on http://127.0.0.1:%d  model=%s  base=%s",
            port, litellm_model, litellm_api_base or "(provider default)",
        )
        # Signal the main thread that we're ready
        ready_event.set()
        # Keep running forever
        await asyncio.Event().wait()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start())
    except Exception:
        logger.exception("Flock LiteLLM proxy failed to start")
        # Signal anyway so the main thread doesn't hang
        ready_event.set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_flock_proxy(
    litellm_model: str,
    litellm_api_key: str = "",
    litellm_api_base: str = "",
) -> str:
    """Start the Flock LiteLLM proxy if not already running.

    Args:
        litellm_model: LiteLLM model string, e.g. ``ollama/qwen3-80b``,
            ``openai/zai-org-glm-5-1``.
        litellm_api_key: API key for the provider (optional for local).
        litellm_api_base: Provider base URL (optional -- LiteLLM infers).

    Returns:
        The proxy base URL (e.g. ``http://127.0.0.1:18199``).

    Raises:
        RuntimeError: If the proxy fails to start within 5 seconds.

    Multi-instance support:
        Set ``FLOCK_INSTANCES`` to a comma-separated list of
        ``model@base_url`` entries for round-robin load balancing
        across multiple LLM backends (e.g. multiple Ollama instances
        on rented GPUs).  When set, the proxy cycles through instances
        on each request.
    """
    global _proxy_started, _instances, _instance_cycle

    # Parse multi-instance config if provided
    raw_instances = os.environ.get("FLOCK_INSTANCES", "")
    if raw_instances:
        parsed = _parse_instances(raw_instances, litellm_api_key)
        if parsed:
            _instances = parsed
            _instance_cycle = None  # reset cycle
            logger.info(
                "Flock multi-instance: %d backends configured",
                len(_instances),
            )
            for i, inst in enumerate(_instances):
                logger.info(
                    "  instance[%d]: model=%s base=%s",
                    i, inst.model, inst.api_base or "(provider default)",
                )

    with _proxy_lock:
        if _proxy_started:
            return f"http://127.0.0.1:{_FLOCK_PROXY_PORT}"

        ready_event = threading.Event()
        t = threading.Thread(
            target=_run_proxy,
            args=(litellm_model, litellm_api_key, litellm_api_base,
                  _FLOCK_PROXY_PORT, ready_event),
            daemon=True,
            name="flock-litellm-proxy",
        )
        t.start()

        # Wait for the proxy to actually bind and be ready
        if not ready_event.wait(timeout=5.0):
            logger.error("Flock LiteLLM proxy failed to start within 5s")
            raise RuntimeError(
                "Flock LiteLLM proxy failed to start -- check logs"
            )

        # Verify the thread is still alive (it might have crashed after
        # setting ready_event in the except handler)
        if not t.is_alive():
            logger.error("Flock LiteLLM proxy thread died during startup")
            raise RuntimeError(
                "Flock LiteLLM proxy thread died -- check logs"
            )

        _proxy_started = True

    proxy_url = f"http://127.0.0.1:{_FLOCK_PROXY_PORT}"
    logger.info("Flock LiteLLM proxy available at %s", proxy_url)
    return proxy_url


def get_flock_proxy_url() -> Optional[str]:
    """Return the proxy URL if running, else None."""
    if _proxy_started:
        return f"http://127.0.0.1:{_FLOCK_PROXY_PORT}"
    return None
