# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Lightweight reverse proxy for Flock ↔ Venice compatibility.

Flock sends ``response_format: {type: json_schema, ...}`` on every request,
but Venice (and many OpenAI-compatible providers) reject this with
"response_format is not supported by this model".

This module runs a tiny aiohttp server on localhost that:
1. Accepts Flock's HTTP requests
2. Strips the ``response_format`` field from the JSON body
3. Forwards the cleaned request to the real LLM API
4. Returns the upstream response verbatim

Usage in corpus_store.py:
    Point Flock's BASE_URL at ``http://127.0.0.1:{FLOCK_PROXY_PORT}``
    instead of the real API base URL.

The proxy is started once via ``start_flock_proxy()`` and runs in a
background thread for the lifetime of the process.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

_FLOCK_PROXY_PORT = int(os.environ.get("FLOCK_PROXY_PORT", "18199"))
_proxy_started = False
_proxy_lock = threading.Lock()


def _clean_request(data: dict) -> dict:
    """Strip fields from the request that Venice/compatible providers reject.

    Also stash the ``response_format`` schema so we can use it to wrap
    plain-text responses back into the expected JSON structure.
    Returns (cleaned_data, original_schema_or_None).
    """
    schema = data.pop("response_format", None)
    return data, schema


def _wrap_content_as_json(content: str, schema: dict | None) -> str:
    """If Flock expected json_schema output, wrap plain-text into it.

    Flock's C++ ``ExtractCompletionOutput`` does
    ``nlohmann::json::parse(content)`` and then indexes into the result
    as an object.  Without ``response_format`` the model returns plain
    text, which Flock can't parse.  We wrap it here.
    """
    if not schema:
        return content

    # Already a JSON *object*?  Return as-is.
    # (Flock needs an object — a bare number/string/array won't work.)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return content
    except (json.JSONDecodeError, TypeError):
        pass

    # Wrap in the expected schema: {"items": ["<content>"]}
    # This matches the default Flock schema for llm_complete.
    return json.dumps({"items": [content]})


def _clean_response(data: dict, schema: dict | None = None) -> dict:
    """Normalise the response to strict OpenAI format for Flock.

    Venice adds extra fields (venice_parameters, reasoning_content, etc.)
    that Flock's C++ JSON parser chokes on.  We strip them here and
    ensure the content matches the expected json_schema structure.
    """
    # Remove top-level Venice extensions
    data.pop("venice_parameters", None)
    data.pop("kv_transfer_params", None)

    # Clean each choice
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        # reasoning_content is Venice-specific; Flock doesn't expect it
        msg.pop("reasoning_content", None)
        # 'name' and 'tool_calls' should only be present if meaningful
        if msg.get("name") is None:
            msg.pop("name", None)
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)

        # Wrap plain-text content in expected JSON schema
        if "content" in msg and isinstance(msg["content"], str):
            msg["content"] = _wrap_content_as_json(msg["content"], schema)

    # Ensure usage fields are integers (some providers return strings)
    usage = data.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in usage and isinstance(usage[key], str):
            try:
                usage[key] = int(usage[key])
            except (ValueError, TypeError):
                pass
    # Remove non-standard usage sub-fields
    usage.pop("prompt_tokens_details", None)
    usage.pop("cache_read_input_tokens", None)

    return data


async def _proxy_handler(request: web.Request) -> web.Response:
    """Forward request to the real API, fixing request and response."""
    real_base = request.app["real_base_url"].rstrip("/")
    target_url = f"{real_base}{request.path}"

    # Read and clean the request body
    body = await request.read()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    original_schema = None
    if body:
        try:
            data = json.loads(body)
            logger.debug("Flock request to %s: %s", request.path, json.dumps(data)[:500])
            data, original_schema = _clean_request(data)
            body = json.dumps(data).encode()
            headers["Content-Length"] = str(len(body))
        except (json.JSONDecodeError, TypeError):
            pass  # forward raw body if not JSON

    # Forward to real API
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=body,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp_body = await resp.read()

            # Clean the response body for Flock compatibility
            if resp.status == 200 and resp_body:
                try:
                    resp_data = json.loads(resp_body)
                    logger.debug("Upstream response: %s", json.dumps(resp_data)[:500])
                    resp_data = _clean_response(resp_data, original_schema)
                    resp_body = json.dumps(resp_data).encode()
                except (json.JSONDecodeError, TypeError):
                    pass  # return raw body if not JSON

            return web.Response(
                status=resp.status,
                body=resp_body,
                content_type=resp.content_type,
            )


def _run_proxy(
    real_base_url: str, port: int, ready_event: threading.Event,
) -> None:
    """Run the proxy server in a background thread."""
    import asyncio

    async def _start() -> None:
        app = web.Application()
        app["real_base_url"] = real_base_url
        app.router.add_route("*", "/{path_info:.*}", _proxy_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", port)
        await site.start()
        logger.info(
            "Flock proxy started on http://127.0.0.1:%d -> %s",
            port, real_base_url,
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
        logger.exception("Flock proxy failed to start")
        # Signal anyway so the main thread doesn't hang — it will check
        # _proxy_started which remains False on failure.
        ready_event.set()


def start_flock_proxy(real_base_url: str) -> str:
    """Start the Flock proxy if not already running.

    Returns the proxy base URL (e.g. ``http://127.0.0.1:18199``).
    Raises RuntimeError if the proxy fails to start within 5 seconds.
    """
    global _proxy_started

    with _proxy_lock:
        if _proxy_started:
            return f"http://127.0.0.1:{_FLOCK_PROXY_PORT}"

        ready_event = threading.Event()
        t = threading.Thread(
            target=_run_proxy,
            args=(real_base_url, _FLOCK_PROXY_PORT, ready_event),
            daemon=True,
            name="flock-proxy",
        )
        t.start()

        # Wait for the proxy to actually bind and be ready
        if not ready_event.wait(timeout=5.0):
            logger.error(
                "Flock proxy failed to start within 5 seconds"
            )
            raise RuntimeError(
                "Flock proxy failed to start — check logs for details"
            )

        # Verify the thread is still alive (it might have crashed after
        # setting ready_event in the except handler)
        if not t.is_alive():
            logger.error("Flock proxy thread died during startup")
            raise RuntimeError(
                "Flock proxy thread died during startup — "
                "check logs for details"
            )

        _proxy_started = True

    proxy_url = f"http://127.0.0.1:{_FLOCK_PROXY_PORT}"
    logger.info("Flock proxy available at %s", proxy_url)
    return proxy_url


def get_flock_proxy_url() -> Optional[str]:
    """Return the proxy URL if running, else None."""
    if _proxy_started:
        return f"http://127.0.0.1:{_FLOCK_PROXY_PORT}"
    return None
