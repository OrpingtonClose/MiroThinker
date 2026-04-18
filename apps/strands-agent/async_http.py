# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Async HTTP client for MiroThinker research tools.

Provides a shared httpx.AsyncClient with:
- Configurable timeouts (default 30s connect, 60s read)
- Automatic retry with exponential backoff on 429 / 5xx
- Rate-limit awareness (respects Retry-After headers)
- Proper User-Agent for academic API politeness

All tool modules should use `async_get()` / `async_post()` instead of
raw `httpx.get()` to get timeout, retry, and rate-limit handling for free.

Strands SDK note: @tool-decorated async functions are awaited directly
in the event loop (no thread overhead), making this strictly better than
sync httpx.get() which requires asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

# ── Shared client configuration ──────────────────────────────────────

_CONNECT_TIMEOUT = float(os.environ.get("TOOL_CONNECT_TIMEOUT", "30"))
_READ_TIMEOUT = float(os.environ.get("TOOL_READ_TIMEOUT", "60"))
_MAX_RETRIES = int(os.environ.get("TOOL_MAX_RETRIES", "3"))
_USER_AGENT = "MiroThinker/1.0 (research agent; mailto:research@miromind.ai)"

_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT,
    read=_READ_TIMEOUT,
    write=30.0,
    pool=10.0,
)

_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
}

# Module-level client — created lazily on first use.
# httpx.AsyncClient is safe for concurrent use from multiple coroutines.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Get or create the shared async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT,
            headers=_HEADERS,
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )
    return _client


async def close_client() -> None:
    """Close the shared client. Call during shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


# ── Retry-aware request helpers ──────────────────────────────────────


async def async_get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> httpx.Response:
    """GET with automatic retry on 429/5xx and timeout handling.

    Args:
        url: Request URL.
        params: Query parameters.
        headers: Extra headers (merged with defaults).
        timeout: Override read timeout for this request (seconds).
        max_retries: Override max retry count.

    Returns:
        httpx.Response on success.

    Raises:
        httpx.HTTPStatusError: On non-retryable 4xx errors.
        httpx.TimeoutException: After all retries exhausted.
    """
    client = _get_client()
    retries = max_retries if max_retries is not None else _MAX_RETRIES
    request_timeout = (
        httpx.Timeout(connect=_CONNECT_TIMEOUT, read=timeout, write=30.0, pool=10.0)
        if timeout
        else None
    )
    merged_headers = {**_HEADERS, **(headers or {})}

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.get(
                url,
                params=params,
                headers=merged_headers,
                timeout=request_timeout,
            )

            # Success — return immediately
            if resp.status_code < 400:
                return resp

            # Rate limited — respect Retry-After
            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp)
                if attempt < retries:
                    wait = retry_after or min(2 ** attempt, 30)
                    logger.info(
                        "Rate limited (429) on %s — waiting %.1fs (attempt %d/%d)",
                        url, wait, attempt + 1, retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()

            # Server error — retry with backoff
            if resp.status_code >= 500:
                if attempt < retries:
                    wait = min(2 ** attempt, 30)
                    logger.info(
                        "Server error (%d) on %s — retrying in %.1fs",
                        resp.status_code, url, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()

            # Other 4xx — don't retry, return as-is
            return resp

        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt < retries:
                wait = min(2 ** attempt, 15)
                logger.warning(
                    "Timeout on %s — retrying in %.1fs (attempt %d/%d)",
                    url, wait, attempt + 1, retries,
                )
                await asyncio.sleep(wait)
                continue
            raise

        except httpx.ConnectError as exc:
            last_exc = exc
            if attempt < retries:
                wait = min(2 ** attempt, 15)
                logger.warning(
                    "Connection error on %s — retrying in %.1fs",
                    url, wait,
                )
                await asyncio.sleep(wait)
                continue
            raise

    # Should not reach here, but just in case
    if last_exc:
        raise last_exc
    raise httpx.TimeoutException(f"All {retries} retries exhausted for {url}")


async def async_post(
    url: str,
    *,
    json: dict | None = None,
    data: str | bytes | None = None,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> httpx.Response:
    """POST with automatic retry on 429/5xx and timeout handling.

    Same retry semantics as async_get().
    """
    client = _get_client()
    retries = max_retries if max_retries is not None else _MAX_RETRIES
    request_timeout = (
        httpx.Timeout(connect=_CONNECT_TIMEOUT, read=timeout, write=30.0, pool=10.0)
        if timeout
        else None
    )
    merged_headers = {**_HEADERS, **(headers or {})}

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.post(
                url,
                json=json,
                content=data,
                params=params,
                headers=merged_headers,
                timeout=request_timeout,
            )

            if resp.status_code < 400:
                return resp

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp)
                if attempt < retries:
                    wait = retry_after or min(2 ** attempt, 30)
                    logger.info("Rate limited (429) — waiting %.1fs", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()

            if resp.status_code >= 500:
                if attempt < retries:
                    wait = min(2 ** attempt, 30)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()

            return resp

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 15))
                continue
            raise

    if last_exc:
        raise last_exc
    raise httpx.TimeoutException(f"All retries exhausted for POST {url}")


def _parse_retry_after(resp: httpx.Response) -> float | None:
    """Parse Retry-After header (seconds or HTTP-date)."""
    val = resp.headers.get("Retry-After")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None
