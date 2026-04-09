# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-tool callback — source-level context budget + per-provider rate limiting.

What this callback does:
1. **Source-level context budget for Exa** — injects ``textMaxCharacters`` and
   ``numResults`` into every Exa tool call so results are bounded at the source.
2. **Per-provider rate limiting** — separate threading.Semaphore per MCP provider
   (Brave, Exa, Firecrawl, Kagi) so one slow provider doesn't block others.
3. **Dashboard tool_start tracking** — records each tool call in the collector.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from dashboard import get_active_collector

logger = logging.getLogger(__name__)

# ── Per-provider rate limiting ───────────────────────────────────────────
# Separate semaphores prevent one slow provider from blocking others.
# Default concurrency of 3 per provider is generous — most providers
# handle 3 concurrent requests without rate-limiting.
# We use threading.Semaphore (not asyncio.Semaphore) because the
# before_tool_callback is synchronous and needs non-blocking acquire.
_BRAVE_CONCURRENCY = int(os.environ.get("BRAVE_CONCURRENCY", "3"))
_EXA_CONCURRENCY = int(os.environ.get("EXA_CONCURRENCY", "3"))
_FIRECRAWL_CONCURRENCY = int(os.environ.get("FIRECRAWL_CONCURRENCY", "3"))
_KAGI_CONCURRENCY = int(os.environ.get("KAGI_CONCURRENCY", "3"))

_provider_semaphores: Dict[str, threading.Semaphore] = {
    "brave": threading.Semaphore(_BRAVE_CONCURRENCY),
    "exa": threading.Semaphore(_EXA_CONCURRENCY),
    "firecrawl": threading.Semaphore(_FIRECRAWL_CONCURRENCY),
    "kagi": threading.Semaphore(_KAGI_CONCURRENCY),
}

# Map tool names to provider keys
_TOOL_TO_PROVIDER: Dict[str, str] = {
    "brave_web_search": "brave",
    "brave_local_search": "brave",
    "brave_image_search": "brave",
    "brave_video_search": "brave",
    "brave_news_search": "brave",
    "brave_summarizer": "brave",
    "web_search_exa": "exa",
    "web_search_advanced_exa": "exa",
    "crawling_exa": "exa",
    "get_code_context_exa": "exa",
    "firecrawl_scrape": "firecrawl",
    "firecrawl_search": "firecrawl",
    "firecrawl_crawl": "firecrawl",
    "firecrawl_map": "firecrawl",
    "firecrawl_extract": "firecrawl",
    "kagi_search": "kagi",
    "kagi_summarize": "kagi",
    "kagi_fastgpt": "kagi",
    "kagi_enrich_web": "kagi",
    "kagi_enrich_news": "kagi",
}


def _get_provider(tool_name: str) -> Optional[str]:
    """Return provider key for a tool, or None if not rate-limited."""
    return _TOOL_TO_PROVIDER.get(tool_name)

# Source-level context budget defaults for Exa tools.
# These cap how much TEXT the Exa API returns per result — not truncation,
# just telling the API to send less, like numResults caps how many results.
EXA_DEFAULT_TEXT_MAX_CHARS = 10_000  # emergency cap per-result (chars)
EXA_DEFAULT_NUM_RESULTS = 8  # max results per search
EXA_CRAWL_TEXT_MAX_CHARS = 15_000  # crawling a specific URL gets more


# ── helpers ──────────────────────────────────────────────────────────────────


def _enforce_exa_budget(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Inject source-level size controls into Exa tool calls.

    This is NOT truncation — it tells the Exa API to return less data per
    result, the same way numResults tells it to return fewer results.
    The LLM's instruction says to use these params, but it doesn't always
    comply, so we enforce them deterministically here.
    """
    if tool_name in ("web_search_exa", "web_search_advanced_exa"):
        if "numResults" not in args:
            args["numResults"] = EXA_DEFAULT_NUM_RESULTS
        # Coerce numResults to int and cap at 10 — LLM sometimes sends
        # string values like "8" instead of 8, which downstream rejects.
        try:
            nr = int(args.get("numResults", 0))
            args["numResults"] = min(nr, 10)
        except (TypeError, ValueError):
            args["numResults"] = EXA_DEFAULT_NUM_RESULTS
        # Inject textMaxCharacters if LLM didn't set it
        if "textMaxCharacters" not in args:
            args["textMaxCharacters"] = EXA_DEFAULT_TEXT_MAX_CHARS
        # Enable highlights for focused excerpts
        if "enableHighlights" not in args:
            args["enableHighlights"] = True

    elif tool_name == "crawling_exa":
        # Crawling a specific URL: allow more text but still bounded
        if "textMaxCharacters" not in args:
            args["textMaxCharacters"] = EXA_CRAWL_TEXT_MAX_CHARS
        if "enableHighlights" not in args:
            args["enableHighlights"] = True

    elif tool_name == "get_code_context_exa":
        if "numResults" not in args:
            args["numResults"] = EXA_DEFAULT_NUM_RESULTS

    return args


# ── callback ─────────────────────────────────────────────────────────────────


def before_tool_callback(
    tool: Any, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict[str, Any]]:
    """
    ADK before_tool_callback.

    Returns *None* to allow the tool call to proceed, or a dict/string
    that ADK will use as the tool result (blocking the actual execution).
    """
    tool_name: str = tool.name if hasattr(tool, "name") else str(tool)

    # ── Source-level context budget for Exa ───────────────────────────
    _enforce_exa_budget(tool_name, args)

    # Use function_call_id as the per-invocation key.  ADK assigns a
    # unique function_call_id to every tool invocation, so parallel calls
    # never collide — unlike shared session-state keys.
    call_id = getattr(tool_context, "function_call_id", "") or tool_name

    # ── Per-provider rate limiting ────────────────────────────────────
    provider = _get_provider(tool_name)
    if provider:
        sem = _provider_semaphores[provider]
        # Non-blocking acquire — if semaphore is full, log a warning
        # but proceed anyway (we don't want to deadlock the pipeline).
        # threading.Semaphore.acquire(blocking=False) returns True on
        # success, False when the semaphore cannot be acquired.
        acquired = sem.acquire(blocking=False)
        if acquired:
            # Key by function_call_id — unique per invocation, no overwrites
            tool_context.state[f"_provider_sem_{call_id}"] = provider
            logger.debug(
                "Provider semaphore acquired: %s",
                provider,
            )
        else:
            # Semaphore full — proceed without holding it
            logger.debug(
                "Provider semaphore full for %s, proceeding without limit",
                provider,
            )

    # ── Dashboard: track tool start ───────────────────────────────────
    tool_context.state[f"_tool_start_time_{call_id}"] = time.time()
    _c = get_active_collector()
    if _c:
        agent_name = getattr(tool_context, "agent_name", "researcher")
        args_summary = str(args)[:300] if args else ""
        _c.tool_start(tool_name, agent_name, args_summary)

    return None  # allow execution
