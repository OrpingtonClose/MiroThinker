# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-tool callback — tool result truncation + dashboard tracking.

What this callback does:
1. **Tool result truncation** — caps large tool results (scrapes, crawls) to
   keep context usage under control.  Always active, not just DEMO_MODE.
2. **Dashboard tool_end tracking** — records duration and result size.
3. **Provider semaphore release** — releases per-provider rate-limit semaphore
   acquired in before_tool_callback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from callbacks.before_tool import _get_provider, _provider_semaphores
from dashboard import get_active_collector

logger = logging.getLogger(__name__)

# Maximum length for tool results before truncation.
# This is ALWAYS enforced (not just DEMO_MODE) to prevent context overflow.
# Scrape/crawl results are the main offender — they can be 100K+ chars.
TOOL_RESULT_MAX_CHARS = int(os.environ.get("TOOL_RESULT_MAX_CHARS", "25000"))

# Legacy alias for DEMO_MODE
DEMO_SCRAPE_MAX_LENGTH = 20_000


# Tools whose results are often very large and should be truncated.
_TRUNCATABLE_TOOLS = {
    "scrape", "scrape_website", "firecrawl_scrape", "firecrawl_crawl",
    "crawling_exa", "web_search_exa", "web_search_advanced_exa",
    "brave_web_search", "brave_news_search",
}


def _maybe_truncate_result(tool_name: str, result_text: str) -> str:
    """Truncate large tool results to stay within context budget.

    Always active — this is the primary defence against context overflow.
    In DEMO_MODE, uses the tighter DEMO_SCRAPE_MAX_LENGTH.
    """
    if not isinstance(result_text, str):
        return result_text

    max_chars = TOOL_RESULT_MAX_CHARS
    if os.environ.get("DEMO_MODE") == "1":
        max_chars = min(max_chars, DEMO_SCRAPE_MAX_LENGTH)

    # Only truncate tools known to produce large output
    if tool_name not in _TRUNCATABLE_TOOLS:
        # Still enforce a generous hard cap on any tool (100K chars)
        if len(result_text) > 100_000:
            logger.info(
                "Truncating unexpected large result from %s: %d -> 100000 chars",
                tool_name, len(result_text),
            )
            return result_text[:100_000] + "\n[truncated]"
        return result_text

    if len(result_text) <= max_chars:
        return result_text

    # Try JSON-aware truncation for structured results
    try:
        parsed = json.loads(result_text)
        text = parsed.get("text", "")
        if text and len(text) > max_chars:
            parsed["text"] = text[:max_chars] + "\n[truncated]"
            return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    # Plain truncation
    logger.info(
        "Truncating %s result: %d -> %d chars",
        tool_name, len(result_text), max_chars,
    )
    return result_text[:max_chars] + "\n[truncated]"


# ── callback ─────────────────────────────────────────────────────────────────


def after_tool_callback(
    tool: Any, args: Dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[Dict[str, Any]]:
    """
    ADK after_tool_callback.

    Returns *None* to keep the original result, or a replacement value
    that ADK substitutes as the tool result.
    """
    tool_name: str = tool.name if hasattr(tool, "name") else str(tool)
    result_text = str(tool_response) if tool_response is not None else ""

    # ── Release provider semaphore ───────────────────────────────────
    provider = _get_provider(tool_name)
    if provider:
        sem_key = f"_provider_sem_{provider}"
        if tool_context.state.get(sem_key):
            _provider_semaphores[provider].release()
            tool_context.state[sem_key] = False

    # ── Dashboard: track tool end ────────────────────────────────────
    start_time = tool_context.state.get("_tool_start_time", 0)
    duration = time.time() - start_time if start_time else 0.0
    _c = get_active_collector()
    if _c:
        agent_name = getattr(tool_context, "agent_name", "researcher")
        _c.tool_end(
            tool_name, agent_name, duration,
            result_chars=len(result_text),
        )

    # ── Tool result truncation (always active) ───────────────────────
    truncated = _maybe_truncate_result(tool_name, result_text)
    if truncated != result_text:
        return {"result": truncated}

    return None  # keep the original result
