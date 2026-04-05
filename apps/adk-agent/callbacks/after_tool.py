# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-tool callback implementing Algorithm 4 (Bad Result Detection).

Checks tool results for known error patterns and returns a replacement
error message so the LLM can recover gracefully.  Also completes
Algorithm 2 by recording successful queries in session state, and
truncates scrape results in DEMO_MODE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from dashboard.registry import get_collector
from utils.query_key import build_query_key

logger = logging.getLogger(__name__)

# Maximum length for scrape results in demo mode
DEMO_SCRAPE_MAX_LENGTH = 20_000


def _is_bad_result(tool_name: str, result_text: str) -> Optional[str]:
    """
    Algorithm 4: detect bad / error results and return a replacement message.

    Returns None when the result is fine; returns an error string otherwise.
    """
    if not result_text:
        return None

    text = str(result_text)

    # Unknown tool
    if text.startswith("Unknown tool:"):
        return text

    # Generic execution error
    if text.startswith("Error executing tool"):
        return text

    # Empty Google search results
    if tool_name == "google_search":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and parsed.get("organic") == []:
                return "Search returned no results. Try rephrasing your query."
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _maybe_truncate_scrape(tool_name: str, result_text: str) -> str:
    """In DEMO_MODE, truncate scrape_website results to 20K chars."""
    if os.environ.get("DEMO_MODE") != "1":
        return result_text
    if tool_name not in ("scrape", "scrape_website"):
        return result_text
    if not isinstance(result_text, str):
        return result_text

    try:
        parsed = json.loads(result_text)
        text = parsed.get("text", "")
        if text and len(text) > DEMO_SCRAPE_MAX_LENGTH:
            text = text[:DEMO_SCRAPE_MAX_LENGTH]
        return json.dumps({"text": text}, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        if len(result_text) > DEMO_SCRAPE_MAX_LENGTH:
            return result_text[:DEMO_SCRAPE_MAX_LENGTH]
        return result_text


# ── callback ─────────────────────────────────────────────────────────────────


def _emit_tool_call_end(
    tool_name: str,
    tool_context: ToolContext,
    result_text: str,
    error: str = "",
    was_bad_result: bool = False,
) -> None:
    """Emit TOOL_CALL_END so every TOOL_CALL_START gets a matching end event."""
    collector = get_collector()
    if not collector:
        return
    from dashboard.models import DashboardEvent, EventType

    start_t = tool_context.state.get("_tool_start_time", time.time())
    duration = round(time.time() - start_t, 4)
    collector.emit_sync(
        DashboardEvent(
            event_type=EventType.TOOL_CALL_END,
            agent_name=getattr(tool_context, "agent_name", ""),
            turn=collector.current_turn,
            data={
                "tool_name": tool_name,
                "duration_secs": duration,
                "result_size_chars": len(result_text),
                "error": error,
                "was_bad_result": was_bad_result,
            },
        )
    )


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

    # ── Algorithm 4: Bad Result Detection ───────────────────────────────
    state = tool_context.state
    collector = get_collector()

    bad = _is_bad_result(tool_name, result_text)
    if bad is not None:
        logger.warning("Bad result detected for %s: %s", tool_name, bad[:200])
        if collector:
            from dashboard.models import DashboardEvent, EventType

            collector.emit_sync(
                DashboardEvent(
                    event_type=EventType.BAD_RESULT_DETECTED,
                    agent_name=getattr(tool_context, "agent_name", ""),
                    turn=collector.current_turn,
                    data={
                        "tool_name": tool_name,
                        "reason": bad[:200],
                        "error_message": bad[:500],
                    },
                )
            )
        # Emit TOOL_CALL_END even on bad results so start/end counts match
        _emit_tool_call_end(tool_name, tool_context, result_text, error=bad[:200], was_bad_result=True)
        return {"error": bad}

    # ── Complete Algorithm 2: record successful query ───────────────────
    if "seen_queries" not in state:
        state["seen_queries"] = {}

    query_key = build_query_key(tool_name, args)
    if query_key is not None:
        state["seen_queries"][query_key] = state["seen_queries"].get(query_key, 0) + 1

    # ── DEMO_MODE truncation ────────────────────────────────────────────
    truncated = _maybe_truncate_scrape(tool_name, result_text)
    if truncated != result_text:
        # Emit TOOL_CALL_END before returning truncated result
        _emit_tool_call_end(tool_name, tool_context, truncated)
        return {"result": truncated}

    # Emit tool-call-end event for the normal path
    _emit_tool_call_end(tool_name, tool_context, result_text)

    return None  # keep the original result
