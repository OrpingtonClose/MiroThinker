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

import json
import logging
import os
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from callbacks.before_tool import _classify_query
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

    # Empty / error search results (handles both legacy JSON and MCP text)
    if tool_name in ("google_search", "brave_web_search", "firecrawl_search"):
        # MCP servers return plain-text or multi-part text; legacy tools
        # returned JSON with {"organic": [...]} or {"data": [...]}
        stripped = text.strip()
        if not stripped or stripped in ("[]", "{}", "null"):
            return "Search returned no results. Try rephrasing your query."
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                if parsed.get("error"):
                    return f"Search API error: {parsed['error']}"
                # Legacy Brave wrapper format
                if parsed.get("organic") == []:
                    return "Search returned no results. Try rephrasing your query."
                # Firecrawl format
                if parsed.get("data") == [] and parsed.get("success") is not False:
                    return "Search returned no results. Try rephrasing your query."
            if isinstance(parsed, list) and len(parsed) == 0:
                return "Search returned no results. Try rephrasing your query."
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _maybe_truncate_scrape(tool_name: str, result_text: str) -> str:
    """In DEMO_MODE, truncate scrape_website results to 20K chars."""
    if os.environ.get("DEMO_MODE") != "1":
        return result_text
    if tool_name not in ("scrape", "scrape_website", "firecrawl_scrape"):
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

    bad = _is_bad_result(tool_name, result_text)
    if bad is not None:
        logger.warning("Bad result detected for %s: %s", tool_name, bad[:200])
        return {"error": bad}

    # ── Complete Algorithm 2: record successful query ───────────────────
    if "seen_queries" not in state:
        state["seen_queries"] = {}

    agent_name: str = getattr(tool_context, "agent_name", "")
    query_key = build_query_key(tool_name, args, agent_name=agent_name)
    if query_key is not None:
        state["seen_queries"][query_key] = state["seen_queries"].get(query_key, 0) + 1

    # ── Complete Algorithm 9: record successful search category ──────────
    if tool_name in ("brave_web_search", "firecrawl_search"):
        search_query = args.get("q", "") or args.get("query", "")
        if search_query:
            category = _classify_query(search_query)
            cat_counts_key = "search_category_counts"
            if cat_counts_key not in state:
                state[cat_counts_key] = {}
            cat_counts: Dict[str, int] = state[cat_counts_key]
            cat_counts[category] = cat_counts.get(category, 0) + 1
            state[cat_counts_key] = cat_counts
            logger.info(
                "Search diversity (post-exec): query '%s' → category '%s' (count: %d)",
                search_query[:60], category, cat_counts[category],
            )

    # ── DEMO_MODE truncation ────────────────────────────────────────────
    truncated = _maybe_truncate_scrape(tool_name, result_text)
    if truncated != result_text:
        return {"result": truncated}

    return None  # keep the original result
