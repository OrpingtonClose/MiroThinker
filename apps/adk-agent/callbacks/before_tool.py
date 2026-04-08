# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-tool callback — source-level context budget for Exa tools.

Algorithm 2 (Dedup Guard) has been removed — ``GlobalInstructionPlugin``
injects cross-cutting "never repeat a query" instructions into every agent's
system prompt, letting the LLM enforce dedup via its own reasoning.

Algorithm 8 (Arg Fix) has been removed — ``ReflectAndRetryToolPlugin``
catches tool execution failures (including wrong parameter names), shows the
LLM structured reflection guidance with the error message, and retries.

What remains:
**Source-level context budget for Exa** — injects ``textMaxCharacters`` and
``numResults`` into every Exa tool call so results are bounded at the source.
This is equivalent to a SQL LIMIT clause: the API simply returns less data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

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

    return None  # allow execution
