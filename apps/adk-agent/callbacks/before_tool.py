# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-tool callback implementing Algorithm 2 (Dedup Guard),
Algorithm 8 (Arg Fix), and source-level context budget for Exa tools.

Source-level Context Budget:
  Injects Exa API parameters (textMaxCharacters, numResults) into every Exa
  tool call so results are bounded at the source — no post-hoc truncation.
  This is equivalent to a SQL LIMIT clause: the API simply returns less data.

Algorithm 8 — Arg Fix:
  Correct common parameter-name mistakes the LLM makes when calling tools.

Algorithm 2 — Dedup Guard:
  Prevent the same tool+query from being executed twice.  Instead of the
  MiroThinker approach of popping history (which ADK doesn't expose), we
  return an error string that replaces the tool result, so the LLM naturally
  adjusts.  An escape hatch allows duplicates through after
  MAX_CONSECUTIVE_ERRORS consecutive blocked attempts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from utils.query_key import build_query_key

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5

# Source-level context budget defaults for Exa tools.
# These cap how much TEXT the Exa API returns per result — not truncation,
# just telling the API to send less, like numResults caps how many results.
# With dynamic compression in after_tool, raw results are summarised into
# structured memory (~500-800 tokens) so we can afford larger per-result
# text from the API.  The cap is an emergency safety rail for pathological
# results (PDF dumps, endless boilerplate) — not the primary constraint.
EXA_DEFAULT_TEXT_MAX_CHARS = 10_000  # emergency cap per-result (chars)
EXA_DEFAULT_NUM_RESULTS = 8          # max results per search
EXA_CRAWL_TEXT_MAX_CHARS = 15_000    # crawling a specific URL gets more


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
        # Cap numResults if LLM asked for too many
        if args.get("numResults", 0) > 10:
            args["numResults"] = 10
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


def _fix_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Algorithm 8: fix common parameter-name mistakes in-place."""
    # scrape_and_extract_info: "description"/"introduction" → "info_to_extract"
    if tool_name == "scrape_and_extract_info" and "info_to_extract" not in args:
        for wrong_name in ("description", "introduction"):
            if wrong_name in args:
                args["info_to_extract"] = args.pop(wrong_name)
                break

    # run_python_code: "code" → "code_block", ensure sandbox_id
    if tool_name == "run_python_code":
        if "code_block" not in args and "code" in args:
            args["code_block"] = args.pop("code")
        if "sandbox_id" not in args:
            args["sandbox_id"] = "default"

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
    state = tool_context.state

    # ── Source-level context budget for Exa ───────────────────────────
    _enforce_exa_budget(tool_name, args)

    # ── Algorithm 8: Arg Fix ────────────────────────────────────────────
    original_args = dict(args)  # snapshot before fix
    _fix_args(tool_name, args)

    if args != original_args:
        logger.info("Arg fix applied for %s: %s", tool_name, list(args.keys()))

    # ── Algorithm 2: Dedup Guard ────────────────────────────────────────
    # Per-agent isolation: each agent has its own dedup cache and
    # consecutive-error counter, matching original MiroThinker where
    # cache_name = agent_id + "_" + tool_name.
    agent_name: str = getattr(tool_context, "agent_name", "")
    dedup_counter_key = f"consecutive_dedup_errors:{agent_name}"

    # Initialise session-state buckets on first use
    if "seen_queries" not in state:
        state["seen_queries"] = {}
    if dedup_counter_key not in state:
        state[dedup_counter_key] = 0

    query_key = build_query_key(tool_name, args, agent_name=agent_name)

    if query_key is not None:
        seen: Dict[str, int] = state["seen_queries"]

        if query_key in seen:
            consecutive = state[dedup_counter_key] + 1
            state[dedup_counter_key] = consecutive

            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                # Escape hatch: allow the duplicate through
                logger.warning(
                    "Dedup escape hatch: allowing duplicate after %d consecutive blocks "
                    "(agent=%s, tool=%s, key=%s)",
                    consecutive,
                    agent_name,
                    tool_name,
                    query_key,
                )
                state[dedup_counter_key] = 0
                # Fall through to allow execution
            else:
                logger.info(
                    "Dedup guard blocked duplicate (agent=%s, tool=%s, key=%s, consecutive=%d)",
                    agent_name,
                    tool_name,
                    query_key,
                    consecutive,
                )
                return {
                    "error": (
                        f"Duplicate query detected for '{query_key}'. "
                        "Use a different search query or URL."
                    )
                }
        else:
            # Not a duplicate — reset the consecutive error counter
            state[dedup_counter_key] = 0
    else:
        # Non-tracked tool call — also reset consecutive counter
        # (matches original MiroThinker algorithm which resets after any successful turn)
        state[dedup_counter_key] = 0

    return None  # allow execution; recording happens in after_tool
