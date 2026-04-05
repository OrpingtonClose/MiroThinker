# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-tool callback implementing Algorithm 2 (Dedup Guard) and
Algorithm 8 (Arg Fix).

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


# ── helpers ──────────────────────────────────────────────────────────────────


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

    # ── Algorithm 8: Arg Fix ────────────────────────────────────────────
    original_args = dict(args)  # snapshot before fix
    _fix_args(tool_name, args)

    if args != original_args:
        logger.info("Arg fix applied for %s: %s", tool_name, list(args.keys()))

    # ── Algorithm 2: Dedup Guard ────────────────────────────────────────

    # Initialise session-state buckets on first use
    if "seen_queries" not in state:
        state["seen_queries"] = {}
    if "consecutive_dedup_errors" not in state:
        state["consecutive_dedup_errors"] = 0

    query_key = build_query_key(tool_name, args)

    if query_key is not None:
        seen: Dict[str, int] = state["seen_queries"]

        if query_key in seen:
            consecutive = state["consecutive_dedup_errors"] + 1
            state["consecutive_dedup_errors"] = consecutive

            if consecutive > MAX_CONSECUTIVE_ERRORS:
                # Escape hatch: allow the duplicate through
                logger.warning(
                    "Dedup escape hatch: allowing duplicate after %d consecutive blocks "
                    "(tool=%s, key=%s)",
                    consecutive,
                    tool_name,
                    query_key,
                )
                state["consecutive_dedup_errors"] = 0
                # Fall through to allow execution
            else:
                logger.info(
                    "Dedup guard blocked duplicate (tool=%s, key=%s, consecutive=%d)",
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
            state["consecutive_dedup_errors"] = 0

    return None  # allow execution; recording happens in after_tool
