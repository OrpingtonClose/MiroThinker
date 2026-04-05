# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Shared helper for building canonical dedup keys from tool calls.

Used by both ``before_tool`` (to check duplicates) and ``after_tool``
(to record successful queries).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_query_key(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Build a canonical dedup key from tool name + relevant arg.

    Returns ``None`` for tool types that are not tracked.
    """
    if tool_name == "google_search":
        return tool_name + "_" + args.get("q", "")
    if tool_name == "sogou_search":
        return tool_name + "_" + args.get("Query", "")
    if tool_name == "scrape_website":
        return tool_name + "_" + args.get("url", "")
    if tool_name == "scrape_and_extract_info":
        return (
            tool_name
            + "_"
            + args.get("url", "")
            + "_"
            + args.get("info_to_extract", "")
        )
    if tool_name == "search_and_browse":
        return tool_name + "_" + args.get("subtask", "")
    return None
