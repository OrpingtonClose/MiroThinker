# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Shared helper for building canonical dedup keys from tool calls.

Used by both ``before_tool`` (to check duplicates) and ``after_tool``
(to record successful queries).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_query_key(
    tool_name: str, args: Dict[str, Any], agent_name: str = ""
) -> Optional[str]:
    """Build a canonical dedup key from tool name + relevant arg.

    Keys are namespaced by *agent_name* so that each agent maintains its
    own independent dedup cache — matching the original MiroThinker
    behaviour where ``cache_name = agent_id + "_" + tool_name``.

    Returns ``None`` for tool types that are not tracked.
    """
    prefix = f"{agent_name}:" if agent_name else ""

    if tool_name == "google_search":
        return prefix + tool_name + "_" + args.get("q", "")
    if tool_name == "sogou_search":
        return prefix + tool_name + "_" + args.get("Query", "")
    if tool_name == "scrape_website":
        return prefix + tool_name + "_" + args.get("url", "")
    if tool_name == "scrape_and_extract_info":
        return (
            prefix
            + tool_name
            + "_"
            + args.get("url", "")
            + "_"
            + args.get("info_to_extract", "")
        )
    if tool_name == "search_and_browse":
        return prefix + tool_name + "_" + args.get("subtask", "")
    if tool_name == "brave_web_search":
        return prefix + tool_name + "_" + args.get("q", "")
    if tool_name == "firecrawl_scrape":
        return prefix + tool_name + "_" + args.get("url", "")
    return None
