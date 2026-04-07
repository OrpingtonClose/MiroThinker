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
        return prefix + tool_name + "_" + (args.get("q", "") or args.get("query", ""))
    if tool_name == "firecrawl_scrape":
        return prefix + tool_name + "_" + args.get("url", "")
    if tool_name == "firecrawl_search":
        return prefix + tool_name + "_" + (args.get("query", "") or args.get("q", ""))
    if tool_name == "firecrawl_map":
        return prefix + tool_name + "_" + args.get("url", "")
    if tool_name == "firecrawl_crawl":
        return prefix + tool_name + "_" + args.get("url", "")
    if tool_name == "firecrawl_extract":
        urls = args.get("urls", "")
        if isinstance(urls, list):
            urls = ",".join(sorted(urls))
        return prefix + tool_name + "_" + str(urls)
    # Exa MCP tools
    if tool_name in ("web_search_exa", "web_search_advanced_exa"):
        return prefix + tool_name + "_" + (args.get("query", "") or args.get("q", ""))
    if tool_name == "crawling_exa":
        return prefix + tool_name + "_" + args.get("url", "")
    # Kagi MCP tools
    if tool_name in ("kagi_search", "kagi_fastgpt", "kagi_enrich_web", "kagi_enrich_news"):
        return prefix + tool_name + "_" + (args.get("query", "") or args.get("q", ""))
    if tool_name == "kagi_summarize":
        return prefix + tool_name + "_" + args.get("url", "")
    # TranscriptAPI MCP tools
    if tool_name == "get_youtube_transcript":
        return prefix + tool_name + "_" + (args.get("video_url", "") or args.get("url", ""))
    if tool_name == "search_youtube":
        return prefix + tool_name + "_" + (args.get("query", "") or args.get("q", ""))
    if tool_name in ("list_channel_videos", "get_channel_latest_videos"):
        return prefix + tool_name + "_" + (args.get("channel", "") or args.get("channel_id", ""))
    if tool_name == "search_channel_videos":
        return prefix + tool_name + "_" + (args.get("channel", "") or args.get("channel_id", "")) + "_" + (args.get("query", "") or args.get("q", ""))
    if tool_name == "list_playlist_videos":
        return prefix + tool_name + "_" + args.get("playlist_id", "")
    return None
