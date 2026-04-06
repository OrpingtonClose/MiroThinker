# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Before-tool callback implementing Algorithm 2 (Dedup Guard),
Algorithm 8 (Arg Fix), and Algorithm 9 (Search Diversity Guard).

Algorithm 8 — Arg Fix:
  Correct common parameter-name mistakes the LLM makes when calling tools.

Algorithm 2 — Dedup Guard:
  Prevent the same tool+query from being executed twice.  Instead of the
  MiroThinker approach of popping history (which ADK doesn't expose), we
  return an error string that replaces the tool result, so the LLM naturally
  adjusts.  An escape hatch allows duplicates through after
  MAX_CONSECUTIVE_ERRORS consecutive blocked attempts.

Algorithm 9 — Search Diversity Guard:
  Track which source categories each brave_web_search query targets.
  When a category has been searched MAX_PER_CATEGORY times, block further
  searches in that category and nudge the LLM toward under-explored ones.
  Categories: forum, vendor, local_language, social_media, news, academic,
  general.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from utils.query_key import build_query_key

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5
MAX_PER_CATEGORY = 3  # max searches per source category before nudging


# ── helpers ──────────────────────────────────────────────────────────────────


# ── Search Diversity Guard (Algorithm 9) ────────────────────────────────────

# Keywords that signal which source category a search query targets.
_CATEGORY_SIGNALS: Dict[str, List[str]] = {
    "forum": [
        "forum", "reddit", "thread", "community", "discussion", "4chan",
        "imageboard", "telegram", "discord", "eroids", "meso-rx",
        "bodybuilding.com", "ask", "experience", "review",
    ],
    "vendor": [
        "vendor", "buy", "shop", "store", "sell", "price", "order",
        "marketplace", "darknet", "peptide", "source", "supplier",
        "warehouse", "wholesale",
    ],
    "local_language": [],  # detected by non-ASCII heuristic
    "social_media": [
        "youtube", "twitter", "tiktok", "instagram", "x.com",
        "video", "vlog", "channel",
    ],
    "news": [
        "news", "article", "investigation", "report", "journalism",
        "documentary", "expose", "bust",
    ],
    "academic": [
        "study", "research", "regulation", "law", "legal", "pubmed",
        "journal", "policy", "government", "official",
    ],
}


def _classify_query(query: str) -> str:
    """Classify a search query into a source category."""
    q_lower = query.lower()

    # Non-ASCII chars suggest a foreign-language query
    non_ascii_ratio = sum(1 for c in query if ord(c) > 127) / max(len(query), 1)
    if non_ascii_ratio > 0.15:
        return "local_language"

    # Check each category's signal words
    best_category = "general"
    best_score = 0
    for category, signals in _CATEGORY_SIGNALS.items():
        score = sum(1 for kw in signals if kw in q_lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category


def _get_underexplored_categories(category_counts: Dict[str, int]) -> List[str]:
    """Return categories that haven't been searched yet or have few searches."""
    all_categories = ["forum", "vendor", "local_language", "social_media", "news", "academic"]
    underexplored = []
    for cat in all_categories:
        count = category_counts.get(cat, 0)
        if count == 0:
            underexplored.append(cat)
    if not underexplored:
        # All categories touched at least once; return those below max
        underexplored = [cat for cat in all_categories if category_counts.get(cat, 0) < MAX_PER_CATEGORY]
    return underexplored


_CATEGORY_HINTS = {
    "forum": "Try searching forums/communities: add 'reddit', 'forum', 'discussion', or 'experience' to your query.",
    "vendor": "Try searching vendor/marketplace sites: add 'buy', 'vendor', 'shop', 'source', or 'supplier' to your query.",
    "local_language": "Try searching in the user's LOCAL LANGUAGE (e.g. Polish, German, Spanish) — not English.",
    "social_media": "Try searching social media: add 'youtube', 'twitter', 'tiktok', or 'video' to your query.",
    "news": "Try searching news/journalism: add 'news', 'article', 'investigation', or 'report' to your query.",
    "academic": "Try searching academic/regulatory sources: add 'study', 'regulation', 'law', or 'pubmed' to your query.",
}


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

    # ── Algorithm 9: Search Diversity Guard ──────────────────────────────
    if tool_name in ("brave_web_search", "firecrawl_search", "web_search_exa", "web_search_advanced_exa"):
        search_query = args.get("q", "") or args.get("query", "")
        if search_query:
            category = _classify_query(search_query)
            cat_counts_key = "search_category_counts"

            if cat_counts_key not in state:
                state[cat_counts_key] = {}

            cat_counts: Dict[str, int] = state[cat_counts_key]

            current_count = cat_counts.get(category, 0)
            if current_count >= MAX_PER_CATEGORY:
                underexplored = _get_underexplored_categories(cat_counts)
                if underexplored:
                    hints = " ".join(
                        _CATEGORY_HINTS.get(c, "") for c in underexplored[:3]
                    ).strip()
                    logger.info(
                        "Diversity guard: category '%s' saturated (%d/%d). "
                        "Nudging toward: %s",
                        category, current_count, MAX_PER_CATEGORY, underexplored,
                    )
                    return {
                        "error": (
                            f"You have already made {current_count} searches in the "
                            f"'{category}' category. Search a DIFFERENT category. "
                            f"Under-explored categories: {', '.join(underexplored)}. "
                            f"{hints}"
                        )
                    }

            # Category count is incremented in after_tool_callback
            # (only after confirming the search succeeded), matching
            # how Algorithm 2 records queries post-execution.
            logger.info(
                "Search diversity: query '%s' → category '%s' (pre-exec, count so far: %d)",
                search_query[:60], category, current_count,
            )

    return None  # allow execution; recording happens in after_tool
