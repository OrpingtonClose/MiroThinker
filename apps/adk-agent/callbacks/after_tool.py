# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-tool callback — tool result truncation + dashboard tracking + corpus ingestion.

What this callback does:
1. **Tool result truncation** — caps large tool results (scrapes, crawls) to
   keep context usage under control.  Always active, not just DEMO_MODE.
2. **Dashboard tool_end tracking** — records duration and result size.
3. **Provider semaphore release** — releases per-provider rate-limit semaphore
   acquired in before_tool_callback.
4. **Search result ingestion** — feeds search tool results into the Flock corpus
   via ``ingest_raw()`` so they get atomised, scored, and deduped alongside
   all other findings.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

from callbacks.before_tool import _TOOL_TO_PROVIDER, _provider_semaphores
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

# Search tools whose results should be ingested into the Flock corpus.
# Each tool maps to its source_type tag for condition tracking.
_SEARCH_TOOLS: dict[str, str] = {
    "brave_web_search": "brave_web_search",
    "brave_news_search": "brave_news_search",
    "web_search_exa": "exa_search",
    "web_search_advanced_exa": "exa_search",
    "crawling_exa": "exa_crawl",
    "kagi_search": "kagi_search",
    "kagi_enrich_web": "kagi_enrich",
    "kagi_enrich_news": "kagi_enrich",
    "firecrawl_scrape": "firecrawl_scrape",
    "firecrawl_crawl": "firecrawl_crawl",
    "firecrawl_search": "firecrawl_search",
    # Academic & reference
    "search_papers": "semantic_scholar",
    "get_paper": "semantic_scholar",
    "get_paper_citations": "semantic_scholar",
    "get_paper_references": "semantic_scholar",
    "search_by_category": "arxiv",
    # Wikipedia
    "search": "wikipedia_search",
    "read": "wikipedia_read",
    # Bright Data
    "search_engine": "brightdata_search",
    "scrape_as_markdown": "brightdata_scrape",
    # Deep research tools
    "perplexity_deep_research": "perplexity_deep",
    "grok_deep_research": "grok_deep",
    "tavily_deep_research": "tavily_deep",
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
    # Use function_call_id as the per-invocation key — same key that
    # before_tool_callback used, and it's an attribute on ToolContext
    # (not shared session state), so parallel calls can't overwrite it.
    call_id = getattr(tool_context, "function_call_id", "") or tool_name
    sem_key = f"_provider_sem_{call_id}"
    held_provider = tool_context.state.get(sem_key, "")
    if held_provider:
        _provider_semaphores[held_provider].release()
        tool_context.state[sem_key] = ""

    # ── Dashboard: track tool end ────────────────────────────────────
    start_time = tool_context.state.get(f"_tool_start_time_{call_id}", 0)
    duration = time.time() - start_time if start_time else 0.0
    _c = get_active_collector()
    if _c:
        agent_name = getattr(tool_context, "agent_name", "researcher")
        _c.tool_end(
            tool_name, agent_name, duration,
            result_chars=len(result_text),
        )

    # ── Record cost for deep research tools ────────────────────────
    _deep_providers = {"perplexity", "grok", "tavily"}
    provider_for_cost = _TOOL_TO_PROVIDER.get(tool_name)
    if provider_for_cost in _deep_providers:
        # Only record cost if the tool actually made an API call.
        # Tools return "[TOOL_ERROR] ... unavailable" when the API key
        # is missing — recording phantom costs would inflate the session
        # total and block properly-configured providers.
        _is_no_call_error = (
            (result_text.startswith("[TOOL_ERROR]") and "unavailable" in result_text)
            or "BUDGET EXCEEDED" in result_text
        )
        if not _is_no_call_error:
            try:
                from tools.cost_tracker import ESTIMATED_COSTS, get_cost_tracker

                tracker = get_cost_tracker()
                estimated = ESTIMATED_COSTS.get(provider_for_cost, 1.0)
                query_text = args.get("query", "") if isinstance(args, dict) else ""
                tracker.record_cost(provider_for_cost, estimated, query_text)
                logger.info(
                    "Recorded deep research cost: %s $%.2f (session total: $%.2f)",
                    provider_for_cost, estimated,
                    tracker.get_session_stats()["session_total_usd"],
                )
            except Exception:
                logger.debug("Cost recording failed for %s", tool_name, exc_info=True)

    # ── Queue search results for corpus ingestion ──────────────────
    # Search tool results are queued for later ingestion via the
    # thread-safe queue in condition_manager.  The actual ingest_raw()
    # call happens in researcher_condition_callback on the main thread,
    # avoiding concurrent DuckDB access from parallel tool callbacks.
    source_type = _SEARCH_TOOLS.get(tool_name)
    if source_type and result_text and len(result_text) > 50:
        try:
            from callbacks.condition_manager import queue_search_result

            corpus_key = tool_context.state.get("_corpus_key", "")
            if corpus_key:
                iteration = tool_context.state.get("_corpus_iteration", 0)
                queue_search_result(
                    corpus_key,
                    result_text[:TOOL_RESULT_MAX_CHARS],
                    source_type,
                    iteration,
                )
                logger.info(
                    "Queued %s result (%d chars) for corpus ingestion",
                    tool_name, min(len(result_text), TOOL_RESULT_MAX_CHARS),
                )
        except Exception:
            logger.debug(
                "Search result queuing skipped for %s",
                tool_name, exc_info=True,
            )

    # ── Tool result truncation (always active) ───────────────────────
    truncated = _maybe_truncate_result(tool_name, result_text)
    if truncated != result_text:
        return {"result": truncated}

    return None  # keep the original result
