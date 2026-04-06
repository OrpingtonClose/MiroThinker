# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-tool callback implementing Algorithm 4 (Bad Result Detection)
and dynamic compression for web tool results.

Checks tool results for known error patterns and returns a replacement
error message so the LLM can recover gracefully.  Also completes
Algorithm 2 by recording successful queries in session state.

Dynamic Compression:
  After each Exa / Firecrawl / Brave tool call, the raw result is
  compressed into a structured memory dict (key facts, sources,
  confidence) that typically fits in ~500-800 tokens.  The LLM still
  sees enough to reason accurately, but the conversation history stays
  lean — giving 200+ rounds of headroom within Venice's 202K-token
  context window.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from utils.query_key import build_query_key

logger = logging.getLogger(__name__)

# Maximum length for scrape results in demo mode
DEMO_SCRAPE_MAX_LENGTH = 20_000

# Compression threshold: results above this size (chars) get compressed.
# Below this threshold, the result is small enough to keep as-is.
COMPRESSION_THRESHOLD = 4_000

# Tools whose results should be dynamically compressed
_COMPRESSIBLE_TOOLS = frozenset({
    # Exa
    "web_search_exa", "web_search_advanced_exa", "crawling_exa",
    "get_code_context_exa", "company_research_exa",
    # Firecrawl
    "firecrawl_scrape", "firecrawl_crawl", "firecrawl_search",
    "firecrawl_extract",
    # Brave
    "brave_web_search",
    # Kagi
    "kagi_search", "kagi_enrich_web", "kagi_enrich_news",
})



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
    if tool_name in ("google_search", "brave_web_search", "firecrawl_search", "web_search_exa", "web_search_advanced_exa", "kagi_search", "kagi_enrich_web", "kagi_enrich_news"):
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


def _extract_urls(text: str) -> List[str]:
    """Extract HTTP(S) URLs from text."""
    return re.findall(r'https?://[^\s"\'>)\]]+', text)


def _extract_key_facts(text: str, max_facts: int = 12) -> List[str]:
    """Extract the most informative sentences / bullet points from text.

    Heuristic: prefer lines that contain numbers, proper nouns (capitalised
    words), or specific data markers (dates, percentages, currencies).
    """
    # Split into sentences / bullet lines
    lines = re.split(r'[\n•\-\*]|(?<=\.)\s+', text)
    scored: List[tuple] = []
    for line in lines:
        line = line.strip()
        if len(line) < 20:
            continue
        score = 0
        # Boost lines with numbers / data
        score += len(re.findall(r'\d', line)) * 0.5
        # Boost lines with capitalised words (proper nouns)
        score += len(re.findall(r'\b[A-Z][a-z]{2,}', line)) * 0.3
        # Boost lines with specific markers
        if any(m in line for m in ('%', '$', '€', '£', 'http', '.com')):
            score += 2
        scored.append((score, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Deduplicate keeping order of score
    seen = set()
    facts = []
    for _, line in scored:
        key = line[:60].lower()
        if key not in seen:
            seen.add(key)
            facts.append(line[:300])  # cap individual fact length
            if len(facts) >= max_facts:
                break
    return facts


def _compress_result(tool_name: str, result_text: str) -> Optional[str]:
    """Compress a large tool result into structured memory.

    Returns a JSON string with {key_facts, sources, char_count, tool}
    or None if compression is not needed / not applicable.
    """
    if tool_name not in _COMPRESSIBLE_TOOLS:
        return None
    if not isinstance(result_text, str):
        return None
    if len(result_text) <= COMPRESSION_THRESHOLD:
        return None  # small enough to keep as-is

    original_len = len(result_text)

    # Extract structured data
    urls = _extract_urls(result_text)
    key_facts = _extract_key_facts(result_text)

    # Build source list from URLs (deduplicated, max 10)
    seen_domains: set = set()
    sources: List[Dict[str, str]] = []
    for url in urls:
        # Extract domain for dedup
        domain_match = re.match(r'https?://([^/]+)', url)
        domain = domain_match.group(1) if domain_match else url
        if domain not in seen_domains and len(sources) < 10:
            seen_domains.add(domain)
            sources.append({"url": url, "domain": domain})

    # Determine confidence based on source count and fact density
    if len(sources) >= 3 and len(key_facts) >= 5:
        confidence = "high"
    elif len(sources) >= 1 and len(key_facts) >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    compressed = {
        "_compressed": True,
        "tool": tool_name,
        "original_chars": original_len,
        "key_facts": key_facts,
        "sources": sources,
        "confidence": confidence,
    }

    result = json.dumps(compressed, ensure_ascii=False)
    logger.info(
        "Compressed %s result: %d chars → %d chars (%d facts, %d sources)",
        tool_name, original_len, len(result), len(key_facts), len(sources),
    )
    return result


def _maybe_truncate_scrape(tool_name: str, result_text: str) -> str:
    """In DEMO_MODE, truncate scrape_website results to 20K chars."""
    if os.environ.get("DEMO_MODE") != "1":
        return result_text
    if tool_name not in ("scrape", "scrape_website", "firecrawl_scrape", "crawling_exa"):
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

    # ── Dynamic Compression ──────────────────────────────────────────────
    # Compress large web tool results into structured memory so the
    # conversation history stays lean.  This is NOT truncation — the LLM
    # sees structured facts+sources, not a chopped-off blob.
    compressed = _compress_result(tool_name, result_text)
    if compressed is not None:
        return {"result": compressed}

    # ── DEMO_MODE truncation ────────────────────────────────────────────
    truncated = _maybe_truncate_scrape(tool_name, result_text)
    if truncated != result_text:
        return {"result": truncated}

    return None  # keep the original result
