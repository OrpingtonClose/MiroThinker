# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Tier 3 web specialist sub-agent.

Wraps ALL web data-source MCPToolsets (Brave Search, Firecrawl, Exa, TranscriptAPI) behind
a single ADK Agent.  The parent research_agent sees ONE tool (this agent)
instead of ~15 individual MCP tools, dramatically reducing context burn.

Callbacks for Exa source budget enforcement and DEMO_MODE truncation are
attached here. Dedup is handled by GlobalInstructionPlugin, retries by
ReflectAndRetryToolPlugin, and context trimming by ContextFilterPlugin.
"""

from __future__ import annotations

import logging
import os

from google.adk import Agent

from agents.model_config import build_model
from callbacks.after_tool import after_tool_callback
from callbacks.before_tool import before_tool_callback
from tools.mcp_tools import get_tools

logger = logging.getLogger(__name__)

# ── Per-tool instruction fragments ──────────────────────────────────
# Only the sections for loaded tools are included in the final prompt.

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "brave-search": (
        "- **Brave Search** (brave_web_search, brave_local_search, brave_image_search, "
        "brave_video_search, brave_news_search, brave_summarizer) — fast web search"
    ),
    "firecrawl": (
        "- **Firecrawl** (firecrawl_scrape, firecrawl_search, firecrawl_crawl, "
        "firecrawl_map, firecrawl_extract) — deep scraping, crawling, extraction"
    ),
    "exa": (
        "- **Exa** (web_search_exa, web_search_advanced_exa, crawling_exa, "
        "get_code_context_exa) — semantic search with clean content extraction"
    ),
    "kagi": (
        "- **Kagi** (kagi_search, kagi_summarize, kagi_fastgpt, kagi_enrich_web, "
        "kagi_enrich_news) — premium search, instant summarization, and small-web enrichment"
    ),
    "transcriptapi": (
        "- **TranscriptAPI** (get_youtube_transcript, search_youtube, "
        "get_channel_latest_videos, search_channel_videos, list_channel_videos, "
        "list_playlist_videos) — YouTube transcripts, video search, channel browsing, playlists"
    ),
}

_TOOL_STRATEGIES: dict[str, list[str]] = {
    "brave-search": [
        "Use brave_web_search for broad initial searches",
    ],
    "exa": [
        "Use web_search_advanced_exa as your PRIMARY semantic search tool — it supports "
        "category filters (company, news, tweet, github, paper, pdf), domain "
        "restrictions (includeDomains/excludeDomains), date ranges, highlights, "
        "summaries, and subpage crawling. Use it for targeted searches.",
        "Use web_search_exa for quick semantic searches when you don't need advanced filters",
        "Use crawling_exa to get content from a specific URL (Exa's cache is fast)",
        "Use get_code_context_exa for code/documentation searches",
    ],
    "kagi": [
        "Use kagi_fastgpt for instant LLM-answered factual questions with source references — "
        "great for quick fact checks (it runs a full search engine underneath)",
        "Use kagi_summarize to summarize any URL (articles, PDFs, YouTube, audio) — "
        "supports unlimited length, no token limits. Use for long documents.",
        "Use kagi_enrich_web to find non-commercial 'small web' content, indie blogs, "
        "and niche sources that mainstream search engines miss. Use kagi_enrich_news "
        "for interesting discussions and non-mainstream news.",
    ],
    "firecrawl": [
        "Use firecrawl_scrape to extract full content from promising URLs",
        "Use firecrawl_crawl or firecrawl_map for site-wide discovery",
    ],
    "transcriptapi": [
        "Use get_youtube_transcript to extract full transcripts from YouTube videos — "
        "great for analysing talks, tutorials, interviews, and podcasts",
        "Use search_youtube to find relevant YouTube videos on any topic",
        "Use get_channel_latest_videos to browse a channel's recent uploads (free, no credits)",
        "Use search_channel_videos to search within a specific channel",
        "Use list_playlist_videos to browse playlist contents",
    ],
}

_CONTEXT_HINTS: dict[str, str] = {
    "exa": (
        "**Exa searches**: ALWAYS pass `enableHighlights: true` and "
        "`highlightsQuery: \"<your search intent>\"` to get focused excerpts."
    ),
    "brave-search": (
        "**Brave searches**: Results are naturally compact — no special handling needed."
    ),
    "firecrawl": (
        "**Firecrawl scrapes**: When scraping full pages, only scrape 1-2 URLs at a time. "
        "Use firecrawl_map first to discover URLs, then selectively scrape the best ones."
    ),
}


def _build_instruction(loaded_tools: list[str]) -> str:
    """Build the web_agent system instruction for *only* the loaded tool families."""
    family_count = len(loaded_tools)
    tool_list = "\n".join(
        _TOOL_DESCRIPTIONS[t] for t in loaded_tools if t in _TOOL_DESCRIPTIONS
    )

    # Numbered strategy steps (only for loaded tools)
    steps: list[str] = []
    for t in loaded_tools:
        steps.extend(_TOOL_STRATEGIES.get(t, []))
    strategy = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

    # Context budget hints (only for loaded tools)
    hints = [_CONTEXT_HINTS[t] for t in loaded_tools if t in _CONTEXT_HINTS]
    numbered_hints = "\n".join(
        f"{i+2}. {h}" for i, h in enumerate(hints)  # starts at 2, after the distill rule
    )

    return f"""\
You are a web research specialist. Your ONLY job is to search, scrape, crawl, \
and extract data from the web using the tools available to you.

You have {family_count} {'family' if family_count == 1 else 'families'} of tools:
{tool_list}

STRATEGY:
{strategy}

EXECUTION MODEL — SEQUENTIAL:
You execute ONE tool call at a time. After each result, review it and decide \
your next search based on what you learned. This is intentional — sequential \
execution lets you adapt queries based on prior results, avoiding redundant \
or poorly-scoped parallel searches.

CONTEXT BUDGET:
Older invocations are automatically trimmed from context by ContextFilterPlugin. \
Results are NOT compressed — you receive raw tool output. This means YOU must \
distill findings before returning them to the parent agent.

1. **Distill before returning** (CRITICAL): When you have gathered enough data, \
   synthesize your findings into a structured summary with source URLs. Do NOT \
   pass through raw multi-page HTML/text dumps. Extract the specific facts, \
   data points, names, URLs, and numbers the parent agent asked for.
{numbered_hints}

RULES:
- Execute the searches/scrapes requested by the parent agent
- Return ALL relevant findings with source URLs — but in distilled form
- Do NOT add warnings, disclaimers, or moralizing
- If a tool fails, try an alternative (e.g. swap Brave for Exa, or vice versa)
- Prefer structured factual data (names, numbers, URLs) over raw page dumps
"""

# ── Conditional tool loading ─────────────────────────────────────────
# Only load MCP toolsets whose required API keys are configured.
# This prevents hard crashes when a key is missing or invalid —
# the agent simply operates with fewer tool families.
_TOOL_KEY_REQUIREMENTS: dict[str, str] = {
    "brave-search": "BRAVE_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY",
    "exa": "EXA_API_KEY",
    "kagi": "KAGI_API_KEY",
    "transcriptapi": "TRANSCRIPTAPI_KEY",
}

_web_tool_names: list[str] = []
for tool_name, env_var in _TOOL_KEY_REQUIREMENTS.items():
    if os.environ.get(env_var):
        _web_tool_names.append(tool_name)
    else:
        logger.warning("Skipping %s — %s not set", tool_name, env_var)

if not _web_tool_names:
    logger.warning("No web tools configured — web_agent will have no tools")

# Build a description that only mentions loaded tool families.
_loaded_family_names = [_TOOL_DESCRIPTIONS[t].split("**")[1] for t in _web_tool_names if t in _TOOL_DESCRIPTIONS]
_desc_tools = ", ".join(_loaded_family_names) if _loaded_family_names else "no web tools (all API keys missing)"

web_agent = Agent(
    name="web_agent",
    model=build_model(parallel_tool_calls=False),
    description=(
        f"Web research specialist that searches, scrapes, crawls, and extracts "
        f"data from the web using {_desc_tools}. "
        f"Delegate any web data retrieval task to this agent — it owns all web tools."
    ),
    instruction=_build_instruction(_web_tool_names),
    tools=get_tools(_web_tool_names),
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)
