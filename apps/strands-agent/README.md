# Strands Venice Agent

Venice GLM-4.7 uncensored research agent built with the [Strands Agents SDK](https://strandsagents.com/).

## What is this?

A deep-research agent that uses Venice AI's GLM-4.7 model (uncensored, OpenAI-compatible) with four MCP-based search tool families for comprehensive web intelligence gathering. Built on Strands Agents, it supports:

- **OpenAI-compatible model provider** — Venice AI with `include_venice_system_prompt: false` for uncensored operation
- **MCP tool integration** — Brave Search, Firecrawl, Exa, Kagi (conditionally loaded based on configured API keys)
- **Streaming responses** — real-time token streaming via `PrintingCallbackHandler`
- **Conversation memory** — sliding window context management (`SlidingWindowConversationManager`)
- **Multi-agent orchestration** — planner + researcher via the agent-as-tool pattern
- **Guardrails** — budget tracking callback for tool call limits and session timeouts
- **OpenTelemetry observability** — built-in OTEL tracing to Phoenix or any OTEL backend

## Install

```bash
# Using uv (recommended)
cd apps/strands-agent
uv sync

# Or using pip
pip install -e .
```

## Configure

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required:
- `VENICE_API_KEY` — Venice AI API key ([get one here](https://venice.ai))

Optional tool keys (agent loads only the tools whose keys are configured):
- `BRAVE_API_KEY` — Brave Search ([get key](https://brave.com/search/api/))
- `FIRECRAWL_API_KEY` — Firecrawl ([get key](https://firecrawl.dev))
- `EXA_API_KEY` — Exa ([get key](https://exa.ai))
- `KAGI_API_KEY` — Kagi ([get key](https://kagi.com/settings?p=api))

Optional observability:
- `OTEL_EXPORTER_OTLP_ENDPOINT` — OTLP endpoint (e.g. `http://localhost:6006/v1/traces` for Phoenix)

## Run

### Single-agent mode (default)

One agent with direct access to all search tools:

```bash
python agent.py
```

### Multi-agent mode

Planner agent delegates to a researcher agent (agent-as-tool pattern):

```bash
python agent.py --multi
```

### Environment variables for guardrails

```bash
MAX_TOOL_CALLS=200      # Max tool calls per session (default: 200)
SESSION_TIMEOUT=3600    # Session timeout in seconds (default: 3600)
```

## Available Tools

### Brave Search
- `brave_web_search` — broad web search
- `brave_local_search` — local business search
- `brave_image_search` — image search
- `brave_video_search` — video search
- `brave_news_search` — news search
- `brave_summarizer` — page summarization

### Firecrawl
- `firecrawl_scrape` — extract full content from URLs
- `firecrawl_search` — search and scrape
- `firecrawl_crawl` — crawl entire sites
- `firecrawl_map` — discover site URLs
- `firecrawl_extract` — structured data extraction

### Exa
- `web_search_exa` — quick semantic search
- `web_search_advanced_exa` — semantic search with category/domain/date filters
- `crawling_exa` — get content from specific URLs
- `get_code_context_exa` — code and documentation search

### Kagi
- `kagi_search` — premium web search
- `kagi_summarize` — summarize URLs (articles, PDFs, YouTube, audio)
- `kagi_fastgpt` — instant LLM-answered questions with sources
- `kagi_enrich_web` — small-web / indie content discovery
- `kagi_enrich_news` — non-mainstream news and discussions

## npm Prerequisites

The MCP servers require Node.js. Most use `npx` (auto-installed), but Exa requires a global install:

```bash
npm install -g exa-mcp-server
# brave, firecrawl use npx (auto-installed)
# kagi uses uvx (auto-installed)
```

This matches the existing requirement from `apps/adk-agent/tools/mcp_tools.py`.

## Architecture

```
┌──────────────────────────────────────────────┐
│  agent.py                                     │
│  ┌────────────┐    ┌───────────────────────┐ │
│  │  Planner   │───▶│  Researcher (as tool) │ │
│  │  (--multi) │    │  ┌─────────────────┐  │ │
│  └────────────┘    │  │ Brave MCP       │  │ │
│       OR           │  │ Firecrawl MCP   │  │ │
│  ┌────────────┐    │  │ Exa MCP         │  │ │
│  │  Single    │────│  │ Kagi MCP        │  │ │
│  │  Agent     │    │  └─────────────────┘  │ │
│  └────────────┘    └───────────────────────┘ │
│                                               │
│  config.py ─── Venice GLM-4.7 (OpenAI compat) │
│  prompts.py ── Uncensored research prompts    │
│  tools.py ──── MCP client wiring              │
└──────────────────────────────────────────────┘
```

## Framework Docs

- Strands Agents SDK: https://strandsagents.com/
- Venice AI API: https://docs.venice.ai/
