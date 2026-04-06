# MiroThinker ADK Agent

A reimplementation of MiroThinker's 8 core algorithms using only
[Google ADK](https://google.github.io/adk-docs/) primitives. No code is
ported from `apps/miroflow-agent/` — only the algorithms are preserved.

## Quick Start

```bash
# Install dependencies (requires uv)
uv sync

# Copy and fill in API keys
cp .env.example .env

# Run the agent
python main.py
```

## Architecture

```
apps/adk-agent/
├── pyproject.toml
├── .env.example
├── README.md
├── main.py                    # Entry point with context compression retry loop
├── agents/
│   ├── research.py            # Main research agent definition
│   ├── browsing.py            # Browsing sub-agent definition
│   └── summary.py             # Summary/final-answer agent definition
├── callbacks/
│   ├── before_tool.py         # Algorithms 2 (dedup guard) + 8 (arg fix)
│   ├── after_tool.py          # Algorithm 4 (bad result detection)
│   ├── before_model.py        # Algorithm 5 (keep-K-recent)
│   └── after_model.py         # Algorithm 7 (intermediate boxed extraction)
├── tools/
│   └── mcp_tools.py           # MCPToolset factory from existing MCP servers
├── prompts/
│   └── templates.py           # Prompt text (from prompt_utils.py, minus XML instructions)
└── utils/
    ├── boxed.py               # \boxed{} extraction logic
    └── input_handler.py       # Import from existing libs (unchanged)
```

## Algorithm Mapping

| # | Algorithm | MiroThinker | ADK Implementation |
|---|-----------|-------------|-------------------|
| 1 | ReAct Loop | Orchestrator while-loop (1200 lines) | ADK `Agent` (built-in) |
| 2 | Dedup Guard | `Orchestrator._check_duplicate_query` | `before_tool_callback` + session state |
| 3 | Delegation | `expose_sub_agents_as_tools` + intercept | `Agent(sub_agents=[...])` native |
| 4 | Bad Result | `ToolExecutor.should_rollback_result` | `after_tool_callback` returns error |
| 5 | Keep-K-Recent | `BaseClient._remove_tool_result_from_messages` | `before_model_callback` |
| 6 | Context Compression | Pipeline retry loop + AnswerGenerator | Outer Python loop around `Runner` |
| 7 | Boxed Extraction | `OutputFormatter._extract_boxed_content` | `after_model_callback` + session state |
| 8 | Arg Fix | `ToolExecutor.fix_tool_call_arguments` | `before_tool_callback` |

### What's Eliminated

MiroThinker's XML parsing (`parsing_utils.py`), MCP tag format instructions
(`prompt_utils.py` lines 85-164), server_name fixing (`parsing_utils.py`
lines 24-121), and format error rollback (`orchestrator.py` lines 180-255)
are **all eliminated** because ADK uses native function calling.

## Key Design Decisions

### Dedup Guard (Algorithm 2)
Since ADK doesn't support `history.pop()` rollback, the dedup guard returns
an error message instead. The LLM naturally adjusts because it sees
"Duplicate query detected" as the tool result. An escape hatch allows
duplicates through after 5 consecutive blocked attempts (matching
MiroThinker's `MAX_CONSECUTIVE_ROLLBACKS=5`).

### Keep-K-Recent (Algorithm 5)
Implemented via `before_model_callback` which modifies the LLM request's
`contents` list in-place, replacing old tool results with a placeholder
string. Also performs a rough context-length check and sets a `force_end`
flag when the threshold is exceeded.

### Context Compression Retry (Algorithm 6)
Implemented as an outer Python `for` loop around the ADK `Runner`. Each
retry creates a **fresh session** (empty history) but prepends failure
experience summaries from prior attempts to the user message.

### Browsing Sub-Agent (Algorithm 3)
The main agent can delegate to `browsing_agent` via ADK's native
`transfer_to_agent` mechanism. The LLM decides when to delegate based on
the browsing agent's description. No fake tool interception needed.

### Playwright Browser
Since ADK's `MCPToolset` doesn't support persistent sessions, the Playwright
browser tools are wrapped as `FunctionTool` instances that internally manage
a shared `PlaywrightSession` (from `miroflow_tools.mcp_servers.browser_session`).

## Configuration

All configuration is via environment variables (see `.env.example`):

- **`ADK_MODEL`**: The model identifier for ADK agents. Supports both
  Gemini native (`gemini-2.0-flash`) and LiteLLM proxy
  (`litellm/openai/<model>`).
- **`KEEP_TOOL_RESULT`**: Number of recent tool results to keep (default: 5).
- **`MAX_CONTEXT_TOKENS`**: Force final answer above this token estimate
  (default: 128000).
- **`CONTEXT_COMPRESS_LIMIT`**: Maximum retry attempts (default: 5).
