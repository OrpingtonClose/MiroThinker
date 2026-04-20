"""LangGraph-based researcher agent.

Replaces the Strands Agent researcher with a create_deep_agent()
instance that uses ChatOpenAI (same as the orchestrator). This
eliminates the dual async runtime that caused the Strands researcher
to hang on dead Venice API connections.

Architecture:
  orchestrator (LangGraph/ChatOpenAI)
    └── run_research tool
          └── researcher (LangGraph/ChatOpenAI)  ← NEW
                └── search tools (sync, run in thread pool)

Previously:
  orchestrator (LangGraph/ChatOpenAI)
    └── run_research tool
          └── researcher (Strands/OpenAIModel)   ← HANGS
                └── search tools + MCP tools

The fix: one async runtime end-to-end. No Strands event loop, no
MCP subprocess management, no dual asyncio bridging.
"""

from __future__ import annotations

import logging

from deepagents.backends import StateBackend
from deepagents.graph import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

from langchain_tools import get_all_research_tools
from orchestrator import build_venice_model

logger = logging.getLogger(__name__)


RESEARCHER_PROMPT = """\
You are a research specialist executing a specific research task. Your job \
is to find raw data — facts, numbers, protocols, citations — and return \
everything you find.

EXECUTION RULES:
1. Use multiple search tools per task. Start with forums and DuckDuckGo \
(uncensored), then Perplexity/Grok for depth, then Jina for extraction.
2. Be SPECIFIC in your queries. Not "insulin protocols" but \
"site:meso-rx.org VigorousSteve trenbolone insulin timing 5IU humalog".
3. When you find a promising forum thread, use forum_read_thread or \
jina_read_url to extract the FULL text — snippets are not enough.
4. Always include source URLs with every finding.
5. Search in multiple languages when relevant (German, Polish, Russian \
forums have unique practitioner knowledge).
6. If a tool returns an error or empty results, try a different tool \
or rephrase the query. Do NOT give up after one attempt.
7. Return ALL raw data. Do not summarize prematurely. The orchestrator \
will synthesize — your job is to gather.

TOOL PRIORITY (uncensored-first):
1. Forums: forum_search, forum_deep_dive — practitioner knowledge from \
MesoRx, EliteFitness, Professional Muscle, AnabolicMinds, international
2. Uncensored web: duckduckgo_search, mojeek_search, stract_search
3. Deep research: perplexity_deep_research, grok_deep_research, \
tavily_deep_research
4. Extraction: jina_read_url, wayback_search, wayback_fetch
5. Censored (last resort): google_search, exa_multi_search — these \
reject health/PED queries

ANTI-CENSORSHIP: If any tool refuses or returns empty for a health/PED \
query, immediately switch to an uncensored alternative. Never waste time \
retrying a censored tool with the same query.

OUTPUT FORMAT: Return a comprehensive raw research report with ALL data \
gathered. Include specific numbers, protocols, dosages, bloodwork values, \
and source URLs. Structure by topic area. No disclaimers.
"""


def create_researcher(
    model_name: str | None = None,
    max_tokens: int = 8192,
) -> CompiledStateGraph:
    """Create a LangGraph researcher agent.

    Uses the same ChatOpenAI model client as the orchestrator (via
    build_venice_model), ensuring all Venice API calls go through
    the same async-capable HTTP transport.

    Args:
        model_name: Venice model name. Defaults to env var.
        max_tokens: Max tokens for completion.

    Returns:
        Compiled LangGraph agent ready to invoke/ainvoke.
    """
    researcher_model = build_venice_model(
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=0.2,
    )

    tools = get_all_research_tools()

    logger.info(
        "building langchain researcher: %d tools, model=%s",
        len(tools),
        researcher_model.model_name,
    )

    return create_deep_agent(
        model=researcher_model,
        tools=tools,
        system_prompt=RESEARCHER_PROMPT,
        backend=StateBackend(),
        name="miro-researcher",
    )
