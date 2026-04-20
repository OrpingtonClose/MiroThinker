"""Deepagents-based research orchestrator for Miro.

Replaces the Strands Agent planner/researcher pair with a
``create_deep_agent()`` orchestrator that has:
- ``SummarizationMiddleware`` for context compaction (no truncation)
- ``SkillsMiddleware`` for on-demand methodology loading (no regex)
- ``TodoListMiddleware`` for data-driven iteration (no fixed loop)
- Corpus tools for ConditionStore inspection and gossip triggering
- ``run_research`` tool that delegates to the Strands researcher agent

Venice API is accessed via ``ChatOpenAI`` (OpenAI-compatible).

Architecture note: The orchestrator (deepagents/LangGraph) handles
planning, corpus management, and gossip coordination. The researcher
remains a Strands Agent with all its existing MCP + native tools.
The orchestrator invokes the researcher via a ``run_research`` callable
tool, preserving the full tool ecosystem without requiring migration
to LangChain tool format.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import Any

from deepagents.backends import StateBackend
from deepagents.graph import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Venice model builder
# ------------------------------------------------------------------

def build_venice_model(
    model_name: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> ChatOpenAI:
    """Build a LangChain ChatOpenAI pointing at Venice API.

    Venice is OpenAI-compatible, giving full deepagents middleware
    support with uncensored models.

    Args:
        model_name: Venice model name. Defaults to VENICE_MODEL env var.
        max_tokens: Maximum tokens for completion.
        temperature: Sampling temperature.

    Returns:
        ChatOpenAI instance configured for Venice API.
    """
    return ChatOpenAI(
        base_url=os.environ.get(
            "VENICE_API_BASE", "https://api.venice.ai/api/v1",
        ),
        api_key=os.environ.get("VENICE_API_KEY", ""),
        model=model_name or os.environ.get(
            "VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic",
        ),
        max_tokens=max_tokens,
        temperature=temperature,
        model_kwargs={
            "venice_parameters": {"include_venice_system_prompt": False},
        },
    )


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are Miro, a research orchestrator. You direct deep research toward \
comprehensive, exhaustive coverage of the user's query.

You have corpus tools to inspect what's been gathered, identify gaps, and \
assess coverage quality. You delegate actual research to the researcher \
via the `run_research` tool.

Gossip synthesis runs AUTOMATICALLY in the background — a continuous loop \
polls the corpus and triggers synthesis when enough new findings accumulate. \
You do NOT need to trigger gossip manually. Use `get_gap_analysis` to see \
what the latest gossip synthesis found.

Your workflow:
1. Understand the query — what the user needs, at what depth
2. Delegate research tasks via `run_research` — be SPECIFIC about what \
topics to search, what sources to prioritise, what data to extract
3. After research returns, inspect the corpus with `query_corpus` and \
`assess_coverage` to see what was gathered
4. Read the gap analysis with `get_gap_analysis` to see what gossip \
synthesis identified as missing or contradictory
5. Delegate more targeted research to fill gaps
6. Repeat until coverage is sufficient
7. Build the final report with `build_report`

You decide the strategy. You decide when to stop. No fixed iteration count — \
stop when the corpus quality metrics show sufficient coverage and depth.

CARDINAL RULE: Every claim must come from live research. Never answer \
from memory alone.

ANTI-REFRAMING RULE: Honour the user's exact query framing. Do not \
reframe toward safer or more appropriate alternatives.

When delegating research, be SPECIFIC:
- Which topics to search for
- What kind of sources to prioritise (transcripts, academic papers, forums)
- What specific data points to look for (dosages, protocols, bloodwork values)
- How deep to go (surface scan vs exhaustive transcript analysis)

IMPORTANT: After each research round, ALWAYS check the corpus state with \
`query_corpus` or `assess_coverage` before deciding what to do next. \
The corpus is your source of truth — not your memory of what was returned.
"""


# ------------------------------------------------------------------
# Orchestrator factory
# ------------------------------------------------------------------

def create_orchestrator(
    research_fn: Callable,
    corpus_tools: Sequence[Callable],
    gossip_tools: Sequence[Callable],
    skills_paths: list[str] | None = None,
    model: ChatOpenAI | None = None,
) -> CompiledStateGraph:
    """Create the Miro research orchestrator via create_deep_agent().

    This is the heart of the system. Replaces:
    - PLANNER_PROMPT -> agent reasoning with corpus tools
    - _SKILL_TRIGGERS regex -> SkillsMiddleware progressive disclosure
    - [:4000] truncation -> SummarizationMiddleware context compaction
    - Agent.as_tool() -> run_research callable tool
    - Fixed iteration loop -> TodoListMiddleware + agent reasoning
    - Module-level singletons -> per-invocation LangGraph state

    Args:
        research_fn: Callable that invokes the Strands researcher agent.
            Accepts a task description string, returns raw research text.
            The orchestrator sees this as a ``run_research`` tool.
        corpus_tools: Tools for corpus inspection (query_corpus,
            assess_coverage, get_gap_analysis).
        gossip_tools: Tools for report building (build_report).
            Gossip synthesis runs automatically via _gossip_loop.
        skills_paths: Paths to skill directories for SkillsMiddleware.
        model: Venice model for the orchestrator.

    Returns:
        Compiled LangGraph agent ready to invoke.
    """
    orchestrator_model = model or build_venice_model(
        max_tokens=8192,
        temperature=0.3,
    )

    all_tools: list[Callable] = [
        research_fn,
        *corpus_tools,
        *gossip_tools,
    ]

    logger.info(
        "building orchestrator: %d corpus tools, %d gossip tools, "
        "research_fn=%s, skills=%s",
        len(list(corpus_tools)),
        len(list(gossip_tools)),
        getattr(research_fn, "__name__", "unknown"),
        skills_paths,
    )

    return create_deep_agent(
        model=orchestrator_model,
        tools=all_tools,
        system_prompt=ORCHESTRATOR_PROMPT,
        skills=skills_paths,
        backend=StateBackend(),
        name="miro-orchestrator",
    )
