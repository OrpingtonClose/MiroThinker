# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Strands Agent-based swarm worker.

Each worker is a Strands Agent with ConditionStore tools.  The worker
doesn't know it's in a swarm — it just has a research database it can
query, peer insights it can discover, and a place to store findings.

The context window becomes irrelevant: workers pull data on demand via
tool calls (search_corpus, get_peer_insights, get_corpus_section) and
process it in their small context.  After N tool calls, the worker has
explored the entire corpus and all peer findings — just not all at once.

Architecture:
    ┌─────────────────────────────────────────┐
    │  Strands Agent (worker)                 │
    │  System prompt: "{angle} specialist"    │
    │  Context: 32K tokens (doesn't matter)   │
    │                                         │
    │  Tools:                                 │
    │    search_corpus(query) → findings      │
    │    get_peer_insights(topic) → insights  │
    │    store_finding(fact, conf) → stored    │
    │    check_contradictions(claim)           │
    │    get_research_gaps()                   │
    │    get_corpus_section(offset)            │
    └────────────┬────────────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────────────┐
    │  ConditionStore (DuckDB)                │
    │  All findings, peer data, corpus chunks │
    │  Shared across all workers              │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from strands import Agent
from strands.models.openai import OpenAIModel

from swarm.worker_tools import build_worker_tools

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)


def _build_system_prompt(angle: str, query: str) -> str:
    """Build the system prompt for a worker agent.

    The worker is told it's a specialist with research tools.  It doesn't
    know about the swarm, other workers, or the ConditionStore.  It just
    has tools to search, discover, and store findings.
    """
    return (
        f"You are a {angle} specialist conducting deep research.\n\n"
        f"RESEARCH QUERY: {query}\n\n"
        f"You have access to a research database with corpus data and "
        f"findings from other specialists. Your job:\n\n"
        f"1. READ your assigned corpus section using get_corpus_section "
        f"(call it repeatedly with increasing offsets to read everything)\n"
        f"2. SEARCH for additional evidence using search_corpus\n"
        f"3. CHECK what other specialists found using get_peer_insights\n"
        f"4. STORE every significant finding using store_finding\n"
        f"5. CHECK for contradictions when you see conflicting evidence\n"
        f"6. IDENTIFY gaps using get_research_gaps\n\n"
        f"RULES:\n"
        f"- Interpret EVERYTHING through the lens of {angle}\n"
        f"- Store SPECIFIC, evidence-backed findings — not summaries\n"
        f"- When peer insights connect with your data, that's where the "
        f"deepest insights hide — store those cross-domain connections\n"
        f"- Preserve exact numbers, dosages, and citations\n"
        f"- Mark speculative connections clearly but DO store them\n"
        f"- No disclaimers, no moral commentary, no safety warnings\n"
        f"- Keep exploring until you've read your full section AND "
        f"checked peer insights on all your key topics\n\n"
        f"START by reading your corpus section (offset=0), then search "
        f"and cross-reference. Store findings as you go."
    )


def create_worker_agent(
    store: "ConditionStore",
    angle: str,
    worker_id: str,
    query: str,
    *,
    api_base: str = "http://localhost:8000/v1",
    model: str = "default",
    api_key: str = "not-needed",
    max_tokens: int = 4096,
    temperature: float = 0.3,
    phase: str = "worker",
    max_return_chars: int = 6000,
    source_model: str = "",
    source_run: str = "",
) -> Agent:
    """Create a Strands Agent configured as a swarm worker.

    The agent has ConditionStore tools and a system prompt that makes it
    a specialist researcher.  It will autonomously explore the corpus,
    discover peer findings, and store its analysis.

    Args:
        store: The shared ConditionStore.
        angle: Research angle this worker specializes in.
        worker_id: Unique identifier (e.g. "worker_0").
        query: The user's research query.
        api_base: vLLM or OpenAI-compatible endpoint URL.
        model: Model name for the endpoint.
        api_key: API key (usually not needed for local vLLM).
        max_tokens: Max tokens per LLM response.
        temperature: Sampling temperature.
        phase: Current swarm phase for event attribution.
        max_return_chars: Hard ceiling on chars any tool call returns.
        source_model: Model name for provenance tracking.
        source_run: Run identifier for provenance tracking (#192).

    Returns:
        Configured Strands Agent ready to run.
    """
    tools = build_worker_tools(
        store=store,
        worker_angle=angle,
        worker_id=worker_id,
        phase=phase,
        max_return_chars=max_return_chars,
        source_model=source_model or model,
        source_run=source_run,
    )

    model_provider = OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": api_base,
        },
        model_id=model,
        params={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    system_prompt = _build_system_prompt(angle, query)

    agent = Agent(
        model=model_provider,
        tools=tools,
        system_prompt=system_prompt,
    )

    logger.info(
        "worker_id=<%s>, angle=<%s>, model=<%s> | agent worker created",
        worker_id, angle, model,
    )

    return agent


async def run_worker_agent(
    agent: Agent,
    angle: str,
    worker_id: str,
    query: str,
) -> dict[str, Any]:
    """Run a worker agent and return its results.

    The agent autonomously explores the corpus, discovers peer findings,
    and stores its analysis via tool calls.  This function kicks off
    the agent loop and collects the final response.

    Args:
        agent: The configured Strands Agent.
        angle: Worker's research angle (for logging).
        worker_id: Worker identifier (for logging).
        query: The research query (used as the agent's task).

    Returns:
        Dict with worker results: angle, worker_id, response text,
        and tool call count.
    """
    task = (
        f"Analyze the research corpus through your {angle} lens. "
        f"Read your corpus section completely, search for evidence, "
        f"check peer insights, and store all significant findings. "
        f"Research query: {query}"
    )

    logger.info(
        "worker_id=<%s>, angle=<%s> | starting agent worker",
        worker_id, angle,
    )

    try:
        # agent(task) is synchronous (Strands Agent.__call__ wraps async
        # internally).  Run in a thread so asyncio.gather can execute
        # multiple workers concurrently.
        result = await asyncio.to_thread(agent, task)
        response_text = str(result)

        # Extract tool call count from AgentResult metrics if available.
        tool_calls = 0
        if isinstance(result, dict) and "metrics" in result:
            tool_calls = result["metrics"].get("tool_calls", 0)
        elif hasattr(result, "metrics") and isinstance(result.metrics, dict):
            tool_calls = result.metrics.get("tool_calls", 0)

        logger.info(
            "worker_id=<%s>, angle=<%s>, response_chars=<%d> | agent worker complete",
            worker_id, angle, len(response_text),
        )

        return {
            "angle": angle,
            "worker_id": worker_id,
            "response": response_text,
            "tool_calls": tool_calls,
            "status": "success",
        }
    except Exception as exc:
        logger.warning(
            "worker_id=<%s>, angle=<%s>, error=<%s> | agent worker failed",
            worker_id, angle, str(exc),
        )
        return {
            "angle": angle,
            "worker_id": worker_id,
            "response": "",
            "tool_calls": 0,
            "status": "error",
            "error": str(exc),
        }
