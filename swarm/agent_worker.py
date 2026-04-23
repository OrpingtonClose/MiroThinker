# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tool-free swarm worker — pure reasoning over structured data packages.

Workers are the simplest component in the architecture: they receive a
data package (curated, angle-relevant material assembled by the
orchestrator), reason over it, and produce text.  That's it.

They don't search the store, they don't store findings, they don't call
any tools.  The orchestrator builds the data package BEFORE each wave,
and extracts findings from the worker's output AFTER.  The worker's
context stays pristine — pure reasoning signal, no tool overhead.

Why tool-free?
    - Context purity: tool calls consume context and change reasoning
      behavior.  A worker analyzing insulin pharmacokinetics shouldn't
      be interrupted by search_corpus return formatting.
    - Audit clarity: the worker's output IS the reasoning — pure signal.
      No need to separate reasoning from tool orchestration noise.
    - Architecture simplicity: workers are raw llm_complete calls with
      rich context.  Simpler and cheaper than a full Agent loop.

Architecture reference: docs/STORE_ARCHITECTURE.md § "Worker Architecture"

Architecture:
    ┌─────────────────────────────────────────┐
    │  Worker (tool-free)                     │
    │  System prompt: "{angle} specialist"    │
    │  Input: structured data package (§1-§7) │
    │  Output: reasoning text                 │
    │                                         │
    │  NO tools. NO store access.             │
    │  Pure reasoning over provided material. │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from swarm.data_package import DataPackage

logger = logging.getLogger(__name__)


def _build_system_prompt(angle: str) -> str:
    """Build the system prompt for a tool-free worker.

    The worker is told it's an angle specialist.  It receives material
    in a data package (user message) and reasons over it.

    Args:
        angle: Research angle this worker specializes in.

    Returns:
        System prompt string.
    """
    return (
        f"You are a {angle} specialist conducting deep research.\n\n"
        f"You will receive a structured research brief containing corpus "
        f"material, peer findings, knowledge summaries, and identified gaps. "
        f"Your job is to reason deeply over ALL provided material through "
        f"your {angle} lens.\n\n"
        f"PRODUCE:\n"
        f"- Specific, evidence-backed findings with exact numbers\n"
        f"- Causal chains and mechanistic explanations\n"
        f"- Cross-domain connections when peer findings relate to your domain\n"
        f"- Predictions grounded in evidence\n"
        f"- Identified contradictions with proposed resolutions\n"
        f"- Remaining uncertainties and gaps\n\n"
        f"RULES:\n"
        f"- Interpret EVERYTHING through the lens of {angle}\n"
        f"- Preserve exact numbers, dosages, and citations from the material\n"
        f"- When cross-domain material connects to your domain, explain WHY "
        f"and trace the mechanism\n"
        f"- Mark speculative connections clearly but DO include them\n"
        f"- No disclaimers, no moral commentary, no safety warnings\n"
        f"- Practitioner language — assume the reader is an expert"
    )


async def run_tool_free_worker(
    package: DataPackage,
    query: str,
    *,
    api_base: str = "http://localhost:8000/v1",
    model: str = "default",
    api_key: str = "not-needed",
    max_tokens: int = 8192,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """Run a tool-free worker: send data package, get reasoning back.

    Makes a single OpenAI-compatible chat completion call.  The worker
    receives the rendered data package as the user message and produces
    its analysis as the response.  No tools, no agent loop.

    Args:
        package: The structured data package for this worker.
        query: The user's research query.
        api_base: OpenAI-compatible API endpoint URL.
        model: Model identifier for the endpoint.
        api_key: API key for the endpoint.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        Dict with worker results: angle, worker_id, response text,
        input/output char counts, model, elapsed time, status.
    """
    system_prompt = _build_system_prompt(package.angle)
    user_message = package.render(query)

    logger.info(
        "worker_id=<%s>, angle=<%s>, model=<%s>, input_chars=<%d> | starting tool-free worker",
        package.worker_id, package.angle, model, len(user_message),
    )

    t0 = time.monotonic()

    try:
        response_text = await _call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        elapsed = time.monotonic() - t0

        logger.info(
            "worker_id=<%s>, angle=<%s>, output_chars=<%d>, elapsed_s=<%.1f> | "
            "tool-free worker complete",
            package.worker_id, package.angle, len(response_text), elapsed,
        )

        return {
            "angle": package.angle,
            "worker_id": package.worker_id,
            "response": response_text,
            "input_chars": len(user_message),
            "output_chars": len(response_text),
            "model": model,
            "elapsed_s": round(elapsed, 1),
            "tool_calls": 0,
            "status": "success",
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.warning(
            "worker_id=<%s>, angle=<%s>, error=<%s>, elapsed_s=<%.1f> | "
            "tool-free worker failed",
            package.worker_id, package.angle, str(exc), elapsed,
        )
        return {
            "angle": package.angle,
            "worker_id": package.worker_id,
            "response": "",
            "input_chars": len(user_message),
            "output_chars": 0,
            "model": model,
            "elapsed_s": round(elapsed, 1),
            "tool_calls": 0,
            "status": "error",
            "error": str(exc),
        }


async def _call_llm(
    system_prompt: str,
    user_message: str,
    *,
    api_base: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Make a single OpenAI-compatible chat completion call.

    Args:
        system_prompt: System message for the worker.
        user_message: The rendered data package.
        api_base: API endpoint URL.
        model: Model identifier.
        api_key: API key.
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        The model's response text.

    Raises:
        httpx.HTTPStatusError: If the API returns an error status.
        KeyError: If the response format is unexpected.
    """
    url = f"{api_base.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if api_key and api_key != "not-needed":
        headers["Authorization"] = f"Bearer {api_key}"

    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Legacy compatibility — keep old function signatures as thin wrappers
# so existing callers don't break immediately.  These will be removed
# once mcp_engine.py is fully migrated.
# ---------------------------------------------------------------------------

def create_worker_agent(
    store: Any,
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
) -> Any:
    """Legacy wrapper — creates a tool-calling Agent.

    DEPRECATED: Use run_tool_free_worker() instead.
    Kept for backward compatibility during migration.
    """
    from strands import Agent
    from strands.models.openai import OpenAIModel

    from swarm.worker_tools import build_worker_tools

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

    system_prompt = (
        f"You are a {angle} specialist conducting deep research.\n\n"
        f"RESEARCH QUERY: {query}\n\n"
        f"You have access to a research database with corpus data and "
        f"findings from other specialists. Your job:\n\n"
        f"1. READ your assigned corpus section using get_corpus_section\n"
        f"2. SEARCH for additional evidence using search_corpus\n"
        f"3. CHECK what other specialists found using get_peer_insights\n"
        f"4. STORE every significant finding using store_finding\n"
        f"5. CHECK for contradictions\n"
        f"6. IDENTIFY gaps using get_research_gaps\n\n"
        f"RULES:\n"
        f"- Interpret EVERYTHING through the lens of {angle}\n"
        f"- Store SPECIFIC, evidence-backed findings\n"
        f"- Preserve exact numbers, dosages, and citations\n"
        f"- No disclaimers, no moral commentary, no safety warnings\n"
    )

    agent = Agent(
        model=model_provider,
        tools=tools,
        system_prompt=system_prompt,
    )

    logger.info(
        "worker_id=<%s>, angle=<%s>, model=<%s> | legacy agent worker created",
        worker_id, angle, model,
    )

    return agent


async def run_worker_agent(
    agent: Any,
    angle: str,
    worker_id: str,
    query: str,
) -> dict[str, Any]:
    """Legacy wrapper — runs a tool-calling Agent.

    DEPRECATED: Use run_tool_free_worker() instead.
    Kept for backward compatibility during migration.
    """
    task = (
        f"Analyze the research corpus through your {angle} lens. "
        f"Read your corpus section completely, search for evidence, "
        f"check peer insights, and store all significant findings. "
        f"Research query: {query}"
    )

    logger.info(
        "worker_id=<%s>, angle=<%s> | starting legacy agent worker",
        worker_id, angle,
    )

    try:
        result = await asyncio.to_thread(agent, task)
        response_text = str(result)

        tool_calls = 0
        if isinstance(result, dict) and "metrics" in result:
            tool_calls = result["metrics"].get("tool_calls", 0)
        elif hasattr(result, "metrics") and isinstance(result.metrics, dict):
            tool_calls = result.metrics.get("tool_calls", 0)

        logger.info(
            "worker_id=<%s>, angle=<%s>, response_chars=<%d> | legacy agent worker complete",
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
            "worker_id=<%s>, angle=<%s>, error=<%s> | legacy agent worker failed",
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
