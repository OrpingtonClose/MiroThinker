# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""LangChain / deepagents implementation of ``ResearchOrchestrator``.

This is the **only** module outside ``orchestrator.py`` that imports
LangChain / LangGraph / deepagents. Consumers of the orchestrator
(e.g. ``main._run_job``) depend solely on ``orchestrator_protocol``.

Backend selection is stubbed for now — see ``MANIFEST.md`` §10.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from orchestrator_protocol import OrchestratorEvent, ResearchOrchestrator

logger = logging.getLogger(__name__)


def _summarise_input(raw: Any, limit: int = 200) -> str:
    """Render tool input as a short string for event payloads."""
    if isinstance(raw, dict):
        try:
            text = json.dumps(raw)
        except (TypeError, ValueError):
            text = str(raw)
    else:
        text = str(raw)
    return text[:limit]


class LangChainOrchestrator(ResearchOrchestrator):
    """Adapts a deepagents ``CompiledStateGraph`` to ``ResearchOrchestrator``.

    Maps LangGraph events (``on_tool_start``, ``on_tool_end``,
    ``on_chat_model_stream``) to the backend-neutral ``OrchestratorEvent``
    type. All LangChain surface area stops at this file.
    """

    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    async def run(self, query: str) -> AsyncIterator[OrchestratorEvent]:
        try:
            async for raw in self._graph.astream_events(
                {"messages": [HumanMessage(content=query)]},
                version="v2",
                config={"recursion_limit": 9_999},
            ):
                mapped = self._map(raw)
                if mapped is not None:
                    yield mapped
        except Exception as exc:
            logger.exception("LangChainOrchestrator: astream_events failed")
            yield OrchestratorEvent(
                type="error",
                name="astream_events",
                data={"error": str(exc)},
            )

    def _map(self, raw: dict[str, Any]) -> OrchestratorEvent | None:
        etype = raw.get("event", "")
        name = raw.get("name", "")
        data = raw.get("data", {}) or {}

        if etype == "on_tool_start":
            return OrchestratorEvent(
                type="tool_start",
                name=name,
                data={"input_summary": _summarise_input(data.get("input", ""))},
            )

        if etype == "on_tool_end":
            output = data.get("output", "")
            output_str = str(output)
            return OrchestratorEvent(
                type="tool_end",
                name=name,
                data={
                    "output": output,
                    "output_summary": (
                        output_str[:200] + "..."
                        if len(output_str) > 200
                        else output_str
                    ),
                },
            )

        if etype == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk is not None and getattr(chunk, "content", ""):
                return OrchestratorEvent(
                    type="stream",
                    name=name,
                    data={"chunk": chunk.content},
                )
            return None

        # Ignore all other LangGraph events (on_chain_*, on_llm_*, etc.).
        return None
