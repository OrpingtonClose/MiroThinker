# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swappable orchestrator protocol.

Defines the minimal interface every research orchestrator backend must
implement. Today we ship a LangChain/deepagents implementation
(``orchestrator_langchain.LangChainOrchestrator``); future backends
(Strands-native, custom, etc.) slot in by implementing ``run``.

Backend selection is intended to be env-driven
(``ORCHESTRATOR_BACKEND=langchain|strands|...``) but only ``langchain``
is implemented today. See ``MANIFEST.md`` for the migration plan.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class OrchestratorEvent:
    """Normalised event emitted by any orchestrator backend.

    The ``type`` field is a small controlled vocabulary; backends may
    attach backend-specific payload under ``data``.

    Standard types:
    - ``tool_start`` — a tool (including launch_*) is about to execute.
      ``name`` is the tool name; ``data`` may include ``input``.
    - ``tool_end`` — tool finished. ``data`` may include ``output``.
    - ``task_launched`` — AsyncTaskPool accepted a new task.
    - ``task_completed`` — AsyncTaskPool task finished successfully.
    - ``task_failed`` — AsyncTaskPool task raised.
    - ``stream`` — partial assistant token stream.
      ``data["chunk"]`` carries the text.
    - ``error`` — unexpected failure inside the orchestrator itself.
    - ``final`` — final assistant content (if the backend surfaces one).
    """

    type: str
    name: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ResearchOrchestrator(Protocol):
    """Protocol every orchestrator backend must implement."""

    def run(self, query: str) -> AsyncIterator[OrchestratorEvent]:
        """Run the orchestrator on a query. Returns an async iterator of events.

        Implementations are expected to:
        - Be safe to call once per instance (concurrent calls on the same
          instance are not supported).
        - Honour cancellation via the underlying event loop (caller wraps
          the iteration in their own cancellation machinery).
        - Not raise from the iterator itself for expected failures;
          emit an ``error`` event instead and return.
        """
        ...
