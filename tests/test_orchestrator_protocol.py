"""Unit tests for ``apps/strands-agent/orchestrator_protocol.py``.

Verifies:

- ``OrchestratorEvent`` dataclass construction / defaults / to-dict-like
  behaviour of the ``data`` field.
- ``ResearchOrchestrator`` is a runtime-checkable Protocol with a ``run``
  method — any class that exposes a compatible ``run`` signature
  structurally satisfies ``isinstance(obj, ResearchOrchestrator)``.
- A minimal async-generator ``run`` implementation yields
  ``OrchestratorEvent`` instances with the expected controlled-vocabulary
  ``type`` values.

These tests do not import LangChain / Strands.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "strands-agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


orchestrator_protocol = importlib.import_module("orchestrator_protocol")
OrchestratorEvent = orchestrator_protocol.OrchestratorEvent
ResearchOrchestrator = orchestrator_protocol.ResearchOrchestrator


# ---------------------------------------------------------------------------
# OrchestratorEvent
# ---------------------------------------------------------------------------


def test_orchestrator_event_defaults():
    ev = OrchestratorEvent(type="tool_start")
    assert ev.type == "tool_start"
    assert ev.name == ""
    assert ev.data == {}


def test_orchestrator_event_with_payload():
    ev = OrchestratorEvent(
        type="task_launched",
        name="launch_research",
        data={"task_id": "task-research-abc", "description": "probe"},
    )
    assert ev.type == "task_launched"
    assert ev.name == "launch_research"
    assert ev.data["task_id"] == "task-research-abc"


def test_orchestrator_event_data_is_per_instance():
    """Regression: the ``data`` default_factory must not be shared."""
    a = OrchestratorEvent(type="tool_start")
    b = OrchestratorEvent(type="tool_start")
    a.data["foo"] = 1
    assert "foo" not in b.data


# ---------------------------------------------------------------------------
# ResearchOrchestrator protocol compliance
# ---------------------------------------------------------------------------


class _MinimalBackend:
    """Structural implementation of ``ResearchOrchestrator`` used to
    verify that any class exposing a compatible ``run`` satisfies the
    ``runtime_checkable`` Protocol.
    """

    async def run(self, query: str):
        yield OrchestratorEvent(type="tool_start", name="launch_research")
        yield OrchestratorEvent(
            type="task_launched",
            name="launch_research",
            data={"task_id": "task-research-xyz"},
        )
        yield OrchestratorEvent(type="stream", data={"chunk": "hello"})
        yield OrchestratorEvent(type="final", data={"content": "final answer"})


def test_minimal_backend_satisfies_protocol():
    backend = _MinimalBackend()
    assert isinstance(backend, ResearchOrchestrator)


def test_protocol_rejects_objects_without_run():
    class _Missing:
        pass

    assert not isinstance(_Missing(), ResearchOrchestrator)


def test_minimal_backend_yields_expected_event_types():
    backend = _MinimalBackend()

    async def _collect():
        out = []
        async for ev in backend.run("hello"):
            out.append(ev)
        return out

    events = asyncio.run(_collect())
    types = [ev.type for ev in events]
    assert types == ["tool_start", "task_launched", "stream", "final"]
    assert all(isinstance(ev, OrchestratorEvent) for ev in events)
    # Sanity-check controlled vocabulary subset documented in the module
    # docstring.
    allowed = {
        "tool_start",
        "tool_end",
        "task_launched",
        "task_completed",
        "task_failed",
        "stream",
        "error",
        "final",
    }
    assert set(types).issubset(allowed)
