# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Dashboard package — real-time pipeline run instrumentation.

The active collector is stored in a :class:`contextvars.ContextVar` so
that each async request (task) in the AG-UI server gets its own
collector without interference from concurrent requests.  The CLI path
(``main.py``) works identically — ``ContextVar`` falls back to a
single implicit context when there is no event-loop task switching.

ADK callbacks call ``get_active_collector()`` and interact with it;
zero overhead when no pipeline run is active (returns ``None``).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dashboard.collector import PipelineCollector

_active_collector: ContextVar[PipelineCollector | None] = ContextVar(
    "_active_collector", default=None
)


def set_active_collector(collector: PipelineCollector | None) -> None:
    """Set (or clear) the active pipeline collector for this async context."""
    _active_collector.set(collector)


def get_active_collector() -> PipelineCollector | None:
    """Return the active pipeline collector for this async context, or None."""
    return _active_collector.get()
