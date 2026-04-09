# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Dashboard package — real-time pipeline run instrumentation.

**Dual-storage design** for the active collector:

1. A :class:`contextvars.ContextVar` ensures each async request (task)
   in the AG-UI server gets its own collector — ADK callbacks that call
   ``get_active_collector()`` always see the correct one even when
   concurrent POST requests overlap.

2. A shared ``dict`` (``_active_collectors``) keyed by ``session_id``
   lets the dashboard SSE stream and REST endpoints observe *all*
   in-progress collectors from any request context.  The dashboard
   calls ``get_all_active_collectors()`` or ``get_any_active_collector()``
   to read from the shared registry.

The CLI path (``main.py``) works identically — both stores are
populated by ``set_active_collector()``.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dashboard.collector import PipelineCollector

# Per-request isolation for ADK callbacks
_active_collector: ContextVar[PipelineCollector | None] = ContextVar(
    "_active_collector", default=None
)

# Shared registry for cross-request dashboard observability
_active_collectors: dict[str, PipelineCollector] = {}
_registry_lock = threading.Lock()


def set_active_collector(collector: PipelineCollector | None) -> None:
    """Set (or clear) the active pipeline collector.

    Updates both the per-request ``ContextVar`` (for ADK callbacks) and
    the shared registry (for dashboard SSE/REST endpoints).

    When *clearing* (``collector is None``), the previous collector for
    this context is automatically removed from the shared registry so
    callers like ``main.py`` don't need to call ``unregister_collector``
    separately.
    """
    old = _active_collector.get()
    _active_collector.set(collector)
    with _registry_lock:
        if collector is not None:
            _active_collectors[collector.session_id] = collector
        elif old is not None:
            _active_collectors.pop(old.session_id, None)


def unregister_collector(session_id: str) -> None:
    """Remove a collector from the shared registry by session_id."""
    with _registry_lock:
        _active_collectors.pop(session_id, None)


def get_active_collector() -> PipelineCollector | None:
    """Return the active pipeline collector for this async context.

    Used by ADK callbacks — returns the collector for the current
    request only.
    """
    return _active_collector.get()


def get_any_active_collector() -> PipelineCollector | None:
    """Return any active collector from the shared registry.

    Used by dashboard endpoints that need cross-request visibility
    (e.g. ``/dashboard/latest``, ``/dashboard/stream``).  Returns
    the most recently registered collector, or ``None``.
    """
    with _registry_lock:
        if _active_collectors:
            # Return the last one added (most recent run)
            return list(_active_collectors.values())[-1]
    return None


def get_all_active_collectors() -> list[PipelineCollector]:
    """Return all active collectors (for multi-run dashboard views)."""
    with _registry_lock:
        return list(_active_collectors.values())
