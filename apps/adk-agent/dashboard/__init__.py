# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Dashboard package — real-time pipeline run instrumentation.

The ``_active_collector`` module-level reference is the rendezvous point
between ADK callbacks (whose signatures are fixed by the framework) and
the :class:`PipelineCollector`.  ``run_pipeline()`` sets it before the
run starts and clears it when the run ends.  Callbacks check
``if _active_collector:`` and call methods on it — zero overhead when
no pipeline run is active.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dashboard.collector import PipelineCollector

_active_collector: PipelineCollector | None = None
_collector_lock = threading.Lock()


def set_active_collector(collector: PipelineCollector | None) -> None:
    """Set (or clear) the active pipeline collector."""
    global _active_collector
    with _collector_lock:
        _active_collector = collector


def get_active_collector() -> PipelineCollector | None:
    """Return the active pipeline collector, or None."""
    return _active_collector
