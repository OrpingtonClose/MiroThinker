# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Module-level collector registry.

ADK's InMemorySessionService does not persist arbitrary Python objects
across get_session() calls, so we store the active DashboardCollector
here at module level where all callbacks can access it.
"""

from __future__ import annotations

from typing import Optional

from dashboard.collector import DashboardCollector

_active_collector: Optional[DashboardCollector] = None


def set_collector(collector: DashboardCollector) -> None:
    """Set the active collector for the current run."""
    global _active_collector
    _active_collector = collector


def get_collector() -> Optional[DashboardCollector]:
    """Get the active collector, or None if not set."""
    return _active_collector


def clear_collector() -> None:
    """Clear the active collector."""
    global _active_collector
    _active_collector = None
