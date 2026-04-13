# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Aspect-oriented cross-cutting concerns for the pipeline block framework.

Each aspect implements the ``Aspect`` interface (before / after / on_error)
and is applied uniformly to every ``PipelineBlock`` by the ``PipelineRunner``.
"""

from aspects.timing import TimingAspect
from aspects.heartbeat import HeartbeatAspect
from aspects.io_validation import InputOutputValidationAspect
from aspects.duckdb_safety import DuckDBSafetyAspect
from aspects.health_gate import HealthGateAspect
from aspects.cost_tracking import CostTrackingAspect
from aspects.corpus_refresh import CorpusRefreshAspect
from aspects.error_escalation import ErrorEscalationAspect

__all__ = [
    "TimingAspect",
    "HeartbeatAspect",
    "InputOutputValidationAspect",
    "DuckDBSafetyAspect",
    "HealthGateAspect",
    "CostTrackingAspect",
    "CorpusRefreshAspect",
    "ErrorEscalationAspect",
]
