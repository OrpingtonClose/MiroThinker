# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""HealthGateAspect -- evaluate block health and track cumulative state.

Replaces the old per-phase ``PipelineHealth`` checks with a unified
aspect that:

1. Tracks per-block health metrics across iterations.
2. Evaluates simple gate conditions after each block.
3. Stores a health summary in session state for downstream consumers.

The gate conditions are intentionally simple -- they flag degraded
states without imposing hard aborts (that's ErrorEscalationAspect's job).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockResult, PipelineBlock,
)

logger = logging.getLogger(__name__)


@dataclass
class _PhaseHealth:
    """Accumulated health for a single block across iterations."""
    total_runs: int = 0
    failures: int = 0
    last_metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class HealthGateAspect(Aspect):
    """Track cumulative block health and evaluate simple gates."""

    name = "health_gate"

    def __init__(self) -> None:
        self._health: dict[str, _PhaseHealth] = {}

    def _get(self, name: str) -> _PhaseHealth:
        if name not in self._health:
            self._health[name] = _PhaseHealth()
        return self._health[name]

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        ph = self._get(block.name)
        ph.total_runs += 1
        ph.last_metrics = dict(result.metrics)

        if result.metrics.get("error") or result.metrics.get("block_failed"):
            ph.failures += 1

        # Simple gate: warn if failure rate > 50% after 2+ runs
        if ph.total_runs >= 2 and ph.failures / ph.total_runs > 0.5:
            msg = (
                f"Block '{block.name}' has >{50}% failure rate "
                f"({ph.failures}/{ph.total_runs})"
            )
            if msg not in ph.warnings:
                ph.warnings.append(msg)
            logger.warning(msg)

        # Persist health summary in state for downstream consumers
        ctx.state["_block_health"] = self.summary()

    async def on_error(
        self, block: PipelineBlock, ctx: BlockContext, error: Exception,
    ) -> Optional[BlockResult]:
        # Don't increment here — after() is always called by the runner
        # (even on error paths) and will handle the counting.
        return None

    def summary(self) -> dict[str, Any]:
        """Return a serialisable health summary."""
        return {
            name: {
                "total_runs": ph.total_runs,
                "failures": ph.failures,
                "failure_rate": (
                    round(ph.failures / ph.total_runs, 2)
                    if ph.total_runs > 0 else 0.0
                ),
                "warnings": ph.warnings[-5:],
            }
            for name, ph in self._health.items()
        }
