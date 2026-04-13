# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""CostTrackingAspect -- track cumulative API cost across blocks.

Snapshots the cost tracker before execution and records the delta
after execution.  Injects cost metrics into the block result.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockResult, PipelineBlock,
)

logger = logging.getLogger(__name__)


def _get_session_cost() -> float:
    """Read cumulative cost from the cost tracker (best-effort)."""
    try:
        from tools.cost_tracker import get_session_cost
        return get_session_cost()
    except Exception:
        return 0.0


class CostTrackingAspect(Aspect):
    """Track API costs per block execution."""

    name = "cost_tracking"

    async def before(
        self, block: PipelineBlock, ctx: BlockContext,
    ) -> Optional[BlockResult]:
        ctx._cost_snapshot = _get_session_cost()
        return None

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        current = _get_session_cost()
        delta = current - ctx._cost_snapshot
        if delta > 0:
            result.metrics["cost_delta"] = round(delta, 4)
            result.metrics["cost_cumulative"] = round(current, 4)
            logger.info(
                "Block '%s' cost: $%.4f (cumulative $%.4f)",
                block.name, delta, current,
            )

        # Update state with cumulative cost
        ctx.state["_cumulative_api_cost"] = current

        # Emit cost event to dashboard
        if ctx.collector is not None and delta > 0:
            try:
                ctx.collector.emit_event("block_cost", data={
                    "phase": block.name,
                    "cost_delta": round(delta, 4),
                    "cost_cumulative": round(current, 4),
                    "iteration": ctx.iteration,
                })
            except Exception:
                pass
