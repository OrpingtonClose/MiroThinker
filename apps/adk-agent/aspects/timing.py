# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""TimingAspect -- measure phase execution time.

Records wall-clock time for each block's execution and injects the
duration into the result's metrics dict.  Also emits a dashboard
event so the frontend can show per-phase timing.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockResult, PipelineBlock,
)

logger = logging.getLogger(__name__)


class TimingAspect(Aspect):
    """Measure wall-clock execution time for every block."""

    name = "timing"

    async def before(
        self, block: PipelineBlock, ctx: BlockContext,
    ) -> Optional[BlockResult]:
        ctx._phase_start_time = time.monotonic()
        return None

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        if ctx._phase_start_time > 0:
            elapsed = time.monotonic() - ctx._phase_start_time
            result.metrics["duration_s"] = round(elapsed, 2)
            logger.info(
                "Block '%s' completed in %.2fs", block.name, elapsed,
            )
            # Emit dashboard event
            if ctx.collector is not None:
                try:
                    ctx.collector.emit_event("phase_timing", data={
                        "phase": block.name,
                        "duration_s": round(elapsed, 2),
                        "iteration": ctx.iteration,
                    })
                except Exception:
                    pass
