# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""HeartbeatAspect -- emit phase_start / phase_end dashboard events.

Provides live observability: the frontend sees which phase is active
and when it completes (or fails).  Zero business logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockResult, PipelineBlock,
)

logger = logging.getLogger(__name__)


class HeartbeatAspect(Aspect):
    """Emit dashboard heartbeat events at phase boundaries."""

    name = "heartbeat"

    async def before(
        self, block: PipelineBlock, ctx: BlockContext,
    ) -> Optional[BlockResult]:
        if ctx.collector is not None:
            try:
                ctx.collector.emit_event("phase_start", data={
                    "phase": block.name,
                    "iteration": ctx.iteration,
                })
            except Exception:
                pass
        logger.info(
            "▶ Block '%s' starting (iteration=%d)",
            block.name, ctx.iteration,
        )
        return None

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        if ctx.collector is not None:
            try:
                ctx.collector.emit_event("phase_end", data={
                    "phase": block.name,
                    "iteration": ctx.iteration,
                    "routing": result.routing.value,
                    "has_error": bool(result.metrics.get("error")),
                })
            except Exception:
                pass
        logger.info(
            "■ Block '%s' finished (routing=%s)",
            block.name, result.routing.value,
        )

    async def on_error(
        self, block: PipelineBlock, ctx: BlockContext, error: Exception,
    ) -> Optional[BlockResult]:
        if ctx.collector is not None:
            try:
                ctx.collector.emit_event("phase_error", data={
                    "phase": block.name,
                    "iteration": ctx.iteration,
                    "error": str(error)[:500],
                })
            except Exception:
                pass
        return None
