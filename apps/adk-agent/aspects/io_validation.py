# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""InputOutputValidationAspect -- validate block I/O against ParamSpecs.

Blocks declare their validation RULES via ``input_specs`` / ``output_specs``.
This aspect owns the CONSEQUENCES: what happens when rules fail.

For CRITICAL blocks, input validation failure short-circuits execution.
For BEST_EFFORT blocks, failures are logged and execution proceeds.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockCriticality, BlockResult,
    PipelineBlock, RoutingHint,
)

logger = logging.getLogger(__name__)


class InputOutputValidationAspect(Aspect):
    """Validate inputs before and outputs after block execution."""

    name = "io_validation"

    async def before(
        self, block: PipelineBlock, ctx: BlockContext,
    ) -> Optional[BlockResult]:
        """Validate input specs.  Short-circuit CRITICAL blocks on failure."""
        errors: list[str] = []
        for spec in block.input_specs:
            value = ctx.state.get(spec.key, spec.default if not spec.required else None)
            ok, msg = spec.validate(value)
            if not ok:
                errors.append(msg)

        if not errors:
            return None

        error_summary = "; ".join(errors)
        logger.warning(
            "Input validation failed for '%s': %s",
            block.name, error_summary,
        )

        if block.criticality == BlockCriticality.CRITICAL:
            return BlockResult(
                metrics={"validation_errors": errors, "short_circuited": True},
                routing=RoutingHint.ABORT,
                diagnosis=f"Input validation failed (CRITICAL): {error_summary}",
            )

        # BEST_EFFORT: log but proceed
        return None

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        """Validate output specs against state_updates."""
        errors: list[str] = []
        for spec in block.output_specs:
            value = result.state_updates.get(spec.key)
            if value is None:
                value = ctx.state.get(spec.key)
            ok, msg = spec.validate(value)
            if not ok:
                errors.append(msg)

        if errors:
            result.metrics["output_validation_errors"] = errors
            logger.warning(
                "Output validation warnings for '%s': %s",
                block.name, "; ".join(errors),
            )
