# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ErrorEscalationAspect -- cumulative error tracking and escalation policy.

This is the ONLY place that decides whether an error aborts the pipeline
or is absorbed.  The runner is dumb plumbing -- it just calls aspects.

Policy:
- CRITICAL blocks: any error → ABORT (pipeline stops).
- BEST_EFFORT blocks: errors are absorbed with CONTINUE routing.
- Cumulative failures: if N consecutive blocks fail (across any
  criticality), escalate to ABORT as a circuit-breaker.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockCriticality, BlockResult,
    PipelineBlock, RoutingHint,
)

logger = logging.getLogger(__name__)

# After this many consecutive block failures, abort regardless of criticality.
_MAX_CONSECUTIVE_FAILURES = int(
    os.environ.get("MAX_CONSECUTIVE_BLOCK_FAILURES", "3"),
)


class ErrorEscalationAspect(Aspect):
    """Decide whether block errors abort or are absorbed.

    This aspect tracks cumulative failures and implements the escalation
    policy.  It should be the LAST aspect in the list so it sees the
    final error state after all other aspects have had a chance to
    handle the error.
    """

    name = "error_escalation"

    def __init__(self) -> None:
        self._consecutive_failures: int = 0
        self._total_errors: int = 0

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        # Reset consecutive failures on successful execution.
        # Only check "block_failed" (set by runner/on_error on real exceptions).
        # Don't check "error" — blocks like SearchExecutorBlock catch their own
        # exceptions and set metrics["error"] as a soft failure; the consecutive
        # failure counter should still reset for those.
        if not result.metrics.get("block_failed"):
            self._consecutive_failures = 0

    async def on_error(
        self, block: PipelineBlock, ctx: BlockContext, error: Exception,
    ) -> Optional[BlockResult]:
        self._consecutive_failures += 1
        self._total_errors += 1

        logger.error(
            "ErrorEscalation: block '%s' failed (criticality=%s, "
            "consecutive=%d, total=%d): %s",
            block.name, block.criticality.value,
            self._consecutive_failures, self._total_errors,
            error,
        )

        # CRITICAL block failure → ABORT
        if block.criticality == BlockCriticality.CRITICAL:
            logger.error(
                "CRITICAL block '%s' failed — aborting pipeline", block.name,
            )
            return BlockResult(
                metrics={
                    "error": str(error),
                    "block_failed": True,
                    "escalation": "critical_abort",
                },
                routing=RoutingHint.ABORT,
                diagnosis=(
                    f"CRITICAL block '{block.name}' failed: {error}"
                ),
            )

        # Circuit breaker: too many consecutive failures
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            logger.error(
                "Circuit breaker: %d consecutive failures — aborting",
                self._consecutive_failures,
            )
            return BlockResult(
                metrics={
                    "error": str(error),
                    "block_failed": True,
                    "escalation": "circuit_breaker",
                    "consecutive_failures": self._consecutive_failures,
                },
                routing=RoutingHint.ABORT,
                diagnosis=(
                    f"Circuit breaker tripped after "
                    f"{self._consecutive_failures} consecutive failures"
                ),
            )

        # BEST_EFFORT: absorb the error, continue pipeline
        return BlockResult(
            metrics={
                "error": str(error),
                "block_failed": True,
                "escalation": "absorbed",
            },
            routing=RoutingHint.CONTINUE,
            diagnosis=(
                f"BEST_EFFORT block '{block.name}' failed (absorbed): {error}"
            ),
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def total_errors(self) -> int:
        return self._total_errors
