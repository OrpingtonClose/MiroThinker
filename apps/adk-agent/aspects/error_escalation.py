# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ErrorEscalationAspect -- typed error handling and escalation policy.

This is the ONLY place that decides whether an error aborts the pipeline
or is absorbed.  The runner is dumb plumbing -- it just calls aspects.

Phase 3: Now uses the typed error taxonomy (PipelineCritical, PipelineDegraded,
PipelineTransient, PipelineIgnorable) for nuanced error handling instead of
treating all exceptions identically.

Policy:
- PipelineCritical → ABORT (always).
- PipelineDegraded → CONTINUE but track in state for quality manifest.
- PipelineTransient → CONTINUE, count occurrences, escalate if chronic.
- PipelineIgnorable → CONTINUE silently (DEBUG log only).
- Untyped exceptions → classify via ``classify_error()`` then apply policy.
- Circuit breaker: N consecutive failures → ABORT regardless.
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

# After this many transient errors from the same source, escalate to degraded.
_TRANSIENT_CHRONIC_THRESHOLD = 3


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
        self._degradations: list[dict] = []
        self._transient_counts: dict[str, int] = {}  # source → count

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        # Reset consecutive failures on successful execution.
        if not result.metrics.get("block_failed"):
            self._consecutive_failures = 0

        # Persist degradations in state for quality manifest
        if self._degradations:
            existing = ctx.state.get("_pipeline_degradations", [])
            ctx.state["_pipeline_degradations"] = existing + self._degradations
            self._degradations = []

    async def on_error(
        self, block: PipelineBlock, ctx: BlockContext, error: Exception,
    ) -> Optional[BlockResult]:
        from models.errors import (
            PipelineCritical, PipelineDegraded, PipelineIgnorable,
            PipelineTransient, classify_error,
        )

        self._consecutive_failures += 1
        self._total_errors += 1

        # Classify the error if it isn't already typed
        typed_error = error
        if not isinstance(error, (PipelineCritical, PipelineDegraded,
                                   PipelineTransient, PipelineIgnorable)):
            typed_error = classify_error(error, source=block.name)

        # ── PipelineIgnorable: DEBUG log, absorb silently ──
        if isinstance(typed_error, PipelineIgnorable):
            logger.debug(
                "ErrorEscalation: ignorable error in '%s': %s",
                block.name, typed_error,
            )
            self._consecutive_failures -= 1  # don't count ignorable
            self._total_errors -= 1
            return BlockResult(
                # block_failed=True prevents after() from resetting
                # _consecutive_failures to 0, preserving any existing
                # chain of real failures.
                metrics={"error": str(typed_error), "escalation": "ignored", "block_failed": True},
                routing=RoutingHint.CONTINUE,
            )

        # ── PipelineTransient: count, escalate if chronic ──
        if isinstance(typed_error, PipelineTransient):
            source = getattr(typed_error, "source", block.name) or block.name
            self._transient_counts[source] = self._transient_counts.get(source, 0) + 1
            count = self._transient_counts[source]

            if count >= _TRANSIENT_CHRONIC_THRESHOLD:
                logger.warning(
                    "ErrorEscalation: transient error from '%s' became chronic "
                    "(%d occurrences) — escalating to degraded",
                    source, count,
                )
                self._degradations.append({
                    "source": source,
                    "error": str(typed_error),
                    "category": "chronic_transient",
                    "count": count,
                })
            else:
                logger.warning(
                    "ErrorEscalation: transient error in '%s' (%d/%d): %s",
                    block.name, count, _TRANSIENT_CHRONIC_THRESHOLD, typed_error,
                )

            return BlockResult(
                metrics={
                    "error": str(typed_error),
                    "block_failed": True,
                    "escalation": "transient_absorbed",
                    "transient_count": count,
                },
                routing=RoutingHint.CONTINUE,
            )

        # ── PipelineCritical: always ABORT ──
        if isinstance(typed_error, PipelineCritical):
            logger.error(
                "ErrorEscalation: CRITICAL error in '%s' — aborting pipeline: %s",
                block.name, typed_error,
            )
            return BlockResult(
                metrics={
                    "error": str(typed_error),
                    "block_failed": True,
                    "escalation": "critical_abort",
                },
                routing=RoutingHint.ABORT,
                diagnosis=f"CRITICAL: {typed_error}",
            )

        # ── PipelineDegraded: absorb but track ──
        # Policy: CONTINUE but track in state for quality manifest.
        # Does NOT defer to block criticality — degraded errors are
        # always absorbed.  Only PipelineCritical triggers ABORT.
        if isinstance(typed_error, PipelineDegraded):
            logger.warning(
                "ErrorEscalation: degraded error in '%s': %s",
                block.name, typed_error,
            )
            self._degradations.append({
                "source": getattr(typed_error, "source", block.name) or block.name,
                "error": str(typed_error),
                "category": "degraded",
            })
            return BlockResult(
                metrics={
                    "error": str(typed_error),
                    "block_failed": True,
                    "escalation": "degraded_absorbed",
                },
                routing=RoutingHint.CONTINUE,
            )

        # ── CRITICAL block (by criticality) → ABORT ──
        if block.criticality == BlockCriticality.CRITICAL:
            logger.error(
                "CRITICAL block '%s' failed — aborting pipeline", block.name,
            )
            return BlockResult(
                metrics={
                    "error": str(typed_error),
                    "block_failed": True,
                    "escalation": "critical_abort",
                },
                routing=RoutingHint.ABORT,
                diagnosis=f"CRITICAL block '{block.name}' failed: {typed_error}",
            )

        # ── Circuit breaker: too many consecutive failures ──
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            logger.error(
                "Circuit breaker: %d consecutive failures — aborting",
                self._consecutive_failures,
            )
            return BlockResult(
                metrics={
                    "error": str(typed_error),
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

        # ── BEST_EFFORT: absorb the error, continue pipeline ──
        return BlockResult(
            metrics={
                "error": str(typed_error),
                "block_failed": True,
                "escalation": "absorbed",
            },
            routing=RoutingHint.CONTINUE,
            diagnosis=f"BEST_EFFORT block '{block.name}' failed (absorbed): {typed_error}",
        )

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def total_errors(self) -> int:
        return self._total_errors

    @property
    def degradation_count(self) -> int:
        return len(self._degradations)
