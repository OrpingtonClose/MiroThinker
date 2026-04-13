# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""BlockMaintainer -- optional per-block agentic LLM monitor.

When a block fails or produces degraded output, the maintainer can
diagnose the issue and suggest corrective action.  This is an
*optional* capability -- blocks work fine without it.

The maintainer is invoked by the ErrorEscalationAspect when a
BEST_EFFORT block fails, giving it a chance to provide a richer
diagnosis before the error is absorbed.

This is a lightweight implementation -- it logs diagnostic context
and emits dashboard events.  A future version could invoke an LLM
to generate repair suggestions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from models.pipeline_block import BlockContext, BlockResult, PipelineBlock

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Structured diagnostic from the maintainer."""
    block_name: str = ""
    error_type: str = ""
    error_message: str = ""
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    severity: str = "INFO"  # INFO / WARNING / ERROR


class BlockMaintainer:
    """Per-block diagnostic monitor.

    Collects failure context and produces structured diagnostics.
    Can be extended with LLM-powered diagnosis in the future.
    """

    def __init__(self) -> None:
        self._history: list[DiagnosticReport] = []

    def diagnose(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        error: Exception,
        result: Optional[BlockResult] = None,
    ) -> DiagnosticReport:
        """Produce a diagnostic report for a block failure.

        Args:
            block: The failed block.
            ctx: The block context at time of failure.
            error: The exception that was raised.
            result: Optional BlockResult if the error was caught.

        Returns:
            A DiagnosticReport with context and suggestions.
        """
        report = DiagnosticReport(
            block_name=block.name,
            error_type=type(error).__name__,
            error_message=str(error)[:1000],
            context_snapshot={
                "iteration": ctx.iteration,
                "user_query": ctx.user_query[:200] if ctx.user_query else "",
                "has_corpus": ctx.corpus is not None,
                "state_keys": list(ctx.state.keys())[:20],
            },
        )

        # Simple heuristic suggestions based on error type
        if "DuckDB" in str(error) or "database" in str(error).lower():
            report.suggestion = (
                "DuckDB error — check for concurrent access or "
                "corrupted corpus file."
            )
            report.severity = "ERROR"
        elif "timeout" in str(error).lower():
            report.suggestion = (
                "Timeout — consider increasing timeout or reducing "
                "search concurrency."
            )
            report.severity = "WARNING"
        elif "api" in str(error).lower() or "key" in str(error).lower():
            report.suggestion = (
                "API error — check API keys and rate limits."
            )
            report.severity = "WARNING"
        else:
            report.suggestion = f"Unexpected {report.error_type} in {block.name}."
            report.severity = "WARNING"

        self._history.append(report)
        logger.info(
            "BlockMaintainer diagnosis for '%s': %s — %s",
            block.name, report.severity, report.suggestion,
        )

        # Emit to dashboard
        if ctx.collector is not None:
            try:
                ctx.collector.emit_event("block_diagnostic", data={
                    "block": block.name,
                    "error_type": report.error_type,
                    "suggestion": report.suggestion,
                    "severity": report.severity,
                    "iteration": ctx.iteration,
                })
            except Exception:
                pass

        return report

    @property
    def history(self) -> list[DiagnosticReport]:
        return list(self._history)

    def summary(self) -> dict[str, Any]:
        """Return a serialisable summary of all diagnostics."""
        return {
            "total_diagnostics": len(self._history),
            "by_block": {},
            "recent": [
                {
                    "block": r.block_name,
                    "error": r.error_type,
                    "suggestion": r.suggestion,
                    "severity": r.severity,
                }
                for r in self._history[-5:]
            ],
        }
