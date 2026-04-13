# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ScoutBlock -- Phase 0 landscape probing.

Wraps ``run_scout_phase()`` as a fenced pipeline block.  The scout
decomposes the query, probes cheap APIs, and classifies sub-questions
into SHALLOW / MODERATE / DEEP tiers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading

from models.pipeline_block import (
    BlockContext, BlockCriticality, BlockResult, ParamSpec,
    PipelineBlock,
)

logger = logging.getLogger(__name__)


class ScoutBlock(PipelineBlock):
    """Phase 0: landscape assessment via cheap search probes."""

    name = "scout"
    needs_corpus = False
    criticality = BlockCriticality.BEST_EFFORT
    is_looped = False

    input_specs = [
        ParamSpec(
            key="user_query",
            expected_type=str,
            validator=lambda v: bool(v and v.strip()),
            description="Non-empty user research query",
        ),
    ]
    output_specs = [
        ParamSpec(
            key="research_findings",
            expected_type=str,
            required=False,
            description="Landscape assessment text injected by scout",
        ),
    ]

    async def execute(self, ctx: BlockContext) -> BlockResult:
        from tools.scout import run_scout_phase

        query = ctx.user_query or ctx.state.get("user_query", "")
        if not query:
            return BlockResult(
                metrics={"skipped": True, "reason": "no_query"},
                diagnosis="Scout skipped: no user query",
            )

        cancel_event = ctx.cancel or threading.Event()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    future = pool.submit(
                        asyncio.run,
                        run_scout_phase(query, ctx.state, _cancel=cancel_event),
                    )
                    future.result(timeout=90)
                finally:
                    cancel_event.set()
                    pool.shutdown(wait=False, cancel_futures=True)
            else:
                await run_scout_phase(query, ctx.state, _cancel=cancel_event)
        except Exception as exc:
            cancel_event.set()
            logger.warning("Scout phase failed: %s", exc)
            return BlockResult(
                metrics={"error": str(exc)},
                diagnosis=f"Scout failed (non-fatal): {exc}",
            )

        findings = ctx.state.get("research_findings", "")
        has_landscape = bool(findings) and findings != "(no findings yet)"

        return BlockResult(
            metrics={
                "has_landscape": has_landscape,
                "landscape_length": len(findings) if has_landscape else 0,
            },
            state_updates={
                "research_findings": findings,
            },
        )
