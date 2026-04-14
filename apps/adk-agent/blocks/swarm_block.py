# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""SwarmSynthesisBlock -- Flock gossip swarm synthesis.

Runs the 3-phase gossip protocol (per-angle workers → peer refinement
→ queen merge) on the full corpus before the final synthesiser.
Replaces the raw atomic conditions with a pre-synthesised narrative.
"""

from __future__ import annotations

import logging

from models.pipeline_block import (
    BlockContext, BlockCriticality, BlockResult, ParamSpec,
    PipelineBlock,
)

logger = logging.getLogger(__name__)


class SwarmSynthesisBlock(PipelineBlock):
    """Pre-synthesiser swarm: gossip protocol over the corpus."""

    name = "swarm_synthesis"
    needs_corpus = True
    criticality = BlockCriticality.BEST_EFFORT
    is_looped = False

    input_specs = [
        ParamSpec(
            key="corpus_for_synthesis",
            expected_type=str,
            required=False,
            description="Raw corpus formatted for synthesiser",
        ),
    ]
    output_specs = [
        ParamSpec(
            key="corpus_for_synthesis",
            expected_type=str,
            required=False,
            description="Swarm-synthesised narrative for final synthesiser",
        ),
    ]

    async def execute(self, ctx: BlockContext) -> BlockResult:
        from callbacks.condition_manager import run_swarm_synthesis

        state = ctx.state
        swarm_report = run_swarm_synthesis(state)

        if swarm_report and swarm_report.strip():
            logger.info(
                "Swarm synthesis produced %d chars", len(swarm_report),
            )
            return BlockResult(
                metrics={
                    "swarm_output_length": len(swarm_report),
                    "swarm_produced": True,
                },
                state_updates={
                    "corpus_for_synthesis": swarm_report,
                },
            )

        logger.warning(
            "Swarm synthesis returned empty — final synthesiser "
            "will read the raw corpus format instead"
        )
        return BlockResult(
            metrics={
                "swarm_output_length": 0,
                "swarm_produced": False,
            },
        )
