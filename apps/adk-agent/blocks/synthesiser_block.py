# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""SynthesiserBlock -- final report generation phase.

The synthesiser reads the swarm-synthesised corpus and produces the
polished final report.  This block handles post-synthesiser metrics
collection.  The QualityManifest (Phase 4) is appended in
SwarmSynthesisBlock BEFORE the synthesiser runs, so it appears in
the final report.
"""

from __future__ import annotations

import logging

from models.pipeline_block import (
    BlockContext, BlockCriticality, BlockResult, ParamSpec,
    PipelineBlock,
)

logger = logging.getLogger(__name__)


class SynthesiserBlock(PipelineBlock):
    """Final phase: polish swarm output into the definitive report."""

    name = "synthesiser"
    needs_corpus = True
    criticality = BlockCriticality.CRITICAL
    is_looped = False

    input_specs = [
        ParamSpec(
            key="corpus_for_synthesis",
            expected_type=str,
            validator=lambda v: bool(v and len(v.strip()) > 50),
            description="Swarm-synthesised corpus (>50 chars)",
        ),
    ]
    output_specs = []  # Output goes to conversation history via ADK

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Post-synthesiser processing: measure what the synthesiser had.

        NOTE: The actual LLM call is handled by ADK's Agent.  This block
        runs as the after_agent_callback wrapper — it collects metrics
        on the synthesiser's input quality.
        """
        state = ctx.state

        # Measure what the synthesiser had to work with
        swarm_input = state.get("corpus_for_synthesis", "")
        corpus_findings = 0

        if ctx.corpus is not None:
            try:
                corpus_findings = ctx.corpus.conn.execute(
                    "SELECT COUNT(*) FROM conditions WHERE row_type = 'finding'"
                ).fetchone()[0]
            except Exception:
                pass

        metrics = {
            "swarm_input_length": len(swarm_input) if swarm_input else 0,
            "corpus_findings": corpus_findings,
        }

        return BlockResult(metrics=metrics)
