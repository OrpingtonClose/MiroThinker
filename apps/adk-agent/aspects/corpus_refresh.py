# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""CorpusRefreshAspect -- refresh corpus-derived state after block execution.

After blocks that modify the corpus (search_executor, maestro), this
aspect updates session state with the latest corpus formatting so
downstream blocks see fresh data.

The block declares ``needs_corpus``.  This aspect handles the
cross-cutting consequence of corpus mutation: state refresh.
"""

from __future__ import annotations

import logging

from models.pipeline_block import (
    Aspect, BlockContext, BlockResult, PipelineBlock,
)

logger = logging.getLogger(__name__)

# Blocks whose execution typically mutates the corpus and require
# a state refresh afterward.
_CORPUS_MUTATORS = {"search_executor", "maestro", "swarm_synthesis"}


class CorpusRefreshAspect(Aspect):
    """Refresh corpus-derived state keys after corpus-mutating blocks."""

    name = "corpus_refresh"

    async def after(
        self, block: PipelineBlock, ctx: BlockContext, result: BlockResult,
    ) -> None:
        if block.name not in _CORPUS_MUTATORS:
            return
        if ctx.corpus is None:
            return

        try:
            iteration = ctx.iteration
            user_query = ctx.user_query

            # Refresh thinker briefing
            briefing = ""
            try:
                briefing = ctx.corpus.synthesise(user_query) if user_query else ""
            except Exception:
                logger.debug("Swarm briefing failed, falling back to format_for_thinker")
            if not briefing:
                briefing = ctx.corpus.format_for_thinker(current_iteration=iteration)

            # Only fill keys the block didn't already set — blocks like
            # MaestroBlock (devil's advocate injection) and SwarmSynthesisBlock
            # (swarm narrative) produce their own values that we must not overwrite.
            result.state_updates.setdefault("research_findings", briefing)
            result.state_updates.setdefault(
                "corpus_for_synthesis",
                ctx.corpus.format_for_synthesiser(),
            )

            # Inject expansion targets (only if block didn't set them)
            if "_expansion_targets" not in result.state_updates:
                expansion_targets = ctx.corpus.get_expansion_targets()
                if expansion_targets:
                    lines = ["=== ENRICHMENT TASKS (from corpus analysis) ==="]
                    for t in expansion_targets[:10]:
                        lines.append(
                            f"- Finding [{t['id']}] needs enrichment via "
                            f"{t['strategy']}: {t['hint']}"
                        )
                    lines.append("=== END ENRICHMENT TASKS ===")
                    result.state_updates["_expansion_targets"] = "\n".join(lines)
                else:
                    result.state_updates["_expansion_targets"] = ""

            logger.debug(
                "Corpus refresh after '%s': briefing=%d chars",
                block.name, len(briefing),
            )
        except Exception as exc:
            logger.warning(
                "Corpus refresh failed after '%s': %s",
                block.name, exc,
            )
