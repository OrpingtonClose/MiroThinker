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

            # Only compute corpus-derived values for keys the block
            # didn't already set.  We must NOT use setdefault() here
            # because Python always evaluates the default argument —
            # that would hit DuckDB even when the key exists, which is
            # unsafe after MaestroBlock starts its background scoring
            # thread (DuckDB connections are not thread-safe).
            #
            # IMPORTANT: Use the CHEAP format_for_thinker() here, NOT
            # synthesise().  The expensive multi-LLM gossip synthesis
            # only runs once in SwarmSynthesisBlock before the final
            # synthesiser.  Running it after every search/maestro
            # iteration would add minutes of latency per loop.
            if "research_findings" not in result.state_updates:
                result.state_updates["research_findings"] = (
                    ctx.corpus.format_for_thinker(current_iteration=iteration)
                )

            if "corpus_for_synthesis" not in result.state_updates:
                result.state_updates["corpus_for_synthesis"] = (
                    ctx.corpus.format_for_synthesiser()
                )

            if "corpus_summary_for_maestro" not in result.state_updates:
                result.state_updates["corpus_summary_for_maestro"] = (
                    ctx.corpus.format_summary_for_maestro()
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

            briefing_len = len(result.state_updates.get("research_findings", ""))
            logger.debug(
                "Corpus refresh after '%s': briefing=%d chars",
                block.name, briefing_len,
            )
        except Exception as exc:
            logger.warning(
                "Corpus refresh failed after '%s': %s",
                block.name, exc,
            )
