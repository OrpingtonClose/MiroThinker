# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""DuckDBSafetyAspect -- ensure DuckDB thread-safety invariant.

Blocks that declare ``needs_corpus = True`` require exclusive DuckDB
access.  This aspect checks for pending scoring threads before
execution and injects the corpus into the context.

The block declares ``needs_corpus``.  This aspect enforces the
thread-safety consequence: wait/skip if DuckDB is contended.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.pipeline_block import (
    Aspect, BlockContext, BlockResult, PipelineBlock, RoutingHint,
)

logger = logging.getLogger(__name__)


class DuckDBSafetyAspect(Aspect):
    """Ensure DuckDB thread-safety for corpus-dependent blocks."""

    name = "duckdb_safety"

    async def before(
        self, block: PipelineBlock, ctx: BlockContext,
    ) -> Optional[BlockResult]:
        if not block.needs_corpus:
            return None

        # Check for DuckDB contention via the per-corpus async lock
        try:
            from callbacks.condition_manager import _get_corpus_lock
            corpus_key = ctx.state.get("_corpus_key", "default")
            lock = _get_corpus_lock(corpus_key)
            if lock.locked():
                logger.warning(
                    "DuckDB safety: async lock held for '%s' "
                    "— skipping block '%s' to avoid concurrent access",
                    corpus_key, block.name,
                )
                return BlockResult(
                    metrics={
                        "skipped": True,
                        "reason": "duckdb_contention",
                    },
                    routing=RoutingHint.CONTINUE,
                    diagnosis=(
                        f"Block '{block.name}' skipped: DuckDB async "
                        f"lock held (key={corpus_key})"
                    ),
                )
        except ImportError:
            pass

        # Inject corpus into context if not already present
        if ctx.corpus is None:
            try:
                from callbacks.condition_manager import _get_corpus
                ctx.corpus = _get_corpus(ctx.state)
            except Exception as exc:
                logger.warning(
                    "DuckDB safety: failed to inject corpus for '%s': %s",
                    block.name, exc,
                )
                if block.needs_corpus:
                    return BlockResult(
                        metrics={"error": f"Corpus unavailable: {exc}"},
                        routing=RoutingHint.CONTINUE,
                        diagnosis=f"Corpus injection failed: {exc}",
                    )

        return None
