# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""DuckDBSafetyAspect -- ensure DuckDB thread-safety invariant.

Blocks that declare ``needs_corpus = True`` require exclusive DuckDB
access.  This aspect **waits** for the per-corpus async lock to become
available instead of silently skipping the block.

Phase 2 fix: changed from skip-on-contention to queue-not-skip.
No blocks are silently dropped; they wait their turn.
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

        # Wait for DuckDB availability instead of skipping (Phase 2).
        # acquire() + release() ensures the lock was free at entry.
        # Individual DuckDB operations within blocks use _safe_corpus_write
        # for fine-grained locking.
        try:
            from callbacks.condition_manager import _get_corpus_lock
            corpus_key = ctx.state.get("_corpus_key", "default")
            lock = _get_corpus_lock(corpus_key)
            if lock.locked():
                logger.info(
                    "DuckDB safety: waiting for lock (key=%s, block=%s)",
                    corpus_key, block.name,
                )
                await lock.acquire()
                lock.release()
                logger.info(
                    "DuckDB safety: lock acquired and released for '%s'",
                    block.name,
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
