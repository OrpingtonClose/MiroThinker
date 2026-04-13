# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""SearchExecutorBlock -- automated multi-API search (no LLM).

Wraps ``run_search_executor()`` as a fenced pipeline block.  Reads
expansion targets from the corpus and strategy queries from the
thinker, fires search APIs programmatically, and ingests results.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading

from models.pipeline_block import (
    BlockContext, BlockCriticality, BlockResult, ParamSpec,
    PipelineBlock,
)

logger = logging.getLogger(__name__)


class SearchExecutorBlock(PipelineBlock):
    """Automated search phase: multi-API fan-out, content extraction."""

    name = "search_executor"
    needs_corpus = True
    criticality = BlockCriticality.BEST_EFFORT
    is_looped = True

    input_specs = [
        ParamSpec(
            key="research_strategy",
            expected_type=str,
            required=False,
            description="Thinker's strategy with extractable queries",
        ),
    ]
    output_specs = []  # Corpus mutation — no direct state output

    async def execute(self, ctx: BlockContext) -> BlockResult:
        from callbacks.condition_manager import _drain_search_queue
        from tools.search_executor import run_search_executor

        state = ctx.state
        iteration = state.get("_corpus_iteration", 0)

        # Set tracing context on corpus
        if ctx.corpus is not None and ctx.collector is not None:
            try:
                ctx.corpus.set_trace_context(
                    session_id=ctx.collector.session_id,
                    iteration=iteration,
                )
            except Exception:
                pass

        # Drain any queued search results from previous tool callbacks
        _drain_search_queue(state)

        # Run the automated search executor
        cancel_event = ctx.cancel or threading.Event()
        stats: dict = {}

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                timed_out = False
                _se_timeout = int(os.environ.get("SEARCH_EXECUTOR_TIMEOUT", "300"))
                try:
                    future = pool.submit(
                        asyncio.run,
                        run_search_executor(state, cancel=cancel_event),
                    )
                    wrapped = asyncio.wrap_future(future)
                    stats = await asyncio.wait_for(wrapped, timeout=_se_timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    logger.warning("Search executor timed out after %ds", _se_timeout)
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(asyncio.wrap_future(future), timeout=5)
                    except (asyncio.TimeoutError, Exception):
                        pass
                    stats = {"timed_out": True}
                finally:
                    if not timed_out:
                        cancel_event.set()
                    pool.shutdown(wait=False, cancel_futures=True)
            else:
                stats = await run_search_executor(state, cancel=cancel_event)

            logger.info("Search executor stats: %s", stats)

            if ctx.collector is not None and isinstance(stats, dict):
                try:
                    ctx.collector.emit_event("search_executor", data=stats)
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Search executor failed (non-fatal): %s", exc, exc_info=True)
            stats = {"error": str(exc)}

        metrics = dict(stats) if isinstance(stats, dict) else {"raw": str(stats)}
        return BlockResult(metrics=metrics)
