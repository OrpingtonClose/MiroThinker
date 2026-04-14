# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Thin adapter: ADK callback → PipelineRunner block execution.

Every ADK callback delegates to the corresponding ``PipelineBlock`` via
``PipelineRunner.run_block()``.  This adapter handles the translation
between ADK's ``CallbackContext`` and the block framework's
``BlockContext`` / ``BlockResult``.

The adapter is intentionally thin — it does NOT contain business logic.
All logic lives in the blocks; all cross-cutting concerns live in aspects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dashboard import get_active_collector
from models.pipeline_block import BlockContext, BlockResult

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext

logger = logging.getLogger(__name__)


def _build_block_context(callback_context: "CallbackContext") -> BlockContext:
    """Build a BlockContext from an ADK CallbackContext.

    Translates ADK's callback-centric view into the block framework's
    dependency-injected context.  The corpus is lazily resolved from
    the module-level ``_corpus_stores`` dict.
    """
    state = callback_context.state
    corpus = None
    try:
        from callbacks.condition_manager import _get_corpus
        corpus = _get_corpus(state)
    except Exception:
        pass

    return BlockContext(
        state=state,
        corpus=corpus,
        collector=get_active_collector(),
        iteration=state.get("_corpus_iteration", 0),
        user_query=state.get("user_query", ""),
    )


async def run_block_from_callback(
    block_name: str,
    callback_context: "CallbackContext",
) -> BlockResult:
    """Run a named block through the PipelineRunner with all aspects.

    This is the single integration point between ADK callbacks and the
    aspect-oriented block pipeline.  Each callback becomes a one-liner
    that calls this function with the appropriate block name.

    Returns the ``BlockResult`` so the caller can translate routing
    hints into ADK-native flow control (e.g. ``actions.escalate``).
    """
    from agents.pipeline import get_pipeline_runner

    runner = get_pipeline_runner()
    ctx = _build_block_context(callback_context)
    result = await runner.run_block(block_name, ctx)

    # Apply state updates from the block result
    runner.apply_state_updates(ctx, result)

    # Log block completion with key metrics
    skipped = result.metrics.get("skipped", False)
    error = result.metrics.get("error")
    if skipped:
        logger.info(
            "Block '%s' skipped: %s",
            block_name, result.metrics.get("reason", "unknown"),
        )
    elif error:
        logger.warning("Block '%s' completed with error: %s", block_name, error)
    else:
        logger.debug("Block '%s' completed successfully", block_name)

    return result
