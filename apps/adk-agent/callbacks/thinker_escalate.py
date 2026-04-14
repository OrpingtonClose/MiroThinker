# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-agent callback for the thinker inside a LoopAgent.

Thin wrapper that delegates to ``ThinkerBlock`` via the block adapter.
All business logic (thought admission, convergence detection, escalation)
lives in the block; all cross-cutting concerns (timing, health, I/O
validation, DuckDB safety, error escalation) are handled by aspects.

ASYNC: ADK supports async callbacks (``inspect.isawaitable`` check in
``base_agent.py``).
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from callbacks.block_adapter import run_block_from_callback
from models.pipeline_block import RoutingHint

logger = logging.getLogger(__name__)


async def thinker_escalate_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Delegate to ThinkerBlock, then translate routing into ADK escalation.

    The block handles thought admission, strategy tracking, convergence
    detection, and EVIDENCE_SUFFICIENT signalling.  This callback
    translates the block's ``RoutingHint.ESCALATE`` into ADK's
    ``actions.escalate = True`` so the ``LoopAgent`` exits.
    """
    result = await run_block_from_callback("thinker", callback_context)

    # Translate block routing into ADK flow control
    if result.routing in (RoutingHint.ESCALATE, RoutingHint.ABORT):
        callback_context.actions.escalate = True
        if result.routing == RoutingHint.ABORT:
            callback_context.state["_pipeline_aborted"] = True
            callback_context.state["_abort_reason"] = result.diagnosis

    return None
