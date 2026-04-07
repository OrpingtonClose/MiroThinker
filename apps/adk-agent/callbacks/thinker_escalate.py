# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-agent callback for the thinker inside a LoopAgent.

When the thinker outputs text containing the sentinel ``EVIDENCE_SUFFICIENT``,
this callback sets ``escalate=True`` on the event actions so that the
enclosing ``LoopAgent`` breaks out of the research loop and hands off to
the synthesiser.

This keeps the thinker 100 % tool-free: it signals "done" via plain text,
and the callback translates that into an ADK-native escalation event.
"""

from __future__ import annotations

import logging
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

_SENTINEL = "EVIDENCE_SUFFICIENT"


def thinker_escalate_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """Check if the thinker signalled that enough evidence has been gathered.

    Reads the thinker's ``output_key`` value (``research_strategy``) from
    session state.  If the text contains the ``EVIDENCE_SUFFICIENT`` sentinel
    the callback sets ``escalate = True`` so the ``LoopAgent`` exits.

    Returns ``None`` so the thinker's original output is preserved.
    """
    strategy = callback_context.state.get("research_strategy", "")
    if _SENTINEL in strategy:
        logger.info("Thinker signalled EVIDENCE_SUFFICIENT — escalating out of research loop")
        callback_context.actions.escalate = True
    return None
