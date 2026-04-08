# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-tool callback — DEMO_MODE truncation.

Bad Result Detection (Algorithm 4) and Dynamic Compression have been removed —
they are now handled by ``ReflectAndRetryToolPlugin`` (catches tool errors,
shows the LLM structured reflection guidance, retries up to N times) and
``ContextFilterPlugin`` (invocation-level context trimming makes per-result
compression unnecessary).

Dedup Guard is now a prompt-based soft guard via ``GlobalInstructionPlugin``
(static instruction, no session-state dependency).

What remains:
1. **DEMO_MODE truncation** — caps scrape results to 20K chars in demo mode.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# Maximum length for scrape results in demo mode
DEMO_SCRAPE_MAX_LENGTH = 20_000


def _maybe_truncate_scrape(tool_name: str, result_text: str) -> str:
    """In DEMO_MODE, truncate scrape_website results to 20K chars."""
    if os.environ.get("DEMO_MODE") != "1":
        return result_text
    if tool_name not in (
        "scrape",
        "scrape_website",
        "firecrawl_scrape",
        "crawling_exa",
    ):
        return result_text
    if not isinstance(result_text, str):
        return result_text

    try:
        parsed = json.loads(result_text)
        text = parsed.get("text", "")
        if text and len(text) > DEMO_SCRAPE_MAX_LENGTH:
            text = text[:DEMO_SCRAPE_MAX_LENGTH]
        return json.dumps({"text": text}, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        if len(result_text) > DEMO_SCRAPE_MAX_LENGTH:
            return result_text[:DEMO_SCRAPE_MAX_LENGTH]
        return result_text


# ── callback ─────────────────────────────────────────────────────────────────


def after_tool_callback(
    tool: Any, args: Dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[Dict[str, Any]]:
    """
    ADK after_tool_callback.

    Returns *None* to keep the original result, or a replacement value
    that ADK substitutes as the tool result.
    """
    tool_name: str = tool.name if hasattr(tool, "name") else str(tool)
    result_text = str(tool_response) if tool_response is not None else ""

    # ── DEMO_MODE truncation ────────────────────────────────────────────
    truncated = _maybe_truncate_scrape(tool_name, result_text)
    if truncated != result_text:
        return {"result": truncated}

    return None  # keep the original result
