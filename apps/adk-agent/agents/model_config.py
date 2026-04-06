# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Shared model configuration for all ADK agents.

For native Gemini models (``gemini-*``), returns a plain string so ADK
uses its built-in handler.  For LiteLLM-routed models, returns a
``LiteLlm`` instance with vendor-specific ``extra_body`` forwarding
(e.g. Venice ``venice_parameters``).
"""

from __future__ import annotations

import json
import os
from typing import Union

from google.adk.models import LiteLlm

# ── Model name ───────────────────────────────────────────────────────
# Strip any :param=value suffix the user may have put in .env — those
# don't work via LiteLLM's model string and we handle them below.
_raw_model = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")
ADK_MODEL_NAME = _raw_model.split(":")[0]

# ── Extra body parameters (vendor-specific) ──────────────────────────
# Only inject venice_parameters when the API base actually points to Venice.
# Sending unknown body fields to stricter providers could cause 400 errors.
_api_base = os.environ.get("OPENAI_API_BASE", "")
_is_venice = "venice.ai" in _api_base

# VENICE_PARAMS is a JSON dict forwarded as ``venice_parameters`` in the
# request body.  Default when using Venice: disable the built-in system
# prompt so our own prompt has full control (critical for uncensored use).
_default_venice_params = json.dumps(
    {"include_venice_system_prompt": False} if _is_venice else {}
)
VENICE_PARAMS: dict = json.loads(
    os.environ.get("VENICE_PARAMS", _default_venice_params)
)

_extra_body: dict = {}
if VENICE_PARAMS:
    _extra_body["venice_parameters"] = VENICE_PARAMS


def build_model(*, parallel_tool_calls: bool = True) -> Union[str, LiteLlm]:
    """Return the model for ADK Agent(model=...).

    * Native Gemini models (``gemini-*``) → plain string (ADK native path).
    * Everything else → ``LiteLlm`` with vendor-specific ``extra_body``.

    Args:
        parallel_tool_calls: Whether the model may emit multiple tool calls
            in a single response.  Set to ``False`` for sub-agents that need
            sequential tool execution (e.g. web_agent) so each result is
            processed before the next search is issued.
    """
    name = ADK_MODEL_NAME

    # Strip the ``litellm/`` prefix — it's an ADK routing convention,
    # not part of the LiteLLM model identifier.
    if name.startswith("litellm/"):
        name = name[len("litellm/"):]

    # Native Gemini models use ADK's built-in handler (no LiteLLM wrapper).
    if name.startswith("gemini"):
        return name

    kwargs: dict = {"extra_body": _extra_body}
    if not parallel_tool_calls:
        kwargs["parallel_tool_calls"] = False

    return LiteLlm(model=name, **kwargs)
