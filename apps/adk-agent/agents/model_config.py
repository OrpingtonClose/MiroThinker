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

# ── Model names ──────────────────────────────────────────────────────
# ADK_MODEL: primary model for research/web/browsing agents.
# ADK_SYNTHESIS_MODEL: model for synthesis/summary agents (defaults to
# ADK_MODEL when not set, so everything uses one model by default).
_raw_model = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")
ADK_MODEL_NAME = _raw_model.split(":")[0]

_raw_synthesis = os.environ.get("ADK_SYNTHESIS_MODEL", "")
ADK_SYNTHESIS_MODEL_NAME = _raw_synthesis.split(":")[0] if _raw_synthesis else ADK_MODEL_NAME

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


# ── Synthesis model config ────────────────────────────────────────────
# When synthesis model uses a different provider (e.g. Venice for
# synthesis vs RunPod for research), we need separate api_key, api_base,
# and extra_body per model.
_synthesis_api_base = os.environ.get("SYNTHESIS_API_BASE", "")
_synthesis_api_key = os.environ.get("SYNTHESIS_API_KEY", "")
_synthesis_is_venice = "venice.ai" in _synthesis_api_base
_synthesis_venice_params: dict = json.loads(
    os.environ.get("VENICE_PARAMS", json.dumps(
        {"include_venice_system_prompt": False} if _synthesis_is_venice else {}
    ))
)
_synthesis_extra_body: dict = {}
if _synthesis_venice_params:
    _synthesis_extra_body["venice_parameters"] = _synthesis_venice_params


def build_model(
    *,
    parallel_tool_calls: bool = True,
    synthesis: bool = False,
) -> Union[str, LiteLlm]:
    """Return the model for ADK Agent(model=...).

    * Native Gemini models (``gemini-*``) → plain string (ADK native path).
    * Everything else → ``LiteLlm`` with vendor-specific ``extra_body``.

    Args:
        parallel_tool_calls: Whether the model may emit multiple tool calls
            in a single response.  Set to ``False`` for sub-agents that need
            sequential tool execution (e.g. web_agent) so each result is
            processed before the next search is issued.
        synthesis: If True, use the synthesis model (ADK_SYNTHESIS_MODEL)
            instead of the primary research model.  Synthesis agents (summary,
            final-answer) can use a different provider (e.g. Venice) while
            research agents use RunPod/Qwen.
    """
    name = ADK_SYNTHESIS_MODEL_NAME if synthesis else ADK_MODEL_NAME
    extra = _synthesis_extra_body if synthesis else _extra_body

    # Strip the ``litellm/`` prefix — it's an ADK routing convention,
    # not part of the LiteLLM model identifier.
    if name.startswith("litellm/"):
        name = name[len("litellm/"):]

    # Native Gemini models use ADK's built-in handler (no LiteLLM wrapper).
    if name.startswith("gemini"):
        return name

    kwargs: dict = {"extra_body": extra}
    if not parallel_tool_calls:
        kwargs["parallel_tool_calls"] = False

    # When synthesis model uses a different provider, pass per-model
    # api_key and api_base so LiteLLM routes to the correct endpoint
    # instead of using the global OPENAI_API_KEY / OPENAI_API_BASE.
    if synthesis and _synthesis_api_key:
        kwargs["api_key"] = _synthesis_api_key
    if synthesis and _synthesis_api_base:
        kwargs["api_base"] = _synthesis_api_base

    return LiteLlm(model=name, **kwargs)
