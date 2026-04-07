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
# ADK_MODEL: primary model for research/executor agents (tool-capable).
# ADK_SYNTHESIS_MODEL: model for synthesis/thinker agents (defaults to
# ADK_MODEL when not set, so everything uses one model by default).
# ADK_THINKER_MODEL: model for the thinker agent (defaults to
# ADK_SYNTHESIS_MODEL, which defaults to ADK_MODEL).
_raw_model = os.environ.get("ADK_MODEL", "litellm/openai/gpt-4o")
ADK_MODEL_NAME = _raw_model.split(":")[0]

_raw_synthesis = os.environ.get("ADK_SYNTHESIS_MODEL", "")
_has_separate_synthesis = bool(_raw_synthesis)
ADK_SYNTHESIS_MODEL_NAME = _raw_synthesis.split(":")[0] if _raw_synthesis else ADK_MODEL_NAME

_raw_thinker = os.environ.get("ADK_THINKER_MODEL", "")
_has_separate_thinker = bool(_raw_thinker)
ADK_THINKER_MODEL_NAME = _raw_thinker.split(":")[0] if _raw_thinker else ADK_SYNTHESIS_MODEL_NAME

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
# When synthesis model uses a different provider (e.g. abliteration.ai
# for synthesis vs Venice for research), we need separate api_key,
# api_base, and extra_body per model.
_synthesis_api_base = os.environ.get("SYNTHESIS_API_BASE", "")
_synthesis_api_key = os.environ.get("SYNTHESIS_API_KEY", "")
_synthesis_is_venice = "venice.ai" in _synthesis_api_base
_synthesis_venice_params: dict = (
    json.loads(os.environ.get(
        "VENICE_PARAMS",
        json.dumps({"include_venice_system_prompt": False}),
    ))
    if _synthesis_is_venice
    else {}
)
_synthesis_extra_body: dict = {}
if _synthesis_venice_params:
    _synthesis_extra_body["venice_parameters"] = _synthesis_venice_params

# ── Thinker model config ─────────────────────────────────────────────
# When the thinker uses a different provider (e.g. abliteration.ai),
# we need separate api_key, api_base, and extra_body.
_thinker_api_base = os.environ.get("THINKER_API_BASE", _synthesis_api_base)
_thinker_api_key = os.environ.get("THINKER_API_KEY", _synthesis_api_key)
_thinker_is_venice = "venice.ai" in _thinker_api_base
_thinker_venice_params: dict = (
    json.loads(os.environ.get(
        "VENICE_PARAMS",
        json.dumps({"include_venice_system_prompt": False}),
    ))
    if _thinker_is_venice
    else {}
)
_thinker_extra_body: dict = {}
if _thinker_venice_params:
    _thinker_extra_body["venice_parameters"] = _thinker_venice_params


def build_model(
    *,
    parallel_tool_calls: bool = True,
    synthesis: bool = False,
    thinker: bool = False,
) -> Union[str, LiteLlm]:
    """Return the model for ADK Agent(model=...).

    * Native Gemini models (``gemini-*``) -> plain string (ADK native path).
    * Everything else -> ``LiteLlm`` with vendor-specific ``extra_body``.

    Args:
        parallel_tool_calls: Whether the model may emit multiple tool calls
            in a single response.  Set to ``False`` for sub-agents that need
            sequential tool execution (e.g. executor) so each result is
            processed before the next search is issued.
        synthesis: If True, use the synthesis model (ADK_SYNTHESIS_MODEL)
            instead of the primary research model.  Synthesis agents (final
            report) can use a different provider (e.g. abliteration.ai)
            while research agents use Venice GLM-4.7.
        thinker: If True, use the thinker model (ADK_THINKER_MODEL).
            Defaults to the synthesis model if not separately configured.
    """
    if thinker:
        name = ADK_THINKER_MODEL_NAME
    elif synthesis:
        name = ADK_SYNTHESIS_MODEL_NAME
    else:
        name = ADK_MODEL_NAME

    # When no separate synthesis/thinker model is configured, inherit the
    # primary model's extra_body so Venice params aren't silently dropped.
    if thinker and _has_separate_thinker and _thinker_api_base:
        extra = _thinker_extra_body
    elif (synthesis or thinker) and _has_separate_synthesis and _synthesis_api_base:
        extra = _synthesis_extra_body
    else:
        extra = _extra_body

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

    # When synthesis/thinker model uses a different provider, pass
    # per-model api_key and api_base so LiteLLM routes correctly.
    if thinker and _thinker_api_key:
        kwargs["api_key"] = _thinker_api_key
    elif (synthesis or thinker) and _synthesis_api_key:
        kwargs["api_key"] = _synthesis_api_key
    if thinker and _thinker_api_base:
        kwargs["api_base"] = _thinker_api_base
    elif (synthesis or thinker) and _synthesis_api_base:
        kwargs["api_base"] = _synthesis_api_base

    return LiteLlm(model=name, **kwargs)
