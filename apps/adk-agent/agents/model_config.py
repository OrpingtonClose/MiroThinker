# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Shared model configuration for all ADK agents.

Builds a ``LiteLlm`` instance that correctly forwards vendor-specific
parameters (e.g. Venice ``venice_parameters``) via ``extra_body``.

LiteLLM's ``:param=value`` model-string syntax does NOT forward
vendor-specific body parameters — they end up ignored.  By using an
explicit ``LiteLlm(extra_body=...)`` we guarantee they reach the API.
"""

from __future__ import annotations

import json
import os

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


def build_model() -> LiteLlm:
    """Return a configured ``LiteLlm`` with vendor params forwarded."""
    return LiteLlm(model=ADK_MODEL_NAME, extra_body=_extra_body)
