# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Model configuration for the Strands Venice agent.

Supports two modes:
  1. Static — uses VENICE_MODEL env var (fast, no probe overhead)
  2. Runtime selection — probes candidate models with a topic-derived
     censorship test and cascades to the first uncensored model.

The ``include_venice_system_prompt: false`` parameter is critical for
uncensored operation — it disables Venice's built-in safety system
prompt so the agent's own prompt has full control.

Set ``MODEL_SELECTION=runtime`` to enable runtime probing.
Set ``VENICE_MODEL_OVERRIDE`` to force a specific model (skips runtime probing).

Reference: apps/adk-agent/agents/model_config.py lines 41-56.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

VENICE_API_BASE = os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "zai-org-glm-5")

# Default model changed from heretic (known empty-content issue) to GLM 5
# (fastest uncensored model in benchmarks: 8157 chars, 24.4s, FC+Reasoning).


def build_model():
    """Build Strands model provider pointing at Venice AI (static mode)."""
    from strands.models.openai import OpenAIModel

    api_key = os.environ.get("VENICE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "VENICE_API_KEY is not set. "
            "Copy .env.example to .env and add your Venice API key."
        )

    return OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": VENICE_API_BASE,
            "timeout": 120.0,
            "max_retries": 2,
        },
        model_id=VENICE_MODEL,
        params={
            "extra_body": {
                "venice_parameters": {"include_venice_system_prompt": False},
            }
        },
    )


def build_model_with_selection(user_query: str):
    """Build Strands model using runtime censorship probing.

    Probes candidate models with a topic-derived test extracted from
    *user_query*.  Returns (model, selection_result) where selection_result
    contains the full probe log for transparency.

    Falls back to static ``build_model()`` if:
      - ``MODEL_SELECTION`` env var is not ``runtime``
      - ``VENICE_MODEL_OVERRIDE`` is explicitly set (user override)
    """
    from model_selector import SelectionResult, build_model_from_selection, select_model

    selection_mode = os.environ.get("MODEL_SELECTION", "static")
    explicit_model = os.environ.get("VENICE_MODEL_OVERRIDE")

    if selection_mode != "runtime":
        logger.info("Static model selection: %s", VENICE_MODEL)
        return build_model(), None

    if explicit_model:
        logger.info("Model override: %s (skipping probe)", explicit_model)
        from strands.models.openai import OpenAIModel

        api_key = os.environ.get("VENICE_API_KEY", "")
        if not api_key:
            raise RuntimeError("VENICE_API_KEY is not set.")
        return OpenAIModel(
            client_args={"api_key": api_key, "base_url": VENICE_API_BASE, "timeout": 120.0, "max_retries": 2},
            model_id=explicit_model,
            params={"extra_body": {"venice_parameters": {"include_venice_system_prompt": False}}},
        ), None

    logger.info("Runtime model selection — probing candidates...")
    selection = select_model(user_query)
    logger.info("Selected: %s (%s)", selection.label, selection.model_id)
    logger.info("\n%s", selection.summary())

    model = build_model_from_selection(selection)
    return model, selection
