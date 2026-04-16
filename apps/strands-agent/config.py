# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Model configuration for the Strands Venice agent.

Configures the Venice GLM-4.7 model using Strands' OpenAI-compatible
provider.  The ``include_venice_system_prompt: false`` parameter is
critical for uncensored operation — it disables Venice's built-in
safety system prompt so the agent's own prompt has full control.

Reference: apps/adk-agent/agents/model_config.py lines 41-56.
"""

import os

from dotenv import load_dotenv

load_dotenv()

VENICE_API_BASE = os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic")


def build_model(temperature: float | None = None):
    """Build Strands model provider pointing at Venice AI.

    Args:
        temperature: Optional sampling temperature override.  When *None*
            the API default is used.  Higher values (e.g. 1.2) make the
            model more creative/divergent — useful for breaking out of
            repetitive search patterns.
    """
    from strands.models.openai import OpenAIModel

    api_key = os.environ.get("VENICE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "VENICE_API_KEY is not set. "
            "Copy .env.example to .env and add your Venice API key."
        )

    params: dict = {
        "extra_body": {
            "venice_parameters": {"include_venice_system_prompt": False},
            "reasoning": {"effort": "high"},
        }
    }
    if temperature is not None:
        params["temperature"] = temperature

    return OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": VENICE_API_BASE,
        },
        model_id=VENICE_MODEL,
        params=params,
    )
