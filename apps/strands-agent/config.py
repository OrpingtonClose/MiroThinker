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

VENICE_API_KEY = os.environ["VENICE_API_KEY"]
VENICE_API_BASE = os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "zai-org-glm-4.7")


def build_model():
    """Build Strands model provider pointing at Venice AI."""
    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        client_args={
            "api_key": VENICE_API_KEY,
            "base_url": VENICE_API_BASE,
        },
        model_id=VENICE_MODEL,
        params={
            "extra_body": {"venice_parameters": {"include_venice_system_prompt": False}}
        },
    )
