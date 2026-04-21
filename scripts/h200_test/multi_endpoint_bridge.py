# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Multi-endpoint swarm bridge for 8×H200 deployment.

Extends the single-endpoint swarm_bridge.py to support routing worker
requests across multiple vLLM instances (one per GPU). Each GPU serves
a different uncensored model for epistemic diversity.

For the 1×H200 test, all endpoints point to the same vLLM instance.
For 8×H200 production, each endpoint is a separate GPU + model.

Configuration via environment variable:
    SWARM_ENDPOINTS — comma-separated list of vLLM base URLs
    e.g. "http://gpu0:8000/v1,http://gpu1:8001/v1,..."

    If not set, falls back to SWARM_API_BASE (single endpoint).

The bridge round-robins worker requests across available endpoints.
Queen and serendipity can be pinned to a specific endpoint via:
    SWARM_QUEEN_ENDPOINT — override for queen (default: first endpoint)
    SWARM_SERENDIPITY_ENDPOINT — override for serendipity
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import httpx

from swarm.config import CompleteFn, SwarmConfig
from swarm.engine import GossipSwarm

logger = logging.getLogger(__name__)


# ── Localhost guard ───────────────────────────────────────────────────

_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _is_localhost(url: str) -> bool:
    """Check if a URL points to localhost."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in _ALLOWED_HOSTS


def _assert_all_localhost(urls: list[str]) -> None:
    """Raise if any URL is not localhost.

    For multi-VM deployments where GPUs are on different machines,
    set SWARM_ALLOW_REMOTE=1 to bypass this guard.
    """
    allow_remote = os.environ.get("SWARM_ALLOW_REMOTE", "0") == "1"
    if allow_remote:
        return

    for url in urls:
        if not _is_localhost(url):
            msg = (
                f"Endpoint <{url}> is not localhost. For multi-VM deployments, "
                f"set SWARM_ALLOW_REMOTE=1 to allow remote endpoints. "
                f"Ensure all endpoints are on a private network."
            )
            raise RuntimeError(msg)


# ── Endpoint resolution ──────────────────────────────────────────────

def _resolve_endpoints() -> list[str]:
    """Resolve vLLM endpoint URLs from environment."""
    endpoints_str = os.environ.get("SWARM_ENDPOINTS", "")
    if endpoints_str:
        endpoints = [e.strip() for e in endpoints_str.split(",") if e.strip()]
    else:
        base = os.environ.get(
            "SWARM_API_BASE",
            os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
        )
        endpoints = [base]

    _assert_all_localhost(endpoints)
    return endpoints


def _resolve_model_for_endpoint(endpoint: str, idx: int) -> str:
    """Resolve which model to use for a given endpoint.

    Checks SWARM_MODEL_N (e.g. SWARM_MODEL_0, SWARM_MODEL_1, ...) first,
    then falls back to SWARM_WORKER_MODEL.
    """
    # Per-endpoint model override
    model = os.environ.get(f"SWARM_MODEL_{idx}", "")
    if model:
        return model

    # Global worker model
    return os.environ.get("SWARM_WORKER_MODEL", "huihui-ai/Qwen3.5-32B-abliterated")


# ── Completion functions ─────────────────────────────────────────────

async def _call_endpoint(
    prompt: str,
    endpoint: str,
    model: str,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> str:
    """Call a single vLLM endpoint."""
    url = f"{endpoint}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Produce your analysis."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.exception(
                "model=<%s>, endpoint=<%s> | endpoint call failed",
                model, endpoint,
            )
            return ""


class MultiEndpointRouter:
    """Round-robin router across multiple vLLM endpoints.

    Each call to ``worker_complete`` goes to the next endpoint in sequence.
    Queen and serendipity are pinned to dedicated endpoints.
    """

    def __init__(self) -> None:
        self.endpoints = _resolve_endpoints()
        self.models = {
            ep: _resolve_model_for_endpoint(ep, i)
            for i, ep in enumerate(self.endpoints)
        }

        # Round-robin iterator for worker requests
        self._worker_cycle = itertools.cycle(range(len(self.endpoints)))

        # Queen endpoint (first by default, or override)
        queen_ep = os.environ.get("SWARM_QUEEN_ENDPOINT", "")
        self.queen_endpoint = queen_ep if queen_ep else self.endpoints[0]
        self.queen_model = os.environ.get(
            "SWARM_QUEEN_MODEL",
            self.models.get(self.queen_endpoint, "huihui-ai/Qwen3.5-32B-abliterated"),
        )

        # Serendipity endpoint (last by default, or override)
        seren_ep = os.environ.get("SWARM_SERENDIPITY_ENDPOINT", "")
        self.serendipity_endpoint = seren_ep if seren_ep else self.endpoints[-1]
        self.serendipity_model = os.environ.get(
            "SWARM_SERENDIPITY_MODEL",
            self.models.get(
                self.serendipity_endpoint,
                "huihui-ai/Qwen3.5-32B-abliterated",
            ),
        )

        logger.info(
            "endpoints=<%d>, queen=<%s/%s>, serendipity=<%s/%s> | "
            "multi-endpoint router initialized",
            len(self.endpoints),
            self.queen_endpoint, self.queen_model,
            self.serendipity_endpoint, self.serendipity_model,
        )
        for i, ep in enumerate(self.endpoints):
            logger.info(
                "endpoint_%d=<%s>, model=<%s>",
                i, ep, self.models[ep],
            )

    async def worker_complete(self, prompt: str) -> str:
        """Route worker request to next endpoint via round-robin."""
        idx = next(self._worker_cycle)
        endpoint = self.endpoints[idx]
        model = self.models[endpoint]
        return await _call_endpoint(prompt, endpoint, model)

    async def queen_complete(self, prompt: str) -> str:
        """Route queen request to the dedicated queen endpoint."""
        return await _call_endpoint(
            prompt, self.queen_endpoint, self.queen_model,
            max_tokens=32768, temperature=0.3,
        )

    async def serendipity_complete(self, prompt: str) -> str:
        """Route serendipity request to the dedicated endpoint."""
        return await _call_endpoint(
            prompt, self.serendipity_endpoint, self.serendipity_model,
            max_tokens=16384, temperature=0.5,
        )

    def build_swarm(self, config: SwarmConfig | None = None) -> GossipSwarm:
        """Create a GossipSwarm wired to this router's endpoints."""
        return GossipSwarm(
            complete=self.worker_complete,
            worker_complete=self.worker_complete,
            queen_complete=self.queen_complete,
            serendipity_complete=self.serendipity_complete,
            config=config or SwarmConfig(),
        )
