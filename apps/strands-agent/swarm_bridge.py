# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Bridge between strands-agent and GossipSwarm engine.

Provides async CompleteFn wrappers that call Venice API directly,
and a top-level ``gossip_synthesize`` function that runs the full
gossip pipeline on a research corpus.

The strands-agent researcher gathers raw data (TranscriptAPI, web
search, etc.).  That raw output becomes the corpus for GossipSwarm,
which adds multi-worker angle analysis, adversarial gossip rounds,
serendipity cross-connections, and a queen merge — the quality
mechanisms that make research trustworthy.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``swarm.*`` is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import httpx

from swarm.config import SwarmConfig
from swarm.engine import GossipSwarm, SwarmResult

logger = logging.getLogger(__name__)

# ── Venice API completion functions ───────────────────────────────────

_VENICE_API_BASE = os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1")
_VENICE_API_KEY = os.environ.get("VENICE_API_KEY", "")

# Per-phase model selection.
# Workers: fast uncensored model (same as agent default).
# Queen: best available writer — cleaner English, no CJK corruption.
# Serendipity: best cross-domain reasoner.
_WORKER_MODEL = os.environ.get(
    "SWARM_WORKER_MODEL",
    os.environ.get("VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic"),
)
_QUEEN_MODEL = os.environ.get("SWARM_QUEEN_MODEL", _WORKER_MODEL)
_SERENDIPITY_MODEL = os.environ.get("SWARM_SERENDIPITY_MODEL", _WORKER_MODEL)


async def _venice_complete(
    prompt: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Call Venice API chat completions endpoint.

    Returns the assistant message content, or empty string on failure.
    """
    url = f"{_VENICE_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {_VENICE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Produce your analysis."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "venice_parameters": {"include_venice_system_prompt": False},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("model=<%s> | venice API call failed", model)
            return ""


async def worker_complete(prompt: str) -> str:
    """CompleteFn for swarm workers — fast uncensored model."""
    return await _venice_complete(prompt, _WORKER_MODEL)


async def queen_complete(prompt: str) -> str:
    """CompleteFn for queen merge — best writer model."""
    return await _venice_complete(
        prompt, _QUEEN_MODEL, max_tokens=8192, temperature=0.3,
    )


async def serendipity_complete(prompt: str) -> str:
    """CompleteFn for serendipity bridge — cross-domain reasoner."""
    return await _venice_complete(prompt, _SERENDIPITY_MODEL)


# ── Gossip synthesis entry point ──────────────────────────────────────


async def gossip_synthesize(
    corpus: str,
    query: str,
) -> SwarmResult:
    """Run full gossip swarm pipeline on a research corpus.

    Args:
        corpus: Raw research output from the strands-agent researcher.
        query: The user's original research query.

    Returns:
        SwarmResult with user_report, knowledge_report, metrics, etc.
    """
    config = SwarmConfig()

    swarm = GossipSwarm(
        complete=worker_complete,
        worker_complete=worker_complete,
        queen_complete=queen_complete,
        serendipity_complete=serendipity_complete,
        config=config,
    )

    logger.info(
        "corpus_chars=<%d>, workers=<%d>, gossip_rounds=<%d> | starting gossip synthesis",
        len(corpus), config.max_workers, config.gossip_rounds,
    )

    result = await swarm.synthesize(corpus=corpus, query=query)

    logger.info(
        "llm_calls=<%d>, elapsed_s=<%.1f>, user_report_chars=<%d>, "
        "knowledge_report_chars=<%d> | gossip synthesis complete",
        result.metrics.total_llm_calls,
        result.metrics.total_elapsed_s,
        len(result.user_report),
        len(result.knowledge_report),
    )

    return result
