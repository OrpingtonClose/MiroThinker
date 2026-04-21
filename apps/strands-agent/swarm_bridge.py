# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Bridge between strands-agent and GossipSwarm engine.

Provides async CompleteFn wrappers that call a **localhost-only**
OpenAI-compatible endpoint (Ollama) and a top-level
``gossip_synthesize`` function that runs the full gossip pipeline
on a research corpus.

The strands-agent researcher gathers raw data (TranscriptAPI, web
search, etc.).  That raw output becomes the corpus for GossipSwarm,
which adds multi-worker angle analysis, adversarial gossip rounds,
serendipity cross-connections, and a queen merge — the quality
mechanisms that make research trustworthy.

Security:
    A localhost guard (``_assert_localhost``) enforces that all swarm
    LLM calls stay on the local machine.  Remote API URLs are rejected
    at startup to prevent accidental data leakage to external services.
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
from urllib.parse import urlparse

from swarm.config import SwarmConfig
from swarm.engine import GossipSwarm, SwarmResult
from swarm.lineage import LineageStore

logger = logging.getLogger(__name__)

# ── Localhost guard ───────────────────────────────────────────────────

_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _assert_localhost(url: str) -> None:
    """Reject any URL whose host is not localhost.

    The swarm MUST use local models only.  This guard prevents
    accidental data leakage to remote APIs.

    Raises:
        ValueError: If the URL points to a non-localhost host.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in _LOCALHOST_HOSTS:
        msg = (
            f"Swarm completion URL must be localhost, got {url!r} "
            f"(host={host!r}). Set SWARM_API_BASE to a localhost URL "
            f"(e.g. http://localhost:11434/v1). Remote APIs are forbidden "
            f"for swarm workers."
        )
        raise ValueError(msg)


# ── Local model completion functions (Ollama, OpenAI-compatible) ──────

_SWARM_API_BASE = os.environ.get(
    "SWARM_API_BASE",
    os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
)

# Validate at import time — fail fast if misconfigured
_assert_localhost(_SWARM_API_BASE)

# Per-phase model selection.
# Workers: Gemma-4-26B abliterated (fast, uncensored, parallel).
# Queen: Qwen3.5-27B-Claude-Opus abliterated (best writer).
# Serendipity: same as queen (best cross-domain reasoner).
_WORKER_MODEL = os.environ.get("SWARM_WORKER_MODEL", "gemma-4-uncensored")
_QUEEN_MODEL = os.environ.get("SWARM_QUEEN_MODEL", "qwen-claude-opus")
_SERENDIPITY_MODEL = os.environ.get("SWARM_SERENDIPITY_MODEL", _QUEEN_MODEL)

logger.info(
    "swarm_api_base=<%s>, worker_model=<%s>, queen_model=<%s>, "
    "serendipity_model=<%s> | swarm bridge configured (localhost-only)",
    _SWARM_API_BASE, _WORKER_MODEL, _QUEEN_MODEL, _SERENDIPITY_MODEL,
)


async def _local_complete(
    prompt: str,
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Call localhost OpenAI-compatible chat completions endpoint.

    Works with Ollama (``/v1/chat/completions``), vLLM, llama.cpp server,
    or any local OpenAI-compatible API.

    Returns the assistant message content, or empty string on failure.
    """
    # Re-validate at call time in case env was mutated
    _assert_localhost(_SWARM_API_BASE)

    url = f"{_SWARM_API_BASE}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Produce your analysis."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("model=<%s>, url=<%s> | local LLM call failed", model, url)
            return ""


async def worker_complete(prompt: str) -> str:
    """CompleteFn for swarm workers — Gemma-4 uncensored (local)."""
    return await _local_complete(prompt, _WORKER_MODEL)


async def queen_complete(prompt: str) -> str:
    """CompleteFn for queen merge — Qwen-Claude-Opus (local)."""
    return await _local_complete(
        prompt, _QUEEN_MODEL, max_tokens=8192, temperature=0.3,
    )


async def serendipity_complete(prompt: str) -> str:
    """CompleteFn for serendipity bridge — cross-domain reasoner (local)."""
    return await _local_complete(prompt, _SERENDIPITY_MODEL)


# ── Gossip synthesis entry point ──────────────────────────────────────


async def gossip_synthesize(
    corpus: str,
    query: str,
    on_event: "Callable[[dict], Awaitable[None]] | None" = None,
    cancel_event: "asyncio.Event | None" = None,
    corpus_delta_fn: "Callable[[], Awaitable[str]] | None" = None,
    lineage_store: "LineageStore | None" = None,
) -> SwarmResult:
    """Run full gossip swarm pipeline on a research corpus.

    Args:
        corpus: Raw research output from the strands-agent researcher.
        query: The user's original research query.
        on_event: Optional async callback for streaming progress events.
            Called with structured dicts at each phase boundary and
            gossip round completion.
        cancel_event: Optional asyncio.Event checked between gossip
            rounds.  If set, the swarm stops early and returns a
            partial result.
        corpus_delta_fn: Optional async callback that returns new findings
            as formatted text.  Called between gossip rounds to inject
            external data from producers running in parallel.
        lineage_store: Optional LineageStore for persisting all swarm
            phase outputs.  When provided, every bee's work (synthesis,
            gossip rounds, serendipity, queen merge) is documented in
            the store.

    Returns:
        SwarmResult with user_report, knowledge_report, metrics, etc.
    """
    config = SwarmConfig()
    config.corpus_delta_fn = corpus_delta_fn
    if lineage_store is not None:
        config.lineage_store = lineage_store

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

    result = await swarm.synthesize(
        corpus=corpus,
        query=query,
        on_event=on_event,
        cancel_event=cancel_event,
    )

    logger.info(
        "llm_calls=<%d>, elapsed_s=<%.1f>, user_report_chars=<%d>, "
        "knowledge_report_chars=<%d> | gossip synthesis complete",
        result.metrics.total_llm_calls,
        result.metrics.total_elapsed_s,
        len(result.user_report),
        len(result.knowledge_report),
    )

    return result
