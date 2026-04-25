#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Run the gossip swarm pipeline.

The pipeline makes smart decisions at runtime — most configuration is
resolved automatically from the environment, the angles, and the corpus.

What the pipeline decides on its own:
    - Worker count: one per angle (override via SWARM_MAX_WORKERS)
    - Enrichment: runs when angles define enrichment_queries
    - Corpus: auto-discovers mega_corpus.txt or youtube_corpus_combined.txt
    - Flock backend: always local vLLM (scales with hardware, no rate limits)
    - Queen routing: local vLLM by default, OpenRouter when OPENROUTER_API_KEY set
    - Corpus validation: refuses <50K chars, warns <500K
    - Serendipity, hive memory, diversity gossip: always enabled

Usage:
    # Start vLLM first (in another terminal):
    ./launch_vllm.sh huihui-ai/Qwen3.5-32B-abliterated

    # Simplest invocation — everything auto-resolved:
    python run_swarm_test.py

    # With explicit corpus:
    python run_swarm_test.py --corpus /path/to/corpus.txt

    # With existing enriched database:
    python run_swarm_test.py --db enriched.duckdb

Environment variables:
    SWARM_API_BASE          — vLLM endpoint (default: http://localhost:8000/v1)
    SWARM_WORKER_MODEL      — model name served by vLLM
    SWARM_QUEEN_MODEL       — queen model (routes to OpenRouter if OPENROUTER_API_KEY set)
    SWARM_SERENDIPITY_MODEL — serendipity model (same as worker for single-GPU)
    SWARM_MAX_WORKERS       — cap on concurrent workers (default: one per angle)
    SWARM_GOSSIP_ROUNDS     — gossip rounds (default: 3)
    OPENROUTER_API_KEY      — set to route queen to OpenRouter
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_STRANDS_AGENT = str(Path(__file__).resolve().parents[2] / "apps" / "strands-agent")
if _STRANDS_AGENT not in sys.path:
    sys.path.insert(0, _STRANDS_AGENT)

from angles import ALL_ANGLES, REQUIRED_ANGLE_LABELS, AngleDefinition, get_swarm_query
from corpus import ConditionStore
from swarm.config import SwarmConfig
from swarm.engine import GossipSwarm, SwarmResult
from swarm.lineage import LineageStore
from tracing import (
    init_tracing,
    shutdown_tracing,
    trace_event,
    trace_phase,
    traced_complete_fn,
    upload_output_dir_to_b2,
    upload_traces_to_b2,
)

logger = logging.getLogger(__name__)


# ── vLLM completion function ─────────────────────────────────────────

def _get_api_base() -> str:
    """Resolve the vLLM API base URL."""
    return os.environ.get("SWARM_API_BASE", "http://localhost:8000/v1")


def _get_model(env_var: str, default: str) -> str:
    """Resolve a model name from environment."""
    return os.environ.get(env_var, default)


async def _vllm_complete(
    prompt: str,
    model: str,
    api_base: str,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> str:
    """Call vLLM OpenAI-compatible chat completions endpoint."""
    import httpx

    url = f"{api_base}/chat/completions"
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
                "model=<%s>, url=<%s> | vLLM call failed", model, url,
            )
            return ""


def make_complete_fn(
    model: str,
    api_base: str,
    max_tokens: int = 16384,
    temperature: float = 0.3,
):
    """Create a CompleteFn bound to a specific model and endpoint.

    Args:
        model: Resolved model identifier.
        api_base: Resolved API base URL.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.
    """

    async def _complete(prompt: str) -> str:
        return await _vllm_complete(
            prompt, model, api_base, max_tokens, temperature,
        )

    return _complete


# ── OpenRouter completion function ───────────────────────────────────


async def _openrouter_complete(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    """Call OpenRouter chat completions endpoint.

    Uses the same chat format as vLLM (system + user message) for
    compatibility with the Flock's prefix caching prompt structure.

    Args:
        prompt: The full prompt (system message content).
        model: OpenRouter model identifier.
        api_key: OpenRouter API key.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        The completion text, or empty string on failure.
    """
    import httpx

    url = "https://openrouter.ai/api/v1/chat/completions"
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
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning(
                    "model=<%s>, error=<%s> | openrouter call returned error",
                    model, data["error"],
                )
                return ""
            return data["choices"][0]["message"]["content"]
        except Exception:
            logger.exception(
                "model=<%s> | openrouter call failed", model,
            )
            return ""


def make_openrouter_complete_fn(
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> Callable[[str], Awaitable[str]]:
    """Create a CompleteFn targeting OpenRouter.

    Reads OPENROUTER_API_KEY from environment.

    Args:
        model: OpenRouter model identifier.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        An async completion callable.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set — OpenRouter calls will fail")

    async def _complete(prompt: str) -> str:
        return await _openrouter_complete(
            prompt, model, api_key, max_tokens, temperature,
        )

    return _complete


# ── DeepSeek API completion function ──────────────────────────────────
#
# Direct DeepSeek API for V4 Flash workers — cheap, fast, uncensored,
# unlimited concurrency (API-based, no GPU cost for bees).

# Model constants must be defined before functions that use them as defaults
_DEEPSEEK_V4_FLASH_MODEL = "deepseek-v4-flash"  # via DeepSeek API
_DEEPSEEK_V4_PRO_MODEL = "deepseek-ai/DeepSeek-V4-Pro"  # local vLLM
_DEEPSEEK_V4_PRO_OPENROUTER = "deepseek/deepseek-v4-pro"  # via OpenRouter


async def _deepseek_api_complete(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> str:
    """Call DeepSeek API chat completions endpoint.

    Uses the OpenAI-compatible chat format.  DeepSeek V4 Flash returns
    separate reasoning_content (chain-of-thought) and content fields —
    we concatenate both for maximum analytical depth.

    Args:
        prompt: The full prompt (system message content).
        model: DeepSeek model identifier (e.g. "deepseek-v4-flash").
        api_key: DeepSeek API key.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        The completion text (reasoning + content), or empty string on failure.
    """
    import httpx

    url = "https://api.deepseek.com/v1/chat/completions"
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
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning(
                    "model=<%s>, error=<%s> | deepseek api returned error",
                    model, data["error"],
                )
                return ""
            msg = data["choices"][0]["message"]
            # V4 models return reasoning_content (CoT) + content
            reasoning = msg.get("reasoning_content", "") or ""
            content = msg.get("content", "") or ""
            # Concatenate reasoning and content for full analytical depth
            if reasoning and content:
                return f"<reasoning>\n{reasoning}\n</reasoning>\n\n{content}"
            return content or reasoning
        except Exception:
            logger.exception(
                "model=<%s> | deepseek api call failed", model,
            )
            return ""


def make_deepseek_api_complete_fn(
    model: str = _DEEPSEEK_V4_FLASH_MODEL,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> Callable[[str], Awaitable[str]]:
    """Create a CompleteFn targeting the DeepSeek API.

    Reads DEEPSEEK_API_KEY from environment.

    Args:
        model: DeepSeek model identifier.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        An async completion callable.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set — DeepSeek API calls will fail")

    async def _complete(prompt: str) -> str:
        return await _deepseek_api_complete(
            prompt, model, api_key, max_tokens, temperature,
        )

    return _complete


# ── Multi-instance vLLM management ────────────────────────────────────
#
# Manages multiple vLLM processes across GPUs for the multi-model swarm.
# Phase 1 (bees+gossip): DeepSeek V4 Flash via API (unlimited concurrency)
# Phase 2 (Flock): DeepSeek V4 Pro on 8×H200 via vLLM docker

# Default models for each phase
_KIMI_LINEAR_MODEL = "huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated"
_QWEN3_235B_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# DeepSeek V4 models defined above (before make_deepseek_api_complete_fn)

# Legacy remote worker models via OpenRouter
_LING_MODEL = "inclusionai/ling-2.6-1t:free"

# vLLM base port — instances use 8000, 8001, 8002, ...
_VLLM_BASE_PORT = 8000

# Track running vLLM subprocesses, log handles, and Docker containers for cleanup
_vllm_processes: list[subprocess.Popen[bytes]] = []
_vllm_log_handles: list[object] = []
_docker_containers: list[str] = []


def _compute_instances_per_gpu(
    model_vram_gb: float,
    gpu_vram_gb: float = 143.7,
    kv_reserve_gb: float = 10.0,
) -> int:
    """Compute how many model instances fit on a single GPU.

    Packs as many instances as possible while reserving space for
    KV cache per instance.

    Args:
        model_vram_gb: Estimated model weight VRAM in GB.
        gpu_vram_gb: Total GPU VRAM in GB (H200 = 143.7).
        kv_reserve_gb: VRAM reserved per instance for KV cache.

    Returns:
        Number of instances that fit on one GPU (minimum 1).
    """
    usable = gpu_vram_gb * 0.95  # 5% overhead for CUDA context
    per_instance = model_vram_gb + kv_reserve_gb
    count = int(usable // per_instance)
    return max(1, count)


# Known model VRAM estimates (FP8 weights) in GB
_MODEL_VRAM_ESTIMATES: dict[str, float] = {
    _KIMI_LINEAR_MODEL: 48.0,   # 48B MoE at FP8
    _QWEN3_235B_MODEL: 235.0,   # 235B MoE at FP8, needs TP
}


async def start_vllm_instances(
    model: str,
    num_gpus: int = 8,
    tp_per_instance: int = 1,
    max_model_len: int = 131072,
    *,
    instances_per_gpu: int | None = None,
    model_vram_gb: float | None = None,
    extra_vllm_args: list[str] | None = None,
) -> list[str]:
    """Start vLLM instances packed across GPUs.

    Automatically computes how many instances fit per GPU based on
    model weight size.  For small MoE models (e.g. Kimi-Linear 48B
    with 3B active at FP8 = ~48 GB), this packs 2 instances per
    143 GB H200 GPU = 16 instances across 8 GPUs.

    Args:
        model: HuggingFace model id or local path.
        num_gpus: Total available GPUs.
        tp_per_instance: Tensor-parallel degree per instance.
        max_model_len: Context window per instance.
        instances_per_gpu: Override auto-computed packing density.
        model_vram_gb: Model weight size in GB (auto-looked up if None).
        extra_vllm_args: Additional CLI args passed to vLLM.

    Returns:
        List of API base URLs (e.g. ["http://localhost:8000/v1", ...]).
    """
    import httpx

    # Compute packing density
    if instances_per_gpu is None:
        vram = model_vram_gb or _MODEL_VRAM_ESTIMATES.get(model, 50.0)
        instances_per_gpu = _compute_instances_per_gpu(vram)

    num_instances = instances_per_gpu * (num_gpus // tp_per_instance)
    # Each instance's share of GPU memory
    mem_util = round(0.95 / instances_per_gpu, 2)

    logger.info(
        "model=<%s>, instances_per_gpu=<%d>, total_instances=<%d>, "
        "mem_util_each=<%.2f>, tp=<%d> | packing vLLM instances",
        model, instances_per_gpu, num_instances, mem_util, tp_per_instance,
    )

    endpoints: list[str] = []

    for i in range(num_instances):
        port = _VLLM_BASE_PORT + i
        # Pack instances onto GPUs: instances 0..N-1 on GPU 0,
        # instances N..2N-1 on GPU 1, etc.
        gpu_idx = (i // instances_per_gpu) * tp_per_instance
        gpu_ids = ",".join(
            str(gpu_idx + g) for g in range(tp_per_instance)
        )

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_ids}
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--tensor-parallel-size", str(tp_per_instance),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(mem_util),
            "--trust-remote-code",
            "--dtype", "auto",
            "--disable-log-requests",
        ]
        if extra_vllm_args:
            cmd.extend(extra_vllm_args)

        logger.info(
            "instance=<%d>, port=<%d>, gpus=<%s>, model=<%s>, "
            "mem_util=<%.2f> | starting vLLM",
            i, port, gpu_ids, model, mem_util,
        )

        log_fh = open(f"/workspace/vllm_{port}.log", "w")  # noqa: SIM115
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        _vllm_processes.append(proc)
        _vllm_log_handles.append(log_fh)
        endpoints.append(f"http://localhost:{port}/v1")

    # Wait for all instances to become healthy
    logger.info(
        "instances=<%d> | waiting for vLLM instances to become healthy",
        num_instances,
    )
    healthy = set()
    for attempt in range(120):  # up to 10 min
        await asyncio.sleep(5)
        async with httpx.AsyncClient(timeout=5.0) as client:
            for idx, ep in enumerate(endpoints):
                if idx in healthy:
                    continue
                try:
                    resp = await client.get(f"{ep}/models")
                    if resp.status_code == 200:
                        healthy.add(idx)
                        logger.info(
                            "instance=<%d>, endpoint=<%s> | vLLM healthy",
                            idx, ep,
                        )
                except Exception:
                    pass
        if len(healthy) == len(endpoints):
            break

    if len(healthy) < len(endpoints):
        failed = [
            endpoints[i] for i in range(len(endpoints)) if i not in healthy
        ]
        logger.warning(
            "healthy=<%d>, total=<%d>, failed=<%s> | "
            "some vLLM instances did not become healthy",
            len(healthy), len(endpoints), failed,
        )

    logger.info(
        "healthy=<%d>, total=<%d> | vLLM startup complete",
        len(healthy), len(endpoints),
    )
    return [endpoints[i] for i in sorted(healthy)]


def stop_vllm_instances() -> None:
    """Kill all tracked vLLM processes and Docker containers."""
    for proc in _vllm_processes:
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Wait briefly then force-kill stragglers
    for proc in _vllm_processes:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    _vllm_processes.clear()

    # Close log file handles
    for fh in _vllm_log_handles:
        try:
            fh.close()  # type: ignore[union-attr]
        except Exception:
            pass
    _vllm_log_handles.clear()

    # Stop Docker containers started by _start_v4_pro_docker.
    # docker run -d returns immediately so the tracked Popen is already
    # exited — we must explicitly remove the named container to free GPUs.
    for name in list(_docker_containers):
        logger.info("container=<%s> | stopping docker container", name)
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
        )
    _docker_containers.clear()

    # Also kill any orphaned vLLM processes on the host
    subprocess.run(
        ["pkill", "-9", "-f", "vllm.entrypoints"],
        capture_output=True,
    )
    logger.info("vllm_processes=<0> | all vLLM instances stopped")


def build_worker_routing(
    angles: list[str],
    local_endpoints: list[str],
    local_model: str,
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> dict[str, Callable[[str], Awaitable[str]]]:
    """Build per-angle CompleteFn routing for multi-model swarm.

    All worker bees use DeepSeek V4 Flash via the DeepSeek API.
    This is cheap, fast, uncensored (8/8 censorship test pass),
    and has unlimited concurrency (API-based, no GPU cost).

    Each worker is wrapped with tracing for per-angle observability,
    plus a local vLLM fallback if the API is unreachable.

    Args:
        angles: List of angle labels from the swarm config.
        local_endpoints: API base URLs of local vLLM instances (fallback).
        local_model: Model name served by local vLLM (fallback).
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        Dict mapping angle label to traced CompleteFn.
    """
    has_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY", ""))
    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY", ""))
    routing: dict[str, Callable[[str], Awaitable[str]]] = {}

    n = len(angles)

    for idx, angle in enumerate(angles):
        # Local endpoint for fallback (round-robin across instances)
        local_ep = local_endpoints[idx % len(local_endpoints)] if local_endpoints else ""
        local_fn = make_complete_fn(
            local_model, local_ep, max_tokens, temperature,
        ) if local_ep else None

        if has_deepseek:
            # All workers use DeepSeek V4 Flash via API — unlimited concurrency
            raw_fn = make_deepseek_api_complete_fn(
                _DEEPSEEK_V4_FLASH_MODEL, max_tokens, temperature,
            )
            # Wrap with tracing
            traced_fn = traced_complete_fn(
                raw_fn,
                f"worker.{angle}",
                model_name=_DEEPSEEK_V4_FLASH_MODEL,
                backend="deepseek-api",
            )
            routing[angle] = _make_fallback_fn(traced_fn, local_fn, angle)
            logger.info(
                "angle=<%s>, backend=<deepseek-v4-flash>, fallback=<local> | "
                "worker routing assigned",
                angle,
            )
        elif has_openrouter:
            # Fallback: use OpenRouter if DeepSeek API key not available
            raw_fn = make_openrouter_complete_fn(
                _DEEPSEEK_V4_PRO_OPENROUTER, max_tokens, temperature,
            )
            traced_fn = traced_complete_fn(
                raw_fn,
                f"worker.{angle}",
                model_name=_DEEPSEEK_V4_PRO_OPENROUTER,
                backend="openrouter",
            )
            routing[angle] = _make_fallback_fn(traced_fn, local_fn, angle)
            logger.info(
                "angle=<%s>, backend=<deepseek-v4-pro-openrouter>, fallback=<local> | "
                "worker routing assigned",
                angle,
            )
        else:
            # Last resort: local vLLM only
            if local_fn:
                traced_fn = traced_complete_fn(
                    local_fn,
                    f"worker.{angle}",
                    model_name=local_model,
                    backend="vllm-local",
                )
                routing[angle] = traced_fn
            logger.info(
                "angle=<%s>, backend=<local> | "
                "worker routing assigned (no API keys available)",
                angle,
            )

    logger.info(
        "total_angles=<%d>, backend=<%s> | worker routing complete",
        n,
        "deepseek-v4-flash" if has_deepseek else (
            "openrouter" if has_openrouter else "local"
        ),
    )
    return routing


def _make_fallback_fn(
    primary: Callable[[str], Awaitable[str]],
    fallback: Callable[[str], Awaitable[str]] | None,
    angle: str,
    max_retries: int = 3,
) -> Callable[[str], Awaitable[str]]:
    """Wrap a primary CompleteFn with retry + local fallback.

    Tries primary up to max_retries times.  On exhaustion, falls back
    to the local CompleteFn.  If no fallback is available, returns empty.

    Args:
        primary: Remote CompleteFn (e.g. OpenRouter).
        fallback: Local CompleteFn (e.g. Kimi-Linear vLLM).
        angle: Angle label for logging.
        max_retries: Retries before falling back.

    Returns:
        A resilient CompleteFn.
    """

    async def _resilient(prompt: str) -> str:
        for attempt in range(1, max_retries + 1):
            result = await primary(prompt)
            if result:
                return result
            logger.warning(
                "angle=<%s>, attempt=<%d>, max=<%d> | "
                "remote call returned empty, retrying",
                angle, attempt, max_retries,
            )
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # exponential backoff

        # Primary exhausted — fall back to local
        if fallback is not None:
            logger.warning(
                "angle=<%s> | remote exhausted after %d attempts, "
                "falling back to local",
                angle, max_retries,
            )
            return await fallback(prompt)

        logger.error(
            "angle=<%s> | remote exhausted and no fallback available",
            angle,
        )
        return ""

    return _resilient


async def swap_to_flock_model(
    model: str = _DEEPSEEK_V4_PRO_MODEL,
    num_gpus: int = 8,
    max_model_len: int = 1048576,
    *,
    use_docker: bool = True,
) -> str:
    """Kill current vLLM instances and start the Flock model on all GPUs.

    For DeepSeek V4 Pro, uses the official vLLM deepseekv4-cu130 Docker
    image with data-parallel-size=8 and hybrid attention optimisations
    (FP8 KV cache, FP4 indexer, 256-token blocks, expert parallelism).

    For other models, falls back to the standard vLLM Python entrypoint
    with tensor parallelism.

    Args:
        model: Model for Flock evaluation.
        num_gpus: GPUs available.
        max_model_len: Context window.
        use_docker: Use Docker for V4 Pro (recommended for production).

    Returns:
        API base URL of the Flock vLLM instance.
    """
    logger.info(
        "model=<%s>, gpus=<%d>, docker=<%s> | swapping to Flock model",
        model, num_gpus, use_docker,
    )
    stop_vllm_instances()

    # Give CUDA a moment to release memory
    await asyncio.sleep(10)

    is_v4_pro = "DeepSeek-V4-Pro" in model or "deepseek-v4-pro" in model.lower()

    if is_v4_pro and use_docker:
        # Use the official vLLM deepseekv4-cu130 Docker image
        return await _start_v4_pro_docker(model, num_gpus)

    # Fallback: standard vLLM Python entrypoint with tensor parallelism
    endpoints = await start_vllm_instances(
        model=model,
        num_gpus=num_gpus,
        tp_per_instance=num_gpus,
        max_model_len=max_model_len,
        instances_per_gpu=1,
    )

    if not endpoints:
        logger.error("flock model failed to start — no healthy endpoints")
        return ""

    logger.info(
        "endpoint=<%s> | Flock model ready",
        endpoints[0],
    )
    return endpoints[0]


async def _start_v4_pro_docker(
    model: str = _DEEPSEEK_V4_PRO_MODEL,
    num_gpus: int = 8,
) -> str:
    """Start DeepSeek V4 Pro via the official vLLM Docker image.

    Uses the exact command from the vLLM blog post "DeepSeek V4 in vLLM:
    Efficient Long-context Attention" (April 24, 2026):
    - vllm/vllm-openai:deepseekv4-cu130 image
    - data-parallel-size 8 (NOT tensor-parallel-size)
    - FP8 KV cache + FP4 indexer cache
    - 256-token blocks for hybrid CSA+HCA attention
    - Expert parallelism for MoE efficiency
    - Full+piecewise CUDA graph compilation
    - DeepSeek V4 tokenizer and reasoning parser

    Args:
        model: HuggingFace model identifier.
        num_gpus: GPU count for data parallelism.

    Returns:
        API base URL (http://localhost:8000/v1) or empty on failure.
    """
    import httpx

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    cmd = [
        "docker", "run",
        "--gpus", "all",
        "--ipc=host",
        "-p", "8000:8000",
        "-v", f"{hf_home}:/root/.cache/huggingface",
        "-d",  # detach so we can poll for readiness
        "--name", "vllm-v4-pro",
        "vllm/vllm-openai:deepseekv4-cu130",
        model,
        "--trust-remote-code",
        "--kv-cache-dtype", "fp8",
        "--block-size", "256",
        "--enable-prefix-caching",
        "--enable-expert-parallel",
        "--data-parallel-size", str(num_gpus),
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE", "custom_ops":["all"]}',
        "--attention_config.use_fp4_indexer_cache=True",
        "--tokenizer-mode", "deepseek_v4",
        "--tool-call-parser", "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser", "deepseek_v4",
    ]

    logger.info(
        "model=<%s>, gpus=<%d> | starting V4 Pro via Docker "
        "(vllm/vllm-openai:deepseekv4-cu130)",
        model, num_gpus,
    )

    # Remove any existing container with the same name
    subprocess.run(
        ["docker", "rm", "-f", "vllm-v4-pro"],
        capture_output=True,
    )

    log_fh = open("/workspace/vllm_v4_pro_docker.log", "w")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    _vllm_processes.append(proc)
    _vllm_log_handles.append(log_fh)
    _docker_containers.append("vllm-v4-pro")

    endpoint = "http://localhost:8000/v1"

    # Wait for the container to become healthy (model loading can take
    # 10-30 minutes for V4 Pro depending on download speed)
    logger.info(
        "endpoint=<%s> | waiting for V4 Pro to become healthy "
        "(this may take 10-30 minutes for model loading)",
        endpoint,
    )

    for attempt in range(360):  # up to 30 min
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{endpoint}/models")
                if resp.status_code == 200:
                    logger.info(
                        "attempt=<%d>, endpoint=<%s> | V4 Pro Docker healthy",
                        attempt, endpoint,
                    )
                    return endpoint
        except Exception:
            if attempt % 12 == 0:  # log every minute
                logger.info(
                    "attempt=<%d> | still waiting for V4 Pro Docker...",
                    attempt,
                )

    logger.error("V4 Pro Docker did not become healthy after 30 minutes")
    return ""


# ── Flock backend resolution ─────────────────────────────────────────

# Known OpenRouter models for the Flock evaluation phase.
# Each entry maps a CLI choice to (model_id, BackendConfig factory).
_FLOCK_BACKENDS: dict[str, tuple[str, str]] = {
    "ling-free": ("inclusionai/ling-2.6-1t:free", "free_tier"),
    "ling-flash-free": ("inclusionai/ling-2.6-flash:free", "free_tier"),
    "deepseek-flash": ("deepseek/deepseek-v4-flash", "paid_api"),
    "deepseek-pro": ("deepseek/deepseek-v4-pro", "paid_api"),
}


def resolve_flock_backend(
    choice: str,
    local_model: str,
    local_api_base: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> tuple[Callable[[str], Awaitable[str]], "BackendConfig | None"]:
    """Resolve the Flock backend from a CLI choice string.

    Returns a (complete_fn, backend_config) tuple.  When choice is
    "local", backend_config is None (no risk wrapping needed).

    Args:
        choice: CLI backend choice (e.g. "ling-free", "local").
        local_model: Model for the local vLLM endpoint.
        local_api_base: Local vLLM API base URL.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        Tuple of (completion_callable, optional_backend_config).
    """
    from swarm.backend import BackendConfig

    if choice == "local":
        fn = make_complete_fn(local_model, local_api_base, max_tokens, temperature)
        return fn, None

    if choice not in _FLOCK_BACKENDS:
        logger.warning(
            "flock_backend=<%s> | unknown backend, falling back to local", choice,
        )
        fn = make_complete_fn(local_model, local_api_base, max_tokens, temperature)
        return fn, None

    model_id, tier_name = _FLOCK_BACKENDS[choice]

    # Build the raw completion function targeting OpenRouter
    raw_fn = make_openrouter_complete_fn(model_id, max_tokens, temperature)

    # Build a local fallback for when the remote backend fails
    local_fallback = BackendConfig.self_hosted(
        name="local-vllm-fallback",
        model=local_model,
        api_base=local_api_base,
    )

    # Build the risk-aware backend config
    if tier_name == "free_tier":
        backend_config = BackendConfig.free_tier(
            name=choice,
            model=model_id,
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            fallback=local_fallback,
        )
    else:
        backend_config = BackendConfig.paid_api(
            name=choice,
            model=model_id,
            api_base="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            fallback=local_fallback,
        )

    logger.info(
        "flock_backend=<%s>, model=<%s>, tier=<%s> | "
        "flock will use remote backend with risk-aware wrapping",
        choice, model_id, tier_name,
    )

    return raw_fn, backend_config


# ── Corpus loading ───────────────────────────────────────────────────

def load_corpus_from_file(path: str) -> str:
    """Load corpus text from a file."""
    with open(path) as f:
        return f.read()


def load_corpus_from_store(store: ConditionStore) -> str:
    """Export ConditionStore findings as a structured corpus for the swarm."""
    findings = store.get_findings(limit=100000)
    if not findings:
        return ""

    sections: dict[str, list[str]] = {}
    for row in findings:
        if isinstance(row, dict):
            fact = row.get("fact", str(row))
            angle = row.get("angle", "general")
        else:
            fact = str(row)
            angle = "general"

        sections.setdefault(angle, []).append(fact)

    # Build structured corpus with angle headers
    parts = []
    for angle, facts in sections.items():
        parts.append(f"## {angle}\n")
        for fact in facts:
            parts.append(fact)
        parts.append("")

    return "\n\n".join(parts)


# ── Main test runner ─────────────────────────────────────────────────

# Default OpenRouter queen model — used when auto-routing decides the queen
# should go remote but SWARM_QUEEN_MODEL is not explicitly set to an
# OpenRouter-compatible identifier.
_OPENROUTER_QUEEN_DEFAULT = "deepseek/deepseek-r1"


async def run_swarm_test(
    corpus: str,
    query: str,
    config: SwarmConfig,
    output_dir: str = ".",
    queen_routing: str = "local",
    resolved_model: str = "",
    local_endpoints: list[str] | None = None,
) -> SwarmResult:
    """Run the full gossip swarm pipeline and save results.

    Args:
        corpus: Full corpus text to analyze.
        query: Research query for the swarm.
        config: Pre-configured SwarmConfig (token budgets already sized).
        output_dir: Directory for output files.
        queen_routing: "local" or "openrouter" — decided by main() based on
            whether synthesis input fits in local context window.
        resolved_model: Model name discovered by main() via vLLM probing.
            When provided, used as the default instead of the hardcoded
            fallback.  Ensures the gossip engine uses the same model that
            main() discovered from the running vLLM instance.
        local_endpoints: API base URLs of local vLLM instances for
            multi-model worker routing.  When provided, builds a
            worker_complete_map that distributes angles across local
            instances + remote OpenRouter models.
    """
    api_base = _get_api_base()

    # Use the model main() discovered from vLLM, falling back to env var
    # then hardcoded default only if nothing was discovered.
    _fallback = resolved_model or "huihui-ai/Qwen3.5-32B-abliterated"
    worker_model = _get_model("SWARM_WORKER_MODEL", _fallback)
    serendipity_model = _get_model("SWARM_SERENDIPITY_MODEL", worker_model)

    # Queen model resolution depends on routing decision:
    # - "local": use SWARM_QUEEN_MODEL env var, fallback to worker_model
    # - "openrouter": use SWARM_QUEEN_MODEL env var, fallback to a known
    #   OpenRouter model (local vLLM names are NOT valid OpenRouter ids)
    queen_model_env = os.environ.get("SWARM_QUEEN_MODEL", "")
    if queen_routing == "openrouter":
        queen_model = queen_model_env or _OPENROUTER_QUEEN_DEFAULT
    else:
        queen_model = queen_model_env or worker_model

    worker_fn = make_complete_fn(
        worker_model, api_base,
        max_tokens=config.worker_max_tokens,
        temperature=config.worker_temperature,
    )

    # Queen routing is decided by main() based on context window analysis.
    # "openrouter" = synthesis input exceeds local context, use remote model.
    # "local" = input fits, or no OPENROUTER_API_KEY available.
    _has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY", ""))
    if queen_routing == "openrouter" and _has_openrouter_key:
        queen_fn = make_openrouter_complete_fn(
            queen_model,
            max_tokens=config.queen_max_tokens,
            temperature=config.queen_temperature,
        )
        logger.info(
            "queen_model=<%s>, routing=<openrouter> | queen routed to OpenRouter "
            "(synthesis input exceeds local context window)",
            queen_model,
        )
    else:
        queen_fn = make_complete_fn(
            queen_model, api_base,
            max_tokens=config.queen_max_tokens,
            temperature=config.queen_temperature,
        )
        if queen_routing == "openrouter" and not _has_openrouter_key:
            logger.warning(
                "queen_model=<%s> | queen should use OpenRouter but "
                "OPENROUTER_API_KEY not set — falling back to local with truncation risk",
                queen_model,
            )

    serendipity_fn = make_complete_fn(
        serendipity_model, api_base,
        max_tokens=config.worker_max_tokens,
        temperature=0.5,  # Slightly higher for serendipity creativity
    )

    # Build per-worker routing map when local endpoints are available.
    # This enables multi-model swarms: local Kimi-Linear instances +
    # remote Ling/DeepSeek workers confronting each other during gossip.
    worker_map: dict[str, Callable[[str], Awaitable[str]]] | None = None
    if local_endpoints:
        worker_map = build_worker_routing(
            angles=list(REQUIRED_ANGLE_LABELS),
            local_endpoints=local_endpoints,
            local_model=worker_model,
            max_tokens=config.worker_max_tokens,
            temperature=config.worker_temperature,
        )

    # Wrap completion functions with tracing before building the swarm
    worker_fn = traced_complete_fn(
        worker_fn, "swarm.worker",
        model_name=worker_model, backend="vllm-local",
    )
    queen_fn = traced_complete_fn(
        queen_fn, "swarm.queen",
        model_name=queen_model,
        backend="openrouter" if queen_routing == "openrouter" else "vllm-local",
    )
    serendipity_fn = traced_complete_fn(
        serendipity_fn, "swarm.serendipity",
        model_name=serendipity_model, backend="vllm-local",
    )

    swarm = GossipSwarm(
        complete=worker_fn,
        worker_complete=worker_fn,
        queen_complete=queen_fn,
        serendipity_complete=serendipity_fn,
        config=config,
        worker_complete_map=worker_map,
    )

    # Progress callback — emits OTel events alongside structured logs
    async def on_event(event: dict) -> None:
        phase = event.get("type", "unknown")
        logger.info("swarm_event=<%s> | %s", phase, json.dumps(event, default=str))
        # Emit OTel event on the current span
        trace_event(
            f"swarm.{phase}",
            {k: str(v) for k, v in event.items()},
        )

    logger.info(
        "corpus_chars=<%d>, workers=<%d>, gossip_rounds=<%d>, angles=<%d> | "
        "starting swarm test",
        len(corpus), config.max_workers, config.gossip_rounds,
        len(config.required_angles),
    )

    t0 = time.monotonic()

    # Run the full swarm under a pipeline-level tracing span
    tracer = __import__("tracing").get_tracer()
    with tracer.start_as_current_span(
        "pipeline.swarm_synthesis",
        attributes={
            "pipeline.corpus_chars": len(corpus),
            "pipeline.workers": config.max_workers,
            "pipeline.gossip_rounds": config.gossip_rounds,
            "pipeline.angles": len(config.required_angles),
            "pipeline.queen_routing": queen_routing,
            "pipeline.worker_model": worker_model,
            "pipeline.queen_model": queen_model,
        },
    ) as pipeline_span:
        result = await swarm.synthesize(
            corpus=corpus,
            query=query,
            on_event=on_event,
        )
        elapsed = time.monotonic() - t0

        # Record pipeline-level metrics on the span
        pipeline_span.set_attribute("pipeline.elapsed_s", round(elapsed, 1))
        pipeline_span.set_attribute("pipeline.total_llm_calls", result.metrics.total_llm_calls)
        pipeline_span.set_attribute("pipeline.total_workers", result.metrics.total_workers)
        pipeline_span.set_attribute("pipeline.gossip_rounds_executed", result.metrics.gossip_rounds_executed)
        pipeline_span.set_attribute("pipeline.gossip_converged_early", result.metrics.gossip_converged_early)
        pipeline_span.set_attribute("pipeline.user_report_chars", len(result.user_report))
        pipeline_span.set_attribute("pipeline.knowledge_report_chars", len(result.knowledge_report))

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # User report
    user_report_path = output_path / f"user_report_{timestamp}.md"
    with open(user_report_path, "w") as f:
        f.write(result.user_report)

    # Knowledge report
    knowledge_report_path = output_path / f"knowledge_report_{timestamp}.md"
    with open(knowledge_report_path, "w") as f:
        f.write(result.knowledge_report)

    # Metrics
    metrics_path = output_path / f"metrics_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "total_elapsed_s": elapsed,
                "total_llm_calls": result.metrics.total_llm_calls,
                "total_workers": result.metrics.total_workers,
                "gossip_rounds_executed": result.metrics.gossip_rounds_executed,
                "gossip_converged_early": result.metrics.gossip_converged_early,
                "serendipity_produced": result.metrics.serendipity_produced,
                "phase_times": result.metrics.phase_times,
                "angles_detected": result.angles_detected,
                "user_report_chars": len(result.user_report),
                "knowledge_report_chars": len(result.knowledge_report),
                "worker_summaries_chars": {
                    k: len(v) for k, v in result.worker_summaries.items()
                },
            },
            f, indent=2,
        )

    # Worker summaries (individual)
    for worker_id, summary in result.worker_summaries.items():
        worker_path = output_path / f"worker_{worker_id}_{timestamp}.md"
        with open(worker_path, "w") as f:
            f.write(summary)

    # Serendipity insights
    if result.serendipity_insights:
        seren_path = output_path / f"serendipity_{timestamp}.md"
        with open(seren_path, "w") as f:
            f.write(result.serendipity_insights)

    logger.info(
        "elapsed_s=<%.1f>, llm_calls=<%d>, user_report_chars=<%d>, "
        "knowledge_report_chars=<%d> | swarm test complete",
        elapsed, result.metrics.total_llm_calls,
        len(result.user_report), len(result.knowledge_report),
    )

    print(f"\n{'═' * 60}")
    print(f"  SWARM TEST COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Elapsed:            {elapsed:.1f}s")
    print(f"  LLM calls:          {result.metrics.total_llm_calls}")
    print(f"  Workers:            {result.metrics.total_workers}")
    print(f"  Gossip rounds:      {result.metrics.gossip_rounds_executed}")
    print(f"  Converged early:    {result.metrics.gossip_converged_early}")
    print(f"  Serendipity:        {result.metrics.serendipity_produced}")
    print(f"  User report:        {len(result.user_report):,} chars")
    print(f"  Knowledge report:   {len(result.knowledge_report):,} chars")
    print(f"  Angles:             {result.angles_detected}")
    print(f"\n  Output directory:   {output_path.resolve()}")
    print(f"  User report:        {user_report_path.name}")
    print(f"  Knowledge report:   {knowledge_report_path.name}")
    print(f"  Metrics:            {metrics_path.name}")
    print(f"{'═' * 60}\n")

    return result


# ── Agentic runtime decisions ────────────────────────────────────────
#
# Each function below inspects the live environment and makes a decision
# that used to be a CLI flag.  Every decision is logged with reasoning
# so the operator can audit why the pipeline chose what it chose.

_CORPUS_MIN_HARD = 50_000      # below this → ERROR + exit (hallucination zone)
_CORPUS_MIN_SOFT = 500_000     # below this → WARNING (quality risk)
_DEFAULT_TEMPERATURE = 0.3     # static choice — not derived from model probing

# Approximate chars-per-token ratio for sizing calculations.
# Conservative (high) estimate ensures we don't overflow context.
_CHARS_PER_TOKEN = 3.5

# Reserve fraction of context window for output generation.
_OUTPUT_RESERVE_FRAC = 0.25


@dataclass
class ModelCapabilities:
    """What we discovered about the local model by probing the endpoint."""

    model_id: str = ""
    context_window: int = 0      # max_model_len from vLLM
    alive: bool = False

    @property
    def usable_input_tokens(self) -> int:
        """Tokens available for input after reserving space for output."""
        return int(self.context_window * (1 - _OUTPUT_RESERVE_FRAC))

    @property
    def usable_input_chars(self) -> int:
        """Chars available for input (conservative estimate)."""
        return int(self.usable_input_tokens * _CHARS_PER_TOKEN)


@dataclass
class CorpusCoverage:
    """Per-angle coverage analysis of the corpus."""

    angle_label: str
    keyword_hits: int = 0
    estimated_chars: int = 0
    thin: bool = False


async def _probe_vllm(api_base: str) -> ModelCapabilities:
    """Probe the vLLM endpoint to discover model capabilities.

    Calls /v1/models to get the model id and max_model_len (context
    window).  This drives token budget sizing and queen routing.
    Falls back to conservative defaults if the endpoint is unreachable.
    """
    import httpx

    caps = ModelCapabilities()
    url = f"{api_base}/models"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        models = data.get("data", [])
        if models:
            m = models[0]
            caps.model_id = m.get("id", "")
            # vLLM exposes max_model_len in the model object
            caps.context_window = int(
                m.get("max_model_len", 0)
                or m.get("context_window", 0)
                or 0
            )
            caps.alive = True

        logger.info(
            "model_id=<%s>, context_window=<%d>, usable_input=<%d> tokens | "
            "vLLM probed successfully",
            caps.model_id, caps.context_window, caps.usable_input_tokens,
        )
    except Exception as exc:
        logger.warning(
            "api_base=<%s>, error=<%s> | vLLM probe failed — "
            "using conservative defaults (32K context assumed)",
            api_base, exc,
        )
        caps.context_window = 32768
        caps.alive = False

    return caps


def _analyze_corpus_coverage(
    corpus: str,
    angles: list[AngleDefinition],
) -> list[CorpusCoverage]:
    """Analyze how well the corpus covers each angle.

    Scans for angle keywords (key_compounds, key_interactions, and words
    from the angle description) to estimate per-angle coverage.  Angles
    with thin coverage get prioritized for enrichment.
    """
    corpus_lower = corpus.lower()
    results = []

    for angle in angles:
        # Build keyword set from angle metadata
        keywords: set[str] = set()
        for compound in angle.key_compounds:
            keywords.add(compound.lower())
        for interaction in angle.key_interactions:
            keywords.add(interaction.lower())
        # Extract significant words from description (>4 chars, not stopwords)
        for word in angle.description.split():
            w = word.strip(".,;:()").lower()
            if len(w) > 4:
                keywords.add(w)

        # Count keyword occurrences
        hits = 0
        for kw in keywords:
            hits += corpus_lower.count(kw)

        # Estimate chars relevant to this angle (rough: 200 chars per hit)
        est_chars = min(hits * 200, len(corpus))

        # "Thin" = fewer than 5 keyword hits per 100K corpus chars
        thin = hits < max(5, len(corpus) // 100_000 * 5)

        cov = CorpusCoverage(
            angle_label=angle.label,
            keyword_hits=hits,
            estimated_chars=est_chars,
            thin=thin,
        )
        results.append(cov)

        log_fn = logger.warning if thin else logger.info
        log_fn(
            "angle=<%s>, keyword_hits=<%d>, estimated_chars=<%d>, thin=<%s> | "
            "corpus coverage for angle",
            angle.label, hits, est_chars, thin,
        )

    thin_count = sum(1 for c in results if c.thin)
    if thin_count > 0:
        logger.warning(
            "thin_angles=<%d>/%d | %d angles have thin corpus coverage — "
            "enrichment should target these first",
            thin_count, len(results),
            thin_count,
        )

    return results


def _decide_token_budgets(
    caps: ModelCapabilities,
) -> dict[str, int]:
    """Size all token budgets based on the actual model context window.

    Instead of hardcoded defaults, scales budgets proportionally to
    what the model can actually handle.
    """
    # Apply fallback directly to caps so that derived @property methods
    # (usable_input_tokens, usable_input_chars) also use the fallback.
    if not caps.context_window:
        caps.context_window = 32768
    ctx = caps.context_window

    # Worker output: 25% of context, min 4096, max 16384
    worker_max = max(4096, min(16384, ctx // 4))

    # Queen output: 50% of context, min 8192, max 65536
    queen_max = max(8192, min(65536, ctx // 2))

    # Max section chars per worker: scale with usable input
    max_section = int(caps.usable_input_chars * 0.8)  # 80% of usable input

    # Context budget for queen merge: usable input chars
    context_budget = caps.usable_input_chars

    # Report generation budget
    report_max_tokens = max(8192, min(32768, ctx // 3))
    report_max_chars = int(report_max_tokens * _CHARS_PER_TOKEN)

    budgets = {
        "worker_max_tokens": worker_max,
        "queen_max_tokens": queen_max,
        "max_section_chars": max_section,
        "context_budget": context_budget,
        "report_max_tokens": report_max_tokens,
        "report_max_chars": report_max_chars,
        "max_return_chars": min(6000, max_section // 10),
    }

    logger.info(
        "context_window=<%d>, worker_max=<%d>, queen_max=<%d>, "
        "max_section=<%d>, context_budget=<%d> | "
        "token budgets sized from model capabilities",
        ctx, worker_max, queen_max, max_section, context_budget,
    )

    return budgets


def _decide_queen_routing(
    corpus_chars: int,
    worker_count: int,
    caps: ModelCapabilities,
) -> tuple[str, str]:
    """Decide whether the queen should run locally or via OpenRouter.

    Estimates the queen's input size from the corpus and worker count,
    compares it to the local model's context window, and routes
    accordingly.

    Returns:
        Tuple of (routing_decision, reason) where decision is
        "local" or "openrouter".
    """
    # Estimate queen input: each worker produces a summary, plus query and framing
    # Conservative: assume each worker summary is ~max_summary_chars
    est_summary_per_worker = min(corpus_chars // max(worker_count, 1), 100_000)
    est_queen_input_chars = (
        est_summary_per_worker * worker_count  # worker summaries
        + 5_000                                 # query + framing prompt
    )
    est_queen_input_tokens = int(est_queen_input_chars / _CHARS_PER_TOKEN)

    has_openrouter = bool(os.environ.get("OPENROUTER_API_KEY", ""))
    local_can_fit = est_queen_input_tokens < caps.usable_input_tokens

    if local_can_fit:
        reason = (
            f"queen input ~{est_queen_input_tokens} tokens fits in local "
            f"context ({caps.usable_input_tokens} usable)"
        )
        logger.info(
            "queen_input_est=<%d> tokens, local_usable=<%d> tokens | "
            "queen will run locally — input fits",
            est_queen_input_tokens, caps.usable_input_tokens,
        )
        return "local", reason

    if has_openrouter:
        reason = (
            f"queen input ~{est_queen_input_tokens} tokens exceeds local "
            f"context ({caps.usable_input_tokens} usable) — routing to "
            f"OpenRouter (128K+ context available)"
        )
        logger.info(
            "queen_input_est=<%d> tokens, local_usable=<%d> tokens | "
            "queen routed to OpenRouter — input too large for local model",
            est_queen_input_tokens, caps.usable_input_tokens,
        )
        return "openrouter", reason

    reason = (
        f"queen input ~{est_queen_input_tokens} tokens exceeds local "
        f"context ({caps.usable_input_tokens} usable) but no "
        f"OPENROUTER_API_KEY — queen will run locally with TRUNCATION"
    )
    logger.warning(
        "queen_input_est=<%d> tokens, local_usable=<%d> tokens | "
        "queen will be truncated — set OPENROUTER_API_KEY for full synthesis",
        est_queen_input_tokens, caps.usable_input_tokens,
    )
    return "local", reason


def _prioritize_enrichment(
    coverage: list[CorpusCoverage],
    angles: list[AngleDefinition],
) -> list[AngleDefinition]:
    """Reorder angles so thin-coverage angles get enriched first.

    Thin angles go first (sorted by ascending keyword hits), then
    well-covered angles.  This ensures limited enrichment budget goes
    to the angles that need it most.
    """
    coverage_map = {c.angle_label: c for c in coverage}
    angle_map = {a.label: a for a in angles}

    thin = [
        (coverage_map[a.label].keyword_hits, a)
        for a in angles
        if a.label in coverage_map and coverage_map[a.label].thin
    ]
    thin.sort(key=lambda x: x[0])  # fewest hits first

    ok = [
        a for a in angles
        if a.label not in coverage_map or not coverage_map[a.label].thin
    ]

    reordered = [a for _, a in thin] + ok

    if thin:
        thin_labels = [a.label for _, a in thin]
        logger.info(
            "thin_angles=<%s> | enrichment will prioritize these angles",
            ", ".join(thin_labels),
        )

    return reordered


def _should_enrich() -> int:
    """Decide whether to run enrichment based on angles configuration.

    Returns the total number of enrichment queries across all angles,
    or 0 if enrichment should be skipped.
    """
    try:
        total_eq = sum(len(a.enrichment_queries) for a in ALL_ANGLES)
        if total_eq > 0:
            logger.info(
                "enrichment_queries=<%d> | angles define enrichment queries — "
                "enrichment will run automatically",
                total_eq,
            )
        return total_eq
    except Exception:
        return 0


def _auto_discover_corpus() -> str:
    """Find the best corpus file available on this machine.

    Searches standard locations in priority order.  Prefers larger files
    (more data = better swarm output).
    """
    candidates = [
        Path(__file__).resolve().parent / "mega_corpus.txt",
        Path.home() / "data" / "youtube_corpus_combined.txt",
        Path.home() / "repos" / "MiroThinker" / "scripts" / "h200_test" / "mega_corpus.txt",
    ]

    # Pick the largest available file, not just the first
    best_path = ""
    best_size = 0
    for candidate in candidates:
        if candidate.exists():
            size = candidate.stat().st_size
            logger.info(
                "corpus_candidate=<%s>, size=<%d> | found corpus file",
                candidate, size,
            )
            if size > best_size:
                best_path = str(candidate)
                best_size = size

    if best_path:
        logger.info(
            "corpus_path=<%s>, size=<%d> | selected largest available corpus",
            best_path, best_size,
        )
    return best_path


def _auto_workers() -> int:
    """Decide worker count from angles and environment.

    One worker per angle, capped by SWARM_MAX_WORKERS env var if set.
    """
    angle_count = len(REQUIRED_ANGLE_LABELS)
    env_cap = int(os.environ.get("SWARM_MAX_WORKERS", "0"))
    workers = env_cap if env_cap > 0 else angle_count
    logger.info(
        "angles=<%d>, env_cap=<%d>, workers=<%d> | worker count resolved",
        angle_count, env_cap, workers,
    )
    return workers


def _validate_corpus(corpus: str) -> None:
    """Validate corpus size — exit on dangerously small, warn on suboptimal."""
    n = len(corpus)
    if n < _CORPUS_MIN_HARD:
        logger.error(
            "corpus_chars=<%d> | corpus dangerously small (<%d chars) — "
            "model WILL hallucinate, refusing to run",
            n, _CORPUS_MIN_HARD,
        )
        sys.exit(1)
    elif n < _CORPUS_MIN_SOFT:
        logger.warning(
            "corpus_chars=<%d> | corpus below recommended minimum (%d) — "
            "output quality may be poor",
            n, _CORPUS_MIN_SOFT,
        )
    else:
        logger.info("corpus_chars=<%d> | corpus size validated", n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the gossip swarm pipeline.  The pipeline probes the vLLM "
            "endpoint, analyzes the corpus, sizes token budgets from the "
            "actual context window, and routes the queen based on whether "
            "the synthesis input fits locally."
        ),
    )
    # ── Meaningful operator choices only ──────────────────────────
    parser.add_argument(
        "--corpus", default="",
        help="Path to corpus file (auto-discovers largest available if omitted)",
    )
    parser.add_argument(
        "--db", default="",
        help="Path to existing DuckDB database with enriched corpus",
    )
    parser.add_argument(
        "--query", default="",
        help="Override the research query (defaults to angles.get_swarm_query())",
    )
    parser.add_argument(
        "--engine", choices=["gossip", "mcp"], default="gossip",
        help="Swarm engine architecture (default: gossip)",
    )
    parser.add_argument(
        "--model", default="",
        help="Local model name (overrides SWARM_WORKER_MODEL env var)",
    )
    parser.add_argument(
        "--api-base", default="",
        help="vLLM API base URL (overrides SWARM_API_BASE env var)",
    )
    parser.add_argument(
        "--output-dir", default="swarm_results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--source-run", default="",
        help="Run identifier for provenance (auto-generated if omitted)",
    )
    parser.add_argument(
        "--multi-model", action="store_true",
        help=(
            "Enable multi-model swarm: worker bees use DeepSeek V4 Flash "
            "via API (unlimited concurrency), then swap to DeepSeek V4 Pro "
            "on 8×H200 via vLLM Docker for Flock evaluation"
        ),
    )
    parser.add_argument(
        "--local-model", default=_KIMI_LINEAR_MODEL,
        help="Model for local vLLM instances (fallback for workers)",
    )
    parser.add_argument(
        "--flock-model", default=_DEEPSEEK_V4_PRO_MODEL,
        help="Model for Flock evaluation phase (default: DeepSeek V4 Pro)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=8,
        help="Number of GPUs available (default: 8 for H200)",
    )
    parser.add_argument(
        "--trace-dir", default="/workspace/traces",
        help="Directory for OpenTelemetry trace files",
    )
    parser.add_argument(
        "--upload-b2", action="store_true",
        help="Upload traces and outputs to Backblaze B2 after run completes",
    )
    parser.add_argument(
        "--b2-bucket", default="mirothinker-traces",
        help="B2 bucket name for trace/output uploads",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # ══════════════════════════════════════════════════════════════
    #  PHASE 0: Initialise tracing
    # ══════════════════════════════════════════════════════════════

    resolved_source_run = args.source_run or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    tracer, trace_path = init_tracing(
        run_id=resolved_source_run,
        trace_dir=args.trace_dir,
    )

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1: Probe the environment
    # ══════════════════════════════════════════════════════════════

    api_base = args.api_base or _get_api_base()
    default_model = args.model or _get_model(
        "SWARM_WORKER_MODEL", "huihui-ai/Qwen3.5-32B-abliterated",
    )

    # Probe vLLM to discover model context window and capabilities.
    # This drives EVERY downstream sizing decision.
    caps = asyncio.run(_probe_vllm(api_base))

    # If the probe found a model id, prefer it over the env/default
    # (it's what vLLM is actually serving)
    if caps.model_id and not args.model:
        logger.info(
            "model_override=<%s> → <%s> | using model discovered from vLLM",
            default_model, caps.model_id,
        )
        default_model = caps.model_id

    # Size token budgets from actual context window
    budgets = _decide_token_budgets(caps)

    # Workers: one per angle
    resolved_workers = _auto_workers()
    resolved_gossip_rounds = int(os.environ.get("SWARM_GOSSIP_ROUNDS", "3"))

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2: Load and analyze corpus
    # ══════════════════════════════════════════════════════════════

    corpus = ""
    store = None

    # Step 1: Load from existing DuckDB if provided
    if args.db:
        store = ConditionStore(db_path=args.db)
        corpus = load_corpus_from_store(store)
        logger.info("corpus_chars=<%d> | loaded from existing database", len(corpus))

    # Step 2: Load from corpus file (auto-discover if not specified)
    corpus_path = args.corpus or _auto_discover_corpus()
    if corpus_path:
        file_corpus = load_corpus_from_file(corpus_path)
        corpus = f"{corpus}\n\n{file_corpus}" if corpus else file_corpus
        logger.info(
            "corpus_chars=<%d> | loaded from file %s",
            len(corpus), corpus_path,
        )

    # Step 3: Analyze corpus coverage per angle BEFORE enrichment
    # so we know which angles are thin and should be prioritized
    coverage = _analyze_corpus_coverage(corpus, list(ALL_ANGLES)) if corpus else []

    # Step 4: Enrichment — prioritize thin angles
    enrichment_count = _should_enrich()
    if enrichment_count > 0:
        from datetime import datetime, timezone
        from enrich_corpus import enrich_all

        if store is None:
            store = ConditionStore(db_path="enriched_corpus.duckdb")

        # Reorder angles so thin-coverage ones get enriched first
        prioritized_angles: list[AngleDefinition] | None = None
        if coverage:
            prioritized_angles = _prioritize_enrichment(coverage, list(ALL_ANGLES))
            thin_labels = [
                c.angle_label for c in coverage if c.thin
            ]
            if thin_labels:
                logger.info(
                    "enrichment_priority=<%s> | thin angles targeted first",
                    ", ".join(thin_labels),
                )

        pre_enrich_ts = datetime.now(timezone.utc).isoformat()
        enrich_results = enrich_all(
            store, max_per_query=10, extract_full_text=True,
            angles=prioritized_angles,
        )

        # Evaluate enrichment quality per angle
        for angle_label, count in (enrich_results or {}).items():
            if count == 0:
                logger.warning(
                    "angle=<%s>, results=<0> | enrichment returned nothing — "
                    "corpus may be thin for this angle, consider manual data gathering",
                    angle_label,
                )

        enriched = store.export_delta(since=pre_enrich_ts)
        if enriched:
            corpus = f"{corpus}\n\n{enriched}" if corpus else enriched
            logger.info(
                "enriched_chars=<%d>, total_corpus=<%d> | enrichment complete (delta only)",
                len(enriched), len(corpus),
            )

    if not corpus:
        logger.error("no corpus found — provide --corpus or --db, or place mega_corpus.txt alongside this script")
        sys.exit(1)

    _validate_corpus(corpus)

    # ══════════════════════════════════════════════════════════════
    #  PHASE 3: Build config from discovered capabilities
    # ══════════════════════════════════════════════════════════════

    config = SwarmConfig()
    config.required_angles = list(REQUIRED_ANGLE_LABELS)
    config.max_workers = resolved_workers
    config.gossip_rounds = resolved_gossip_rounds
    config.enable_serendipity = True
    config.enable_full_corpus_gossip = True
    config.enable_semantic_assignment = True
    config.enable_misassignment = True
    config.misassignment_ratio = 0.25
    config.enable_hive_memory = True
    config.enable_diversity_aware_gossip = True

    # Apply token budgets sized from the actual model context window
    config.worker_max_tokens = budgets["worker_max_tokens"]
    config.queen_max_tokens = budgets["queen_max_tokens"]
    config.max_section_chars = budgets["max_section_chars"]
    config.context_budget = budgets["context_budget"]

    # Decide queen routing: does the synthesis input fit locally?
    queen_routing, queen_reason = _decide_queen_routing(
        corpus_chars=len(corpus),
        worker_count=resolved_workers,
        caps=caps,
    )

    logger.info(
        "workers=<%d>, gossip_rounds=<%d>, angles=<%d>, engine=<%s>, "
        "queen=<%s>, context_window=<%d> | full config resolved\n"
        "  queen reasoning: %s",
        resolved_workers, resolved_gossip_rounds,
        len(REQUIRED_ANGLE_LABELS), args.engine,
        queen_routing, caps.context_window,
        queen_reason,
    )

    logger.info(
        "corpus_chars=<%d>, sections_est=<%d> | corpus ready",
        len(corpus),
        len(corpus) // config.max_section_chars + 1,
    )

    # ── Wire corpus_delta_fn for gossip rounds ───────────────────
    if store is not None:
        from datetime import datetime, timezone
        _watermark = datetime.now(timezone.utc).isoformat()

        async def _corpus_delta() -> str:
            """Fetch new findings added to the store since last check."""
            nonlocal _watermark
            delta = store.export_delta(since=_watermark)
            _watermark = datetime.now(timezone.utc).isoformat()
            return delta

        config.corpus_delta_fn = _corpus_delta

    query = args.query or get_swarm_query()

    # ══════════════════════════════════════════════════════════════
    #  PHASE 4: Run the selected engine
    # ══════════════════════════════════════════════════════════════

    if args.engine == "mcp":
        from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine

        if store is None:
            store = ConditionStore(db_path="mcp_swarm.duckdb")

        # Flock always uses local vLLM — thousands of queries, must scale
        # with hardware not rate limits
        _flock_fn, flock_backend_config = resolve_flock_backend(
            choice="local",
            local_model=default_model,
            local_api_base=api_base,
            max_tokens=budgets["worker_max_tokens"],
            temperature=_DEFAULT_TEMPERATURE,
        )

        mcp_config = MCPSwarmConfig(
            max_workers=resolved_workers,
            max_waves=3,
            convergence_threshold=5,
            api_base=api_base,
            model=default_model,
            api_key="not-needed",
            max_tokens=budgets["worker_max_tokens"],
            temperature=_DEFAULT_TEMPERATURE,
            required_angles=list(REQUIRED_ANGLE_LABELS),
            report_max_tokens=budgets["report_max_tokens"],
            enable_serendipity_wave=True,
            source_model=default_model,
            source_run=resolved_source_run,
            max_return_chars=budgets["max_return_chars"],
            compact_every_n_waves=3,
            enable_rolling_summaries=True,
            report_max_chars=budgets["report_max_chars"],
            flock_backend_config=flock_backend_config,
            flock_complete=_flock_fn,
        )

        complete_fn = make_complete_fn(
            default_model, api_base,
            max_tokens=budgets["worker_max_tokens"],
            temperature=_DEFAULT_TEMPERATURE,
        )

        engine = MCPSwarmEngine(
            store=store,
            complete=complete_fn,
            config=mcp_config,
        )

        async def _run_mcp() -> None:
            result = await engine.synthesize(
                corpus=corpus,
                query=query,
            )
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            report_path = output_path / f"mcp_report_{timestamp}.md"
            with open(report_path, "w") as f:
                f.write(result.report)

            metrics_path = output_path / f"mcp_metrics_{timestamp}.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "engine": "mcp",
                    "model": default_model,
                    "api_base": api_base,
                    "context_window": caps.context_window,
                    "queen_routing": queen_routing,
                    "queen_reason": queen_reason,
                    "budgets": budgets,
                    "max_workers": resolved_workers,
                    "corpus_coverage": {
                        c.angle_label: {
                            "keyword_hits": c.keyword_hits,
                            "thin": c.thin,
                        }
                        for c in coverage
                    },
                    "serendipity_enabled": True,
                    "source_model": default_model,
                    "source_run": resolved_source_run,
                    "flock_backend": "local",
                    "total_elapsed_s": result.metrics.total_elapsed_s,
                    "total_waves": result.metrics.total_waves,
                    "total_findings_stored": result.metrics.total_findings_stored,
                    "total_tool_calls": result.metrics.total_tool_calls,
                    "findings_per_wave": result.metrics.findings_per_wave,
                    "phase_times": result.metrics.phase_times,
                    "convergence_reason": result.metrics.convergence_reason,
                    "angles_detected": result.angles_detected,
                    "report_chars": len(result.report),
                }, f, indent=2)

            print(f"\n{'═' * 60}")
            print(f"  MCP SWARM COMPLETE")
            print(f"{'═' * 60}")
            print(f"  Model:              {default_model}")
            print(f"  Context window:     {caps.context_window:,} tokens")
            print(f"  Queen routing:      {queen_routing}")
            print(f"  Flock:              local (always)")
            print(f"  Workers:            {resolved_workers} (matched to {len(REQUIRED_ANGLE_LABELS)} angles)")
            print(f"  Run:                {resolved_source_run}")
            print(f"  Elapsed:            {result.metrics.total_elapsed_s:.1f}s")
            print(f"  Waves:              {result.metrics.total_waves}")
            print(f"  Findings:           {result.metrics.total_findings_stored}")
            print(f"  Tool calls:         {result.metrics.total_tool_calls}")
            print(f"  Convergence:        {result.metrics.convergence_reason}")
            print(f"  Report:             {len(result.report):,} chars")
            print(f"  Angles:             {result.angles_detected}")
            print(f"  Output:             {output_path.resolve()}")
            print(f"{'═' * 60}\n")

        asyncio.run(_run_mcp())
    else:
        # Original gossip engine
        if store is not None:
            config.lineage_store = store

        if args.multi_model:
            # ── Multi-model pipeline ─────────────────────────────────
            # Phase 1: Start local vLLM instances packed across GPUs,
            #          run bees + gossip + serendipity + queen.
            # Phase 2: Swap to Flock model, run evaluation.
            async def _run_multi_model() -> None:
                logger.info(
                    "local_model=<%s>, flock_model=<%s>, gpus=<%d> | "
                    "starting multi-model pipeline",
                    args.local_model, args.flock_model, args.num_gpus,
                )

                try:
                    # Phase 1: Start local instances (auto-packed per GPU)
                    # Inside try so partially-started processes are cleaned
                    # up if start_vllm_instances raises mid-loop.
                    local_eps = await start_vllm_instances(
                        model=args.local_model,
                        num_gpus=args.num_gpus,
                        max_model_len=131072,
                    )

                    if not local_eps:
                        logger.error("no local vLLM instances started — aborting")
                        return
                    # Probe first instance to discover model name
                    local_caps = await _probe_vllm(local_eps[0])
                    local_model_name = local_caps.model_id

                    # Re-compute token budgets from actual context window.
                    # main() probed vLLM before instances existed and got
                    # fallback 32K budgets.  Now we have the real context
                    # window (e.g. 131K for Kimi-Linear) so budgets must
                    # be resized to avoid severe undersizing.
                    actual_budgets = _decide_token_budgets(local_caps)
                    config.worker_max_tokens = actual_budgets["worker_max_tokens"]
                    config.queen_max_tokens = actual_budgets["queen_max_tokens"]
                    config.max_section_chars = actual_budgets["max_section_chars"]
                    config.context_budget = actual_budgets["context_budget"]

                    # Re-evaluate queen routing with actual capabilities.
                    # main() computed queen_routing using stale 32K fallback
                    # because no vLLM was running yet.  With the real context
                    # window the queen may fit locally.
                    actual_queen_routing, queen_reason = _decide_queen_routing(
                        corpus_chars=len(corpus),
                        worker_count=resolved_workers,
                        caps=local_caps,
                    )
                    logger.info(
                        "queen_routing=<%s>, reason=<%s> | "
                        "queen routing re-evaluated with actual context",
                        actual_queen_routing, queen_reason,
                    )

                    logger.info(
                        "local_instances=<%d>, local_model=<%s>, "
                        "context_window=<%d> | Phase 1 ready "
                        "(budgets resized from actual context)",
                        len(local_eps), local_model_name,
                        local_caps.context_window,
                    )

                    # Run bees + gossip + serendipity + queen
                    result = await run_swarm_test(
                        corpus=corpus,
                        query=query,
                        config=config,
                        output_dir=args.output_dir,
                        queen_routing=actual_queen_routing,
                        resolved_model=local_model_name,
                        local_endpoints=local_eps,
                    )

                    logger.info(
                        "phase1_elapsed=<%.1f>s, worker_summaries=<%d> | "
                        "Phase 1 complete (bees + gossip + queen)",
                        result.metrics.total_elapsed_s,
                        len(result.worker_summaries),
                    )

                    # Phase 2: Swap to Flock model
                    flock_ep = await swap_to_flock_model(
                        model=args.flock_model,
                        num_gpus=args.num_gpus,
                    )

                    if flock_ep:
                        flock_caps = await _probe_vllm(flock_ep)
                        logger.info(
                            "flock_model=<%s>, context_window=<%d> | "
                            "Phase 2 ready — Flock model loaded",
                            flock_caps.model_id,
                            flock_caps.context_window,
                        )
                        # Flock evaluation would run here using flock_ep
                        # (FlockQueryManager integration is a separate step)
                    else:
                        logger.warning(
                            "flock model swap failed — skipping Flock evaluation"
                        )
                finally:
                    stop_vllm_instances()

            asyncio.run(_run_multi_model())
        else:
            asyncio.run(run_swarm_test(
                corpus=corpus,
                query=query,
                config=config,
                output_dir=args.output_dir,
                queen_routing=queen_routing,
                resolved_model=default_model,
            ))

    # ══════════════════════════════════════════════════════════════
    #  POST-RUN: Flush traces and upload to B2
    # ══════════════════════════════════════════════════════════════

    shutdown_tracing()

    if args.upload_b2:
        logger.info(
            "run_id=<%s>, bucket=<%s> | uploading traces and outputs to B2",
            resolved_source_run, args.b2_bucket,
        )
        trace_urls = upload_traces_to_b2(
            trace_dir=args.trace_dir,
            run_id=resolved_source_run,
            bucket_name=args.b2_bucket,
        )
        output_urls = upload_output_dir_to_b2(
            output_dir=args.output_dir,
            run_id=resolved_source_run,
            bucket_name=args.b2_bucket,
        )
        all_urls = trace_urls + output_urls
        if all_urls:
            print(f"\n{'═' * 60}")
            print(f"  B2 UPLOAD COMPLETE")
            print(f"{'═' * 60}")
            for url in all_urls:
                print(f"  {url}")
            print(f"{'═' * 60}\n")
    else:
        logger.info(
            "trace_dir=<%s> | traces saved locally (use --upload-b2 to push to B2)",
            args.trace_dir,
        )


if __name__ == "__main__":
    main()
