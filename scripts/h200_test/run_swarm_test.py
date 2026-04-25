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

    swarm = GossipSwarm(
        complete=worker_fn,
        worker_complete=worker_fn,
        queen_complete=queen_fn,
        serendipity_complete=serendipity_fn,
        config=config,
    )

    # Progress callback
    async def on_event(event: dict) -> None:
        phase = event.get("type", "unknown")
        logger.info("swarm_event=<%s> | %s", phase, json.dumps(event, default=str))

    logger.info(
        "corpus_chars=<%d>, workers=<%d>, gossip_rounds=<%d>, angles=<%d> | "
        "starting swarm test",
        len(corpus), config.max_workers, config.gossip_rounds,
        len(config.required_angles),
    )

    t0 = time.monotonic()
    result = await swarm.synthesize(
        corpus=corpus,
        query=query,
        on_event=on_event,
    )
    elapsed = time.monotonic() - t0

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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1: Probe the environment
    # ══════════════════════════════════════════════════════════════

    api_base = args.api_base or _get_api_base()
    default_model = args.model or _get_model(
        "SWARM_WORKER_MODEL", "huihui-ai/Qwen3.5-32B-abliterated",
    )
    resolved_source_run = args.source_run or f"run_{time.strftime('%Y%m%d_%H%M%S')}"

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

        asyncio.run(run_swarm_test(
            corpus=corpus,
            query=query,
            config=config,
            output_dir=args.output_dir,
            queen_routing=queen_routing,
            resolved_model=default_model,
        ))


if __name__ == "__main__":
    main()
