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
from pathlib import Path
from typing import Awaitable, Callable

# Ensure repo root is importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_STRANDS_AGENT = str(Path(__file__).resolve().parents[2] / "apps" / "strands-agent")
if _STRANDS_AGENT not in sys.path:
    sys.path.insert(0, _STRANDS_AGENT)

from angles import ALL_ANGLES, REQUIRED_ANGLE_LABELS, get_swarm_query
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

async def run_swarm_test(
    corpus: str,
    query: str,
    config: SwarmConfig,
    output_dir: str = ".",
) -> SwarmResult:
    """Run the full gossip swarm pipeline and save results."""
    # Default model name — user sets SWARM_WORKER_MODEL to match what vLLM serves
    default_model = "huihui-ai/Qwen3.5-32B-abliterated"

    api_base = _get_api_base()
    worker_model = _get_model("SWARM_WORKER_MODEL", default_model)
    queen_model = _get_model("SWARM_QUEEN_MODEL", default_model)
    serendipity_model = _get_model("SWARM_SERENDIPITY_MODEL", default_model)

    worker_fn = make_complete_fn(
        worker_model, api_base,
        max_tokens=config.worker_max_tokens,
        temperature=config.worker_temperature,
    )
    # Route queen to OpenRouter when OPENROUTER_API_KEY is set AND the queen
    # model differs from the worker model.  The '/' heuristic alone cannot
    # distinguish HuggingFace names (huihui-ai/Qwen3.5-32B-abliterated) from
    # OpenRouter identifiers (deepseek/deepseek-r1) — both use org/model format.
    _has_openrouter_key = bool(os.environ.get("OPENROUTER_API_KEY", ""))
    if _has_openrouter_key and queen_model != worker_model:
        queen_fn = make_openrouter_complete_fn(
            queen_model,
            max_tokens=config.queen_max_tokens,
            temperature=config.queen_temperature,
        )
        logger.info(
            "queen_model=<%s> | queen routed to OpenRouter (OPENROUTER_API_KEY set)",
            queen_model,
        )
    else:
        queen_fn = make_complete_fn(
            queen_model, api_base,
            max_tokens=config.queen_max_tokens,
            temperature=config.queen_temperature,
        )
        if queen_model != worker_model and not _has_openrouter_key:
            logger.warning(
                "queen_model=<%s> | queen differs from worker but no OPENROUTER_API_KEY — using local vLLM",
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


# ── Smart runtime decisions ──────────────────────────────────────────
#
# The pipeline makes its own decisions about configuration based on what
# it discovers at runtime.  The operator provides only meaningful choices
# (what data, what query, where to write output).  Everything else is
# determined by inspecting the environment, the angles, and the corpus.

_CORPUS_MIN_HARD = 50_000      # below this → ERROR + exit (hallucination zone)
_CORPUS_MIN_SOFT = 500_000     # below this → WARNING (quality risk)
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_REPORT_MAX_TOKENS = 8192
_DEFAULT_REPORT_MAX_CHARS = 24_000
_DEFAULT_MAX_RETURN_CHARS = 6_000
_DEFAULT_COMPACT_EVERY = 3
_DEFAULT_CONVERGENCE_THRESHOLD = 5
_DEFAULT_MAX_WAVES = 3


def _auto_discover_corpus() -> str:
    """Find the best corpus file available on this machine.

    Searches standard locations in priority order, returns the path
    to the first file found, or empty string if none.
    """
    candidates = [
        Path(__file__).resolve().parent / "mega_corpus.txt",
        Path.home() / "data" / "youtube_corpus_combined.txt",
        Path.home() / "repos" / "MiroThinker" / "scripts" / "h200_test" / "mega_corpus.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info(
                "corpus_path=<%s>, size=<%d> | auto-discovered corpus file",
                candidate, candidate.stat().st_size,
            )
            return str(candidate)
    return ""


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
            "Run the gossip swarm pipeline.  Most configuration is resolved "
            "automatically — workers match angle count, enrichment fires when "
            "angles define queries, Flock always uses local vLLM, queen routes "
            "to OpenRouter when OPENROUTER_API_KEY is set."
        ),
    )
    # ── Meaningful operator choices only ──────────────────────────
    parser.add_argument(
        "--corpus", default="",
        help="Path to corpus file (auto-discovers if omitted)",
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

    # ── Resolve all configuration from environment + angles ───────
    api_base = args.api_base or _get_api_base()
    default_model = args.model or _get_model(
        "SWARM_WORKER_MODEL", "huihui-ai/Qwen3.5-32B-abliterated",
    )
    resolved_source_run = args.source_run or f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    # Workers match angle count
    resolved_workers = _auto_workers()

    # Gossip rounds from env or default
    resolved_gossip_rounds = int(os.environ.get("SWARM_GOSSIP_ROUNDS", "3"))

    # Build SwarmConfig — all features enabled, no flags to disable them
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

    logger.info(
        "workers=<%d>, gossip_rounds=<%d>, angles=<%d>, engine=<%s> | "
        "swarm config resolved",
        resolved_workers, resolved_gossip_rounds,
        len(REQUIRED_ANGLE_LABELS), args.engine,
    )

    # ── Corpus: discover → load → enrich → validate ──────────────
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

    # Step 3: Auto-enrich when angles define enrichment queries.
    # Use export_delta to capture only NEW findings from enrichment —
    # avoids duplicating pre-existing DB findings already in corpus.
    enrichment_count = _should_enrich()
    if enrichment_count > 0:
        from datetime import datetime, timezone
        from enrich_corpus import enrich_all
        if store is None:
            store = ConditionStore(db_path="enriched_corpus.duckdb")
        pre_enrich_ts = datetime.now(timezone.utc).isoformat()
        enrich_all(store, max_per_query=10, extract_full_text=True)
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

    # Step 4: Validate corpus size
    _validate_corpus(corpus)

    logger.info(
        "corpus_chars=<%d>, sections_est=<%d> | corpus ready",
        len(corpus),
        len(corpus) // config.max_section_chars + 1,
    )

    # ── Wire corpus_delta_fn for gossip rounds ───────────────────
    # Uses a watermark so only NEW findings since last check are returned.
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

    # ── Run the selected engine ──────────────────────────────────
    if args.engine == "mcp":
        from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine

        if store is None:
            store = ConditionStore(db_path="mcp_swarm.duckdb")

        # Flock always uses local vLLM — it fires thousands of queries,
        # remote APIs can't scale and would cost a fortune
        _flock_fn, flock_backend_config = resolve_flock_backend(
            choice="local",
            local_model=default_model,
            local_api_base=api_base,
            max_tokens=_DEFAULT_MAX_TOKENS,
            temperature=_DEFAULT_TEMPERATURE,
        )

        mcp_config = MCPSwarmConfig(
            max_workers=resolved_workers,
            max_waves=_DEFAULT_MAX_WAVES,
            convergence_threshold=_DEFAULT_CONVERGENCE_THRESHOLD,
            api_base=api_base,
            model=default_model,
            api_key="not-needed",
            max_tokens=_DEFAULT_MAX_TOKENS,
            temperature=_DEFAULT_TEMPERATURE,
            required_angles=list(REQUIRED_ANGLE_LABELS),
            report_max_tokens=_DEFAULT_REPORT_MAX_TOKENS,
            enable_serendipity_wave=True,
            source_model=default_model,
            source_run=resolved_source_run,
            max_return_chars=_DEFAULT_MAX_RETURN_CHARS,
            compact_every_n_waves=_DEFAULT_COMPACT_EVERY,
            enable_rolling_summaries=True,
            report_max_chars=_DEFAULT_REPORT_MAX_CHARS,
            flock_backend_config=flock_backend_config,
            flock_complete=_flock_fn,
        )

        complete_fn = make_complete_fn(
            default_model, api_base,
            max_tokens=_DEFAULT_MAX_TOKENS,
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
                    "max_tokens": _DEFAULT_MAX_TOKENS,
                    "temperature": _DEFAULT_TEMPERATURE,
                    "convergence_threshold": _DEFAULT_CONVERGENCE_THRESHOLD,
                    "max_workers": resolved_workers,
                    "max_waves": _DEFAULT_MAX_WAVES,
                    "report_max_tokens": _DEFAULT_REPORT_MAX_TOKENS,
                    "serendipity_enabled": True,
                    "source_model": default_model,
                    "source_run": resolved_source_run,
                    "flock_backend": "local",
                    "flock_backend_model": default_model,
                    "flock_backend_tier": "self_hosted",
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
        ))


if __name__ == "__main__":
    main()
