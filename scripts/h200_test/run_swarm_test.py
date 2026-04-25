#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Run a full swarm test on 1×H200 with enriched corpus.

This is the main test runner. It:
1. Loads or creates a ConditionStore (optionally pre-enriched)
2. Configures swarm angles for the 8-compound ramping protocol
3. Connects to the local vLLM endpoint
4. Runs the full gossip swarm pipeline
5. Outputs both user_report and knowledge_report

Usage:
    # Start vLLM first (in another terminal):
    ./launch_vllm.sh huihui-ai/Qwen3.5-32B-abliterated

    # Then run the swarm:
    python run_swarm_test.py --corpus existing_corpus.txt
    python run_swarm_test.py --db enriched.duckdb
    python run_swarm_test.py --enrich  # enrich + run in one go

Environment variables:
    SWARM_API_BASE          — vLLM endpoint (default: http://localhost:8000/v1)
    SWARM_WORKER_MODEL      — model name served by vLLM
    SWARM_QUEEN_MODEL       — queen model (same as worker for single-GPU test)
    SWARM_SERENDIPITY_MODEL — serendipity model (same as worker for single-GPU)
    SWARM_MAX_WORKERS       — number of concurrent workers (default: 8)
    SWARM_GOSSIP_ROUNDS     — gossip rounds (default: 3)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run gossip swarm test on 1×H200",
    )
    parser.add_argument(
        "--corpus", default="",
        help="Path to corpus text file (auto-discovers mega_corpus.txt or youtube_corpus if empty)",
    )
    parser.add_argument(
        "--db", default="",
        help="Path to DuckDB database with enriched corpus",
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Run enrichment pipeline before swarm (auto-enabled when angles have enrichment_queries)",
    )
    parser.add_argument(
        "--skip-enrich", action="store_true",
        help="Skip auto-enrichment even when angles have enrichment_queries",
    )
    parser.add_argument(
        "--min-corpus-chars", type=int, default=500000,
        help="Minimum corpus size in chars — warns if below this (default: 500000)",
    )
    parser.add_argument(
        "--output-dir", default="swarm_results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Override max workers (0 = use env/default)",
    )
    parser.add_argument(
        "--gossip-rounds", type=int, default=0,
        help="Override gossip rounds (0 = use env/default)",
    )
    parser.add_argument(
        "--query", default="",
        help="Override the default swarm query",
    )
    parser.add_argument(
        "--engine", choices=["gossip", "mcp"], default="gossip",
        help="Swarm engine: 'gossip' (original) or 'mcp' (agent-workers with tools)",
    )
    parser.add_argument(
        "--waves", type=int, default=3,
        help="Max worker waves for MCP engine (default: 3)",
    )
    parser.add_argument(
        "--model", default="",
        help="Model name served by vLLM (overrides SWARM_WORKER_MODEL env var)",
    )
    parser.add_argument(
        "--api-base", default="",
        help="vLLM API base URL (overrides SWARM_API_BASE env var)",
    )
    parser.add_argument(
        "--api-key", default="not-needed",
        help="API key for endpoint (default: not-needed for local vLLM)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Max tokens per worker LLM response (default: 4096)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature for workers (default: 0.3)",
    )
    parser.add_argument(
        "--convergence-threshold", type=int, default=5,
        help="Stop when new findings per wave drops below this (default: 5)",
    )
    parser.add_argument(
        "--report-max-tokens", type=int, default=8192,
        help="Max tokens for final report generation (default: 8192)",
    )
    parser.add_argument(
        "--no-serendipity", action="store_true",
        help="Disable the serendipity cross-domain wave",
    )
    parser.add_argument(
        "--source-model", default="",
        help="Model name for provenance tracking in store",
    )
    parser.add_argument(
        "--source-run", default="",
        help="Run identifier for provenance (e.g. run_042)",
    )
    parser.add_argument(
        "--max-return-chars", type=int, default=6000,
        help="Hard ceiling on chars any tool call returns (default: 6000)",
    )
    parser.add_argument(
        "--compact-every", type=int, default=3,
        help="Run store compaction every N waves (0 = disable, default: 3)",
    )
    parser.add_argument(
        "--no-rolling-summaries", action="store_true",
        help="Disable rolling knowledge summaries between waves",
    )
    parser.add_argument(
        "--report-max-chars", type=int, default=24000,
        help="Max prompt chars for report generation (default: 24000)",
    )
    parser.add_argument(
        "--flock-backend", default="local",
        choices=["local", "ling-free", "ling-flash-free", "deepseek-flash", "deepseek-pro"],
        help=(
            "Backend for the Flock evaluation phase. "
            "'local' uses the same vLLM endpoint as workers. "
            "'ling-free' uses Ling 2.6-1T via OpenRouter free tier (HIGH RISK). "
            "'ling-flash-free' uses Ling 2.6-Flash via OpenRouter free tier. "
            "'deepseek-flash' uses DeepSeek V4-Flash via OpenRouter (paid). "
            "'deepseek-pro' uses DeepSeek V4-Pro via OpenRouter (paid). "
            "Default: local"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    # Build SwarmConfig
    config = SwarmConfig()
    config.required_angles = list(REQUIRED_ANGLE_LABELS)
    config.enable_serendipity = True
    config.enable_full_corpus_gossip = True
    config.enable_semantic_assignment = True
    config.enable_misassignment = True
    config.misassignment_ratio = 0.25
    config.enable_hive_memory = True
    config.enable_diversity_aware_gossip = True

    if args.workers > 0:
        config.max_workers = args.workers
    if args.gossip_rounds > 0:
        config.gossip_rounds = args.gossip_rounds

    # ── Corpus loading with auto-discovery and validation ──────────
    corpus = ""
    store = None

    # Auto-discover corpus if --corpus not provided
    corpus_path = args.corpus
    if not corpus_path:
        _candidates = [
            Path(__file__).resolve().parent / "mega_corpus.txt",
            Path.home() / "data" / "youtube_corpus_combined.txt",
            Path.home() / "repos" / "MiroThinker" / "scripts" / "h200_test" / "mega_corpus.txt",
        ]
        for candidate in _candidates:
            if candidate.exists():
                corpus_path = str(candidate)
                logger.info(
                    "corpus_path=<%s> | auto-discovered corpus file",
                    corpus_path,
                )
                break

    if args.enrich or args.db:
        db_path = args.db or "enriched_corpus.duckdb"
        store = ConditionStore(db_path=db_path)

        if args.enrich:
            from enrich_corpus import enrich_all
            logger.info("running enrichment pipeline...")
            enrich_all(store, max_per_query=10, extract_full_text=True)

        corpus = load_corpus_from_store(store)
        logger.info("corpus_chars=<%d> | loaded from store", len(corpus))

    if corpus_path:
        file_corpus = load_corpus_from_file(corpus_path)
        if corpus:
            corpus = f"{corpus}\n\n{file_corpus}"
        else:
            corpus = file_corpus
        logger.info("corpus_chars=<%d> | loaded from file %s", len(corpus), corpus_path)

    # Auto-enrichment: if angles have enrichment_queries and user didn't
    # explicitly pass --enrich or --skip-enrich, auto-enrich
    if not args.enrich and not args.skip_enrich and not args.db:
        try:
            from angles import ALL_ANGLES
            total_eq = sum(len(a.enrichment_queries) for a in ALL_ANGLES)
            if total_eq > 0:
                logger.info(
                    "enrichment_queries=<%d> | auto-enriching (use --skip-enrich to disable)",
                    total_eq,
                )
                from enrich_corpus import enrich_all
                if store is None:
                    store = ConditionStore(db_path="enriched_corpus.duckdb")
                enrich_all(store, max_per_query=10, extract_full_text=True)
                enriched = load_corpus_from_store(store)
                if enriched:
                    corpus = f"{corpus}\n\n{enriched}" if corpus else enriched
                    logger.info(
                        "enriched_chars=<%d>, total_corpus=<%d> | auto-enrichment complete",
                        len(enriched), len(corpus),
                    )
        except ImportError:
            logger.debug("angles module not available for auto-enrichment check")

    if not corpus:
        logger.error("no corpus provided — use --corpus, --db, or --enrich")
        sys.exit(1)

    # Corpus size validation
    if len(corpus) < 50000:
        logger.error(
            "corpus_chars=<%d> | corpus dangerously small (<50K chars) — "
            "model WILL hallucinate. provide a larger corpus",
            len(corpus),
        )
        sys.exit(1)
    elif len(corpus) < args.min_corpus_chars:
        logger.warning(
            "corpus_chars=<%d>, min_corpus_chars=<%d> | corpus below recommended minimum — "
            "output quality may be poor. use --min-corpus-chars to adjust threshold",
            len(corpus), args.min_corpus_chars,
        )

    logger.info(
        "corpus_chars=<%d>, sections_est=<%d> | corpus loaded and validated",
        len(corpus),
        len(corpus) // config.max_section_chars + 1,
    )

    # Wire corpus_delta_fn for external research during gossip rounds
    if store is not None:
        async def _corpus_delta() -> str:
            """Fetch new findings added to the store since last check."""
            new_findings = store.get_findings(limit=500)
            if not new_findings:
                return ""
            return load_corpus_from_store(store)

        config.corpus_delta_fn = _corpus_delta

    query = args.query or get_swarm_query()

    if args.engine == "mcp":
        # MCP engine: agent-workers with ConditionStore tools
        from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine

        if store is None:
            store = ConditionStore(db_path="mcp_swarm.duckdb")

        api_base = args.api_base or _get_api_base()
        default_model = args.model or _get_model(
            "SWARM_WORKER_MODEL", "huihui-ai/Qwen3.5-32B-abliterated",
        )

        # Resolve source_model from flag or model name
        resolved_source_model = args.source_model or default_model
        resolved_source_run = args.source_run or f"run_{time.strftime('%Y%m%d_%H%M%S')}"

        # Resolve Flock backend (local vLLM or remote API with risk tier)
        _flock_fn, flock_backend_config = resolve_flock_backend(
            choice=args.flock_backend,
            local_model=default_model,
            local_api_base=api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        mcp_config = MCPSwarmConfig(
            max_workers=config.max_workers,
            max_waves=args.waves,
            convergence_threshold=args.convergence_threshold,
            api_base=api_base,
            model=default_model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            required_angles=list(REQUIRED_ANGLE_LABELS),
            report_max_tokens=args.report_max_tokens,
            enable_serendipity_wave=not args.no_serendipity,
            source_model=resolved_source_model,
            source_run=resolved_source_run,
            max_return_chars=args.max_return_chars,
            compact_every_n_waves=args.compact_every,
            enable_rolling_summaries=not args.no_rolling_summaries,
            report_max_chars=args.report_max_chars,
            flock_backend_config=flock_backend_config,
            flock_complete=_flock_fn,
        )

        # The MCP engine needs a simple completion function for
        # angle detection and report generation (non-agent calls)
        complete_fn = make_complete_fn(
            default_model, api_base,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
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
            # Save results
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
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "convergence_threshold": args.convergence_threshold,
                    "max_workers": config.max_workers,
                    "max_waves": args.waves,
                    "report_max_tokens": args.report_max_tokens,
                    "serendipity_enabled": not args.no_serendipity,
                    "source_model": resolved_source_model,
                    "source_run": resolved_source_run,
                    "max_return_chars": args.max_return_chars,
                    "compact_every_n_waves": args.compact_every,
                    "rolling_summaries_enabled": not args.no_rolling_summaries,
                    "report_max_chars": args.report_max_chars,
                    "flock_backend": args.flock_backend,
                    "flock_backend_model": (
                        flock_backend_config.model if flock_backend_config else default_model
                    ),
                    "flock_backend_tier": (
                        flock_backend_config.risk_tier.value if flock_backend_config else "self_hosted"
                    ),
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
            print(f"  MCP SWARM TEST COMPLETE")
            print(f"{'═' * 60}")
            print(f"  Model:              {default_model}")
            print(f"  Flock backend:      {args.flock_backend}")
            if flock_backend_config:
                print(f"  Flock model:        {flock_backend_config.model}")
                print(f"  Flock risk tier:    {flock_backend_config.risk_tier.value}")
            print(f"  Source model:       {resolved_source_model}")
            print(f"  Source run:         {resolved_source_run}")
            print(f"  API base:           {api_base}")
            print(f"  Temperature:        {args.temperature}")
            print(f"  Max tokens:         {args.max_tokens}")
            print(f"  Max return chars:   {args.max_return_chars}")
            print(f"  Report max chars:   {args.report_max_chars}")
            print(f"  Convergence:        {args.convergence_threshold}")
            print(f"  Serendipity:        {not args.no_serendipity}")
            print(f"  Rolling summaries:  {not args.no_rolling_summaries}")
            print(f"  Compact every:      {args.compact_every} waves")
            print(f"  Elapsed:            {result.metrics.total_elapsed_s:.1f}s")
            print(f"  Waves:              {result.metrics.total_waves}")
            print(f"  Findings stored:    {result.metrics.total_findings_stored}")
            print(f"  Tool calls:         {result.metrics.total_tool_calls}")
            print(f"  Findings/wave:      {result.metrics.findings_per_wave}")
            print(f"  Convergence:        {result.metrics.convergence_reason}")
            print(f"  Report:             {len(result.report):,} chars")
            print(f"  Angles:             {result.angles_detected}")
            print(f"\n  Output directory:   {output_path.resolve()}")
            print(f"  Report:             {report_path.name}")
            print(f"  Metrics:            {metrics_path.name}")
            print(f"{'═' * 60}\n")

        asyncio.run(_run_mcp())
    else:
        # Original gossip engine
        # Use LineageStore backed by ConditionStore if available
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
