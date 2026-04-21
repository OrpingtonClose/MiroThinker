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
import re
import sys
import time
from pathlib import Path

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


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from model output."""
    return _THINK_RE.sub("", text).strip()


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
            raw = data["choices"][0]["message"]["content"]
            return _strip_thinking(raw)
        except Exception:
            logger.exception(
                "model=<%s>, url=<%s> | vLLM call failed", model, url,
            )
            return ""


def make_complete_fn(
    model_env: str,
    default_model: str,
    max_tokens: int = 16384,
    temperature: float = 0.3,
):
    """Create a CompleteFn bound to a specific model and endpoint."""
    api_base = _get_api_base()
    model = _get_model(model_env, default_model)

    async def _complete(prompt: str) -> str:
        return await _vllm_complete(
            prompt, model, api_base, max_tokens, temperature,
        )

    return _complete


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

    worker_fn = make_complete_fn(
        "SWARM_WORKER_MODEL", default_model,
        max_tokens=config.worker_max_tokens,
        temperature=config.worker_temperature,
    )
    queen_fn = make_complete_fn(
        "SWARM_QUEEN_MODEL", default_model,
        max_tokens=config.queen_max_tokens,
        temperature=config.queen_temperature,
    )
    serendipity_fn = make_complete_fn(
        "SWARM_SERENDIPITY_MODEL", default_model,
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
        help="Path to corpus text file",
    )
    parser.add_argument(
        "--db", default="",
        help="Path to DuckDB database with enriched corpus",
    )
    parser.add_argument(
        "--enrich", action="store_true",
        help="Run enrichment pipeline before swarm",
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

    # Cap max gossip rounds to prevent context overflow in queen merge
    # with 32K context models. 5 rounds is enough for convergence.
    config.max_gossip_rounds = int(os.environ.get("SWARM_MAX_GOSSIP_ROUNDS", "5"))

    if args.workers > 0:
        config.max_workers = args.workers
    if args.gossip_rounds > 0:
        config.gossip_rounds = args.gossip_rounds

    # Load corpus
    corpus = ""
    store = None

    if args.enrich or args.db:
        db_path = args.db or "enriched_corpus.duckdb"
        store = ConditionStore(db_path=db_path)

        if args.enrich:
            from enrich_corpus import enrich_all
            logger.info("running enrichment pipeline...")
            enrich_all(store, max_per_query=10)

        corpus = load_corpus_from_store(store)
        logger.info("corpus_chars=<%d> | loaded from store", len(corpus))

    if args.corpus:
        file_corpus = load_corpus_from_file(args.corpus)
        if corpus:
            corpus = f"{corpus}\n\n{file_corpus}"
        else:
            corpus = file_corpus
        logger.info("corpus_chars=<%d> | loaded from file", len(corpus))

    if not corpus:
        logger.error("no corpus provided — use --corpus, --db, or --enrich")
        sys.exit(1)

    # Use LineageStore backed by ConditionStore if available
    if store is not None:
        config.lineage_store = store

    query = args.query or get_swarm_query()

    asyncio.run(run_swarm_test(
        corpus=corpus,
        query=query,
        config=config,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
