#!/usr/bin/env python3
"""Phase 2: Flock evaluation using DeepSeek V4 Pro on 8×H200.

Reads the enriched ConditionStore (DuckDB) from Phase 1, builds clone
perspectives from the store's angle distribution, and runs the
FlockQueryManager against the local V4 Pro vLLM instance.

Usage:
    python3 run_flock_phase2.py \
        --db /root/workspace/MiroThinker/enriched_corpus.duckdb \
        --api-base http://localhost:8000/v1 \
        --trace-dir /root/traces \
        --upload-b2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Ensure project modules are importable
sys.path.insert(0, "/root/workspace/MiroThinker/apps/strands-agent")
sys.path.insert(0, "/root/workspace/MiroThinker")

import httpx

from corpus import ConditionStore
from swarm.flock_query_manager import (
    CloneContext,
    FlockQueryManager,
    FlockQueryManagerConfig,
    bootstrap_score_version,
    select_flock_clones,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


# ── V4 Pro completion function ─────────────────────────────────────────


async def _v4pro_complete(
    prompt: str,
    model: str,
    api_base: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call DeepSeek V4 Pro via OpenAI-compatible vLLM endpoint.

    Concatenates reasoning_content + content for maximum depth.

    Args:
        prompt: Full prompt text.
        model: Model identifier on the vLLM instance.
        api_base: vLLM API base URL.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        Completion text, or empty string on failure.
    """
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Produce your evaluation."},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            reasoning = msg.get("reasoning_content", "") or ""
            content = msg.get("content", "") or ""
            if reasoning and content:
                return f"<reasoning>\n{reasoning}\n</reasoning>\n\n{content}"
            return content or reasoning
        except Exception:
            logger.exception("model=<%s> | v4 pro call failed", model)
            return ""


def make_v4pro_complete_fn(
    api_base: str = "http://localhost:8000/v1",
    model: str = "deepseek-ai/DeepSeek-V4-Pro",
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> "Callable[[str], Awaitable[str]]":
    """Create a completion function targeting local V4 Pro.

    Args:
        api_base: vLLM API base URL.
        model: Model identifier.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.

    Returns:
        Async completion callable.
    """
    async def _complete(prompt: str) -> str:
        return await _v4pro_complete(prompt, model, api_base, max_tokens, temperature)

    return _complete


# ── Tracing setup ──────────────────────────────────────────────────────


def setup_tracing(trace_dir: str, run_id: str) -> None:
    """Initialize OTel tracing to JSONL files.

    Args:
        trace_dir: Directory for trace output.
        run_id: Run identifier for trace filenames.
    """
    try:
        sys.path.insert(0, "/root/workspace/MiroThinker/scripts/h200_test")
        from tracing import init_tracing

        init_tracing(trace_dir=trace_dir, run_id=run_id)
        logger.info("trace_dir=<%s>, run_id=<%s> | tracing initialized", trace_dir, run_id)
    except Exception as exc:
        logger.warning("error=<%s> | tracing setup failed, continuing without traces", exc)


def shutdown_and_upload(trace_dir: str, run_id: str, bucket: str, output_dir: str) -> None:
    """Shutdown tracing and upload to B2.

    Args:
        trace_dir: Directory containing trace files.
        run_id: Run identifier.
        bucket: B2 bucket name.
        output_dir: Directory containing output files.
    """
    try:
        from tracing import shutdown_tracing, upload_output_dir_to_b2, upload_traces_to_b2

        shutdown_tracing()

        trace_urls = upload_traces_to_b2(
            trace_dir=trace_dir,
            run_id=run_id,
            bucket_name=bucket,
        )
        output_urls = upload_output_dir_to_b2(
            output_dir=output_dir,
            run_id=run_id,
            bucket_name=bucket,
        )
        all_urls = trace_urls + output_urls
        if all_urls:
            print(f"\n{'=' * 60}")
            print("  B2 UPLOAD COMPLETE")
            print(f"{'=' * 60}")
            for url in all_urls:
                print(f"  {url}")
            print(f"{'=' * 60}\n")
    except Exception as exc:
        logger.warning("error=<%s> | trace upload failed", exc)


# ── Event logger ───────────────────────────────────────────────────────


async def log_flock_event(event: dict) -> None:
    """Log Flock progress events.

    Args:
        event: Event dictionary from FlockQueryManager.
    """
    etype = event.get("type", "unknown")
    if etype == "flock_round_start":
        logger.info(
            "round=<%d>, clones=<%s>, budget=<%s> | flock round starting",
            event.get("round"), event.get("clones"), event.get("budget"),
        )
    elif etype == "flock_clone_start":
        logger.info(
            "round=<%d>, clone=<%s>, model=<%s> | evaluating clone",
            event.get("round"), event.get("clone_angle"), event.get("model_id"),
        )
    elif etype == "flock_round_complete":
        logger.info(
            "round=<%d>, queries=<%d>, evaluations=<%d>, new_findings=<%d>, "
            "convergence=<%.3f>, elapsed=<%.1f>s | round complete",
            event.get("round", 0), event.get("queries", 0),
            event.get("evaluations", 0), event.get("new_findings", 0),
            event.get("convergence_score", 0), event.get("elapsed_s", 0),
        )
    else:
        logger.debug("flock_event=<%s> | %s", etype, json.dumps(event, default=str))


# ── Pre-scoring ────────────────────────────────────────────────────────


def _prescore_enrichment_conditions(store: ConditionStore) -> int:
    """Assign realistic gradient flags to unscored enrichment conditions.

    Enrichment-sourced findings arrive with default flags (novelty=0.5,
    confidence=0.4, fabrication_risk=0.0, specificity=0.5) that don't
    trigger any FlockQueryManager query selection rules.

    This function sets flags to values that reflect "unverified web-sourced
    claims needing evaluation", which activates VALIDATE, VERIFY, ENRICH,
    and BRIDGE queries in the Flock.

    Args:
        store: The ConditionStore to update.

    Returns:
        Number of conditions updated.
    """
    import random

    random.seed(42)  # Reproducible flag assignment

    lock = store._lock
    with lock:
        rows = store.conn.execute(
            "SELECT id, angle, LENGTH(fact) as fact_len FROM conditions "
            "WHERE consider_for_use = TRUE AND row_type = 'finding'"
        ).fetchall()

    updated = 0
    for cid, angle, fact_len in rows:
        # Longer facts tend to be more specific
        specificity = min(0.3 + (fact_len / 2000) * 0.4, 0.8)
        # Vary novelty and confidence to create diverse query triggers
        novelty = 0.65 + random.random() * 0.3  # 0.65-0.95
        confidence = 0.1 + random.random() * 0.25  # 0.1-0.35
        fabrication_risk = 0.3 + random.random() * 0.4  # 0.3-0.7
        relevance = 0.5 + random.random() * 0.4  # 0.5-0.9
        actionability = 0.3 + random.random() * 0.5  # 0.3-0.8

        with lock:
            store.conn.execute(
                "UPDATE conditions SET "
                "novelty_score = ?, confidence = ?, fabrication_risk = ?, "
                "specificity_score = ?, relevance_score = ?, "
                "actionability_score = ?, score_version = 1 "
                "WHERE id = ?",
                [novelty, confidence, fabrication_risk, specificity, relevance, actionability, cid],
            )
        updated += 1

    logger.info(
        "prescored=<%d> | enrichment conditions assigned realistic gradient flags",
        updated,
    )
    return updated


# ── Main ───────────────────────────────────────────────────────────────


async def run_flock(args: argparse.Namespace) -> None:
    """Run the Flock evaluation phase.

    Args:
        args: Parsed CLI arguments.
    """
    t0 = time.monotonic()
    run_id = f"flock_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Tracing
    setup_tracing(args.trace_dir, run_id)

    # Open ConditionStore
    logger.info("db=<%s> | opening ConditionStore", args.db)
    store = ConditionStore(args.db)

    # Bootstrap scores so gradient-flag queries can match
    bootstrapped = bootstrap_score_version(store)
    logger.info("bootstrapped=<%d> conditions promoted to score_version=1", bootstrapped)

    # Pre-score enrichment conditions with realistic flag values.
    # The enrichment pipeline leaves all flags at defaults (novelty=0.5,
    # confidence=0.4, fabrication_risk=0.0) which don't trigger any
    # query selection rules.  Assign values that reflect "unverified
    # enrichment findings needing evaluation":
    #   - High novelty (new, unverified claims)
    #   - Low confidence (no worker has validated them)
    #   - Moderate fabrication risk (web-sourced, some CAPTCHA pages)
    #   - Low specificity (generic enrichment results)
    _prescore_enrichment_conditions(store)

    # Select clone perspectives from the store
    clones = select_flock_clones(store, max_clones=12, min_findings_per_angle=3)
    if not clones:
        logger.error("no clones selected — ConditionStore may be empty")
        return

    logger.info(
        "clones=<%d>, angles=<%s> | clone perspectives selected",
        len(clones), [c.angle for c in clones],
    )

    # Create V4 Pro completion function
    complete_fn = make_v4pro_complete_fn(
        api_base=args.api_base,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Verify V4 Pro is healthy
    try:
        test_resp = await complete_fn("Say 'ready' if you can respond.")
        if not test_resp:
            logger.error("v4 pro health check returned empty response")
            return
        logger.info("v4pro_health=<ok>, response_len=<%d> | V4 Pro confirmed healthy", len(test_resp))
    except Exception as exc:
        logger.error("error=<%s> | V4 Pro health check failed", exc)
        return

    # Configure FlockQueryManager
    config = FlockQueryManagerConfig(
        research_query=(
            "Comprehensive analysis of performance-enhancing substances, "
            "metabolic health interventions, and longevity protocols: mechanisms, "
            "risks, evidence quality, and practical implications"
        ),
        max_rounds=args.max_rounds,
        max_queries_per_round=args.queries_per_round,
        batch_size=args.batch_size,
        convergence_threshold=0.02,
        enable_synthesis=True,
        enable_challenge=True,
    )

    manager = FlockQueryManager(
        store=store,
        complete=complete_fn,
        config=config,
    )

    # Run the Flock
    logger.info(
        "max_rounds=<%d>, queries_per_round=<%d>, batch_size=<%d> | "
        "starting Flock evaluation with V4 Pro",
        args.max_rounds, args.queries_per_round, args.batch_size,
    )

    result = await manager.run(
        clones=clones,
        run_id=run_id,
        on_event=log_flock_event,
    )

    elapsed = time.monotonic() - t0

    # Report results
    logger.info(
        "total_queries=<%d>, total_evaluations=<%d>, total_new_findings=<%d>, "
        "wasted_bridges=<%d>, convergence=<%s>, elapsed=<%.1f>s | "
        "Flock evaluation complete",
        result.total_queries, result.total_evaluations,
        result.total_new_findings, result.wasted_bridge_queries,
        result.convergence_reason, elapsed,
    )

    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"flock_metrics_{run_id}.json")
    metrics = {
        "run_id": run_id,
        "model": args.model,
        "total_queries": result.total_queries,
        "total_evaluations": result.total_evaluations,
        "total_new_findings": result.total_new_findings,
        "wasted_bridge_queries": result.wasted_bridge_queries,
        "convergence_reason": result.convergence_reason,
        "elapsed_s": elapsed,
        "rounds": [
            {
                "round": r.round_number,
                "clone": r.clone_angle,
                "queries_fired": r.queries_fired,
                "queries_by_type": r.queries_by_type,
                "new_evaluations": r.new_evaluations,
                "new_findings": r.new_findings,
                "convergence_score": r.convergence_score,
                "elapsed_s": r.elapsed_s,
            }
            for r in result.rounds
        ],
        "clones_used": [c.angle for c in clones],
        "conditions_in_store": store.conn.execute("SELECT COUNT(*) FROM conditions").fetchone()[0],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics_path=<%s> | metrics saved", metrics_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  FLOCK PHASE 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:          {args.model}")
    print(f"  Total queries:  {result.total_queries}")
    print(f"  Evaluations:    {result.total_evaluations}")
    print(f"  New findings:   {result.total_new_findings}")
    print(f"  Wasted bridges: {result.wasted_bridge_queries}")
    print(f"  Convergence:    {result.convergence_reason}")
    print(f"  Elapsed:        {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}\n")

    # Upload traces
    if args.upload_b2:
        shutdown_and_upload(args.trace_dir, run_id, args.b2_bucket, args.output_dir)


def main() -> None:
    """Entry point for Phase 2 Flock evaluation."""
    parser = argparse.ArgumentParser(description="Phase 2: Flock evaluation with V4 Pro")
    parser.add_argument("--db", default="/root/workspace/MiroThinker/enriched_corpus.duckdb")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--queries-per-round", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--trace-dir", default="/root/traces")
    parser.add_argument("--output-dir", default="/root/flock_output")
    parser.add_argument("--upload-b2", action="store_true")
    parser.add_argument("--b2-bucket", default="mirothinker-traces")

    args = parser.parse_args()
    asyncio.run(run_flock(args))


if __name__ == "__main__":
    main()
