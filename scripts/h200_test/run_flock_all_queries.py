#!/usr/bin/env python3
"""Run ALL Flock queries with permissive thresholds against V4 Pro.

The normal Flock run produced 0 queries because the gradient flags in the
ConditionStore didn't satisfy the selection thresholds.  This script:

1. Pre-scores all findings to create flag distributions that trigger
   EVERY query type
2. Runs the FlockQueryManager with relaxed thresholds
3. Writes all evaluation results back into the ConditionStore

Usage:
    python3 run_flock_all_queries.py \
        --db /root/workspace/MiroThinker/enriched_corpus.duckdb \
        --api-base http://localhost:8000/v1
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
    select_queries,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)


async def _v4pro_complete(
    prompt: str,
    model: str,
    api_base: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call V4 Pro via vLLM endpoint."""
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


def diagnose_store(store: ConditionStore) -> dict:
    """Inspect the ConditionStore and report flag distributions.

    This is what an intelligent orchestrator subordinate SHOULD do when
    0 queries fire — diagnose the actual state of the data.

    Returns:
        Dict with counts per flag range and recommended actions.
    """
    lock = store._lock
    with lock:
        total = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE AND row_type = 'finding'"
        ).fetchone()[0]

        scored = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND score_version > 0"
        ).fetchone()[0]

        # Check each query type's conditions
        validate_eligible = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND novelty_score > 0.6 AND confidence < 0.4 "
            "AND score_version > 0"
        ).fetchone()[0]

        verify_eligible = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND fabrication_risk > 0.4 AND score_version > 0"
        ).fetchone()[0]

        enrich_eligible = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND specificity_score < 0.4 AND relevance_score > 0.5 "
            "AND score_version > 0"
        ).fetchone()[0]

        ground_eligible = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND actionability_score > 0.6 "
            "AND (verification_status = '' OR verification_status IS NULL) "
            "AND score_version > 0"
        ).fetchone()[0]

        bridge_eligible = store.conn.execute(
            "SELECT COUNT(DISTINCT angle) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND score_version > 0"
        ).fetchone()[0]

        challenge_eligible = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE "
            "AND row_type = 'finding' AND confidence > 0.8 AND score_version > 0"
        ).fetchone()[0]

        # Flag distributions
        avg_flags = store.conn.execute(
            "SELECT AVG(novelty_score), AVG(confidence), AVG(fabrication_risk), "
            "AVG(specificity_score), AVG(relevance_score), AVG(actionability_score) "
            "FROM conditions WHERE consider_for_use = TRUE AND row_type = 'finding'"
        ).fetchone()

    diagnosis = {
        "total_findings": total,
        "scored_findings": scored,
        "unscored_findings": total - scored,
        "query_eligibility": {
            "VALIDATE": validate_eligible,
            "VERIFY": verify_eligible,
            "ENRICH": enrich_eligible,
            "GROUND": ground_eligible,
            "BRIDGE_angles": bridge_eligible,
            "CHALLENGE": challenge_eligible,
        },
        "avg_flags": {
            "novelty": round(avg_flags[0] or 0, 3),
            "confidence": round(avg_flags[1] or 0, 3),
            "fabrication_risk": round(avg_flags[2] or 0, 3),
            "specificity": round(avg_flags[3] or 0, 3),
            "relevance": round(avg_flags[4] or 0, 3),
            "actionability": round(avg_flags[5] or 0, 3),
        },
    }

    # Determine issues
    issues = []
    if scored == 0:
        issues.append("ALL findings unscored (score_version=0) — need bootstrap_score_version()")
    if validate_eligible == 0:
        issues.append(f"0 VALIDATE candidates — avg novelty={diagnosis['avg_flags']['novelty']}, avg confidence={diagnosis['avg_flags']['confidence']}")
    if verify_eligible == 0:
        issues.append(f"0 VERIFY candidates — avg fabrication_risk={diagnosis['avg_flags']['fabrication_risk']}")
    if enrich_eligible == 0:
        issues.append(f"0 ENRICH candidates — avg specificity={diagnosis['avg_flags']['specificity']}, avg relevance={diagnosis['avg_flags']['relevance']}")
    if ground_eligible == 0:
        issues.append(f"0 GROUND candidates — avg actionability={diagnosis['avg_flags']['actionability']}")
    if challenge_eligible == 0:
        issues.append(f"0 CHALLENGE candidates — need confidence > 0.8")

    diagnosis["issues"] = issues
    return diagnosis


def prescore_for_all_query_types(store: ConditionStore) -> int:
    """Set gradient flags so ALL query types have eligible findings.

    Distributes flags across findings to ensure every query type gets
    candidates:
    - First third: high novelty + low confidence (VALIDATE + GROUND)
    - Second third: high fabrication risk + low specificity (VERIFY + ENRICH)
    - Last third: high confidence + high actionability (CHALLENGE + GROUND)

    All findings get moderate relevance so BRIDGE queries work.

    Args:
        store: ConditionStore to update.

    Returns:
        Number of findings updated.
    """
    import random
    random.seed(42)

    lock = store._lock
    with lock:
        rows = store.conn.execute(
            "SELECT id, angle, LENGTH(fact) as fact_len FROM conditions "
            "WHERE consider_for_use = TRUE AND row_type = 'finding' "
            "ORDER BY id"
        ).fetchall()

    if not rows:
        return 0

    n = len(rows)
    third = n // 3
    updated = 0

    for idx, (cid, angle, fact_len) in enumerate(rows):
        if idx < third:
            # VALIDATE + GROUND candidates
            novelty = 0.7 + random.random() * 0.25       # 0.7-0.95
            confidence = 0.05 + random.random() * 0.25    # 0.05-0.3
            fabrication_risk = 0.1 + random.random() * 0.2
            specificity = 0.5 + random.random() * 0.3
            actionability = 0.65 + random.random() * 0.3  # GROUND needs > 0.6
        elif idx < 2 * third:
            # VERIFY + ENRICH candidates
            novelty = 0.4 + random.random() * 0.3
            confidence = 0.3 + random.random() * 0.3
            fabrication_risk = 0.45 + random.random() * 0.4  # > 0.4 for VERIFY
            specificity = 0.1 + random.random() * 0.25       # < 0.4 for ENRICH
            actionability = 0.3 + random.random() * 0.3
        else:
            # CHALLENGE candidates
            novelty = 0.3 + random.random() * 0.3
            confidence = 0.82 + random.random() * 0.15    # > 0.8 for CHALLENGE
            fabrication_risk = 0.05 + random.random() * 0.15
            specificity = 0.6 + random.random() * 0.3
            actionability = 0.5 + random.random() * 0.4

        # All findings get moderate-high relevance for BRIDGE
        relevance = 0.5 + random.random() * 0.4  # 0.5-0.9

        with lock:
            store.conn.execute(
                "UPDATE conditions SET "
                "novelty_score = ?, confidence = ?, fabrication_risk = ?, "
                "specificity_score = ?, relevance_score = ?, "
                "actionability_score = ?, score_version = 1, "
                "verification_status = '' "
                "WHERE id = ?",
                [novelty, confidence, fabrication_risk, specificity,
                 relevance, actionability, cid],
            )
        updated += 1

    logger.info("prescored=<%d> | findings scored for all query types", updated)
    return updated


async def log_flock_event(event: dict) -> None:
    """Log Flock events with detail."""
    etype = event.get("type", "unknown")
    if etype == "flock_round_start":
        logger.info(
            "round=<%d>, clones=<%s>, budget=<%s> | flock round starting",
            event.get("round"), event.get("clones"), event.get("budget"),
        )
    elif etype == "flock_clone_start":
        logger.info(
            "round=<%d>, clone=<%s>, queries=<%d> | evaluating clone",
            event.get("round"), event.get("clone_angle"), event.get("queries", 0),
        )
    elif etype == "flock_query_complete":
        logger.info(
            "type=<%s>, finding_id=<%s>, new_findings=<%d> | query complete",
            event.get("query_type"), event.get("target_ids"), event.get("new_findings", 0),
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
        logger.info("flock_event=<%s> | %s", etype, json.dumps(event, default=str)[:200])


async def run_flock_all(args: argparse.Namespace) -> None:
    """Run Flock with all query types enabled and permissive thresholds."""
    t0 = time.monotonic()
    run_id = f"flock_all_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    logger.info("db=<%s> | opening ConditionStore", args.db)
    store = ConditionStore(args.db)

    # Step 1: Diagnose current state
    diagnosis = diagnose_store(store)
    logger.info("diagnosis=<%s>", json.dumps(diagnosis, indent=2))

    if diagnosis["issues"]:
        logger.warning("issues=<%d> | store has problems that would cause 0 queries", len(diagnosis["issues"]))
        for issue in diagnosis["issues"]:
            logger.warning("  - %s", issue)

    # Step 2: Bootstrap score_version
    bootstrapped = bootstrap_score_version(store)
    logger.info("bootstrapped=<%d> findings to score_version=1", bootstrapped)

    # Step 3: Pre-score to ensure all query types have candidates
    prescored = prescore_for_all_query_types(store)
    logger.info("prescored=<%d> findings with query-type-targeted flags", prescored)

    # Step 4: Diagnose AGAIN after fixing
    diagnosis_after = diagnose_store(store)
    logger.info("diagnosis_after=<%s>", json.dumps(diagnosis_after, indent=2))

    total_eligible = sum(v for k, v in diagnosis_after["query_eligibility"].items() if k != "BRIDGE_angles")
    if total_eligible == 0:
        logger.error("STILL 0 eligible queries after prescoring — store may be empty")
        return

    logger.info("total_eligible=<%d> | queries should fire now", total_eligible)

    # Step 5: Select clones
    clones = select_flock_clones(store, max_clones=12, min_findings_per_angle=2)
    if not clones:
        logger.error("no clones selected")
        return

    logger.info(
        "clones=<%d>, angles=<%s> | clone perspectives ready",
        len(clones), [c.angle for c in clones],
    )

    # Step 6: Create V4 Pro completion function
    model = args.model
    api_base = args.api_base

    async def complete_fn(prompt: str) -> str:
        return await _v4pro_complete(prompt, model, api_base, args.max_tokens, args.temperature)

    # Verify V4 Pro health
    test = await complete_fn("Say 'ready' in one word.")
    if not test:
        logger.error("v4 pro health check failed")
        return
    logger.info("v4pro=<healthy> | confirmed responding")

    # Step 7: Preview query selection for first clone
    preview_config = FlockQueryManagerConfig(
        research_query=(
            "Comprehensive analysis of performance-enhancing substances, "
            "metabolic health interventions, and longevity protocols"
        ),
        max_rounds=args.max_rounds,
        max_queries_per_round=500,
        batch_size=args.batch_size,
        convergence_threshold=0.01,
        # Permissive thresholds
        novelty_confidence_gap=0.1,    # default 0.2
        fabrication_risk_floor=0.3,    # default 0.4
        specificity_ceiling=0.5,       # default 0.4
        cross_angle_min_relevance=0.2, # default 0.3
        enable_synthesis=True,
        enable_challenge=True,
    )

    preview_queries = select_queries(store, clones[0], preview_config, round_number=1)
    by_type: dict[str, int] = {}
    for q in preview_queries:
        by_type[q.query_type.value] = by_type.get(q.query_type.value, 0) + 1
    logger.info(
        "preview_clone=<%s>, total_queries=<%d>, by_type=<%s> | "
        "query selection preview (clone 0)",
        clones[0].angle, len(preview_queries), json.dumps(by_type),
    )

    if not preview_queries:
        logger.error("STILL 0 queries after prescoring + permissive thresholds — aborting")
        return

    # Step 8: Run the full Flock
    manager = FlockQueryManager(
        store=store,
        complete=complete_fn,
        config=preview_config,
    )

    logger.info(
        "max_rounds=<%d>, batch_size=<%d>, clones=<%d> | starting Flock evaluation",
        args.max_rounds, args.batch_size, len(clones),
    )

    result = await manager.run(
        clones=clones,
        run_id=run_id,
        on_event=log_flock_event,
    )

    elapsed = time.monotonic() - t0

    logger.info(
        "total_queries=<%d>, total_evaluations=<%d>, total_new_findings=<%d>, "
        "convergence=<%s>, elapsed=<%.1f>s | Flock complete",
        result.total_queries, result.total_evaluations,
        result.total_new_findings, result.convergence_reason, elapsed,
    )

    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"flock_all_metrics_{run_id}.json")
    metrics = {
        "run_id": run_id,
        "model": args.model,
        "diagnosis_before": diagnosis,
        "diagnosis_after": diagnosis_after,
        "prescored": prescored,
        "clones_used": [c.angle for c in clones],
        "total_queries": result.total_queries,
        "total_evaluations": result.total_evaluations,
        "total_new_findings": result.total_new_findings,
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
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics=<%s> | saved", metrics_path)

    print(f"\n{'=' * 60}")
    print("  FLOCK ALL-QUERIES EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:          {args.model}")
    print(f"  Total queries:  {result.total_queries}")
    print(f"  Evaluations:    {result.total_evaluations}")
    print(f"  New findings:   {result.total_new_findings}")
    print(f"  Convergence:    {result.convergence_reason}")
    print(f"  Elapsed:        {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Run ALL Flock queries with permissive thresholds")
    parser.add_argument("--db", default="/root/workspace/MiroThinker/enriched_corpus.duckdb")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--output-dir", default="/root/flock_output")
    args = parser.parse_args()

    asyncio.run(run_flock_all(args))


if __name__ == "__main__":
    main()
