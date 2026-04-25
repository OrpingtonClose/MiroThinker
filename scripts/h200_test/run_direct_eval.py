#!/usr/bin/env python3
"""Direct V4 Pro evaluation of all findings in the ConditionStore.

No Flock framework — just reads every finding, sends it to V4 Pro for
evaluation, and writes the evaluation text back into the store as a
child row (row_type='evaluation') linked to the original finding.

Also updates the original finding's gradient flags based on V4 Pro's
assessment.

Usage:
    python3 run_direct_eval.py \
        --db /root/workspace/MiroThinker/enriched_corpus.duckdb \
        --api-base http://localhost:8000/v1 \
        --concurrency 20
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

EVAL_PROMPT_TEMPLATE = """You are a critical scientific evaluator. You have deep expertise in pharmacology, endocrinology, exercise physiology, and clinical research methodology.

FINDING TO EVALUATE (from angle: {angle}):
{fact}

SOURCE: {source}

TASK: Evaluate this finding thoroughly. Your evaluation must include:

1. **Claim Assessment**: Is the core claim well-supported by evidence? Rate confidence 0.0-1.0.
2. **Source Quality**: How reliable is this source? Rate trust 0.0-1.0.
3. **Novelty**: How novel or surprising is this finding? Rate 0.0-1.0.
4. **Specificity**: How specific and actionable is the claim? Rate 0.0-1.0.
5. **Fabrication Risk**: Could this be fabricated, exaggerated, or taken out of context? Rate risk 0.0-1.0.
6. **Key Caveats**: What are the most important limitations, confounders, or context the reader needs?
7. **Cross-Domain Connections**: What other domains or findings does this relate to?
8. **Verification Status**: One of: verified, plausible, uncertain, dubious, refuted.

End your response with a JSON block on the last line in exactly this format:
```json
{{"confidence": 0.X, "trust": 0.X, "novelty": 0.X, "specificity": 0.X, "fabrication_risk": 0.X, "verification_status": "STATUS"}}
```"""


async def call_v4pro(
    prompt: str,
    model: str,
    api_base: str,
    max_tokens: int,
    temperature: float,
    timeout: float = 600.0,
) -> str:
    """Call V4 Pro and return response text.

    Args:
        prompt: Full prompt text.
        model: Model identifier on vLLM.
        api_base: vLLM API base URL.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        timeout: HTTP timeout in seconds.

    Returns:
        Response text, or empty string on failure.
    """
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
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


def parse_scores(text: str) -> dict[str, float | str]:
    """Extract the JSON score block from evaluation text.

    Args:
        text: Full evaluation response text.

    Returns:
        Dict with confidence, trust, novelty, specificity,
        fabrication_risk, verification_status. Empty dict on failure.
    """
    import re

    # Find JSON block — look for last ```json ... ``` or last { ... } line
    patterns = [
        r"```json\s*\n(\{[^}]+\})\s*\n```",
        r"(\{\"confidence\"[^}]+\})",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                continue
    return {}


async def evaluate_finding(
    finding: dict,
    model: str,
    api_base: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Evaluate a single finding via V4 Pro.

    Args:
        finding: Dict with id, fact, angle, source_url, source_type.
        model: Model identifier on vLLM.
        api_base: vLLM API base URL.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        semaphore: Concurrency limiter.

    Returns:
        Dict with evaluation text and parsed scores, or None on failure.
    """
    async with semaphore:
        prompt = EVAL_PROMPT_TEMPLATE.format(
            angle=finding["angle"] or "general",
            fact=finding["fact"],
            source=finding["source_url"] or finding["source_type"] or "unknown",
        )

        text = await call_v4pro(prompt, model, api_base, max_tokens, temperature)
        if not text:
            return None

        scores = parse_scores(text)

        return {
            "finding_id": finding["id"],
            "angle": finding["angle"],
            "evaluation_text": text,
            "scores": scores,
        }


async def run_evaluations(args: argparse.Namespace) -> None:
    """Run all evaluations and write results back to ConditionStore.

    Args:
        args: Parsed CLI arguments.
    """
    import duckdb
    import threading

    t0 = time.monotonic()
    run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Open DuckDB
    logger.info("db=<%s> | opening ConditionStore", args.db)
    conn = duckdb.connect(args.db)
    lock = threading.RLock()

    # Ensure evaluation columns exist
    for col, typedef in (
        ("evaluation_text", "TEXT DEFAULT ''"),
        ("evaluation_count", "INTEGER DEFAULT 0"),
        ("last_evaluated_at", "TEXT DEFAULT ''"),
    ):
        try:
            conn.execute(
                f"ALTER TABLE conditions ADD COLUMN IF NOT EXISTS {col} {typedef}"
            )
        except Exception:
            pass

    # Read all findings
    with lock:
        rows = conn.execute(
            "SELECT id, fact, angle, source_url, source_type "
            "FROM conditions "
            "WHERE consider_for_use = TRUE AND row_type = 'finding' "
            "ORDER BY id"
        ).fetchall()

    findings = [
        {"id": r[0], "fact": r[1], "angle": r[2], "source_url": r[3], "source_type": r[4]}
        for r in rows
    ]

    logger.info("findings=<%d> | loaded findings from store", len(findings))

    if not findings:
        logger.error("no findings in store — nothing to evaluate")
        return

    # Verify V4 Pro is alive
    try:
        test = await call_v4pro(
            "Say 'ready' in one word.", args.model, args.api_base, 32, 0.0,
        )
        if not test:
            logger.error("v4 pro health check failed")
            return
        logger.info("v4pro=<healthy> | proceeding with evaluations")
    except Exception as exc:
        logger.error("error=<%s> | v4 pro unreachable", exc)
        return

    # Run evaluations with concurrency limit
    semaphore = asyncio.Semaphore(args.concurrency)
    completed = 0
    failed = 0
    eval_rows_written = 0

    # Process in batches for progress reporting
    batch_size = args.concurrency * 2
    now = datetime.now(timezone.utc).isoformat()

    for batch_start in range(0, len(findings), batch_size):
        batch = findings[batch_start:batch_start + batch_size]

        tasks = [
            evaluate_finding(f, args.model, args.api_base, args.max_tokens, args.temperature, semaphore)
            for f in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("error=<%s> | evaluation task raised", result)
                failed += 1
                continue
            if result is None:
                failed += 1
                continue

            completed += 1
            scores = result["scores"]
            finding_id = result["finding_id"]
            eval_text = result["evaluation_text"]

            # Write evaluation as child row
            with lock:
                # Get next ID
                max_id = conn.execute("SELECT MAX(id) FROM conditions").fetchone()[0] or 0
                eval_id = max_id + 1

                conn.execute(
                    "INSERT INTO conditions "
                    "(id, fact, source_type, source_ref, row_type, "
                    "parent_id, consider_for_use, angle, created_at, "
                    "source_model, source_run, phase) "
                    "VALUES (?, ?, 'v4pro-eval', ?, 'evaluation', "
                    "?, TRUE, ?, ?, ?, ?, 'flock-direct')",
                    [
                        eval_id, eval_text, str(finding_id), finding_id,
                        result["angle"], now, args.model, run_id,
                    ],
                )
                eval_rows_written += 1

                # Update original finding's scores if we parsed them
                if scores:
                    confidence = scores.get("confidence")
                    trust = scores.get("trust")
                    novelty = scores.get("novelty")
                    specificity = scores.get("specificity")
                    fabrication_risk = scores.get("fabrication_risk")
                    verification = scores.get("verification_status", "")

                    updates = []
                    params = []
                    if isinstance(confidence, (int, float)):
                        updates.append("confidence = ?")
                        params.append(float(confidence))
                    if isinstance(trust, (int, float)):
                        updates.append("trust_score = ?")
                        params.append(float(trust))
                    if isinstance(novelty, (int, float)):
                        updates.append("novelty_score = ?")
                        params.append(float(novelty))
                    if isinstance(specificity, (int, float)):
                        updates.append("specificity_score = ?")
                        params.append(float(specificity))
                    if isinstance(fabrication_risk, (int, float)):
                        updates.append("fabrication_risk = ?")
                        params.append(float(fabrication_risk))
                    if verification:
                        updates.append("verification_status = ?")
                        params.append(str(verification))

                    updates.append("evaluation_count = evaluation_count + 1")
                    updates.append("last_evaluated_at = ?")
                    params.append(now)
                    updates.append("score_version = 2")

                    params.append(finding_id)
                    conn.execute(
                        f"UPDATE conditions SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )

        elapsed = time.monotonic() - t0
        logger.info(
            "completed=<%d>, failed=<%d>, total=<%d>, eval_rows=<%d>, "
            "elapsed=<%.1f>s | batch progress",
            completed, failed, len(findings), eval_rows_written, elapsed,
        )

    elapsed = time.monotonic() - t0

    # Summary
    logger.info(
        "completed=<%d>, failed=<%d>, eval_rows=<%d>, elapsed=<%.1f>s | "
        "direct evaluation complete",
        completed, failed, eval_rows_written, elapsed,
    )

    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"direct_eval_metrics_{run_id}.json")
    metrics = {
        "run_id": run_id,
        "model": args.model,
        "findings_evaluated": completed,
        "findings_failed": failed,
        "eval_rows_written": eval_rows_written,
        "elapsed_s": elapsed,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("metrics=<%s> | saved", metrics_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("  DIRECT V4 PRO EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:       {args.model}")
    print(f"  Evaluated:   {completed}/{len(findings)}")
    print(f"  Failed:      {failed}")
    print(f"  Eval rows:   {eval_rows_written}")
    print(f"  Elapsed:     {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}\n")

    conn.close()


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Direct V4 Pro evaluation of ConditionStore findings")
    parser.add_argument("--db", default="/root/workspace/MiroThinker/enriched_corpus.duckdb")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--output-dir", default="/root/flock_output")
    args = parser.parse_args()

    asyncio.run(run_evaluations(args))


if __name__ == "__main__":
    main()
