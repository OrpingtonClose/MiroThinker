#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Run MCP engine experiments with different configurations.

Tests the MCP-powered worker architecture against cloud LLM APIs
(OpenAI, Together, etc.) with a matrix of configurations to measure:
- Convergence behavior (findings per wave, when it stabilizes)
- Tool call patterns (which tools workers prefer, how many calls)
- Store growth (total findings, per-angle distribution)
- Report quality (length, coverage of angles)
- Timing (per-wave, per-phase, total)

Each experiment gets a fresh ConditionStore and produces:
- metrics JSON (machine-readable)
- report markdown (human-readable)
- store snapshot (DuckDB queries for analysis)

Usage:
    export OPENAI_API_KEY=sk-...
    python run_mcp_experiments.py --corpus /path/to/corpus.md
    python run_mcp_experiments.py --corpus /path/to/corpus.md --experiment-id 3
    python run_mcp_experiments.py --list  # show experiment matrix
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ensure repo root and strands-agent are importable
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_STRANDS_AGENT = str(Path(__file__).resolve().parents[2] / "apps" / "strands-agent")
for p in (_REPO_ROOT, _STRANDS_AGENT):
    if p not in sys.path:
        sys.path.insert(0, p)

from corpus import ConditionStore
from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

@dataclass
class ExperimentDef:
    """A single experiment configuration."""

    name: str
    description: str
    max_workers: int = 3
    max_waves: int = 2
    convergence_threshold: int = 5
    temperature: float = 0.3
    max_tokens: int = 4096
    enable_serendipity_wave: bool = True
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    corpus_slice: str = "full"  # "full", "half", "quarter"
    required_angles: list[str] = field(default_factory=list)


def build_experiment_matrix() -> list[ExperimentDef]:
    """Build the full experiment matrix."""
    experiments = []

    # ── Experiment 1: Baseline — minimal config ──────────────────────
    experiments.append(ExperimentDef(
        name="baseline_3w_2wave",
        description=(
            "Baseline: 3 workers, 2 waves, default convergence. "
            "Establishes minimum viable configuration."
        ),
        max_workers=3,
        max_waves=2,
        convergence_threshold=3,
        temperature=0.3,
        enable_serendipity_wave=False,
    ))

    # ── Experiment 2: More workers ───────────────────────────────────
    experiments.append(ExperimentDef(
        name="more_workers_5w_2wave",
        description=(
            "5 workers, 2 waves. Tests whether more specialist angles "
            "produce richer findings without more waves."
        ),
        max_workers=5,
        max_waves=2,
        convergence_threshold=5,
        temperature=0.3,
        enable_serendipity_wave=False,
    ))

    # ── Experiment 3: More waves ─────────────────────────────────────
    experiments.append(ExperimentDef(
        name="more_waves_3w_4wave",
        description=(
            "3 workers, 4 waves. Tests iterative refinement — do workers "
            "find more on successive passes through the store?"
        ),
        max_workers=3,
        max_waves=4,
        convergence_threshold=3,
        temperature=0.3,
        enable_serendipity_wave=False,
    ))

    # ── Experiment 4: Serendipity wave ───────────────────────────────
    experiments.append(ExperimentDef(
        name="serendipity_3w_2wave",
        description=(
            "3 workers, 2 waves + serendipity wave. Tests whether the "
            "cross-domain discovery pass finds connections workers missed."
        ),
        max_workers=3,
        max_waves=2,
        convergence_threshold=3,
        temperature=0.3,
        enable_serendipity_wave=True,
    ))

    # ── Experiment 5: Higher temperature ─────────────────────────────
    experiments.append(ExperimentDef(
        name="high_temp_3w_2wave",
        description=(
            "3 workers, 2 waves, temperature=0.7. Tests whether higher "
            "creativity produces more diverse findings or more noise."
        ),
        max_workers=3,
        max_waves=2,
        convergence_threshold=3,
        temperature=0.7,
        enable_serendipity_wave=False,
    ))

    # ── Experiment 6: Low convergence threshold ──────────────────────
    experiments.append(ExperimentDef(
        name="strict_convergence_3w_4wave",
        description=(
            "3 workers, 4 waves, convergence_threshold=10. Tests whether "
            "a stricter threshold causes early stopping and saves calls."
        ),
        max_workers=3,
        max_waves=4,
        convergence_threshold=10,
        temperature=0.3,
        enable_serendipity_wave=False,
    ))

    # ── Experiment 7: Full config — max everything ───────────────────
    experiments.append(ExperimentDef(
        name="full_config_5w_3wave_seren",
        description=(
            "5 workers, 3 waves, serendipity, default threshold. "
            "The intended production configuration."
        ),
        max_workers=5,
        max_waves=3,
        convergence_threshold=5,
        temperature=0.3,
        enable_serendipity_wave=True,
    ))

    # ── Experiment 8: Required angles preset (no auto-detection) ────
    experiments.append(ExperimentDef(
        name="preset_angles_5w_3wave",
        description=(
            "5 workers, 3 waves, PRESET angles from h200_test/angles.py. "
            "Tests whether explicit angle definitions fix the catch-all "
            "problem seen in auto-detected angles."
        ),
        max_workers=5,
        max_waves=3,
        convergence_threshold=5,
        temperature=0.3,
        enable_serendipity_wave=True,
        required_angles=[
            "Insulin & GH protocols — Milos Sarcev framework",
            "Testosterone & Trenbolone pharmacokinetics",
            "Ancillaries & health marker management",
            "Micronutrient interactions — compound synergies",
            "Ramping strategy & phase periodization",
        ],
    ))

    # ── Experiment 9: Small corpus slice ─────────────────────────────
    experiments.append(ExperimentDef(
        name="small_corpus_3w_2wave",
        description=(
            "3 workers, 2 waves, quarter corpus. Tests behavior with "
            "minimal data — does the architecture degrade gracefully?"
        ),
        max_workers=3,
        max_waves=2,
        convergence_threshold=3,
        temperature=0.3,
        enable_serendipity_wave=False,
        corpus_slice="quarter",
    ))

    return experiments


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

async def run_single_experiment(
    experiment: ExperimentDef,
    corpus: str,
    query: str,
    output_dir: Path,
) -> dict:
    """Run a single experiment and return results."""
    logger.info(
        "experiment=<%s> | starting: %s",
        experiment.name, experiment.description,
    )

    # Apply corpus slicing
    if experiment.corpus_slice == "half":
        corpus_text = corpus[:len(corpus) // 2]
    elif experiment.corpus_slice == "quarter":
        corpus_text = corpus[:len(corpus) // 4]
    else:
        corpus_text = corpus

    # Get API key
    api_key = os.environ.get(experiment.api_key_env, "")
    if not api_key:
        logger.error(
            "experiment=<%s>, env=<%s> | API key not found",
            experiment.name, experiment.api_key_env,
        )
        return {"experiment": experiment.name, "error": f"Missing {experiment.api_key_env}"}

    # Create fresh store for this experiment
    store = ConditionStore(db_path="")  # in-memory

    # Build config
    config = MCPSwarmConfig(
        max_workers=experiment.max_workers,
        max_waves=experiment.max_waves,
        convergence_threshold=experiment.convergence_threshold,
        api_base=experiment.api_base,
        model=experiment.model,
        api_key=api_key,
        max_tokens=experiment.max_tokens,
        temperature=experiment.temperature,
        required_angles=experiment.required_angles,
        enable_serendipity_wave=experiment.enable_serendipity_wave,
    )

    # Build completion function for angle detection / report generation
    async def _complete(prompt: str) -> str:
        import httpx
        url = f"{experiment.api_base}/chat/completions"
        payload = {
            "model": experiment.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Produce your analysis."},
            ],
            "max_tokens": experiment.max_tokens,
            "temperature": experiment.temperature,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                logger.warning(
                    "experiment=<%s>, error=<%s> | completion call failed",
                    experiment.name, str(exc),
                )
                return ""

    engine = MCPSwarmEngine(
        store=store,
        complete=_complete,
        config=config,
    )

    # Collect events
    events: list[dict] = []

    async def _on_event(event: dict) -> None:
        events.append(event)
        logger.info(
            "experiment=<%s>, event=<%s> | %s",
            experiment.name, event.get("type", "?"),
            json.dumps(event, default=str)[:200],
        )

    # Run
    t0 = time.monotonic()
    try:
        result = await engine.synthesize(
            corpus=corpus_text,
            query=query,
            on_event=_on_event,
        )
    except Exception as exc:
        logger.error(
            "experiment=<%s>, error=<%s> | engine failed",
            experiment.name, str(exc),
        )
        return {
            "experiment": experiment.name,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - t0, 1),
        }

    elapsed = time.monotonic() - t0

    # Query store for detailed analysis
    with store._lock:
        total_findings = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE"
        ).fetchone()[0]

        tool_call_count = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE row_type = 'tool_call'"
        ).fetchone()[0]

        # Tool call breakdown
        tool_breakdown = store.conn.execute(
            """SELECT source_ref, COUNT(*) as cnt
               FROM conditions
               WHERE row_type = 'tool_call'
               GROUP BY source_ref
               ORDER BY cnt DESC""",
        ).fetchall()

        # Findings per angle
        angle_distribution = store.conn.execute(
            """SELECT angle, COUNT(*) as cnt, AVG(confidence) as avg_conf
               FROM conditions
               WHERE consider_for_use = TRUE
                 AND row_type IN ('finding', 'thought', 'insight')
                 AND angle != ''
               GROUP BY angle
               ORDER BY cnt DESC""",
        ).fetchall()

        # Findings per worker
        worker_findings = store.conn.execute(
            """SELECT source_ref, COUNT(*) as cnt
               FROM conditions
               WHERE consider_for_use = TRUE
                 AND source_type = 'worker_analysis'
               GROUP BY source_ref
               ORDER BY cnt DESC""",
        ).fetchall()

    # Build results
    results = {
        "experiment": experiment.name,
        "description": experiment.description,
        "config": {
            "max_workers": experiment.max_workers,
            "max_waves": experiment.max_waves,
            "convergence_threshold": experiment.convergence_threshold,
            "temperature": experiment.temperature,
            "model": experiment.model,
            "enable_serendipity_wave": experiment.enable_serendipity_wave,
            "corpus_slice": experiment.corpus_slice,
            "corpus_chars": len(corpus_text),
        },
        "metrics": {
            "total_elapsed_s": round(elapsed, 1),
            "total_waves": result.metrics.total_waves,
            "total_findings_stored": result.metrics.total_findings_stored,
            "total_tool_calls_engine": result.metrics.total_tool_calls,
            "total_tool_calls_store": tool_call_count,
            "findings_per_wave": result.metrics.findings_per_wave,
            "convergence_reason": result.metrics.convergence_reason,
            "phase_times": {k: round(v, 1) for k, v in result.metrics.phase_times.items()},
        },
        "store_analysis": {
            "total_findings": total_findings,
            "tool_call_breakdown": [
                {"tool": ref, "count": cnt}
                for ref, cnt in tool_breakdown
            ],
            "angle_distribution": [
                {"angle": angle, "count": cnt, "avg_confidence": round(avg_conf, 3)}
                for angle, cnt, avg_conf in angle_distribution
            ],
            "worker_findings": [
                {"worker": ref, "count": cnt}
                for ref, cnt in worker_findings
            ],
        },
        "report_chars": len(result.report),
        "angles_detected": result.angles_detected,
        "events": events,
    }

    # Save outputs
    exp_dir = output_dir / experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(exp_dir / "report.md", "w") as f:
        f.write(result.report)

    # Save store snapshot as queryable summary
    with open(exp_dir / "store_snapshot.json", "w") as f:
        with store._lock:
            all_findings = store.conn.execute(
                """SELECT id, fact, confidence, angle, row_type,
                          source_type, source_ref, phase, created_at
                   FROM conditions
                   ORDER BY id ASC"""
            ).fetchall()
        snapshot = [
            {
                "id": row[0], "fact": row[1][:200], "confidence": row[2],
                "angle": row[3], "row_type": row[4], "source_type": row[5],
                "source_ref": row[6], "phase": row[7], "created_at": row[8],
            }
            for row in all_findings
        ]
        json.dump(snapshot, f, indent=2, default=str)

    logger.info(
        "experiment=<%s>, elapsed_s=<%.1f>, findings=<%d>, tool_calls=<%d>, "
        "report_chars=<%d> | experiment complete",
        experiment.name, elapsed, total_findings, tool_call_count,
        len(result.report),
    )

    return results


def generate_summary_report(all_results: list[dict], output_dir: Path) -> str:
    """Generate a markdown summary comparing all experiments."""
    lines = [
        "# MCP Engine Experiment Results\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M UTC')}\n",
        "## Configuration Matrix\n",
        "| Experiment | Workers | Waves | Conv.Thresh | Temp | Serendipity | Corpus |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in all_results:
        if "error" in r and "config" not in r:
            lines.append(f"| {r['experiment']} | ERROR | - | - | - | - | - |")
            continue
        c = r["config"]
        lines.append(
            f"| {r['experiment']} | {c['max_workers']} | {c['max_waves']} | "
            f"{c['convergence_threshold']} | {c['temperature']} | "
            f"{'Yes' if c['enable_serendipity_wave'] else 'No'} | "
            f"{c['corpus_slice']} ({c['corpus_chars']:,} chars) |"
        )

    lines.extend([
        "\n## Results Summary\n",
        "| Experiment | Elapsed(s) | Waves Run | Findings | Tool Calls | Report Chars | Convergence |",
        "|---|---|---|---|---|---|---|",
    ])

    for r in all_results:
        if "error" in r and "metrics" not in r:
            lines.append(f"| {r['experiment']} | {r.get('elapsed_s', '?')} | ERROR | - | - | - | {r['error'][:50]} |")
            continue
        m = r["metrics"]
        lines.append(
            f"| {r['experiment']} | {m['total_elapsed_s']} | {m['total_waves']} | "
            f"{m['total_findings_stored']} | {m['total_tool_calls_store']} | "
            f"{r['report_chars']:,} | {m['convergence_reason'][:40]} |"
        )

    # Per-wave findings comparison
    lines.extend([
        "\n## Findings Per Wave\n",
        "| Experiment | Wave 1 | Wave 2 | Wave 3 | Wave 4 |",
        "|---|---|---|---|---|",
    ])

    for r in all_results:
        if "metrics" not in r:
            continue
        fpw = r["metrics"]["findings_per_wave"]
        cols = [str(fpw[i]) if i < len(fpw) else "-" for i in range(4)]
        lines.append(f"| {r['experiment']} | {' | '.join(cols)} |")

    # Angle distribution
    lines.extend([
        "\n## Angle Coverage\n",
    ])

    for r in all_results:
        if "store_analysis" not in r:
            continue
        lines.append(f"\n### {r['experiment']}\n")
        ad = r["store_analysis"]["angle_distribution"]
        if ad:
            lines.append("| Angle | Findings | Avg Confidence |")
            lines.append("|---|---|---|")
            for a in ad:
                lines.append(f"| {a['angle'][:50]} | {a['count']} | {a['avg_confidence']:.3f} |")

    # Tool call patterns
    lines.extend([
        "\n## Tool Call Patterns\n",
    ])

    for r in all_results:
        if "store_analysis" not in r:
            continue
        lines.append(f"\n### {r['experiment']}\n")
        tc = r["store_analysis"]["tool_call_breakdown"]
        if tc:
            # Aggregate by tool name (strip worker prefix)
            tool_counts: dict[str, int] = {}
            for t in tc:
                tool_name = t["tool"].split("/")[-1] if "/" in t["tool"] else t["tool"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + t["count"]
            lines.append("| Tool | Total Calls |")
            lines.append("|---|---|")
            for tool, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {tool} | {cnt} |")

    # Key observations
    lines.extend([
        "\n## Key Observations\n",
        "_(Auto-generated from metrics comparison)_\n",
    ])

    # Find best/worst by findings
    valid = [r for r in all_results if "metrics" in r]
    if valid:
        best_findings = max(valid, key=lambda r: r["metrics"]["total_findings_stored"])
        worst_findings = min(valid, key=lambda r: r["metrics"]["total_findings_stored"])
        fastest = min(valid, key=lambda r: r["metrics"]["total_elapsed_s"])
        slowest = max(valid, key=lambda r: r["metrics"]["total_elapsed_s"])

        lines.append(f"- **Most findings:** {best_findings['experiment']} ({best_findings['metrics']['total_findings_stored']} findings)")
        lines.append(f"- **Fewest findings:** {worst_findings['experiment']} ({worst_findings['metrics']['total_findings_stored']} findings)")
        lines.append(f"- **Fastest:** {fastest['experiment']} ({fastest['metrics']['total_elapsed_s']}s)")
        lines.append(f"- **Slowest:** {slowest['experiment']} ({slowest['metrics']['total_elapsed_s']}s)")

        # Check convergence patterns
        converged_early = [r for r in valid if "converged" in r["metrics"]["convergence_reason"]]
        if converged_early:
            lines.append(f"- **Converged early:** {len(converged_early)}/{len(valid)} experiments")

        # Wave 2 vs Wave 1 delta
        wave_deltas = []
        for r in valid:
            fpw = r["metrics"]["findings_per_wave"]
            if len(fpw) >= 2 and fpw[0] > 0:
                delta = (fpw[1] - fpw[0]) / fpw[0] * 100
                wave_deltas.append((r["experiment"], delta))
        if wave_deltas:
            lines.append("\n**Wave 2 vs Wave 1 finding change:**")
            for name, delta in wave_deltas:
                direction = "+" if delta > 0 else ""
                lines.append(f"  - {name}: {direction}{delta:.0f}%")

    report = "\n".join(lines)

    with open(output_dir / "EXPERIMENT_SUMMARY.md", "w") as f:
        f.write(report)

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MCP engine experiments",
    )
    parser.add_argument(
        "--corpus", required=False, default=None,
        help="Path to corpus text/markdown file (required unless --list)",
    )
    parser.add_argument(
        "--output-dir", default="experiment_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiment-id", type=int, default=-1,
        help="Run a specific experiment by index (0-based). -1 = run all.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_experiments",
        help="List all experiments and exit",
    )
    parser.add_argument(
        "--query", default="",
        help="Override the default research query",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    experiments = build_experiment_matrix()

    if args.list_experiments:
        # --list doesn't need a corpus file
        pass
    elif not args.corpus:
        parser.error("--corpus is required when not using --list")

    if args.list_experiments:
        print(f"\n{'═' * 60}")
        print(f"  MCP EXPERIMENT MATRIX ({len(experiments)} experiments)")
        print(f"{'═' * 60}")
        for i, exp in enumerate(experiments):
            print(f"\n  [{i}] {exp.name}")
            print(f"      {exp.description}")
            print(f"      workers={exp.max_workers}, waves={exp.max_waves}, "
                  f"conv={exp.convergence_threshold}, temp={exp.temperature}, "
                  f"seren={exp.enable_serendipity_wave}, corpus={exp.corpus_slice}")
        print(f"\n{'═' * 60}\n")
        return

    # Load corpus
    with open(args.corpus) as f:
        corpus = f.read()
    logger.info("corpus_chars=<%d> | loaded from %s", len(corpus), args.corpus)

    # Default query
    query = args.query or (
        "Synthesize a comprehensive, practitioner-grade ramping bodybuilding "
        "cycle protocol covering testosterone, trenbolone, insulin (grounded "
        "in Milos Sarcev's timing framework), growth hormone, turinabol, "
        "boldenone, actovegin, and LGD-4033. For each phase, specify exact "
        "compounds, dosages, frequencies, timing windows, micronutrient "
        "support, bloodwork markers, and transition criteria. Explain the "
        "pharmacokinetic reasoning behind every decision."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select experiments
    if args.experiment_id >= 0:
        if args.experiment_id >= len(experiments):
            logger.error("experiment_id=%d out of range (0-%d)", args.experiment_id, len(experiments) - 1)
            sys.exit(1)
        to_run = [experiments[args.experiment_id]]
    else:
        to_run = experiments

    # Run experiments
    all_results = []
    for i, exp in enumerate(to_run):
        print(f"\n{'─' * 60}")
        print(f"  Running experiment {i + 1}/{len(to_run)}: {exp.name}")
        print(f"{'─' * 60}\n")

        result = asyncio.run(run_single_experiment(
            experiment=exp,
            corpus=corpus,
            query=query,
            output_dir=output_dir,
        ))
        all_results.append(result)

        # Brief pause between experiments
        if i < len(to_run) - 1:
            logger.info("pausing 5s between experiments...")
            time.sleep(5)

    # Generate summary
    summary = generate_summary_report(all_results, output_dir)
    print(f"\n{'═' * 60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Results in: {output_dir.resolve()}")
    print(f"  Summary:    {output_dir.resolve()}/EXPERIMENT_SUMMARY.md")
    print(f"{'═' * 60}\n")
    print(summary)


if __name__ == "__main__":
    main()
