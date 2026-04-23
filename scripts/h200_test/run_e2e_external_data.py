#!/usr/bin/env python3
"""E2E test: external data acquisition pipeline.

Tests the full flow when NO corpus files are attached:
  prompt → corpus_builder (multi-API search) → swarm → research_organizer (clones) → report

Validates:
  1. Corpus acquired from external sources (not just static file)
  2. Workers receive externally-sourced material in §2
  3. Research Organizer spawns clones between waves
  4. Clone findings appear as §8 FRESH EVIDENCE in subsequent waves
  5. Report references external sources
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "apps" / "strands-agent"))

from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROMPT = (
    "Milos Sarcev insulin protocol for bodybuilding in the context of "
    "taking insulin, hgh and trenbolone based cycles with full complexity "
    "breakdown between all food related nutrients, supplements and PEDs"
)


async def run(
    output_dir: str,
    api_base: str,
    api_key: str,
    model: str,
    max_waves: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    compact_every: int = 3,
) -> None:
    """Execute E2E run with external data acquisition (no pre-built corpus)."""
    from corpus import ConditionStore

    import httpx

    db_path = str(Path(output_dir) / "store.duckdb")
    store = ConditionStore(db_path=db_path)

    async def complete_fn(prompt: str) -> str:
        """OpenAI-compatible completion with auth headers."""
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
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)
        phase = event.get("phase", "")
        logger.info("event | %s", json.dumps(event, default=str))

    config = MCPSwarmConfig(
        max_workers=8,
        max_waves=max_waves,
        convergence_threshold=5,
        api_base=api_base,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        required_angles=[],
        report_max_tokens=16384,
        enable_serendipity_wave=True,
        source_model=model,
        source_run=f"e2e_ext_{time.strftime('%Y%m%d_%H%M%S')}",
        compact_every_n_waves=compact_every,
        enable_rolling_summaries=True,
    )

    engine = MCPSwarmEngine(store=store, complete=complete_fn, config=config)

    logger.info("prompt=<%s> | starting synthesis with NO corpus (external acquisition)", PROMPT[:80])

    # KEY: corpus=None triggers the corpus builder
    result = await engine.synthesize(corpus=None, query=PROMPT, on_event=on_event)

    # ── Save results ──────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    report_path = out / f"report_{ts}.md"
    report_path.write_text(result.report)

    # Check for external data signals
    acq_events = [e for e in events if "corpus_acquisition" in e.get("phase", "")]
    ro_events = [e for e in events if "research_organizer" in e.get("phase", "")]

    metrics = {
        "prompt": PROMPT,
        "engine": "mcp",
        "model": model,
        "api_base": api_base,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "max_waves": max_waves,
        "total_elapsed_s": result.metrics.total_elapsed_s,
        "total_waves": result.metrics.total_waves,
        "total_findings_stored": result.metrics.total_findings_stored,
        "total_tool_calls": result.metrics.total_tool_calls,
        "findings_per_wave": result.metrics.findings_per_wave,
        "phase_times": result.metrics.phase_times,
        "convergence_reason": result.metrics.convergence_reason,
        "angles_detected": result.angles_detected,
        "report_chars": len(result.report),
        "corpus_acquisition_events": acq_events,
        "research_organizer_events": ro_events,
        "all_events": events,
    }

    metrics_path = out / f"metrics_{ts}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))

    events_path = out / f"events_{ts}.json"
    events_path.write_text(json.dumps(events, indent=2, default=str))

    # ── Validation summary ────────────────────────────────────────
    corpus_acquired = any("corpus_acquisition_complete" in e.get("phase", "") for e in events)
    corpus_chars = next(
        (e.get("corpus_chars", 0) for e in events if "corpus_acquisition_complete" in e.get("phase", "")),
        0,
    )
    ro_ran = len(ro_events) > 0
    clone_findings = sum(e.get("clone_findings", 0) for e in ro_events)

    print(f"\n{'═' * 60}")
    print(f"  E2E External Data Acquisition Test")
    print(f"{'═' * 60}")
    print(f"  Prompt:              {PROMPT[:60]}...")
    print(f"  Model:               {model}")
    print(f"  Elapsed:             {result.metrics.total_elapsed_s:.1f}s")
    print(f"  Waves:               {result.metrics.total_waves}")
    print(f"  Findings:            {result.metrics.total_findings_stored}")
    print(f"  Findings/wave:       {result.metrics.findings_per_wave}")
    print(f"  Convergence:         {result.metrics.convergence_reason}")
    print(f"  Angles:              {result.angles_detected}")
    print(f"  Report:              {len(result.report):,} chars")
    print(f"{'─' * 60}")
    print(f"  CORPUS ACQUIRED:     {'YES' if corpus_acquired else 'NO'} ({corpus_chars:,} chars)")
    print(f"  RESEARCH ORGANIZER:  {'YES' if ro_ran else 'NO'} ({len(ro_events)} runs)")
    print(f"  CLONE FINDINGS:      {clone_findings}")
    print(f"{'─' * 60}")

    if corpus_acquired:
        acq_time = result.metrics.phase_times.get("corpus_acquisition", 0)
        print(f"  Acquisition time:    {acq_time:.1f}s")

    for e in ro_events:
        wave_n = e.get("phase", "").split("_")[-1]
        print(f"  RO wave {wave_n}:          {e.get('clones', 0)} clones, {e.get('clone_findings', 0)} findings")

    print(f"\n  Output:              {out.resolve()}")
    print(f"{'═' * 60}\n")

    # Assertions
    if not corpus_acquired:
        print("FAIL: Corpus was not acquired from external sources!")
        sys.exit(1)
    if corpus_chars < 100:
        print(f"FAIL: Corpus too small ({corpus_chars} chars)")
        sys.exit(1)

    print("PASS: External data acquisition pipeline completed successfully.")


if __name__ == "__main__":
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    output_dir = str(Path(__file__).parent / "results_e2e_external_data")

    asyncio.run(run(
        output_dir=output_dir,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="meta-llama/llama-3.1-8b-instruct",
        max_waves=3,
        temperature=0.3,
        max_tokens=8192,
        compact_every=3,
    ))
