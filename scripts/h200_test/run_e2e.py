#!/usr/bin/env python3
"""Run a single E2E swarm query with organic angle detection.

No hardcoded angles. No trimming. Serendipity always on.
The prompt determines everything.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure swarm/ and apps/ are importable
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


def load_corpus(path: str) -> str:
    """Load corpus from a text file."""
    with open(path) as f:
        return f.read()


async def run(
    corpus_path: str,
    output_dir: str,
    api_base: str,
    api_key: str,
    model: str,
    max_waves: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 8192,
    compact_every: int = 3,
) -> None:
    """Execute a single E2E swarm run."""
    from corpus import ConditionStore

    corpus = load_corpus(corpus_path)
    logger.info("corpus_chars=<%d> | loaded", len(corpus))

    # Fresh store per run
    db_path = str(Path(output_dir) / "store.duckdb")
    store = ConditionStore(db_path=db_path)

    # Build completion function for angle detection + report gen
    # Uses auth-aware httpx client (OpenRouter requires Bearer token)
    import httpx

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
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=600.0) as client:
            try:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                logger.exception(
                    "model=<%s>, url=<%s> | completion call failed", model, url,
                )
                return ""

    config = MCPSwarmConfig(
        max_workers=8,  # upper bound — actual workers = angles detected
        max_waves=max_waves,
        convergence_threshold=5,
        api_base=api_base,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        # No required_angles — organic detection from prompt
        required_angles=[],
        report_max_tokens=16384,
        enable_serendipity_wave=True,
        source_model=model,
        source_run=f"e2e_{time.strftime('%Y%m%d_%H%M%S')}",
        compact_every_n_waves=compact_every,
        enable_rolling_summaries=True,
    )

    engine = MCPSwarmEngine(
        store=store,
        complete=complete_fn,
        config=config,
    )

    logger.info("prompt=<%s> | starting synthesis", PROMPT[:80])

    result = await engine.synthesize(corpus=corpus, query=PROMPT)

    # Save results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    report_path = out / f"report_{ts}.md"
    report_path.write_text(result.report)

    metrics = {
        "prompt": PROMPT,
        "engine": "mcp",
        "model": model,
        "api_base": api_base,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "convergence_threshold": config.convergence_threshold,
        "max_workers": config.max_workers,
        "max_waves": max_waves,
        "serendipity_enabled": True,
        "rolling_summaries_enabled": True,
        "compact_every_n_waves": compact_every,
        "required_angles": [],
        "total_elapsed_s": result.metrics.total_elapsed_s,
        "total_waves": result.metrics.total_waves,
        "total_findings_stored": result.metrics.total_findings_stored,
        "total_tool_calls": result.metrics.total_tool_calls,
        "findings_per_wave": result.metrics.findings_per_wave,
        "phase_times": result.metrics.phase_times,
        "convergence_reason": result.metrics.convergence_reason,
        "angles_detected": result.angles_detected,
        "report_chars": len(result.report),
    }

    metrics_path = out / f"metrics_{ts}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'═' * 60}")
    print(f"  E2E Run Complete")
    print(f"{'═' * 60}")
    print(f"  Prompt:       {PROMPT[:60]}...")
    print(f"  Model:        {model}")
    print(f"  Elapsed:      {result.metrics.total_elapsed_s:.1f}s")
    print(f"  Waves:        {result.metrics.total_waves}")
    print(f"  Findings:     {result.metrics.total_findings_stored}")
    print(f"  Tool calls:   {result.metrics.total_tool_calls}")
    print(f"  Findings/wave:{result.metrics.findings_per_wave}")
    print(f"  Convergence:  {result.metrics.convergence_reason}")
    print(f"  Report:       {len(result.report):,} chars")
    print(f"  Angles:       {result.angles_detected}")
    print(f"  Output:       {out.resolve()}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    corpus_path = str(Path(__file__).parent / "test_corpus.txt")
    output_dir = str(Path(__file__).parent / "results_e2e_baseline")

    asyncio.run(run(
        corpus_path=corpus_path,
        output_dir=output_dir,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        model="meta-llama/llama-3.1-8b-instruct",
        max_waves=3,
        temperature=0.3,
        max_tokens=8192,
        compact_every=3,
    ))
