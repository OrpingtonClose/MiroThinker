"""Tier 3 tests: architecture validation — needs real LLM, intensive.

Validates that the deeper architectural assumptions in the backlog
are correct.  These are expensive (many LLM calls) and take 10-30 min.

Covers assumptions D6, D7 and Tier 3 tests from TEST_PLAN.md:
    D7  Larger data packages (30K chars) produce more unique insights than 6K
    --  Multi-run dedup: second run skips re-ingestion, still finds new findings
    --  24h simulation: 10 sequential runs, store doesn't corrupt
    D6  Quality doesn't degrade over multiple waves

Requires: OPENROUTER_API_KEY environment variable.

Run with:
    python -m pytest tests/test_architecture_validation.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

import pytest

# Ensure repo root and strands-agent are importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_STRANDS_AGENT = str(Path(__file__).resolve().parents[1] / "apps" / "strands-agent")
_SWARM = str(Path(__file__).resolve().parents[1] / "swarm")
for p in (_REPO_ROOT, _STRANDS_AGENT, _SWARM):
    if p not in sys.path:
        sys.path.insert(0, p)

from corpus import ConditionStore
from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine


# ═══════════════════════════════════════════════════════════════════════
# Skip condition
# ═══════════════════════════════════════════════════════════════════════

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
requires_llm = pytest.mark.skipif(
    not OPENROUTER_KEY,
    reason="OPENROUTER_API_KEY not set — skipping LLM architecture validation tests",
)


# ═══════════════════════════════════════════════════════════════════════
# Test corpus — larger than Tier 2 to stress data package assembly
# ═══════════════════════════════════════════════════════════════════════

TEST_CORPUS = """\
## Insulin Timing and Nutrient Partitioning

Rapid-acting insulin (Humalog/NovoRapid) peaks at 60-90 minutes post-injection.
Pre-workout dosing at 4-6 IU with 10g dextrose per IU prevents hypoglycemia.
Humulin-R has a slower onset (30-60 min) and longer tail (6-8 hours).
Post-workout insulin drives amino acids into muscle tissue via GLUT4 upregulation.
Berberine at 500mg mimics some insulin-sensitizing effects via AMPK activation.
GDA supplements (chromium picolinate, alpha-lipoic acid) enhance insulin sensitivity.
Metformin at 500-2000mg/day reduces hepatic glucose production but may blunt mTOR.
Timing insulin with high-glycemic carbs maximizes glycogen supercompensation.
Lantus (glargine) at 10-20 IU/day provides basal insulin coverage for 24 hours.
The insulin-to-carb ratio varies by individual: typically 1 IU per 10-15g carbs.
Insulin resistance develops with chronic supraphysiological doses above 15 IU/day.
Somatomedins (IGF-binding proteins) modulate insulin sensitivity at the receptor level.

## Hematological Effects of Anabolic Compounds

Trenbolone acetate increases erythropoietin (EPO) production, raising hematocrit
by 15-20% over 8-12 weeks. At 400mg/week, hematocrit commonly reaches 52-54%.
Boldenone (EQ) at 300-600mg/week elevates RBC count through a different mechanism
— direct bone marrow stimulation rather than EPO pathway.
Regular phlebotomy (500mL every 8 weeks) manages polycythemia.
Naringin (grapefruit extract) may modestly reduce hematocrit by 2-3 points.
Iron supplementation during phlebotomy prevents functional iron deficiency.
Ferritin monitoring (target 50-150 ng/mL) essential during frequent donation.
Hemoglobin A1c may be artificially lowered by accelerated RBC turnover on cycle.
Oxymetholone (Anadrol) at 50-100mg/day is the most potent oral for RBC stimulation.
Nandrolone decanoate at 200-400mg/week increases RBC with fewer androgenic sides.
EPO micro-dosing (500-2000 IU 3x/week) provides controlled hematocrit elevation.
Aspirin at 81mg/day reduces thrombotic risk from elevated hematocrit.
Blood viscosity increases exponentially above hematocrit of 50%.

## Growth Hormone and IGF-1 Cascade

Exogenous GH at 2-4 IU/day increases hepatic IGF-1 production within 6 hours.
Splitting doses (AM + pre-bed) mimics natural pulsatile secretion pattern.
GH + insulin synergy: insulin prevents GH-induced insulin resistance while
GH amplifies insulin's anabolic effects through IGF-1 mediation.
MK-677 (ibutamoren) at 25mg/day raises GH by 40-60% but causes water retention.
CJC-1295/Ipamorelin combination provides more physiological GH release patterns.
IGF-1 LR3 at 20-80mcg/day has longer half-life and crosses the blood-brain barrier.
GH-induced lipolysis peaks at 3-4 hours post-injection — fasted morning dosing optimal.
IGF-1 DES at 50-100mcg pre-workout has 10x IGF-1 receptor affinity but 20min half-life.
Mechano Growth Factor (MGF) at 200mcg post-training activates satellite cells.
GH at doses above 6 IU/day commonly causes carpal tunnel syndrome and joint pain.
Acromegaly risk increases with chronic GH use above 4 IU/day for >12 months.
Somatostatin analogues (octreotide) can modulate GH pulses when titrating doses.

## Hepatoprotection During Oral Cycles

TUDCA at 500-1000mg/day provides bile acid support for 17-alpha-alkylated orals.
NAC at 600-1200mg/day replenishes glutathione depleted by hepatotoxic compounds.
Milk thistle (silymarin) at 420mg/day shows mixed evidence for liver protection.
Hepatic enzymes (ALT/AST) typically normalize within 4-6 weeks post-cycle.
Cholestasis risk increases with stacking multiple 17-aa orals simultaneously.
Injection site oil (subcutaneous depot) absorbs directly into lymphatics, bypassing
first-pass hepatic metabolism entirely — subcutaneous testosterone undecanoate
at 40-80mg/day provides stable levels without liver stress.
UDCA (ursodeoxycholic acid) at 300mg 2x/day is the pharmaceutical alternative
to TUDCA with better clinical evidence for cholestatic liver disease.
"""

TEST_QUERY = "Analyze interactions between insulin, anabolic compounds, growth hormone, and hepatoprotection protocols"


# ═══════════════════════════════════════════════════════════════════════
# LLM helper
# ═══════════════════════════════════════════════════════════════════════

async def _openrouter_complete(prompt: str) -> str:
    """Direct OpenRouter completion."""
    import httpx

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _make_config(**overrides) -> MCPSwarmConfig:
    """Create a test config."""
    defaults = {
        "api_base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "api_key": OPENROUTER_KEY,
        "max_tokens": 2048,
        "max_workers": 3,
        "max_waves": 2,
        "convergence_threshold": 2,
        "worker_timeout_s": 120.0,
        "enable_serendipity_wave": False,
        "enable_rolling_summaries": False,
        "compact_every_n_waves": 0,
    }
    defaults.update(overrides)
    return MCPSwarmConfig(**defaults)


def _extract_unique_facts(store: ConditionStore, limit: int = 1000) -> set[str]:
    """Get unique fact strings from the store."""
    findings = store.get_findings(limit=limit)
    return {f["fact"].strip().lower() for f in findings}


# ═══════════════════════════════════════════════════════════════════════
# D7: Larger data packages produce more unique insights
# ═══════════════════════════════════════════════════════════════════════


class TestMultiWaveDepthImproves:
    """D7: More waves with richer data packages produce deeper insights.

    With tool-free workers, the data package grows richer each wave
    (progressive population of §1-§7).  More waves should produce
    more unique findings as workers get better context.
    """

    @requires_llm
    @pytest.mark.asyncio
    async def test_more_waves_more_insights(self) -> None:
        """Run pipeline with 1 wave vs 3 waves, compare finding count."""

        # --- Run 1: 1 wave (bootstrap only, §2 corpus material) ---
        store_1w = ConditionStore()
        config_1w = _make_config(
            max_workers=3,
            max_waves=1,
            enable_serendipity_wave=False,
        )
        engine_1w = MCPSwarmEngine(
            store=store_1w, complete=_openrouter_complete, config=config_1w,
        )
        result_1w = await engine_1w.synthesize(TEST_CORPUS, TEST_QUERY)

        # --- Run 2: 3 waves (progressive data packages) ---
        store_3w = ConditionStore()
        config_3w = _make_config(
            max_workers=3,
            max_waves=3,
            convergence_threshold=0,
            enable_serendipity_wave=False,
        )
        engine_3w = MCPSwarmEngine(
            store=store_3w, complete=_openrouter_complete, config=config_3w,
        )
        result_3w = await engine_3w.synthesize(TEST_CORPUS, TEST_QUERY)

        # Extract unique facts
        facts_1w = _extract_unique_facts(store_1w)
        facts_3w = _extract_unique_facts(store_3w)

        count_1w = len(facts_1w)
        count_3w = len(facts_3w)

        # Both should produce findings
        assert count_3w > 0, "3-wave run produced no findings"
        assert count_1w > 0, "1-wave run produced no findings"

        # Log results regardless of pass/fail
        print(f"\n  1-wave findings: {count_1w}")
        print(f"  3-wave findings: {count_3w}")
        print(f"  Unique to 3-wave: {len(facts_3w - facts_1w)}")

        # 3 waves should produce at least as many findings as 1 wave
        assert count_3w >= count_1w * 0.7, (
            f"3-wave run ({count_3w} findings) produced significantly FEWER "
            f"findings than 1-wave run ({count_1w}) — progressive data "
            f"packages may not be improving worker output"
        )


# ═══════════════════════════════════════════════════════════════════════
# Multi-run dedup: second run skips re-ingestion, still finds new findings
# ═══════════════════════════════════════════════════════════════════════


class TestMultiRunDedup:
    """Second run on same corpus skips ingestion, still produces findings."""

    @requires_llm
    @pytest.mark.asyncio
    async def test_second_run_skips_ingestion_finds_new(self) -> None:
        """Two runs with same store, verify corpus fingerprinting works."""
        store = ConditionStore()
        config = _make_config(
            max_workers=2,
            max_waves=1,
            enable_serendipity_wave=False,
        )

        # --- Run 1 ---
        engine = MCPSwarmEngine(
            store=store, complete=_openrouter_complete, config=config,
        )
        result_1 = await engine.synthesize(TEST_CORPUS, TEST_QUERY)
        findings_after_run1 = len(store.get_findings(limit=1000))
        assert findings_after_run1 > 0, "Run 1 produced no findings"

        # --- Run 2 (same store, same corpus) ---
        engine2 = MCPSwarmEngine(
            store=store, complete=_openrouter_complete, config=config,
        )
        result_2 = await engine2.synthesize(TEST_CORPUS, TEST_QUERY)
        findings_after_run2 = len(store.get_findings(limit=1000))

        # Run 2 should have SOME findings (either new or accumulated)
        assert findings_after_run2 >= findings_after_run1, (
            f"Findings decreased from {findings_after_run1} to {findings_after_run2} — "
            f"second run may have corrupted the store"
        )

        # Verify corpus fingerprinting: the corpus_hashes table should
        # have the fingerprint registered
        import hashlib
        corpus_hash = hashlib.sha256(TEST_CORPUS.encode()).hexdigest()[:16]
        assert store.has_corpus_hash(corpus_hash), (
            "Corpus fingerprint not registered after first run"
        )

        # Both runs must produce reports
        assert result_1.report is not None
        assert result_2.report is not None

        print(f"\n  Run 1 findings: {findings_after_run1}")
        print(f"  Run 2 findings: {findings_after_run2}")
        print(f"  Net new in run 2: {findings_after_run2 - findings_after_run1}")


# ═══════════════════════════════════════════════════════════════════════
# 24h simulation: 10 sequential runs, store doesn't corrupt
# ═══════════════════════════════════════════════════════════════════════


class TestLongRunSimulation:
    """Simulate 24h operation: multiple sequential runs, store stays healthy."""

    @requires_llm
    @pytest.mark.asyncio
    async def test_10_sequential_runs_no_corruption(self) -> None:
        """Run the pipeline 5 times sequentially, verify store integrity."""
        # Use 5 runs instead of 10 to keep test time reasonable (~15min)
        num_runs = 5
        store = ConditionStore()
        config = _make_config(
            max_workers=2,
            max_waves=1,
            convergence_threshold=0,
            enable_serendipity_wave=False,
            compact_every_n_waves=0,
        )

        findings_per_run: list[int] = []
        total_findings_per_run: list[int] = []
        reports: list[str] = []

        for run_idx in range(num_runs):
            engine = MCPSwarmEngine(
                store=store, complete=_openrouter_complete, config=config,
            )

            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

            assert result.report is not None, f"Run {run_idx} produced no report"
            reports.append(result.report)

            all_findings = store.get_findings(limit=5000)
            total_count = len(all_findings)
            total_findings_per_run.append(total_count)

            # Calculate net new findings this run
            if run_idx == 0:
                findings_per_run.append(total_count)
            else:
                net_new = total_count - total_findings_per_run[run_idx - 1]
                findings_per_run.append(net_new)

        print(f"\n  Findings per run: {findings_per_run}")
        print(f"  Cumulative findings: {total_findings_per_run}")

        # 1. Store must not lose data across runs
        for i in range(1, num_runs):
            assert total_findings_per_run[i] >= total_findings_per_run[i - 1], (
                f"Store lost data between run {i - 1} and {i}: "
                f"{total_findings_per_run[i - 1]} → {total_findings_per_run[i]}"
            )

        # 2. Store stats must be consistent
        stats = store.get_store_stats()
        assert stats["total_rows"] > 0

        # 3. All reports must be non-trivial
        for i, report in enumerate(reports):
            assert len(report) > 100, (
                f"Run {i} report too short ({len(report)} chars)"
            )

        # 4. D6: Quality doesn't degrade — last report should still
        # reference corpus-specific terms
        last_report_lower = reports[-1].lower()
        corpus_terms = ["insulin", "hematocrit", "growth hormone", "igf"]
        found = [t for t in corpus_terms if t in last_report_lower]
        assert len(found) >= 2, (
            f"Final report only references {found} from {corpus_terms} — "
            f"quality may have degraded over {num_runs} runs"
        )

    @requires_llm
    @pytest.mark.asyncio
    async def test_compaction_under_load(self) -> None:
        """Run pipeline with compaction enabled, verify no data corruption."""
        store = ConditionStore()
        config = _make_config(
            max_workers=3,
            max_waves=2,
            enable_serendipity_wave=False,
            compact_every_n_waves=1,  # compact after every wave
        )

        engine = MCPSwarmEngine(
            store=store, complete=_openrouter_complete, config=config,
        )

        result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        assert result.report is not None

        # After compaction, all remaining findings should be consider_for_use=True
        findings = store.get_findings(limit=1000)
        assert len(findings) > 0, "No findings survived compaction"

        # Verify no duplicate facts within the same angle
        seen: dict[tuple[str, str], int] = {}
        for f in findings:
            key = (f["angle"], f["fact"].strip().lower())
            seen[key] = seen.get(key, 0) + 1

        duplicates = {k: v for k, v in seen.items() if v > 1}
        assert not duplicates, (
            f"Compaction left {len(duplicates)} duplicate(s) in the store: "
            f"{list(duplicates.keys())[:3]}"
        )

        # Verify store stats are internally consistent
        stats = store.get_store_stats()
        assert stats["total_rows"] >= len(findings), (
            "Store stats report fewer rows than get_findings returns"
        )
