#!/usr/bin/env python3
"""Live GossipSwarm end-to-end test against the Venice API.

Runs the full 4-worker, 3-round gossip swarm on a multi-domain biomedical
corpus (Trenbolone / insulin resistance / PAG stress / sulfation /
GH-AMPK rhythm / astrocytic ECM), with corpus expansion deltas between
gossip rounds (KATP material after round 1, NAC/STS material after round 2).

This is not a unit test — it hits a live API and writes human-readable
results to ``docs/test_results/live_system_test/``.

Usage:
    VENICE_MODEL=deepseek-v3.2 python tests/live_swarm_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarm.config import SwarmConfig  # noqa: E402
from swarm.engine import GossipSwarm  # noqa: E402

VENICE_API_URL = "https://api.venice.ai/api/v1/chat/completions"
VENICE_API_KEY = os.environ.get("VENICE_API_KEY")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "deepseek-v3.2")

RESULTS_DIR = REPO_ROOT / "docs" / "test_results" / "live_system_test"


# =============================================================================
# Corpus — 6 sections
# =============================================================================

CORPUS_INITIAL = """\
## Section 1: Trenbolone and mTORC1 Signaling

Trenbolone acetate hydrolyses to 19-nor-delta-9,11-testosterone, which binds the androgen
receptor (AR) with high affinity (Kd ~0.3-0.6 nM) and resists 5α-reduction and aromatization.
Ligand-bound AR drives two mTORC1 activation routes:

1. Pharmacological (IRS-1-independent): AR recruits c-Src via its ligand-binding domain,
   inducing Src autophosphorylation at Tyr416. Src phosphorylates the p85 regulatory subunit
   of PI3K at Tyr468, enabling PI3K → PDK1 → Akt activation without IRS-1 tyrosine
   phosphorylation. Akt → TSC2 (inhibitory phosphorylation at Thr1462) → Rheb-GTP → mTORC1.
2. Nutritional (IRS-1-dependent): Insulin → IRS-1(Tyr) → PI3K → Akt → TSC2 inhibition → mTORC1.
3. Mechanical: PLD-derived phosphatidic acid (PA) displaces FKBP38 from mTOR at the lysosomal
   surface, potentiating mTORC1 regardless of upstream PI3K state.

These three routes are not additive in practice — they share downstream Rheb/mTOR capacity.

## Section 2: Insulin Resistance via Serine Phosphorylation

Chronic AR activation upregulates SOCS-3 and induces TNF-α / IL-6 in skeletal muscle. JNK and
IKKβ, recruited by these cytokines, phosphorylate IRS-1 at Ser307 (human) / Ser312 (rodent).
In parallel, AR-induced PKCθ/PKCζ (sustained DAG accumulation) targets IRS-1 Ser636/639
(human) / Ser632/635 (rodent), and mTORC1/S6K1 feedback phosphorylates IRS-1 Ser1101.

Serine phosphorylation sterically blocks IRS-1 binding to the p85 subunit of PI3K and promotes
dissociation of the IRS-1/p85 complex, attenuating canonical insulin → PI3K → Akt signaling.
The net outcome is impaired GLUT4 translocation and reduced glycogen synthase activity —
classic peripheral insulin resistance.

Chronic trenbolone therefore creates a specific molecular bind: maximal hypertrophy requires
both androgen (AR → satellite cell activation, MyoD/myogenin, ribosomal biogenesis) and
insulin (Akt → mTORC1 → cap-dependent translation), but AR-driven serine phosphorylation
attenuates the insulin arm, creating dependency on supraphysiological insulin to overcome
the blockade.

## Section 3: PAG Columnar Physiology

The periaqueductal gray (PAG) is organized into longitudinal columns with distinct defensive
roles:

- dorsolateral PAG (dlPAG): active defense (fight/flight) — sympathetic activation,
  catecholamine release, vocalization, confrontational behaviour.
- ventrolateral PAG (vlPAG): passive defense (freeze/shutdown) — parasympathetic surge,
  endogenous opioid release, analgesia, immobility.

Column switching: healthy organisms alternate between active and passive coping depending on
threat controllability. Chronic stress causes column-switching dysfunction — animals get stuck
in one mode regardless of context.

PAG integrates:
- ascending interoceptive signals from NTS / parabrachial
- descending cortical input (mPFC → PAG)
- catecholamine input from locus coeruleus
- opioidergic modulation from enkephalin/dynorphin systems

Catecholamine clearance in PAG depends on three enzymes: COMT (methylation, uses SAM),
SULT1A3/PST-M (sulfation, uses PAPS), and MAO (oxidative deamination). CaMKIIδ in PAG
projection neurons integrates Ca²⁺ signals and receives input from insulin/IGF-1 via IRS-1.

## Section 4: Sulfation / PAPS Pool Dynamics

PAPS (3'-phosphoadenosine-5'-phosphosulfate) is the universal sulfate donor for every
sulfotransferase isoform. Synthesis: cysteine → sulfate (via cysteine dioxygenase, sulfite
oxidase) → PAPS (via PAPSS1/2 — PAPS synthase).

SULT1A3 preferentially sulfates catecholamines (dopamine, norepinephrine); sulfation
terminates signaling and promotes renal clearance.

Competition: catecholamine sulfation vs. estrogen/xenobiotic sulfation vs. proteoglycan
sulfation (heparan sulfate, chondroitin sulfate) — all draw from the same finite PAPS pool.

Depletion of sulfate/PAPS slows ALL sulfation reactions globally. Tissues with high
catecholamine flux (PAG, adrenal medulla, sympathetic ganglia) are most vulnerable to PAPS
depletion under chronic sympathetic drive.

mTORC1 ↔ sulfur crosstalk: S6K1 can transcriptionally regulate SULT1A3 expression; mTORC1
activation broadly increases sulfotransferase translation; AMPK (mTORC1 antagonist)
regulates cysteine dioxygenase activity and thus sulfate supply.

## Section 5: GH-AMPK Inhale/Exhale Rhythm

Growth hormone pulses drive a ~2-3 h oscillation between anabolic (mTORC1-high, AMPK-low)
and catabolic (AMPK-high, mTORC1-low) states:

- GH peak → AMPK up (via CaMKKβ / LKB1) → TSC2 activation (Thr1462, GAP-enhancing) → mTORC1
  down. AMPK also phosphorylates Raptor at Ser792, directly inhibiting mTORC1 assembly.
- GH nadir → insulin/Akt predominates → TSC2 inhibition → mTORC1 up.

Temporal phases of trenbolone pharmacology:
- 0-6 h: AR → Src → p85 bypass dominates (mTORC1 active)
- 6-24 h: GH/AMPK oscillation dominates (net catabolic windows re-emerge)
- 24-48 h: insulin/nutrient signaling restoration (if IRS-1 serine load permits)

The inhale/exhale rhythm is essential for protein-synthesis efficiency — chronic mTORC1
suppression of autophagy without AMPK cycles leads to misfolded protein accumulation.

## Section 6: Astrocytic ECM and LOX Cross-Linking

PAG astrocytes regulate extracellular K⁺ and glutamate, release ATP via Panx1 hemichannels,
and generate Ca²⁺ waves that coordinate column switching across PAG projection neurons.

ECM stiffness — determined largely by collagen/elastin cross-linking by lysyl oxidase (LOX)
— affects astrocytic process motility and therefore column-switch plasticity. Chronic
sympathetic drive elevates LOX expression (via AR, TGF-β, and hypoxia signaling),
progressively stiffening the PAG ECM and reducing astrocytic process motility.

A stiff ECM mechanically locks astrocytic territories, constraining the Ca²⁺ wave
reconfigurations needed to switch the neuronal column from dl → vl activity. This is a slow
structural change (days-weeks) that compounds the faster sulfation / catecholamine-clearance
bottleneck.
"""


DELTA_AFTER_ROUND_1 = """\
## Delta (round 1 → round 2): KATP / Kir3.2 channels in vlPAG

Newly surfaced material:

- KATP channels (Kir6.2 / SUR1 composition) in vlPAG GABAergic interneurons are sensitive to
  the ATP/ADP ratio and act as local metabolic-state sensors. When open they hyperpolarize
  the GABAergic interneuron pool, disinhibiting vlPAG opioid neurons → freeze / shutdown.
- AMPK directly phosphorylates KATP subunits, biasing them toward the open state. Therefore
  AMPK activity is a prerequisite for entering vlPAG (passive defense) mode.
- Kir3.2 (GIRK) channels in vlPAG are opened postsynaptically by μ/δ-opioid receptor
  activation; they mediate the postsynaptic inhibition that enforces the freeze response.
  AMPK potentiates Kir3.2 conductance.
- Implication: chronic mTORC1 dominance (low AMPK) starves BOTH KATP opening AND Kir3.2
  potentiation, so vlPAG cannot engage the passive-defense program even when catecholamine
  tone is clamped — the organism sits in dlPAG by default.
"""


DELTA_AFTER_ROUND_2 = """\
## Delta (round 2 → round 3): NAC, sulfate supplementation, and STS

Newly surfaced intervention material:

- N-acetylcysteine (NAC) is deacetylated to cysteine, which feeds cysteine dioxygenase →
  sulfate → PAPS. NAC at 600-1800 mg/day measurably elevates plasma sulfate in humans within
  48 h; brain PAPS restoration has been shown in rodent models at equivalent doses within
  72 h.
- Inorganic sulfate (Na₂SO₄, MgSO₄) bypasses the cysteine-dioxygenase step. Epsom-salt oral
  loading raises plasma sulfate faster than NAC but has GI tolerability limits.
- STS (sodium thiosulfate) is a sulfide donor that enters the mitochondrial sulfur pool via
  SQOR and rhodanese; it indirectly contributes to sulfate via persulfide oxidation.
- Crucially, NAC also replenishes glutathione and lowers TNF-α / IL-6 expression,
  simultaneously reducing the cytokine → JNK / IKKβ → IRS-1 Ser307 arm of insulin resistance
  (i.e. NAC targets two bottlenecks at once).
- The key falsifiable prediction: if PAPS depletion is the dominant bottleneck for column
  locking, inorganic-sulfate loading should restore column-switching flexibility within
  3-5 days of supplementation, measurable as a partial return of vlPAG-dominant responses to
  controllable stressors, BEFORE LOX-driven ECM stiffness has time to reverse (ECM remodelling
  takes weeks). A dissociation between rapid behavioral rescue (days) and slow ECM
  normalization (weeks) would localize the bottleneck to sulfation rather than ECM.
"""


QUERY = (
    "Chronic Trenbolone use appears to lock organisms into the dlPAG (fight/flight) PAG "
    "column. Synthesize the full molecular mechanism across these six domains — "
    "Trenbolone / mTORC1, IRS-1 serine phosphorylation / insulin resistance, PAG columnar "
    "physiology, PAPS / sulfation, GH-AMPK inhale-exhale rhythm, and astrocytic ECM — and "
    "propose a single upstream intervention target that would most efficiently restore "
    "column-switching plasticity, with a falsifiable biomarker or behavioural readout."
)


# =============================================================================
# Venice API adapter
# =============================================================================

class VeniceStats:
    calls: int = 0
    total_latency: float = 0.0
    errors: int = 0


def make_venice_complete(model: str, concurrency: int):
    """Build an async `(prompt: str) -> str` completion fn that calls Venice."""
    sem = asyncio.Semaphore(concurrency)

    async def complete(prompt: str) -> str:
        async with sem:
            print(f"  [venice] -> call, prompt={len(prompt)} chars", flush=True)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2500,
                # DeepSeek-V3.2 on Venice is a reasoning model — by default
                # it spends most of the output-token budget on `reasoning_content`
                # and returns little / empty `content`. For a pipeline-mechanics
                # test we disable the reasoning pass so calls return quickly
                # and the `content` field is populated directly. The user
                # explicitly authorized this ("validating pipeline mechanics,
                # not model quality").
                "venice_parameters": {
                    "disable_thinking": True,
                    "strip_thinking_response": True,
                    "include_venice_system_prompt": False,
                },
            }
            headers = {
                "Authorization": f"Bearer {VENICE_API_KEY}",
                "Content-Type": "application/json",
            }
            # Retry loop for 429/5xx
            last_err: Exception | None = None
            for attempt in range(6):
                start = time.monotonic()
                try:
                    # DeepSeek-V3.2 via Venice routinely takes 3-5 minutes for a
                    # 2-3K-token completion on worker synthesis / gossip prompts
                    # (~10KB of generated text), so we budget generously.
                    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                        resp = await client.post(VENICE_API_URL, json=payload, headers=headers)
                    elapsed = time.monotonic() - start
                    if resp.status_code == 200:
                        VeniceStats.calls += 1
                        VeniceStats.total_latency += elapsed
                        data = resp.json()
                        choice = data["choices"][0]["message"]
                        content = choice.get("content") or ""
                        usage = data.get("usage", {})
                        print(
                            f"  [venice] <- 200 in {elapsed:.1f}s | "
                            f"in={usage.get('prompt_tokens')} "
                            f"out={usage.get('completion_tokens')} "
                            f"content_chars={len(content)}",
                            flush=True,
                        )
                        # DeepSeek-R1 puts chain of thought in "reasoning_content"
                        # and final answer in "content". Both the <think>...</think>
                        # style and separate field style are seen in the wild.
                        # We keep just the final `content`; callers want answers.
                        return content
                    if resp.status_code in (408, 429, 500, 502, 503, 504):
                        backoff = min(60, 5 * (attempt + 1))
                        print(
                            f"  Venice {resp.status_code} (attempt {attempt+1}): "
                            f"backing off {backoff}s",
                            flush=True,
                        )
                        last_err = RuntimeError(
                            f"Venice {resp.status_code}: {resp.text[:300]}"
                        )
                        await asyncio.sleep(backoff)
                        continue
                    # Non-retryable
                    raise RuntimeError(
                        f"Venice {resp.status_code}: {resp.text[:500]}"
                    )
                except (httpx.RequestError, httpx.ReadTimeout, httpx.ConnectError) as e:
                    backoff = min(60, 5 * (attempt + 1))
                    print(
                        f"  Venice net-error ({type(e).__name__}) "
                        f"(attempt {attempt+1}): {e!r}; backing off {backoff}s",
                        flush=True,
                    )
                    last_err = e
                    await asyncio.sleep(backoff)
                    continue
            VeniceStats.errors += 1
            raise RuntimeError(f"Venice call failed after retries: {last_err}")

    return complete


# =============================================================================
# Test harness
# =============================================================================

async def run_swarm() -> dict:
    if not VENICE_API_KEY:
        raise SystemExit("VENICE_API_KEY not set in environment")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Corpus starts as the 6-section initial text; a delta queue is drained by
    # `corpus_delta_fn` (the engine calls this between gossip rounds).
    pending_deltas: list[str] = [DELTA_AFTER_ROUND_1, DELTA_AFTER_ROUND_2]

    async def corpus_delta_fn() -> str:
        if pending_deltas:
            return pending_deltas.pop(0)
        return ""

    config = SwarmConfig(
        max_workers=4,
        max_concurrency=2,   # throttle live API to avoid rate limits
        gossip_rounds=3,
        min_gossip_rounds=3,  # force full 3 rounds even if convergence early
        max_gossip_rounds=3,  # hard cap — engine defaults this to 10 otherwise
        enable_serendipity=True,
        enable_adaptive_rounds=False,  # force full 3 rounds regardless
        enable_semantic_assignment=True,
        enable_full_corpus_gossip=True,
        enable_diversity_aware_gossip=True,
        corpus_delta_fn=corpus_delta_fn,
        worker_temperature=0.3,
        queen_temperature=0.3,
        # DeepSeek-V3.2 on Venice is slow (~100 tok/s effective). Cap output
        # tightly so each call finishes in under ~3-4 minutes.
        worker_max_tokens=2500,
        queen_max_tokens=4000,
        max_summary_chars=6000,
        max_section_chars=20000,
    )

    complete_fn = make_venice_complete(VENICE_MODEL, concurrency=config.max_concurrency)
    swarm = GossipSwarm(complete=complete_fn, config=config)

    events: list[dict] = []

    async def on_event(evt: dict) -> None:
        evt = {**evt, "_recv_utc": datetime.now(timezone.utc).isoformat()}
        events.append(evt)
        kind = evt.get("type") or evt.get("phase") or "event"
        round_n = evt.get("round")
        extra = f" round={round_n}" if round_n is not None else ""
        print(f"  [event] {kind}{extra}", flush=True)

    print(f"=== LIVE SWARM TEST ({VENICE_MODEL} via Venice) ===", flush=True)
    print(f"Corpus: {len(CORPUS_INITIAL)} chars, 6 sections", flush=True)
    print(
        f"Config: workers={config.max_workers} concurrency={config.max_concurrency} "
        f"gossip_rounds={config.gossip_rounds} serendipity={config.enable_serendipity}",
        flush=True,
    )

    start = time.monotonic()
    result = await swarm.synthesize(
        corpus=CORPUS_INITIAL,
        query=QUERY,
        on_event=on_event,
    )
    elapsed = time.monotonic() - start

    print("", flush=True)
    print("=== DONE ===", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"Venice calls: {VeniceStats.calls}", flush=True)
    print(
        f"Venice avg latency: "
        f"{VeniceStats.total_latency / max(1, VeniceStats.calls):.2f}s",
        flush=True,
    )
    print(f"Angles: {result.angles_detected}", flush=True)
    print(
        f"Gossip rounds executed: {result.metrics.gossip_rounds_executed} "
        f"(configured: {result.metrics.gossip_rounds_configured})",
        flush=True,
    )
    print(f"Total LLM calls (engine-reported): {result.metrics.total_llm_calls}", flush=True)
    print(
        f"User report: {len(result.user_report)} chars | "
        f"Knowledge report: {len(result.knowledge_report)} chars | "
        f"Serendipity: {len(result.serendipity_insights)} chars",
        flush=True,
    )

    return {
        "elapsed_s": elapsed,
        "venice_calls": VeniceStats.calls,
        "venice_avg_latency_s": (
            VeniceStats.total_latency / VeniceStats.calls if VeniceStats.calls else 0.0
        ),
        "venice_errors": VeniceStats.errors,
        "events": events,
        "result": result,
    }


def _metrics_as_dict(metrics) -> dict:
    return {
        "total_llm_calls": metrics.total_llm_calls,
        "total_workers": metrics.total_workers,
        "gossip_rounds_configured": metrics.gossip_rounds_configured,
        "gossip_rounds_executed": metrics.gossip_rounds_executed,
        "gossip_converged_early": metrics.gossip_converged_early,
        "serendipity_produced": metrics.serendipity_produced,
        "phase_times": metrics.phase_times,
        "worker_input_chars": metrics.worker_input_chars,
        "worker_output_chars": metrics.worker_output_chars,
        "total_elapsed_s": metrics.total_elapsed_s,
        "gossip_info_gain": metrics.gossip_info_gain,
        "degradations": metrics.degradations,
    }


def save_results(run: dict) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result = run["result"]
    metrics = _metrics_as_dict(result.metrics)
    run_id = f"live_swarm_{VENICE_MODEL}_{ts}".replace("/", "_")

    # JSON summary (no raw prompts — just outputs + metrics + events)
    json_path = RESULTS_DIR / f"{run_id}.json"
    json_payload = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": VENICE_MODEL,
        "venice_api_url": VENICE_API_URL,
        "elapsed_s": run["elapsed_s"],
        "venice_calls": run["venice_calls"],
        "venice_avg_latency_s": run["venice_avg_latency_s"],
        "venice_errors": run["venice_errors"],
        "corpus_initial_chars": len(CORPUS_INITIAL),
        "corpus_deltas_chars": [len(DELTA_AFTER_ROUND_1), len(DELTA_AFTER_ROUND_2)],
        "query": QUERY,
        "angles_detected": result.angles_detected,
        "engine_metrics": metrics,
        "worker_summaries": result.worker_summaries,
        "serendipity_insights": result.serendipity_insights,
        "user_report": result.user_report,
        "knowledge_report": result.knowledge_report,
        "events": run["events"],
    }
    with json_path.open("w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Wrote {json_path}", flush=True)

    # Markdown report
    md_path = RESULTS_DIR / "LIVE_SWARM_TEST_RESULTS.md"
    with md_path.open("w") as f:
        f.write("# Live GossipSwarm End-to-End Test\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  \n")
        f.write(f"**Model:** `{VENICE_MODEL}` via Venice API  \n")
        f.write(
            "**Config:** max_workers=4, max_concurrency=2, gossip_rounds=3, "
            "min_gossip_rounds=3, adaptive_rounds=OFF, serendipity=ON, "
            "corpus deltas=2 (KATP material after round 1, NAC/STS after round 2)  \n"
        )
        f.write(
            f"**Corpus:** 6 sections, {len(CORPUS_INITIAL)} chars initial + "
            f"{len(DELTA_AFTER_ROUND_1)} chars delta1 + {len(DELTA_AFTER_ROUND_2)} "
            f"chars delta2  \n\n"
        )
        f.write("## Runtime\n\n")
        f.write(f"- Wall time: **{run['elapsed_s']:.1f}s**\n")
        f.write(f"- Venice calls: **{run['venice_calls']}**\n")
        f.write(
            f"- Venice avg latency: **{run['venice_avg_latency_s']:.2f}s**\n"
        )
        f.write(f"- Venice retry errors: {run['venice_errors']}\n")
        f.write(f"- Engine-reported LLM calls: {metrics['total_llm_calls']}\n")
        f.write(
            f"- Gossip rounds executed: {metrics['gossip_rounds_executed']} "
            f"(configured {metrics['gossip_rounds_configured']}, "
            f"converged_early={metrics['gossip_converged_early']})\n"
        )
        f.write(f"- Serendipity produced: {metrics['serendipity_produced']}\n\n")
        f.write("### Phase times\n\n")
        f.write("| Phase | Seconds |\n|---|---:|\n")
        for phase, secs in metrics["phase_times"].items():
            f.write(f"| {phase} | {secs:.2f} |\n")
        f.write("\n## Angles detected\n\n")
        for a in result.angles_detected:
            f.write(f"- {a}\n")
        f.write("\n## Worker summaries (final gossip-round output)\n\n")
        for ang, summary in result.worker_summaries.items():
            f.write(f"### {ang}\n\n")
            f.write(summary or "*(empty)*")
            f.write("\n\n")
        f.write("## Serendipity insights\n\n")
        f.write(result.serendipity_insights or "*(none)*")
        f.write("\n\n## Queen user-report\n\n")
        f.write(result.user_report or "*(empty)*")
        f.write("\n\n## Queen knowledge-report\n\n")
        f.write(result.knowledge_report or "*(empty)*")
        f.write("\n")
    print(f"Wrote {md_path}", flush=True)


def main() -> None:
    run = asyncio.run(run_swarm())
    save_results(run)


if __name__ == "__main__":
    main()
