#!/usr/bin/env python3
"""Execute the employment list — batch swarm runs across disaggregated topics.

Usage:
    # Wave 1: All 48 topics with DeepSeek V3.2
    python run_employment.py --wave 1

    # Wave 2: Cross-validation (12 key topics x alternative models)
    python run_employment.py --wave 2

    # Single topic test
    python run_employment.py --topics A1 --model deepseek-chat --provider deepseek
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure swarm/ and apps/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "apps" / "strands-agent"))

from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Topic definitions (48 topics, 8 clusters)
# ---------------------------------------------------------------------------

TOPICS: dict[str, str] = {
    # Cluster A — Insulin & Metabolic Signaling
    "A1": (
        "Milos Sarcev insulin protocol — exact timing, dosing schedule, "
        "nutrient pairing, GH synchronization for bodybuilding in the "
        "context of taking insulin, hgh and trenbolone based cycles with "
        "full complexity breakdown between all food related nutrients, "
        "supplements and PEDs"
    ),
    "A2": (
        "Rapid-acting insulin analog pharmacokinetics — Humalog vs "
        "NovoRapid vs Apidra onset/peak/duration curves, subcutaneous "
        "absorption variables, temperature and injection site effects"
    ),
    "A3": (
        "GLUT4 translocation in skeletal muscle vs adipose tissue — "
        "insulin-mediated glucose partitioning, exercise-induced "
        "translocation without insulin, contraction signaling via AMPK, "
        "implications for body composition during PED use"
    ),
    "A4": (
        "mTOR pathway activation by insulin — mTORC1 vs mTORC2 "
        "divergence, rapamycin interaction, leucine co-activation, "
        "protein synthesis rate, interaction with anabolic steroids "
        "and IGF-1 signaling"
    ),
    "A5": (
        "Insulin resistance from chronic supraphysiological use — "
        "receptor downregulation kinetics, beta-cell exhaustion timeline, "
        "reversibility after cessation, biomarkers for early detection "
        "in bodybuilders using insulin"
    ),
    "A6": (
        "Insulin sensitivity enhancers for bodybuilders — berberine AMPK "
        "activation, metformin hepatic glucose suppression, alpha-lipoic "
        "acid, chromium picolinate, cinnamon extract, inositol — "
        "mechanisms and dosing protocols"
    ),
    "A7": (
        "Hypoglycemia pathophysiology during exogenous insulin use — "
        "glucagon counter-regulation failure, neuroglycopenia cascade, "
        "adrenergic warning signs, glucose threshold shifts in trained "
        "athletes, emergency protocols"
    ),
    "A8": (
        "Hepatic vs peripheral insulin sensitivity divergence — NAFLD "
        "risk during insulin use, de novo lipogenesis, visceral fat "
        "accumulation, liver enzyme correlation with insulin dosing, "
        "interaction with oral anabolic steroids"
    ),

    # Cluster B — Trenbolone Deep Dive
    "B1": (
        "Trenbolone acetate veterinary origins — Finaplix-H cattle "
        "implant design, Revalor-H combination with estradiol, feed "
        "efficiency mechanism, USDA residue limits, how it became "
        "a bodybuilding compound"
    ),
    "B2": (
        "17-beta-trenbolone environmental persistence — waterway "
        "contamination from feedlot runoff, photodegradation half-life, "
        "endocrine disruption in fish via vitellogenin induction, "
        "soil binding kinetics"
    ),
    "B3": (
        "Trenbolone androgen receptor binding kinetics — 5x testosterone "
        "affinity, nuclear translocation rate, AR upregulation vs "
        "saturation, dose-response curve nonlinearity, comparison "
        "to other 19-nor compounds"
    ),
    "B4": (
        "Trenbolone and glucose metabolism — GLUT4 effects in skeletal "
        "muscle, insulin sensitization at muscle cell level, nutrient "
        "repartitioning mechanism, glycogen supercompensation, "
        "interaction with exogenous insulin"
    ),
    "B5": (
        "Trenbolone progesterone and prolactin axis — 19-nor "
        "progestational activity, prolactin elevation mechanism, "
        "cabergoline/dostinex management protocols, gynecomastia "
        "differential diagnosis (estrogenic vs prolactin-mediated)"
    ),
    "B6": (
        "Trenbolone metabolite cascade — epitrenbolone, trendione, "
        "17-alpha-trenbolone detection windows in urine and blood, "
        "metabolite bioactivity vs parent compound, anti-doping "
        "implications"
    ),
    "B7": (
        "Trenbolone ester pharmacokinetics — acetate (ED/EOD, 72h "
        "half-life) vs enanthate (2x/wk, 5-7d) vs "
        "hexahydrobenzylcarbonate (Parabolan, 14d) — absorption "
        "kinetics, blood level stability, practical dosing"
    ),
    "B8": (
        "Trenbolone cardiovascular toxicity — left ventricular "
        "remodeling, cardiac fibrosis pathways, atherosclerosis "
        "acceleration, HDL suppression magnitude vs other AAS, "
        "dose-dependency of cardiac damage"
    ),
    "B9": (
        "Trenbolone neuropsychiatric effects — insomnia mechanism "
        "via GABA-A interference, night sweats from sympathetic "
        "activation, aggression from amygdala AR density, "
        "5-alpha-reduced neurosteroid disruption"
    ),
    "B10": (
        "Trenbolone renal impact — proteinuria incidence, creatinine "
        "elevation vs actual GFR decline, kidney stress biomarkers "
        "KIM-1 and NGAL, dose-response for nephrotoxicity, "
        "interaction with dehydration and diuretics"
    ),

    # Cluster C — Boldenone & Hematology
    "C1": (
        "Boldenone undecylenate veterinary pharmacology — Equipoise "
        "equine origins, undecylenate ester very long half-life of 14 "
        "days, appetite stimulation mechanism via ghrelin pathway, "
        "steady-state accumulation kinetics"
    ),
    "C2": (
        "Boldenone EPO stimulation mechanism — erythropoietin vs "
        "direct bone marrow stimulation debate, comparison to "
        "nandrolone RBC effects, oxymetholone (Anadrol) as most "
        "potent oral erythropoietic AAS"
    ),
    "C3": (
        "Polycythemia management during AAS use — therapeutic "
        "phlebotomy protocols (500mL/8wk), naringin and grapefruit "
        "extract, blood donation eligibility, viscosity thresholds "
        "for clinical intervention"
    ),
    "C4": (
        "Dihydroboldenone (DHB / 1-testosterone) — boldenone "
        "metabolite via 5-alpha-reduction, anabolic-to-androgenic "
        "ratio, post-injection pain reputation, standalone use "
        "vs as metabolite pathway"
    ),
    "C5": (
        "Ferritin depletion dynamics during chronic phlebotomy — "
        "iron loss per donation, hepcidin regulation of absorption, "
        "oral vs IV iron supplementation, ferritin target range "
        "50-150 ng/mL, functional iron deficiency symptoms"
    ),
    "C6": (
        "Hematocrit monitoring and blood viscosity during AAS — "
        "testing frequency protocols, danger thresholds above 54%, "
        "stroke and PE risk modeling, altitude training confound, "
        "dehydration artifact in blood tests"
    ),

    # Cluster D — GH / IGF-1 Cascade
    "D1": (
        "Growth hormone pulsatile secretion — circadian GHRH and "
        "somatostatin oscillation, sleep-dependent GH surge, "
        "exogenous GH disruption of endogenous pulsatility, "
        "fasting amplification of GH release"
    ),
    "D2": (
        "GH dose-response curves — 2 IU anti-aging vs 4 IU "
        "bodybuilding vs 8+ IU professional level, diminishing "
        "returns modeling, side effect escalation (edema, CTS, "
        "insulin resistance) at each tier"
    ),
    "D3": (
        "IGF-1 hepatic production — GH receptor JAK2/STAT5 "
        "signaling cascade, liver health dependency, alcohol and "
        "NAFLD impact on IGF-1 synthesis, blood level targets "
        "300-500 ng/mL for bodybuilders"
    ),
    "D4": (
        "MGF (mechano growth factor) splice variant biology — "
        "local vs systemic IGF-1 isoforms, exercise-induced MGF "
        "expression, synthetic MGF peptide pharmacology, satellite "
        "cell activation mechanism"
    ),
    "D5": (
        "IGF-1 LR3 pharmacology — long-acting analog with 20h "
        "half-life vs 15 min for endogenous, dosing protocols "
        "20-100 mcg bilateral, cancer risk debate, comparison "
        "to endogenous GH-to-IGF-1 pathway"
    ),
    "D6": (
        "GH-induced insulin resistance — compensatory "
        "hyperinsulinemia, the Milos Sarcev GH plus insulin "
        "synergy window (GH 30 min before insulin), hepatic vs "
        "peripheral resistance divergence during GH use"
    ),
    "D7": (
        "GH and lipolysis — fatty acid mobilization mechanism, "
        "fasting protocol optimization, meal timing interference "
        "(insulin blunts GH-stimulated lipolysis within 30 min), "
        "contest prep application for bodybuilders"
    ),
    "D8": (
        "GH secretagogues — CJC-1295 DAC and no-DAC, Ipamorelin, "
        "MK-677 ibutamoren oral, GHRP-6 and GHRP-2 hunger effects, "
        "combined peptide protocols vs exogenous GH cost-benefit "
        "analysis"
    ),
    "D9": (
        "Somatostatin feedback loop — GHRH pulsatility regulation, "
        "arginine somatostatin suppression, L-DOPA pathway, feedback "
        "reset after GH cessation, rebound protocols for restoring "
        "natural GH production"
    ),

    # Cluster E — Micronutrient-PED Interactions
    "E1": (
        "Magnesium and insulin receptor sensitivity — 400-800 mg/day, "
        "glycinate vs citrate vs threonate bioavailability, TRPM6/7 "
        "channel regulation, magnesium-dependent ATP binding at "
        "insulin receptor tyrosine kinase"
    ),
    "E2": (
        "Zinc and aromatase modulation — 30-50 mg/day, CYP19A1 "
        "inhibition mechanism, testosterone-to-estradiol ratio "
        "optimization, zinc-copper antagonism, immune function "
        "during AAS immunosuppression"
    ),
    "E3": (
        "Vitamin D and androgen receptor density — 5000-10000 IU/day, "
        "VDR polymorphisms Fok1 and Bsm1, free testosterone "
        "correlation, 25-OH-D target levels 60-80 ng/mL, "
        "seasonal variation in AAS efficacy"
    ),
    "E4": (
        "Iron absorption interference during AAS and phlebotomy — "
        "calcium inhibition (separate by 2h), tannin and phytate "
        "chelation, hepcidin as master regulator, ascorbic acid "
        "enhancement, timing protocol"
    ),
    "E5": (
        "Potassium requirements during insulin use — intracellular "
        "K+ shift mechanism, cardiac arrhythmia risk at below 3.5 "
        "mEq/L, 99 mg supplementation per IU insulin rule, banana "
        "and coconut water as sources"
    ),
    "E6": (
        "TUDCA vs NAC hepatoprotection during oral AAS — bile acid "
        "mechanism (TUDCA) vs glutathione precursor (NAC), oral AAS "
        "liver stress pathways cholestasis vs oxidative, combination "
        "protocol, timing with meals"
    ),
    "E7": (
        "Taurine and muscle cramping during AAS — electrolyte shift "
        "mechanism, cell volumizing effect, bile acid conjugation, "
        "3-5 g/day dosing, cardiovascular protective effects during "
        "steroid use"
    ),
    "E8": (
        "Omega-3 cardiovascular protection during AAS — EPA vs DHA "
        "ratio for triglycerides vs inflammation, HDL recovery "
        "acceleration, 3-5 g/day pharmaceutical grade, interaction "
        "with aspirin during AAS use"
    ),
    "E9": (
        "Chromium picolinate insulin sensitization — GLUT4 "
        "enhancement mechanism, diabetic research evidence at "
        "500-1000 mcg, bodybuilding crossover, GTF glucose "
        "tolerance factor history, synergy with berberine"
    ),
    "E10": (
        "Vitamin K2 MK-7 and calcium metabolism during GH use — "
        "arterial calcification prevention, osteocalcin "
        "carboxylation, GH-induced calcium mobilization, 200 mcg/day "
        "synergy with vitamin D3 and magnesium"
    ),

    # Cluster F — Additional Compound Deep Dives
    "F1": (
        "Nandrolone Deca-Durabolin joint lubrication — synovial "
        "fluid production, collagen type III synthesis, 19-nor "
        "structure and progestational activity, therapeutic vs "
        "bodybuilding dosing protocols"
    ),
    "F2": (
        "Oxandrolone Anavar nitrogen retention — SHBG reduction "
        "mechanism, muscle wasting clinical research in burns and "
        "HIV, hepatic metabolism as C-17aa, women dosing protocols "
        "and virilization thresholds"
    ),
    "F3": (
        "Testosterone ester pharmacokinetics — cypionate 8d half-life "
        "vs enanthate 4.5d vs propionate 0.8d vs undecanoate oral "
        "21d release curves, blood level stability, injection "
        "frequency optimization"
    ),
    "F4": (
        "Aromatase inhibitor pharmacology — anastrozole reversible "
        "nonsteroidal vs letrozole reversible nonsteroidal stronger "
        "vs exemestane suicidal steroidal, estrogen rebound risk, "
        "bone density impact from crashed E2"
    ),
    "F5": (
        "PCT pharmacology — tamoxifen SERM breast tissue vs "
        "clomiphene SERM hypothalamic vs enclomiphene pure "
        "trans-isomer, HPTA recovery kinetics, HCG during cycle "
        "vs PCT timing, recovery timeline"
    ),
    "F6": (
        "5-alpha reductase and AAS interaction — DHT conversion "
        "rates by compound, finasteride plus nandrolone paradox "
        "creating more androgenic DHN, compound-specific 5-alpha "
        "reductase susceptibility, hair loss management"
    ),
    "F7": (
        "Myostatin inhibition approaches — follistatin gene therapy, "
        "ACE-031 clinical trials, natural epicatechin from dark "
        "chocolate, satellite cell activation by AAS, mTOR and Akt "
        "pathway convergence with myostatin"
    ),
    "F8": (
        "Clenbuterol beta-2-agonist mechanism — muscle preservation "
        "during caloric deficit, cardiac hypertrophy risk from "
        "beta-1 cross-reactivity, receptor downregulation 2wk on "
        "2wk off, ketotifen resensitization protocol"
    ),
    "F9": (
        "GLP-1 agonists in bodybuilding context — semaglutide and "
        "tirzepatide appetite suppression, insulin sensitization "
        "mechanism, contest prep application, muscle preservation "
        "debate, interaction with exogenous insulin"
    ),

    # Cluster G — Clinical Monitoring & Management
    "G1": (
        "Blood pressure management on AAS — RAAS system interaction "
        "with androgens, ACE inhibitors lisinopril, ARBs telmisartan, "
        "nebivolol beta-blocker with NO, dose-response by compound "
        "and stack"
    ),
    "G2": (
        "Left ventricular hypertrophy monitoring on AAS — "
        "echocardiogram markers IVSd and LVPWd, AAS-induced "
        "pathological vs exercise-induced physiological hypertrophy, "
        "reversibility after cessation, risk stratification"
    ),
    "G3": (
        "Lipid management during AAS — HDL suppression magnitude by "
        "compound (oral much worse than injectable), LDL and ApoB "
        "elevation, niacin, EPA/DHA high-dose, statin considerations "
        "and myopathy risk on AAS"
    ),
    "G4": (
        "Liver enzyme interpretation on AAS — AST/ALT elevation "
        "patterns by compound type (C-17aa orals vs injectables), "
        "cholestasis markers GGT ALP bilirubin, TUDCA and NAC "
        "intervention thresholds"
    ),
    "G5": (
        "Kidney function monitoring on AAS — eGFR interpretation "
        "when taking creatine plus AAS (both elevate creatinine), "
        "cystatin C as superior marker, proteinuria screening, "
        "trenbolone-specific nephrotoxicity"
    ),

    # Cluster H — Cross-Domain Bridges
    "H1": (
        "Veterinary-to-human pharmacology pipeline — trenbolone "
        "cattle, boldenone horses, stanozolol racing, nandrolone "
        "veterinary anemia: how animal drugs became human PEDs, "
        "regulatory gaps and gray market chemistry"
    ),
    "H2": (
        "Diabetes research crossover to bodybuilding — metformin "
        "repurposing, GLP-1 agonists semaglutide, insulin analogs "
        "Humalog NovoRapid, berberine from TCM, SGLT2 inhibitors, "
        "continuous glucose monitors for insulin users"
    ),
    "H3": (
        "Endocrine disruptor environmental science — 17-beta-"
        "trenbolone in waterways, ethinylestradiol from "
        "contraceptives, bisphenol A, phthalates — aquatic organism "
        "feminization, human fertility impact from environmental "
        "exposure"
    ),
    "H4": (
        "Cancer biology intersection with PED pharmacology — "
        "IGF-1 and mTOR signaling in tumor growth, androgen receptor "
        "role in prostate cancer, estrogen receptor modulation by "
        "SERMs, apoptosis pathway disruption by AAS"
    ),
    "H5": (
        "Aging and longevity research crossover — GH decline "
        "somatopause, testosterone decline andropause, NAD+ "
        "precursors NMN and NR, telomere length and AAS, "
        "mitochondrial function, senolytics interaction with "
        "hormone replacement"
    ),
}

# ---------------------------------------------------------------------------
# Provider configurations
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """API configuration for a model provider."""

    api_base: str
    api_key_env: str
    model: str
    max_tokens: int = 8192
    temperature: float = 0.3


PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        model="deepseek-chat",
    ),
    "gemini-flash": ProviderConfig(
        api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GOOGLE_API_KEY",
        model="gemini-2.5-flash",
    ),
    "grok-fast": ProviderConfig(
        api_base="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
        model="grok-4-1-fast-non-reasoning",
    ),
    "mistral-large": ProviderConfig(
        api_base="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        model="mistral-large-latest",
    ),
    "kimi": ProviderConfig(
        api_base="https://api.moonshot.cn/v1",
        api_key_env="KIMI_API_KEY",
        model="kimi-k2.5",
        temperature=1.0,  # Kimi requires temperature=1
    ),
    "openrouter": ProviderConfig(
        api_base="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model="deepseek/deepseek-chat-v3-0324:free",
    ),
    "groq": ProviderConfig(
        api_base="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.3-70b-versatile",
    ),
    "openai": ProviderConfig(
        api_base="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model="gpt-4.1-mini",
    ),
}

# ---------------------------------------------------------------------------
# Wave definitions
# ---------------------------------------------------------------------------

# Wave 1: All 48 topics with primary worker
WAVE1_TOPICS = list(TOPICS.keys())

# Wave 2: High-serendipity topics with alternative models
WAVE2_RUNS: list[tuple[list[str], str]] = [
    (["A1", "B1", "B4", "C2", "D6", "F9"], "gemini-flash"),
    (["A1", "B1", "B4", "C2", "D6", "F9"], "grok-fast"),
    (["A1", "A4", "B3", "D3", "F7", "H4"], "mistral-large"),
    (["A1", "B9", "D8", "F5", "G2", "H5"], "kimi"),
]

# Wave 3: Orchestrator comparison (A1 only)
WAVE3_PROVIDERS = ["deepseek", "gemini-flash", "grok-fast", "openai"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_single_topic(
    topic_id: str,
    query: str,
    provider: ProviderConfig,
    provider_name: str,
    corpus_path: str,
    base_output_dir: str,
    max_waves: int = 3,
) -> dict[str, Any]:
    """Run a single swarm topic and return metrics."""
    from corpus import ConditionStore

    output_dir = Path(base_output_dir) / provider_name / topic_id
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get(provider.api_key_env, "")
    if not api_key:
        logger.error(
            "topic=<%s>, provider=<%s> | missing API key %s",
            topic_id, provider_name, provider.api_key_env,
        )
        return {"topic": topic_id, "provider": provider_name, "status": "ERROR", "error": f"missing {provider.api_key_env}"}

    corpus = Path(corpus_path).read_text()

    db_path = str(output_dir / "store.duckdb")
    store = ConditionStore(db_path=db_path)

    import httpx

    async def complete_fn(prompt: str) -> str:
        url = f"{provider.api_base}/chat/completions"
        payload = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Produce your analysis."},
            ],
            "max_tokens": provider.max_tokens,
            "temperature": provider.temperature,
        }
        headers = {"Content-Type": "application/json"}
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
                    "topic=<%s>, model=<%s> | completion failed",
                    topic_id, provider.model,
                )
                return ""

    run_id = f"emp_{provider_name}_{topic_id}_{time.strftime('%Y%m%d_%H%M%S')}"

    config = MCPSwarmConfig(
        max_workers=8,
        max_waves=max_waves,
        convergence_threshold=5,
        api_base=provider.api_base,
        model=provider.model,
        api_key=api_key,
        max_tokens=provider.max_tokens,
        temperature=provider.temperature,
        required_angles=[],
        report_max_tokens=16384,
        enable_serendipity_wave=True,
        source_model=provider.model,
        source_run=run_id,
        compact_every_n_waves=3,
        enable_rolling_summaries=True,
    )

    engine = MCPSwarmEngine(store=store, complete=complete_fn, config=config)

    logger.info(
        "topic=<%s>, provider=<%s>, model=<%s> | starting",
        topic_id, provider_name, provider.model,
    )

    t0 = time.time()
    try:
        result = await engine.synthesize(corpus=corpus, query=query)
        elapsed = time.time() - t0

        # Save report
        ts = time.strftime("%Y%m%d_%H%M%S")
        (output_dir / f"report_{ts}.md").write_text(result.report)

        metrics = {
            "topic": topic_id,
            "query": query[:100],
            "provider": provider_name,
            "model": provider.model,
            "status": "OK",
            "elapsed_s": round(elapsed, 1),
            "total_waves": result.metrics.total_waves,
            "total_findings": result.metrics.total_findings_stored,
            "findings_per_wave": result.metrics.findings_per_wave,
            "convergence_reason": result.metrics.convergence_reason,
            "angles_detected": result.angles_detected,
            "report_chars": len(result.report),
            "run_id": run_id,
        }

        (output_dir / f"metrics_{ts}.json").write_text(json.dumps(metrics, indent=2))

        logger.info(
            "topic=<%s>, provider=<%s> | done in %.1fs — %d findings, %d angles, %s",
            topic_id, provider_name, elapsed,
            result.metrics.total_findings_stored,
            len(result.angles_detected),
            result.metrics.convergence_reason,
        )

        return metrics

    except Exception as exc:
        elapsed = time.time() - t0
        logger.exception("topic=<%s>, provider=<%s> | FAILED", topic_id, provider_name)
        error_metrics = {
            "topic": topic_id,
            "provider": provider_name,
            "model": provider.model,
            "status": "ERROR",
            "error": str(exc)[:200],
            "elapsed_s": round(elapsed, 1),
        }
        (output_dir / "error.json").write_text(json.dumps(error_metrics, indent=2))
        return error_metrics


async def run_batch(
    topic_ids: list[str],
    provider_name: str,
    corpus_path: str,
    base_output_dir: str,
    max_concurrent: int = 6,
    max_waves: int = 3,
) -> list[dict[str, Any]]:
    """Run a batch of topics with concurrency control."""
    provider = PROVIDERS[provider_name]
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_run(topic_id: str) -> dict[str, Any]:
        async with semaphore:
            return await run_single_topic(
                topic_id=topic_id,
                query=TOPICS[topic_id],
                provider=provider,
                provider_name=provider_name,
                corpus_path=corpus_path,
                base_output_dir=base_output_dir,
                max_waves=max_waves,
            )

    tasks = [limited_run(tid) for tid in topic_ids]
    return await asyncio.gather(*tasks)


async def run_wave1(corpus_path: str, output_dir: str, max_concurrent: int = 6) -> None:
    """Wave 1: All 48 topics with DeepSeek V3.2."""
    logger.info("wave=<1> | starting — %d topics with deepseek", len(WAVE1_TOPICS))
    results = await run_batch(
        topic_ids=WAVE1_TOPICS,
        provider_name="deepseek",
        corpus_path=corpus_path,
        base_output_dir=output_dir,
        max_concurrent=max_concurrent,
    )
    # Summary
    ok = [r for r in results if r.get("status") == "OK"]
    err = [r for r in results if r.get("status") == "ERROR"]
    total_findings = sum(r.get("total_findings", 0) for r in ok)
    logger.info(
        "wave=<1> | complete — %d OK, %d ERROR, %d total findings",
        len(ok), len(err), total_findings,
    )
    summary_path = Path(output_dir) / "wave1_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nWave 1 Summary: {len(ok)} OK, {len(err)} ERROR, {total_findings} findings")
    print(f"Saved to: {summary_path}")


async def run_wave2(corpus_path: str, output_dir: str, max_concurrent: int = 4) -> None:
    """Wave 2: Cross-validation with alternative models."""
    logger.info("wave=<2> | starting — cross-validation runs")
    all_results = []
    for topics, provider_name in WAVE2_RUNS:
        logger.info("wave=<2>, provider=<%s> | %d topics", provider_name, len(topics))
        results = await run_batch(
            topic_ids=topics,
            provider_name=provider_name,
            corpus_path=corpus_path,
            base_output_dir=output_dir,
            max_concurrent=max_concurrent,
        )
        all_results.extend(results)

    summary_path = Path(output_dir) / "wave2_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    ok = [r for r in all_results if r.get("status") == "OK"]
    print(f"\nWave 2 Summary: {len(ok)} OK, {len(all_results) - len(ok)} ERROR")


async def run_wave3(corpus_path: str, output_dir: str) -> None:
    """Wave 3: Orchestrator comparison — A1 only, different providers."""
    logger.info("wave=<3> | starting — orchestrator comparison on A1")
    all_results = []
    for provider_name in WAVE3_PROVIDERS:
        results = await run_batch(
            topic_ids=["A1"],
            provider_name=provider_name,
            corpus_path=corpus_path,
            base_output_dir=output_dir,
            max_concurrent=1,
        )
        all_results.extend(results)

    summary_path = Path(output_dir) / "wave3_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWave 3 Summary: {len(all_results)} orchestrator comparison runs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Employment list runner")
    parser.add_argument("--wave", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--topics", nargs="+", help="Run specific topics only")
    parser.add_argument("--provider", default="deepseek", choices=list(PROVIDERS.keys()))
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--max-waves", type=int, default=3)
    parser.add_argument("--corpus", default=str(Path(__file__).parent / "test_corpus.txt"))
    parser.add_argument("--output", default=str(Path(__file__).parent / "employment_results"))
    args = parser.parse_args()

    if args.topics:
        # Run specific topics with specified provider
        asyncio.run(run_batch(
            topic_ids=args.topics,
            provider_name=args.provider,
            corpus_path=args.corpus,
            base_output_dir=args.output,
            max_concurrent=args.max_concurrent,
            max_waves=args.max_waves,
        ))
    elif args.wave == 1:
        asyncio.run(run_wave1(args.corpus, args.output, args.max_concurrent))
    elif args.wave == 2:
        asyncio.run(run_wave2(args.corpus, args.output))
    elif args.wave == 3:
        asyncio.run(run_wave3(args.corpus, args.output))


if __name__ == "__main__":
    main()
