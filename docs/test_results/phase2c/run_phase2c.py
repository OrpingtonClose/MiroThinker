#!/usr/bin/env python3
"""Phase 2C — Qwen3-235B-A22B-Instruct-2507-FP8 context degradation curve.

Same needle-in-haystack methodology as Phase 2B (Kimi-Linear) for direct comparison.
Sizes: 131K, 250K, 500K, 750K, 1M (the full 1M curve the user requested).

Uses stateless direct-pair prompt (the Flock pattern that worked at 3K for Kimi-Linear;
Qwen3 should also handle prefix-cached but direct-pair is the across-model comparable baseline).
"""
import json, time, asyncio, aiohttp, statistics, random, sys
from pathlib import Path

URL = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen3-235b"
OUT = Path("/root/phase2c_out")
CHARS_PER_TOKEN = 3.5

DISTRACTORS = [
    "Propranolol is a non-selective beta blocker used for hypertension and arrhythmia.",
    "Lisinopril is an ACE inhibitor that reduces vascular resistance.",
    "Atorvastatin competitively inhibits HMG-CoA reductase in hepatic cholesterol synthesis.",
    "Metformin improves insulin sensitivity and reduces hepatic gluconeogenesis.",
    "Warfarin is a vitamin K antagonist; INR monitoring is required.",
    "Omeprazole irreversibly inhibits H+/K+-ATPase proton pumps in parietal cells.",
    "Levothyroxine is a synthetic T4 hormone replacement; TSH-guided dosing.",
    "Amlodipine is a dihydropyridine calcium-channel blocker for hypertension.",
    "Sertraline is an SSRI increasing serotonin availability in the synaptic cleft.",
    "Gabapentin modulates alpha-2-delta subunit of voltage-gated calcium channels.",
    "Losartan is an angiotensin-II receptor blocker; no dry cough side effect.",
    "Furosemide inhibits Na/K/2Cl cotransport in the thick ascending loop of Henle.",
    "Hydrochlorothiazide inhibits NaCl symport in the distal convoluted tubule.",
    "Ciprofloxacin inhibits bacterial DNA gyrase (topoisomerase II).",
    "Amoxicillin is a beta-lactam that inhibits bacterial cell-wall synthesis.",
    "Azithromycin blocks 50S ribosomal subunit preventing bacterial translation.",
    "Ibuprofen non-selectively inhibits COX-1 and COX-2; dose-limited GI toxicity.",
    "Acetaminophen mechanism is weak COX inhibition with central antipyretic effect.",
    "Dexamethasone is a long-acting glucocorticoid with 25x cortisol potency.",
    "Insulin glargine is a basal long-acting analogue with 24-hour coverage.",
]

NEEDLE = "Alirocumab is a monoclonal antibody targeting PCSK9 to lower LDL cholesterol by up to 60%."
NEEDLE_QUERY = "alirocumab PCSK9 LDL reduction"


def build_doc(approx_tokens, insert_needle, needle_pos_frac=0.5, rng=None):
    rng = rng or random.Random(42)
    target_chars = int(approx_tokens * CHARS_PER_TOKEN)
    parts = []
    current = 0
    while current < target_chars:
        s = rng.choice(DISTRACTORS)
        parts.append(s)
        current += len(s) + 1
    text = " ".join(parts)
    if insert_needle:
        pos = int(len(text) * needle_pos_frac)
        while pos < len(text) and text[pos] != " ":
            pos += 1
        text = text[:pos] + " " + NEEDLE + " " + text[pos:]
    return text


async def judge(session, query, doc, timeout_s=1800):
    prompt = f"Answer with exactly RELEVANT or NOT_RELEVANT.\nQuery: {query}. Document: {doc}"
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 16, "temperature": 0.0}
    t0 = time.perf_counter()
    try:
        async with session.post(URL, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            data = await r.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        content = data["choices"][0]["message"]["content"].strip().upper()
        predicted = (content.startswith("RELEVANT") and
                     not content.startswith("NOT_RELEVANT") and
                     not content.startswith("NOT RELEVANT"))
        return predicted, content[:60], latency_ms, data["usage"]["prompt_tokens"], None
    except Exception as e:
        return None, None, (time.perf_counter()-t0)*1000, None, repr(e)


async def run_size(size_tokens):
    print(f"--- size={size_tokens} tokens ---", flush=True)
    results = []
    positions = [0.1, 0.5, 0.9]
    async with aiohttp.ClientSession() as session:
        for i, pos_frac in enumerate(positions):
            doc = build_doc(size_tokens, insert_needle=True, needle_pos_frac=pos_frac,
                            rng=random.Random(1000 + i))
            predicted, raw, latency, ptok, err = await judge(session, NEEDLE_QUERY, doc)
            correct = (predicted is True)
            results.append({"size": size_tokens, "case": f"needle@{pos_frac}", "expected": True,
                            "predicted": predicted, "correct": correct, "raw": raw,
                            "latency_ms": round(latency, 1), "prompt_tokens": ptok, "error": err})
            print(f"  needle@{pos_frac}  predicted={predicted}  correct={correct}  "
                  f"latency={latency:.0f}ms  ptok={ptok}  err={err}", flush=True)
        for i in range(3):
            doc = build_doc(size_tokens, insert_needle=False, rng=random.Random(2000 + i))
            predicted, raw, latency, ptok, err = await judge(session, NEEDLE_QUERY, doc)
            correct = (predicted is False)
            results.append({"size": size_tokens, "case": f"no_needle_{i}", "expected": False,
                            "predicted": predicted, "correct": correct, "raw": raw,
                            "latency_ms": round(latency, 1), "prompt_tokens": ptok, "error": err})
            print(f"  no_needle_{i}  predicted={predicted}  correct={correct}  "
                  f"latency={latency:.0f}ms  ptok={ptok}  err={err}", flush=True)
    correct = sum(1 for r in results if r["correct"])
    latencies = [r["latency_ms"] for r in results if r["error"] is None]
    return {
        "size_tokens": size_tokens,
        "num_trials": len(results),
        "correct": correct,
        "accuracy": round(correct/len(results), 3),
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "max_latency_ms": round(max(latencies), 1) if latencies else None,
        "avg_prompt_tokens": round(statistics.mean(
            [r["prompt_tokens"] for r in results if r["prompt_tokens"]]), 0)
            if any(r["prompt_tokens"] for r in results) else None,
        "errors": sum(1 for r in results if r["error"]),
        "trials": results,
    }


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else \
            [131000, 250000, 500000, 750000, 1000000]
    all_summaries = []
    for s in sizes:
        summ = await run_size(s)
        all_summaries.append(summ)
        json.dump(summ, open(OUT / f"size_{s}.json", "w"), indent=2)
        json.dump(all_summaries, open(OUT / "phase2c_summary.json", "w"), indent=2)
        print(json.dumps({k: v for k, v in summ.items() if k != "trials"}, indent=2), flush=True)

asyncio.run(main())
