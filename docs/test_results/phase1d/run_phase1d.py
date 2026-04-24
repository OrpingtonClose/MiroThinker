#!/usr/bin/env python3
"""Phase 1D — Qwen3-235B-A22B-Instruct-2507-FP8 Flock baseline.

Runs both regimes:
- Prefix-cached corpus (canonical Flock pattern, same as Phase 1A / 1C-1)
- Stateless direct-pair (the pattern that worked for Kimi-Linear at 96%)

50 judgments × 2 patterns × 1 reps, plus 200 judgments high-throughput variant.
"""
import json, time, asyncio, aiohttp, statistics, sys
from pathlib import Path

PHASE1A_JSON = Path("/root/phase1a_1A-1.json")
OUT = Path("/root/phase1d_out")
URL = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen3-235b"

SYSTEM_INSTRUCTION = (
    "You are a precise relevance classifier for a pharmacology research corpus. "
    "Given a USER QUERY and a candidate FINDING from the corpus below, decide whether the "
    "finding directly answers or materially informs the query. Reply with exactly one word: "
    "RELEVANT or NOT_RELEVANT. Do not explain."
)


def build_corpus(findings):
    lines = ["# CORPUS (41 pharmacology findings):"]
    for i, f in enumerate(findings, 1):
        lines.append(f"[{i:02d}] {f}")
    return "\n".join(lines)


async def judge_prefix(session, sys_text, judgment):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user",
             "content": f"QUERY: {judgment['query']}\nFINDING: {judgment['finding']}\nRespond RELEVANT or NOT_RELEVANT."},
        ],
        "max_tokens": 8, "temperature": 0.0,
    }
    return await _call(session, payload, judgment)


async def judge_direct(session, judgment):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user",
             "content": f"Answer with exactly RELEVANT or NOT_RELEVANT.\nQuery: {judgment['query']}. Document: {judgment['finding']}"},
        ],
        "max_tokens": 16, "temperature": 0.0,
    }
    return await _call(session, payload, judgment)


async def _call(session, payload, judgment):
    t0 = time.perf_counter()
    try:
        async with session.post(URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as r:
            data = await r.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        content = data["choices"][0]["message"]["content"].strip().upper()
        predicted = (content.startswith("RELEVANT") and
                     not content.startswith("NOT_RELEVANT") and
                     not content.startswith("NOT RELEVANT"))
        return {
            **judgment, "predicted": predicted, "correct": predicted == judgment["expected"],
            "raw_response": content[:60], "latency_ms": round(latency_ms, 1),
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"], "error": None,
        }
    except Exception as e:
        return {**judgment, "predicted": None, "correct": False, "raw_response": None,
                "latency_ms": None, "error": repr(e),
                "prompt_tokens": None, "completion_tokens": None}


async def run(test_id, judge_fn, judgments, concurrency, repetitions, sys_text=None):
    sem = asyncio.Semaphore(concurrency)
    work = judgments * repetitions
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as s:
        async def bound(j):
            async with sem:
                if sys_text is not None:
                    return await judge_fn(s, sys_text, j)
                return await judge_fn(s, j)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(bound(j) for j in work))
        total = time.perf_counter() - t0
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r["error"])
    latencies = sorted([r["latency_ms"] for r in results if r["error"] is None])
    summary = {
        "test_id": test_id, "model": MODEL,
        "serving": "vLLM 0.19.1, TP=4, prefix-caching ON, 4xH200 SXM, max_model_len=1048576",
        "num_judgments": len(results), "correct": correct, "errors": errors,
        "accuracy": round(correct/len(results), 4),
        "total_time_s": round(total, 2),
        "judgments_per_second": round(len(results)/total, 2),
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "p50_latency_ms": round(latencies[len(latencies)//2], 1) if latencies else None,
        "p95_latency_ms": round(latencies[int(len(latencies)*0.95)], 1) if latencies else None,
        "total_prompt_tokens": sum((r["prompt_tokens"] or 0) for r in results),
        "total_completion_tokens": sum((r["completion_tokens"] or 0) for r in results),
        "concurrent": concurrency, "repetitions": repetitions,
    }
    return summary, results


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pha = json.load(open(PHASE1A_JSON))
    raw = pha.get("judgments") or pha.get("results")
    simple = [{"query": j["query"], "finding": j["finding"], "expected": j["expected"]} for j in raw]
    findings = []
    seen = set()
    for j in simple:
        if j["finding"] not in seen:
            seen.add(j["finding"])
            findings.append(j["finding"])
    corpus = build_corpus(findings)
    sys_text = f"{SYSTEM_INSTRUCTION}\n\n{corpus}"

    # 1D-1: canonical prefix-cached, 50 judgments, concurrency=5
    print("[1D-1] canonical prefix-cached, 50 judgments, concurrency=5", flush=True)
    s1, r1 = await run("1D-1", judge_prefix, simple, 5, 1, sys_text=sys_text)
    print(json.dumps(s1, indent=2), flush=True)
    json.dump({**s1, "judgments": r1}, open(OUT / "1D-1.json", "w"), indent=2)

    # 1D-2: canonical prefix-cached, 200 judgments, concurrency=20
    print("\n[1D-2] canonical prefix-cached, 200 judgments, concurrency=20", flush=True)
    s2, r2 = await run("1D-2", judge_prefix, simple, 20, 4, sys_text=sys_text)
    print(json.dumps(s2, indent=2), flush=True)
    json.dump({**s2, "judgments": r2}, open(OUT / "1D-2.json", "w"), indent=2)

    # 1D-3: stateless direct-pair, 50 judgments, concurrency=5
    print("\n[1D-3] stateless direct-pair, 50 judgments, concurrency=5", flush=True)
    s3, r3 = await run("1D-3", judge_direct, simple, 5, 1)
    print(json.dumps(s3, indent=2), flush=True)
    json.dump({**s3, "judgments": r3}, open(OUT / "1D-3.json", "w"), indent=2)

    summary = {"1D-1": s1, "1D-2": s2, "1D-3": s3}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

asyncio.run(main())
