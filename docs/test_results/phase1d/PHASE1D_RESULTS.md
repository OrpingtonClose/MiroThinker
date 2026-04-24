# Phase 1D — Qwen3-235B-A22B-Instruct-2507-FP8 Flock Baseline

**Date:** 2026-04-23
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (235 B total, 22 B active, MoE, non-thinking)
**Serving:** vLLM 0.19.1 on 4×H200 SXM (Japan, vast.ai instance 35487543), TP=4, prefix-caching ON, bf16 KV, FP8 weights, `--max-model-len 1048576`, YaRN 4.0× rope scaling (262 K → 1 M)
**Corpus:** same 41-finding synthetic pharmacology corpus + 50 ground-truth judgments used for Phase 1A (Llama-70B) and Phase 1C (Kimi-Linear), sourced from `phase1a/1A-1.json`

## Headline

**Qwen3-235B-A22B is the new champion Flock driver.** It matches Llama-70B's accuracy on the canonical prefix-cached pattern (98 % vs 98 %) while delivering **~22× higher throughput** at concurrency=20 (137 j/s vs Llama's 6.3 j/s baseline). Unlike Kimi-Linear-48B, the prefix-cached pattern succeeds — standard softmax attention recovers the precise key-based lookup that Kimi's linear attention could not.

## Results

| Test | Pattern | Judgments | Concurrency | Accuracy | Avg latency | p50 | p95 | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1D-1 | prefix-cached (canonical Flock) | 50 | 5 | **98 %** (49/50) | 125 ms | 108 ms | 283 ms | 39.2 j/s |
| 1D-2 | prefix-cached (canonical Flock) | 200 | 20 | **98 %** (196/200) | 139 ms | 138 ms | 173 ms | **137.0 j/s** |
| 1D-3 | stateless direct-pair | 50 | 5 | 96 % (48/50) | 96 ms | 94 ms | 113 ms | 50.6 j/s |

Raw per-judgment JSON: [`1D-1.json`](./1D-1.json), [`1D-2.json`](./1D-2.json), [`1D-3.json`](./1D-3.json).
Harness: [`run_phase1d.py`](./run_phase1d.py).

## Comparison with other Flock-candidate models

| Model | Prefix-cached accuracy | Prefix-cached throughput | Direct-pair accuracy | Verdict |
|---|---:|---:|---:|---|
| Llama-70B abliterated (Phase 1A) | 98 % | 6.3 j/s @ 848 ms avg | — | baseline driver |
| Qwen3.5-35B thinking | 54 % | — | — | **REJECTED** — thinking model incompatible |
| Kimi-Linear-48B-A3B abliterated (Phase 1C) | 54 % | — | 96 % @ 51 j/s | stateless-only |
| **Qwen3-235B-A22B-Instruct-2507 FP8 (Phase 1D)** | **98 %** | **137 j/s @ 139 ms avg** | 96 % @ 51 j/s | **champion driver** |

## Token economics

- 1D-1 + 1D-2 + 1D-3 combined: 277 595 prompt tokens, 1 345 completion tokens.
- The 200-judgment 1D-2 run completed in 1.46 s wall-clock at conc=20. Prefix-cache hit rate was high (system+corpus reused across all 200 judgments in the run), which is the exact amortisation advantage Flock was designed for. A full 10 000-judgment production Flock pass would cost roughly 73 s of GPU time at this rate versus ~25 min on Llama-70B.

## Why Qwen3-235B succeeds where Kimi-Linear failed

Kimi-Linear's 54 % failure on this exact prompt pattern was diagnosed in [PHASE1C_RESULTS.md](../phase1c/PHASE1C_RESULTS.md) as a consequence of its hybrid linear-attention architecture: the compressed recurrent state it uses for long-range dependencies cannot perform precise key-based lookup when the query references corpus items by ID ("Is finding [03] RELEVANT to …?"). Qwen3-235B-A22B-Instruct-2507 uses standard softmax attention with GQA (94 layers, 64 Q heads, 4 KV heads, head dim 128), which preserves the full KV matrix and supports the lookup — confirming the hypothesis that any model with conventional attention should pass this pattern given sufficient scale.

## Errors

- 1D-1: 1 miss (false negative on a borderline-caveat finding).
- 1D-2: 4 misses across 200 judgments, all of the same false-negative-with-caveat-text pattern.
- 1D-3: 2 misses (same pattern).

No refusals, no off-task responses, no malformed outputs across 300 judgments.

## Notes on the FP8 checkpoint

- FP8 weights (block-wise quantised, E4M3, 128×128 block) × 24 shards ≈ 220 GB on disk.
- rope_scaling was patched to YaRN 4.0× (262 K → 1 M) because the shipped config has `rope_scaling: null` despite the README claiming 1 M support. Patch is a one-liner in `config.json`; no code change required.
- KV cache with bf16 dtype fits 4×H200 at `--gpu-memory-utilization 0.92` with 67.77 GiB free per GPU, supporting a total KV pool of 1 511 904 tokens and max concurrency of 1.44× at 1 M tokens per request.
- Attempted `--kv-cache-dtype fp8_e5m2` fails with `ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints.` — kept bf16 KV, which fits anyway.

## Suggested next uses

1. **Swap Qwen3-235B-A22B-Instruct-2507 in as the default Flock driver** in `apps/adk-agent/models/` — single config change. The 22× throughput win directly translates to wall-clock reduction on corpus synthesis.
2. **Re-test the transplant POC** ([`../transplant_poc/summary.md`](../transplant_poc/summary.md)) against a large (≥100 K token) corpus served by this Qwen3 instance, using it as both the bee and the architect. The original Gemini-based transplant POC saw control-beats-transplant on a 3 K corpus; a larger corpus should expose whether the transplant technique wins at scale.
