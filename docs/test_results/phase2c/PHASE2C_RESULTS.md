# Phase 2C — Qwen3-235B-A22B-Instruct-2507-FP8 Context Degradation Curve (to 1 M)

**Date:** 2026-04-23
**Model:** `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` (FP8 weights, bf16 KV cache)
**Serving:** vLLM 0.19.1 on 4×H200 SXM, TP=4, prefix-caching ON, `--max-model-len 1048576`, YaRN 4.0× rope scaling
**Methodology:** needle-in-haystack identical to Phase 2B (Kimi-Linear). Same 20-item distractor pool, same needle (`Alirocumab is a monoclonal antibody targeting PCSK9 to lower LDL cholesterol by up to 60%.`), same query (`alirocumab PCSK9 LDL reduction`). Per size, 3 needle-present positions (10 %, 50 %, 90 %) and 3 needle-absent controls — **6 trials per size, 30 trials total.**

## Headline

**Qwen3-235B-A22B-Instruct-2507 preserves perfect retrieval across the entire 1 M-token context window.** 30/30 trials correct across 5 context sizes; no degradation curve observed. This is a striking contrast with Kimi-Linear-48B which showed a flat 50 % chance-floor from 32 K onward. Qwen3 is the first model in this evaluation that maintains Llama-70B-style precision at scales > 100 K tokens.

## Results

| Size target | Actual prompt tokens | Trials | Correct | Accuracy | Avg latency | Max latency |
|---:|---:|---:|---:|---:|---:|---:|
| 131 000 | 116 410 | 6 | **6** | **100 %** | 13.6 s | 14.3 s |
| 250 000 | 221 958 | 6 | **6** | **100 %** | 32.6 s | 41.3 s |
| 500 000 | 444 260 | 6 | **6** | **100 %** | 148.2 s | 150.4 s |
| 750 000 | 666 408 | 6 | **6** | **100 %** | 325.5 s | 326.1 s |
| 1 000 000 | 888 354 | 6 | **6** | **100 %** | 566.0 s | 566.7 s |

Raw per-trial JSON: [`size_131000.json`](./size_131000.json), [`size_250000.json`](./size_250000.json), [`size_500000.json`](./size_500000.json), [`size_750000.json`](./size_750000.json), [`size_1000000.json`](./size_1000000.json).
Aggregated: [`phase2c_summary.json`](./phase2c_summary.json).
Harness: [`run_phase2c.py`](./run_phase2c.py).

The actual prompt-token counts come in below the target sizes because the distractor text packs more characters per token than the 3.5 estimate used for padding. The effective maximum-context cell still reached 888 K tokens — roughly 85 % of the 1 M max-model-len window, well past the 262 K native context.

## Comparison with prior context-degradation curves

| Size | Llama-70B (Phase 2A) | Kimi-Linear (Phase 2B) | **Qwen3-235B (Phase 2C)** |
|---:|:---:|:---:|:---:|
| 32 K | 100 % | 50 % (chance) | — |
| 65 K | 100 % | 50 % (chance) | — |
| 91 K | 100 % | — | — |
| 100 K | — | 50 % (chance) | — |
| 131 K | — | — | **100 %** |
| 250 K | — | — | **100 %** |
| 500 K | — | — | **100 %** |
| 750 K | — | — | **100 %** |
| 1 M | — | — | **100 %** |

Llama-70B topped out at 91 K (its native 128 K window). Kimi-Linear collapsed to chance at 32 K. **Qwen3-235B is the first model tested to maintain 100 % retrieval through 1 M tokens.**

## Latency scaling

Prefill-dominated cost grows roughly quadratically-to-superquadratically with prompt length (attention is O(n²) even with FlashAttention), and the numbers bear that out:

| Size | ptok | Latency | ms per 1 K ptok |
|---:|---:|---:|---:|
| 131 000 | 116 K | 13.6 s | 117 |
| 250 000 | 222 K | 32.6 s | 147 |
| 500 000 | 444 K | 148 s | 334 |
| 750 000 | 666 K | 325 s | 488 |
| 1 000 000 | 888 K | 566 s | 637 |

The per-token cost roughly 5.4× from 131 K to 1 M. For production Flock use over large corpora, amortising a single 1 M prefill across thousands of downstream judgments via prefix-caching is the only route to acceptable wall-clock — which is exactly what the Flock architecture is designed for.

## Cost

| Cell | Wall-clock |
|---|---:|
| 131 K | 82 s |
| 250 K | 196 s |
| 500 K | 889 s |
| 750 K | 1 953 s |
| 1 M | 3 396 s |
| **Total Phase 2C** | ~6 516 s (~108 min) |

At vast.ai 4×H200 SXM @ $17.24/hr, Phase 2C alone = ~$31.

## Interpretation

Three observations worth highlighting:

1. **The YaRN-scaled 1 M context is functionally real on this model.** The shipped `config.json` has `rope_scaling: null` despite the README advertising 1 M; patching it to `{"type":"yarn","factor":4.0,"original_max_position_embeddings":262144}` is sufficient to make vLLM honour the full 1 M max-model-len *and* the model continues to retrieve correctly at positions ≥ 750 K. This is meaningfully different from "technically accepts 1 M tokens but degrades" which is what most post-hoc context-extended models do.
2. **Prefix-caching is essential at these sizes.** A 566 s prefill is painful for an interactive workflow, but if 10 000 downstream Flock judgments share the same 1 M prefix, per-judgment amortised cost drops below 60 ms — competitive with the Phase 1D numbers.
3. **No accuracy degradation anywhere.** The retrieval task is simple (precise 20-word needle, unambiguous query), but so was Phase 2B's, and Kimi-Linear failed even that at 32 K. The architectural factor that matters — standard softmax with GQA — holds the line end-to-end.

## Files

- `size_{131000,250000,500000,750000,1000000}.json` — full per-trial records
- `phase2c_summary.json` — aggregated array for all sizes
- `run_phase2c.py` — harness used on the H200 VM
- This writeup
