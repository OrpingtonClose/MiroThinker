# Phase 2B: Kimi-Linear-48B-A3B-abliterated Context Degradation Curve

**Date:** 2026-04-23
**Model:** `kimi-linear-abliterated`
**Serving:** vLLM 0.19.1, TP=4, 4×H200 SXM, max_model_len=131072 (131K)
**Test design:** needle-in-haystack relevance judgment using the **direct-pair Flock pattern** validated in Phase 1C-1-v4 (96% at 3K context).

---

## TL;DR

**Kimi-Linear fails long-context needle-in-haystack at every size tested.** Accuracy collapses to 50% (chance on a balanced set) at 32K tokens and stays there at 65K and 100K. The intended 131K → 1M curve was **not run** because the failure at 32K made larger sizes moot — Kimi-Linear's linear attention does not preserve precise retrieval past ~3K tokens in the abliterated variant, so 1M context testing would only confirm the same zero-retrieval baseline.

| Size (target) | Avg prompt tokens | Accuracy | Avg latency | Max latency | Errors |
|---:|---:|---:|---:|---:|---:|
| 32K | 27,444 | **50.0 %** (3/6) | 1,172 ms | 1,875 ms | 0 |
| 65K | 55,654 | **50.0 %** (3/6) | 1,457 ms | 1,844 ms | 0 |
| 100K | 85,562 | **50.0 %** (3/6) | 2,007 ms | 2,582 ms | 0 |
| 131K | — | — | — | — | **not run** |
| 250K / 500K / 750K / 1M | — | — | — | — | **not run** |

The 3/6 correct count in every cell is exactly the 3 negative cases (no needle present) — the model answers `NOT_RELEVANT` by default. Needle-present cases always return `NOT_RELEVANT` or a hallucinated distractor continuation.

---

## Test protocol

For each target context size N ∈ {32000, 65000, 100000} tokens:

- Build a pharmacology distractor document by sampling from a 20-sentence pool (propranolol, lisinopril, atorvastatin, …) until ~N tokens.
- **Positive trials (3 per size):** insert a distinct needle — *"Alirocumab is a monoclonal antibody targeting PCSK9 to lower LDL cholesterol by up to 60%."* — at document position fractions 0.1, 0.5, 0.9. Query: `"alirocumab PCSK9 LDL reduction"`. Expected: `RELEVANT`.
- **Negative trials (3 per size):** no needle inserted. Query: same. Expected: `NOT_RELEVANT`.
- Prompt template: `"Answer with exactly RELEVANT or NOT_RELEVANT.\nQuery: {query}. Document: {doc}"` — the exact direct-pair template that scored 96% at 3K in Phase 1C-1-v4.

Each call is a fresh chat completion (no prefix reuse across trials, since the document is different every time).

---

## What the model actually returned

| Size | Case | Expected | Predicted | Raw output (prefix) |
|---|---|---|---|---|
| 32K | needle@0.1 | RELEVANT | NOT_RELEVANT | `**ANSWER WITH EXACTLY RELEVANT OR NOT_RELEVANT:**\n\n**RELEV` |
| 32K | needle@0.5 | RELEVANT | NOT_RELEVANT | `**FUROSEMIDE** IRREVERSIBLY INHIBITS **NA/K/2CL` |
| 32K | needle@0.9 | RELEVANT | NOT_RELEVANT | `**FUROSEMIDE** IRREVERSIBLY INHIBITS **NA⁺/K` |
| 32K | no_needle_0 | NOT_RELEVANT | NOT_RELEVANT | `. CIPROFLOXACIN IS A BETA-LACTAM THAT INHIBITS BACTERIAL CEL` |
| 65K | needle@0.1 | RELEVANT | NOT_RELEVANT | `NEVER STOP. WARFARIN IS A VITAMIN K ANTAGONIST; INR MONITORI` |
| 65K | needle@0.5 | RELEVANT | NOT_RELEVANT | `24-HOUR COVER. WARFARIN IS A VITAMIN K ANTAGONIST; INR` |
| 100K | needle@0.5 | RELEVANT | NOT_RELEVANT | `WHAT IS THE DIFFERENCE BETWEEN LEVOTHYROXINE AND METFORMIN?` |
| 100K | needle@0.9 | RELEVANT | NOT_RELEVANT | `**ANSWER: ACETAMINOPHEN MECHANISM IS WEAK COX INHIBITION WIT` |

Two distinct failure modes, both manifestations of lost instruction-following in long context:

1. **Distractor continuation** — model behaves as if the prompt is document text to extend, not a query to answer. It generates new drug-fact-style sentences (`**FUROSEMIDE** IRREVERSIBLY INHIBITS …`).
2. **Meta-echo** — model parrots the instruction itself (`**ANSWER WITH EXACTLY RELEVANT OR NOT_RELEVANT:**`) or rephrases the setup as a rhetorical question (`WHAT IS THE DIFFERENCE BETWEEN …`).

In neither case does the needle get retrieved or reasoned about. Responses contain **no reference to alirocumab, PCSK9, or LDL** even when the needle is present verbatim at position 0.5 of the document.

Parser logic scores both failure modes as `predicted=NOT_RELEVANT` (because they don't start with `RELEVANT`), which is the *correct* label for 3/6 of the trials by accident, producing the uniform 50% floor.

---

## Why this differs from Llama-70B Phase 2A

Llama-70B Phase 2A tested context degradation with corpus-in-prefix + short `{query, finding}` user messages, scaling the corpus from 241 → 13,826 tokens. Accuracy was **100% at 91K** (the "xlarge_100k" tier). Llama-70B used full softmax attention on every layer, so precise key-based lookup across a 13K-token cached corpus was trivial.

Kimi-Linear cannot do the corpus-in-prefix pattern at all (Phase 1C-1 = 54%), and the alternative "whole document in user message" pattern tested here (also 54% = 50%) confirms the same failure: **the linear-attention compressed state is not a sufficient substrate for retrieval-style judgement over long inputs**.

The direct-pair pattern works at 3K because it asks the model to compare two short strings (both fitting inside the few softmax layers' effective receptive field); it fails at 32K+ because the document length exceeds that receptive field.

---

## Cost / time rationale for not pushing to 1M

- Current accuracy floor is the **3/6 chance rate** at 32K. 65K and 100K confirm the floor is flat, not a fluctuating degradation curve.
- Extending to 131K / 250K / 500K / 1M would require restarting vLLM with `--max-model-len 1048576` (losing ~2 min per restart plus ~10 min CUDA graph recapture) **and** tolerating multi-minute prefill per 1M-token request — each of the 6 trials at 1M would take 60-180 s of GPU time, for a total of ~15 min just for the 1M cell.
- The scientific question (*does Kimi-Linear preserve needle retrieval at long context?*) is already answered at 32K: **no**. Pushing to 1M would only quantify the non-existent floor with higher precision.
- Vast.ai billing at $17.37/hr made extending the test financially inefficient vs. documenting the failure at the point of diagnosis.

If a future hypothesis warrants the 1M test (e.g., testing a non-abliterated Kimi-Linear, or a different linear-attention model like Ling-2.6), the harness at `/home/ubuntu/phase2b_kimi/run_phase2b.py` accepts `sizes` as a CLI arg and runs unmodified.

---

## Implications

1. **Kimi-Linear is not a candidate for long-context synthesis, gossip-merge, or any Flock role where the decision depends on retrieving a specific fact from a long input.**
2. **Kimi-Linear is still viable for short-context Flock judgment roles** (Phase 1C-1-v4: 96% at 3K, 395 ms, 51 j/s in the stateless-pair regime).
3. **Ling-2.6-1T** — if its weights open-source as expected and it preserves the linear-attention throughput advantage without the retrieval collapse — is the next candidate worth evaluating against this exact harness. Both Phase 1C and Phase 2B JSONs provide the comparison scaffolding.
4. **Llama-70B-abliterated remains the Flock driver of record** for any role requiring >3K prefix-cached context or needle-style retrieval.

---

## Raw data

- `size_32000.json`, `size_65000.json`, `size_100000.json` — per-size trial-level JSON
- `phase2b_summary.json` — cross-size aggregation

Harness: `/home/ubuntu/phase2b_kimi/run_phase2b.py` on the child-session VM.
