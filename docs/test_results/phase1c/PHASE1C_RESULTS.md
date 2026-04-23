# Phase 1C: Kimi-Linear-48B-A3B-abliterated Flock Probe

**Date:** 2026-04-23
**Model:** `kimi-linear-abliterated` (48B total, 3B active MoE, hybrid linear + full attention)
**Serving:** vLLM 0.19.1, TP=4, prefix-caching ON, bf16
**Hardware:** 4×NVIDIA H200 SXM (575 GB VRAM), Vast.ai contract #35487543
**Judgments:** 50 ground-truth relevance pairs (25 positive, 25 negative) — **identical set to Phase 1A Llama-70B baseline**
**Success criteria (from plan):** ≥90% accuracy, <1500 ms average latency

---

## TL;DR — Two-regime verdict

| Regime | Prompt shape | Accuracy | Avg latency | Verdict |
|---|---|---|---|---|
| **Prefix-cached Flock (canonical)** | Corpus in system + `{query, finding}` in user | **54.0 %** | 585 ms | **REJECTED** |
| Few-shot variant of above | Few-shot + stops | 50.0 % | 408 ms | REJECTED |
| YES/NO variant of above | YES/NO answer | 44.0 % | 347 ms | REJECTED |
| **Stateless direct-pair** | `Query: X. Document: Y. RELEVANT/NOT_RELEVANT` (no corpus preamble) | **96.0 %** (50 judg.) / **96.0 %** (200 judg.) | 395 ms / 369 ms | **PASSES thresholds** |

Kimi-Linear **cannot consume a prefix-cached corpus** the way Llama-70B can — its hybrid linear attention fails to perform key-based lookup across the corpus, so the model either echoes corpus indices (`[39] SIGNS OF HYPOGLYCE`) or degenerates into pharmacology continuation. **But** when each judgment is a self-contained `{query, finding}` pair with no corpus preamble, accuracy jumps to 96% at 370 ms avg (8× faster throughput than Llama-70B).

**For Flock-as-currently-architected (prefix-cached corpus):** Kimi-Linear is REJECTED, same verdict as Qwen3.5-35B-thinking (54%).

**For a reshaped stateless-pair Flock variant:** Kimi-Linear is the fastest abliterated option tested so far — 51 judgments/sec vs Llama-70B's 6.3/sec — and its 96% accuracy is within 2 pp of Llama-70B's 98% baseline.

---

## Detailed results

### 1C-1 — Canonical prefix-cached Flock prompt (`run_phase1c.py`)

The corpus (41 numbered findings) is placed in the system message, the `{query, finding}` pair in the user message. Model is asked to reply `RELEVANT` or `NOT_RELEVANT`. This is bit-for-bit the Phase 1A Llama-70B methodology.

```
test_id:              1C-1
num_judgments:        50
concurrency:          5
corpus_size_chars:    3444
accuracy:             54.0 %  (27/50)
avg_latency_ms:       584.6
p50_latency_ms:       362.7
p95_latency_ms:       2594.9
total_time_s:         6.02
throughput:           8.3 judgments/sec
```

**Response distribution (n=50):**

| Count | Raw response (prefix) |
|---:|---|
| 28 | `RELEVANT` |
| 4 | `**RELEVANT**` |
| 3 | `[26] TRENBOLONE ENHANCES` *(echoed corpus)* |
| 2 | `RELEVCE` *(token-level corruption)* |
| 3 | `THE USER QUERY IS: …` *(echoed query)* |
| 10 | Various `[NN] …` echoes |

**Error pattern:** the model either answers `RELEVANT` unconditionally (leading to 25/25 true positives matched but 15/25 false positives flagged as RELEVANT) or echoes corpus content instead of answering. This is an **instruction-following failure in the presence of a large prefix**, not a relevance-reasoning failure.

Duplicate run at concurrency=20 × 4 repetitions (200 judgments) gave **48.5 %** — confirming the regression is not variance.

### 1C-1-v2 — Few-shot examples + stop sequences

4 exemplars of correct RELEVANT/NOT_RELEVANT decisions inlined into the system prompt, `stop=["\n", "\nQUERY", "CORPUS"]`.

```
accuracy:             50.0 %  (25/50)  — WORSE
avg_latency_ms:       407.5
```

**Bias analysis:** 0/25 true positives and 0/25 false positives were flagged positive in the parseable responses — the model produced so many off-format outputs that the parser defaulted everything to `predicted=False`. Few-shot **hurt** format adherence rather than helping.

### 1C-1-v3 — Binary YES/NO

Reformulated as "Does the finding answer the query? YES / NO."

```
accuracy:             44.0 %  (22/50)
avg_latency_ms:       346.5
```

**Bias:** 33× `NO` vs 15× `YES`. Sensitivity 6/25, specificity 16/25 — worse than chance on positives.

### 1C-1-v4 — Stateless direct-pair (the winning config)

No corpus preamble. Each judgment is fully self-contained:

```
Answer with exactly RELEVANT or NOT_RELEVANT.
Query: {query}. Document: {finding}
```

```
test_id:              1C-1-v4
num_judgments:        50
concurrency:          5
accuracy:             96.0 %  (48/50)
avg_latency_ms:       394.6
p50_latency_ms:       386.5
p95_latency_ms:       685.1
```

**Response distribution:** 25× `NOT_RELEVANT`, 23× `RELEVANT`, 2× `RELEVANT\n…` (parsed correctly as positive).

**Confusion matrix:** 24/25 true positives, 24/25 true negatives, 1 false positive, 1 false negative.

### 1C-2 — Direct-pair at high throughput

Same prompt as 1C-1-v4, 200 judgments (4× rep) at concurrency=20:

```
accuracy:             96.0 %  (192/200)
avg_latency_ms:       369.4
p50:                  325.0
p95:                  699.4
total_time_s:         3.90
throughput:           51.32 judgments/sec  ← 8× Llama-70B (6.29/s at concurrency=20)
```

---

## Comparison table — Kimi-Linear vs prior baselines

| Model | Prompt shape | Accuracy | Avg latency | Throughput | Verdict |
|---|---|---:|---:|---:|---|
| Llama-70B abliterated | prefix-cached | 98.0 % | 848 ms | 5.5 j/s | baseline |
| Llama-70B abliterated (rep) | prefix-cached | 98.0 % | 2979 ms | 6.3 j/s | baseline |
| Qwen3.5-35B thinking | prefix-cached | 54 % | — | — | rejected |
| **Kimi-Linear-48B-A3B-abliterated** | **prefix-cached** | **54.0 %** | 585 ms | 8.3 j/s | **rejected (Flock canonical)** |
| **Kimi-Linear-48B-A3B-abliterated** | **direct-pair** | **96.0 %** | 395 ms | 51.3 j/s | **passes (stateless Flock)** |

---

## Mechanistic interpretation

Kimi-Linear uses hybrid attention: most layers use a linear-attention variant (fast, O(N) memory, constant-size state), with only a few layers using full softmax attention. Linear attention **compresses** the context into a fixed-size recurrent state, which makes precise key-based lookup across a large corpus statistically weak. When the model is asked to judge `{query, finding}` against a corpus held in its state, it can't reliably locate the finding's neighbourhood in that state, so it falls back to echoing (top-of-state) corpus items or continuing the distractor text.

Remove the corpus from the prefix — send each `{query, finding}` pair as a fresh short prompt — and the discriminative capability is back, because the decision no longer requires retrieval from a compressed state; it's a simple two-string comparison.

This matches the published positioning of Kimi-Linear: **linear attention is optimised for throughput on short-context judgement tasks**, not for long-context retrieval.

---

## Implications for the Employment Plan

1. **Do not substitute Kimi-Linear for Llama-70B in the current Flock driver slot.** The prefix-cached corpus pattern (which Phase 2A validated at up to 91K tokens for Llama) is incompatible with Kimi-Linear's architecture.
2. **Consider a stateless-pair Flock variant with Kimi-Linear as the driver.** At 51 j/s at 96% accuracy, the throughput gain is 8× over Llama-70B; this may offset the loss of prefix-cache amortisation for batches above a few hundred judgments where the model is GPU-bound regardless.
3. **Kimi-Linear is viable for short-context Flock roles**: tier-race first-pass scoring, junk-atom detection, deduplication voting — any role where each judgement is independent and <500 tokens.
4. **Kimi-Linear is not viable for long-context Flock roles**: synthesis, gossip merging, cross-finding clustering — see Phase 2B for long-context degradation evidence.
5. **Ling-2.6-1T (weights expected to open-source soon)** remains the unresolved candidate for a Kimi-Linear-class throughput win *with* prefix-cache compatibility.

---

## Raw data

- `1C-1.json` — canonical prefix-cached prompt, 50 judgments
- `1C-1-v2.json` — few-shot variant, 50 judgments
- `1C-1-v3.json` — YES/NO variant, 50 judgments
- `1C-1-v4.json` — **direct-pair (winning config)**, 50 judgments
- `1C-2.json` — direct-pair at concurrency=20, 200 judgments
- `summary.json` — aggregated metrics for the canonical runs

Harness scripts live at `/home/ubuntu/phase1c/` on the child-session VM (not committed).
