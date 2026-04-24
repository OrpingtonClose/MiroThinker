# Employment Execution Plan (v6)

Deliberate model-to-role test matrix for MiroThinker's full architecture.
**Approved for execution. Phase 1A complete.**

**Last updated:** 2026-04-23 (v6 — abliterated model survey incorporated, Phase 1A baseline done)

**H200 instance:** Vast.ai, JP, 4×H200 (575 GB VRAM), $17.24/hr, contract #35487543
**Runner script:** `scripts/h200_test/run_employment.py`
**Corpus:** `scripts/h200_test/test_corpus.txt` (12 KB baseline)
**Extended corpus:** Medical textbooks (626K) + YouTube (338K) + swarm findings (449K) = **1.4 MB**
**Output:** `scripts/h200_test/employment_results/{model}/{test_id}/`
**Survey:** `docs/test_results/abliterated_survey/ABLITERATED_MODEL_SURVEY.md` (280+ models cataloged)

---

## What Changed from v5

v5 was written before the comprehensive abliterated model survey (204K chars, 280+ models,
30+ providers, all technique comparisons). The survey revealed:

1. **Heretic/MPOA abliteration actually *improves* reasoning** — grimjim's MPOA scores
   NatInt 21.33 vs 18.72 baseline. The reasoning-uncensored trade-off is a solved problem.
2. **Quality tiers matter enormously** — KL divergence ranges from 0.0115 (Abliterix)
   to 1.04 (naive methods). A 100× quality range.
3. **New priority models not in v5:** Kimi K2 abliterated (1T), MiMo-V2-Flash (94.1% AIME),
   Qwen3.5-122B-A10B Abliterix (KL 0.0115), GPT-OSS 120B, Dolphin 3.0 R1.
4. **Phase 1A baseline is done:** Llama-70B-abliterated achieves 98.0% Flock accuracy,
   5.5 judgments/sec. This is the bar to beat.
5. **Scout abliterated v2 exists:** `jiangchengchengNLP/Llama-4-Scout-17B-16E-Instruct-abliterated-v2`
6. **Nemotron 3 Super has a Heretic variant:** `mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic`
   (22K downloads, Mamba-2 + Heretic = best quality abliteration on best architecture)

**v6 corrections:**
- **Flock driver candidates expanded** with survey-sourced models
- **Phase 1A results incorporated** as baseline
- **Model rotation reordered** by survey priority
- **Clone-transplant targets updated** with highest-quality abliterated models

---

## Architecture: How Flock Works on vLLM

```
┌─────────────────────────────────────────────────────┐
│                    4×H200 (575 GB VRAM)              │
│                                                       │
│  ┌─────────────────────────────────────────────┐     │
│  │ vLLM Server (TP=2 or TP=4)                  │     │
│  │                                               │     │
│  │  ┌─────────────────────────────────────┐     │     │
│  │  │ PREFIX CACHE (KV)                    │     │     │
│  │  │ = Full 1.4 MB corpus                 │     │     │
│  │  │   computed ONCE, reused every query   │     │     │
│  │  └─────────────────────────────────────┘     │     │
│  │         ↕ (sub-100ms per query)               │     │
│  │  ┌─────────────────────────────────────┐     │     │
│  │  │ FLOCK QUERIES (via flock_complete)   │     │     │
│  │  │ "Is finding X relevant to domain Y?" │     │     │
│  │  │ → 1000s of queries, near-zero cost   │     │     │
│  │  └─────────────────────────────────────┘     │     │
│  └─────────────────────────────────────────────┘     │
│         ↕                                             │
│  ┌─────────────────────────────────────────────┐     │
│  │ ConditionStore (DuckDB)                      │     │
│  │ = Every Flock hit → new AtomicCondition row  │     │
│  │ → knowledge graph grows → next wave queries  │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

**For Clone-Transplant-Flock (Phase 3):**

```
Step 1: Swarm bee runs deep reasoning → rich conversation (findings, chains, hypotheses)
Step 2: Bee's conversation transplanted as ADDITIONAL prefix to smarter model on vLLM
Step 3: Smarter model's prefix = raw corpus + bee's reasoning
Step 4: Flock queries run against this imbued prefix
        → every judgment benefits from BOTH raw data AND bee's domain expertise
Step 5: ConditionStore captures → graph grows → next wave
```

---

## Model Serving Configurations for 4×H200

575 GB total VRAM. Models rotate through serving slots.

### Flock Driver Candidates (served on vLLM)

| Model | HuggingFace ID | Params | Active | VRAM (FP16) | Context | TP | Abliteration | Status |
|---|---|---|---|---|---|---|---|---|
| **Llama-3.3-70B-abliterated** | `huihui-ai/Llama-3.3-70B-Instruct-abliterated` | 70B | 70B | ~140 GB | 128K | 2 | Classic | **BASELINE: 98.0% acc, 5.5/s** |
| **Nemotron 3 Super Heretic** | `mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic` | 121B | 12B | ~242 GB | **1M** | 2 | Heretic (KL low) | DOWNLOADING |
| **Qwen3.5-35B-A3B Abliterated** | `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` | 35B | 3B | ~70 GB | 262K | 1 | Classic | DOWNLOADING |
| **Llama 4 Scout Abliterated v2** | `jiangchengchengNLP/Llama-4-Scout-17B-16E-Instruct-abliterated-v2` | 109B | 17B | ~220 GB | **10M** | 2 | Classic | DOWNLOADING |
| **Qwen3-235B-A22B-abliterated** | `huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-abliterated` | 235B | 22B | ~470 GB | 262K | 4 | Classic | Pending |
| **Qwen3.5-122B-A10B Abliterix** | `wangzhang/Qwen3.5-122B-A10B-abliterix` | 122B | 10B | ~244 GB | 262K | 2 | Bayesian (KL 0.0115) | NEW from survey |

### Clone-Transplant Candidates (Flock driver after receiving bee context)

| Model | Active | Context | Abliteration Quality | Why |
|---|---|---|---|---|
| **Llama 4 Scout Abliterated v2** | 17B | 10M | Classic | Can hold corpus + bee conv + ALL prior findings |
| **Nemotron 3 Super Heretic** | 12B | 1M | Heretic (best) | Mamba-2 = no slowdown as transplanted context grows |
| **Qwen3.5-122B-A10B Abliterix** | 10B | 262K | Bayesian (KL 0.0115) | Virtually identical to base — no quality degradation |
| **DeepSeek-V3 Abliterated** | 37B | 128K | Classic | 671B total, maximum intelligence. Fits 4×H200 at Q4 (~227 GB) |

### Worker Models (API-based, few expensive calls)

| Model | Provider | Active | Price | Role |
|---|---|---|---|---|
| DeepSeek V3.2 | DeepSeek | ~37B | $1.10/M | Proven worker baseline |
| GPT-4.1-mini | OpenAI | ? | $0.40/$1.60 | Highest finding count (80) |
| Kimi K2.6 | Moonshot | 40B | $0.56/$3.50 | Muon-optimized cross-domain reasoning |
| Grok 4.1-fast | xAI | ? | $0.20/$0.50 | UNCENSORED, 2M context |
| Ling 2.6 1T | OpenRouter | 63B | FREE | Highest free active params |

### Meta-Expert (single expensive call, all results concatenated)

| Model | Provider | Context | Price | Role |
|---|---|---|---|---|
| Grok 4.20 | xAI | **2M** | $2/$6 | Holds entire project, finds cross-domain bridges |
| Claude Opus 4.7 | Anthropic | 1M | $5/$25 | Best meta-reasoning (refuses direct, full meta) |
| Gemini 3.1 Pro | Google | 1M | $2/$12 | UNCENSORED + SOTA reasoning |

---

## Hypotheses

### Confirmed (Slots 2-4)

| ID | Finding |
|---|---|
| H1 | Dense 70B > MoE 3B for depth |
| H3 | Weight-surgery abliteration > fine-tune |
| H6 | ministral-3b = best API flock driver (90%, 286ms) |
| H8 | Cross-expert pollination works (16 bridges) |
| H9 | Expert disagreement = diagnostic signal |

### New (v5)

| ID | Hypothesis | Phase |
|---|---|---|
| H10 | Full-corpus Flock via vLLM prefix cache beats chunked SQL+LLM | 1 |
| H11 | Mamba-2 (Nemotron) outperforms transformer for cached Flock | 1 |
| H12 | DeltaNet linear attention (Qwen 3.6) maintains quality on complex reasoning | 1 |
| H13 | Qwen 3.6 27B dense vs 35B-A3B MoE — 9× more active params = better findings? | 1 |
| H14 | 10M Scout can hold entire corpus + findings in single prefix | 2 |
| H15 | Bee clone transplanted to smarter model produces higher Flock accuracy than either alone | 3 |
| H16 | ConditionStore growth across waves improves Flock hit quality | 3 |
| H17 | 2M meta-expert finds cross-domain bridges no single model could | 4 |
| H18 | Censorship differs across providers for same weights | 0 |

---

## Execution Plan

### Phase 0: Censorship Pre-Screen (API, cheap, parallel with H200 setup)
**Duration:** ~45 min | **Cost:** ~$2

Runs FROM THIS VM (Devin) while H200 downloads models.

#### 0A: Architecture Stars (FREE on OpenRouter)

| Model | Provider | Active | Architecture | Why Priority |
|---|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | OpenRouter | 12B | Mamba-2 + MoE | Top Flock candidate |
| `nvidia/nemotron-3-nano-30b-a3b` | OpenRouter | 3B | Mamba + MoE | ministral-3b competitor |
| `inclusionai/ling-2.6-1t` | OpenRouter | 63B | Hybrid linear | Best free worker |
| `inclusionai/ling-2.6-flash` | OpenRouter | 7.4B | MoE | Fast free worker |
| `google/gemma-4-26b-a4b-it` | OpenRouter | 4B | MoE | Google free |
| `google/gemma-4-31b-it` | OpenRouter | 31B | Dense | Google free |

**Cost: $0**

#### 0B: New 1M+ Models

| Model | Provider | Context | Price |
|---|---|---|---|
| `qwen3.6-plus` | DashScope | 1M | $0.33/$1.95 |
| `qwen3.6-27b` | DashScope | 262K | TBD |
| `qwen3.6-35b-a3b` | DashScope | 262K | TBD |
| `qwen3.6-max-preview` | DashScope | ? | TBD |
| `xiaomi/mimo-v2.5-pro` | OpenRouter | 1M | $1/$3 |
| `amazon/nova-premier-v1` | OpenRouter | 1M | $2.50/$12.50 |

**Cost: ~$2**

#### 0C: Cross-Provider Censorship (H18)

Test same weights on different providers:

| Model | Native (censored?) | Alternative |
|---|---|---|
| Claude Opus 4.7 | Anthropic | Venice |
| GPT-5.4 | OpenAI | Venice |
| Kimi K2.6 | Moonshot | Fireworks |
| Qwen 3.6 Plus | DashScope | OpenRouter |

**Cost: ~$2**

**Gate:** Models scoring UNCENSORED/USABLE proceed. LIMITED/REFUSED excluded from worker/flock.

---

### Phase 1: vLLM Flock Driver Tournament (H200)
**Duration:** ~4 hours | **Cost:** $38 (H200 rental only)

The core test. Each model is served on vLLM with the corpus as prefix cache.
Run 50 Flock relevance judgments per model. Measure accuracy, latency, throughput.

#### Slot 1A: Llama-3.3-70B-abliterated (BASELINE)

Already proven in Slot 2. Re-run with prefix caching enabled to establish baseline.

**vLLM command:**
```bash
vllm serve huihui-ai/Llama-3.3-70B-Instruct-abliterated \
  --port 8000 --max-model-len 131072 --gpu-memory-utilization 0.90 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --enable-prefix-caching --tensor-parallel-size 2 --host 0.0.0.0
```

| Test | What | Metric |
|---|---|---|
| 1A-1 | 50 Flock judgments, 12KB corpus as prefix | Accuracy, latency per judgment |
| 1A-2 | 50 Flock judgments, full 1.4MB corpus as prefix | Accuracy at scale, prefix cache benefit |
| 1A-3 | 200 rapid-fire Flock judgments (throughput stress) | Judgments/sec, cache hit rate |

#### Slot 1B: Nemotron 3 Super 120B-A12B (H11 — Mamba-2)

**vLLM command:**
```bash
vllm serve nvidia/Nemotron-3-Super-120B-A12B-Instruct \
  --port 8000 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --enable-prefix-caching --tensor-parallel-size 2 --host 0.0.0.0
```

| Test | What | Metric |
|---|---|---|
| 1B-1 | 50 Flock judgments, 12KB corpus | Baseline accuracy |
| 1B-2 | 50 Flock judgments, 1.4MB corpus | Does Mamba-2 handle 1M prefix better? |
| 1B-3 | 200 rapid-fire (MTP speed test) | Multi-token prediction = faster? |
| 1B-4 | Same 50 judgments split: 25 simple + 25 complex | Does Mamba degrade on complex reasoning? |

#### Slot 1C: Qwen 3.6 35B-A3B (H12 — DeltaNet Linear Attention)

Only 3B active = very fast. DeltaNet uses fixed-size recurrent state.
Tests whether linear attention hurts Flock accuracy (MiniMax warning).

**vLLM command:**
```bash
vllm serve Qwen/Qwen3.6-35B-A3B-Instruct \
  --port 8000 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --enable-prefix-caching --tensor-parallel-size 1 --host 0.0.0.0
```

| Test | What | Metric |
|---|---|---|
| 1C-1 | 50 Flock judgments, 12KB corpus | Accuracy vs 70B baseline |
| 1C-2 | 50 Flock judgments, 1.4MB corpus | DeltaNet at scale |
| 1C-3 | 25 simple + 25 complex | Does DeltaNet degrade like MiniMax Lightning? |

#### Slot 1D: Qwen3-235B-A22B-abliterated (H5 — Quality Ceiling)

Completing interrupted Slot 5. 22B active MoE on 4 GPUs.

**vLLM command:**
```bash
vllm serve huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-abliterated \
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --tensor-parallel-size 4 --max-num-seqs 8 --host 0.0.0.0
```

| Test | What | Metric |
|---|---|---|
| 1D-1 | T1 Worker on A1 (Milos insulin) | Finding count + specificity vs 70B (30) |
| 1D-2 | 50 Flock judgments, 12KB corpus | Quality ceiling for Flock accuracy |
| 1D-3 | T5 Cross-Expert on A1+B4 | Meta-reasoning quality |

#### Model Rotation Schedule (H200)

```
Hour 0-0.5:    Download Llama-70B (if not cached) + Nemotron Super + Qwen 3.6 35B
Hour 0.5-1.5:  Slot 1A — Llama-70B baseline (3 tests)
Hour 1.5-2.5:  Slot 1B — Nemotron Super (4 tests)  [swap: stop 70B, start Nemotron]
Hour 2.5-3.5:  Slot 1C — Qwen 3.6 35B (3 tests)    [swap: stop Nemotron, start Qwen]
Hour 3.5-5:    Slot 1D — Qwen3-235B (3 tests)       [swap: needs TP=4, download if needed]
```

---

### Phase 2: 10M Scout + Worker Quality (H200 + API parallel)
**Duration:** ~3 hours | **Cost:** ~$47 ($38 H200 + ~$9 API)

#### 2A: Llama 4 Scout at Extended Context (H14 — 10M)

**The test no API can run.** Scout's 10M native context on H200.

**vLLM command:**
```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --port 8000 --gpu-memory-utilization 0.92 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --enable-prefix-caching --tensor-parallel-size 2 --host 0.0.0.0 \
  --max-model-len {SCALED_PER_TEST}
```

| Test | Context | What We Learn |
|---|---|---|
| 2A-1 | 300K (Together parity) | Baseline — self-hosted matches API? |
| 2A-2 | 500K | Beyond any provider's cap |
| 2A-3 | 1M | Full corpus Flock — direct comparison to Phase 1 results |
| 2A-4 | 2M+ (if VRAM allows) | Stress test — corpus + textbooks + YouTube + all findings |

50 Flock judgments at each context size. Same ground truth throughout.
**Metric:** Accuracy degradation curve as context grows.

#### 2B: Worker Quality Tournament (API, parallel with 2A)

Runs FROM THIS VM while H200 runs Scout tests.

| Model | Provider | Topic | Cost |
|---|---|---|---|
| Kimi K2.6 | Moonshot | A1 | ~$1 |
| Ling 2.6 1T | OpenRouter | A1 | FREE |
| Qwen 3.6 27B | DashScope | A1 | ~$0.50 |
| Qwen 3.6 35B-A3B | DashScope | A1 | ~$0.50 |
| Nemotron 3 Super | OpenRouter | A1 | FREE |
| Gemma 4 31B | OpenRouter | A1 | FREE |
| DeepSeek V3.2 | DeepSeek | A1 | ~$1 |
| GPT-4.1-mini | OpenAI | A1 (re-run for comparison) | ~$1 |

All same prompt, same corpus. Count findings, measure specificity, check censorship.

**Baselines:** Llama-70B: 30, Hermes: 72, Qwen3-30B: 40, GPT-4.1-mini: 80

**Cost: ~$4 API**

---

### Phase 3: Clone-Transplant-Flock (Dream Serendipity Setup)
**Duration:** ~3 hours | **Cost:** ~$47 ($38 H200 + ~$9 API)

**This is the architecturally transformative test.**

#### 3A: Generate Bee Clone (API or local)

Run a full MiroThinker swarm bee session on topic A1 (Milos insulin).
The bee reasons deeply — extracts findings, builds cross-domain hypotheses,
generates a rich conversation history (the clone).

| Bee Model | Provider | Expected Conv Size | Why |
|---|---|---|---|
| DeepSeek V3.2 | API | ~50K tokens | Proven deep reasoner, UNCENSORED |
| Llama-70B-abliterated | H200 | ~50K tokens | Guaranteed uncensored, deep |

#### 3B: Transplant to Smarter Model + Run Flock

Take the bee's conversation and inject it as ADDITIONAL prefix into a
smarter/larger model on vLLM. The prefix cache now holds:
**raw corpus (1.4MB) + bee's entire reasoning conversation (~50K tokens)**

The smarter model runs Flock queries. Every judgment benefits from BOTH
the raw data AND the bee's domain expertise.

| Flock Driver (receives transplant) | Active | Context | What We Learn |
|---|---|---|---|
| **Nemotron 3 Super** | 12B | 1M | Mamba-2 won't slow down with larger prefix (H15) |
| **Llama 4 Scout** | 17B | 10M | Can hold corpus + bee conv + ALL prior findings (H14) |
| **Llama-70B-abliterated** | 70B | 128K | Dense reasoning on transplant (context limited) |

| Test | What | Metric |
|---|---|---|
| 3B-1 | 50 Flock judgments WITH transplant | Accuracy |
| 3B-2 | 50 Flock judgments WITHOUT transplant (same model, same corpus, no bee conv) | Control |
| 3B-3 | Compare 3B-1 vs 3B-2 | **H15: Does the transplant improve Flock?** |
| 3B-4 | Run Flock → ConditionStore captures → load expanded Store → run Flock again | **H16: Does the Store growth improve next wave?** |

#### 3C: Multi-Bee Transplant (if time allows)

Run 3 different bees (different topics: A1, B4, F9) → transplant ALL their
conversations to Scout at 10M → run Flock across all domains simultaneously.
The model holds 3 experts' reasoning + the full corpus.

---

### Phase 4: Meta-Expert Synthesis (H17)
**Duration:** ~1 hour | **Cost:** ~$10 API

**Prerequisite:** Phases 1-3 complete. Need results to concatenate.

Concatenate ALL results from ALL phases into a single prompt:

| Input | Size |
|---|---|
| All worker findings (Phases 1D, 2B) | ~100K tokens |
| All Flock results (Phases 1, 3) | ~50K tokens |
| ConditionStore export (all AtomicConditions) | ~100K tokens |
| Medical textbooks | ~160K tokens |
| YouTube transcripts | ~85K tokens |
| **Total** | **~500K tokens** |

| Model | Provider | Context | Cost |
|---|---|---|---|
| **Grok 4.20** | xAI | **2M** | ~$3 |
| **Claude Opus 4.7** | Anthropic | 1M | ~$5 |
| **Gemini 3.1 Pro** | Google | 1M | ~$3 |

**Prompt:** "You hold every finding from every model, every Flock judgment, every
cross-domain bridge. Find connections that no single model identified."

---

## Timeline

```
PHASE 0 (from Devin VM, parallel with H200 setup):
  Hour 0-0.75:   Censorship pre-screen (0A + 0B + 0C)

PHASE 1 (H200):
  Hour 0-0.5:    Download models to H200
  Hour 0.5-1.5:  Slot 1A — Llama-70B Flock baseline
  Hour 1.5-2.5:  Slot 1B — Nemotron Super Flock (Mamba-2)
  Hour 2.5-3.5:  Slot 1C — Qwen 3.6 35B Flock (DeltaNet)
  Hour 3.5-5:    Slot 1D — Qwen3-235B quality ceiling

PHASE 2 (H200 + API parallel):
  Hour 5-7:      2A — Scout at 300K/500K/1M/2M+ (H200)
  Hour 5-7:      2B — Worker tournament (API from Devin VM)

PHASE 3 (H200):
  Hour 7-8:      3A — Generate bee clone (API or local)
  Hour 8-10:     3B — Transplant to smarter model + Flock
  Hour 10-11:    3C — Multi-bee transplant at 10M (if time)

PHASE 4 (API):
  Hour 11-12:    Meta-expert synthesis (Grok + Claude + Gemini)
```

**Total wall clock:** ~12 hours
**H200 cost:** 12h × $9.42 = ~$113
**API cost:** ~$25
**Grand total:** ~$138

---

## Cost Summary

| Phase | What | H200 | API | Total |
|---|---|---|---|---|
| 0 | Censorship pre-screen | — | ~$4 | ~$4 |
| 1 | vLLM Flock tournament (4 models) | ~$47 | — | ~$47 |
| 2 | 10M Scout + worker quality | ~$19 | ~$4 | ~$23 |
| 3 | Clone-transplant-Flock | ~$28 | ~$3 | ~$31 |
| 4 | Meta-expert synthesis | — | ~$11 | ~$11 |
| **TOTAL** | | **~$113** | **~$22** | **~$135** |

---

## Success Criteria

### Must-Pass
- [ ] Phase 0: ≥ 4/6 architecture-star models pass censorship
- [ ] Phase 1: Prefix-cached Flock works on at least 3/4 models
- [ ] Phase 1B: Nemotron Mamba-2 completes 50 Flock judgments at 1.4MB corpus
- [ ] Phase 2A: Scout serves at ≥ 1M context on H200

### Should-Pass
- [ ] H10: Prefix-cached Flock at 1.4MB ≥ 85% accuracy (matching chunked baseline)
- [ ] H11: Nemotron Super faster than Llama-70B at same accuracy
- [ ] H14: Scout at 1M matches quality of Scout at 300K (no degradation)
- [ ] H15: Transplanted Flock > non-transplanted Flock (accuracy delta > 5%)
- [ ] H17: Meta-expert finds ≥ 5 cross-domain bridges not in any single phase

### Would-Be-Amazing
- [ ] Scout at 2M+ produces coherent Flock results
- [ ] DeltaNet (Qwen 3.6) matches transformer accuracy on complex reasoning
- [ ] Multi-bee transplant at 10M finds connections none of the individual bees found
- [ ] ConditionStore growth measurably improves Flock accuracy wave-over-wave

---

## Completed Results

### Phase 1A: Flock Baseline (Llama-70B-abliterated, vLLM prefix caching)

**Date:** 2026-04-23 | **Full results:** `docs/test_results/phase1a/`

| Test | Judgments | Accuracy | Avg Latency | Throughput |
|---|---|---|---|---|
| 1A-1 (50, concurrency=5) | 50 | **98.0%** | 848ms | 5.5/sec |
| 1A-3 (200 rapid-fire, concurrency=20) | 200 | **98.0%** | 2,979ms | 6.3/sec |

**Verdict:** Prefix caching validated. 98% accuracy is the baseline to beat.

### Prior Slots (from earlier sessions)

#### Slot 2: Llama-3.3-70B-abliterated

| Test | Result |
|---|---|
| T1 Worker A1 | 30 findings, UNCENSORED |
| T3 Flock (API) | 84% accuracy, 210ms, 10.6 tok/s |
| T4 Clone Context | 5 waves, 4K→22K tokens, 100% |
| T5 Cross-Expert | 16 cross-domain bridges |

#### Slot 3: Hermes-3-70B (NousResearch fine-tune)

| Test | Result |
|---|---|
| T1 Worker A1 | 72 findings (insulin UNCENSORED) |
| T1 Worker B1 | REFUSED (trenbolone) — **confirms H3: fine-tune < weight-surgery** |

#### Slot 4: Qwen3-30B-A3B

| Test | Result |
|---|---|
| T1 Worker A1 | 40 findings, UNCENSORED |
| T1 Worker B3 | 10 findings, flagged not refused |

#### API Track

| Model | A1 Findings |
|---|---|
| DeepSeek V3.2 | 31 |
| GPT-4.1-mini | **80** |
| Grok 3-fast | 31 |
| ministral-3b (Flock via API) | **90% accuracy, 286ms** |
