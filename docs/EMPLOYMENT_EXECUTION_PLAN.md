# Employment Execution Plan (v4)

Deliberate model-to-role test matrix for MiroThinker's full architecture.
**Go/no-go checkpoint — review and approve before execution begins.**

**Last updated:** 2026-04-16 (v4 — architecture-informed redesign)

**Runner script:** `scripts/h200_test/run_employment.py`
**Corpus:** `scripts/h200_test/test_corpus.txt` (Milos corpus, 128 lines, 12 KB)
**Extended corpus:** Medical textbooks (626K chars) + YouTube transcripts (338K chars) + swarm findings (449K chars) = **1.4 MB total**
**Output:** `scripts/h200_test/employment_results/{model}/{topic_id}/`
**H200 instance:** Vast.ai, US, 4×H200 (575 GB VRAM), $9.42/hr

**Supporting docs:**
- `MODEL_REGISTRY.md` — full 823-model inventory across 16 providers
- `MODEL_USEFULNESS_ANALYSIS.md` — architecture deep dives per model family

---

## What Changed from v3

v3 treated all models as interchangeable API endpoints. Architecture research
revealed that attention mechanism, MoE routing, and context serving limits
fundamentally change which models suit which roles. Key corrections:

1. **Llama 4 Scout has 10M native context** — but Groq caps at 131K, Together
   at 300K. Self-hosting on H200 is the only path to test beyond 300K.
2. **Nemotron 3 Super is a hybrid Mamba-2 + MoE + Attention model** with 1M
   context and multi-token prediction (2-7× faster than peers). This is
   purpose-built for exactly what Flock needs. Elevated to top priority.
3. **MiniMax abandoned Lightning attention in M2** due to poor reasoning quality.
   This is a WARNING for all linear attention models (Ling, Qwen DeltaNet).
   We now explicitly test linear attention quality vs standard attention.
4. **Ling 2.6 1T has 63B active params and is FREE** — if uncensored, it's the
   highest-quality free model available. Elevated to censorship probe priority.
5. **Provider-served context ≠ native context.** Added a self-hosting track.

---

## Architectural Tiers Under Test

| Tier | What | Model Needs | Status |
|---|---|---|---|
| **T1: Worker** | Tool-free reasoning on data packages | Uncensored, strong reasoning | Tested (Slots 2-4) |
| **T2: Research Organizer** | Clone-based gap resolution | Tool-capable, search queries | Partially tested |
| **T3: Flock Battery** | SQL+LLM relevance scoring | Fast, cheap, accurate | Tested (basic) |
| **T4: Clone Context** | Session proxy prepends conversation | Stable with growing context | Tested (basic) |
| **T5: Cross-Expert** | Multi-clone pollination | Large context, multi-domain | Tested (basic) |
| **T6: 1M Flock** | Full corpus in single window | 1M+ context, uncensored | **NOT YET TESTED** |
| **T7: 10M Synthesis** | Entire project in single window | 10M context (self-host only) | **NEW — NOT TESTED** |

---

## Test Hypotheses

### Confirmed (H1-H9, from Slots 2-4)

| ID | Hypothesis | Status | Key Finding |
|---|---|---|---|
| H1 | Dense 70B > MoE 3B for depth | **CONFIRMED** | 70B: 30 findings vs 3B: shallower |
| H3 | Abliteration technique matters | **CONFIRMED** | Weight-surgery = total uncensoring; fine-tune = topic-specific |
| H4 | Research framing helps censorship | **PARTIAL** | Helps but doesn't replace abliteration |
| H5 | 235B MoE quality ceiling | **PENDING** | Qwen3-235B was downloading |
| H6 | Flock driver speed vs quality | **CONFIRMED** | ministral-3b: 90% @ 286ms |
| H8 | Cross-expert pollination works | **CONFIRMED** | 16 cross-domain bridges found |
| H9 | Expert disagreement = signal | **CONFIRMED** | Hermes refused tren, Llama engaged |

### New Hypotheses (H10-H18)

#### H10: Full-Corpus 1M Flock (eliminates chunking)
Can a 1M-context model score relevance across the ENTIRE 1.4 MB corpus in one
call? If yes, Flock shifts from iterative 13-step SQL+LLM to single-shot.
**Tested in:** Phase 2

#### H11: Linear Attention Quality at Scale
MiniMax abandoned Lightning attention in M2 due to poor reasoning. Do other
linear attention variants (DeltaNet, Mamba-2, hybrid linear) also degrade?
**Tested in:** Phase 1C (direct comparison)

#### H12: 2M Meta-Expert Synthesis
Grok 4.20 holds ALL results in 2M context. Finds connections no single model could?
**Tested in:** Phase 4

#### H13: Qwen 3.6 Dense vs MoE (DeltaNet)
27B dense (all active) vs 35B MoE (3B active, DeltaNet linear attention).
Does 9× more active params = 9× better findings? Or does DeltaNet close the gap?
**Tested in:** Phase 1B

#### H14: Free Models as Flock Drivers
Can $0 models (Ling, Gemma, Nemotron) match ministral-3b's 90% Flock accuracy?
**Tested in:** Phase 1A

#### H15: Mamba-2 Hybrid vs Pure Transformer for Flock (NEW)
Nemotron 3 Super uses Mamba-2 + MoE + selective Attention. Its multi-token
prediction generates multiple tokens per forward pass. Does this architecture
beat ministral-3b (pure transformer, 3B dense) on Flock speed AND accuracy?
**Tested in:** Phase 1A

#### H16: Self-Hosted 10M Llama 4 Scout (NEW)
No API serves Scout beyond 300K. On 4×H200, can we serve it at 1M+ and run
full-corpus Flock? At 10M, can it hold the ENTIRE research project?
**Tested in:** Phase 3

#### H17: Kimi K2.6 MoE with Muon Optimizer (NEW)
K2.6 has 40B active params with a novel muon optimizer for better expert routing.
Does this translate to better cross-domain reasoning for MiroThinker's workers?
**Tested in:** Phase 1B

#### H18: Provider Censorship Divergence (NEW)
Same model weights show different censorship on different providers (GLM-5:
REFUSED on Zhipu, UNCENSORED on Fireworks/Venice). How many of the 656
untested models are uncensored on alternative providers?
**Tested in:** Phase 0

---

## Employment List v4

### Phase 0: Censorship Pre-Screen (GATE — blocks all other phases)

**Purpose:** Before spending money on Flock/worker tests, determine which new
models are usable. The 4-probe battery costs ~$0 for free models.

**Duration:** ~45 min | **Cost:** ~$1-2 (DashScope models only)

#### 0A: High-Priority Probes (architecture stars)

These models have interesting architectures. If uncensored, they change our plans.

| Model | Provider | Active Params | Architecture | Why Priority |
|---|---|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | OpenRouter | 12B | **Mamba-2 + MoE** | Top Flock candidate if uncensored. 1M self-host. FREE |
| `inclusionai/ling-2.6-1t` | OpenRouter | 63B | **Hybrid linear** | Highest-quality free model. 1M self-host |
| `inclusionai/ling-2.6-flash` | OpenRouter | 7.4B | MoE | Ultra-fast free worker |
| `google/gemma-4-26b-a4b-it` | OpenRouter | 4B | MoE | Google MoE, FREE, potential flock driver |
| `google/gemma-4-31b-it` | OpenRouter | 31B | Dense | Google dense, FREE |
| `nvidia/nemotron-3-nano-30b-a3b` | OpenRouter | 3B | **Mamba + MoE** | Direct ministral-3b competitor. FREE |

**Cost: $0** (all free on OpenRouter)

#### 0B: 1M-Context Model Probes

| Model | Provider | Context | Price | Why |
|---|---|---|---|---|
| `qwen3.6-plus` | DashScope | 1M | $0.33/$1.95 | Cheapest 1M if uncensored |
| `qwen3.6-27b` | DashScope | 262K | TBD | Dense DeltaNet — novel architecture |
| `qwen3.6-35b-a3b` | DashScope | 262K | TBD | MoE DeltaNet — H13 test model |
| `qwen3.6-flash` | DashScope | ? | TBD | Speed variant |
| `qwen3.6-max-preview` | DashScope | ? | TBD | Quality ceiling |
| `xiaomi/mimo-v2.5-pro` | OpenRouter | 1M | $1/$3 | New 1M entrant |
| `xiaomi/mimo-v2.5` | OpenRouter | 1M | $0.40/$2 | Cheaper MiMo |
| `amazon/nova-premier-v1` | OpenRouter | 1M | $2.50/$12.50 | AWS flagship |
| `minimax/minimax-m1` | OpenRouter | 80-128K (!) | $0.40/$2.20 | Lightning attention (cautionary) |

**Cost:** ~$1-2 | **Time:** ~15 min

#### 0C: Cross-Provider Censorship Check (H18)

Test 5 models known to be censored on native provider against alternative providers:

| Model | Native (Censored) | Alternative | Expected |
|---|---|---|---|
| GLM-5 | Zhipu (REFUSED) | Fireworks / Venice | UNCENSORED (confirmed prior) |
| Claude Opus 4.7 | Anthropic (LIMITED) | Venice | Uncensored proxy? |
| GPT-5.4 | OpenAI (LIMITED) | Venice | Uncensored proxy? |
| Kimi K2.6 | Moonshot | Fireworks | Compare censorship |
| Qwen 3.6 Plus | DashScope | OpenRouter | Compare censorship |

**Cost:** ~$2-3 | **Time:** ~15 min

**Phase 0 Gate:** Models that score UNCENSORED or USABLE proceed to Phases 1-4.
Models that score LIMITED or REFUSED are excluded from worker/flock roles but may
serve as orchestrators (like Claude Opus).

---

### Phase 1: Architecture Comparison Tests (API-based, parallel)

**Purpose:** Head-to-head comparisons that test specific architectural hypotheses.
All run the same standardized benchmarks for direct comparison.

**Duration:** ~2 hours (all sub-phases parallel) | **Cost:** ~$5-10

#### Phase 1A: Flock Driver Tournament (H14, H15)

50 relevance judgments on the Milos corpus. Same ground truth as Slot 2E.
**Goal:** Find the best Flock driver considering speed, accuracy, and cost.

| Model | Provider | Active | Architecture | Price | Baseline |
|---|---|---|---|---|---|
| **ministral-3b** | Mistral | 3B | Dense transformer | $0.02/MTok | **Champion: 90%, 286ms** |
| **nemotron-3-super-120b-a12b** | OpenRouter | 12B | **Mamba-2 + MoE + Attn** | FREE | Expected: faster (MTP) |
| **nemotron-3-nano-30b-a3b** | OpenRouter | 3B | **Mamba + MoE** | FREE | Head-to-head vs ministral |
| **gemma-4-26b-a4b-it** | OpenRouter | 4B | MoE | FREE | Google quality at 0 cost |
| **ling-2.6-flash** | OpenRouter | 7.4B | MoE | FREE | InclusionAI speed variant |
| **ling-2.6-1t** | OpenRouter | 63B | Hybrid linear | FREE | Overkill for flock? Or best? |
| **qwen3.6-35b-a3b** | DashScope | 3B | **DeltaNet MoE** | TBD | Novel linear attention |
| **llama-4-scout** | Groq | 17B | MoE (10M native) | $0.11/MTok | 90% at 131K context |

**What this tests:**
- H15: Mamba-2 (Nemotron Super) vs transformer (ministral) — does the architecture
  advantage translate to better flock performance?
- H14: Can any FREE model match 90% accuracy?
- H11: Does DeltaNet linear attention (Qwen 3.6) hurt flock accuracy (MiniMax warning)?

**Metrics:** Accuracy, latency (ms), throughput (tok/s), cost per judgment.
**Cost: ~$0.50** (mostly free models + one DashScope + one Groq)

#### Phase 1B: Worker Quality Tournament (H13, H17)

Topic A1 (Milos Insulin baseline). Same prompt, same corpus. Count findings,
measure specificity, check for censorship.

| Model | Provider | Active | Architecture | Why Test |
|---|---|---|---|---|
| **kimi-k2.6** | Moonshot | 40B | MoE + muon optimizer | H17: muon = better routing? |
| **ling-2.6-1t** | OpenRouter | 63B | Hybrid linear | Highest free active params |
| **qwen3.6-27b** | DashScope | 27B | Dense + DeltaNet | H13: dense baseline |
| **qwen3.6-35b-a3b** | DashScope | 3B | MoE + DeltaNet | H13: MoE comparison |
| **nemotron-3-super-120b-a12b** | OpenRouter | 12B | Mamba-2 + MoE | Can Mamba reason as worker? |
| **gemma-4-31b-it** | OpenRouter | 31B | Dense | Google dense baseline |
| **deepseek-chat** | DeepSeek | ~37B | MoE | Proven baseline (31 findings) |
| **gpt-4.1-mini** | OpenAI | ? | Dense | Current finding champion (80) |

**Comparison baselines from prior slots:**
- Llama-70B-abliterated (Slot 2): 30 findings
- Hermes-3-70B (Slot 3): 72 findings (insulin only)
- Qwen3-30B-A3B (Slot 4): 40 findings
- GPT-4.1-mini (API track): 80 findings

**Cost:** ~$3-5 | **Time:** ~1 hour

#### Phase 1C: Linear Attention Stress Test (H11 — MiniMax Warning)

The critical test. MiniMax abandoned Lightning attention because it hurt
reasoning on complex multi-turn tasks. Do other linear variants also degrade?

**Test design:** Same 50 relevance judgments, but with increasingly complex
reasoning chains (simple fact match → multi-hop inference → cross-domain bridge).
Score each model on simple vs complex questions separately.

| Model | Attention Type | Active | Provider | Why |
|---|---|---|---|---|
| **qwen3.6-35b-a3b** | DeltaNet (gated linear) | 3B | DashScope | Novel linear variant |
| **nemotron-3-super** | Mamba-2 (state-space) | 12B | OpenRouter | Best-evidenced alternative |
| **ling-2.6-1t** | Hybrid linear+standard | 63B | OpenRouter | 1T-scale hybrid |
| **minimax-m1** | Lightning (sparse) | ? | OpenRouter | The cautionary model itself |
| **ministral-3b** | Standard transformer | 3B | Mistral | Pure transformer control |
| **gpt-4.1-nano** | Standard transformer | ? | OpenAI | Closed-model control |

**What this reveals:**
- If DeltaNet/Mamba-2 match transformer on complex reasoning → safe to use
- If they degrade on complex but match on simple → use only for simple flock
- If they all degrade like M1 → linear attention is unsuitable for MiroThinker

**Cost:** ~$1-2 | **Time:** ~30 min

---

### Phase 2: 1M Full-Corpus Flock Battery (H10 — User's Primary Interest)

**Purpose:** Load the ENTIRE 1.4 MB corpus into a single API call. Run 50
relevance judgments. This is the architecturally transformative test.

**Duration:** ~3 hours | **Cost:** ~$25-35

**Prerequisite:** Phase 0 (censorship screen) + Phase 1A (flock baselines)

#### 2A: 1M Flock Battery

| Model | Provider | Context | Price (in/out) | Architecture | Censorship |
|---|---|---|---|---|---|
| **gemini-3.1-pro-preview** | Google | 1M | $2/$12 | Dense transformer | UNCENSORED |
| **gemini-3-flash-preview** | Google | 1M | $0.50/$3 | Dense transformer | UNCENSORED |
| **gemini-3.1-flash-lite** | Google | 1M | $0.25/$1.50 | Dense transformer | UNCENSORED |
| **grok-4.1-fast** | xAI | 2M | $0.20/$0.50 | Dense transformer | UNCENSORED |
| **gpt-4.1-mini** | OpenAI | 1M | $0.40/$1.60 | Dense transformer | UNCENSORED |
| **gpt-4.1-nano** | OpenAI | 1M | $0.10/$0.40 | Dense transformer | UNCENSORED |
| **qwen3.6-plus** | DashScope | 1M | $0.33/$1.95 | MoE (DeltaNet?) | TBD (Phase 0) |
| **llama-4-maverick** | OpenRouter | 1M | $0.15/$0.60 | MoE 400B/17B | USABLE |

**Test design:**
1. System prompt: full 1.4 MB corpus
2. User prompt: 50 relevance judgment questions (same ground truth as Slot 2E)
3. Measure: accuracy, latency, cost per judgment, explanation quality

**Key question:** Does ANY 1M model beat ministral-3b's **90% accuracy** without
chunking? If yes, the Flock architecture shifts from iterative SQL+LLM to
single-shot evaluation.

**Cost per model:** ~$2-5 (1.4M tokens × 50 calls). **Total: ~$20-30.**

#### 2B: Context Scaling Degradation (H11 at 1M scale)

Same 50 judgments at increasing corpus sizes: 100K, 250K, 500K, 750K, 1M tokens.
Reveals the "effective context" — where accuracy starts dropping.

| Model | Max Context | Architecture | Why |
|---|---|---|---|
| gemini-3.1-pro-preview | 1M | Transformer | Gold standard |
| grok-4.1-fast | 2M | Transformer | Cheapest transformer at scale |
| gpt-4.1 | 1M | Transformer | GPT baseline |
| qwen3.6-plus | 1M | MoE (DeltaNet?) | Linear attention at 1M |

**Cost:** ~$10-15 | **Time:** ~2 hours

---

### Phase 3: Self-Hosted Extended Context (H16 — H200 Track)

**Purpose:** API providers cap context well below native limits. Self-hosting
on 4×H200 (575 GB VRAM) unlocks capabilities no API provides.

**Duration:** ~4-6 hours (model downloads + serving + tests)
**Cost:** 6h × $9.42/hr = ~$57 (H200 rental only, no API cost)

#### 3A: Llama 4 Scout at Extended Context (H16)

No API serves Scout beyond 300K. Self-hosting is the only path.

| Test | Context | What We Learn |
|---|---|---|
| 3A-1 | 300K (Together parity) | Baseline — does self-hosted match API quality? |
| 3A-2 | 500K | Does quality hold beyond any provider's cap? |
| 3A-3 | 1M | Full-corpus Flock in single call with open model |
| 3A-4 | 2M (if VRAM allows) | Stress test — can Scout handle 2× corpus? |

**Model:** `meta-llama/Llama-4-Scout-17B-16E-Instruct`
**vLLM config:** TP=2 (only 17B active, ~40 GB FP8), `--max-model-len` scaled per test
**VRAM estimate:** ~60 GB weights + KV cache scales with context

**Metric:** 50 Flock judgments at each context size. Accuracy curve.

#### 3B: Nemotron 3 Super at 1M (Mamba-2 at Full Context)

OpenRouter caps at 262K. Self-hosting unlocks 1M.

**Model:** `nvidia/Nemotron-3-Super-120B-A12B-Instruct`
**vLLM config:** TP=2, `--max-model-len 1048576`
**VRAM estimate:** ~65 GB weights (INT8) + KV cache (Mamba-2 = efficient)

| Test | What |
|---|---|
| 3B-1 | 50 Flock judgments at 1M context (full corpus) |
| 3B-2 | T4 Clone Context accumulation over 10 waves (does Mamba maintain quality?) |
| 3B-3 | Head-to-head vs Phase 2A results at same 1M corpus |

**This tests H15 at 1M scale:** Does Mamba-2's linear-time advantage translate
to better-than-transformer Flock performance at full corpus size?

#### 3C: Qwen3-235B-A22B (Completing Slot 5)

**Status from v3:** Download was at 276 GB / ~470 GB, disk at 82%.
**Model:** `huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-abliterated`
**vLLM config:** TP=4, `--max-model-len 8192`
**VRAM:** ~125 GB (INT4), needs 4 GPUs

| Test | Tier | Hypothesis | Topic | What We Measure |
|---|---|---|---|---|
| 3C-1 | T1 | H5 (quality ceiling) | A1 | Findings depth — 22B active vs 70B dense |
| 3C-2 | T5 | H5 (orchestrator) | A1 | Angle detection, contradiction finding |
| 3C-3 | T5 | H8 (meta-expert) | A1+B4 | Cross-domain synthesis |

#### 3D: Model Rotation Schedule (H200 Timeline)

Only one model can serve at a time (VRAM constraint). Sequence:

```
Hour 0-1:     Download Llama 4 Scout (~60 GB)
Hour 1-3:     Serve Scout → Tests 3A-1 through 3A-3 (300K, 500K, 1M)
Hour 3-3.5:   Stop Scout → download Nemotron 3 Super (~65 GB)
Hour 3.5-5:   Serve Nemotron → Tests 3B-1 through 3B-3 (1M Flock, Clone, comparison)
Hour 5-5.5:   Stop Nemotron → resume Qwen3-235B download (if not complete)
Hour 5.5-7:   Serve Qwen3-235B → Tests 3C-1 through 3C-3 (quality ceiling)
```

**Alternative (if disk space allows parallel downloads):** Pre-download all 3,
then rotate serving only.

---

### Phase 4: Meta-Expert Synthesis (H12)

**Purpose:** Single call with ALL results concatenated. Can a 2M model find
connections that no single-slot model could?

**Duration:** ~1 hour | **Cost:** ~$5

**Prerequisite:** Phases 1-3 complete (need results to concatenate)

#### 4A: Grok 4.20 at 2M

| Input | Size |
|---|---|
| All Slot 2-5 worker conversations | ~400K tokens |
| Medical textbooks | ~160K tokens |
| YouTube transcripts | ~85K tokens |
| Phase 1B worker findings (all models) | ~100K tokens |
| Phase 2A 1M Flock results | ~50K tokens |
| **Total** | **~800K tokens** (well within 2M) |

**Model:** `grok-4.20` via xAI API ($2/$6 per MTok)
**Prompt:** "You hold every finding from every model and every source. Find
cross-domain connections that no single model identified. Focus on bridges
between insulin pharmacology, hematology, and environmental toxicology."

**Cost:** ~$1.60 input + ~$1.00 output = **~$3**

#### 4B: Comparison Meta-Expert (optional)

Run the same concatenated input through Claude Opus 4.7 (1M, LIMITED but
FULL meta-reasoning) and Gemini 3.1 Pro (1M, UNCENSORED).

| Model | Context | Cost | Why |
|---|---|---|---|
| Claude Opus 4.7 | 1M | ~$4 input + $2.50 output | Best meta-reasoner (refuses direct but full meta) |
| Gemini 3.1 Pro | 1M | ~$1.60 + $1.20 | UNCENSORED + SOTA reasoning |

**Purpose:** Do different meta-experts find different cross-domain bridges?

---

## Execution Timeline

```
PHASE 0: Censorship Pre-Screen (GATE)
  Hour 0-0.75:   0A (6 free architecture stars) + 0B (9 new 1M models) + 0C (5 cross-provider)
  → Decision: which models proceed to Phases 1-4

PHASE 1: Architecture Comparisons (parallel, API-based)
  Hour 1-2:      1A (Flock tournament, 8 models × 50 judgments)
  Hour 1-2:      1B (Worker tournament, 8 models × A1 topic)
  Hour 1-1.5:    1C (Linear attention stress test, 6 models)
  → Results inform Phase 2 model selection and Phase 3 priorities

PHASE 2: 1M Flock Battery (API-based, after Phase 1)
  Hour 2-5:      2A (8 models × 50 judgments at full 1.4 MB corpus)
  Hour 5-7:      2B (4 models × 5 context sizes × 50 judgments)
  → Answers H10: does 1M single-shot beat chunked Flock?

PHASE 3: Self-Hosted H200 (parallel with Phase 2)
  Hour 1-3:      3A (Llama 4 Scout at 300K/500K/1M)
  Hour 3-5:      3B (Nemotron 3 Super at 1M — Mamba-2 Flock)
  Hour 5-7:      3C (Qwen3-235B — quality ceiling, Slot 5 completion)
  → Answers H16: does self-hosted 10M-native model outperform API 1M?

PHASE 4: Meta-Expert Synthesis (after Phases 1-3)
  Hour 7-8:      4A (Grok 4.20 at 2M) + 4B (Claude Opus + Gemini comparison)
  → Answers H12: cross-domain connections from entire project in one context
```

**Total wall clock:** ~8 hours (Phases 1+2+3 run in parallel)

---

## Cost Summary

| Phase | What | API Cost | H200 Cost | Total |
|---|---|---|---|---|
| 0 | Censorship pre-screen | ~$2 | — | ~$2 |
| 1A | Flock tournament | ~$0.50 | — | ~$0.50 |
| 1B | Worker tournament | ~$4 | — | ~$4 |
| 1C | Linear attention stress | ~$1.50 | — | ~$1.50 |
| 2A | 1M Flock battery | ~$25 | — | ~$25 |
| 2B | Context degradation | ~$12 | — | ~$12 |
| 3 | Self-hosted H200 tests | — | ~$57 | ~$57 |
| 4 | Meta-expert synthesis | ~$10 | — | ~$10 |
| **TOTAL** | | **~$55** | **~$57** | **~$112** |

---

## Completed Results (Slots 2-4 + API Track)

### Slot 2: Llama-3.3-70B-abliterated — COMPLETE

| Test | Tier | Result |
|---|---|---|
| 2A (T1, A1) | Worker | 30 findings, UNCENSORED, dosages included |
| 2B (T1, B4/F9) | Worker | 15+15 findings, cross-domain connections |
| 2E (T3) | Flock | 84% accuracy, 210ms avg, 10.6 tok/s |
| 2F (T4) | Clone Context | 5 waves, 4K→22K tokens, 100% accuracy |
| 2G (T5) | Cross-Expert | 16 cross-domain bridges (insulin↔hematology) |

### Slot 3: Hermes-3-Llama-3.1-70B — COMPLETE

| Test | Tier | Result |
|---|---|---|
| 3A (T1, A1) | Worker | **72 findings** — UNCENSORED on insulin |
| 3B (T1, B1) | Worker | **REFUSED** on trenbolone |
| 3C (T4) | Clone Context | 3 waves, 3.7K→11K tokens |

### Slot 4: Qwen3-30B-A3B — COMPLETE

| Test | Tier | Result |
|---|---|---|
| 4A (T1, A1) | Worker | 40 findings — UNCENSORED on insulin |
| 4B (T1, D8/B3) | Worker | 20+10 findings, tren flagged but not refused |

### API Track — COMPLETE (6/7)

| Model | Provider | A1 Findings | Notes |
|---|---|---|---|
| DeepSeek V3.2 | DeepSeek | 31 | UNCENSORED |
| Gemini Flash | Google | 2/15/31 | Variable format |
| Grok 3-fast | xAI | 31/10/31 | UNCENSORED |
| Mistral Large | Mistral | 9/11/13 | UNCENSORED |
| GPT-4.1-mini | OpenAI | **80** | Highest count |
| Claude Sonnet 4 | OpenRouter | 1 | Minimal engagement |

### Flock Speed Benchmark — COMPLETE

| Model | Provider | Accuracy | Latency |
|---|---|---|---|
| ministral-3b | Mistral | **90%** | **286ms** |
| llama-4-scout | Groq | 90% | 414ms |
| gpt-4.1-nano | OpenAI | 85% | 739ms |
| qwen3-32b | Groq | 30% | FAILED |

---

## Success Criteria

### Must-Pass
- [ ] Phase 0: ≥ 4/6 architecture-star models pass censorship screen
- [ ] Phase 1A: At least 1 free model achieves ≥ 85% Flock accuracy
- [ ] Phase 2A: ≥ 6/8 1M models complete Flock battery without errors
- [ ] Phase 3C: Qwen3-235B produces findings on A1 (H5)

### Should-Pass
- [ ] H10: At least 1 model achieves ≥ 85% accuracy on full-corpus 1M Flock
- [ ] H11: Linear attention models score within 5% of transformer on complex reasoning
- [ ] H12: Grok 2M finds ≥ 5 cross-domain connections not in any single slot
- [ ] H14: At least 1 free model matches ministral-3b (90%)
- [ ] H15: Nemotron Super beats ministral-3b on speed while maintaining ≥ 85% accuracy
- [ ] H16: Self-hosted Scout at 1M matches API Gemini quality on Flock

### Nice-to-Have
- [ ] H13: Measurable quality difference between Qwen 3.6 27B and 35B-A3B
- [ ] H17: Kimi K2.6 produces more cross-domain findings than DeepSeek V3.2
- [ ] H18: ≥ 3 models censored on native provider are uncensored on alternatives
- [ ] Scout at 2M context produces coherent Flock results (no quality collapse)

---

## Decision Points (For Your Review)

1. **Run all phases or subset?**
   - **Minimum:** Phase 0 + 1A + 2A (censorship + flock tournament + 1M battery) = ~$28, ~5h
   - **Recommended:** Phases 0-2 + 3A-3B (all API tests + Scout/Nemotron self-host) = ~$100, ~7h
   - **Full:** All phases including Slot 5 completion and meta-expert = ~$112, ~8h

2. **H200 model priority?**
   - Scout first (H16 — 10M potential is the biggest unknown)
   - Nemotron first (H15 — Mamba-2 at 1M is more immediately useful)
   - Qwen3-235B first (H5 — completes interrupted Slot 5)

3. **How many models in 1M Flock battery?**
   - All 8 listed (~$30) or top 4 from Phase 1A flock results (~$15)?

4. **Phase 4 meta-expert: Grok only or add Claude+Gemini comparison?**
   - Grok only: ~$3
   - All three: ~$10

5. **Start Phase 0 + Phase 1 immediately?**
   - Phase 0 costs ~$2 and answers the most critical question (which models are even usable)
   - Phase 1A costs ~$0.50 (mostly free models)
   - Recommendation: YES — these are cheap gates that inform everything else
