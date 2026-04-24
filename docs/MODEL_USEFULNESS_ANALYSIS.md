# Model Usefulness Analysis for MiroThinker Roles

Cursory research on architecture, strengths, and practical usefulness of every
interesting model family in the registry. Focus: which models are viable for
which MiroThinker tiers (T1-T6), and what makes each architecturally unique.

**Last updated:** 2026-04-16

---

## The Big Picture: Context Window Reality Check

Native context ≠ API-served context. Providers routinely cap below the model's
native capability. This table shows what you ACTUALLY get:

| Model | Native Context | Groq | Together | OpenRouter | Fireworks | Self-Host (vLLM) |
|---|---|---|---|---|---|---|
| **Llama 4 Scout** | **10M** | 131K | 300K (→1M planned) | ? | ? | **10M possible** (needs 4×H200+) |
| **Llama 4 Maverick** | **10M** | — | 500K | 1M | ? | **10M possible** |
| **Nemotron 3 Super** | **1M** | ? | ? | 262K (free) | ? | **1M** (Mamba-2 = efficient) |
| **Gemini 3.1 Pro** | **1M** | — | — | — | — | N/A (closed) |
| **Grok 4.20** | **2M** | — | — | — | — | N/A (closed) |
| **GPT-4.1** | **1M** | — | — | 1M | — | N/A (closed) |
| **Claude Opus 4.7** | **1M** | — | — | 1M | — | N/A (closed) |
| **MiniMax M1** | **1M** | — | — | 80-128K (!) | — | N/A (closed) |
| **Ling 2.6 1T** | **1M (self-host)** | — | — | 262K (free) | — | **1M** (hybrid linear) |
| **Kimi K2.6** | **256K** | — | — | 256K | 256K | — |
| **Qwen 3.6 Plus** | **1M** | — | — | — | — | DashScope API: 1M |

**Key insight:** For 1M+ Flock batteries, only **Gemini 3.1 Pro**, **Grok 4.20**,
**GPT-4.1**, **Claude Opus 4.7**, and **Qwen 3.6 Plus** serve full 1M via API.
For open models, **self-hosting is mandatory** to unlock full context.

---

## Model Family Deep Dives

### 1. Llama 4 Scout — The 10M Context MoE

| Property | Value |
|---|---|
| Architecture | MoE: 109B total, **17B active** per token, 16 experts |
| Context | **10M native** (trained at 256K, length-generalized to 10M) |
| Attention | Grouped-Query Attention (GQA) with QK-Norm |
| Strengths | Massive context for full-corpus single-shot; low active params = fast |
| Weaknesses | MoE routing latency variance; providers cap well below 10M |
| Censorship | Groq: 90% accuracy on Flock, UNCENSORED (from our probes) |

**MiroThinker Role Suitability:**
- **T6 (10M Flock Battery): ★★★★★** — The only open model that can hold the entire
  corpus + all findings + medical textbooks in a single context window. If self-hosted
  on 4×H200 at even 1M, it transforms the Flock architecture from iterative to single-shot.
- **T1 (Worker): ★★★☆☆** — 17B active is modest for deep reasoning. Good for breadth,
  weaker on depth compared to 70B dense.
- **T3 (Flock Driver): ★★★★☆** — Already proven on Groq at 131K (90% accuracy, 414ms).
  At 10M, could do Flock over the entire corpus without chunking.

**Why it's interesting:** The 10M context isn't just a number — it means you could load
every medical textbook, every YouTube transcript, every prior swarm finding, and every
cross-domain connection into a SINGLE query. The model sees everything simultaneously.
No chunking artifacts, no lost connections between chunks. The question is whether
17B active params can actually reason well over 10M tokens.

**Provider reality:** Groq caps at 131K, Together at 300K. To test beyond 300K,
self-hosting on the H200 is the only path.

---

### 2. Llama 4 Maverick — Scout's Bigger Sibling

| Property | Value |
|---|---|
| Architecture | MoE: 400B total, **17B active** per token (same active as Scout!) |
| Context | **1M** (some sources say 10M, needs verification) |
| Strengths | Higher quality reasoning than Scout at same inference cost |
| Weaknesses | 400B total = larger download, more VRAM for weights |

**MiroThinker Role Suitability:**
- **T1 (Worker): ★★★★☆** — Better reasoning quality than Scout (more experts, better routing)
  while keeping the same 17B active inference cost.
- **T5 (Cross-Expert): ★★★★☆** — Good for meta-expert synthesis with its larger expert pool.
- **T3 (Flock Driver): ★★★☆☆** — Same active params as Scout but slower to serve due to weight size.

**Why it's interesting:** Same inference cost as Scout (17B active) but 400B total params
means more specialized expert knowledge. The trade-off: needs more VRAM to load the full
model. On 4×H200 (575 GB), you could serve it in INT4/INT8.

---

### 3. Nemotron 3 Super 120B-A12B — The Hybrid Architecture Star

| Property | Value |
|---|---|
| Architecture | **Hybrid: Mamba-2 + MoE + Attention** (LatentMoE) |
| Parameters | 120B total, **12B active** per token |
| Context | **1M tokens** (with 32K output) |
| Attention | Selective attention layers + Mamba-2 state-space for linear-time sequences |
| Special | Multi-token prediction (MTP) for 2-7× faster generation |
| Speed | 2.2-7.5× faster than GPT-OSS-120B and Qwen3.5-122B |
| Strengths | Agentic reasoning, multi-agent coordination, long-context |
| Available | OpenRouter (free at 262K), self-host for 1M |

**MiroThinker Role Suitability:**
- **T3 (Flock Driver): ★★★★★** — 1M context + extreme speed (MTP) + 12B active = ideal
  for high-throughput relevance judgments over the full corpus. Could dethrone ministral-3b.
- **T1 (Worker): ★★★★☆** — 12B active is modest but the Mamba-2 layers handle long context
  better than pure transformers. Good for workers that need to process large corpus chunks.
- **T4 (Clone Context): ★★★★★** — Mamba-2's linear-time sequence processing means context
  accumulation doesn't slow down. Perfect for the clone pattern where context grows each wave.
- **T5 (Cross-Expert): ★★★★☆** — Strong agentic reasoning benchmarks suggest good meta-expert quality.

**Why it's VERY interesting:** This is arguably the most architecturally innovative model
in the registry. The Mamba-2 + MoE + Attention hybrid means:
1. **Linear-time on long sequences** (Mamba-2 layers) — no quadratic attention blowup
2. **Sparse efficiency** (MoE) — only 12B params active per token
3. **Selective precision** (Attention layers) — full attention where it matters most
4. **Multi-token prediction** — generates multiple tokens per forward pass = faster

This combination is purpose-built for exactly what MiroThinker needs: fast inference
over very long contexts with good reasoning quality. The 1M context at 12B active
params is the best ratio in the entire registry.

**FREE on OpenRouter at 262K.** Self-hosting unlocks 1M. This should be a top priority test.

---

### 4. Ling 2.6 1T — The 1-Trillion-Parameter Hybrid Linear

| Property | Value |
|---|---|
| Architecture | Hybrid linear attention + standard attention, **1T total params** |
| Active Params | ~63B (MoE, 1T/63B from OpenRouter listing) |
| Context | 262K on OpenRouter, **1M if self-hosted** |
| Attention | Hybrid: linear attention (sub-quadratic) in lower layers, standard in upper |
| Price | **FREE** on OpenRouter |
| Strengths | Massive parameter count, efficient long-context via linear attention |
| Weaknesses | Hybrid linear may underperform on short dense reasoning |

**MiroThinker Role Suitability:**
- **T6 (1M Flock Battery): ★★★★☆** — If self-hosted at 1M, hybrid linear attention
  means it handles the full corpus more efficiently than pure transformers. 63B active
  = strong reasoning quality.
- **T1 (Worker): ★★★★★** — 63B active params (largest in registry except GPT/Claude) + FREE
  = unlimited worker runs at no cost. If UNCENSORED under research framing, this is the
  ideal free worker model.
- **T4 (Clone Context): ★★★★★** — Linear attention = context accumulation has sub-quadratic
  cost. Perfect for the clone pattern where context grows each wave.

**Why it's interesting:** 1T total parameters at $0 cost. The hybrid linear attention means
it doesn't hit the quadratic wall on long contexts. 63B active params means reasoning
quality comparable to GPT-4.1 class models. **Needs censorship probe.**

---

### 5. Kimi K2.6 — The Muon-Optimized Reasoning Giant

| Property | Value |
|---|---|
| Architecture | MoE: **1T total, ~40B active** per token |
| Context | **256K** |
| Optimizer | **Muon** (custom optimizer for MoE sparsity) |
| Attention | MLA (Multi-Latent Attention) + cross-modal |
| Strengths | Cross-domain reasoning, multimodal, graduate-level benchmarks |
| Weaknesses | High memory for multimodal tokens, expert imbalance on niche tasks |
| Price | Competitive on Moonshot/Fireworks |
| Censorship | **UNCENSORED** on our probes (K2.5 confirmed) |

**MiroThinker Role Suitability:**
- **T1 (Worker): ★★★★★** — 40B active params + UNCENSORED + strong reasoning = excellent
  research worker. Already proven uncensored on Kimi K2.5.
- **T5 (Cross-Expert): ★★★★☆** — Strong cross-domain reasoning from muon-optimized training.
- **T3 (Flock Driver): ★★★☆☆** — 40B active is heavy for flock speed benchmarks, but
  quality will be high.

**Why it's interesting:** The muon optimizer is a training innovation that produces better
expert routing in MoE models. This means the 40B active params are better-selected for
each query than standard MoE models. The 60% speed improvement over K2.5 (confirmed in
deep-search-portal exploration) makes it a strong practical choice.

---

### 6. Qwen 3.6 Family — DeltaNet Linear Attention Pioneer

| Variant | Params | Active | Architecture | Context |
|---|---|---|---|---|
| qwen3.6-27b | 27B | **27B (all)** | Dense + DeltaNet | 262K |
| qwen3.6-35b-a3b | 35B | **3B** | MoE + DeltaNet linear | 262K |
| qwen3.6-flash | ? | ? | ? | ? |
| qwen3.6-plus | ? | ? | ? | **1M** |
| qwen3.6-max-preview | ? | ? | ? | ? |

**DeltaNet linear attention:** Uses a fixed-size recurrent state instead of growing KV
cache. This means:
- **Constant memory** regardless of context length
- **Linear compute** (not quadratic) for long sequences
- **40% KV cache reduction** while matching dense performance

**MiroThinker Role Suitability:**
- **T1 (Worker, 27B dense): ★★★★☆** — All 27B params active = good reasoning depth.
  Dense model means no expert routing overhead. DeltaNet handles long corpus chunks well.
- **T3 (Flock Driver, 35B-A3B MoE): ★★★★★** — Only 3B active = extremely fast + DeltaNet
  linear attention = efficient on long contexts. Could be the new flock champion if quality
  holds at 3B active.
- **T6 (1M Flock, Plus): ★★★★☆** — 1M context on DashScope API. DeltaNet = efficient at
  1M. Needs probe.

**Why it's interesting:** The H13 hypothesis (dense 27B vs MoE 35B-A3B) tests whether 9×
more active parameters produces proportionally better findings. DeltaNet is also the most
novel attention mechanism here — if it works well for Flock relevance scoring, it could
change the architecture recommendation.

---

### 7. MiniMax M1 — Lightning Attention (Cautionary Tale)

| Property | Value |
|---|---|
| Architecture | Dense + **Lightning attention** |
| Context | **1M native**, but API caps at **80-128K** |
| Strengths | Long document analysis, math, coding, tool usage |
| Weaknesses | Slower on heavy prompts, "rigid" per user reports |
| M2 successor | **Abandoned Lightning attention** → switched to standard MoE (204K) |

**MiroThinker Role Suitability:**
- **T3 (Flock Driver): ★★☆☆☆** — Lightning attention was abandoned in M2 due to poor
  reasoning/multi-turn accuracy. If M2 dropped it, the quality concerns are real.
- **T1 (Worker): ★★★☆☆** — OK for long documents but "rigid" and slower than alternatives.

**Why it's a cautionary tale:** MiniMax tried Lightning attention for 1M context, found
it hurt reasoning quality on complex tasks, and reverted to standard MoE in M2. This
is a warning signal for all linear attention variants — long context efficiency may come
at the cost of reasoning precision. **Test Qwen 3.6 DeltaNet and Ling hybrid linear
carefully before relying on them.**

---

### 8. Gemma 4 — Google's Efficient Open Models

| Variant | Params | Active | Architecture | Context |
|---|---|---|---|---|
| gemma-4-26b-a4b-it | 26B | **4B** | MoE | 262K |
| gemma-4-31b-it | 31B | **31B** | Dense | 262K |

**MiroThinker Role Suitability:**
- **T3 (Flock Driver, 26B-A4B): ★★★★☆** — 4B active = very fast. 85 tok/s on consumer
  hardware. FREE on OpenRouter. Good flock candidate if quality holds.
- **T1 (Worker, 31B dense): ★★★☆☆** — 31B dense is decent but smaller than our 70B
  abliterated workers. Better for quick passes than deep reasoning.

**Why it's interesting:** FREE on OpenRouter. The 26B-A4B MoE is a Google-trained model
with only 4B active — if it can match ministral-3b's 90% Flock accuracy, it could be
a zero-cost Flock driver with better reasoning quality.

---

### 9. Nemotron 3 Nano 30B-A3B — NVIDIA's Speed Specialist

| Property | Value |
|---|---|
| Architecture | Hybrid Mamba-Attention MoE |
| Parameters | 30B total, **3B active** |
| Context | ~128-256K |
| Price | **FREE** on OpenRouter |

**MiroThinker Role Suitability:**
- **T3 (Flock Driver): ★★★★☆** — 3B active + Mamba efficiency = very fast. Same active
  params as ministral-3b but with Mamba hybrid architecture.
- **T1 (Worker): ★★☆☆☆** — Too small for deep research reasoning.

**Why it's interesting:** Direct competitor to ministral-3b for the Flock driver role, but
with Mamba architecture for better long-context handling. FREE. Worth a head-to-head test.

---

### 10. GLM-5 — DeepSeek Sparse on Huawei Chips

| Property | Value |
|---|---|
| Architecture | MoE: **744B total, 40B active** per token |
| Attention | MLA + DeepSeek Sparse Attention |
| Context | 128K |
| Training | 28.5T tokens on **Huawei chips** (no NVIDIA) |
| Censorship | **REFUSED on native Zhipu**, UNCENSORED on Fireworks/Venice |

**MiroThinker Role Suitability:**
- **T1 (Worker via Fireworks/Venice): ★★★★☆** — 40B active + uncensored on alternative
  providers = strong worker. The DeepSeek Sparse Attention is efficient.
- **T5 (Cross-Expert): ★★★☆☆** — Good reasoning but the Zhipu censorship means you must
  use third-party providers.

**Why it's interesting:** 744B total params is the largest open MoE besides DeepSeek V3.
But the censorship pattern is the most instructive: same weights, REFUSED on native Zhipu,
UNCENSORED on Fireworks. Provider-level censorship wrapping is real.

---

## Role-Optimized Recommendations (Updated)

### T1 Worker (Deep Reasoning)
| Priority | Model | Why |
|---|---|---|
| 1 | **Kimi K2.6** (40B active, UNCENSORED) | Highest active params + proven uncensored |
| 2 | **Ling 2.6 1T** (63B active, FREE) | Massive reasoning power at zero cost — needs probe |
| 3 | **Llama-3.3-70B-abliterated** (70B dense, local) | Proven, guaranteed uncensored |
| 4 | **DeepSeek V3.2** (37B active, UNCENSORED) | Proven in prior swarm runs |
| 5 | **GLM-5 via Fireworks** (40B active, UNCENSORED) | Strong but needs provider routing |

### T3 Flock Driver (Speed + Relevance Accuracy)
| Priority | Model | Why |
|---|---|---|
| 1 | **ministral-3b** (3B dense, 90%, 286ms) | Proven champion |
| 2 | **Nemotron 3 Super** (12B active, 1M, FREE) | Mamba-2 + MTP = fastest architecture |
| 3 | **Qwen 3.6 35B-A3B** (3B active, DeltaNet) | Linear attention + tiny active = fast+efficient |
| 4 | **Gemma 4 26B-A4B** (4B active, FREE) | Google quality at zero cost |
| 5 | **Nemotron 3 Nano** (3B active, FREE) | Mamba + same active as ministral-3b |
| 6 | **Llama 4 Scout** (17B active, Groq: 131K) | Proven 90% accuracy on Groq |

### T4 Clone Context (Context Accumulation)
| Priority | Model | Why |
|---|---|---|
| 1 | **Nemotron 3 Super** (Mamba-2, 1M) | Linear-time sequence = no slowdown as context grows |
| 2 | **Ling 2.6 1T** (hybrid linear, 1M self-host) | Sub-quadratic context accumulation |
| 3 | **Qwen 3.6 27B** (DeltaNet, 262K) | Fixed-size recurrent state = constant memory |
| 4 | **Llama-3.3-70B** (local, vLLM prefix cache) | Proven in Slot 2F |

### T5 Cross-Expert / Meta-Expert
| Priority | Model | Why |
|---|---|---|
| 1 | **Claude Opus 4.7** (1M, apex reasoning) | Best meta-reasoning, orchestrator-grade |
| 2 | **Grok 4.20** (2M reasoning, UNCENSORED) | 2M context holds all results |
| 3 | **Gemini 3.1 Pro** (1M, UNCENSORED) | SOTA reasoning + full context |
| 4 | **Kimi K2.6** (40B active, muon-optimized) | Strong cross-domain reasoning |

### T6 Full-Corpus Single-Shot (1M+ Flock Battery)
| Priority | Model | Why |
|---|---|---|
| 1 | **Gemini 3.1 Pro** (1M API, UNCENSORED) | Best quality at 1M via API |
| 2 | **Grok 4.1 Fast** (2M API, $0.20/M, UNCENSORED) | Cheapest 1M+ API path |
| 3 | **Llama 4 Scout** (10M self-host) | If quality holds at 10M, game-changer |
| 4 | **Nemotron 3 Super** (1M self-host, FREE) | Mamba-2 = efficient at 1M |
| 5 | **GPT-4.1** (1M API, UNCENSORED) | Proven quality, expensive |
| 6 | **Qwen 3.6 Plus** (1M DashScope, UNTESTED) | Cheapest 1M if uncensored |

---

## Key Architectural Insights

### Linear Attention Variants — Promise vs Reality

Four models use novel attention mechanisms. The MiniMax M1 cautionary tale is critical:

| Model | Attention Type | Claimed Benefit | Real-World Evidence |
|---|---|---|---|
| Qwen 3.6 | DeltaNet (gated linear) | 40% KV cache reduction | Benchmarks good, needs real-world test |
| Ling 2.6 | Hybrid linear+standard | 50-70% memory reduction | Strong but may underperform on short dense reasoning |
| MiniMax M1 | Lightning (sparse) | 1M at 2-3× speed | **ABANDONED in M2** — poor reasoning/multi-turn accuracy |
| Nemotron 3 | Mamba-2 (state-space) | Linear-time sequences | NVIDIA reports 2-7× speed, strong benchmarks |

**Verdict:** Mamba-2 (Nemotron) has the strongest evidence. DeltaNet (Qwen) looks
promising but unproven at scale. Hybrid linear (Ling) is sound architecture but needs
testing. Lightning (MiniMax) was tried and failed.

### MoE Efficiency Comparison

| Model | Total/Active | Ratio | Inference Cost Relative |
|---|---|---|---|
| Llama 4 Scout | 109B/17B | 6.4× | ★★★★ (fast) |
| Llama 4 Maverick | 400B/17B | 23.5× | ★★★★ (same speed as Scout!) |
| Kimi K2.6 | 1T/40B | 25× | ★★★ (moderate) |
| GLM-5 | 744B/40B | 18.6× | ★★★ (moderate) |
| Nemotron Super | 120B/12B | 10× | ★★★★★ (fast + Mamba) |
| Nemotron Nano | 30B/3B | 10× | ★★★★★ (fastest) |
| Qwen 3.6 MoE | 35B/3B | 11.7× | ★★★★★ (fastest) |
| Gemma 4 MoE | 26B/4B | 6.5× | ★★★★ (fast) |
| Ling 2.6 1T | 1T/63B | 15.9× | ★★★ (moderate, but FREE) |

**Best efficiency:** Nemotron Nano and Qwen 3.6 MoE (3B active). Best quality per
inference dollar: Ling 2.6 1T (63B active, FREE) and Kimi K2.6 (40B active, competitive).

### The Self-Hosting Question

For MiroThinker's H200 fleet, these models can unlock capabilities no API provides:

| Model | Self-Host VRAM (FP8) | Context Unlocked | Why Self-Host |
|---|---|---|---|
| Llama 4 Scout | ~60 GB (17B active) | **10M** | No API serves >300K |
| Nemotron 3 Super | ~65 GB | **1M** | OpenRouter caps at 262K |
| Ling 2.6 1T | ~500 GB (needs 4×H200) | **1M** | OpenRouter caps at 262K |
| Llama 4 Maverick | ~210 GB (needs 2×H200) | **1M+** | Together caps at 500K |

**On 4×H200 (575 GB VRAM):** You could serve Llama 4 Scout at full 10M context, or
Ling 2.6 1T at 1M, or multiple smaller models in parallel.

---

## Priority Tests (Updated)

Based on this research, the highest-value tests to run:

1. **Censorship probe: Ling 2.6 1T, Nemotron 3 Super, Gemma 4, Qwen 3.6 family**
   — These are FREE/cheap models with strong architectures. If uncensored, they
   transform the economics of MiroThinker.

2. **Flock benchmark: Nemotron 3 Super vs ministral-3b**
   — Mamba-2 architecture + 12B active vs 3B dense. If Super matches accuracy at
   higher speed, it's the new Flock champion with 1M context bonus.

3. **1M Flock battery: Gemini 3.1 Pro, Grok 4.1 Fast, Llama 4 Scout (self-host)**
   — The T6 tier test. Can full-corpus single-shot beat iterative chunked Flock?

4. **DeltaNet vs Standard: Qwen 3.6 35B-A3B vs ministral-3b on 50 judgments**
   — Tests whether linear attention helps or hurts relevance scoring.

5. **10M stress test: Llama 4 Scout on H200 with full corpus**
   — The ultimate test. Self-host Scout, load everything, ask it to find
   cross-domain connections. If it works, the entire Flock architecture changes.
