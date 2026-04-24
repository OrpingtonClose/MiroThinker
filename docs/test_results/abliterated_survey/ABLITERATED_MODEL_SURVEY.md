# Comprehensive Abliterated/Uncensored LLM Survey

**Source:** External LLM research survey (April 2026)
**Scope:** 280+ models across 20+ families, 30+ providers
**Reference deployment:** 4×NVIDIA H200 SXM (575 GB VRAM)

---

## Executive Summary

The abliterated model ecosystem on HuggingFace has industrialized. What began as FailSpy's single `abliterator` library (May 2024) is now a fully automated supply chain producing uncensored variants of every major foundation model within hours of release.

**Key numbers:**
- 500+ uncensored model checkpoints on HuggingFace
- 4,967+ results for "abliterated", 2,164+ for "heretic"
- huihui-ai alone maintains 80+ abliterated models
- DavidAU hosts 200+ models including GGUF variants
- Heretic tool has spawned 1,300+ community-created models

---

## Abliteration Techniques (Quality Ranked)

### Tier 1: Best Quality Preservation

| Technique | Creator | KL Divergence | Refusal Rate | Reasoning Impact |
|---|---|---|---|---|
| **Heretic (TPE Bayesian)** | p-e-w | 0.16 (Gemma-3-12B) | 3/100 | Minimal — co-minimizes refusals + KL |
| **MPOA (Magnitude-Preserving)** | grimjim | Low | Low | **Improves** NatInt (21.33 vs 18.72 baseline) |
| **Biprojection (Norm-Preserving)** | TrevorJS | 0.068 (Gemma-4-E4B) | 0.7% | Near-zero degradation |
| **ARA (Heretic variant)** | trohrbaugh | 0.012 (Gemma-4-31B) | 5/100 | Minimal |
| **Abliterix (Bayesian)** | wangzhang | 0.0115 (Qwen3.5-122B) | 0.5% | Near-zero |

### Tier 2: Good Quality

| Technique | Creator | Notes |
|---|---|---|
| **Standard abliteration** | huihui-ai | Automated pipeline, covers all major models. Some reasoning degradation on math (GSM8K) |
| **Lorablation** | mlabonne | LoRA extraction — transfers uncensoring across model generations |
| **Dataset filtering** | Eric Hartford (Dolphin) | Full fine-tune on curated datasets. Most natural outputs |
| **EGA (Expert-Granular)** | TrevorJS/OBLITERATUS | For MoE models — per-expert abliteration. Required for Gemma 4 MoE |

### Tier 3: Aggressive (Use With Caution)

| Technique | Notes |
|---|---|
| Classic orthogonalization | Original method. Can degrade GSM8K by up to 18.81% |
| Nuclear (OBLITERATUS) | All techniques combined. Maximum force, maximum risk |
| Aggressive uncensoring (HauhauCS) | 0/465 refusals but quality impact unclear |

**Key insight:** Modern Heretic/MPOA/Biprojection abliteration has **reversed the reasoning-uncensored trade-off**. grimjim's MPOA actually *improves* reasoning scores, suggesting safety alignment processing occupies computational capacity that can be reclaimed.

---

## Priority Models for 4×H200 (575 GB VRAM)

### Category A: Flock Driver (Fast, Long-Context, Uncensored)

| Model | Repo ID | Total/Active | Context | VRAM (Q4) | VRAM (FP16) | vLLM | Why |
|---|---|---|---|---|---|---|---|
| **Llama 4 Scout Abliterated** | `jiangchengchengNLP/Llama-4-Scout-17B-16E-Instruct-abliterated-v2` | 109B/17B | **10M** | ~65 GB | ~220 GB | Yes | Only 10M model. Entire corpus in one shot |
| **Nemotron 3 Super 120B Heretic** | `mradermacher/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-heretic` | 121B/12B | 1M | ~70 GB | ~242 GB | Yes | Mamba-2 hybrid. Linear-time context. 22K downloads |
| **Qwen3.5-35B-A3B Abliterated** | `huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated` | 35B/3B | 262K | ~14-16 GB | ~70 GB | Yes | Best efficiency/capacity balance. DeltaNet |
| **GLM-4.7-Flash Uncensored** | `DavidAU/GLM-4.7-Flash-Uncensored-Heretic-NEO-CODE` | 30B/2B | 200K | ~14 GB | ~60 GB | Yes | Tiny active params, agentic coding |
| **MiMo-V2-Flash Abliterated** | `huihui-ai/Huihui-MiMo-V2-Flash-BF16-abliterated` | ~309B/~15B | 128K | ~40 GB | — | Yes | Highest AIME (94.1%) |

### Category B: Deep Reasoning Worker (Dense, High Quality)

| Model | Repo ID | Params | Context | VRAM (FP16) | vLLM | Why |
|---|---|---|---|---|---|---|
| **Llama-3.3-70B Abliterated** | `huihui-ai/Llama-3.3-70B-Instruct-abliterated` | 70B | 128K | ~140 GB | Yes | Proven baseline. Consensus general-purpose |
| **Qwen3.5-27B Abliterated** | `huihui-ai/Huihui-Qwen3.5-27B-abliterated` | 27B | 262K | ~54 GB | Yes | 189K downloads. SWE-bench 72.4%, GPQA 85.5% |
| **DeepSeek-R1-Distill-Llama-70B Ablit** | `huihui-ai/DeepSeek-R1-Distill-Llama-70B-abliterated` | 70B | 128K | ~140 GB | Yes | Highest reasoning scores at 70B |
| **Dolphin 3.0 R1 Mistral 24B** | `cognitivecomputations/Dolphin3.0-R1-Mistral-24B` | 24B | 32K | ~48 GB | Yes | Only R1-trained uncensored. 92%+ GSM8K |
| **QwQ-32B Abliterated** | `huihui-ai/QwQ-32B-abliterated` | 32B | 32K | ~64 GB | Yes | MATH-500 90.6%, strong thought chain |
| **Hermes-3-Llama-3.1-70B Lorablated** | `mlabonne/Hermes-3-Llama-3.1-70B-lorablated` | 70B | 128K | ~140 GB | Yes | Best creative quality via LoRA preservation |

### Category C: Maximum Scale (Quality Ceiling)

| Model | Repo ID | Total/Active | Context | VRAM (Q4) | vLLM | Why |
|---|---|---|---|---|---|---|
| **Qwen3-235B-A22B Abliterated** | `huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-abliterated` | 235B/22B | 262K | ~125 GB | Yes | AIME 89.2%, largest abliterated Qwen3 |
| **Qwen3.5-397B-A17B Abliterated** | `huihui-ai/Huihui-Qwen3.5-397B-A17B-abliterated-GGUF` | 397B/17B | 262K | ~200 GB | Yes | Largest abliterated Qwen. Frontier scale |
| **Qwen3.5-122B-A10B Abliterix** | `wangzhang/Qwen3.5-122B-A10B-abliterix` | 122B/10B | 262K | ~65 GB | Yes | KL 0.0115 — best quality preservation ever |
| **DeepSeek-V3 Abliterated** | `huihui-ai/DeepSeek-V3-abliterated` | 671B/37B | 128K | ~227 GB | Yes | Only abliterated 671B. Fits 4×H200 at Q4 |
| **Kimi K2 Abliterated** | `huihui-ai/kimi-k2-abliterated` | ~1T/~32B | 256K | ~40 GB+ | Yes | Largest abliterated model (1.026T params) |
| **Nemotron Ultra 253B Abliterated** | `huihui-ai/Llama-3.1-Nemotron-Ultra-253B-v1-abliterated` | 253B | 128K | ~160 GB | Yes | Largest dense Nemotron. ~30% of 4×H200 |

### Category D: Small/Fast (Edge, Flock Driver, Auxiliary)

| Model | Repo ID | Total/Active | VRAM | vLLM | Why |
|---|---|---|---|---|---|
| **Gemma 4 E4B Uncensored** | `TrevorJS/gemma-4-E4B-it-uncensored` | 4.5B | ~2.6 GB | Yes | KL 0.068. Tiny flock driver candidate |
| **Gemma 4 26B-A4B Uncensored** | `TrevorJS/gemma-4-26B-A4B-it-uncensored` | 25.2B/3.8B | ~15 GB | Yes | First MoE abliteration via EGA |
| **GPT-OSS 20B Abliterated** | `DavidAU/OpenAi-GPT-oss-20b-abliterated` | 21B/3.6B | ~12 GB | Yes | Apache 2.0, mixed tasks |

---

## Key Providers

| Provider | Models | Specialty | Quality |
|---|---|---|---|
| **huihui-ai** | 200+ | Automated abliteration of every major release | Good (standard). v2 iterations fix encoding issues |
| **mlabonne** | 20+ | Layerwise abliteration, lorablation | High — pioneered per-layer approach |
| **DavidAU** | 30+ | Dark Champion MoE, Heretic variants, creative merges | Variable — creative but experimental |
| **p-e-w** | 6+ official + 1,300 community | Heretic tool (TPE Bayesian optimization) | **Highest** — KL 0.16 with 3/100 refusals |
| **grimjim** | 10+ | MPOA (magnitude-preserving orthogonal ablation) | **Highest** — improves reasoning benchmarks |
| **TrevorJS** | 10+ | Biprojected abliteration, EGA for MoE | **Highest** — KL 0.068 on Gemma 4 |
| **nicoboss** | 20+ | LoRA fine-tune uncensored, Reasoner variants | Good — restores reasoning post-uncensoring |
| **Eric Hartford** | 50+ | Dolphin series (dataset filtering) | High — most natural outputs |
| **NousResearch** | 15+ | Hermes series ("aligned to user") | High — strong function calling and RP |
| **TheDrummer** | 10+ | Rocinante, Cydonia, Skyfall, Fallen series | High for creative writing |
| **Sao10K** | 10+ | Stheno, Euryale, Kunou (QLoRA) | High for roleplay |
| **wangzhang** | 5+ | Abliterix (Bayesian optimization) | **Highest** — KL 0.0115 |

---

## vLLM Compatibility Summary

**All abliterated models work identically to base models in vLLM.** Abliteration modifies weight values only — no architecture/tokenizer/attention changes.

| Architecture | vLLM Executor | Required Flags |
|---|---|---|
| Llama (3.1/3.2/3.3/4) | `LlamaForCausalLM` | None |
| Mistral (7B/24B) | `MistralForCausalLM` | `--tokenizer-mode mistral` |
| Mixtral (8×7B, 8×22B) | `MixtralForCausalLM` | `--enable-expert-parallel` |
| Qwen2/2.5/3/3.5 | `Qwen2ForCausalLM` / `Qwen3ForCausalLM` | None (v0.8.4+ for Qwen3) |
| Qwen MoE | `Qwen2MoeForCausalLM` | `--enable-expert-parallel` |
| DeepSeek V3/R1 | `DeepSeekV3ForCausalLM` | `--trust-remote-code --enable-expert-parallel` |
| Gemma (2/3/4) | `GemmaForCausalLM` | None |
| Nemotron | `LlamaForCausalLM` | None |

**Reasoning parser:** For models with `<think>...</think>` tags (DeepSeek R1, Qwen3 thinking):
```bash
--enable-reasoning-parser deepseek_r1 --reasoning-parser-deepseek-r1-stop-string "</think>"
```

---

## VRAM Calculation for 4×H200

**Formula:**
```
Total VRAM = Weights + KV Cache + Activations + Overhead
```

**Quick estimates:**
- Short context: `Weight Size × 1.3`
- Long context (100K+): `Weight Size × 1.8`

**Multi-model deployment examples (all fit in 575 GB):**
- Nemotron 70B Q8_0 (75 GB) + Cydonia 24B Q8_0 (25 GB) + R1-Distill-32B Q8_0 (35 GB) + Magistral 24B Q8_0 (25 GB) = 160 GB total, 415 GB free
- DeepSeek-V3-abliterated Q5_K_M (~270 GB) = single largest abliterated model that fits
- Llama 4 Scout FP16 (~220 GB) = fits on 2×H200, leaves 2 GPUs free

---

## Self-Service Abliteration Tools

For models without pre-abliterated versions:

| Tool | Creator | Ease | Methods | Best For |
|---|---|---|---|---|
| **OBLITERATUS** | Pliny the Liberator | Zero-code (HF Spaces) | 13 methods + EGA | One-click abliteration of any model |
| **Heretic** | p-e-w | CLI | TPE Bayesian | Highest quality (lowest KL divergence) |
| **Gabliteration** | Goekdeniz Gulmez | CLI | Adaptive multi-directional | Novel architectures |
| **remove-refusals-with-transformers** | huihui-ai | Script | Standard abliteration | Quick automated pipeline |
| **llm-abliteration** | grimjim (NousResearch fork) | Notebook | MPOA | Research-grade quality |

---

## Implications for MiroThinker

### Flock Architecture
The survey confirms that vLLM prefix caching with abliterated models is the optimal Flock architecture:
- **Every model in the survey is vLLM-compatible**
- Models like Scout (10M), Nemotron Super (1M Mamba-2), and Qwen3.5-35B-A3B (262K DeltaNet) are purpose-built for the prefix-cached Flock use case
- Multi-model deployment on 4×H200 enables role-specialized ensembles

### Model Selection for Employment Plan v6
Based on survey findings, priority models for testing:

1. **Flock Drivers (fast, uncensored, long-context):**
   - Qwen3.5-35B-A3B Abliterated (3B active, 262K, ~14 GB) — fastest
   - Nemotron 3 Super Heretic (12B active, 1M, ~70 GB) — Mamba-2
   - Llama 4 Scout Abliterated v2 (17B active, 10M, ~65 GB) — longest context
   
2. **Workers (deep reasoning):**
   - Llama-3.3-70B Abliterated (proven baseline)
   - Qwen3.5-27B Abliterated (262K context, GPQA 85.5%)
   - DeepSeek-R1-Distill-70B Abliterated (highest reasoning at 70B)

3. **Clone-Transplant targets (receive bee conversation as prefix):**
   - Llama 4 Scout at 10M (holds everything)
   - Qwen3.5-122B-A10B Abliterix (KL 0.0115 — virtually identical to base)
   - DeepSeek-V3-abliterated at Q4 (671B/37B active — maximum intelligence)

4. **Quality ceiling (meta-expert):**
   - DeepSeek-V3-abliterated Q4_K_M (~227 GB) — fits 4×H200
   - Qwen3.5-397B-A17B at Q4 (~200 GB) — largest abliterated Qwen

### New Models NOT in Previous Plan
The survey revealed several models we hadn't considered:
- **Kimi K2 Abliterated** (1.026T params, ~32B active, 256K) — largest abliterated model ever
- **MiMo-V2-Flash Abliterated** (94.1% AIME — highest reasoning score)
- **Qwen3.5-122B-A10B Abliterix** (KL 0.0115 — best quality preservation)
- **GPT-OSS 120B** (117B/5.1B active — near-proprietary reasoning, Apache 2.0)
- **Dolphin 3.0 R1** (only R1-trained uncensored model — 92%+ GSM8K)
- **Hermes-3-70B Lorablated** (best creative quality via LoRA preservation)
