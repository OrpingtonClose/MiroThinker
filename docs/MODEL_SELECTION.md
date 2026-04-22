# Model Selection: Uncensored SOTA for Tool-Free Swarm Workers

> May 2026 snapshot. Models and quantizations evolve fast — re-evaluate
> monthly against the hard filters below.

## Hard Filters

Every model considered for MiroThinker must pass **all three**:

1. **Uncensored.** The research domains (PED protocols, hematological
   manipulation, insulin timing, off-label pharmacology) trigger refusals
   on aligned models. A worker that refuses to analyze trenbolone
   hematocrit data is useless regardless of capability. Acceptable paths:
   open weights with no safety filter, abliterated variants (HauhauCS,
   huihui_ai), or models where the base weights are permissive.

2. **Locally runnable on 8×H200 (640 GB VRAM).** No API-only models.
   The 24-hour run must not depend on external services, rate limits, or
   billing surprises.

3. **Quantized to fit the allocation.** Raw BF16 weights for 1T-param
   models don't fit. The model must have proven GGUF/AWQ quantizations
   that preserve quality at the target bit width.

---

## Architecture Context: Why Tool-Free Workers Change Model Selection

Workers in the MiroThinker swarm are **tool-free** (see
`SWARM_WAVE_ARCHITECTURE.md` §Data Package). They receive a structured
data package assembled by the orchestrator and produce analytical output.
They do not call `search_corpus`, `store_finding`, or any other tool.

This means:

- **Tool-calling ability is irrelevant.** Agentic benchmarks (Toolathon,
  SWE-Bench) don't predict worker quality.
- **Analytical reasoning over long input is everything.** The worker
  reads a 10-50K token data package and must find non-obvious
  connections, contradictions, and implications.
- **Context window size directly determines data package richness.** A
  32K-context worker sees a curated 6K-char slice. A 200K-context worker
  sees the full finding set. A 1M-context worker sees everything
  including raw evidence chains.
- **Structured output matters.** Workers must produce typed findings
  (fact + confidence + relationship type) that the cataloguer can ingest.
- **Speed matters.** 5-7 workers run in parallel per wave, 30-60 runs
  in 24 hours. Slow inference means fewer waves means shallower analysis.

---

## Candidate Models (May 2026)

### Tier 1: Recommended for Production

#### GLM-5.1 (744B total / 40B active) — **Primary worker model**

| Property | Value |
|----------|-------|
| Uncensored | Yes — open source, no safety filter on GGUF quants |
| Architecture | MoE (40B active) |
| Context | 200K+ (UD-Q2_K_XL) |
| Key benchmarks | SWE-Verified 77.8%, AIME 92.7%, GPQA Diamond 86.0% |
| Coding score | 45.3 (94.6% of Claude Opus 4.6) |
| Sources | unsloth/GLM-5.1-GGUF, ubergarm/GLM-5.1-GGUF |

**Quantizations:**

| Quant | Use case |
|-------|----------|
| UD-Q4_K_XL | **Recommended.** Best quality/VRAM balance for H200 |
| UD-Q5_K_S/M/XL | Higher fidelity when VRAM allows |
| UD-Q2_K_XL | Maximum context (200K+) on constrained VRAM |
| UD-Q6_K/XL, UD-Q8_K_XL | Diminishing returns; use only for validation |

**Why GLM-5.1 for workers:**

- 40B active parameters — enough depth for analytical reasoning, not
  just summarization
- MoE efficiency — fast inference despite large total parameter count
- 200K context with UD-Q2_K_XL — data packages can be 30-50K chars
  (5× the current 6K budget) without truncation
- Uncensored natively — no abliteration artifacts
- Best open-source coding score — coding benchmarks correlate with
  systematic analytical ability (following logic chains, identifying
  edge cases, structured output)
- No multimodal overhead — pure text reasoning, which is all workers need

**Risks:**

- 200K context is good but not unlimited. At wave 50+ with 10K+
  findings, data packages may still need curation.
- MoE routing quality degrades at very low quantizations. UD-Q2_K_XL
  may lose subtle reasoning ability.

---

#### GPT-OSS 20B — **Catalogue / Flock clone model**

| Property | Value |
|----------|-------|
| Uncensored | Yes — open weights |
| Architecture | Dense 20B |
| Context | TBD (likely 128K+) |
| Key benchmarks | Strong reasoning, near o4-mini class |
| Source | OpenAI open-source release |

**Quantizations:**

| Quant | Use case |
|-------|----------|
| UD-Q4_K_S | **Recommended.** ~10-25 GB, fits alongside workers |
| UD-Q5_K_S | Higher quality for complex relationship typing |

**Why GPT-OSS 20B for catalogue/Flock:**

- Relationship typing ("how does finding A relate to finding B?") is a
  structured classification task — 20B dense is sufficient
- Hundreds of short calls per wave — speed and throughput matter more
  than depth
- Small enough to co-locate on a GPU already running a worker instance
- Open weights = uncensored
- Dense architecture = predictable latency (no MoE routing variance)

---

### Tier 2: Strong Alternatives

#### GPT-OSS 120B — **All-rounder / report generation**

| Property | Value |
|----------|-------|
| Uncensored | Yes — open weights |
| Architecture | MoE (~12-50B active) |
| Key benchmarks | Near o4-mini reasoning, GPQA PhD 80.9% |
| VRAM | ~35-60 GB at UD-Q4_K_S |

Good backup worker model if GLM-5.1 has quality issues on specific
research domains. Strong candidate for report generation where you want
maximum reasoning depth in a single long-output call.

#### MiniMax-M2.5 (1T / ~46B active) — **Alternative worker**

| Property | Value |
|----------|-------|
| Uncensored | Yes — local GGUF variants |
| Architecture | Hybrid MoE |
| Key benchmarks | SWE-Verified 80.2%, BrowseComp 76.3% |
| VRAM | 100-200+ GB at UD-IQ4_K_S |

Top agentic performance, but the agentic strength is wasted on tool-free
workers. The VRAM footprint is heavy — 2-3 H200s per instance. Consider
only if GLM-5.1 proves insufficient on reasoning depth.

#### Qwen3.6-35B-A3B — **Lightweight worker**

| Property | Value |
|----------|-------|
| Uncensored | Yes — HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive |
| Architecture | MoE (3B active) |
| Context | 128K estimated |
| VRAM | ~10-15 GB at Q4_K_P |

**Quantizations:** Q2_K_P, Q4_K_P, Q4 (Ollama)

Extremely lightweight — could run 5+ instances on a single H200. But
3B active parameters may be too shallow for deep corpus analysis. Workers
might just summarize instead of reasoning about non-obvious connections.
Test head-to-head against GLM-5.1 before committing.

#### Kimi K2.5 / K2.6 — **Report generation specialist**

| Property | Value |
|----------|-------|
| Uncensored | Yes — community quants |
| Architecture | 1T MoE |
| Key benchmarks | AIME 96.1%, HMMT 95.4% (K2.5), 262K context (K2.6) |
| VRAM | ~550 GB at INT4 |

The strongest pure reasoning model available. The 262K context on K2.6
can hold an entire store summary for report generation. But ~550 GB
VRAM means it consumes 7 of 8 H200s — impractical as a worker model.
Reserve for report generation only if you can dedicate the cluster to
a single report-gen pass after all waves complete.

---

### Tier 3: Watch List (Not Currently Viable)

#### Kimi-Linear-48B-A3B — **Best architecture, wrong censorship**

| Property | Value |
|----------|-------|
| Uncensored | **NO** — aligned instruct model, no abliterated variant |
| Architecture | Hybrid KDA + MLA (3:1), 3B active |
| Context | Native 1M tokens |
| KV cache | 75% smaller than full attention |
| VRAM | 35-60 GB at 4-bit for 1M context |

**This is the most architecturally interesting model on the list.**

The hybrid linear attention with 1M native context at 35-60 GB VRAM
would eliminate the entire token budget machinery (PR #181). Workers
could see the FULL store contents — all findings, all evidence chains,
all peer insights — in a single context window. The 75% KV cache
reduction means faster inference at long contexts.

**But it fails the hard filter: not uncensored.** No HauhauCS, no
huihui_ai, no community abliteration exists. The instruct version has
standard Moonshot safety alignment that will refuse PED/pharmacology
research.

**If an abliterated variant drops, immediately re-evaluate.** The
architecture (1M context + 3B active + 75% KV savings) is perfect for
tool-free workers processing large data packages.

#### Qwen3.6-397B-A17B — **Too heavy without sharding infrastructure**

Uncensored Q4 exists (~200 GB footprint) but requires 3×H200 for a
single instance, leaving only 5 GPUs for everything else. Worth
exploring if you need deeper reasoning than GLM-5.1 provides and can
sacrifice worker parallelism.

#### MiniMax-M2.7 — **Unproven benchmarks**

ELO ~1495 on internal suite and 97% skill compliance sound impressive
but lack independent SWE/AIME/GPQA verification. Watch for public
benchmark results before considering.

---

## Recommended H200 Allocation

### Primary Configuration

```
GPU 0-4:  GLM-5.1 UD-Q4_K_XL  (workers, one per GPU)
GPU 5-6:  GPT-OSS 20B UD-Q4_K_S  (catalogue/Flock clones)
GPU 7:    GLM-5.1 UD-Q4_K_XL  (report generation)
```

**Workers (5 GPUs):** Each GPU runs one GLM-5.1 instance. 40B active
parameters provide deep analytical reasoning. 200K context allows
data packages up to ~50K chars without truncation. Five parallel
workers cover five research angles per wave.

**Catalogue (2 GPUs):** GPT-OSS 20B handles relationship typing for
the QR ontology graph. Hundreds of short "how does A relate to B?"
calls per wave. Dense 20B is fast and predictable. Two GPUs provide
throughput headroom for high-wave runs.

**Report (1 GPU):** Reuses GLM-5.1 for final synthesis. Single call
with the full store summary. Could swap to GPT-OSS 120B if report
quality needs more depth (would need 1-2 GPUs).

### Alternative: Uniform Deployment

```
GPU 0-7:  GLM-5.1 UD-Q4_K_XL  (all roles)
```

Simpler ops — one model, one set of inference parameters. Workers,
catalogue, and report all use GLM-5.1. Trade catalogue throughput
for operational simplicity. Recommended for initial testing.

### High-Reasoning Configuration (Report-Heavy)

```
GPU 0-4:  GLM-5.1 UD-Q4_K_XL  (workers)
GPU 5:    GPT-OSS 20B UD-Q4_K_S  (catalogue)
GPU 6-7:  GPT-OSS 120B UD-Q4_K_S  (report generation)
```

Dedicates 2 GPUs to a stronger report model. Use when report quality
is the bottleneck (e.g., final deliverable runs after exploration
phases are complete).

---

## Impact on Token Budget Machinery

The current codebase (PR #181, PR #206) implements aggressive token
budgets designed for 32K-context models:

| Parameter | Current Value | With GLM-5.1 (200K) | With Kimi-Linear (1M, if uncensored) |
|-----------|--------------|---------------------|--------------------------------------|
| `max_return_chars` | 6,000 | **30,000-50,000** | **500,000+** (effectively unlimited) |
| `report_max_chars` | 24,000 | **100,000-150,000** | **500,000+** |
| Pagination in `get_corpus_section` | Required | Still useful for 150MB corpus | Optional |
| Rolling summaries | Critical for context management | Useful for orientation, not required for fitting | Purely optional |
| `_truncate_to_budget()` | Hard-capped at 6K | Raise cap, keep as safety net | Remove or set to 500K |
| Compaction urgency | Every 3 waves | Every 10+ waves | Only when store > 500K rows |

**Action:** After confirming GLM-5.1 as the worker model, raise
`max_return_chars` to 30,000 and `report_max_chars` to 100,000. Keep
the truncation machinery as a safety net but set limits that reflect
the actual context capacity.

---

## Validation Plan

### Stage 2 PoC: Head-to-Head Comparison

Before committing to GLM-5.1, run this test (see issue #202):

1. Prepare a data package from a real 5-wave run (~3000 findings)
2. Run GLM-5.1 (UD-Q4_K_XL) as worker with 30K-char data package
3. Run Qwen3.6-35B-A3B (Q4_K_P uncensored) with same data package
4. Run GLM-5.1 with the current 6K-char truncated data package
5. Human-judge output quality: connection depth, specificity, non-obvious insights
6. Measure: tokens/second, findings stored per wave, unique insights ratio

**Pass criteria:** GLM-5.1 with 30K package must produce ≥15% more
unique insights than GLM-5.1 with 6K package. If Qwen3.6 at 3B active
produces ≥90% the quality of GLM-5.1, prefer it for throughput.

### Monthly Re-Evaluation Triggers

- New abliterated variant of a hybrid-linear model (Kimi-Linear class)
- New MoE model with >200K context and uncensored weights
- Benchmark results for MiniMax-M2.7 or GPT-OSS variants
- Unsloth Dynamic quantization improvements that meaningfully
  change VRAM requirements

---

## Model Intelligence Profiles

### For Reference: Reasoning Depth by Active Parameters

These profiles describe analytical capability relevant to tool-free
workers processing structured data packages. Agentic/tool-calling
benchmarks are excluded as irrelevant.

**GLM-5.1 (40B active):**
Strong systematic analysis. Can follow multi-step logic chains across
a large evidence set. Identifies edge cases and contradictions. The
coding benchmark (94.6% of Claude Opus) is the best proxy for the
structured analytical reasoning workers need. Weakness: no multimodal,
and GPQA Diamond (86.0%) suggests the deepest abstract reasoning
isn't quite frontier-class. For corpus analysis, this doesn't matter.

**GPT-OSS 120B (~12-50B active):**
Broad reasoning comparable to o4-mini. Good at synthesis and structured
classification. The 120B total parameter count provides wide knowledge
coverage. Better than GLM-5.1 for report generation (longer coherent
output), but MoE routing overhead makes it slightly less efficient for
the many-short-calls pattern of catalogue work.

**GPT-OSS 20B (dense):**
Solid structured classification. Good at "how does A relate to B?"
judgments that catalogue/Flock operations require. The dense architecture
means consistent latency — no MoE routing variance. Not deep enough
for worker-level analytical reasoning but perfect for typed edge
assignment in the QR ontology graph.

**Qwen3.6-35B-A3B (3B active):**
Fast but shallow. Good at surface-level analysis and summarization.
The 3B active parameter count limits the ability to find non-obvious
connections in large data packages. May produce "Part 1-5" style
outputs rather than genuine analytical insights. Best reserved for
high-throughput, low-depth tasks or as a baseline comparison.

**Kimi K2.5/K2.6 (MoE, ~550 GB VRAM):**
The strongest pure reasoning available. AIME 96.1% suggests frontier
mathematical/logical reasoning. The 262K context on K2.6 makes it
excellent for single-shot report generation where the entire store
can fit in context. Impractical for workers due to VRAM requirements.

**Kimi-Linear-48B-A3B (3B active, 1M context):**
Architecturally revolutionary but censored. If uncensored, the 1M
native context with 75% KV cache reduction would make it the ideal
worker model — every finding, every evidence chain, every peer insight
in a single context window. The 3B active parameter depth is a concern
(same as Qwen3.6) but the massive context may compensate by giving
the model more evidence to reason over. Watch list until abliterated.

---

## Design Invariants

14. **Uncensored is a hard filter, not a preference.** No model enters
    production without proven uncensored operation on target research
    domains.

15. **Model allocation is configuration, not architecture.** The swarm
    engine takes `--model` and `--api-base` flags. Swapping models
    requires zero code changes.

16. **Token budgets are safety nets, not design constraints.** Set
    limits to 80% of the model's proven context capacity. Never design
    data packages around the limit — design them around what the worker
    needs, then verify they fit.

17. **Re-evaluate monthly.** The model landscape moves fast. An
    abliterated hybrid-linear model (Kimi-Linear class) would change
    the entire token budget architecture. Keep the machinery but don't
    over-invest in optimizing for current limits.
