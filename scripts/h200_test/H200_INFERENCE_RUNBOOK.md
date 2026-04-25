# H200 Inference Runbook — DeepSeek V4 Pro & Kimi Linear 48B

Low-level operational findings from deploying two SOTA models on 8×NVIDIA H200
(Vast.ai, Iceland) for the GossipSwarm phase-swap pipeline.

**Hardware**: 8× NVIDIA H200 SXM5, 143,771 MiB (140.4 GiB) VRAM each, compute
capability 9.0 (Hopper), 2 TB NVMe, ~2 TB system RAM.

**vLLM version**: v0.1.dev15833+g62d441ee8 (V1 engine).

---

## 1. DeepSeek V4 Pro 1.6T (Flock Evaluation Phase)

### 1.1 Model Profile

| Property | Value |
|----------|-------|
| Full name | `deepseek-ai/DeepSeek-V4-Pro` |
| Architecture | MoE, 384 routed experts, top-8 routing |
| Total params | ~1.6T |
| Active params | 49B per token |
| Attention | Hybrid CSA (Compressed Shared Attention) + HCA (Hybrid Chunked Attention) |
| Native context | 1,048,576 tokens (1M) |
| Weight size on disk | ~900 GB (64 safetensors shards) |
| Download time | ~30 min at 500 MB/s from HuggingFace |

### 1.2 What Worked

**Final working configuration (TP=8)**:
```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-prefix-caching \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --tokenizer-mode deepseek_v4
```

**Measured values**:
- Weight memory: ~102.68 GiB total across 8 GPUs (~12.8 GiB per GPU)
- KV cache memory: 17.55 GiB allocated
- Max context: 1,048,576 tokens
- Test query `"What is 2+2?"` → `"Four"` in 2 tokens, sub-second latency

### 1.3 What Failed and Why

#### Attempt 1: DP=8 (blog recipe) — OOM

The official vLLM blog post for V4 Pro uses `--data-parallel-size 8` with a
dedicated Docker image (`vllm/vllm-openai:deepseekv4-cu130`). This **replicates
the full model on every GPU** — each GPU holds ~127 GiB of weights. On B200
(192 GB VRAM) this leaves ~65 GB for KV cache. On H200 (143 GB) it leaves only
~16 GB, which is insufficient for CUDA graph profiling.

```
torch.OutOfMemoryError: Tried to allocate 10.50 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 4.25 GiB is free.
```

**Lesson**: The DP=8 recipe is designed for Blackwell (B200/B300 with 192+ GB).
H200 requires TP=8 (tensor parallelism — shards the model across GPUs).

#### Attempt 2: TP=8 + CUDA graphs — Worker death

With TP=8, per-GPU weight memory drops to ~12.8 GiB, but CUDA graph
compilation still OOM'd during the warmup phase. The graph profiler allocates
temporary buffers proportional to max_model_len, exhausting remaining VRAM.

**Fix**: `--compilation-config '{"cudagraph_mode":"NONE"}'` disables CUDA graph
capture entirely. Trade-off: ~2-3× slower decode throughput vs. compiled graphs.

#### Attempt 3: FP4 indexer cache — Blackwell-only

```
--attention_config.use_fp4_indexer_cache=True
```

This flag enables the Lightning Indexer with FP4 quantized attention indices.
It requires Blackwell architecture (sm_100+). On Hopper (sm_90), vLLM silently
fails to initialize the attention backend.

**Fix**: Removed the flag entirely. FP8 KV cache (`--kv-cache-dtype fp8`) is
the maximum quantization level available on H200.

### 1.4 H200-Specific Constraints

| Constraint | H200 (Hopper) | B200 (Blackwell) |
|------------|---------------|-------------------|
| Max VRAM per GPU | 143.7 GiB | 192 GiB |
| Compute capability | 9.0 | 10.0+ |
| FP4 indexer cache | NOT supported | Supported |
| DP=8 (V4 Pro) | OOM — weights alone use 127 GiB | Works (65 GiB spare) |
| TP=8 (V4 Pro) | Works (~12.8 GiB/GPU weights) | Works |
| CUDA graphs + 1M context | OOM during profiling | Likely works |
| FP8 KV cache | Supported | Supported |

### 1.5 Disk Requirements

V4 Pro requires **≥1.5 TB** of usable disk space:
- Model weights: ~900 GB (safetensors)
- HuggingFace cache overhead: ~50 GB (index files, config, tokenizer)
- vLLM temporary files during loading: ~200 GB peak
- OS + dependencies: ~100 GB

A 500 GB disk caused the first Vast.ai instance to fail. 2 TB is the safe
minimum.

---

## 2. Kimi Linear 48B-A3B-Instruct (Worker Synthesis Phase)

### 2.1 Model Profile

| Property | Value |
|----------|-------|
| Full name | `moonshotai/Kimi-Linear-48B-A3B-Instruct` |
| Architecture | Hybrid linear attention (KDA + MLA, 3:1 ratio) |
| Total params | 48B |
| Active params | 3B (MoE-style routing) |
| Attention layers | 36 linear (KDA) + 12 standard (MLA) |
| Native context | 1,048,576 tokens (1M) |
| Weight size (BF16) | 91.50 GiB (safetensors) |
| Weight size (FP8) | ~46.66 GiB per instance after online quantization |
| Download time | ~3 min at 500 MB/s |

### 2.2 Working Configuration (8 instances × TP=1 × FP8)

```bash
# Launched 8 times, one per GPU (CUDA_VISIBLE_DEVICES=0..7)
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 524288 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enforce-eager \
    --port 800X  # 8000-8007
```

### 2.3 Measured Performance (Per Instance)

| Metric | Value |
|--------|-------|
| Weight load time | 22.23 seconds |
| Model memory (FP8) | ~46.66 GiB |
| Available KV cache memory | 26.06 GiB |
| KV cache capacity | 1,733,184 tokens |
| Max concurrency at 524K context | 12.67× (theoretical) |
| Attention block size | 3,776 tokens (auto-sized for hybrid Mamba/attention) |
| Prefill throughput | 684–1,526 tokens/s |
| Decode throughput (eager) | 20.2–21.0 tokens/s |
| GPU VRAM used | 141,793 MiB (~141.7 GiB) of 143,771 MiB |
| Prefix cache hit rate | 0% (each instance serves different workers) |

### 2.4 What Failed and Why

#### CUDA Graph OOM at 512K Context

With `max_model_len=1048576` (1M tokens), CUDA graph profiling attempted to
allocate 59 GiB for the warmup pass:

```
torch.OutOfMemoryError: Tried to allocate 59.00 GiB.
GPU 0 has a total capacity of 139.80 GiB of which 32.35 GiB is free.
```

Memory breakdown at failure:
- Model weights (FP8): 46.66 GiB
- PyTorch allocator: 58.89 GiB (pre-allocated KV cache + activations)
- PyTorch reserved: 47.81 GiB
- CUDA graph profiling request: 59.00 GiB
- Total requested: ~212 GiB > 143.7 GiB available

**Fix**: Two changes applied together:
1. `--enforce-eager` — disables CUDA graphs entirely (no profiling allocation)
2. `--max-model-len 524288` — reduces KV cache pre-allocation from 1M to 512K

Either change alone might suffice, but both were applied for reliability.

#### Slow Tokenizer Warning

```
WARNING: Using a slow tokenizer. This might cause a significant slowdown.
```

Kimi Linear's tokenizer doesn't have a fast (Rust-backed) implementation.
Tokenization is Python-based and adds latency on long prompts. For 50K-token
worker inputs, this adds ~2-3 seconds per request (negligible vs. 13 min
generation time).

#### Mamba Cache Alignment

```
WARNING: Mamba cache mode is set to 'align' for KimiLinearForCausalLM
by default when prefix caching is enabled
```

The hybrid architecture's linear attention layers use Mamba-style state caching.
When prefix caching is enabled, vLLM auto-aligns Mamba page boundaries with
attention page boundaries. The attention block size was auto-set to 3,776 tokens
(instead of the default 16) to ensure attention pages are ≥ Mamba pages.

This is normal behavior — not an error. But it means prefix cache blocks are
larger than typical, reducing cache granularity.

### 2.5 VRAM Budget (Per H200 GPU, Single Instance)

```
┌─────────────────────────────────────────────────────┐
│ NVIDIA H200 — 143,771 MiB (140.4 GiB)              │
├─────────────────────────────────────────────────────┤
│ Model weights (FP8 online quant)     46.66 GiB      │
│ KV cache pre-allocation              26.06 GiB      │
│ PyTorch runtime / activations        ~65 GiB         │
│ CUDA context + driver                ~3 GiB          │
├─────────────────────────────────────────────────────┤
│ Total used                           ~141.7 GiB      │
│ Free (gpu_memory_utilization=0.95)   ~2 GiB          │
└─────────────────────────────────────────────────────┘
```

### 2.6 Multi-Instance Density Analysis

| Configuration | Instances | Context/inst | Gen tok/s/inst | Aggregate tok/s |
|---------------|-----------|-------------|----------------|-----------------|
| 1× TP=8 (BF16) | 1 | 1M | ~60-100 (estimated w/ graphs) | 60-100 |
| 8× TP=1 (FP8, eager) | 8 | 512K | 21 | **168** |
| 8× TP=1 (FP8, graphs, 131K) | 8 | 131K | 60-100 (projected) | **480-800** |
| 16× TP=1 (FP8, 2/GPU) | 16 | ~128K | ~15-18 (memory-constrained) | ~240-288 |

**Key insight**: 8× TP=1 at 21 tok/s aggregate (168 tok/s) is already 1.7-2.8×
faster than 1× TP=8 for parallel workloads, despite eager mode throttling per-
instance speed. CUDA graphs would push the 8× TP=1 config to 480-800 tok/s
aggregate — a 3-5× improvement.

---

## 3. Phase-Swap Architecture

### 3.1 Sequence

```
Phase 1: Worker Synthesis (Kimi Linear)
├── Launch 8 vLLM instances (TP=1, FP8, ports 8000-8007)
├── Load weights: 22s per instance (parallel across GPUs)
├── Run 12 workers × 3 gossip rounds
├── Save worker context to disk (ContextPersistence)
└── Kill all 8 vLLM processes

Phase 2: Flock Evaluation (DeepSeek V4 Pro)
├── Launch 1 vLLM instance (TP=8, all 8 GPUs)
├── Load weights: ~5 min (900 GB from NVMe cache)
├── Run 10,800 Flock evaluation queries
├── Prefix caching: shared prefix cached once, queries hit cache
└── Kill vLLM process

Post-Pipeline:
├── Aggregate traces from Phase 1 + Phase 2
├── Upload to B2 (mirothinker-traces bucket)
└── Generate final metrics report
```

### 3.2 Context Persistence Between Phases

Worker outputs from Phase 1 are saved to disk as JSON:
```
/root/swarm_results/worker_context/
├── round_1_state.json    # All 12 worker summaries after round 1
├── round_2_state.json    # After gossip round 2
├── round_3_state.json    # Final worker state
└── worker_outputs.json   # Cross-phase transfer file for Flock
```

Phase 2 loads `worker_outputs.json` as context for Flock evaluation queries.
Workers can also be rehydrated between gossip rounds if the model must be
reloaded (e.g., after an OOM crash).

### 3.3 GPU Memory Transition

```
Phase 1 (per GPU):           Phase 2 (per GPU):
┌──────────────────┐         ┌──────────────────┐
│ Kimi Linear FP8  │         │ V4 Pro shard     │
│ 46.66 GiB        │         │ ~12.8 GiB        │
├──────────────────┤         ├──────────────────┤
│ KV cache          │         │ KV cache          │
│ 26.06 GiB        │         │ 17.55 GiB        │
├──────────────────┤         ├──────────────────┤
│ Runtime           │         │ Runtime           │
│ ~68 GiB           │         │ ~113 GiB          │
└──────────────────┘         └──────────────────┘
Total: 141.7 GiB            Total: ~143 GiB
```

V4 Pro in TP=8 mode uses far less weight memory per GPU but requires all 8 GPUs
to be in the same process (inter-GPU communication via NVLink).

---

## 4. Observed Bottlenecks and Optimization Opportunities

### 4.1 Eager Mode Decode Penalty

**Current**: 21 tok/s per instance (eager mode)
**Projected with CUDA graphs**: 60-100+ tok/s per instance

Eager mode disables torch.compile and CUDA graph capture. Every decode step
recomputes the full forward pass without kernel fusion or graph replay. This is
the single largest throughput penalty.

**Proposed fix**: Reduce `max_model_len` from 524K to 131K (still 2.6× larger
than the biggest worker input at 50K tokens), which shrinks KV cache pre-
allocation and leaves room for CUDA graph profiling buffers.

### 4.2 Zero Prefix Cache Hits

**Observed**: 0% prefix cache hit rate across all 8 instances.

Each instance serves different workers with different prompts. The shared system
prompt is identical across all workers, but since workers are round-robin
distributed across instances, no single instance processes the same prefix
twice within a round.

**Proposed fixes**:
1. Route workers with the same system prompt prefix to the same instance
2. Structure prompts so the corpus prefix is identical (sort sections
   deterministically) — enables automatic prefix dedup within an instance
3. For gossip rounds: reuse the same instance for the same worker across
   rounds, so the prefix from round N is cached for round N+1

### 4.3 Load Imbalance (12 Workers, 8 Instances)

Round 1 map phase: 8 workers run in parallel, 4 wait. Total wall time
determined by the slowest of the first 8 plus the slowest of the remaining 4.

**Measured**: 1575.5 seconds (26.3 min) for 12 workers across 8 instances.
Optimal (if perfectly balanced): 1575.5 × 8/12 = 1050s (17.5 min).
Overhead from queueing: ~50%.

**Proposed fix**: Split each worker's corpus section into 2-3 sub-chunks and
pipeline them so all 8 instances stay continuously busy.

### 4.4 Slow Tokenizer

Kimi Linear lacks a fast (Rust) tokenizer. For 50K-token inputs, Python-based
tokenization adds ~2-3 seconds per request. Negligible for the current workload
(generation dominates at 13 min per worker) but would matter at higher
throughput.

### 4.5 KV Cache Sizing Headroom

At 512K context with FP8 KV cache, each instance pre-allocates 26.06 GiB for
1,733,184 tokens of KV cache. Our workers use at most ~65K tokens (50K input +
16K output). This means 96% of the pre-allocated KV cache goes unused.

Reducing `max_model_len` to 131K would free ~20 GiB per GPU, which is exactly
what's needed for CUDA graph profiling (~10-15 GiB) while still providing 2×
headroom over the largest worker input.

---

## 5. Pipeline Run Metrics (Pipeline 7)

### 5.1 Timeline

| Event | Timestamp (UTC) | Elapsed |
|-------|-----------------|---------|
| vLLM instances launched | 19:18:10 | 0:00 |
| All 8 instances healthy | 19:21:05 | 2:55 |
| Corpus analysis complete | 19:21:05 | 2:55 |
| Semantic scoring (2 LLM calls) | 19:22:36–19:23:29 | 4:26–5:19 |
| Worker assignment complete | 19:23:29 | 5:19 |
| Round 1 map phase complete | 19:49:44 | 31:34 |
| Round 1 map LLM time | — | 1575.5s (26.3 min) |
| Gossip/reduce phase started | 19:49:48 | 31:38 |

### 5.2 vLLM Request Distribution (Round 1)

| Port | Instance | Completions (map) | Completions (gossip) |
|------|----------|-------------------|----------------------|
| 8000 | GPU 0 | 5 | 3+ |
| 8001 | GPU 1 | 3 | 3+ |
| 8002 | GPU 2 | 3 | 4+ |
| 8003 | GPU 3 | 4 | 1+ |
| 8004 | GPU 4 | 4 | 1+ |
| 8005 | GPU 5 | 3 | 1+ |
| 8006 | GPU 6 | 3 | 1+ |
| 8007 | GPU 7 | 3 | 2+ |

### 5.3 Tracing

- Trace file: `/root/traces/run_20260425_191810_spans.jsonl`
- Size at last check: 16.5 KB (growing, spans flush on completion)
- Spans include: model name, backend, prompt/response sizes, latency, phase

### 5.4 Corpus

- Source: `/root/workspace/MiroThinker/scripts/h200_test/mega_corpus.txt`
- Content: YouTube transcripts (DrJasonFung, Gabrielle Lyon, etc.) + PubMed/
  Semantic Scholar enrichment
- Size: 2,685,757 chars, 772 sections
- Domain: PED/fitness/metabolic health (insulin, anabolics, GH, GLP-1, etc.)

---

## 6. vLLM Configuration Reference

### 6.1 Flags That Matter on H200

| Flag | Purpose | H200 Notes |
|------|---------|------------|
| `--tensor-parallel-size N` | Shard model across N GPUs | Use for V4 Pro (TP=8). Use TP=1 for small models (Kimi Linear) |
| `--data-parallel-size N` | Replicate model on N GPUs | **NOT for H200** with V4 Pro (OOM). Works for small models |
| `--quantization fp8` | Online FP8 weight quantization | ~50% VRAM reduction. CutlassFP8ScaledMM kernel selected automatically on Hopper |
| `--kv-cache-dtype fp8` | FP8 KV cache values | ~50% KV cache VRAM reduction |
| `--enforce-eager` | Disable CUDA graphs + torch.compile | Required when CUDA graphs OOM. Costs 2-3× decode speed |
| `--enable-prefix-caching` | Cache common prefixes | Works with hybrid attention. Block size auto-adjusted for Mamba alignment |
| `--block-size 256` | Attention block size | Used for V4 Pro. Kimi Linear auto-sets to 3776 for Mamba alignment |
| `--gpu-memory-utilization 0.95` | VRAM allocation fraction | Default 0.9. Set to 0.95 to maximize KV cache on tight budgets |
| `--max-model-len N` | Maximum sequence length | Reduce from native max to save VRAM for CUDA graphs |
| `--attention_config.use_fp4_indexer_cache` | FP4 Lightning Indexer | **Blackwell only** (sm_100+). Crashes silently on H200 |
| `--compilation-config '{"cudagraph_mode":"NONE"}'` | Disable CUDA graphs explicitly | Alternative to `--enforce-eager` with finer control |

### 6.2 Kimi Linear Specific Behavior

- **Mamba cache alignment**: Auto-enabled when prefix caching is on. Attention
  block size inflated from 16 to 3776 to match Mamba page boundaries.
- **Chunked prefill**: Auto-enabled with `max_num_batched_tokens=8192`.
- **FP8 online quantization**: Uses `CutlassFP8ScaledMMLinearKernel` (auto-
  selected for Hopper).
- **Tokenizer**: Slow (Python-based). No fast Rust tokenizer available.
- **Hybrid layers**: 36 linear (KDA) + 12 standard (MLA). The linear layers
  have constant-size state regardless of context length. Only the 12 MLA layers
  contribute to KV cache growth.

### 6.3 DeepSeek V4 Pro Specific Behavior

- **Expert parallelism**: Available via `--enable-expert-parallel` (DP mode
  only). Not applicable in TP=8 mode on H200.
- **Tokenizer**: Custom `deepseek_v4` tokenizer mode required (`--tokenizer-
  mode deepseek_v4`).
- **Reasoning**: Supports separate CoT via `--reasoning-parser deepseek_v4`.
- **Docker image**: `vllm/vllm-openai:deepseekv4-cu130` (specific build with
  CUDA 13.0, Blackwell optimizations). **Not required on H200** — standard
  vLLM works with TP=8.

---

## 7. Cost Analysis

| Phase | Model | Config | Duration | Vast.ai Cost |
|-------|-------|--------|----------|-------------|
| Weight download (V4 Pro) | — | — | ~30 min | $14.46 |
| Weight download (Kimi Linear) | — | — | ~3 min | $1.45 |
| Phase 1: Worker synthesis (current) | Kimi Linear | 8×TP=1 FP8 eager | ~90 min (est. 3 rounds) | $43.30 |
| Phase 1: Worker synthesis (optimized) | Kimi Linear | 8×TP=1 FP8 graphs | ~25 min (est. 3 rounds) | $12.05 |
| Phase 2: Flock evaluation | V4 Pro | 1×TP=8 | ~30 min (est.) | $14.46 |
| **Total (current config)** | | | **~2.5 hours** | **~$72** |
| **Total (optimized)** | | | **~1 hour** | **~$29** |

Vast.ai rate: $28.91/hr for 8×H200.

---

## 8. Why Kimi Linear — and When to Use Something Else

### 8.1 Kimi Linear's Only Differentiator Is 1M Context

Kimi Linear 48B-A3B exists for one reason: native 1,048,576-token context
via hybrid linear attention (36 KDA + 12 MLA layers). The linear layers have
constant-size state regardless of sequence length, so the KV cache at 1M
tokens is **75% smaller** than a full-attention model of the same size, and
decoding at 1M context is **6.3× faster** than equivalent full-attention.

**If you are not using the 1M context, you are paying a tax for no benefit.**

At ≤256K context, Kimi Linear's hybrid attention provides no meaningful
advantage over standard attention — the KV cache is small enough to fit in
VRAM either way, and decode speed at short context is similar. Meanwhile,
newer models with identical 3B active params offer better quality, smaller
weights, multimodal capability, and broader ecosystem support.

### 8.2 The Replacement: Qwen3.6-35B-A3B (April 2026)

Qwen3.6-35B-A3B was released on April 14, 2026 — 6 months after Kimi Linear
(October 2025). It uses a nearly identical hybrid architecture (GatedDeltaNet
+ Gated Attention in 3:1 ratio, same as Kimi Linear's KDA + MLA in 3:1
ratio), but improves on every other dimension:

| Property | Kimi Linear 48B-A3B | Qwen3.6-35B-A3B |
|----------|--------------------:|------------------:|
| Total params | 48B | **35B** (lighter) |
| Active params | 3B | 3B (same) |
| Hybrid attention ratio | 3:1 (KDA:MLA) | 3:1 (GatedDeltaNet:GQA) |
| Native context | **1,048,576** (1M) | 262,144 (262K) |
| Extended context (YaRN) | — | 1,010,000 (~1M) |
| FP8 weights | ~47 GB | **~35 GB** |
| Multimodal (vision) | No | **Yes** |
| Uncensored variant | Yes (huihui-ai) | **Yes** (HauhauCS, 0/465 refusals) |
| SWE-bench Verified | N/A | **73.4** |
| AIME 2026 | N/A | **92.7** |
| vLLM support | Yes | **Yes** (GatedDeltaNet merged Mar 2026) |
| License | MIT | Apache 2.0 |

### 8.3 Decision Matrix

| Context needed | Best model | Best GPU | Cost/hr | Rationale |
|---------------|------------|---------|--------:|-----------|
| ≤131K tokens | **Qwen3.6-35B-A3B** | H100 PCIe (80 GB) | **$1.53** | Smaller weights (35 GB), fits with CUDA graphs, newer/smarter, multimodal |
| 131K-262K tokens | **Qwen3.6-35B-A3B** | H100 PCIe (80 GB) | **$1.53** | Native 262K context, no YaRN needed |
| 262K-512K tokens | Kimi Linear 48B or Qwen3.6 (YaRN) | H200 (143 GB) | $2.32 | Kimi Linear native; Qwen3.6 needs YaRN extrapolation |
| **512K-1M tokens** | **Kimi Linear 48B** | **H200 (143 GB)** | **$2.32** | **Only model with native 1M + proven hybrid attention at this length** |

**Rule**: If your input is under 262K tokens, use Qwen3.6-35B-A3B. It's
smaller, faster, smarter, and $0.79/hr cheaper per GPU. Kimi Linear only
justifies its 47 GB weight tax when you are genuinely operating in the
512K-1M token range where its native linear attention dominance kicks in.

### 8.4 Qwen3.6-35B-A3B on H100 PCIe

```
FP8 weights:             ~35 GB
KV cache at 131K (FP8):   ~5 GB
CUDA graph profiling:    ~10 GB
Runtime:                  ~3 GB
────────────────────────────────
Total:                   ~53 GB of 80 GB (27 GB spare)
```

27 GB spare means CUDA graphs work comfortably — projected 60-100+ tok/s
per instance. At $1.53/hr per H100 PCIe, a 12-GPU fleet costs $18.36/hr
and delivers an estimated 720-1200 aggregate tok/s.

### 8.5 Why Not Cheaper GPUs?

- **L40S / A6000 (48 GB)**: FP8 weights for Kimi Linear (~47 GB) or even
  Qwen3.6 (~35 GB) leave too little room for KV cache + runtime at 131K
  context. INT4 quantization would fit but degrades MoE expert routing
  quality and no official INT4 quants exist for either model
- **RTX 4090 (24 GB)**: Cannot fit either model at FP8 on a single card.
  TP=2 across two 4090s uses PCIe interconnect (no NVLink), adding 5-10×
  latency to every cross-GPU op
- **A100 (80 GB, ~$1.30/hr)**: Fits FP8 weights but Ampere lacks native FP8
  tensor cores. Emulated FP8 is 30-40% slower than native Hopper FP8.
  The $0.23/hr savings vs H100 doesn't justify the throughput penalty

---

## 9. Exploiting the Full 1M Context — The Whole Point of Kimi Linear

If we're running Kimi Linear, we must be using the 1M context. Otherwise
Qwen3.6-35B-A3B is strictly better (see section 8). This section describes
the architecture required to actually operate at 1M tokens.

### 9.1 VRAM Requirements for 1M Context

The KV cache scales differently for Kimi Linear's two layer types:

| Layer type | Count | KV cache scaling | Size at 1M tokens |
|-----------|------:|-----------------|-------------------:|
| Linear (KDA) | 36 | **Constant** — fixed-size state | ~94 MB total |
| Standard (MLA) | 12 | **Linear** — grows with sequence | ~49 GB (BF16) / ~25 GB (FP8) |
| **Total KV cache** | 48 | | **~25 GB (FP8)** |

This is the hybrid attention payoff: a full-attention 48B model would need
~100 GB of KV cache at 1M tokens. Kimi Linear needs only ~25 GB because 36
of 48 layers have constant-size state.

Total VRAM per instance at 1M context:
```
Model weights (FP8):     47 GB
KV cache (FP8, 1M):     25 GB
Runtime / activations:  ~10 GB
────────────────────────────────
Subtotal:               ~82 GB  ← minimum for eager mode at 1M
CUDA graph profiling:  +15-60 GB ← depends on graph capture strategy
```

### 9.2 GPU Selection: H200 Is Required

| GPU | VRAM | Fits 1M eager? | Fits 1M + CUDA graphs? |
|-----|------|:--------------:|:----------------------:|
| **H200 (143 GB)** | 143 GB | **Yes** (61 GB spare) | Depends — OOM'd at 512K, untested at 1M with PIECEWISE |
| H100 80GB | 80 GB | **No** (~82 GB needed) | No |
| A100 80GB | 80 GB | **No** | No |
| B200 (192 GB) | 192 GB | **Yes** (110 GB spare) | **Likely yes** |

H100 PCIe (80 GB) cannot serve Kimi Linear at 1M context — weights (47 GB)
+ KV cache (25 GB) + runtime (10 GB) = ~82 GB > 80 GB. Full 1M context
**requires H200 or Blackwell**. This is why the H200 fleet exists — not for
131K workloads (where H100 + Qwen3.6 would be cheaper and faster), but for
the 1M context that only Kimi Linear can natively serve.

### 9.3 Throughput at 1M Context (Eager Mode)

Our H200 run proved CUDA graphs OOM at 512K context (59 GiB profiling
allocation). At 1M the profiling buffer would be even larger. Eager mode
(`--enforce-eager`) is required, limiting decode to ~21 tok/s per instance.

| Metric | Value |
|--------|------:|
| Prefill throughput | 684-1526 tok/s |
| Prefill time for 1M tokens | **11-24 min** |
| Decode throughput (eager) | ~21 tok/s |
| Decode time for 16K output | ~13 min |
| **Total per request (1M in + 16K out)** | **~24-37 min** |

For a 12-worker swarm on 8× H200:
- Round time: ~75-111 min (vs 26 min at 512K, vs ~5 min at 131K with graphs)
- Cost per round: $28.91/hr × ~1.5 hr = **~$43 per round**

**Potential workaround**: vLLM's `cudagraph_mode: "PIECEWISE"` captures
graphs for only the decode phase (smaller buffers). This might fit at 1M on
H200 — untested. If it works, decode could jump from 21 to 60-100 tok/s,
cutting total request time from ~37 min to ~24 min.

### 9.4 Corpus Design for 1M Inputs

Our current corpus is 2.69M chars (~672K tokens). Split across 12 workers,
each gets only ~56K tokens — nowhere near 1M. To justify Kimi Linear's 1M
context, the pipeline must restructure how workers receive data:

**Strategy 1: Every worker sees the full corpus**
- Give each worker the entire 672K-token corpus (not a slice)
- Workers differentiate via angle-specific system prompts, not corpus slicing
- Pro: eliminates information silos, enables cross-domain insight
- Con: 8 instances × 672K tokens = high prefill cost (~7-16 min each)
- Prefix caching critical: corpus prefix is identical across workers,
  so instances 2-8 get near-instant prefill from instance 1's cache

**Strategy 2: Multi-round gossip accumulation**
- Round 1: each worker starts with 56K tokens (corpus slice)
- Each gossip round appends 50-100K tokens of peer summaries
- By round 5-6: workers have accumulated 300-600K tokens of context
- Pro: natural growth toward 1M; earlier rounds are fast
- Con: later rounds hit the eager-mode throughput wall

**Strategy 3: Fewer workers, larger shares**
- 3-4 workers instead of 12, each with 168-224K tokens of corpus
- Remaining context budget (~776-832K tokens) reserved for gossip
  accumulation and reasoning chains
- Pro: fewer requests, better GPU utilization
- Con: fewer angles means less diversity of analysis

**Strategy 4: Dramatically larger corpus**
- Expand from 2.69M chars to 30M+ chars (full book collections, entire
  channel archives, multi-source datasets)
- Each of 12 workers gets ~2.5M chars (~625K tokens) per slice
- Pro: genuinely needs 1M context per worker
- Con: requires much more data collection upfront

### 9.5 Prefix Caching Is Non-Negotiable at 1M

At 1M context, prefill dominates the time budget (11-24 min). Without prefix
caching, every request re-processes the entire 1M-token input from scratch.
With caching, subsequent requests that share a prefix skip the cached portion.

Requirements for effective prefix caching at 1M:
1. **Worker-instance affinity**: Same worker always routes to the same GPU.
   The KV cache from round N stays resident for round N+1
2. **Deterministic prompt ordering**: The cached prefix must be byte-identical
   to what was previously processed. Any change (even whitespace) invalidates
   the cache
3. **Corpus-first prompt structure**: Put the shared corpus at the START of
   the prompt (before angle-specific instructions) so the common prefix is
   maximized across workers on the same instance
4. **Sufficient KV cache capacity**: At 1M context with FP8, each sequence
   uses 25 GB. The H200 has 61 GB spare → can cache ~2.4 full 1M sequences
   simultaneously

If strategy 1 (every worker sees full corpus) is used with affinity routing:
- Instance 1 processes worker A (672K tokens): 11-24 min prefill
- Instance 1 then processes worker B (same corpus, different angle): **<1 min**
  prefill (672K tokens cached, only the angle-specific suffix is new)

### 9.6 Recommended Architecture

```
Hardware:  8× H200 (only GPU with enough VRAM for 1M, short of Blackwell)
Instances: 8× TP=1 (one per GPU)
Quantization: FP8 weights + FP8 KV cache
Context:   1,048,576 tokens (full native — this is why we chose Kimi Linear)
Mode:      --enforce-eager (CUDA graphs OOM at 1M)
Caching:   --enable-prefix-caching with affinity routing
Corpus:    Strategy 1 (full corpus to every worker) or Strategy 2 (accumulation)

Expected per-instance:
  First request:  24-37 min (full 1M prefill + 16K decode)
  Cached request: ~13-15 min (prefix hit, decode only)
  
Expected fleet (8× H200):
  Round 1 (cold): ~75-111 min (no prefix cache)
  Round 2+ (warm): ~40-50 min (prefix cache hits from round 1)
  Cost: $28.91/hr × ~1.5 hr/round = ~$43 per round
```

### 9.7 When NOT to Use Kimi Linear

| Scenario | Use instead | Why |
|----------|-------------|-----|
| Worker inputs <200K tokens | **Qwen3.6-35B-A3B** on H100 PCIe | 3× cheaper, 3-5× faster, smarter benchmarks |
| Need multimodal (images) | **Qwen3.6-35B-A3B** | Kimi Linear is text-only |
| Need maximum tok/s | **Qwen3.6-35B-A3B** with CUDA graphs | 60-100 tok/s vs 21 tok/s (eager) |
| Budget-constrained | **Qwen3.6-35B-A3B** on H100 PCIe | $1.53/hr vs $2.32/hr per GPU |
| Input genuinely exceeds 262K tokens | **Kimi Linear** on H200 | The one scenario where it wins |

---

## 10. Lessons Learned

1. **H200 is not B200.** Official blog recipes targeting Blackwell (DP=8, FP4
   indexer) will silently fail or OOM on Hopper. Always check compute
   capability and adjust.

2. **CUDA graph profiling is the hidden VRAM consumer.** The model fits, the KV
   cache fits, but graph warmup needs an additional ~60 GiB temporary
   allocation. Reduce `max_model_len` or use `--enforce-eager` to dodge this.

3. **Multi-instance TP=1 beats single-instance TP=8 for parallel workloads.**
   Even at 21 tok/s (eager), 8 instances aggregate 168 tok/s vs. ~60-100 tok/s
   from a single TP=8 instance. The latency per request is higher, but
   throughput for batch workloads is substantially better.

4. **FP8 online quantization is free quality-wise.** CutlassFP8 kernel
   selection is automatic on Hopper. No accuracy calibration needed — weights
   are quantized online during loading. The ~50% VRAM savings enable multi-
   instance configurations that wouldn't fit at BF16.

5. **Prefix caching needs deliberate prompt design.** Round-robin distribution
   of workers to instances guarantees 0% cache hit rate. Affinity routing
   (same worker → same instance across rounds) is needed to exploit the cache.

6. **Kimi Linear's hybrid attention complicates block sizing.** The Mamba cache
   alignment inflates attention blocks from 16 to 3776 tokens. This reduces
   prefix cache granularity but is unavoidable when prefix caching is enabled.

7. **Disk is a first-class resource.** V4 Pro's 900 GB weights plus cache
   overhead require ≥1.5 TB. A 500 GB disk killed the first Vast.ai instance.
   Always provision 2× the weight size.

8. **Phase-swap latency is dominated by model loading, not GPU clearing.** GPU
   memory is freed instantly when processes die. The bottleneck is reading
   weights from NVMe (~5 min for V4 Pro, ~22s for Kimi Linear). NVMe caching
   makes subsequent loads faster.
