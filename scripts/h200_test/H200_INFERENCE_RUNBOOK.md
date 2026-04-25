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
| Prefix cache hit rate (round 1) | 0% (each instance serves different workers) |
| Prefix cache hit rate (cumulative, 3 rounds) | 39% on heaviest instance (611K hits / 1.57M queries) |

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

### 4.2 Low Prefix Cache Hit Rate

**Round 1 observed**: 0% hit rate — each instance serves different workers with
different prompts, so no prefix is reused within a single round.

**After 3 rounds observed**: 39% hit rate on the heaviest instance (611K token
hits out of 1.57M token queries on port 8000). Gossip rounds partially reuse
prior-round prefixes when the same worker is routed back to the same instance.
This is accidental affinity, not deliberate — a proper affinity-routing scheme
would push this to 80-90%.

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
| Gossip round 1 complete | 20:04:49 | 46:39 |
| Gossip round 1 info_gain | — | 0.836 |
| Gossip round 1 LLM time | — | 904.8s (15.1 min) |
| Gossip round 2 complete | 20:31:00 | 1:12:50 |
| Gossip round 2 info_gain | — | 0.526 |
| Gossip round 2 LLM time | — | 2475.9s (41.3 min) |
| Gossip round 3 started | 20:31:02 | 1:12:52 |
| Gossip round 3 hive hits | — | 11 workers |
| Gossip round 3 complete | 20:57:42 | 1:39:32 |
| Gossip round 3 info_gain | — | 0.664 (↑ from 0.526) |
| Gossip round 3 LLM time | — | 4077.2s (67.9 min) |
| Gossip round 4 started (bonus) | 20:57:43 | 1:39:33 |

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

## 10. Runtime Context Analysis — The Gossip Accumulation Gap

### 10.1 Per-Instance Token Usage (After 3 Gossip Rounds)

Measured cumulative prompt + generation tokens across all rounds:

| Instance | Port | Prompt tokens | Gen tokens | Total | Running | Load share |
|----------|------|-------------:|----------:|---------:|:-------:|----------:|
| GPU 0 | 8000 | 1,565,270 | 81,260 | 1,646,530 | Yes | **28.5%** |
| GPU 1 | 8001 | 1,017,545 | 63,578 | 1,081,123 | Yes | 18.7% |
| GPU 2 | 8002 | 543,241 | 63,059 | 606,300 | Yes | 10.5% |
| GPU 3 | 8003 | 497,431 | 51,480 | 548,911 | Idle | 9.5% |
| GPU 4 | 8004 | 354,495 | 29,879 | 384,374 | Yes | 6.6% |
| GPU 5 | 8005 | 285,820 | 62,047 | 347,867 | Yes | 6.0% |
| GPU 6 | 8006 | 475,930 | 62,059 | 537,989 | Yes | 9.3% |
| GPU 7 | 8007 | 260,793 | 33,473 | 294,266 | Idle | 5.1% |
| **Total** | | **5,000,525** | **446,835** | **5,447,360** | | |

### 10.2 Load Imbalance

GPU 0 processed **6× more prompt tokens** than GPU 7 (1.57M vs 261K). This is
a direct consequence of round-robin worker assignment without load awareness.
Workers assigned to angles with larger corpus sections (e.g., insulin-protocols)
generate larger prompts and are disproportionately routed to lower-numbered
instances.

**Impact**: GPU 0 was the bottleneck in every round. Other GPUs sat idle waiting
for it to finish. The theoretical speedup from 8 GPUs is 8×, but the actual
speedup was closer to 4-5× due to this imbalance.

**Fix**: Load-aware routing that assigns workers to instances based on estimated
prompt size, not round-robin order.

### 10.3 Context Utilization — The Core Problem

Each individual worker request uses **50-80K prompt tokens** (corpus slice +
system prompt + gossip history from prior rounds). With 512K max context
configured (and 1M available on Kimi Linear), this means:

```
Actual context used per request:  50,000-80,000 tokens
Configured max context:          524,288 tokens (512K)
Kimi Linear native context:    1,048,576 tokens (1M)
Utilization vs configured:       10-15%
Utilization vs native:           5-8%
```

**This is the fundamental waste.** We are paying the H200 tax ($2.32/hr per GPU)
and the eager-mode speed penalty (21 tok/s vs 60-100 tok/s) for a 1M context
window that we fill to 5-8%. At this utilization, Qwen3.6-35B-A3B on H100 PCIe
($1.53/hr) with CUDA graphs (60-100 tok/s) would be strictly superior.

Kimi Linear is only justified if we **actually fill** the context window to
500K+ tokens per request.

### 10.4 Why Context Stays Small — The Gossip Architecture Gap

The current gossip protocol operates as follows:

```
Round 1: worker receives corpus_slice (~50K tokens) + system prompt (~2K)
         → generates synthesis (~16K tokens)
         → total context: ~68K tokens

Round 2: worker receives corpus_slice (~50K) + system prompt (~2K)
         + gossip_from_peers (~11 summaries × ~2K each = ~22K)
         → total context: ~74K tokens

Round 3: worker receives corpus_slice (~50K) + system prompt (~2K)
         + gossip_from_peers (~22K) + hive_memory (~5K)
         → total context: ~79K tokens
```

Context grows by only **~5-11K tokens per round** because the gossip payload is
compressed summaries, not full synthesis outputs. After 3 rounds, context has
grown from 68K to 79K — a **16% increase**. To reach 500K+ would require:

- **~40 gossip rounds** at the current 11K/round accumulation rate, OR
- **Full synthesis injection** instead of compressed summaries (each peer's full
  16K output × 11 peers = 176K per round — reaching 500K+ by round 3)

### 10.5 Deep Gossip Accumulation — Architecture for 1M Utilization

To justify Kimi Linear's 1M context, the gossip protocol must accumulate
substantially more context per round. Four design approaches:

**Approach A: Full-output gossip (aggressive)**

Instead of compressing peer outputs into ~2K summaries, inject full synthesis
outputs. Each peer contributes ~16K tokens per round.

```
Round 1: 50K corpus + 2K system = 52K context (same as today)
Round 2: 50K corpus + 2K system + 11×16K peer outputs = 228K context
Round 3: 50K corpus + 2K system + 11×32K accumulated = 404K context
Round 4: 50K corpus + 2K system + 11×48K accumulated = 580K context  ← 1M target zone
Round 5: 50K corpus + 2K system + 11×64K accumulated = 756K context
```

This fills 1M by round 5-6. Trade-off: massively more prompt tokens to prefill,
so each round takes much longer (11-24 min prefill at 1M). Prefix caching with
affinity routing is critical — without it, every round re-prefills everything.

**Approach B: Full-corpus workers (redistribute input)**

Give every worker the entire corpus (672K tokens) instead of a slice (50K).
Workers differentiate by their angle/perspective, not by which data they see.

```
Round 1: 672K corpus + 2K system = 674K context
Round 2: 672K corpus + 2K system + gossip = 696K-850K context
```

Immediately fills 64-81% of 1M. Workers see cross-domain connections that
slice-based workers physically cannot. Trade-off: prefill time for 672K tokens
is 7-16 minutes per request. Only viable with prefix caching (all workers
share the same corpus prefix, so instances 2-8 hit the cache after instance 1
prefills).

**Approach C: Progressive context growth (moderate)**

Start with sliced corpus (50K) for fast Round 1 results, then expand context
each round by injecting more raw corpus sections plus full peer outputs.

```
Round 1: 50K slice + 2K system = 52K (fast cold start)
Round 2: 50K slice + 100K additional corpus + 50K peer outputs = 202K
Round 3: 50K slice + 200K corpus + 100K peer outputs = 352K
Round 4: Full 672K corpus + 150K accumulated gossip = 824K
```

Balances speed (fast initial round) with depth (progressive context growth).
Prefix caching benefits compound as more of the corpus is shared.

**Approach D: Fewer workers, larger shares**

4 workers instead of 12, each getting 168K of corpus (672K / 4). After 3 rounds
of full-output gossip, each worker has ~500K+ context.

```
Round 1: 168K corpus + 2K system = 170K
Round 2: 168K corpus + 3×16K peer outputs = 216K
Round 3: 168K corpus + 3×32K accumulated = 264K
Round 4: 168K corpus + 3×48K accumulated = 312K
```

Slower to reach 1M (needs ~12 rounds), but each worker sees 3× more corpus from
the start. Reducing from 12 to 4 workers also eliminates load imbalance (4
workers on 8 GPUs = 2 idle GPUs, or 4 GPUs with affinity routing).

### 10.6 Prefix Cache Effectiveness Over Time

| Round | Prompt tokens queried | Cache hits | Hit rate | Notes |
|-------|----------------------:|----------:|---------:|-------|
| 1 (map) | ~600K (estimated) | 0 | 0% | Cold start, no prior prefix |
| 1 (gossip) | ~400K (estimated) | ~100K | ~25% | System prompt reuse |
| 2 | ~800K (estimated) | ~250K | ~31% | Partial prefix overlap from R1 |
| 3 (so far) | ~500K+ | ~260K+ | ~39% | Accidental affinity building up |

The hit rate is climbing because the round-robin router sometimes sends the same
worker back to the same instance (accidental affinity). With deliberate affinity
routing, round 2+ would see 80-90% hit rates — turning 11-24 min cold prefills
into <1 min warm prefills.

### 10.7 Information Gain Decay

| Round | Info gain | Δ from prior | Interpretation |
|-------|----------:|:------------:|----------------|
| 1 (gossip) | 0.836 | — | Workers learned substantially from peers |
| 2 (gossip) | 0.526 | −37% | Diminishing returns — workers converging |
| 3 (gossip) | 0.664 | **+26%** | Hive memory injection reversed the decay |

Info gain dropped 37% from round 1 to round 2, as expected from compressed
gossip summaries. But round 3 **reversed the trend** — info gain rose 26% back
to 0.664. The difference: round 3 injected hive memory (cross-pollination
insights from the queen's merge pass) into 11 of 12 workers. This added novel
cross-domain connections that compressed peer summaries alone could not provide.

**Implication for deep gossip architecture**: Hive memory injection is the
mechanism that sustains info gain across rounds. Pure peer-to-peer gossip
converges by round 2-3, but injecting synthesized cross-domain insights from a
coordinator (queen) re-opens divergent reasoning paths. A 1M context strategy
should alternate between peer gossip rounds and hive memory injection rounds
to maintain info gain above 0.5 across 5+ rounds.

---

## 11. Lessons Learned

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

9. **Round-robin routing causes severe load imbalance.** GPU 0 processed 6×
   more tokens than GPU 7 (1.57M vs 261K). Workers with larger corpus
   sections dominate lower-numbered instances. Load-aware routing based on
   estimated prompt size would equalize GPU utilization.

10. **Context utilization is the make-or-break metric for Kimi Linear.** At
    50-80K tokens per request (5-8% of 1M), we are paying the H200/eager-mode
    tax for nothing. If context stays below 262K, switch to Qwen3.6 on H100.
    Only commit to Kimi Linear when the architecture guarantees 500K+ fill.

11. **Gossip summaries are too compressed to drive context growth.** The current
    protocol compresses each peer's synthesis into ~2K tokens. Over 3 rounds,
    context grows from 68K to 79K — a 16% increase. Full-output gossip (16K
    per peer per round) would reach 500K+ by round 4-5.

12. **Accidental affinity still yields 39% prefix cache hits.** Even without
    deliberate routing, the round-robin pattern occasionally sends the same
    worker back to the same instance. The hit rate climbed from 0% (round 1)
    to 39% (after round 3). Deliberate affinity routing would push this to
    80-90%, cutting prefill time by 5-10× for warm rounds.

13. **Info gain decays fast with compressed gossip.** Info gain dropped from
    0.836 (round 1) to 0.526 (round 2) — a 37% decrease. Workers are
    converging prematurely because compressed summaries lose the detail needed
    to challenge and extend each other's reasoning. Full-output gossip would
    sustain higher info gain by preserving raw argumentation.

---

## 12. vLLM Throughput Optimization — V4 Pro on H200

### 12.1 The Problem

V4 Pro at TP=8 on H200 was running at ~15s per reasoning query with only
10-15 concurrent requests. For Flock evaluation (500+ queries) and creative
SQL-driven analysis (100+ deep queries), this throughput creates multi-hour
runtimes. The root cause: conservative default settings that over-provision
KV cache memory and disable hardware acceleration features.

### 12.2 Baseline Configuration (Conservative)

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --max-model-len 65536 \
    --kv-cache-dtype fp8 \
    --enforce-eager \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90
```

**Baseline throughput**: ~15s per query at concurrency=10. VRAM usage:
~135,889 MiB per GPU (94.5% of 143,771 MiB). Most VRAM allocated to KV
cache for 65K context that queries never filled — actual prompts used
5-10K tokens.

### 12.3 Optimized Configuration

```bash
vllm serve deepseek-ai/DeepSeek-V4-Pro \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.95 \
    --block-size 256 \
    --tokenizer-mode deepseek_v4
```

### 12.4 Optimization Breakdown

| Change | Before | After | Impact |
|--------|--------|-------|--------|
| `--max-model-len` | 65536 | 16384 | Frees ~75% of KV cache. Actual prompts are 5-10K tokens — 65K was massively over-provisioned. More cache blocks available → more concurrent requests in the batch scheduler. **Biggest single improvement.** |
| `--gpu-memory-utilization` | 0.90 | 0.95 | Frees ~40 GiB more across 8 GPUs for KV cache. At TP=8, per-GPU weight memory is only ~12.8 GiB, so 0.95 is safe — 5% headroom (7.2 GiB/GPU) is sufficient for PyTorch runtime allocations. |
| `--enforce-eager` | Enabled | **Removed** | Allows CUDA graph compilation. With max-model-len reduced to 16K, graph profiling VRAM fits within the freed memory. CUDA graphs eliminate Python overhead on the decode path — 2-3× faster decode throughput. Note: at 65K this caused OOM; at 16K it works. |
| `--enable-chunked-prefill` | Disabled | Enabled | Allows new requests to start prefilling while existing requests are in the decode phase. Without this, the scheduler waits for all decode steps to complete before admitting new prefills, creating idle bubbles. Improves latency under load. |
| `--block-size 256` | Default | 256 | Aligns KV cache blocks with V4 Pro's attention architecture (CSA/HCA layers use 256-token attention windows). Reduces internal fragmentation. |
| `--tokenizer-mode deepseek_v4` | Default | deepseek_v4 | Uses the architecture-specific tokenizer mode. Avoids fallback to slow Python tokenizer. |

### 12.5 Expected vs Measured Impact

| Metric | Before | After (expected) | Multiplier |
|--------|--------|-------------------|------------|
| Max concurrent requests | ~15 | ~50-60 | 3-4× |
| Per-query latency (8K gen) | ~15s | ~5-8s | 2-3× |
| Effective throughput (queries/min) | ~4 | ~12-15 | 3-4× |
| KV cache blocks available | N (65K budget) | ~4N (16K budget) | 4× |
| VRAM for KV cache | ~123 GiB total | ~130 GiB total | 1.06× |

### 12.6 When NOT to Use These Optimizations

- **If prompts exceed 16K tokens**: Increase `--max-model-len` accordingly.
  Kimi Linear worker synthesis uses 50-80K context — these optimizations are
  V4 Pro Flock-specific where prompts are known to be <16K.
- **If CUDA graph compilation OOMs**: Fall back to `--enforce-eager`. This
  happens when max-model-len is too large for available VRAM after weight
  loading. The threshold on H200 TP=8 is approximately 32K.
- **If prefix cache hit rate drops**: `--block-size 256` may reduce cache
  granularity for prompts that share very short common prefixes. Monitor
  `vllm:prefix_cache_hit_rate` and revert to default block size if needed.

### 12.7 Decision Framework: Context Length vs Throughput

The core trade-off: `max-model-len` directly controls how many concurrent
requests can be batched. For the creative Flock pipeline:

```
Throughput ∝ (available_KV_blocks) ∝ (1 / max_model_len)

At 65K max_model_len:  ~15 concurrent, ~4 queries/min
At 32K max_model_len:  ~30 concurrent, ~8 queries/min
At 16K max_model_len:  ~50 concurrent, ~15 queries/min
At  8K max_model_len:  ~80 concurrent, ~20 queries/min
```

Choose max-model-len as the smallest power of 2 that exceeds your longest
prompt + max_tokens. For Flock evaluation prompts (~3K prompt + 8K gen = 11K),
16384 is the right choice.

---

## 13. DeepSeek V4 Flash — Bulk Query Companion

### 13.1 Model Profile

| Property | Value |
|----------|-------|
| Full name | `deepseek-ai/DeepSeek-V4-Flash` |
| Architecture | Same as V4 Pro (MoE, CSA+HCA) but smaller |
| Weight size (FP8) | ~149 GB (46 safetensors shards) |
| Download time | ~2.5 min at 1+ GB/s on H200 NVMe |

### 13.2 Sizing for H200

At 149 GB FP8, Flash fits on TP=2 (75 GB/GPU) with ample room for KV
cache. It cannot run alongside V4 Pro (TP=8 occupies all 8 GPUs at
135 GB/GPU). Two deployment strategies:

**Phase-swap**: Kill V4 Pro → start Flash on TP=2 → run bulk queries →
kill Flash → restart V4 Pro for deep queries. Phase-swap latency is
dominated by model loading (~60s for Flash, ~5 min for V4 Pro).

**Single-model**: If V4 Pro throughput is sufficient after optimization
(Section 12), skip Flash entirely and run everything on V4 Pro. This
avoids phase-swap overhead and maximizes reasoning depth.

### 13.3 When to Use Flash vs V4 Pro

| Criterion | Use V4 Pro | Use Flash |
|-----------|-----------|-----------|
| Query requires deep reasoning | Yes | No |
| Query count > 500 | Consider Flash | Yes |
| Max depth is priority | Yes | No |
| Throughput is priority | No | Yes |
| Available time > 2 hours | Either | Either |
| Available time < 30 min | No | Yes |

---

## 14. Creative Flock Query Architecture

### 14.1 SQL-Driven Pattern Discovery

The creative Flock uses DuckDB analytical queries to discover patterns
in the ConditionStore, then sends those patterns to V4 Pro for deep
reasoning. This is a prototype for the intelligent orchestrator (Issue #264).

**SQL query types implemented**:

| Query Type | SQL Technique | Purpose |
|------------|---------------|---------|
| CONTRADICTION_MINE | Cross-join + keyword intersection + confidence divergence | Find claims that conflict across angles |
| OUTLIER_DETECT | Window functions (AVG, STDDEV, PERCENT_RANK, NTILE OVER PARTITION BY angle) | Find findings that deviate from angle norms |
| INFORMATION_DESERT | CTE angle pairs + LEFT JOIN cross-eval counts | Find angle pairs with zero cross-references |
| KEYWORD_CLUSTER | CASE-based substance tagging + GROUP_CONCAT + LIST aggregation | Create ad-hoc semantic clusters (cluster_id was unassigned) |
| CONFIDENCE_GRADIENT | PERCENT_RANK global + partitioned + CASE categorization | Classify findings into confidence tiers |
| ANGLE_HEALTH | Full aggregate dashboard (AVG, MIN, MAX, STDDEV, conditional SUM) | Orchestrator-level angle health report |
| SOURCE_CREDIBILITY | CASE URL pattern matching + trust aggregation | Assess reliability by source type |
| DOSE_MENTIONS | regexp_extract_all for dose/duration patterns | Find quantitative findings for interpolation |

### 14.2 LLM-Driven Reasoning Query Types

Beyond standard Flock (VALIDATE, VERIFY, ENRICH, GROUND, BRIDGE, CHALLENGE,
SYNTHESIZE), the creative Flock adds:

| Query Type | Model | Purpose |
|------------|-------|---------|
| AGGREGATE | V4 Pro (deep) | Cross-store strategic research planning — full corpus review |
| MECHANISM_CHAIN | V4 Pro (deep) | Link 3-5 findings into end-to-end mechanistic pathways |
| PRACTITIONER_SIM | V4 Pro (deep) | Role-play as specific practitioners to design protocols |
| COUNTERFACTUAL | V4 Pro (deep) | "If this finding were false, what else breaks?" |
| GAP_DETECT | V4 Pro (deep) | Identify critical missing questions per angle |
| DEVIL_ADVOCATE | V4 Pro (deep) | Construct strongest counter-arguments to high-confidence claims |
| DOSE_RESPONSE | V4 Pro (deep) | Interpolate dose-response curves from scattered mentions |
| RISK_CASCADE | V4 Pro (deep) | Map worst-case failure scenarios from high-risk findings |
| TEMPORAL_REASON | V4 Pro (deep) | Week-by-week effect timelines for specific protocols |
| PREDICTION_GEN | V4 Pro (deep) | Generate testable predictions the corpus implies but never states |

### 14.3 Trace Infrastructure

Every creative query produces a `row_type='trace'` row in the ConditionStore
containing: the SQL that discovered the pattern, finding IDs surfaced, the
prompt sent, the model response, confidence extracted, and wall-clock timing.
This creates full provenance for the orchestrator's decisions — a future
intelligent orchestrator can replay these traces to learn query selection
strategies.
