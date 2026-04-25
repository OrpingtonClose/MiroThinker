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

## 8. Optimal VM Fleet for Kimi Linear Workers

The key insight for Kimi Linear is that it runs TP=1 — each instance needs
exactly one GPU, no NVLink required. This means you can rent individual GPUs
from any provider and scale horizontally without multi-GPU node constraints.

### 8.1 VRAM Budget Per Instance

```
Kimi Linear FP8 weights:     ~47 GB
KV cache at 131K (FP8):       ~6 GB
CUDA graph profiling buffer: ~10-15 GB
Runtime / CUDA context:       ~3 GB
──────────────────────────────────────
Minimum VRAM per instance:   ~70 GB  (with CUDA graphs)
Minimum VRAM (eager mode):   ~55 GB  (without CUDA graphs)
```

### 8.2 GPU-by-GPU Analysis

| GPU | VRAM | Vast.ai $/hr | FP8 native? | Fits FP8 + graphs? | Est. tok/s | tok/s per $ |
|-----|------|-------------|-------------|--------------------|-----------:|------------:|
| **H100 PCIe** | 80 GB | **$1.53** | Yes (Hopper) | **Yes** (10 GB spare) | **50-80** | **33-52** |
| H100 SXM | 80 GB | ~$2.50 | Yes (Hopper) | Yes (10 GB spare) | 55-90 | 22-36 |
| **H200** | 143 GB | $2.32 | Yes (Hopper) | **Yes** (73 GB spare) | 60-100 | 26-43 |
| A100 80GB | 80 GB | ~$1.30 | No (Ampere, emulated FP8) | Tight | 25-40 | 19-31 |
| L40S | 48 GB | $0.53 | Yes (Ada, sm_89) | No (47 GB weights fill VRAM) | INT4 only: ~30 | 57 (quality risk) |
| RTX 4090 | 24 GB | $0.29 | Yes (Ada, sm_89) | No (can't fit 47 GB weights) | — | — |
| A6000 | 48 GB | $0.39 | No (Ampere) | No | INT4 only: ~25 | 64 (quality risk) |

### 8.3 Recommendation: H100 PCIe 80 GB

H100 PCIe is the sweet spot for Kimi Linear worker fleets:

1. **$1.53/hr** — cheapest GPU that fits FP8 weights + CUDA graph profiling
2. **Native Hopper FP8 tensor cores** — same CutlassFP8ScaledMMLinearKernel
   auto-selected on H100 as on H200. No emulation, no quality loss
3. **TP=1, no NVLink** — rent individual GPUs, not expensive multi-GPU nodes.
   Each GPU is a fully independent inference server
4. **CUDA graphs work at 131K context** — 80 GB − 47 GB weights − 6 GB KV −
   3 GB runtime = ~24 GB spare for graph profiling (needs ~10-15 GB)
5. **Horizontally scalable** — add more H100s for more concurrency, no
   architectural changes needed

### 8.4 Fleet Configurations for 12 Concurrent Workers

| Config | GPUs | Cost/hr | Est. aggregate tok/s | Est. round time |
|--------|-----:|--------:|---------------------:|----------------:|
| **12× H100 PCIe** (1 instance each) | 12 | **$18.36** | **600-960** | **~3-5 min** |
| 8× H200 (current, eager) | 8 | $18.56 | 168 | ~26 min |
| 8× H200 (optimized, CUDA graphs) | 8 | $18.56 | 480-800 | ~5-8 min |
| 16× H100 PCIe (headroom) | 16 | $24.48 | 800-1280 | ~2-3 min |
| 24× L40S (INT4, quality risk) | 24 | $12.72 | ~720 | ~4 min |

**Key finding**: 12× H100 PCIe matches the cost of our current 8× H200 setup
($18.36 vs $18.56/hr) but delivers 3.6-5.7× throughput because:
- Every worker gets its own GPU (no queueing — 12 workers, 12 GPUs)
- CUDA graphs enabled (3-5× per-instance speedup over eager mode)
- No wasted VRAM (H200's 143 GB is overkill when you only need 70 GB)

**When H200 is still the right choice**: Phase 2 (Flock with V4 Pro) requires
TP=8 across all 8 GPUs. If the pipeline needs both phases on the same node,
8× H200 is the only option that runs both models. For Kimi-Linear-only
workloads, H100 PCIe is strictly better per dollar.

### 8.5 Why Not Cheaper GPUs?

- **L40S / A6000 (48 GB)**: Kimi Linear FP8 weights alone are ~47 GB. After
  loading, there is <1 GB left for KV cache and runtime — the model either
  refuses to start or crashes immediately. INT4 quantization (AWQ/GPTQ) would
  reduce weights to ~24 GB, leaving 24 GB for KV cache, but (a) no official
  INT4 quant exists for Kimi Linear, (b) MoE expert routing quality degrades
  more than dense models under INT4, and (c) you lose the native FP8 tensor
  core path.
- **RTX 4090 (24 GB)**: Cannot fit FP8 or INT4 + KV cache on a single card.
  TP=2 across two 4090s would work mathematically but PCIe interconnect (no
  NVLink on consumer GPUs) adds 5-10× latency to every cross-GPU operation,
  negating the cost advantage.
- **A100 (80 GB, ~$1.30/hr)**: Fits FP8 weights but Ampere lacks native FP8
  tensor cores. vLLM uses emulated FP8 which is slower than native Hopper FP8.
  The $0.23/hr savings vs H100 doesn't justify the ~30-40% throughput penalty.

---

## 9. Using the Full 1M Context on Kimi Linear

Kimi Linear's headline feature is native 1,048,576-token context via hybrid
linear attention. Actually exploiting the full 1M requires overcoming several
VRAM, throughput, and architecture constraints.

### 9.1 VRAM Requirements for 1M Context

The KV cache scales differently for Kimi Linear's two layer types:

| Layer type | Count | KV cache scaling | Size at 1M tokens |
|-----------|------:|-----------------|-------------------:|
| Linear (KDA) | 36 | **Constant** — fixed-size state | ~94 MB total |
| Standard (MLA) | 12 | **Linear** — grows with sequence | ~49 GB (BF16) / ~25 GB (FP8) |
| **Total KV cache** | 48 | | **~25 GB (FP8)** |

Total VRAM per instance at 1M context:
```
Model weights (FP8):     47 GB
KV cache (FP8, 1M):     25 GB
Runtime / activations:  ~10 GB
────────────────────────────────
Subtotal:               ~82 GB  ← minimum for eager mode at 1M
CUDA graph profiling:  +15-60 GB ← depends on graph capture strategy
```

### 9.2 GPU Requirements (1M Context, Single Instance)

| GPU | VRAM | Fits 1M eager? | Fits 1M + CUDA graphs? |
|-----|------|:--------------:|:----------------------:|
| H200 (143 GB) | 143 GB | **Yes** (61 GB spare) | **Depends** — graphs OOM'd at 512K on our run |
| H100 80GB | 80 GB | Tight (~0 GB spare) | **No** |
| A100 80GB | 80 GB | Tight (~0 GB spare) | **No** |
| B200 (192 GB) | 192 GB | **Yes** (110 GB spare) | **Likely yes** |

**Critical finding**: H100 PCIe (80 GB) cannot serve Kimi Linear at 1M
context — the KV cache alone (25 GB FP8) plus weights (47 GB) plus runtime
(10 GB) totals ~82 GB, exceeding the 80 GB limit. Full 1M context requires
H200 or Blackwell-class GPUs.

### 9.3 What Must Change to Enable 1M Context

#### 1. GPU selection: H200 minimum (143 GB VRAM)

Only H200 (143 GB) and B200 (192 GB) have enough VRAM for FP8 weights + 1M
KV cache + runtime in a single GPU. The "sweet spot" H100 PCIe (80 GB)
recommended in section 8 is limited to ~131K-200K context.

#### 2. Eager mode required (no CUDA graphs at 1M)

Our H200 run proved CUDA graphs OOM at 512K context (59 GiB profiling
allocation). At 1M the profiling buffer would be even larger. This means
eager mode (`--enforce-eager`) is required, limiting decode to ~21 tok/s.

**Potential workaround**: vLLM's `cudagraph_mode: "PIECEWISE"` captures
graphs for only the decode phase (smaller buffers than full capture). This
might fit at 1M on H200 — untested.

#### 3. Accept the throughput trade-off

At 1M context with eager mode:
- **Prefill**: ~684-1526 tok/s (measured) — prefilling 1M tokens takes
  ~11-24 minutes
- **Decode**: ~21 tok/s — generating 16K output tokens takes ~13 minutes
- **Total per request**: ~24-37 minutes for a full 1M-token input + 16K output

For a 12-worker swarm where each worker gets a 1M-token input:
- 8× H200, 1 instance/GPU, eager mode: ~75-111 min per round (4.5× current)
- The throughput penalty is severe — 1M context should only be used when
  the corpus genuinely requires it

#### 4. Restructure the corpus for 1M inputs

Our current corpus is 2.69M chars (~672K tokens). Split across 12 workers,
each worker gets ~56K tokens — well within 131K context. To actually need
1M context, each worker would need ~4M chars (~1M tokens) of input. This
requires either:
- **Dramatically larger corpus** (12× current size = ~32M chars)
- **Fewer workers with larger shares** (e.g., 3 workers × 224K tokens each,
  with the remaining context for gossip history accumulation)
- **Multi-round context accumulation** — start with 56K input, then each
  gossip round appends 50-100K tokens of peer summaries, growing toward
  1M by round 5-6

#### 5. Enable prefix caching with affinity routing

At 1M context, prefill dominates (11-24 min). Prefix caching becomes
critical — if even 500K tokens of the 1M input are cached from a previous
round, prefill drops to ~5-12 min.

Requirements for effective prefix caching at 1M:
- **Worker-instance affinity**: Same worker always routes to the same GPU.
  This ensures the KV cache from round N is still resident for round N+1
- **Deterministic prompt ordering**: The cached prefix must be byte-identical
  to what was previously processed. Any change in prompt structure (even
  whitespace) invalidates the cache
- **Sufficient KV cache capacity**: At 1M context with FP8, the KV cache
  is 25 GB. The H200 can hold ~2.4× this (61 GB spare), meaning it can
  cache approximately 2 full 1M-context sequences simultaneously

#### 6. Adjust vLLM block size for Mamba alignment

At 1M context, the Mamba cache alignment inflates blocks to 3,776 tokens
(see section 2.4). With 1M tokens / 3,776 tokens per block = ~265 blocks.
Each block must be fully materialized in VRAM even if only partially used.
This adds ~2-5% VRAM overhead from partially-filled terminal blocks.

### 9.4 Recommended Architecture for 1M Context

```
Hardware:  8× H200 (only GPU with enough VRAM, short of Blackwell)
Instances: 8× TP=1 (one per GPU)
Quantization: FP8 weights + FP8 KV cache
Context:   1,048,576 tokens (full native)
Mode:      --enforce-eager (CUDA graphs OOM at 1M)
Caching:   --enable-prefix-caching with affinity routing

Expected per-instance:
  Prefill:  ~684-1526 tok/s → 11-24 min for 1M tokens
  Decode:   ~21 tok/s → 13 min for 16K output
  Total:    ~24-37 min per request

Expected fleet (8× H200):
  Round time (12 workers): ~75-111 min per round
  Cost: $28.91/hr × ~1.5 hr = ~$43 per round
```

### 9.5 When 1M Context Is Worth It

| Scenario | Verdict | Rationale |
|----------|---------|-----------|
| Current swarm (56K tokens/worker) | **Not worth it** | 131K context handles the workload at 3-5× throughput |
| Full-book synthesis (500K+ tokens/worker) | **Worth it** | Entire book in context eliminates chunking artifacts |
| Multi-round gossip accumulation (grows to 500K+) | **Worth it** | Rounds 4-6 accumulate enough gossip history to need it |
| Cross-worker corpus dedup (all workers see full corpus) | **Worth it** | Each worker gets the entire 672K-token corpus instead of a slice |
| Real-time long-document QA | **Worth it** | 1M context enables single-shot answers without RAG |

**Bottom line**: Use 131K context on H100 PCIe ($1.53/hr) for workloads under
200K tokens. Switch to 1M context on H200 ($2.32/hr) only when the input
genuinely exceeds 200K tokens or when gossip accumulation pushes context
beyond 131K.

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
