# H200 Swarm Test — Bodybuilding Cycle Protocol Synthesis

Single-GPU test configuration for the gossip swarm on 1×H200 (141GB VRAM).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   1× H200 GPU                        │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │  vLLM (Qwen3.5-32B-abliterated, FP16)     │     │
│  │  64GB weights + 77GB KV cache               │     │
│  │  Continuous batching — 8 workers in parallel │     │
│  └──────────┬──────────────────────────────────┘     │
│             │ OpenAI-compatible API (:8000/v1)        │
└─────────────┼───────────────────────────────────────┘
              │
   ┌──────────▼──────────┐
   │  GossipSwarm Engine  │
   │                      │
   │  8 angles:           │
   │  ├─ Insulin & GH     │──── Phase 2-4 backbone
   │  ├─ Test & Tren      │──── Phase 1-4 foundation
   │  ├─ Ancillaries      │──── All phases safety
   │  ├─ Oral compounds   │──── Phase 1,3,4
   │  ├─ Boldenone        │──── Phase 2-4
   │  ├─ Practitioner     │──── Real-world grounding
   │  ├─ Micronutrients   │──── Hidden interactions ★
   │  └─ Ramping strategy │──── 4-phase structure
   │                      │
   │  3 gossip rounds     │
   │  + serendipity       │
   │  + queen merge       │
   └──────────────────────┘
```

## Quick Start

### 1. Launch vLLM (Terminal 1)

```bash
# Default: Qwen3.5-32B abliterated, FP16, port 8000
./launch_vllm.sh

# Or specify model:
VLLM_MODEL=huihui-ai/Llama-3.3-70B-Instruct-abliterated \
VLLM_DTYPE=float8 \
./launch_vllm.sh
```

### 2. Enrich corpus + run swarm (Terminal 2)

```bash
# Enrich from web + run swarm in one go:
python run_swarm_test.py --enrich --output-dir results/

# Or with an existing corpus file:
python run_swarm_test.py --corpus my_corpus.txt --output-dir results/

# Or from a pre-enriched DuckDB:
python run_swarm_test.py --db enriched.duckdb --output-dir results/
```

### 3. Enrichment only (no swarm)

```bash
# Dry run — see what would be gathered:
python enrich_corpus.py --dry-run

# Full enrichment to DuckDB:
python enrich_corpus.py --db enriched.duckdb

# Export as JSON:
python enrich_corpus.py --db enriched.duckdb --export corpus.json
```

## Environment Variables

### vLLM Server

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `huihui-ai/Qwen3.5-32B-abliterated` | HuggingFace model ID |
| `VLLM_PORT` | `8000` | API server port |
| `VLLM_MAX_MODEL_LEN` | `32768` | Max context length |
| `VLLM_GPU_UTIL` | `0.92` | GPU memory utilization (0.0-1.0) |
| `VLLM_DTYPE` | `auto` | Data type: auto, float16, bfloat16, float8 |
| `HF_TOKEN` | — | HuggingFace token for gated models |

### Swarm Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SWARM_API_BASE` | `http://localhost:8000/v1` | vLLM endpoint |
| `SWARM_WORKER_MODEL` | Model served by vLLM | Model name for workers |
| `SWARM_QUEEN_MODEL` | Same as worker | Model name for queen |
| `SWARM_SERENDIPITY_MODEL` | Same as worker | Model for serendipity |
| `SWARM_MAX_WORKERS` | `6` | Concurrent workers |
| `SWARM_GOSSIP_ROUNDS` | `3` | Gossip refinement rounds |

### Search API Keys (optional — enrichment works without them)

| Variable | Service | Tier |
|----------|---------|------|
| `BRAVE_API_KEY` | Brave Search | 1 (uncensored) |
| `JINA_API_KEY` | Jina Reader (full text) | 2 (extraction) |
| `NCBI_API_KEY` | PubMed (higher limits) | Academic |

DuckDuckGo and forum search work without any API keys.

## Swarm Angles

8 angles covering the full compound matrix + cross-cutting layers:

| # | Angle | Purpose |
|---|-------|---------|
| 1 | Insulin & GH — Milos framework | Timing backbone for the entire protocol |
| 2 | Testosterone & Trenbolone | Pharmacokinetics, dose-response, interactions |
| 3 | Ancillaries & health markers | Bloodwork, safety, side effect management |
| 4 | Oral compounds (tbol, LGD, actovegin) | Hepatic load, receptor selectivity, recovery |
| 5 | Boldenone & EQ interactions | EPO, appetite, hematocrit compounding |
| 6 | Practitioner protocols | Real-world dosing, adjustments, experience |
| 7 | **Micronutrient interactions** ★ | Hidden mineral/vitamin dependencies |
| 8 | Ramping & periodization | 4-phase structure, transition criteria |

Angle 7 (Micronutrients) is the serendipity accelerator. It hunts for
interactions like tren→iron depletion, Mg→insulin sensitivity, Zn→aromatase,
B vitamins→liver methylation that compound-specific workers miss.

## Test Configurations

### Test 1: Baseline (32B FP16, single model)

```bash
VLLM_MODEL=huihui-ai/Qwen3.5-32B-abliterated \
./launch_vllm.sh
```

64GB weights, 77GB KV cache. Maximum context per worker.
This is the baseline — 8 workers sharing one model via continuous batching.

### Test 2: 70B FP8

```bash
VLLM_MODEL=huihui-ai/Llama-3.3-70B-Instruct-abliterated \
VLLM_DTYPE=float8 \
./launch_vllm.sh
```

70GB weights (FP8), 71GB KV cache. Deeper reasoning, slightly less context.
Compare quality of cross-domain connections vs 32B.

### Test 3: A/B epistemic diversity test

Run Test 1 (Qwen3.5-32B) and Test 2 (Llama-3.3-70B) sequentially on the
same corpus with the same query. Compare outputs to measure whether different
model families find genuinely different insights.

## Output

Each run produces in the output directory:

- `user_report_TIMESTAMP.md` — Concise protocol narrative (queen merge)
- `knowledge_report_TIMESTAMP.md` — Full structured report preserving all findings
- `metrics_TIMESTAMP.json` — Swarm telemetry (timing, LLM calls, convergence)
- `worker_N_TIMESTAMP.md` — Individual worker analyses
- `serendipity_TIMESTAMP.md` — Cross-angle surprise connections

## Scaling to 8×H200

After validating on 1×H200, scale to 8 GPUs:

1. Launch 8 vLLM instances (one per GPU, different models for epistemic diversity)
2. Update `SWARM_API_BASE` to a load balancer or list of endpoints
3. Set `SWARM_MAX_WORKERS=48` (6 per GPU × 8 GPUs)
4. The swarm engine handles the rest — same gossip protocol, more perspectives
