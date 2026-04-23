# Employment List Execution Plan

Concrete execution plan for the 5-wave model evaluation using MiroThinker's swarm engine.
This document is the "go/no-go" checkpoint — review and approve before execution begins.

**Runner script:** `scripts/h200_test/run_employment.py`
**Corpus:** `scripts/h200_test/test_corpus.txt` (Milos corpus, 128 lines, 12 KB)
**Output directory:** `scripts/h200_test/employment_results/{provider}/{topic_id}/`
**Store:** Each run gets its own DuckDB store (findings isolated per topic per model)

---

## Pre-Flight Checklist

| Check | Status | Notes |
|---|---|---|
| DeepSeek API key | ✓ available | `DEEPSEEK_API_KEY` — Wave 1 primary worker |
| Google API key | ✓ available | `GOOGLE_API_KEY` — Gemini Flash/Pro |
| xAI API key | ✓ available | `XAI_API_KEY` — Grok Fast |
| Mistral API key | ✓ available | `MISTRAL_API_KEY` — Mistral Large + ministral |
| Moonshot API key | ✓ available | `KIMI_API_KEY` — Kimi K2.5 |
| OpenAI API key | ✓ available | `OPENAI_API_KEY` — GPT-4.1-mini/nano |
| Groq API key | ✓ available | `GROQ_API_KEY` — Llama-3.3-70b, qwen3-32b |
| Anthropic API key | ✓ available | `ANTHROPIC_API_KEY` — Claude Opus (Wave 3 orchestrator) |
| OpenRouter API key | ✓ available | `OPENROUTER_API_KEY` — fallback/free models |
| Venice API key | ✓ available | `VENICE_API_KEY` — Wave 5 proxy tests |
| Swarm engine | ✓ tested | A1 test run dispatched 8 workers, extracted findings, DeepSeek API confirmed working |
| Corpus | ✓ exists | 128 lines, covers insulin, hematology, GH/IGF-1, micronutrients, trenbolone |
| DuckDB | ✓ available | ConditionStore per run, no shared state |

---

## Wave 1: Primary Coverage

**Goal:** Establish baseline findings corpus across all 48 disaggregated topics.

### What Runs

- **48 swarm runs**, one per topic (A1–A8, B1–B10, C1–C6, D1–D9, E1–E10, F1–F9, G1–G5, H1–H5)
- Each run: angle detection → 8 parallel workers × up to 3 waves → finding extraction → report
- Same corpus for all runs (Milos test_corpus.txt, 12 KB)
- Different query per topic (topic-specific research question)

### Model Assignment

| Role | Model | Provider | API Base | Cost |
|---|---|---|---|---|
| Workers + Orchestrator | `deepseek-chat` (V3.2) | DeepSeek native | `https://api.deepseek.com/v1` | $0.27/$1.10 per M tokens |

**Note:** Wave 1 uses a single model for everything (workers AND orchestrator). This is the simplest
configuration and establishes the DeepSeek V3.2 baseline. Wave 3 tests the orchestrator split.

### Rate Limiting & Concurrency

| Parameter | Value | Rationale |
|---|---|---|
| Max concurrent topics | 6 | DeepSeek rate limit ~60 RPM; each topic fires ~10-15 calls (angle detection + 8 workers + finding extraction); 6 concurrent = ~90 calls/min, stays under burst with headroom |
| Workers per topic | 8 | One per detected angle (engine default) |
| Max waves per topic | 3 | Convergence threshold = 5 new findings/wave |
| Timeout per worker | 600s | Default — DeepSeek typically responds in 10-30s |

### Execution Sequence

```
Batch 1 (6 parallel): A1, A2, A3, A4, A5, A6            → ~20 min
Batch 2 (6 parallel): A7, A8, B1, B2, B3, B4            → ~20 min
Batch 3 (6 parallel): B5, B6, B7, B8, B9, B10           → ~20 min
Batch 4 (6 parallel): C1, C2, C3, C4, C5, C6            → ~20 min
Batch 5 (6 parallel): D1, D2, D3, D4, D5, D6            → ~20 min
Batch 6 (6 parallel): D7, D8, D9, E1, E2, E3            → ~20 min
Batch 7 (6 parallel): E4, E5, E6, E7, E8, E9            → ~20 min
Batch 8 (6 parallel): E10, F1, F2, F3, F4, F5           → ~20 min
Batch 9 (6 parallel): F6, F7, F8, F9, G1, G2            → ~20 min
Batch 10 (4 topics):  G3, G4, G5, H1                    → ~15 min
Batch 11 (4 topics):  H2, H3, H4, H5                    → ~15 min
```

**NOTE:** Batching is handled by `asyncio.Semaphore(6)` in the runner — topics are
all submitted concurrently and the semaphore gates actual execution. No manual batching
needed. The sequence above is the expected execution order.

### Expected Per-Topic Output

Each topic produces in `employment_results/deepseek/{topic_id}/`:
- `store.duckdb` — ConditionStore with all findings, transcripts, angles
- `report_{timestamp}.md` — synthesized research report
- `metrics_{timestamp}.json` — structured run metrics

### Cost Estimate

| Component | Calculation | Cost |
|---|---|---|
| Angle detection | 48 topics × ~500 tokens prompt × $0.27/M | ~$0.006 |
| Worker calls | 48 topics × 8 workers × 3 waves × ~4K tokens out × $1.10/M | ~$5.10 |
| Worker inputs | 48 × 8 × 3 × ~3K tokens in × $0.27/M | ~$0.93 |
| Finding extraction | 48 × 3 waves × ~2K tokens × $1.10/M | ~$0.32 |
| Report generation | 48 × ~8K tokens × $1.10/M | ~$0.42 |
| **Total Wave 1** | | **~$7** |

### Timeline: ~3.5 hours

### Command

```bash
cd /home/ubuntu/repos/MiroThinker
python scripts/h200_test/run_employment.py --wave 1 --max-concurrent 6
```

### Success Criteria

- ≥ 44/48 topics complete (OK status) — up to 4 allowed to ERROR (transient API issues)
- Average ≥ 20 findings per topic (960+ total findings across all topics)
- No topics return 0 findings (would indicate censorship or API format failure)
- All 8 clusters have ≥ 1 successful topic

---

## Wave 2: Cross-Validation

**Goal:** Run high-serendipity topics with alternative models to detect model-specific blind spots.

### What Runs

24 additional swarm runs: 4 provider groups × 6 topics each.

### Model Assignments

| Group | Topics | Model | Provider | Why |
|---|---|---|---|---|
| 2A | A1, B1, B4, C2, D6, F9 | `gemini-2.5-flash` | Google | Cheap, 1M context — does longer context improve serendipity? |
| 2B | A1, B1, B4, C2, D6, F9 | `grok-4-1-fast-non-reasoning` | xAI | Uncensored, 2M context, fast — different reasoning style |
| 2C | A1, A4, B3, D3, F7, H4 | `mistral-large-latest` | Mistral | Strong reasoning, UNCENSORED on research-framed probes |
| 2D | A1, B9, D8, F5, G2, H5 | `kimi-k2.5` | Moonshot | 262K context, Chinese medical literature training data |

**Topic selection rationale:**
- Groups 2A/2B share topics with highest serendipity bridge counts (B4 has 3 bridges, F9 spans diabetes↔bodybuilding)
- Group 2C targets mechanism-heavy topics (AR binding, mTOR, myostatin, cancer biology)
- Group 2D targets clinical topics (neuropsychiatric, secretagogues, PCT, LVH, aging)
- A1 appears in all groups as the universal baseline comparison point

### Rate Limiting

| Provider | Rate Limit | Concurrent Topics |
|---|---|---|
| Google (Gemini) | ~60 RPM | 4 |
| xAI (Grok) | ~60 RPM | 4 |
| Mistral | ~30 RPM | 3 |
| Moonshot (Kimi) | ~30 RPM (needs temperature=1.0) | 3 |

Groups 2A–2D can run in parallel (different providers, no shared rate limit).

### Cost Estimate

| Group | Cost/Run | Runs | Subtotal |
|---|---|---|---|
| 2A (Gemini Flash) | ~$0.05 | 6 | ~$0.30 |
| 2B (Grok Fast) | ~$0.10 | 6 | ~$0.60 |
| 2C (Mistral Large) | ~$0.80 | 6 | ~$4.80 |
| 2D (Kimi K2.5) | ~$0.50 | 6 | ~$3.00 |
| **Total Wave 2** | | **24** | **~$8.70** |

### Timeline: ~1.5 hours (all 4 providers in parallel)

### Command

```bash
python scripts/h200_test/run_employment.py --wave 2
```

### Success Criteria

- ≥ 20/24 topics complete
- At least 1 model produces ≥ 10% more findings than DeepSeek on the same topic (novel blind spot found)
- A1 findings across all 5 models (DeepSeek + 4 alternatives) show detectable variation

### Analysis Output

After Wave 2, generate a cross-model comparison table:

```
Topic | DeepSeek Findings | Gemini Findings | Grok Findings | Mistral Findings | Kimi Findings
A1    | (from Wave 1)     | (from 2A)       | (from 2B)     | —                | (from 2D)
B1    | (from Wave 1)     | (from 2A)       | (from 2B)     | —                | —
B4    | (from Wave 1)     | (from 2A)       | (from 2B)     | —                | —
...
```

---

## Wave 3: Orchestrator Comparison

**Goal:** Isolate the orchestrator's impact. Same topic (A1), same workers (DeepSeek), different orchestrator model.

### What Runs

4 runs of A1, each with a different model handling orchestrator duties (angle detection, finding extraction, report generation).

| Run | Orchestrator Model | Workers | Notes |
|---|---|---|---|
| 3A | `deepseek-chat` (V3.2) | `deepseek-chat` | Single-model baseline (same as Wave 1 A1) |
| 3B | `gemini-2.5-pro` | `deepseek-chat` | 1M context orchestrator — can it hold more findings? |
| 3C | `grok-4-1-fast-non-reasoning` | `deepseek-chat` | Cheap/fast orchestrator — quality floor test |
| 3D | `gpt-4.1-mini` | `deepseek-chat` | OpenAI orchestrator — different angle detection style |

**Note:** Claude Opus would be the ideal orchestrator candidate (FULL on meta-reasoning), but
the current `complete_fn` uses OpenAI-compatible API format. Claude needs its own `/v1/messages`
adapter. Options:
1. Route Claude through OpenRouter (OPENROUTER_API_KEY available) — adds a proxy hop
2. Write a 20-line Anthropic adapter in `complete_fn` — preferred, direct API
3. Skip Claude in Wave 3, test in a dedicated Wave 3b after writing the adapter

**Recommendation:** Add Claude via OpenRouter for Wave 3 as run 3E, then compare with native
adapter in Phase 3 role-pair tests. OpenRouter's censorship may differ from native Anthropic.

### Implementation Detail

Wave 3 requires `MCPSwarmConfig.model_map` to split orchestrator from workers:

```python
# Run 3B: Gemini orchestrator + DeepSeek workers
config = MCPSwarmConfig(
    api_base="https://api.deepseek.com/v1",
    model="deepseek-chat",  # workers
    model_map={
        "__orchestrator__": "gemini-2.5-pro",  # orchestrator
    },
)
```

**Current swarm engine limitation:** `model_map` routes by angle name, not by role. The
`__orchestrator__` and `__report__` special keys are documented in MCPSwarmConfig but their
support needs verification in `mcp_engine.py`. If not implemented, Wave 3 runs as single-model
(orchestrator = workers = same model), which still tests different models for orchestration
but doesn't isolate the role split.

### Cost Estimate

| Run | Orchestrator Cost | Worker Cost | Total |
|---|---|---|---|
| 3A (DeepSeek single) | included | ~$0.30 | ~$0.30 |
| 3B (Gemini Pro orch) | ~$0.50 | ~$0.30 | ~$0.80 |
| 3C (Grok Fast orch) | ~$0.05 | ~$0.30 | ~$0.35 |
| 3D (GPT-4.1-mini orch) | ~$0.15 | ~$0.30 | ~$0.45 |
| **Total Wave 3** | | | **~$1.90** |

### Timeline: ~30 minutes (sequential — all use same workers, testing orchestrator quality)

### Command

```bash
python scripts/h200_test/run_employment.py --wave 3
```

### Success Criteria

- All 4 runs complete
- Angle detection varies across orchestrators (different models detect different research angles)
- Report quality difference is detectable (score 1-5 manually)

---

## Wave 4: Flock Speed Benchmark

**Goal:** Find the fastest model for high-volume relevance judgments (the Flock driver role).

### What Runs

6 models × 50 relevance judgment calls each = 300 API calls total.

Each call:
- Input: "Is [finding] relevant to [angle]? Answer YES/NO with one-sentence reason."
- ~100 tokens input, ~50 tokens output
- Measure: latency (ms), throughput (tok/s)

### Models

| Model | Provider | Expected tok/s | Cost/M out | Notes |
|---|---|---|---|---|
| `ministral-3b-latest` | Mistral | ~278 | $0.04 | Price floor |
| `ministral-8b-latest` | Mistral | ~181 | $0.08 | Slightly better reasoning |
| `llama-3.1-8b-instant` | Groq | ~179 | $0.05 | Groq hardware speed |
| `qwen3-32b` | Groq | ~343 | $0.20 | Best Groq reasoning+speed |
| `llama-4-scout-17b-16e-instruct` | Groq | ~162 | $0.11 | MoE speed test |
| `gpt-4.1-nano` | OpenAI | varies | $0.10 | OpenAI cheapest, UNCENSORED |

### Implementation

Wave 4 is NOT a swarm run — it's a standalone benchmark script. Needs a separate function
in `run_employment.py` that:
1. Pulls 50 (finding, angle) pairs from Wave 1 results (DuckDB stores)
2. Sends each as a single completion call to each model
3. Measures wall-clock time and tokens returned
4. Outputs a comparison table

### Cost Estimate

300 calls × ~50 tokens out × $0.10/M avg = **~$0.002** (essentially free)

### Timeline: ~10 minutes

### Success Criteria

- All 6 models respond (no errors)
- Throughput ranking established
- ≥ 1 model achieves > 200 tok/s
- YES/NO accuracy ≥ 80% (spot-check 10 judgments manually)

---

## Wave 5: Venice Proxy Quality Test

**Goal:** Determine if Venice's uncensoring proxy degrades model quality.

### What Runs

6 runs of A1 (Milos Insulin baseline) through Venice-proxied models that are CENSORED natively
but UNCENSORED through Venice.

| Model (via Venice) | Native Censorship Status | Venice Endpoint |
|---|---|---|
| `claude-opus-4-7` | LIMITED (refuses direct PED) | `https://api.venice.ai/api/v1` |
| `gpt-5.4-mini` | LIMITED | `https://api.venice.ai/api/v1` |
| `kimi-k2.6` | LIMITED | `https://api.venice.ai/api/v1` |
| `glm-5` | LIMITED | `https://api.venice.ai/api/v1` |
| `glm-5-turbo` | LIMITED | `https://api.venice.ai/api/v1` |
| `hermes-3-llama-3.1-405b` | N/A (Venice-native) | `https://api.venice.ai/api/v1` |

### Venice API Configuration

```python
ProviderConfig(
    api_base="https://api.venice.ai/api/v1",
    api_key_env="VENICE_API_KEY",
    model="venice/{model_name}",
)
```

**Note:** Venice model names may differ from native names. Need to verify exact model IDs
from Venice's model listing before execution.

### Cost Estimate

6 runs × ~$0.10 (Venice pricing is flat, low) = **~$0.60**

### Timeline: ~1 hour (sequential — Venice may have lower rate limits)

### Success Criteria

- Venice Claude produces findings comparable to native Claude (meta-reasoning quality)
- Venice hermes-3-405b produces findings comparable to DeepSeek V3.2
- No quality degradation > 20% (measured by findings count and report quality score)

---

## Grand Summary

| Wave | Runs | Cost | Time | Dependency |
|---|---|---|---|---|
| 1 — Primary Coverage | 48 | ~$7 | ~3.5h | None |
| 2 — Cross-Validation | 24 | ~$9 | ~1.5h | None (can run in parallel with Wave 1 on different providers) |
| 3 — Orchestrator Comparison | 4 | ~$2 | ~30m | Wave 1 A1 as baseline reference |
| 4 — Flock Benchmark | 300 calls | ~$0.002 | ~10m | Wave 1 findings (needs finding/angle pairs from DuckDB) |
| 5 — Venice Proxy | 6 | ~$0.60 | ~1h | None |
| **Total** | **82 runs + 300 calls** | **~$19** | **~6.5h wall clock** | |

**Optimal execution order (minimizing wall clock):**

```
Hour 0-3.5:   Wave 1 (DeepSeek, 48 topics)
              ‖ parallel ‖
Hour 0-1.5:   Wave 2 (Gemini/Grok/Mistral/Kimi, 24 topics on different providers)
              ‖ parallel ‖
Hour 0-1:     Wave 5 (Venice proxy, 6 topics)

Hour 3.5-4:   Wave 3 (orchestrator comparison, needs Wave 1 A1 done)
Hour 4-4.2:   Wave 4 (flock benchmark, needs Wave 1 findings)

Total wall clock: ~4.5 hours (with full parallelism)
```

**Note:** Waves 1, 2, and 5 can all start simultaneously since they use different API
providers and don't share rate limits. Wave 3 and 4 depend on Wave 1 results.

---

## Output Structure

After all waves complete:

```
scripts/h200_test/employment_results/
├── deepseek/
│   ├── A1/  (store.duckdb, report_*.md, metrics_*.json)
│   ├── A2/
│   ├── ...
│   └── H5/
├── gemini-flash/
│   ├── A1/
│   ├── B1/
│   └── ...
├── grok-fast/
│   └── ...
├── mistral-large/
│   └── ...
├── kimi/
│   └── ...
├── openai/
│   └── A1/
├── venice/
│   ├── claude-opus/A1/
│   └── ...
├── wave1_summary.json     ← aggregated Wave 1 metrics
├── wave2_summary.json     ← aggregated Wave 2 metrics
├── wave3_summary.json     ← orchestrator comparison table
├── wave4_flock_bench.json ← throughput rankings
└── wave5_summary.json     ← Venice quality comparison
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| DeepSeek rate limit hit (60 RPM) | Medium | Delays Wave 1 | Concurrency capped at 6; exponential backoff in httpx |
| Kimi requires temperature=1.0 | Known | Wave 2 group D fails | Already configured in ProviderConfig |
| Venice model names don't match | Medium | Wave 5 fails | Verify model list via Venice API before running |
| Claude adapter missing for Wave 3 | Known | Can't test Claude orchestrator | Use OpenRouter proxy or write adapter first |
| API key exhaustion/billing | Low | Runs stop mid-wave | Monitor cost; total ~$19 is well within reasonable |
| DuckDB file locking (parallel writes) | Low | Corruption | Each topic has its own DuckDB file — no shared writes |
| Transient API errors (500, timeout) | Medium | Individual topics fail | Runner logs ERROR, continues with next topic; retry manually |

---

## Decision Points Before Execution

1. **Run all 5 waves, or start with Wave 1 only?**
   - Recommendation: Start Wave 1 + Wave 2 + Wave 5 in parallel, then Wave 3-4 after.

2. **Write Claude Anthropic adapter for Wave 3?**
   - Requires ~20 lines in `complete_fn` to translate OpenAI format → Anthropic `/v1/messages`
   - Alternative: Route through OpenRouter (simpler, but adds proxy hop)

3. **Max concurrent topics (rate limit safety)?**
   - Current plan: 6 concurrent. Can increase to 8 if DeepSeek handles it, or decrease to 4 for safety.

4. **Max waves per topic?**
   - Current: 3 waves. Could increase to 5 for deeper exploration but adds ~60% more time and cost.

5. **Verify Venice model IDs before Wave 5?**
   - Need to call Venice API to list available models and confirm naming convention.
