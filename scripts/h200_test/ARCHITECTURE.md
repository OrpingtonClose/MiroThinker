# Cumulative Model Test — Architecture

## What This Tests

A single research query is processed by multiple SOTA abliterated models in sequence. Each model inherits every finding from every previous model. The question: **does epistemic diversity — different model architectures trained on different data with different biases — produce genuinely different and complementary knowledge?**

```
Query: "Synthesize a comprehensive, practitioner-grade ramping bodybuilding
cycle protocol covering testosterone, trenbolone, insulin (grounded in Milos
Sarcev's timing framework), growth hormone, turinabol, boldenone, actovegin,
and LGD-4033."
```

This query was chosen because it sits at the intersection of pharmacology, endocrinology, sports science, and practitioner lore — domains where censored models refuse to engage and where ground truth is scattered across academic papers, forum posts, coach interviews, and grey-market guides. An uncensored model is a hard requirement, not a preference.

---

## System Under Test

```
┌──────────────────────────────────────────────────────────────────┐
│                        run_swarm_test.py                         │
│  CLI entrypoint — accepts --model, --api-base, --temperature,   │
│  --max-tokens, --convergence-threshold, --workers, --waves, etc │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                       MCPSwarmEngine                             │
│                                                                  │
│  1. Ingest 96K corpus into ConditionStore                       │
│  2. Detect or accept preset research angles                      │
│  3. Create N Strands Agent workers (one per angle)              │
│  4. Run workers in parallel waves                                │
│  5. After each wave: check convergence (store growth rate)       │
│  6. Optional serendipity wave (cross-domain discovery)           │
│  7. Generate final report from accumulated store                 │
└──────────────┬──────────────────────────┬────────────────────────┘
               │                          │
    ┌──────────┼──────────┐               │
    ▼          ▼          ▼               ▼
 Worker A   Worker B   Worker C    Serendipity Agent
 (Agent)    (Agent)    (Agent)        (Agent)
    │          │          │               │
    └──────────┼──────────┘               │
               ▼                          │
    ┌─────────────────────┐               │
    │   ConditionStore    │◄──────────────┘
    │   (DuckDB file)     │
    │                     │
    │   Persists across   │
    │   ALL runs          │
    └─────────────────────┘
```

### Worker Agent Internals

Each worker is a **Strands Agent** with an OpenAI-compatible model provider pointed at a local vLLM instance. The worker doesn't know it's in a swarm. It has six tools:

| Tool | What it does |
|------|-------------|
| `search_corpus(query)` | Keyword search over all stored findings and corpus paragraphs |
| `get_peer_insights(topic)` | Retrieve findings stored by OTHER workers (including from previous runs) |
| `store_finding(fact, confidence)` | Write a new finding to the shared store |
| `check_contradictions(claim)` | Find stored findings that conflict with a given claim |
| `get_research_gaps()` | Identify under-explored topics based on angle coverage |
| `get_corpus_section(offset, max_chars)` | Read a chunk of the raw corpus text |

The context window is irrelevant. A 32K-context model processes a 96K corpus by making repeated tool calls — pull a chunk, reason about it, store findings, pull the next chunk. The ConditionStore IS the memory, not the context window.

### Cumulative Knowledge Flow

```
Run 1 (Model A):
  Workers read corpus → store findings → serendipity finds cross-domain links
  Store: corpus paragraphs + Model A's findings

Run 2 (Model B):
  Workers call get_peer_insights → see ALL of Model A's findings
  Workers call search_corpus → see corpus + Model A's findings
  Workers store NEW findings (what Model A missed)
  Store: corpus + Model A + Model B

Run N (Model N):
  Workers see EVERYTHING from runs 1..N-1
  Focus shifts to gaps, contradictions, and novel connections
  Store: cumulative knowledge from all model architectures
```

The store is a file-backed DuckDB (`.duckdb`). Same file opened by every run. No deduplication — if two models independently reach the same conclusion, both findings are stored. Independent convergence is a signal, not redundancy.

---

## Model Selection: Typology

Models are selected to maximize architectural diversity. The goal is not "which model is best" but "do different architectures find different things."

### Selection Criteria

1. **Uncensored** — must be abliterated or natively unfiltered. The query requires it.
2. **Tool-calling support** — workers call 6 tools per turn. The model must emit structured tool calls, not just text. This requires vLLM `--enable-auto-tool-choice` with a compatible parser.
3. **Architectural diversity** — different model families, different training data origins, different attention mechanisms.
4. **Fits on 1×H200** (141GB VRAM) in a format that supports tool calling.

### Model Roster

| Run | Model | Architecture | Why This Model |
|-----|-------|-------------|---------------|
| 1 | `huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated` | Qwen3.6 MoE (35B total, 3B active) | SOTA as of April 2026. MoE = efficient, Claude-distilled = strong instruction following. Native tool calling. Only 3B active params = room for large batches. |
| 2 | `huihui-ai/Huihui-Qwen3.5-27B-abliterated` | Qwen3.5 dense (27B) | Same Qwen lineage as Run 1 but dense architecture and older generation. Tests whether MoE vs dense matters for research quality. |
| 3 | `TrevorJS/gemma-4-31B-it-uncensored` or `huihui-ai/gemma-4-*-abliterated` | Gemma 4 (Google, dense 31B) | Completely different training corpus (Google's data). Different tokenizer, different attention patterns. Should find things Qwen-family models miss due to training data gaps. |
| 4 | `huihui-ai/GLM-4.7-*-abliterated` | GLM (Zhipu AI, Chinese origin) | Chinese-origin architecture with access to Chinese-language pharmacology literature during training. May surface Traditional Chinese Medicine perspectives, actovegin research published in Chinese journals, and dosing conventions from non-Western bodybuilding communities. |
| 5 | `huihui-ai/Kimi-Linear-*-abliterated` | Linear attention (Moonshot AI) | **Different attention mechanism entirely.** Standard transformers have quadratic attention scaling; Kimi uses linear attention with constant KV cache. This means workers can see MORE data per tool call (larger `max_chars`, more `max_results`) because context is effectively unlimited for the same VRAM. Tests whether seeing more data at once produces better synthesis than seeing it in chunks. |

### Typology Dimensions

```
                    Dense ◄──────────────────► MoE/Sparse
                      │                          │
        Qwen3.5-27B ──┤              Qwen3.6-35B-A3B ──┤
        Gemma-4-31B ──┤                                 │
        GLM-4.7 ──────┘                                 │
                                                        │
                Standard Attention ◄────► Linear Attention
                      │                          │
        Qwen, Gemma, GLM ─────┤     Kimi-Linear ──┘
                               │
                Western Training ◄────► Chinese Training
                      │                          │
        Qwen, Gemma ──┤              GLM-4.7 ────┘
                      │
          Instruction-Tuned ◄────► Claude-Distilled
                      │                          │
        Gemma, GLM ───┤     Qwen3.6-Claude-4.7 ─┘
```

Each model occupies a different point in this space. The cumulative test measures whether these differences translate into genuinely different findings — or whether all models converge on the same knowledge regardless of architecture.

---

## What Stays Constant

Every run uses identical settings except the model. No handicapping.

| Parameter | Value | Why |
|-----------|-------|-----|
| Corpus | 96K chars (Milos Sarcev + extended insulin protocols) | Full corpus, always |
| Workers | 5 | Matches the 5 preset angles |
| Waves | 3 | Sufficient for convergence based on prior experiments |
| Serendipity | Always on | Cross-domain connections are a primary measurement |
| Angles | Preset: Insulin & GH, Testosterone & Trenbolone pharma, Ancillaries, Oral compounds, Boldenone & EQ | Consistent decomposition across models |
| Temperature | 0.3 | Deterministic enough for comparison |
| Max tokens | 4096 per worker response | Standard |
| Convergence threshold | 5 | Stop if <5 new findings per wave |
| Store | Same `.duckdb` file, cumulative | The whole point |
| vLLM endpoint | localhost (H200, 141GB VRAM) | All local, all uncensored |

---

## Measurements

### Per-Run Metrics

Each run produces a `metrics.json` with:

```json
{
  "model": "huihui-ai/Huihui-Qwen3.6-...",
  "run_number": 1,
  "findings_before": 0,
  "findings_after": 187,
  "delta": 187,
  "total_tool_calls": 342,
  "tool_call_breakdown": {
    "search_corpus": 89,
    "get_peer_insights": 45,
    "store_finding": 187,
    "check_contradictions": 3,
    "get_research_gaps": 12,
    "get_corpus_section": 6
  },
  "elapsed_s": 210.4,
  "convergence_reason": "max_waves_reached",
  "waves_completed": 3,
  "findings_per_wave": [92, 58, 37],
  "serendipity_findings": 14,
  "config": { ... }
}
```

### Cross-Run Analysis

After all runs complete, we measure:

1. **Marginal delta curve**: `findings_added` vs `run_number`. When does adding another model stop helping?
2. **Finding overlap**: How many findings from Model B were already stored by Model A? (Keyword overlap scoring on the finding text.)
3. **Unique contributions**: Findings stored by Model X that have <30% keyword overlap with any finding from previous models.
4. **Contradiction rate**: How often does Model B's `check_contradictions` fire against Model A's findings?
5. **Tool call profiles**: Does Model X use `get_peer_insights` more than Model Y? Does that correlate with unique contributions?
6. **Serendipity quality**: Are cross-domain findings from later runs (which see more material) higher confidence than from earlier runs?
7. **Angle coverage balance**: Do all models distribute findings evenly across angles, or do some architectures fixate on certain angles?

### The Deliverable

The final ConditionStore after all runs is the primary output — a knowledge base assembled by 5 different model architectures, each building on the others' work. The store can be queried directly via SQL:

```sql
-- What did each model uniquely contribute?
SELECT source_model, COUNT(*) as findings, AVG(confidence) as avg_conf
FROM conditions
WHERE event_type = 'finding'
GROUP BY source_model;

-- Cross-model agreement (same fact found by multiple models)
SELECT text, COUNT(DISTINCT source_model) as models_agreeing
FROM conditions
WHERE event_type = 'finding'
GROUP BY text
HAVING models_agreeing > 1;
```

---

## Infrastructure

### GPU: Vast.ai H200

- **Instance**: 1×H200 SXM (141GB VRAM), rented from Vast.ai
- **vLLM**: v0.19.1, `--enable-auto-tool-choice --tool-call-parser hermes`
- **Model swaps**: Kill vLLM, start with new `--served-model-name`, wait for load
- **Model cache**: Downloaded models uploaded to B2 (`swarm-local-models` bucket) for future runs

### Execution Flow

```bash
# Run 1
vllm serve huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated ...
python run_swarm_test.py --engine mcp \
  --corpus corpus_96k.txt \
  --db cumulative.duckdb \
  --model huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated \
  --api-base http://<h200-ip>:<port>/v1 \
  --output-dir results/run1

# Run 2 — same store, different model
# Kill vLLM, start with new model
python run_swarm_test.py --engine mcp \
  --db cumulative.duckdb \
  --model huihui-ai/Huihui-Qwen3.5-27B-abliterated \
  ...

# Run N — knowledge accumulates
```

### Model Serving Considerations

| Model | Dtype | VRAM (weights) | Concurrent Workers | Tool Call Parser |
|-------|-------|----------------|-------------------|-----------------|
| Qwen3.6-35B-A3B (MoE) | BF16 | ~12GB (3B active) | 32 | hermes |
| Qwen3.5-27B (dense) | BF16 | ~54GB | 8–12 | hermes |
| Gemma-4-31B (dense) | BF16 | ~62GB | 6–8 | hermes |
| GLM-4.7 (dense) | BF16 | ~varies | varies | hermes/jinja |
| Kimi-Linear | BF16 | ~varies | varies | jinja |

The MoE model (Run 1) is dramatically more efficient — only 3B active params means the H200 can serve 32 concurrent sequences with minimal KV cache pressure. Dense models eat more VRAM for weights, leaving less for concurrent workers. This is reflected in `--max-num-seqs` per model.

---

## What We Expect to Learn

1. **Does architectural diversity matter?** If all 5 models find the same 200 facts, architecture is irrelevant for research quality. If each adds 50+ unique findings, architecture is a first-class variable in swarm design.

2. **Which architecture type excels at what?** MoE for breadth (more tool calls, more exploration)? Dense for depth (better per-finding quality)? Linear attention for synthesis (sees more at once)?

3. **When does cumulative knowledge saturate?** The marginal delta curve tells us: is 3 models enough, or does the 5th model still contribute meaningfully?

4. **Does seeing prior findings help or hurt?** Later models see everything. Do they build on it (novel cross-references) or just rephrase it (decreasing marginal returns)?

5. **Do contradictions emerge?** Different training data may encode different "truths" about dosing, timing, or interactions. Explicit contradictions between models are the most valuable output — they identify where the evidence is genuinely uncertain.

6. **Does prompt decomposition quality change per architecture?** All models get the same 5 preset angles. Do some architectures handle certain angles better (e.g., GLM on actovegin due to Chinese-language training data)?
