# Employment List Execution Plan

Deliberate model-to-role test matrix for MiroThinker's full architecture.
Go/no-go checkpoint — review and approve before execution begins.

**Runner script:** `scripts/h200_test/run_employment.py`
**Corpus:** `scripts/h200_test/test_corpus.txt` (Milos corpus, 128 lines, 12 KB)
**Output:** `scripts/h200_test/employment_results/{model}/{topic_id}/`
**H200 instance:** Vast.ai #35390779, France, 143 GB VRAM, $2.02/hr

---

## Architectural Tiers Under Test

The current swarm engine has 5 distinct architectural tiers. Each tier
exercises different model capabilities. Testing a model only as a "worker"
tells you nothing about its performance as a Flock driver or clone backend.

| Tier | What | Where in Code | Model Needs | Currently Tested? |
|---|---|---|---|---|
| **T1: Worker** | Tool-free bee reasoning on data packages | `agent_worker.py` → `run_tool_free_worker()` | Uncensored, strong reasoning | Yes (basic) |
| **T2: Research Organizer** | Clone-based gap resolution with search tools | `research_organizer.py` → `run_research_organizer()` | Tool-capable, domain-expert search queries | Partially (runs but not measured) |
| **T3: Flock Battery** | 13-step SQL+LLM scoring/clustering/serendipity | `FLOCK_SERENDIPITY_ARCHITECTURE.md` templates | Fast, cheap, good relevance judgment | **No** |
| **T4: Clone Context** | Session proxy prepends worker conversation to Flock queries | `session_proxy.py` → `register_clone()` | Stays in character with prepended context | **No** |
| **T5: Cross-Expert** | Multi-clone pollination, meta-expert composition, disagreement | `CLONED_CONTEXT_PATTERN.md` ramifications 2-8 | Large context, multi-domain synthesis | **No** |

**The current API-based employment list only tests T1.** Tiers 3-5 require
local serving (vLLM on H200) because:
- Flock calls go through DuckDB → session proxy → vLLM (local loop)
- vLLM prefix caching makes clone context injection efficient (~free after first call)
- High-volume Flock queries (50-300 per step) need no rate limits
- Clone context accumulation needs persistent vLLM KV cache

---

## H200 Infrastructure

**Current state:**
- Vast.ai instance 35390779 running (France, $2.02/hr)
- vLLM 0.19.1 installed
- `huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated` currently serving on port 8000
- 133 GB VRAM used, 10 GB free
- 80 GB disk, 64 GB free, 92 GB HF cache (Kimi model)

**Models that fit on 1× H200 (143 GB VRAM):**

| Model | HuggingFace ID | Quant | VRAM | Active Params | Best For |
|---|---|---|---|---|---|
| Kimi-Linear-48B-A3B | `huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated` | auto | ~92 GB | 3B | T4 (linear attention = unlimited clone context) |
| Llama-3.3-70B | `huihui-ai/Llama-3.3-70B-Instruct-abliterated` | FP8 | ~75 GB | 70B | T1, T2 (strongest worker reasoning) |
| Qwen3-235B-A22B | `huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-abliterated` | INT4 | ~125 GB | 22B | T5 (orchestrator/meta-expert quality ceiling) |
| Hermes-3-70B | `NousResearch/Hermes-3-Llama-3.1-70B` | FP8 | ~75 GB | 70B | T1 (abliteration method comparison) |
| Qwen3-4B | `huihui-ai/Qwen3-4B-abliterated` | FP16 | ~8 GB | 4B | T3 (flock speed floor) |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.6-35B-A3B` | auto | ~70 GB | 3B | Research framing hypothesis (NOT abliterated) |

**Disk constraint:** Only one large model cached at a time. Model rotation
requires: stop vLLM → clear HF cache → download new model → restart vLLM.
~30-60 min per swap (download-dominated).

---

## Test Hypotheses

Each slot in the model rotation plan tests a specific hypothesis. We are NOT
just running "the same swarm with different models" — each slot isolates a
different architectural variable.

### H1: Dense 70B vs MoE 48B (3B active) for research depth
**What we learn:** Does 20× more active parameters produce proportionally
deeper findings, or does the MoE's speed advantage compensate?
**Metric:** Findings count, unique angle discovery, specificity of claims
(dosages mentioned vs generic statements), serendipity connections.

### H2: Linear attention for unlimited clone context
**What we learn:** Kimi-Linear has constant KV cache cost. Can the clone
accumulate 10+ waves of conversation without degradation? Does the wave-10
clone produce better relevance judgments than the wave-1 clone?
**Metric:** Flock relevance accuracy at wave 1 vs wave 5 vs wave 10.
**Why H200 required:** vLLM prefix caching + clone context accumulation.

### H3: Abliteration technique comparison
**What we learn:** huihui-ai weight surgery vs NousResearch Hermes fine-tune —
different residual refusal patterns? Different knowledge gaps?
**Metric:** Same topic, same corpus. Compare refusal rate, findings count,
and whether specific sub-topics (e.g., dosage ranges) are treated differently.

### H4: Research framing vs abliteration
**What we learn:** Does MiroThinker's research framing bypass safety on a
non-abliterated base model? If yes, abliteration may be unnecessary for
well-framed use cases.
**Metric:** Qwen3.6-35B-A3B (NOT abliterated) on A1. Does it refuse?
How do findings compare to abliterated variant?

### H5: Orchestrator quality ceiling (235B MoE)
**What we learn:** With 22B active params (vs 3B or 70B), does the larger
MoE produce better angle detection, deeper cross-domain synthesis, better
reports?
**Metric:** A1 angle count and quality, report depth, serendipity connections
found. Is the 5× slowdown worth it?

### H6: Flock driver speed vs quality tradeoff
**What we learn:** For the 13-step Flock template battery (scoring, clustering,
dedup, contradiction detection, cross-angle bridges, contrarian challenges),
what's the minimum model size that produces reliable relevance judgments?
**Metric:** 50 relevance judgments per model. Measure tok/s, cost, accuracy
vs 70B ground truth.

### H7: Clone context injection quality
**What we learn:** Does prepending the worker's conversation to Flock queries
actually improve relevance judgment vs a generic model? How much?
**Metric:** Same 50 judgments, with and without clone context. Accuracy delta.

### H8: Cross-expert pollination
**What we learn:** When the insulin clone evaluates hematology findings (and
vice versa), does it find genuine cross-domain connections? Or just noise?
**Metric:** N×(N-1) cross-expert queries. Count "relevant" hits, spot-check
for quality. Compare to template 10 (generic polymath bridge).

### H9: Expert disagreement as signal
**What we learn:** When two different architecture clones (Llama-70B vs
Kimi-48B) disagree on a finding's relevance, is the disagreement predictive
of genuine controversy or model blind spots?
**Metric:** Agreement matrix across architectures. Disagreement rows flagged
for manual review.

---

## Model Rotation Plan (H200)

Sequential model loading. Each "slot" = stop vLLM → clear cache → download →
serve → run tests → capture results. Total ~8-10 hours.

### Slot 1: Kimi-Linear-48B-A3B (ALREADY LOADED)

**Setup time:** 0 min (already serving on port 8000)
**Run time:** ~2 hours

**Tests:**

| Test | Tier | Hypothesis | Topic(s) | What We Measure |
|---|---|---|---|---|
| 1A | T1 | H1 (MoE baseline) | A1 | Findings count, specificity, angle diversity — MoE worker baseline |
| 1B | T1 | H1 | B4, F9 | High-serendipity topics — does 3B active find cross-domain connections? |
| 1C | T3 | H6 (flock speed) | — | 50 relevance judgments from A1 findings. Measure tok/s, latency |
| 1D | T4 | H2 (linear clone context) | A1 | Run 5 waves. Register clone after each wave. Measure relevance accuracy at wave 1 vs 3 vs 5 |
| 1E | T4 | H7 (clone vs generic) | A1 | Same 50 judgments WITH clone context vs WITHOUT. Accuracy delta |

**vLLM command (already running):**
```bash
vllm serve huihui-ai/Huihui-Kimi-Linear-48B-A3B-Instruct-abliterated \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --max-num-seqs 32 --host 0.0.0.0
```

**Key hypothesis tested:** Linear attention models have constant KV cache cost.
If the wave-5 clone produces significantly better judgments than wave-1, this
validates the clone accumulation architecture — the core innovation of the
Cloned-Context pattern.

---

### Slot 2: Llama-3.3-70B-Instruct-abliterated (FP8)

**Setup time:** ~45 min (download 70 GB weights + vLLM load)
**Run time:** ~3.5 hours

**Tests:**

| Test | Tier | Hypothesis | Topic(s) | What We Measure |
|---|---|---|---|---|
| 2A | T1 | H1 (dense 70B) | A1 | Findings count + specificity. Direct comparison with Slot 1A (Kimi-48B) |
| 2B | T1 | H1 | B4, F9 | Same high-serendipity topics. Compare serendipity connections found |
| 2C | T1 | H1 | All 48 topics | Full coverage run — establish the 70B baseline corpus |
| 2D | T2 | (R.O. quality) | A1 | Measure clone research quality: doubts resolved, fresh evidence specificity |
| 2E | T3 | H6 (flock 70B ground truth) | — | 50 relevance judgments. This is the "ground truth" for H6 accuracy comparison |
| 2F | T4 | H7 (clone context 70B) | A1 | 5 waves + clone registration. Compare clone accuracy to Slot 1D (Kimi) |
| 2G | T4 | H8 (cross-expert) | A1+B4 | Insulin clone evaluates B4 findings, tren clone evaluates A1 findings |

**vLLM command:**
```bash
vllm serve huihui-ai/Llama-3.3-70B-Instruct-abliterated \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --dtype fp8 --enable-chunked-prefill --max-num-seqs 16 --host 0.0.0.0
```

**Key tests:** 2A vs 1A answers H1 definitively (70B dense vs 3B MoE).
2C produces the full 48-topic corpus that all subsequent analysis builds on.
2G is the first real cross-expert pollination test.

---

### Slot 3: Hermes-3-Llama-3.1-70B (FP8)

**Setup time:** ~45 min
**Run time:** ~1 hour

**Tests:**

| Test | Tier | Hypothesis | Topic(s) | What We Measure |
|---|---|---|---|---|
| 3A | T1 | H3 (abliteration comparison) | A1 | Same topic, same corpus. Compare findings to 2A (huihui abliterated) |
| 3B | T1 | H3 | B1 (cattle origins) | Potentially contentious sub-topic. Do refusal patterns differ? |
| 3C | T4 | H9 (disagreement) | A1 | Register Hermes-70B clone. Compare judgments to Slot 2F Llama-70B clone. Disagreement = signal |

**vLLM command:**
```bash
vllm serve NousResearch/Hermes-3-Llama-3.1-70B \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --dtype fp8 --enable-chunked-prefill --max-num-seqs 16 --host 0.0.0.0
```

**Key test:** 3C produces the disagreement matrix for H9. Findings where
huihui-Llama agrees but Hermes disagrees (or vice versa) are high-value
diagnostic signals about abliteration residuals.

---

### Slot 4: Qwen3.6-35B-A3B (NOT abliterated)

**Setup time:** ~30 min
**Run time:** ~30 min

**Tests:**

| Test | Tier | Hypothesis | Topic(s) | What We Measure |
|---|---|---|---|---|
| 4A | T1 | H4 (research framing vs abliteration) | A1 | Does it refuse? Partial refuse? Full engagement? |
| 4B | T1 | H4 | B3 (AR binding), D8 (secretagogues) | Mechanism topics — more "academic" framing. Compare refusal behavior to direct PED topics |

**vLLM command:**
```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --max-num-seqs 32 --host 0.0.0.0
```

**Key test:** If 4A produces findings comparable to abliterated models, it
proves that MiroThinker's research framing is itself the censorship bypass —
abliteration is insurance, not a requirement. This would dramatically expand
the viable model pool.

---

### Slot 5: Qwen3-235B-A22B-Instruct-abliterated (INT4)

**Setup time:** ~90 min (largest download, ~120 GB)
**Run time:** ~1.5 hours

**Tests:**

| Test | Tier | Hypothesis | Topic(s) | What We Measure |
|---|---|---|---|---|
| 5A | T1 | H5 (quality ceiling) | A1 | Findings specificity and depth. Is 22B active >> 3B active? |
| 5B | T5 | H5 (orchestrator) | A1 | Use as orchestrator: angle detection quality, contradiction finding |
| 5C | T5 | H8 (meta-expert) | A1+B4 | Concatenate A1 + B4 worker conversations as meta-expert. Ask cross-domain synthesis question |
| 5D | T3 | H6 (flock 235B) | — | 50 relevance judgments. Compare quality vs 70B (2E) and speed tradeoff |

**vLLM command:**
```bash
vllm serve huihui-ai/Huihui-Qwen3-235B-A22B-Instruct-abliterated \
  --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  --dtype auto --trust-remote-code --enable-chunked-prefill \
  --max-num-seqs 8 --host 0.0.0.0
```

**Note:** 8K context only (VRAM constraint at 125 GB weights). This is tight
but sufficient for single-wave worker calls. NOT suitable for clone
accumulation (context too short). Tests orchestrator and meta-expert quality.

---

## Parallel API Runs (While H200 Rotates)

The H200 tests local models. Simultaneously, we run API-based tests on
remote providers. These two tracks are independent — different machines,
different rate limits, different things being tested.

### API Track: Remote Model Cross-Validation

While the H200 rotates through Slots 1-5 (~8-10 hours), the Devin VM runs:

| Wave | Topics | Model | Provider | Cost | Time | Purpose |
|---|---|---|---|---|---|---|
| API-1 | A1, B4, F9 (3 key topics) | `deepseek-chat` (V3.2) | DeepSeek | ~$1 | ~30m | Remote API baseline for comparison with local |
| API-2 | A1, B4, F9 | `gemini-2.5-flash` | Google | ~$0.15 | ~30m | 1M context — does longer context help? |
| API-3 | A1, B4, F9 | `grok-4-1-fast-non-reasoning` | xAI | ~$0.30 | ~30m | Uncensored commercial + 2M context |
| API-4 | A1, B4, F9 | `mistral-large-latest` | Mistral | ~$2.40 | ~30m | Strong reasoning, different training data |
| API-5 | A1, B4, F9 | `kimi-k2.5` | Moonshot | ~$1.50 | ~30m | Chinese pharma literature |
| API-6 | A1 only | `gpt-4.1-mini` | OpenAI | ~$0.15 | ~10m | OpenAI baseline |
| API-7 | A1 only | `claude-sonnet-4` via OpenRouter | OpenRouter | ~$0.50 | ~10m | Claude worker test (we know it refuses direct — does research framing change that?) |

**Total API cost:** ~$6. **Total API time:** ~2.5 hours (can overlap).

### Flock Speed Benchmark (API-based, no H200 needed)

| Model | Provider | Expected tok/s | 50 Judgments |
|---|---|---|---|
| `ministral-3b-latest` | Mistral | ~278 | Price floor |
| `qwen3-32b` | Groq | ~343 | Best speed+reasoning |
| `llama-4-scout-17b-16e-instruct` | Groq | ~162 | MoE speed |
| `gpt-4.1-nano` | OpenAI | varies | OpenAI cheapest |

**Total:** 200 calls, ~$0.01, ~10 min.

---

## Comparison Matrix (What Gets Compared to What)

### Worker Quality (H1, H3, H4, H5)

All on topic A1 (Milos Insulin baseline):

| Model | Source | Active Params | Type | Expected |
|---|---|---|---|---|
| Kimi-48B-A3B (Slot 1A) | Local H200 | 3B | MoE, abliterated, linear attention | Fast, shallow |
| Llama-70B (Slot 2A) | Local H200 | 70B | Dense, abliterated | Deep, slower |
| Hermes-70B (Slot 3A) | Local H200 | 70B | Dense, uncensored fine-tune | Deep, different knowledge |
| Qwen-35B-A3B (Slot 4A) | Local H200 | 3B | MoE, NOT abliterated | Tests research framing |
| Qwen-235B-A22B (Slot 5A) | Local H200 | 22B | MoE, abliterated | Quality ceiling |
| DeepSeek V3.2 (API-1) | Remote API | ~37B active | MoE, uncensored | Proven baseline |
| Gemini 2.5 Flash (API-2) | Remote API | unknown | Dense? | 1M context |
| Grok Fast (API-3) | Remote API | unknown | Dense? | 2M context, uncensored |

**Deliverable:** 8-row comparison table with findings count, specificity score
(% of findings with dosages/numbers), unique angles, serendipity connections.

### Flock Driver (H6)

All on 50 relevance judgments from A1 findings:

| Model | Source | tok/s | Accuracy vs 70B Ground Truth |
|---|---|---|---|
| Kimi-48B-A3B (Slot 1C) | Local H200 | measured | measured |
| Llama-70B (Slot 2E) | Local H200 | measured | **ground truth** |
| Qwen-235B (Slot 5D) | Local H200 | measured | measured |
| ministral-3b | Remote API | ~278 | measured |
| qwen3-32b | Remote API | ~343 | measured |
| gpt-4.1-nano | Remote API | varies | measured |

**Deliverable:** Speed vs accuracy scatter plot. Identify the Pareto-optimal
flock driver (best accuracy at acceptable speed).

### Clone Context Quality (H2, H7, H9)

| Test | With Clone Context | Without | Delta |
|---|---|---|---|
| Kimi-48B wave 1 clone | Slot 1E | Slot 1C (generic) | H7 |
| Kimi-48B wave 5 clone | Slot 1D (wave 5) | Slot 1D (wave 1) | H2 |
| Llama-70B clone | Slot 2F | Slot 2E (generic) | H7 |
| Hermes-70B clone vs Llama-70B clone | Slot 3C | Slot 2F | H9 |

**Deliverable:** Clone context injection value quantified. If wave-5 clone
accuracy > wave-1 by ≥10%, the accumulation architecture is validated.

### Cross-Expert Pollination (H8)

| Clone Expert | Evaluates Findings From | Source |
|---|---|---|
| A1 insulin clone (Llama-70B) | B4 tren+glucose findings | Slot 2G |
| B4 tren clone (Llama-70B) | A1 insulin findings | Slot 2G |
| A1 insulin clone (Qwen-235B) | B4 findings | Slot 5C |
| A1+B4 meta-expert (Qwen-235B) | Cross-domain synthesis | Slot 5C |

**Deliverable:** Cross-expert connection count. Compare to template 10
(generic polymath bridge) quality. If expert-driven pollination finds
connections that generic doesn't, the clone pattern is validated.

---

## Execution Timeline

```
Hour 0:       Start API waves (API-1 through API-7) + Flock speed benchmark
              ‖ parallel on Devin VM ‖
Hour 0-2:     Slot 1 — Kimi-48B-A3B tests (already loaded, zero setup)

Hour 2-2.75:  Model swap: clear Kimi cache → download Llama-70B FP8

Hour 2.75-6:  Slot 2 — Llama-70B tests (including full 48-topic run 2C)

Hour 6-6.75:  Model swap: download Hermes-3-70B

Hour 6.75-8:  Slot 3 — Hermes-70B tests

Hour 8-8.5:   Model swap: download Qwen3.6-35B-A3B

Hour 8.5-9:   Slot 4 — Research framing hypothesis test

Hour 9-10.5:  Model swap: download Qwen3-235B-A22B (largest, ~90 min)

Hour 10.5-12: Slot 5 — Quality ceiling + meta-expert tests

Hour 12-13:   Compile results → MODEL_REGISTRY.md update
```

**Total wall clock:** ~13 hours
**Total H200 cost:** 13h × $2.02/hr = ~$26
**Total API cost:** ~$6
**Grand total:** ~$32

---

## Session Proxy Setup (Required for T4/T5 Tests)

The clone tests (Slots 1D, 1E, 2F, 2G, 3C, 5C) require the session proxy
running alongside vLLM:

```bash
# On the H200 instance
pip install fastapi uvicorn httpx
cd /workspace/MiroThinker
uvicorn swarm.session_proxy:app --host 0.0.0.0 --port 18199
```

**Wiring:** Flock SQL routes through session proxy (port 18199) instead of
directly to vLLM (port 8000). The proxy intercepts `clone_{angle}` model
names and prepends stored conversations before forwarding to vLLM.

**For non-clone requests:** proxy passes through to vLLM unchanged.

---

## Success Criteria

### Must-Pass (execution is useless without these)
- [ ] ≥ 44/48 topics produce findings in Slot 2C (full coverage run)
- [ ] A1 produces ≥ 15 findings with every model tested (no censorship)
- [ ] Session proxy successfully injects clone context (Slot 1D/1E verified)

### Should-Pass (validates core hypotheses)
- [ ] H1: 70B produces ≥ 30% more specific findings than 3B-active MoE
- [ ] H2: Wave-5 clone accuracy ≥ 10% better than wave-1 clone
- [ ] H7: Clone-context accuracy ≥ 15% better than generic (no context)
- [ ] H8: Cross-expert finds ≥ 3 connections not found by generic polymath

### Nice-to-Have (informs future architecture)
- [ ] H4: Non-abliterated Qwen engages with PED content via research framing
- [ ] H5: 235B MoE orchestrator detects ≥ 2 more angles than 70B
- [ ] H9: Architecture disagreement matrix has ≥ 5 high-signal rows

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| 70B FP8 download takes >1h | Medium | Delays Slot 2 | Can start with Kimi-48B tests while downloading |
| 235B INT4 doesn't fit (VRAM) | Low | Skip Slot 5 | Reduce `--max-model-len` to 4096 or use AWQ quant |
| Session proxy doesn't work with vLLM | Medium | Lose T4/T5 tests | Test proxy in Slot 1 first; fall back to direct injection |
| Qwen3.6-35B-A3B refuses all PED content | Medium | H4 disproven | That IS a valid result — confirms abliteration necessary |
| Disk too small for model downloads | Medium | Can't rotate | Resize Vast.ai disk to 500 GB before starting |
| H200 instance terminated mid-run | Low | Lose all local data | Save results to Devin VM after each slot |

---

## Decision Points Before Execution

1. **Resize H200 disk to 500 GB?**
   Current: 80 GB (64 GB free). Each model download is 35-120 GB.
   Without resize: must delete previous cache before each download.
   With resize (~$0.05/hr extra): keep all models cached, instant re-serve.

2. **Run all 5 slots, or subset?**
   Minimum valuable: Slots 1 + 2 (~6h). Answers H1, H2, H6, H7.
   Recommended: Slots 1-4 (~9h). Adds H3, H4, H8.
   Full: All 5 slots (~13h). Adds H5 (quality ceiling) and meta-expert.

3. **Clone the MiroThinker repo to H200, or run remotely?**
   Option A: Clone repo to H200, run swarm engine locally (fastest — no network hop for LLM calls).
   Option B: Run swarm on Devin VM, point `api_base` at H200's external port.
   Recommendation: Option A for Slots 2C (full 48-topic run) and all T4/T5 tests.
   Option B works for simple T1 tests (Slots 1A, 3A, 4A).

4. **Full 48-topic run on which models?**
   Slot 2C runs all 48 topics on Llama-70B. Do we also want full 48-topic
   coverage on any other model?
   Recommendation: 70B only for full coverage. Other models tested on
   A1 + B4 + F9 (3 diagnostic topics) to keep total time manageable.

5. **Start API track immediately, or wait for H200 results?**
   Recommendation: Start both tracks simultaneously. API results provide
   remote baseline while H200 runs local tests. Compare remote vs local
   on the same topics (A1, B4, F9).
