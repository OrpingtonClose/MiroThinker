# MCP Swarm — External Data Test Scenarios

**Query:** "Synthesize a comprehensive, practitioner-grade ramping bodybuilding cycle protocol covering testosterone, trenbolone, insulin (grounded in Milos Sarcev's timing framework), growth hormone, turinabol, boldenone, actovegin, and LGD-4033. For each phase, specify exact compounds, dosages, frequencies, timing windows, micronutrient support, bloodwork markers, and transition criteria. Explain the pharmacokinetic reasoning behind every decision."

**Constants across all scenarios:** Full 96K corpus, full external toolset, serendipity always on, **local uncensored models only** (vLLM on H200).

---

## Design Principle: Cumulative Knowledge

**The ConditionStore persists across all runs.** Each model inherits every finding from every previous model. Knowledge accumulates — it is never discarded.

```
Run 1 (Qwen-32B):     Store = corpus + Qwen discoveries
Run 2 (Llama-70B):     Store = corpus + Qwen discoveries + Llama discoveries
Run 3 (Gemma-31B):     Store = corpus + Qwen + Llama + Gemma discoveries
...
Run 10 (max waves):    Store = corpus + ALL previous models' discoveries
```

**What this means for each run:**
- Workers call `search_corpus()` and `get_peer_insights()` — they see EVERYTHING stored by all previous models
- A model doesn't re-discover what an earlier model already found. It builds on it.
- Each model's unique contribution is measurable: `new findings stored = total after run N - total after run N-1`
- Contradictions between models are explicitly capturable: Model B calls `check_contradictions` on Model A's findings
- The serendipity wave gets richer with every run — more material to find cross-domain connections in
- **The final store after all 10 runs is the deliverable** — a knowledge base assembled by 7+ architectures

**Execution order is strategic.** Early models lay the factual foundation. Middle models find what the foundation missed. Late models synthesize, contradict, and fill gaps.

**Per-run metrics track marginal contribution:**
- `findings_before`: store row count before this run
- `findings_after`: store row count after this run
- `delta`: what this model uniquely added
- `contradictions_found`: findings that conflict with previous models
- `gaps_filled`: topics that had low coverage before this run, now addressed

---

## Complete Variable Inventory

Every parameter that can be altered between tests, organized by subsystem.

### A. Orchestration (MCPSwarmConfig)

| # | Variable | Default | Range / Options | What it controls |
|---|----------|---------|-----------------|-----------------|
| 1 | `max_workers` | 7 | 1–20 | Number of parallel specialist agents |
| 2 | `max_waves` | 3 | 1–10 | Iterative refinement passes over the store |
| 3 | `convergence_threshold` | 5 | 1–50 | New findings per wave below which the swarm stops early |
| 4 | `temperature` | 0.3 | 0.0–1.0 | Worker sampling temperature |
| 5 | `max_tokens` | 4096 | 1024–16384 | Max tokens per worker LLM response |
| 6 | `report_max_tokens` | 8192 | 2048–32768 | Max tokens for final report generation |
| 7 | `enable_serendipity_wave` | True | True/False | Cross-domain discovery pass (always True for our tests) |

### B. Model Selection (local uncensored only)

| # | Variable | Default | Range / Options | What it controls |
|---|----------|---------|-----------------|-----------------|
| 8 | `model` | `huihui-ai/Qwen3.5-32B-abliterated` | See model table below | Which local uncensored LLM runs the workers |
| 9 | `api_base` | `http://localhost:8000/v1` | localhost only | vLLM endpoint (localhost guard enforced) |
| 10 | `VLLM_DTYPE` | auto | auto, float16, bfloat16, float8 | Quantization level — trades VRAM for quality |
| 11 | `VLLM_MAX_MODEL_LEN` | 32768 | 8192–131072 | Max context length per request |
| 12 | `VLLM_GPU_UTIL` | 0.92 | 0.5–0.95 | GPU memory utilization ceiling |

### C. Angle Strategy

| # | Variable | Default | Range / Options | What it controls |
|---|----------|---------|-----------------|-----------------|
| 13 | `required_angles` | [] (auto-detect) | Hand-crafted list vs empty | Whether angles are preset or LLM-detected |
| 14 | Angle detection `max_angles` | 6 | 2–12 | How many angles the LLM can detect |
| 15 | Corpus preview for detection | 30,000 chars | 5K–full corpus | How much material the LLM sees when detecting angles |
| 16 | Merge strategy | required-first | required-first, interleave | How preset and detected angles combine |

### D. Worker Tool Parameters (hardcoded defaults inside tools)

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 17 | `search_corpus.max_results` | 15 | How many store hits a worker sees per search |
| 18 | `get_peer_insights.max_results` | 10 | How many cross-angle findings per query |
| 19 | `get_corpus_section.max_chars` | 8,000 | Chunk size when reading assigned section |
| 20 | `_extract_terms.top_k` | 20 (search) / 10 (contradictions) | Keyword extraction breadth |
| 21 | Same-angle boost | 1.2× | Relevance boost for findings from same angle |

### E. RAG (rag.py — hive memory)

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 22 | `query_hive.top_k` | 5 | Max cross-angle entries returned |
| 23 | `query_hive.min_score` | 1.5 | Minimum relevance threshold |
| 24 | `extract_concepts.top_k` | 15 | Number of key terms extracted per query |

### F. Serendipity Wave

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 25 | Serendipity temperature | 0.5 (hardcoded) | Creativity level for cross-domain agent |
| 26 | Serendipity worker count | 1 (hardcoded) | How many cross-domain agents run |
| 27 | Serendipity prompt | hardcoded | What the cross-domain agent looks for |

### G. System Prompt

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 28 | Worker system prompt | hardcoded in `_build_system_prompt()` | Instructions, rules, exploration strategy |
| 29 | Report generation prompt | hardcoded in `_generate_report()` | How findings become a final document |

### H. External Tool Set (NEW — not yet implemented)

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 30 | External tools available | None (store-only today) | Which web/academic/forum/OSINT tools workers can call |
| 31 | External tool call budget | Unlimited | Max external API calls per worker per wave |
| 32 | External tool priority | None | Whether workers prefer store or external first |

### I. ConditionStore

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 33 | Storage backend | In-memory DuckDB | Persistence and query performance |
| 34 | `min_confidence` (export) | 0.0 | Floor for findings included in reports |
| 35 | `max_rows` (export) | None (all) | Cap on findings exported for report |
| 36 | **Persistence mode** | **reset per run** | **NEW: persist (accumulate) vs reset between runs** |
| 37 | **Source tagging** | **none** | **NEW: tag each finding with which model/run produced it** |

### J. Multi-Endpoint (8×H200 scaling)

| # | Variable | Default | What it controls |
|---|----------|---------|-----------------|
| 38 | `SWARM_ENDPOINTS` | single localhost | Comma-separated vLLM endpoints for multi-GPU |
| 39 | `SWARM_MODEL_N` | same model | Per-endpoint model override (epistemic diversity) |
| 40 | `SWARM_QUEEN_ENDPOINT` | first endpoint | Which GPU handles report generation |
| 41 | `SWARM_SERENDIPITY_ENDPOINT` | last endpoint | Which GPU handles serendipity |

### K. Not Currently Configurable (but could be)

| # | Variable | What it would control |
|---|----------|----------------------|
| 42 | Per-worker model | Different models for different angles |
| 43 | Per-wave model | Escalate model quality as waves progress |
| 44 | Per-phase model | Different model for report generation vs worker exploration |
| 45 | Multiple serendipity workers | More than one cross-domain agent |
| 46 | Corpus ingestion strategy | Paragraphs vs sections vs fixed-size chunks |
| 47 | Worker collaboration mode | Oblivious (current) vs aware (workers know about each other) |
| 48 | Finding deduplication | Whether the store deduplicates similar findings across workers |
| 49 | Confidence decay | Whether older findings lose confidence over waves |
| 50 | External source weighting | Whether external findings get different confidence than corpus findings |

---

## Available Local Uncensored Models

All served via vLLM on 1×H200 (141GB VRAM). Localhost-only — the swarm guard rejects remote endpoints.

### Tier 1: Already in the codebase / tested

| Model | HuggingFace ID | Params | Dtype | VRAM (weights) | Max Context | Architecture |
|-------|---------------|--------|-------|----------------|-------------|-------------|
| **Qwen3.5-32B-abl** | `huihui-ai/Qwen3.5-32B-abliterated` | 32B | FP16 | ~64GB | 32K | Qwen (dense) |
| **Llama-3.3-70B-abl** | `huihui-ai/Llama-3.3-70B-Instruct-abliterated` | 70B | FP8 | ~70GB | 128K | Llama (dense) |
| **Gemma-4-31B-unc** | `TrevorJS/gemma-4-31B-it-uncensored` | 31B | FP16 | ~62GB | 128K | Gemma (dense) |

### Tier 2: Available from huihui-ai collections (vLLM-compatible, abliterated)

| Model | HuggingFace ID | Params | Dtype | VRAM (weights) | Architecture | Notes |
|-------|---------------|--------|-------|----------------|-------------|-------|
| **GLM-4.7-abl** | `huihui-ai/GLM-4.7-*-abliterated` | 9–32B | FP16 | ~18–64GB | GLM (Zhipu) | Chinese-origin, strong reasoning |
| **Qwen3-Next-abl** | `huihui-ai/Qwen3-Next-*-abliterated` | varies | FP16 | varies | Qwen-Next | Newer Qwen architecture |
| **Mistral-Small-4-abl** | `huihui-ai/Mistral-Small-4-*-abliterated` | 24B | FP16 | ~48GB | Mistral | European-trained, different biases |
| **Kimi-Linear-abl** | `huihui-ai/Kimi-Linear-*-abliterated` | 48B | FP8 | ~48GB | Linear attention | Constant memory — no KV cache scaling |
| **Devstral-abl** | `huihui-ai/Devstral-*-abliterated` | ~24B | FP16 | ~48GB | Mistral (code) | Code-focused but general-capable |

### Tier 3: Other abliterated / uncensored (vLLM-compatible)

| Model | HuggingFace ID | Params | Dtype | VRAM (weights) | Architecture | Notes |
|-------|---------------|--------|-------|----------------|-------------|-------|
| **DeepSeek-R1-Distill-32B-unc** | `richardyoung/Deepseek-R1-Distill-Qwen-32b-uncensored` | 32B | FP16 | ~64GB | Qwen (R1-distilled) | Reasoning model — explicit chain-of-thought |
| **Hermes-3-Llama-3.1-70B** | `NousResearch/Hermes-3-Llama-3.1-70B` | 70B | FP8 | ~70GB | Llama | NousResearch fine-tune, tool-calling trained |
| **Gemma-4-E2B-unc** | `TrevorJS/gemma-4-E2B-it-uncensored` | 5B | FP16 | ~10GB | Gemma (small) | Tiny — speed baseline |

### VRAM Budget on 1×H200 (141GB)

| Config | Weights | KV Cache | Workers | Max Context/Worker |
|--------|---------|----------|---------|-------------------|
| 32B FP16 | 64GB | 77GB | 8 concurrent | ~32K |
| 70B FP8 | 70GB | 71GB | 6 concurrent | ~24K |
| 32B FP16 + 5B FP16 | 74GB | 67GB | mixed | 32K + 128K |

---

## Test Scenarios

### How Accumulation Works

Each scenario inherits the persistent ConditionStore from all previous runs. Workers in Run N see every finding from Runs 1 through N-1 via `search_corpus()` and `get_peer_insights()`.

**New implementation requirements for accumulation:**
1. **Source tagging** — every finding gets a `source_model` field (e.g., `qwen3.5-32b-abl`) and a `source_run` integer. This lets us query "what did Model X uniquely contribute?"
2. **Store persistence** — the DuckDB file persists on disk between runs. Each run opens the same `.duckdb` file.
3. **Delta tracking** — before each run starts, snapshot `SELECT COUNT(*) FROM conditions`. After the run, diff.
4. **No deduplication** — if Model B re-derives a finding that Model A already stored, store it anyway with Model B's source tag. The presence of convergent findings (same conclusion, different source model) is itself a signal — it means two architectures independently agree.

---

### Run 1: Qwen3.5-32B-abliterated — Foundation Layer

**Role in the sequence:** First mover. Lays the factual foundation from the 96K corpus + external tools. Everything it finds becomes the baseline that later models build on.

**What it sees:** Raw 96K corpus + empty store (no prior findings).

| Variable | Value |
|----------|-------|
| `VLLM_MODEL` | `huihui-ai/Qwen3.5-32B-abliterated` |
| `VLLM_DTYPE` | auto (FP16) |
| `VLLM_MAX_MODEL_LEN` | 32768 |
| max_workers | 8 |
| max_waves | 3 |
| temperature | 0.3 |
| max_tokens | 4096 |
| convergence_threshold | 5 |
| required_angles | 8 compound-specific angles from `h200_test/angles.py` |
| serendipity | on |
| external tools | ALL |
| corpus | full 96K |

**Expected contribution:** Broad factual coverage. Qwen is the best-documented model in our setup. Expect 150–200 findings covering the core protocol — dosages, timing, compounds. This creates the skeleton that later models flesh out.

---

### Run 2: Llama-3.3-70B-Instruct-abliterated — Depth Pass

**Role in the sequence:** Reads Qwen's findings, then goes deeper. 70B parameters mean stronger reasoning — it should identify gaps in Qwen's coverage and fill them. Llama's different training data (Meta's web crawl vs Alibaba's) means it may have different pharmacological knowledge baked in.

**What it sees:** 96K corpus + Qwen's ~150–200 findings.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `huihui-ai/Llama-3.3-70B-Instruct-abliterated` | **different architecture + size** |
| `VLLM_DTYPE` | float8 | **FP8 quantization** |
| max_workers | 6 | **fewer** — less KV headroom at 70B |
| Everything else | Same as Run 1 | |

**Expected contribution:** Deeper reasoning on mechanisms. Where Qwen stored "testosterone dose: 500mg/week", Llama should add "why 500mg — aromatization rate at this dose produces X pg/mL estradiol, which is within the therapeutic window because…". Also: findings from domains where Meta's training data is richer (English-language forums, Reddit bodybuilding communities).

**Accumulation metric:** `delta` — how many genuinely new findings did Llama add that Qwen missed? If delta is small, 70B scale doesn't help. If delta is large but on the same topics, Llama is just rephrasing. If delta introduces new topics, that's epistemic diversity.

---

### Run 3: Gemma-4-31B-uncensored — Google's Perspective

**Role in the sequence:** Third architecture, third training corpus. Google's model was trained on different web data — different academic paper coverage, different forum exposure. At 31B it's the same scale as Qwen so any differences are architectural, not parameter-count.

**What it sees:** 96K corpus + Qwen's findings + Llama's additions.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `TrevorJS/gemma-4-31B-it-uncensored` | **Google architecture** |
| Everything else | Same as Run 1 | |

**Expected contribution:** By this point the store has the broad strokes (Qwen) and deeper reasoning (Llama). Gemma's job is to find what both missed. Google's training data is heavy on PubMed, scholarly articles, and health-related web content — it may surface clinical trial data, specific study citations, or evidence-based dosing ranges that the other two didn't have in training.

**Accumulation metric:** `delta` should be smaller than Run 2 (the easy findings are taken). The interesting metric is `unique_topics` — does Gemma introduce entirely new subtopics?

---

### Run 4: DeepSeek-R1-Distill-32B-uncensored — Contradiction Hunter

**Role in the sequence:** The reasoning model. R1-distilled Qwen does explicit chain-of-thought before answering. Its role in the sequence is NOT to add more facts — three models have already covered the factual space. Its role is to **audit what's already there.** With reasoning chains, it should be the first model to heavily use `check_contradictions` — finding cases where Qwen said one thing, Llama said another, and the evidence supports one over the other.

**What it sees:** 96K corpus + 3 models' worth of accumulated findings (~300–400 rows).

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `richardyoung/Deepseek-R1-Distill-Qwen-32b-uncensored` | **reasoning model** |
| max_tokens | 8192 | **doubled** — reasoning traces need more tokens |
| Everything else | Same as Run 1 | |

**Expected contribution:** Contradiction detection. Confidence re-scoring. Evidence chains. The store should gain fewer new findings but higher-quality ones — findings that say "Models 1–3 stored conflicting dosage ranges for trenbolone. The pharmacokinetic data supports X because…". Also: the reasoning traces themselves are valuable. Tag them as `row_type='reasoning_trace'` in the store.

**Accumulation metric:** `contradictions_found` — this is the primary metric. If R1 doesn't use `check_contradictions` more than previous models, the reasoning training isn't helping the swarm.

---

### Run 5: GLM-4.7-abliterated — Alternate Knowledge Base

**Role in the sequence:** GLM (Zhipu AI) is trained on Chinese web + academic data. It has access to Chinese medical literature, Chinese bodybuilding forums, Traditional Chinese Medicine pharmacology databases that no English-origin model was trained on. By Run 5, the store is well-populated — GLM's job is to bring in knowledge from a genuinely different training distribution.

**What it sees:** 96K corpus + 4 models' worth of findings.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `huihui-ai/GLM-4.7-*-abliterated` (largest FP16) | **Chinese architecture** |
| Everything else | Same as Run 1 | |

**Expected contribution:** Cross-cultural pharmacological knowledge. Chinese sports medicine takes a different approach to peptides, GH, and insulin — different dosing philosophies, different ancillary compounds (e.g., astragalus for liver protection vs TUDCA). May also surface actovegin data that English models lack — actovegin is more widely studied and prescribed in China/Russia.

**Accumulation metric:** `unique_topics` — are GLM's new findings on topics the other 4 models never touched? That would prove the training data diversity thesis.

---

### Run 6: Mistral-Small-4-abliterated — European Perspective + More Workers

**Role in the sequence:** Mistral is smaller (24B) but this is a feature, not a bug. At 48GB weights, it leaves 93GB for KV — enough for 10 workers instead of 8. More workers means more angles can run simultaneously. Also: European training data = WADA/anti-doping research, European sports medicine, different regulatory context.

**What it sees:** 96K corpus + 5 models' worth of findings.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `huihui-ai/Mistral-Small-4-*-abliterated` | **European, smaller** |
| max_workers | 10 | **more workers** |
| required_angles | Same 8 + 2 auto-detected from store gaps | **10 angles** |
| Everything else | Same as Run 1 | |

**Expected contribution:** The 2 auto-detected angles are the innovation here. By Run 6, `get_research_gaps()` returns a rich list of under-covered topics. Mistral auto-detects angles specifically targeting those gaps. This tests whether the gap-detection → angle-creation pipeline works as a feedback loop.

Also: European WADA-adjacent knowledge. Detection windows, clearance timelines, competition timing — information that bodybuilders need but that US-trained models may not emphasize.

**Accumulation metric:** `gap_coverage_before` vs `gap_coverage_after` — did the 2 auto-detected angles actually fill the gaps they were targeting?

---

### Run 7: Hermes-3-Llama-3.1-70B — Tool-Calling Specialist

**Role in the sequence:** By Run 7, the store has ~500+ findings. The bottleneck is no longer finding new facts — it's efficiently navigating what's already there plus making good external search queries. Hermes-3 was fine-tuned specifically for tool calling. It should make better `search_corpus` queries, more precise `store_finding` calls, and — critically — actually use `check_contradictions` and external search tools more effectively than models that weren't trained for structured tool use.

**What it sees:** 96K corpus + 6 models' worth of findings.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `NousResearch/Hermes-3-Llama-3.1-70B` | **tool-calling fine-tune** |
| `VLLM_DTYPE` | float8 | **FP8** |
| max_workers | 6 | **fewer — 70B** |
| Everything else | Same as Run 1 | |

**Expected contribution:** Better tool usage patterns. Previous models may have called `search_corpus("insulin")` with vague queries. Hermes should call `search_corpus("Humalog onset time relative to post-workout dextrose bolus, Milos Sarcev protocol")` — specific, targeted, tool-aware queries. The delta in findings may be modest, but the quality and evidence-grounding of new findings should be measurably higher.

**Accumulation metric:** `tool_calls_per_worker` and `external_tool_calls_per_worker` — does Hermes make more and better tool calls? Also: `average_finding_confidence` — are Hermes's findings higher-confidence?

---

### Run 8: Kimi-Linear-abliterated — Unlimited Context

**Role in the sequence:** Linear attention means no KV cache scaling. Where every other model was limited to 32K context per worker, Kimi can hold 128K+ in context with zero additional VRAM. This means each worker can read the ENTIRE accumulated store in one pass. No chunking, no multi-pass. The worker sees everything at once — all 500+ findings from 7 models plus the full 96K corpus.

**What it sees:** Everything. Literally everything in one context window.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | `huihui-ai/Kimi-Linear-*-abliterated` | **linear attention** |
| `VLLM_DTYPE` | float8 | |
| `VLLM_MAX_MODEL_LEN` | 131072 | **4× longer** |
| `get_corpus_section.max_chars` | 96000 | **entire corpus in one read** |
| `search_corpus.max_results` | 100 | **all findings visible** |
| `get_peer_insights.max_results` | 100 | **all peer data visible** |
| Everything else | Same as Run 1 | |

**Expected contribution:** Synthesis over discovery. With the entire accumulated knowledge base visible in one context window, Kimi should produce the most coherent cross-referencing — seeing how Qwen's insulin timing connects to Llama's trenbolone clearance data connects to GLM's actovegin research connects to R1's contradiction analysis. The serendipity wave with Kimi should be especially productive.

**Accumulation metric:** `serendipity_findings_confidence` — does seeing everything at once produce higher-confidence cross-domain connections?

---

### Run 9: A/B Epistemic Diversity — Two Models Simultaneously

**Role in the sequence:** Instead of one model, run two (Qwen-32B + Gemma-31B) via the multi-endpoint bridge, round-robin across workers. This tests whether intra-run model diversity produces different results than the sequential accumulation approach. Some workers reason with Qwen, others with Gemma, all writing to the same store. They see each other's findings in real time within the same wave.

**What it sees:** 96K corpus + 8 models' worth of accumulated findings.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `SWARM_ENDPOINTS` | `http://localhost:8000/v1,http://localhost:8001/v1` | **two endpoints** |
| `SWARM_MODEL_0` | `huihui-ai/Qwen3.5-32B-abliterated` | |
| `SWARM_MODEL_1` | `TrevorJS/gemma-4-31B-it-uncensored` | |
| `VLLM_MAX_MODEL_LEN` | 16384 both | **halved** — VRAM shared |
| Everything else | Same as Run 1 | |

**VRAM budget:** 64GB (Qwen) + 62GB (Gemma) = 126GB weights. Tight — 15GB for KV. May need FP8 for one model.

**Expected contribution:** Tests whether simultaneous diversity (two models in one run) adds value beyond sequential diversity (each model in its own run). The real-time cross-pollination — Qwen worker stores a finding, Gemma worker immediately sees it in the same wave — is something the sequential runs can't do.

**Accumulation metric:** Compare Run 9 delta to the sum of Runs 1 + 3 deltas. If Run 9 produces more unique findings than the sum of the two models running independently, simultaneous diversity has compounding effects.

---

### Run 10: Final Synthesis — Max Waves with Best Model

**Role in the sequence:** The capstone. Uses the strongest model (determined by which model produced the highest-confidence findings and best delta in Runs 1–9) and runs 6 waves with strict convergence. By now the store has 600–800+ findings from 8+ model architectures. This run's job is to exhaustively fill any remaining gaps, resolve any remaining contradictions, and produce the final report from the complete knowledge base.

**What it sees:** Everything from all 9 previous runs.

| Variable | Value | Changed from Run 1 |
|----------|-------|---------------------|
| `VLLM_MODEL` | **Best performer from Runs 1–9** | **data-driven selection** |
| max_waves | 6 | **doubled** |
| convergence_threshold | 3 | **stricter** |
| report_max_tokens | 32768 | **4× for comprehensive report** |
| Everything else | Same as Run 1 | |

**Expected contribution:** Minimal delta in new findings (the knowledge base is nearly complete). Primary output is the **final report** — generated from the complete accumulated store. The report should be dramatically better than any single-run report because it has access to findings from 8+ model architectures, contradiction analysis, cross-domain connections, and gap-filling from multiple training data distributions.

**Accumulation metric:** `convergence_wave` — which wave does the swarm stabilize? If it converges at wave 2 (out of 6), the knowledge base is saturated. If it's still producing new findings at wave 6, there's more to find and more runs would help.

---

## Execution Order

The order is fixed and strategic — each run has a defined role:

| Run | Model | Role | Builds On |
|-----|-------|------|-----------|
| 1 | Qwen3.5-32B-abl | **Foundation** — lay broad factual base | Empty store |
| 2 | Llama-3.3-70B-abl | **Depth** — deeper reasoning, fill gaps | Run 1 |
| 3 | Gemma-4-31B-unc | **Diversity** — third architecture, third training data | Runs 1–2 |
| 4 | DeepSeek-R1-32B-unc | **Audit** — reasoning model finds contradictions | Runs 1–3 |
| 5 | GLM-4.7-abl | **Cross-cultural** — Chinese training data | Runs 1–4 |
| 6 | Mistral-Small-4-abl | **Gap-fill** — auto-detect missing angles, European data | Runs 1–5 |
| 7 | Hermes-3-70B | **Tool mastery** — better search queries, better evidence | Runs 1–6 |
| 8 | Kimi-Linear-abl | **Full-context synthesis** — sees everything at once | Runs 1–7 |
| 9 | Qwen + Gemma (A/B) | **Simultaneous diversity** — real-time cross-pollination | Runs 1–8 |
| 10 | Best from 1–9 | **Final pass** — fill last gaps, generate comprehensive report | Runs 1–9 |

---

## Implementation Requirements

1. **`launch_vllm.sh`** — already supports `VLLM_MODEL`, `VLLM_DTYPE`, `VLLM_MAX_MODEL_LEN` env vars. No changes needed.

2. **ConditionStore persistence** — change from in-memory to file-backed DuckDB. Add `store_path` config option (e.g., `experiments/accumulated_store.duckdb`). Each run opens the same file.

3. **Source tagging** — add `source_model TEXT` and `source_run INTEGER` columns to the conditions table. Every `store_finding()` call tags the finding with the current model and run number.

4. **Delta tracking** — `run_mcp_experiments.py` snapshots row count before/after each run. Writes `{ "run": N, "model": "...", "findings_before": X, "findings_after": Y, "delta": Y-X }` to a JSON log.

5. **`build_worker_tools()`** — add `extra_tools: list` parameter for external tools.

6. **`create_worker_agent()`** — accept `extra_tools` and pass through.

7. **`MCPSwarmConfig`** — add: `extra_tools: list`, `serendipity_temperature: float`, `store_path: str`, `source_model: str`, `source_run: int`.

8. **System prompt update** — add: "You have access to findings from previous research runs by other models. Use `search_corpus` and `get_peer_insights` to see what has already been discovered. Focus on what's MISSING or WRONG, not on re-stating what's already known. You also have web search, academic database, forum, and content extraction tools. Use them when the research database doesn't have what you need. Store everything you find externally as findings."

9. **`run_mcp_experiments.py`** — add `--model`, `--api-base`, `--dtype`, `--run-number`, `--store-path` flags. Add `--resume` flag to continue accumulation from a previous store.

10. **Cost tracking** — no monetary cost for local models. Track GPU-hours, tokens generated, and external API calls.

---

## Measurement Framework

### Per-run metrics

| Metric | What it tells us |
|--------|-----------------|
| `findings_before` | Store state at run start |
| `findings_after` | Store state at run end |
| `delta` | This model's unique contribution (after - before) |
| `new_topics` | Topics this model introduced that no previous model covered |
| `contradictions_found` | Findings that conflict with previous models' findings |
| `convergent_findings` | Findings that independently confirm what a previous model already stored |
| `check_contradictions` call rate | Whether this model actually audits prior work |
| Tool call breakdown (store vs external) | Worker behavior pattern |
| External tool breakdown (which APIs) | What external sources this model gravitates toward |
| Time per wave | Wall-clock per iteration |
| GPU utilization | Whether vLLM is saturated or idle |
| Convergence wave | When this model's workers stopped finding new things |
| Serendipity findings + confidence | Cross-domain connections (richer with more accumulated data) |
| Tokens generated per worker | Model efficiency |
| Average finding confidence | Quality signal |

### Cumulative metrics (tracked across the full sequence)

| Metric | What it tells us |
|--------|-----------------|
| Total store size over time | Knowledge growth curve — does it plateau? |
| Marginal delta per run | Diminishing returns — when does adding another model stop helping? |
| Topic coverage breadth | Are we converging on complete protocol coverage? |
| Contradiction resolution rate | Are later models resolving contradictions found by R1? |
| Cross-architecture agreement | When 3+ architectures independently agree, how reliable is the finding? |
| Serendipity quality over time | Do cross-domain connections improve as the store grows? |
| Final report quality vs Run 1 report | How much better is the accumulated-knowledge report? |

### Cross-model comparisons

- **Runs 1 vs 2 vs 3:** Three architectures — delta comparison reveals training data differences
- **Runs 1 vs 2:** 32B FP16 vs 70B FP8 — does scale produce meaningfully different findings?
- **Runs 1 vs 4:** Same Qwen base — does R1 training change what the model contributes to the swarm?
- **Runs 2 vs 7:** Same Llama 70B — does Hermes tool-calling training improve swarm behavior?
- **Run 8:** Does seeing everything at once (linear attention) produce different synthesis than chunked reading?
- **Run 9 vs (Run 1 + Run 3):** Does simultaneous diversity beat sequential diversity?
- **Knowledge growth curve:** Plot `total_findings` vs run number. Is it linear, logarithmic, or s-curve?
