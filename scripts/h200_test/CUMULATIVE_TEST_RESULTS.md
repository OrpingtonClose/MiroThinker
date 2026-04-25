# Cumulative Model Test Results

5 SOTA abliterated models processed the same bodybuilding research query sequentially against a persistent DuckDB store. Knowledge accumulated run-to-run. This is a dress rehearsal for the 8×H200 deployment where all models run simultaneously with a shared ConditionStore.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Corpus | 96K chars (Milos Sarcev insulin + extended protocols) |
| Workers | 5 per run |
| Max waves | 3 |
| Temperature | 0.3 |
| Max tokens | 4096 |
| Convergence threshold | 5 |
| Serendipity | Always on |
| Store | File-backed DuckDB (same file across all runs) |
| GPU | 1×H200 (Vast.ai instance 35390779) |
| vLLM | 0.19.1, auto tool choice enabled |

## Run Summary

| Run | Model | Architecture | Parser | Total Findings | Delta | Elapsed | Report Size |
|-----|-------|-------------|--------|---------------|-------|---------|-------------|
| 1 | Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated | MoE (A3B), Claude-distilled | qwen3_xml | 661 | +661 | 53.5s | 20,442 chars |
| 2 | Gemma-4-E2B-it-abliterated-v2 | Dense, Google architecture | gemma4 | 1,153 | +492 | 46.1s | 10,359 chars |
| 3 | GLM-4.7-Flash-abliterated | MoE, Chinese architecture (Zhipu) | glm47 | 1,767 | +614 | 103.3s | 15,018 chars |
| 4 | Qwen3.5-27B-abliterated | Dense Qwen | qwen3_xml | 2,396 | +629 | 247.6s | 13,455 chars |
| 5 | Kimi-Linear-48B-A3B-Instruct-abliterated | MoE (A3B), linear attention | kimi_k2 | 2,999 | +603 | 38.0s | 4,914 chars |

## Phase Timing Breakdown

| Run | Model | Ingestion | Wave 1 | Serendipity | Report | Total |
|-----|-------|-----------|--------|-------------|--------|-------|
| 1 | Qwen3.6 | 6.0s | 17.2s | 9.3s | 21.0s | 53.5s |
| 2 | Gemma-4-E2B | 3.4s | 13.7s | 14.2s | 14.8s | 46.1s |
| 3 | GLM-4.7 | 13.3s | 46.3s | 17.7s | 26.1s | 103.3s |
| 4 | Qwen3.5-27B | 55.2s | 103.8s | 24.9s | 63.7s | 247.6s |
| 5 | Kimi-Linear | 10.8s | 11.6s | 8.3s | 7.3s | 38.0s |

## Key Findings

### 1. Every model adds roughly the same marginal delta

Despite wildly different architectures (MoE vs dense, linear attention vs standard, Claude-distilled vs native Qwen), each model added 492–661 findings to the store. The marginal contribution didn't diminish across 5 runs — the 5th model (Kimi-Linear) added 603 findings, comparable to the 1st (Qwen3.6 at 661). This suggests the corpus has not been exhausted and additional models would continue to extract new information.

**Implication for 8×H200**: All 8 model slots would likely be productive. No evidence of saturation at 5 models.

### 2. Speed varies 6.5× across architectures

Kimi-Linear (38s) was 6.5× faster than Qwen3.5-27B (248s) on the same task. The linear attention architecture eliminates KV cache scaling, allowing faster inference. GLM-4.7 was moderately slow (103s) despite being a Flash variant.

**Speed ranking**: Kimi-Linear (38s) > Gemma-4-E2B (46s) > Qwen3.6 (53s) > GLM-4.7 (103s) > Qwen3.5-27B (248s)

### 3. Convergence is immediate — all runs converge at wave 1

Every run converged at wave 1 with 0 new findings below the threshold of 5. Workers read the corpus and peer insights but don't call `store_finding` during their agent loop — the findings come from corpus ingestion during the ingestion phase. The wave/convergence mechanism isn't exercised because the ingestion phase front-loads all the data.

**Implication**: The current architecture functions as a "corpus re-ingestion + report generation" pipeline rather than a multi-wave iterative research system. Workers analyze and synthesize but don't write back discoveries.

### 4. Report quality varies significantly by model

| Model | Report Quality | Notes |
|-------|---------------|-------|
| Qwen3.6 (20K chars) | Longest, most detailed | Starts with thinking traces but delivers comprehensive protocol |
| GLM-4.7 (15K chars) | Structured analysis | Includes explicit analytical framework before content |
| Qwen3.5-27B (13K chars) | Reasoning-heavy | Shows thinking process, slightly truncated due to context overflow |
| Gemma-4-E2B (10K chars) | Clean, well-formatted | Best markdown structure, proper headings |
| Kimi-Linear (5K chars) | Shortest, most concise | Executive summary style, least detail |

### 5. Context overflow hits at Run 4

By Run 4, the accumulated corpus (484K chars) exceeded the 32K token context window for Qwen3.5-27B. One worker failed with a 400 error: `28,673 input tokens + 4,096 output tokens = 32,769 > 32,768 limit`. The run still completed because other workers succeeded.

**Implication for 8×H200**: By Run 8, the corpus will be even larger. Either context windows need to increase, or the system needs smarter corpus chunking. Kimi-Linear's linear attention would handle this better — it can see the entire corpus in one pass.

### 6. Store composition is 91% corpus ingestion

Of 3,305 total rows in the store:
- 2,999 are findings (91% from corpus_section ingestion, not worker analysis)
- 276 are tool_call records
- 30 are raw entries
- Only 1 finding came from wave_1 worker analysis
- 0 findings came from serendipity waves

Workers use tools (`search_corpus`, `get_peer_insights`, `check_contradictions`) during their agent loop, but the findings they produce go into the report, not back into the store. The store serves as a read-only corpus index for workers, not a write-back knowledge base.

### 7. Angle detection degraded to "part N" splitting

Instead of detecting diverse angles (insulin protocols, pharmacokinetics, ancillaries, micronutrients, ramping strategy), the angle detection produced:
- "Insulin & GH protocols — Milos Sarcev framework (part 1–5)"

This happened because the preset angles from `angles.py` were designed for the original 28K corpus, but the corpus grew to 96K+ with extended protocols. The LLM-based angle detection defaulted to splitting the dominant topic into numbered parts rather than identifying orthogonal research domains.

## Findings Distribution

### By Angle

| Angle | Count | % |
|-------|-------|---|
| Insulin & GH protocols (part 1) | 790 | 26.4% |
| Insulin & GH protocols (part 2) | 520 | 17.4% |
| Insulin & GH protocols (part 4) | 417 | 13.9% |
| Insulin & GH protocols (part 3) | 393 | 13.1% |
| Insulin & GH protocols (part 5) | 393 | 13.1% |
| Testosterone & Trenbolone pharmacokinetics | 161 | 5.4% |
| Oral compounds (part 1) | 149 | 5.0% |
| Oral compounds (part 2) | 139 | 4.6% |
| Ancillaries (part 2) | 128 | 4.3% |
| Ancillaries (part 1) | 125 | 4.2% |
| Cross-domain connections | 41 | 1.4% |
| Boldenone & EQ interactions | 8 | 0.3% |

### By Time (mapping to runs)

| Timestamp | Count | Run |
|-----------|-------|-----|
| 07:50 | 223 | (pre-test DeepSeek-R1 remnant) |
| 08:40 | 548 | Run 1: Qwen3.6 |
| 08:53 | 503 | Run 2: Gemma-4-E2B |
| 08:59–09:01 | 709 | Run 3: GLM-4.7 |
| 09:09–09:11 | 714 | Run 4: Qwen3.5-27B |
| 09:39 | 608 | Run 5: Kimi-Linear |

## Infrastructure Notes

### Kimi-Linear tokenizer fix

Kimi-Linear's custom tokenizer (`tokenization_kimi.py`) imports `bytes_to_unicode` from `transformers.models.gpt2.tokenization_gpt2`, which was removed in transformers 5.x. The function was inlined directly into the cached tokenizer files on the H200 to work around this incompatibility.

### Model storage on H200

All models were served from `/dev/shm/hf_hub` (tmpfs) to avoid the 80GB root disk constraint. Previous models must be deleted before loading new ones — accumulated models filled 188GB of shared memory.

### vLLM tool-call parsers by model

| Model Family | Parser |
|-------------|--------|
| Qwen 3.x | qwen3_xml |
| Gemma 4 | gemma4 |
| GLM 4.7 | glm47 |
| Kimi-Linear | kimi_k2 |

## Recommendations for 8×H200

1. **All 5 tested models are productive** — no saturation detected. Fill all 8 GPU slots.
2. **Prioritize Kimi-Linear and Gemma-4-E2B for speed** — they complete in <50s vs 100–250s for others.
3. **Fix the write-back gap** — workers should call `store_finding` during their agent loop, not just produce reports. This would make the cumulative knowledge genuinely grow with each wave, not just with corpus re-ingestion.
4. **Fix angle detection** — the "part N" splitting defeats the purpose of preset angles. Either enforce the preset angles from `angles.py` without LLM modification, or improve the angle merging logic.
5. **Handle context overflow** — by Run 8, the corpus will exceed 32K tokens. Use Kimi-Linear for later runs (linear attention handles long context), or implement smarter corpus chunking that prioritizes high-confidence findings.
6. **Add source_model tagging** — currently there's no way to query "what did Model X uniquely contribute" because findings don't carry model provenance. Adding a `source_model` column to the ConditionStore would enable epistemic diversity analysis.
7. **Multi-endpoint bridge for simultaneous runs** — on 8×H200, use `multi_endpoint_bridge.py` with `SWARM_MODEL_0` through `SWARM_MODEL_7` to route workers to different GPUs for real-time cross-pollination.
