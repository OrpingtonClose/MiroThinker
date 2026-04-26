# SQL-Agentic Architecture Lessons

Findings from the SQL-agentic swarm prototype on 8×H200 with DeepSeek V4 Pro (TP=8) and the Flock DuckDB extension. This document captures what works, what fails, and what the orchestrator should implement based on empirical results across 3 swarm runs (30 virtual agents, 491 findings, 18 research angles).

## 1. Core Insight: SQL IS the Agent

The fundamental discovery: every "virtual agent" in the swarm is a SQL query. SQL is the agent's **perception** (what it discovers in the ConditionStore), the LLM call is the agent's **cognition** (what it reasons about), and the trace row is the agent's **memory** (what it stores for others).

No Python orchestration overhead, no gossip compression, no threshold filters. Raw SQL discovery → LLM reasoning → trace → store, at 30 queries/min on optimized V4 Pro.

This replaces the broken GossipSwarm (Issue #263) entirely. Instead of 12 workers exchanging compressed summaries that destroy diversity, 30 specialized SQL queries each discover a different pattern in the corpus, reason about it via native Flock LLM functions, and write results back. The ConditionStore acts as the shared blackboard.

## 2. Flock SQL Dialect — What Works and What Doesn't

### 2.1 Function Reference (Empirically Verified)

| Function | Type | Purpose | Latency | Reliability |
|----------|------|---------|---------|-------------|
| `llm_complete` | Scalar | Evaluate/reason about each row independently | 4–25s | High (with correct syntax) |
| `llm_filter` | Semantic WHERE | Filter rows by semantic criteria (TRUE/FALSE) | 1–1.5s | High |
| `llm_reduce` | Aggregate | Synthesize grouped rows into one output | 1.5–37s | High |
| `llm_rerank` | Aggregate | Rank rows within a GROUP BY | 0.9–1.8s | Medium (intermittent -1 ID bug) |
| `llm_first` | Aggregate | Pick single best row per group | N/A | **Broken** — returns invalid row IDs with V4 Pro |
| `llm_last` | Aggregate | Pick single worst row per group | N/A | **Broken** — same invalid ID issue |
| `fusion_rrf` | Pure SQL | Reciprocal Rank Fusion across score columns | 0.01s | Perfect (no LLM involved) |
| `fusion_combsum` | Pure SQL | Additive score fusion | 0.01s | Perfect (no LLM involved) |

### 2.2 Critical Syntax Rules

**Named prompts with context_columns (CORRECT):**
```sql
llm_complete(
    {'model_name': 'v4pro'},
    {'prompt_name': 'my-prompt',
     'context_columns': [{'data': column_expression}]}
) AS result
```

**Inline prompt — literal string only (CORRECT):**
```sql
llm_reduce(
    {'model_name': 'v4pro'},
    {'prompt': 'Synthesize these findings into a summary.'}
) AS result
```

**SQL expression in prompt value (BROKEN — Flock rejects this):**
```sql
-- THIS FAILS: "prompt details struct should contain a single key value pair"
llm_complete(
    {'model_name': 'v4pro'},
    {'prompt': 'Analyze: ' || fact_column || ' and explain.'}
) AS result
```

**Mixed prompt + context_columns (BROKEN):**
```sql
-- THIS FAILS: Flock wants ONE key only in the dict
llm_complete(
    {'model_name': 'v4pro'},
    {'prompt': 'Evaluate this', 'context_columns': [{'data': fact}]}
) AS result
```

**Rule**: To include dynamic column values in a prompt, you MUST register a named prompt (with `{{fact}}` placeholder) and pass data via `context_columns`. You cannot use SQL `||` concatenation in the `prompt` value.

**Exception for `llm_reduce`**: The inline `{'prompt': 'literal string'}` format works for aggregate functions because they already receive all grouped rows as implicit context. But you still cannot use SQL expressions in the prompt value.

### 2.3 Context Columns for Multi-Column Data

When an agent needs multiple column values (e.g., two findings for contradiction mining), concatenate them in the `context_columns` expression:

```sql
{'context_columns': [{'data': fact_a || ' vs Finding B: ' || fact_b}]}
```

This is the only way to pass multiple dynamic values into a single prompt call. The template `{{fact}}` will receive the concatenated string.

### 2.4 llm_first / llm_last — Avoid These

`llm_first` and `llm_last` ask the LLM to return a `flock_row_id` identifying which row it picks. With V4 Pro, the model frequently returns `-1` or garbage IDs. This appears to be a model-specific formatting issue — the model doesn't output valid Flock row identifiers.

**Workaround**: Use `llm_rerank` instead. It reorders the GROUP BY results by LLM-assessed quality, and you can take the first/last row from the reranked output via SQL `LIMIT 1`.

### 2.5 llm_rerank — Intermittent Failures

`llm_rerank` works in most cases (3/4 agents succeeded) but occasionally returns `-1` row IDs. This seems to happen when the group contains rows with unusual content (very short facts, metadata-only rows). The orchestrator should:

1. Pre-filter groups to ensure minimum content quality
2. Retry on `-1` with a smaller group size
3. Fall back to score-based SQL ordering if rerank fails twice

## 3. Agent Design Patterns That Work

### 3.1 Six Phases of SQL-Agentic Intelligence

The 30 virtual agents naturally organize into 6 phases, each building on the previous:

**Phase 1 — Scalar Evaluation (llm_complete):** Each finding evaluated individually. Best for quality assessment, bridge-building, adversarial challenge.

**Phase 2 — Semantic Filtering (llm_filter):** Boolean classification of findings. Best for actionability, novelty, safety screening. Extremely fast (1–1.5s for 491 findings).

**Phase 3 — Group Synthesis (llm_reduce):** Aggregate findings by angle, corpus, or theme. This is the gossip replacement — instead of compressed worker summaries, the LLM sees raw findings and synthesizes directly.

**Phase 4 — Ranking (llm_rerank):** Order findings by importance within groups. Replaces subjective human prioritization.

**Phase 5 — Score Fusion (fusion_rrf, fusion_combsum):** Pure SQL, no LLM. Combines multiple score dimensions (confidence, novelty, specificity, actionability, trust) into a single composite ranking. Instant.

**Phase 6 — Deep Specialist Queries (llm_complete + llm_reduce):** The creative agents — contradiction mining, counterfactual probing, mechanism chain building, prediction generation. These use CTEs to discover patterns via SQL, then send them to the LLM for deep reasoning.

### 3.2 Virtual Agent Taxonomy

| Agent Type | Purpose | Flock Function | Example |
|------------|---------|----------------|---------|
| **Evaluator** | Assess individual findings | llm_complete | evaluator-new-corpus |
| **Filter** | Binary classification | llm_filter | actionability-filter, safety-sentinel |
| **Synthesizer** | Merge findings into summaries | llm_reduce | angle-synthesizer, grand-synthesizer |
| **Bridge** | Connect findings across domains | llm_complete | bridge-builder, desert-bridge-proposer |
| **Adversary** | Challenge and stress-test claims | llm_complete/reduce | devil-advocate, counterfactual-prober |
| **Meta-analyst** | Rank, prioritize, detect gaps | llm_rerank/reduce | gap-detector, research-prioritizer |
| **Quality-control** | Audit sources and scores | llm_complete/filter | credibility-auditor, safety-sentinel |
| **Specialist** | Deep domain-specific analysis | llm_reduce | mechanism-chain-builder, dose-response-analyst |
| **Fusion** | Combine scores mathematically | fusion_rrf/combsum | quality-fusion, score-combsum |

### 3.3 CTE Pattern for Context Management

Every aggregate agent must use a CTE to cap the number of findings sent to the LLM. Without this, Flock concatenates ALL grouped rows into one prompt, easily exceeding context limits.

```sql
WITH capped AS (
    SELECT angle, SUBSTR(fact, 1, 400) as fact,
           ROW_NUMBER() OVER (PARTITION BY angle ORDER BY confidence DESC) as rn
    FROM conditions
    WHERE row_type = 'finding' AND consider_for_use = TRUE
)
SELECT angle,
       llm_reduce(
           {'model_name': 'v4pro'},
           {'prompt_name': 'synthesize-angle',
            'context_columns': [{'data': fact}]}
       ) AS synthesis
FROM capped
WHERE rn <= 8
GROUP BY angle
```

**Rules of thumb for `--max-model-len 16384`:**
- `SUBSTR(fact, 1, 400)` per finding
- Max 8–12 findings per GROUP BY group
- Max 6 findings per corpus in cross-corpus queries
- LIMIT 5–8 for cross-join queries (e.g., contradiction mining)

## 4. vLLM Optimization — 10× Throughput

### 4.1 Baseline vs Optimized Config

| Parameter | Baseline | Optimized | Impact |
|-----------|----------|-----------|--------|
| `--max-model-len` | 65536 | 16384 | **Biggest single improvement.** Frees ~75% of KV cache for concurrent requests |
| `--gpu-memory-utilization` | 0.90 | 0.95 | Frees ~40 GB across 8 GPUs for KV cache |
| `--enforce-eager` | Yes | Removed | Enables CUDA graph compilation (51 graphs). 2–3× faster decode |
| `--enable-chunked-prefill` | No | Yes | Allows new requests to start prefilling while others decode |
| Concurrent requests | 10–15 | 37+ | Direct result of smaller KV cache + more memory |

### 4.2 Measured Results

- **Throughput**: 3/min → 30/min (10× improvement)
- **Simple query latency**: 2–3s → 0.19s
- **Complex reasoning latency**: 15s → 4–8s
- **CUDA graphs**: 51 compiled (first time working on H200 with V4 Pro)
- **Total run time for 30 agents**: ~5 minutes

### 4.3 When NOT to Use These Settings

- `--max-model-len 16384` only works if all prompts fit in 16K. If any agent needs longer context (e.g., full document analysis), this will cause silent truncation or errors.
- `--gpu-memory-utilization 0.95` leaves almost no headroom. If the model occasionally needs more memory for certain batch compositions, this can cause OOM. Monitor for sporadic failures.
- Removing `--enforce-eager` risks OOM during CUDA graph capture with very large models. If the model fails to start, add it back.
- `--enable-chunked-prefill` may slightly increase time-to-first-token for individual requests while improving overall throughput.

## 5. ConditionStore as Universal Blackboard

### 5.1 Schema That Works

The ConditionStore needs these row types for SQL-agentic operation:

| row_type | Purpose | Produced By |
|----------|---------|-------------|
| `finding` | Raw factual claims from corpora | Corpus ingestion |
| `evaluation` | V4 Pro quality assessment of a finding | Direct eval script |
| `trace` | Full provenance of an agent's execution | Swarm agents |
| `thought` | Intermediate reasoning | Workers |
| `synthesis` | Merged cross-corpus summaries | Synthesizer agents |
| `worker_metric` | Performance telemetry | Engine |

### 5.2 Gradient Flags That Matter

From the credibility-auditor results, these flags drive meaningful query selection:

- **confidence**: How certain the source is about the claim (0.0–1.0)
- **trust_score**: Source reliability rating
- **fabrication_risk**: Likelihood the finding is fabricated or hallucinated
- **novelty_score**: How surprising/new the finding is
- **specificity**: How precise and measurable the claim is
- **actionability**: Whether a practitioner can act on it directly

**Critical finding**: All 219 YouTube findings had default flag values (novelty=0.5, confidence=0.4, fabrication_risk=0.0) that fell just outside every query threshold. This caused the original Flock to fire 0 queries. The orchestrator MUST detect this condition and either (a) run a prescoring pass to compute real flags, or (b) adjust thresholds to match the actual data distribution.

### 5.3 Corpus Ingestion Pattern

The bodybuilding corpus (272 sections) was ingested by splitting on markdown headers:

```
## Section Title: Content text...
```

Each section becomes one `finding` row with:
- `fact`: Full section text
- `angle`: Inferred from chapter structure (e.g., chapter on amino acids → `amino-acid-biochemistry`)
- `source_ref`: Corpus filename
- `confidence`, `trust_score`, etc.: Set to reasonable defaults for curated content (0.7, 0.8, 0.05 fabrication risk)

This produced 272 findings across 18 angles (largest: amino-acid-biochemistry at 115, smallest: general-biochemistry at 5).

## 6. What the Orchestrator Must Implement (Issue #264)

### 6.1 Pipeline Health Monitoring

The orchestrator (or a subordinate) must continuously check:

1. **Flag distribution audit**: Are all findings at default scores? If avg(novelty) == 0.5 AND avg(confidence) == 0.4 AND avg(fabrication_risk) == 0.0 across the store, the flags haven't been computed yet. Run prescoring before query selection.

2. **Threshold-data alignment**: For each query type, check if ANY findings match the threshold. If 0 findings match VALIDATE thresholds (novelty > 0.6 AND confidence < 0.4), relax the thresholds or report the gap.

3. **Agent success rate tracking**: If an agent fails 2+ times across runs, flag it for redesign (different Flock function or SQL pattern).

4. **Context budget monitoring**: Track prompt token counts. If an agent is hitting context limits, automatically reduce the CTE LIMIT or SUBSTR length.

### 6.2 Self-Correction Actions

When the orchestrator detects a pipeline issue, it should:

| Condition | Detection | Action |
|-----------|-----------|--------|
| 0 queries eligible | All thresholds unmet | Run prescoring pass, then retry |
| All flags at defaults | Stats query on flag distributions | Run direct eval on unscored findings |
| Agent context overflow | Flock error with "too long" | Reduce CTE LIMIT by 50%, retry |
| llm_rerank returns -1 | Error in agent result | Retry with smaller group, then fall back to SQL ORDER BY |
| Cross-join explosion | Agent takes >60s | Add LIMIT to cross-join, reduce pair count |
| Source credibility gap | credibility-auditor flags source | Downweight all findings from that source |

### 6.3 Intelligent Agent Sequencing

Run 3 ran all 30 agents sequentially. The orchestrator should:

1. **Parallel Phase 1-2**: All `llm_complete` and `llm_filter` agents can run concurrently (independent row-level operations)
2. **Sequential Phase 3**: `llm_reduce` agents should run after Phase 1 evaluations update the flags, so synthesis reflects quality-weighted findings
3. **Score fusion after ranking**: `fusion_rrf` and `fusion_combsum` should run after `llm_rerank` produces rankings
4. **Deep specialists last**: Mechanism chains, predictions, and risk cascades benefit from having all prior results in the store

## 7. Flock Prompt Design Lessons

### 7.1 What Makes a Good Flock Prompt

The best-performing prompts share these traits:

- **Role assignment**: "You are a clinical pharmacologist" outperforms generic "Evaluate this"
- **Specific output format**: "Rate confidence 0.0–1.0" produces parseable results
- **Constraint framing**: "Using ONLY the evidence in these findings" prevents hallucination
- **Action-oriented**: "Design a complete daily protocol" outperforms "Summarize findings"

### 7.2 Named Prompt Template Pattern

```python
prompts = {
    "agent-name": (
        "Role assignment sentence. "
        "Task description. "
        "Output format specification. "
        "Constraints. "
        "Data: {{fact}}"
    ),
}
```

The `{{fact}}` placeholder is replaced by whatever is in `context_columns[0].data`. If the agent needs multiple inputs, concatenate them in the SQL expression passed to `context_columns`.

### 7.3 Prompts That Failed

- **Too vague**: "Is this finding valid?" → V4 Pro responds "The claim is not provided" when the finding is just a title/URL
- **Too long**: Prompts over ~500 chars combined with 8+ findings exceed 16K context
- **Multi-task**: "Evaluate AND rate AND explain AND suggest" → model picks one task and ignores others

## 8. Run Results Summary

### 8.1 Three Runs Progression

| Run | Success | Failed | Key Fix |
|-----|---------|--------|---------|
| Run 1 | 22/30 | 8 | Baseline — discovered Flock syntax rules |
| Run 2 | 23/30 | 7 | Fixed cross-corpus-synthesizer (LIMIT 6), llm_reduce CTEs |
| Run 3 | 29/30 | 1 | Registered named prompts for llm_complete, converted llm_first/last to llm_rerank |

### 8.2 Final Agent Results (Run 3)

| Agent | Function | Time | Rows | Status |
|-------|----------|------|------|--------|
| evaluator-new-corpus | llm_complete | 11.7s | 15 | OK |
| bridge-builder | llm_complete | 18.6s | 10 | OK |
| devil-advocate | llm_complete | 4.6s | 10 | OK |
| actionability-filter | llm_filter | 1.4s | 14 | OK |
| novelty-filter | llm_filter | 1.0s | 0 | OK |
| safety-sentinel | llm_filter | 1.4s | 1 | OK |
| angle-synthesizer | llm_reduce | 10.4s | 7 | OK |
| cross-corpus-synthesizer | llm_reduce | 7.1s | 2 | OK |
| mechanism-chain-builder | llm_reduce | 16.4s | 5 | OK |
| protocol-designer | llm_reduce | 1.7s | 1 | OK |
| clinical-auditor | llm_reduce | 9.7s | 1 | OK |
| gap-detector | llm_reduce | 36.7s | 7 | OK |
| priority-ranker | llm_rerank | 1.8s | 0 | FAIL (-1 ID) |
| best-finding-picker | llm_rerank | 1.1s | 7 | OK |
| weakest-finding-picker | llm_rerank | 0.9s | 7 | OK |
| quality-fusion | fusion_rrf | 0.02s | 25 | OK |
| score-combsum | fusion_combsum | 0.01s | 25 | OK |
| contradiction-miner | llm_complete | 4.2s | 5 | OK |
| outlier-explainer | llm_complete | 11.9s | 8 | OK |
| desert-bridge-proposer | llm_complete | 24.6s | 5 | OK |
| dose-response-analyst | llm_reduce | 1.7s | 1 | OK |
| temporal-reasoner | llm_reduce | 3.6s | 1 | OK |
| prediction-generator | llm_reduce | 22.3s | 1 | OK |
| practitioner-sports-med | llm_reduce | 1.0s | 1 | OK |
| counterfactual-prober | llm_complete | 9.2s | 8 | OK |
| credibility-auditor | llm_complete | 8.9s | 17 | OK |
| interaction-mapper | llm_reduce | 2.1s | 1 | OK |
| research-prioritizer | llm_reduce | 24.7s | 1 | OK |
| risk-cascade | llm_reduce | 20.8s | 1 | OK |
| grand-synthesizer | llm_reduce | 21.7s | 1 | OK |

### 8.3 What the Agents Actually Discovered

Notable findings unique to SQL-agentic intelligence (not discovered by individual workers):

- **credibility-auditor**: Bodybuilding corpus rated "highly reliable" (trust=0.8, fab_risk=0.05) vs YouTube testosterone sources at "very low reliability" (conf=0.32, fab_risk=0.64). This should drive automatic source-weighting in future runs.
- **counterfactual-prober**: "If the dual-pathway mTOR convergence framework is wrong, the entire guide's foundation collapses — protocols would revert to standard non-enhanced nutrition." Identifies the single most critical assumption.
- **desert-bridge-proposer**: "AAS-induced impulsivity sensitization predicts increased risk for opioid-related respiratory depression recovery failure" — a novel anabolic-steroids ↔ harm-reduction bridge no individual worker found.
- **prediction-generator**: "Combined leucine and androgen receptor activation will reduce the Sestrin2-mTORC1 dissociation constant below 20 μM through cooperative binding" — testable prediction derived from cross-finding synthesis.
- **gap-detector**: Identified that tyrosine's long-term safety at 2–5g doses across full training cycles has zero findings, and fructose/glucose dual-liver-glycogen repletion lacks dose-response data.

## 9. `instances_per_gpu` Must Be Empirical

Removed `_compute_instances_per_gpu` heuristic entirely. The actual VRAM per instance depends on:

- Model architecture (MoE vs dense, linear attention, MLA compression)
- Quantization (FP8, FP4, INT8, etc.)
- Context length and KV cache allocation
- CUDA graph profiling overhead
- vLLM's internal memory allocator behavior

No formula can capture all of these. `instances_per_gpu` is now a required operator parameter with no auto-calculation fallback.

## 10. Key Takeaway for Future Implementation

The SQL-agentic pattern achieves what the GossipSwarm could not: **diverse, deep analysis without compression-induced convergence**. Each virtual agent sees the raw findings directly (via SQL selection) and reasons about them independently (via Flock LLM calls). The ConditionStore acts as the shared blackboard, and the orchestrator's job is to sequence agents intelligently, monitor pipeline health, and self-correct when things go wrong.

Total cost: ~5 minutes on V4 Pro for 30 agents across 491 findings. This is orders of magnitude cheaper and faster than the GossipSwarm approach, with better quality output.
