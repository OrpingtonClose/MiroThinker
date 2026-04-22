# The Cloned-Context-as-Flock-Backend Pattern: Ramifications

## Enumeration of Core Facts

1. **Workers are tool-free bees.** They receive a data package (curated, angle-relevant material) and reason freely. No tools, no store interaction.

2. **After reasoning, the worker's conversation history is captured** by a Strands hook observer (read-only, `AfterModelCallEvent` / `AfterInvocationEvent`). The transcript is stored as a `worker_transcript` row in the ConditionStore.

3. **The orchestrator clones the worker's conversation** and registers it with a session proxy. The clone is not a separate agent — it is the same conversation history prepended to every Flock request.

4. **Flock's `llm_complete` / `llm_filter` SQL functions** route through the session proxy. The proxy prepends the cloned conversation before forwarding to vLLM. The LLM that answers Flock's question has the worker's full accumulated context.

5. **vLLM prefix caching** makes this efficient. The first call with a conversation prefix is expensive (full KV cache computation). Subsequent calls with the same prefix reuse the cached KV states — nearly free.

6. **Flock supports named models** via `CREATE MODEL('name', 'model_id', 'provider')`. The orchestrator can create per-clone models: `clone_insulin`, `clone_hematology`, etc. The proxy reads the model name from the request and selects the corresponding conversation to prepend.

7. **The existing Flock proxy** (`flock_proxy.py`) already handles LiteLLM routing, `response_format` stripping, JSON wrapping, multi-instance round-robin, and progress callbacks. The session proxy extends this.

8. **The ConditionStore holds everything.** Every operation is a row with `parent_id` lineage. The store is a full audit trail DAG.

9. **The 8×H200 target** provides 143GB VRAM per GPU (1.14TB total). KV cache for a 32B model at 32K tokens ≈ 8GB per conversation. 8 conversations = 64GB — fits alongside model weights (~64GB for 32B fp16) on a single GPU.

10. **The 13-step Flock template battery** already exists — scoring, clustering, dedup, cross-angle bridges, contrarian challenges, consensus detection, surprise scoring. These currently use a generic LLM. With cloned contexts, they use domain experts.

---

## The Pattern, Precisely

```
Wave N begins
    │
    ▼
Orchestrator prepares data packages per angle
(store query → relevance ranking → rolling summaries → cross-domain connections)
    │
    ▼
Workers receive packages, reason freely (tool-free)
    │
    ▼
Hook observer captures each worker's transcript (read-only)
    │
    ▼
Orchestrator registers each worker's conversation with session proxy
    ┌────────────────────────────────────────┐
    │ POST /sessions/clone_insulin           │
    │ Body: {messages: [worker A's history]} │
    │                                        │
    │ POST /sessions/clone_hematology        │
    │ Body: {messages: [worker B's history]} │
    │ ...                                    │
    └────────────────────────────────────────┘
    │
    ▼
DuckDB has per-clone Flock models:
    CREATE MODEL('clone_insulin', 'clone_insulin', 'openai')
    CREATE MODEL('clone_hematology', 'clone_hematology', 'openai')
    │
    ▼
Orchestrator runs Flock SQL:
    SELECT * FROM conditions
    WHERE llm_filter({'model_name': 'clone_insulin'},
                     fact, 'Is this relevant to insulin timing?')
    │
    ▼
Flock sends request → proxy sees model='clone_insulin'
    → prepends insulin worker's conversation → forwards to vLLM
    → vLLM prefix cache: conversation cached → fast response
    │
    ▼
Domain expert answers with full angle context
    │
    ▼
All results stored as rows with parent_id lineage
    │
    ▼
Sessions discarded (or retained — see Ramification 1)
    │
    ▼
Wave N+1 begins with enriched data packages
```

---

## Ramifications

### 1. Expert Persistence: Clones Get Better Over Time

The clone doesn't have to be discarded after one catalogue pass. Over multiple waves, the worker's conversation grows richer — more reasoning, more connections, more domain depth. The clone at wave 10 has 10 waves of accumulated insulin-timing reasoning.

**Consequence:** Flock queries get progressively better. At wave 1, the insulin clone answers "is this relevant?" with basic pharmacology. By wave 10, it answers with nuanced understanding of dose-response curves, timing windows relative to training, interaction with GH protocols. Its relevance judgments become more discriminating.

**The limit:** Context window. A standard transformer at 32K tokens fills up after ~4 waves of rich reasoning (~8K tokens per wave). Options:
- **Linear attention models** (e.g., Kimi-Linear) have constant KV cache cost — effectively unlimited context. The clone can accumulate indefinitely.
- **Rolling distillation:** After each wave, use the clone to generate a "what I've learned" summary. Replace the full conversation with the summary + latest wave. Compressed expertise.
- **Accept the limit:** 4 waves of context is still vastly better than zero (the current generic Flock LLM has no domain context at all).

### 2. Cross-Expert Pollination via Flock Chain

The orchestrator can chain Flock queries through multiple clones:

```sql
-- Step 1: Ask the insulin clone what hematology findings matter for insulin
CREATE TEMP TABLE insulin_relevant AS
SELECT id, fact FROM conditions
WHERE angle = 'hematology'
  AND llm_filter({'model_name': 'clone_insulin'},
                 fact, 'Does this finding have implications for insulin timing?');

-- Step 2: Ask the hematology clone to validate these as genuine hematology
SELECT id, fact FROM insulin_relevant
WHERE llm_filter({'model_name': 'clone_hematology'},
                 fact, 'Is this a substantive hematological finding?');
```

Neither worker cross-pollinated. Neither worker's context was modified. But the orchestrator used BOTH expert clones to find the intersection — material that the insulin expert thinks has insulin implications AND the hematology expert confirms is genuine hematology.

**This replaces the current "cross-angle bridge" template** (template 10 in the Flock serendipity battery), which uses a generic "polymath" LLM prompt. Instead of asking a generic model to "find unexpected connections between angles," you ask each domain expert to evaluate the other domain's findings through their own lens. The intersection is computationally discovered cross-domain connection — more precise and more grounded than a generic prompt.

**Generalization:** For N angles, you can run N×(N-1) cross-expert queries (each clone evaluates every other angle's findings). The resulting connection matrix maps which findings bridge which domains, as judged by actual domain experts.

### 3. Expert Disagreement as Signal

Two clones from different model architectures can be asked the same question about the same finding:

```sql
SELECT id, fact,
       llm_filter({'model_name': 'clone_insulin_qwen'}, fact, 'Accurate claim?') as qwen_agrees,
       llm_filter({'model_name': 'clone_insulin_glm'}, fact, 'Accurate claim?') as glm_agrees
FROM conditions
WHERE angle = 'insulin' AND row_type = 'finding';
```

If Qwen and GLM disagree on a specific finding, that disagreement is a signal:
- **Both agree (high confidence):** The finding is likely robust. Two architecturally different models independently validate it.
- **Both disagree (low confidence):** The finding is likely noise.
- **One agrees, one disagrees (interesting):** The finding sits on a decision boundary. It's either genuinely controversial (high research value) or one model has a blind spot (high diagnostic value).

**Consequence:** Epistemic diversity becomes measurable at the individual-finding level. You don't need to compare final reports — you can quantify agreement on every claim in the store. Disagreement rows become a natural priority queue for further research.

### 4. Clone Composition: Meta-Experts

Concatenate two clone conversations to create a meta-expert:

```python
# Register a meta-expert with both contexts
proxy.register_session("meta_insulin_hematology",
    messages=insulin_worker_messages + hematology_worker_messages)
```

```sql
CREATE MODEL('meta_insulin_hematology', 'meta_insulin_hematology', 'openai');

-- The meta-expert has BOTH domains' reasoning loaded
SELECT llm_complete({'model_name': 'meta_insulin_hematology'},
    {'prompt': 'What interactions exist between insulin protocols and hematological markers?'});
```

This meta-expert can answer questions that require knowledge of BOTH domains simultaneously — because it has both conversation histories as context. No single worker has this capability. The meta-expert is emergent from composition.

**On 8×H200 with prefix caching:**
- 8 individual clones (one per worker/angle)
- 28 pairwise meta-clones (8 choose 2) — each with two conversations concatenated
- 1 omniscient meta-clone (all 8 conversations concatenated)

**Context window pressure:** 2 conversations × ~8K tokens = 16K tokens for a pairwise meta-expert. Fits easily. The omniscient meta-clone at 8 × 8K = 64K tokens needs a model with 64K+ context window (many modern models support 128K). On H200 with 143GB VRAM, the KV cache for 64K tokens at 32B model ≈ 16GB — manageable.

**The omniscient meta-clone is particularly powerful for final report generation.** Instead of a generic summarizer reading findings from the store, the report writer has ALL workers' reasoning loaded as context. It can write a report that reflects the full breadth of angle-specific expertise.

### 5. Self-Improving Data Packages (Bootstrap Loop)

```
Wave N: Worker reasons with basic data package
    → Clone N captures expertise
    → Clone N scores relevance of new findings for wave N+1
    → Data package for N+1 is better curated

Wave N+1: Worker reasons with better material
    → Clone N+1 is a deeper expert (richer context)
    → Clone N+1 scores relevance even more accurately
    → Data package for N+2 is even better curated

Wave N+2: ...
```

**This is a positive feedback loop.** Each wave's clone is a better expert than the previous wave's because the worker had better material. Better material comes from better relevance scoring. Better relevance scoring comes from a better clone. The system bootstraps itself toward optimal data packages.

**Risk:** Positive feedback loops can also lock in. If the wave 1 clone scores a particular sub-topic as highly relevant, the wave 2 worker gets more of that sub-topic, reasons more about it, producing a wave 2 clone that scores it even higher. The angle narrows. Diversity decreases.

**Mitigation:** The serendipity templates (contrarian challenge, angle diversity boost, consensus detector) are designed precisely for this. They counteract narrowing by injecting surprise, boosting minority findings, and spawning dissent. The bootstrap loop makes the expert deeper; the serendipity templates keep it wide.

### 6. The Clone as Validator (Self-Critique Without Context Pollution)

After the worker produces text, the clone can be asked:

```sql
-- What did the worker miss?
SELECT llm_complete({'model_name': 'clone_insulin'},
    {'prompt': 'Given everything you know about this domain, '
               'what important aspects are NOT covered in this analysis? '
               || worker_output});
```

This is self-critique without polluting the worker. The worker never sees the critique. The critique goes into the store as a `thought` row with `parent_id` → the worker transcript. The orchestrator can feed critiques back as part of the next wave's data package.

**Why this is better than having the worker self-critique:** The worker's context is fully occupied with reasoning. Adding "now critique yourself" consumes context and changes the reasoning dynamic. The clone has the same context plus the finished output — it can critique from a position of completeness.

### 7. Clone Persistence Across Runs

Serialize the conversation to the store (already happening via `worker_transcript` rows). In the next run, reconstruct the clone from the transcript:

```python
# Run 50 startup
prior_transcript = store.query(
    "SELECT messages FROM worker_transcripts "
    "WHERE angle = 'insulin' ORDER BY run_number DESC LIMIT 1")
proxy.register_session("clone_insulin", messages=prior_transcript)
```

Run 50's clone starts with Run 49's expertise already loaded. Across 24 hours of continuous operation, each run's clone benefits from all prior reasoning.

**Context window limit:** 49 full conversations don't fit. Solutions:
- **Rolling summaries as compressed clone seeds:** Instead of the full transcript, load the rolling knowledge summary for the angle (which compresses hundreds of findings into a few paragraphs) as the conversation seed. The clone starts with compressed expertise from all prior runs.
- **Progressive distillation:** After each run, the clone generates a "what I've learned" summary. This summary replaces the full conversation. Over 50 runs, the summary accumulates distilled expertise.
- **Linear attention models:** Constant KV cache cost. Load all 49 transcripts. No limit. This is the brute-force solution and it works if you have the right model architecture.

### 8. Competitive Clone Evaluation (Evolutionary Pressure)

Run the same Flock query through all N clones from different model architectures:

```sql
-- Which clone identifies the most relevant findings for its angle?
SELECT 'qwen' as model,
       COUNT(*) as relevant_count
FROM conditions
WHERE llm_filter({'model_name': 'clone_insulin_qwen'}, fact, 'Relevant to insulin timing?')
UNION ALL
SELECT 'glm',
       COUNT(*)
FROM conditions
WHERE llm_filter({'model_name': 'clone_insulin_glm'}, fact, 'Relevant to insulin timing?');
```

Measure:
- **Relevance precision:** What fraction of the clone's "relevant" judgments are confirmed by human review or cross-clone agreement?
- **Disaggregation quality:** How many useful atomic claims does each clone extract from the same source paragraph?
- **Dedup accuracy:** Does the clone correctly identify true duplicates while preserving genuinely different claims?

**Consequence:** The best-performing clone's model architecture gets more worker slots in the next run. This is evolutionary pressure on model selection — measured by Flock query quality, not just report quality. Over 24 hours, the system converges on the optimal model distribution.

### 9. The Serendipity Battery Becomes Expert-Driven

The existing 13-step Flock template battery currently uses a generic LLM (`corpus_model`). With cloned contexts, every template can route through the appropriate expert:

| Template | Current (Generic) | With Cloned Context |
|---|---|---|
| **9. Contrarian Challenge** | Generic sceptic critiques findings | The hematology expert critiques insulin findings. Informed dissent, not generic. |
| **10. Cross-Angle Bridge** | Generic polymath finds connections | Each clone evaluates other angles' findings (Ramification 2). Expert-driven bridging. |
| **4. Detect Contradictions** | Generic model compares pairs | Two clones from different architectures judge the same finding (Ramification 3). Disagreement = contradiction. |
| **6. Compress Redundancy** | Generic model judges duplicates | The domain expert decides whether two findings in its angle are truly redundant. Much more accurate. |
| **12. Consensus Detector** | Generic model generates counter-claims | Clone from a DIFFERENT angle generates counter-claims. Domain-informed dissent. |

**The serendipity templates don't change.** The SQL stays the same. Only the `model_name` parameter changes — from `'corpus_model'` to `'clone_X'`. The upgrade from generic to expert is a configuration change, not an architectural change.

### 10. The Pattern Generalizes

This pattern is not specific to research swarms or MiroThinker. It applies anywhere that:
- Specialized agents accumulate domain context through reasoning
- Database operations (scoring, filtering, classification) would benefit from domain expertise
- The agent's context must not be polluted by operational queries

**Applications beyond this project:**
- **Legal research:** Clone a patent specialist's context to drive relevance scoring in a patent database via SQL
- **Medical diagnosis:** Clone a cardiologist's reasoning context to prioritize cardiac findings in a patient record
- **Code review:** Clone a security specialist's accumulated context to drive vulnerability detection SQL queries in a codebase database
- **Customer support:** Clone a senior agent's context to drive ticket routing decisions via SQL classification
- **Scientific literature review:** Clone a domain expert's context to score paper relevance in a citation database

In each case: the expert reasons freely in its domain, a clone of its context drives database-level operations, the expert is never interrupted or polluted.

### 11. The Session Proxy as Infrastructure Primitive

The session proxy becomes a general-purpose infrastructure component:

```
┌──────────────────────────────────────────────────┐
│              Session Proxy                        │
│                                                  │
│  Sessions:                                       │
│  ┌──────────────────────────────────────────┐   │
│  │ clone_insulin → [conversation messages]   │   │
│  │ clone_hematology → [conversation msgs]    │   │
│  │ meta_insulin_hematology → [concat msgs]   │   │
│  │ omniscient → [all conversations]          │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  Routes:                                         │
│  ┌──────────────────────────────────────────┐   │
│  │ model → session → conversation prepend    │   │
│  │ model → GPU endpoint (per-model routing)  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  Management:                                     │
│  POST /sessions/{id}     Register/update         │
│  DELETE /sessions/{id}   Discard                 │
│  GET /sessions           List active             │
│  POST /compose           Create meta-expert      │
│                                                  │
│  Observability:                                  │
│  All queries logged as audit trail rows          │
│  Token usage per session tracked                 │
│  Cache hit rate per session tracked              │
└──────────────────────────────────────────────────┘
```

**The proxy is the control plane for the expert-clone network.** It manages:
- Per-session conversation contexts
- Meta-expert composition (concatenation)
- GPU routing (direct model-specific requests to the correct vLLM instance)
- Load balancing across GPUs for catalogue operations
- Rate limiting (prevent one angle from dominating inference bandwidth)
- Audit logging (every clone query is a store row)
- Cache management (evict stale conversations, prioritize active clones)

### 12. KV Cache Memory Budget (Real Numbers)

For a 32B parameter model with GQA (8 KV heads, 128 head dim, 64 layers):

```
Per token KV cache = 64 layers × 8 heads × 128 dim × 2 (K+V) × 2 bytes (fp16)
                   = 262,144 bytes ≈ 256 KB per token

32K token conversation = 32,768 × 256 KB = 8 GB

H200 (143 GB VRAM):
  Model weights (32B fp16): ~64 GB
  Remaining for KV cache:   ~79 GB
  Max cached conversations:  79 / 8 ≈ 9 conversations at 32K tokens
  Or: 4 full + 10 half-length conversations
```

For MoE models (e.g., Qwen3 MoE with fewer active KV heads), the per-token KV cost is lower, allowing more cached conversations.

**On 8×H200 with 8 different model instances:**
Each GPU caches conversations for workers using that model. If 2 workers use the same model, that GPU caches 2 conversations. Total system capacity: ~72 cached conversations across all GPUs. More than enough for 8 workers + 28 pairwise meta-experts.

**Critical insight:** MoE models are better for this pattern because their KV cache is smaller, allowing more conversations to be cached simultaneously. Choose MoE for workers whose clones will be used heavily.

### 13. The Fundamental Tension Resolved

In the previous design review, I identified an irresolvable tension:

> - Tool-free workers → can't search the store
> - Workers that catalogue → need tools
> - No context pollution → tool calls pollute context
> - Full audit trail → store grows 10× faster
> - No truncation → larger returns per query

The cloned-context pattern resolves ALL of these:

| Tension | Resolution |
|---|---|
| Tool-free workers can't search the store | Workers don't need to — they receive curated data packages |
| Store operations need domain expertise | Cloned contexts provide expertise without touching the worker |
| No context pollution | The clone is disposable, the worker is pristine |
| Full audit trail + store growth | Cloned experts can do smart compaction, reducing growth |
| No truncation + context limits | Data packages are curated by relevance, not truncated |

The clone IS the bridge between "workers must be pure" and "store operations need intelligence." It carries the worker's expertise without the worker's participation.

---

## Open Questions

1. **How does the Flock proxy know which session to use?** The proxy reads the `model` field from the request body. Flock sends this based on `CREATE MODEL` in DuckDB. The proxy maps model names to sessions. This works with the existing Flock architecture — no extension modification needed.

2. **Can meta-expert contexts be incrementally built?** Instead of concatenating full conversations, could you feed one clone's summary into another clone's context? This would create layered expertise without the token cost of full concatenation.

3. **What happens when the clone's context is stale?** The clone was created from wave N's conversation. By the time catalogue operations run, the store may have wave N+1 data. The clone judges based on wave N understanding. This is acceptable — the clone's value is its domain expertise, not its awareness of the latest store state.

4. **Should clone queries be parallelised across GPUs?** If the insulin worker used Qwen on GPU 0 but there are 200 findings to score, should those queries be batched and distributed across multiple GPUs (each loading the same prefix)? vLLM prefix caching is per-GPU, so replicating the prefix to multiple GPUs multiplies the memory cost but divides the latency.

5. **Can the pattern work with thinking models?** Models that use chain-of-thought (e.g., Qwen3 with thinking enabled) produce richer conversation histories. The clone of a thinking model's conversation would have access to the explicit reasoning chain. This makes the clone's Flock judgments more informed — but also consumes more context window.
