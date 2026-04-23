# MiroThinker: Authoritative System Architecture

*April 2026 — Comprehensive reference for external review*

---

## 1. What MiroThinker Is

MiroThinker is an **agentic research synthesis system** that automates long-horizon information retrieval, cross-domain connection discovery, and evidence-based report generation. It operates continuously (designed for 24-hour unattended runs) against large corpora — medical textbooks, YouTube transcripts, forum posts, academic papers — and produces structured knowledge graphs with full provenance.

The system is built around three core innovations:

1. **Flock** — DuckDB's community extension for LLM-in-SQL, enabling thousands of rapid LLM judgments as composable SQL queries against a persistent data store
2. **The Gossip Swarm** — parallel specialist "bees" with deliberately asymmetric information, whose disagreements and cross-pollination generate unexpected connections (serendipity)
3. **The Cloned Context Pattern** — worker conversation histories transplanted into vLLM's prefix cache, turning accumulated reasoning into reusable expert backends for Flock queries at near-zero marginal cost

The central thesis: **Serendipity = Unexpected AND Relevant.** Every architectural layer exists to manufacture non-obvious cross-domain connections that a single reader would miss, while ensuring those connections serve the user's actual research question.

---

## 2. System Architecture Overview

```
USER PROMPT
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                 │
│  (swarm/mcp_engine.py — the brain)                                   │
│                                                                      │
│  • Ingest corpus into ConditionStore (DuckDB)                        │
│  • Detect research angles from user prompt                           │
│  • Assemble data packages per worker per wave                        │
│  • Launch tool-free workers (bees)                                   │
│  • Capture worker transcripts via hook observer                      │
│  • Register cloned contexts with session proxy                       │
│  • Run Flock SQL for catalogue operations                            │
│  • Check convergence, generate final report                          │
└───────────────┬─────────────────────────────────────────────────────┘
                │
    ┌───────────┼────────────┬──────────────┬──────────────┐
    ▼           ▼            ▼              ▼              ▼
┌────────┐ ┌────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐
│Workers │ │Session │ │ConditionStore│ │Hook Observer│ │  vLLM    │
│(bees)  │ │Proxy   │ │(DuckDB +    │ │(Strands     │ │(GPU      │
│        │ │        │ │ Flock ext)  │ │ hooks)      │ │ cluster) │
│Tool-   │ │Prepends│ │             │ │             │ │          │
│free    │ │cloned  │ │Full audit   │ │Captures     │ │Prefix    │
│reason- │ │context │ │trail DAG    │ │transcripts  │ │caching   │
│ing     │ │to vLLM │ │with lineage │ │read-only    │ │enabled   │
│agents  │ │calls   │ │             │ │             │ │          │
└────────┘ └───┬────┘ └─────────────┘ └─────────────┘ └──────────┘
               │                                            ▲
               └────────────────────────────────────────────┘
                    Forwards with conversation prefix
```

### Component Inventory

| Component | File | Purpose |
|---|---|---|
| Orchestrator | `swarm/mcp_engine.py` | Controls entire lifecycle: data packages, workers, Flock SQL, convergence |
| Worker factory | `swarm/agent_worker.py` | Creates tool-free worker agents with angle-specific worldview prompts |
| Hook observer | `swarm/worker_observer.py` | Strands hook provider capturing worker transcripts (read-only) |
| Session proxy | `swarm/session_proxy.py` | Maps model names to cloned conversations, forwards to vLLM |
| ConditionStore | `apps/strands-agent/corpus.py` | DuckDB-backed store: schema, reads, writes, compaction, fingerprinting |
| Data package builder | `swarm/data_package.py` | Assembles structured research briefs per worker per wave |
| RAG scorer | `swarm/rag.py` | Keyword-based hive memory scoring (bootstrap before clones are available) |
| Angle detector | `swarm/angles.py` | Prompt-driven angle extraction and section-angle assignment |
| Knowledge summarizer | `swarm/summaries.py` | Rolling per-angle knowledge summaries between waves |
| Compactor | `swarm/compactor.py` | Two-phase dedup (exact SQL + semantic via cloned context) |
| Flock proxy | `flock_proxy.py` | LiteLLM routing, response_format handling, multi-instance round-robin |

---

## 3. The ConditionStore: Full Audit Trail with Lineage DAG

### Core Principle

**The store holds everything. Every operation on data — reads, writes, transforms, dedup decisions, relevance judgments, disaggregation — is a row linked by `parent_id`.** Nothing is ever deleted. Nothing is ever truncated. The store is a complete, replayable audit trail.

### The Schema

The ConditionStore is a single DuckDB table (`conditions`) with the Flock extension loaded, providing LLM-in-SQL capabilities (`llm_complete`, `llm_filter`).

Every row represents an **AtomicCondition** — an indivisible unit of information with provenance:

```sql
CREATE TABLE conditions (
    id              INTEGER PRIMARY KEY,
    fact            TEXT NOT NULL,           -- The actual content
    row_type        TEXT NOT NULL,           -- Classification (see below)
    parent_id       INTEGER,                -- Direct parent in the DAG
    related_id      INTEGER,                -- Secondary relationship
    angle           TEXT,                   -- Research angle this belongs to
    source_model    TEXT,                   -- Which model produced this
    source_run      TEXT,                   -- Which run produced this
    source_type     TEXT,                   -- How created (worker_analysis, corpus_ingestion, etc.)
    phase           TEXT,                   -- Swarm phase (worker, serendipity, compact, catalogue)
    consider_for_use BOOLEAN DEFAULT TRUE,  -- Soft-delete flag (never hard-delete)
    confidence      REAL DEFAULT 0.5,
    trust_score     REAL,
    novelty_score   REAL,
    specificity_score REAL,
    relevance_score REAL,
    composite_quality REAL,
    cross_ref_boost REAL DEFAULT 0.0,
    strategy        TEXT,
    created_at      TEXT,
    scored_at       TEXT
);
```

### The Lineage DAG

`parent_id` creates a Directed Acyclic Graph of data provenance. Any claim in the final report can be traced back to the raw corpus paragraph that produced it:

```
Raw corpus paragraph (parent_id=NULL)
  ├── Disaggregated atom 1 (parent_id → raw paragraph)
  │     ├── Worker finding (parent_id → atom 1)
  │     │     └── Dedup decision: kept (parent_id → worker finding)
  │     └── Relevance score for angle X (parent_id → atom 1)
  ├── Disaggregated atom 2 (parent_id → raw paragraph)
  │     └── Worker finding (parent_id → atom 2)
  │           └── Cross-domain insight (parent_id → finding, related_id → finding from different angle)
  └── Disaggregated atom 3 (parent_id → raw paragraph)
        └── Dedup decision: obsolete (parent_id → atom 3, related_id → atom 1)
```

### Row Types

| `row_type` | Description | `parent_id` points to |
|---|---|---|
| `raw` | Original corpus paragraph, ingested verbatim | `NULL` (root node) |
| `finding` | Worker-synthesized claim or extracted fact | Source `raw` or `finding` row |
| `thought` | Intermediate reasoning step | Source row that prompted it |
| `insight` | Cross-domain connection or synthesis | Source finding(s) |
| `dedup_decision` | Record of a compaction judgment | Row that was judged |
| `relevance_score` | Per-angle relevance assessment | Row that was scored |
| `atom` | Disaggregated atomic claim from a larger blob | Source blob row |
| `similarity` | Detected similarity between two findings | One finding (`related_id` → other) |
| `contradiction` | Detected contradiction between findings | One finding (`related_id` → other) |
| `synthesis` | Merged/summarized knowledge briefing | Representative finding from angle |
| `worker_transcript` | Full conversation transcript from a worker's session | `NULL` (audit record) |

### Design Invariants

1. **No truncation. Ever.** The orchestrator controls data package size via disaggregation + relevance ranking. No individual atom is cut off.
2. **No data modification.** Raw corpus, findings, atoms — all immutable once written. `consider_for_use = FALSE` hides but never deletes.
3. **Every operation is a row.** Dedup decisions, relevance scores, atomisation, conversation transcripts — all stored with `parent_id` lineage.
4. **Workers are tool-free.** They receive data packages and reason. They don't interact with the store directly.
5. **The orchestrator runs all store operations.** Compaction, disaggregation, relevance scoring, data package preparation — all orchestrator-level, never worker-level.

---

## 4. Flock: LLM-in-SQL as Composable Intelligence

### What Flock Is

Flock is DuckDB's community extension that embeds LLM calls inside SQL queries. It provides two core functions:

```sql
-- Complete: generate text based on a prompt
SELECT llm_complete({'model_name': 'corpus_model'}, 'Summarize: ' || fact)
FROM conditions WHERE angle = 'insulin';

-- Filter: boolean LLM judgment as a WHERE clause
SELECT * FROM conditions
WHERE llm_filter({'model_name': 'corpus_model'}, fact, 'Is this relevant to insulin timing?');
```

### Why This Architecture Matters

Traditional agents hard-code intelligence in Python. Flock makes intelligence **composable SQL templates**:

| Traditional Agent | Flock Architecture |
|---|---|
| Hard-coded scoring pipeline | SQL templates the orchestrator can adapt |
| Fixed tool sequence | Free-form SQL with LLM functions |
| New feature = new Python class | New feature = new SQL template |
| Requires deployment | Orchestrator can invent at runtime |

### The 13-Step Template Battery

The orchestrator runs a battery of SQL templates against the store, divided into structural operations and serendipity generators:

**Core Templates (1-8): Organize the Corpus**

| Step | Name | Type | What It Does |
|------|------|------|-------------|
| 1 | Assess | Pure SQL | Count by status, find unscored, check expansions |
| 2 | Score | Flock | LLM-score trust, novelty, specificity, relevance |
| 3 | Composite Quality | Pure SQL | Weighted formula across all score dimensions |
| 4 | Detect Contradictions | Flock | Find pairs of findings that contradict each other |
| 5 | Cluster | Flock | Group related findings, mark representatives |
| 6 | Compress Redundancy | Flock | Merge near-duplicates, keep higher-quality version |
| 7 | Flag Weak | Pure SQL | Mark low-quality findings for expansion |
| 8 | Mark Ready | Pure SQL | Graduate findings to 'ready' status |

**Serendipity Templates (9-13): Generate Surprises**

| Step | Name | Type | What It Does |
|------|------|------|-------------|
| 9 | Contrarian Challenge | Flock (per angle) | Devil's advocate critique of top findings per angle |
| 10 | Cross-Angle Bridge | Flock | Polymath connector finds unexpected inter-angle links |
| 11 | Surprise Scoring | Pure SQL | Boost high-novelty outlier findings |
| 12 | Consensus Detector | SQL + Flock | Detect one-sided consensus, spawn targeted dissent |
| 13 | Angle Diversity Boost | Pure SQL | Reward findings from underrepresented angles |

### Serendipity Mechanisms Beyond Flock

Serendipity is woven into every layer:

- **Search Executor:** Contrarian query injection — for 30% of queries, deterministic templates generate `"criticisms of {query}"`, `"evidence against {query}"`, `"{query} controversy OR debate"` variants, interleaved with regular queries.
- **Thought Swarm:** After parallel specialists run, one LLM call asks "What would a polymath notice that domain specialists missed?" Connections become `insight` rows.
- **Condition Manager:** After 2+ iterations with 10+ findings and zero contradictions, a sceptic prompt is appended to the thinker's briefing. Zero LLM cost — just a conditional string append.
- **Angle Diversity Boost:** Pure SQL — rare angles get up to +0.08 quality boost proportional to their rarity: `0.08 * (1 - count/max_count)`.

---

## 5. The Gossip Swarm: Parallel Specialists with Deliberate Friction

### Architecture

The swarm decomposes a research prompt into multiple specialist angles (e.g., molecular biology, clinical pharmacology, hematology, safety, practitioner experience). Each angle is assigned to a **bee** — a tool-free reasoning agent with a specific worldview.

```
Orchestrator
    │
    ├── Bee A (Molecular Biology worldview)
    │     receives: 80% molecular data + 20% practitioner data
    │     produces: mechanistic analysis, pathway explanations
    │
    ├── Bee B (Clinical Pharmacology worldview)
    │     receives: 80% pharmacology data + 20% hematology data
    │     produces: dose-response analysis, drug interactions
    │
    ├── Bee C (Hematology worldview)
    │     receives: 80% hematology data + 20% molecular data
    │     produces: blood marker analysis, iron metabolism
    │
    ├── Bee D (Safety/Risk worldview)
    │     receives: 80% safety data + 20% pharmacology data
    │     produces: contraindications, emergency protocols
    │
    └── Bee E (Practitioner Experience worldview)
          receives: 80% practitioner data + 20% safety data
          produces: real-world observations, bloodwork patterns
```

### Key Design Decisions

**1. Workers are tool-free.** They receive a data package (curated material in their initial prompt) and reason freely. No tools, no store interaction. This ensures:
- Context purity (no tool-call formatting noise)
- Audit clarity (worker output = pure reasoning)
- Architecture simplicity (raw `llm_complete` calls, not full Agent loops)

**2. Deliberate misassignment (20-30% off-angle data).** This is the primary thread-discovery mechanism. A molecular biologist reading `"my hematocrit went from 42 to 49.6 after 8 weeks on 400mg tren-e"` doesn't see "hematocrit went up." It sees:

> 49.6 → hemoglobin ~16.5 → erythropoietin upregulation → but tren doesn't directly stimulate EPO → iron mobilization from hepatic stores → hepcidin suppression → I have a paper on hepcidin regulation in my slice.

Thread found. Because the molecular bee saw raw data from a non-obvious location while holding domain-specific knowledge.

**3. Raw data, not summaries, for cross-pollination.** Summaries kill threads. When a practitioner compresses `"my hematocrit went from 42 to 49.6 after 8 weeks on 400mg tren-e, doc said donate blood"` into `"hematocrit goes up on tren,"` the specific numbers that trigger the molecular bee's recognition are lost.

**4. Bee identity as worldview.** Each bee's system prompt defines a lens through which EVERYTHING is interpreted:

```
You are a molecular biologist. Everything you encounter, you interpret
through cellular mechanisms, receptor binding, gene expression, protein
folding, metabolic pathways. When you read a practitioner's observation
about bloodwork, you don't see "hematocrit went up." You see a signal
about erythropoiesis, iron homeostasis, oxygen-carrying capacity at the
cellular level. Your job is not to summarize what you read — it is to
EXPLAIN what you read through your domain.
```

### Gossip Protocol

```
Round 1: DEEP ANALYSIS
  Each bee analyzes its slice through its worldview.
  Mandatory output: Findings, cross-domain implications, thread-ends.
  → All output persisted to ConditionStore

Round 2+: CROSS-POLLINATION + RAG
  Each bee receives:
  - Its own previous output
  - Its original raw slice
  - Peer summaries from maximally-distant peers
  - "FROM THE HIVE": RAG results from ConditionStore — findings from
    other bees conceptually relevant to this bee's current analysis
  
  → All output persisted to ConditionStore
  → Targeted questions become RAG queries for next round

Convergence when:
  - Thread discovery rate drops (store write rate for new connections)
  - Workers agree (Jaccard similarity above threshold)
  - AND no new external data arrived (corpus delta is empty)
  - Hard ceiling: max_gossip_rounds
```

### Swarm-Internal RAG ("FROM THE HIVE")

Between gossip rounds, the orchestrator:
1. Extracts key concepts from each bee's current summary
2. Queries the ConditionStore for entries matching those concepts (from OTHER bees)
3. Injects the top-K most relevant entries as a "FROM THE HIVE" section in the gossip prompt
4. The bee sees targeted, relevant findings from the persistent store without context overwhelm

---

## 6. The Cloned Context Pattern: Expert Backends for Flock

### The Core Insight

Each worker accumulates angle-specific expertise in its conversation history. This expertise is valuable beyond just the worker's reasoning output — it can drive store operations (disaggregation, relevance scoring, dedup judgments) with domain-specific intelligence.

You don't modify the worker's context. Instead, you **clone the worker's conversation** and use it as the LLM backend for Flock SQL queries.

### How It Works

```
Wave N begins
    │
    ▼
Workers receive data packages, reason freely (tool-free)
    │
    ▼
Hook observer captures each worker's transcript (read-only)
    │
    ▼
Orchestrator registers cloned conversations with session proxy:
    POST /sessions/clone_insulin   → {messages: [worker A's history]}
    POST /sessions/clone_hematology → {messages: [worker B's history]}
    │
    ▼
DuckDB has per-clone Flock models:
    CREATE MODEL('clone_insulin', 'clone_insulin', 'openai')
    │
    ▼
Orchestrator runs Flock SQL:
    SELECT * FROM conditions
    WHERE llm_filter({'model_name': 'clone_insulin'},
                     fact, 'Is this relevant to insulin timing?')
    │
    ▼
Flock → proxy → prepends insulin worker's conversation → vLLM
    → vLLM prefix cache: conversation already cached → fast
    │
    ▼
Domain expert answers with full angle context
    │
    ▼
All results stored as rows with parent_id lineage
    │
    ▼
Wave N+1 begins with enriched data packages
```

### vLLM Prefix Caching

vLLM has built-in prefix caching. When the proxy sends the worker's conversation as a prefix:
- **First call**: Full KV cache computation (expensive)
- **Subsequent calls**: Same prefix detected, cached KV states reused (nearly free)

This means: thousands of Flock queries against the same expert context cost approximately **one** context-load plus one token per query. On GPU hardware, the marginal cost per Flock judgment is sub-millisecond compute + near-zero memory.

### Session Proxy

A thin FastAPI service (~100 lines) between Flock and vLLM:

```python
sessions: dict[str, list[dict]] = {}  # Per-worker conversation contexts

@app.post("/v1/chat/completions")
async def completions(request: ChatRequest):
    session_id = request.headers.get("X-Session-Id")
    if session_id and session_id in sessions:
        messages = sessions[session_id] + request.messages  # Prepend clone
    else:
        messages = request.messages
    return await vllm_client.chat(messages=messages, ...)
```

### Ramifications

**1. Expert Persistence:** Clones get better over time. By wave 10, the insulin clone has 10 waves of accumulated reasoning. Its relevance judgments become progressively more discriminating.

**2. Cross-Expert Pollination:** Chain Flock queries through multiple clones:

```sql
-- Step 1: Ask insulin clone what hematology findings matter for insulin
CREATE TEMP TABLE insulin_relevant AS
SELECT id, fact FROM conditions
WHERE angle = 'hematology'
  AND llm_filter({'model_name': 'clone_insulin'}, fact,
                 'Does this have implications for insulin timing?');

-- Step 2: Ask hematology clone to validate these as genuine hematology
SELECT id, fact FROM insulin_relevant
WHERE llm_filter({'model_name': 'clone_hematology'}, fact,
                 'Is this a substantive hematological finding?');
```

Neither worker cross-pollinated. Neither worker's context was modified. But the orchestrator found the intersection — material that the insulin expert thinks has insulin implications AND the hematology expert confirms is genuine hematology.

**3. Expert Disagreement as Signal:** Two clones from different model architectures answering the same question:

```sql
SELECT id, fact,
  llm_filter({'model_name': 'clone_insulin_qwen'}, fact, 'Accurate?') as qwen,
  llm_filter({'model_name': 'clone_insulin_glm'}, fact, 'Accurate?') as glm
FROM conditions WHERE angle = 'insulin';
```

Disagreement = high research value or model blind spot. Agreement = robust finding. Epistemic diversity becomes measurable at the individual-finding level.

**4. Meta-Experts via Composition:** Concatenate two clone conversations:

```python
proxy.register_session("meta_insulin_hematology",
    messages=insulin_messages + hematology_messages)
```

The meta-expert has BOTH domains' reasoning loaded and can answer questions requiring simultaneous knowledge of both. The omniscient meta-clone (all conversations concatenated) is powerful for final report generation.

**5. Bootstrap Loop (Self-Improving Data Packages):**

```
Wave N:   Worker reasons → Clone N captures expertise
          → Clone N scores relevance for wave N+1
          → Data package for N+1 is better curated

Wave N+1: Worker reasons with better material
          → Clone N+1 is a deeper expert
          → Data package for N+2 is even better

Risk: Positive feedback can narrow. Mitigation: Serendipity templates
(contrarian challenge, diversity boost, consensus detector) counteract narrowing.
```

---

## 7. The Clone-Transplant-Flock Pattern (Dream Serendipity)

This is the system's most ambitious capability, combining all three innovations:

### Architecture

1. **A swarm bee does deep work** — runs a full reasoning session against the corpus, generating 500K-1M tokens of accumulated findings, reasoning chains, and cross-domain hypotheses.

2. **The bee's conversation is transplanted** — injected as prefix into a DIFFERENT, smarter model on vLLM. The bee was perhaps a cheap 3B model. The transplant recipient is a 63B-active or 10M-context model (e.g., Ling 2.6 1T, Llama 4 Scout).

3. **The smarter model runs Flock queries** — it inherits the bee's domain expertise via the transplanted conversation, but applies its own superior reasoning to the Flock judgments.

4. **The ConditionStore captures everything** — each Flock hit becomes a row. The graph grows. Next wave's queries see the expanded graph.

### Why This Is Powerful

The bee provides the "intelligence layer" over raw data. The smarter model provides superior reasoning for binary judgments. The KV prefix cache holds both the raw corpus AND the bee's reasoning about it. Every Flock query benefits from both layers.

The asymmetry is key: **cheap model for broad generation, expensive model for precise judgment.**

### Candidate Models for Clone-Transplant

| Role | Model | Why |
|---|---|---|
| Bee (generator) | Any fast 3B-8B model | Cheap tokens, generates rich conversation |
| Transplant recipient | Ling 2.6 1T (63B active) | Hybrid linear attention, efficient at 1M, free on OpenRouter |
| Transplant recipient | Llama 4 Scout (17B active) | 10M native context, holds ENTIRE store + corpus + findings |
| Transplant recipient | Nemotron 3 Super (12B active) | Mamba-2, linear-time context accumulation |

---

## 8. Model Selection Architecture

### The Problem: Censorship

Medical pharmacology research (the primary use case) involves substances and protocols that commercial LLM providers routinely censor. The system requires models that will engage with detailed drug dosages, administration protocols, and interaction analysis without refusal.

### Abliterated Models

"Abliteration" removes safety training at the weight level via orthogonalization techniques. The primary providers on HuggingFace:

| Provider | Technique | Quality Impact |
|---|---|---|
| huihui-ai | Classic orthogonalization | Minimal KL divergence from base |
| Heretic (TPE) | Bayesian-optimized orthogonalization | Actually IMPROVES reasoning (NatInt 21.33 vs 18.72) |
| MPOA (grimjim) | Norm-preserving orthogonalization | Best quality (KL 0.0 on some models) |
| Abliterix | Multi-pass techniques | KL 0.0115 on Qwen3.5-122B |
| NousResearch | Fine-tune approach (Hermes) | Not true abliteration; retains topic-specific censorship |

**Key finding from testing:** Weight-surgery abliteration (huihui-ai) removes ALL safety training. Fine-tune approach (NousResearch Hermes) preserves topic-specific censorship — insulin protocols pass, steroid protocols are refused.

### Model Roles

| Role | Requirements | Best Candidates |
|---|---|---|
| **Flock Driver** | Fast inference, short answers, no thinking traces, uncensored | Llama-70B-abliterated (tested: 98% accuracy, 5.5/sec), dense non-thinking models |
| **Worker (bee)** | Strong reasoning, uncensored, 32K+ context | Qwen3-235B-abliterated, Llama-70B-abliterated |
| **Clone Recipient** | 1M+ context, strong reasoning, efficient at scale | Ling 2.6 1T, Nemotron 3 Super, Llama 4 Scout |
| **Full-Corpus** | 10M context | Llama 4 Scout (self-hosted, abliterated) |
| **Meta-Expert** | 2M context, frontier reasoning | Grok 4.20, Claude Opus, Gemini 3.1 Pro |

### Critical Finding: Thinking Models Are Unsuitable for Flock

Qwen3.5-35B-A3B (a thinking/reasoning model) was tested as a Flock driver and **failed catastrophically**:

| Metric | Qwen3.5 (thinking) | Llama-70B (non-thinking) |
|---|---|---|
| Accuracy | 54% | 98% |
| Latency | 4,669ms | 848ms |
| Throughput | 1.1/sec | 5.5/sec |
| Tokens/judgment | 598 | ~8 |

The model outputs 400-600 tokens of reasoning traces before every binary answer. With thousands of Flock queries, this is a 75× token overhead. The thinking mode cannot be disabled programmatically. **Architecture mismatch: thinking models are for worker reasoning, not high-throughput Flock judgments.**

---

## 9. Hardware Architecture: vLLM on H200 GPUs

### Target Configuration

- **4× NVIDIA H200 SXM** — 143 GB VRAM each (575 GB total)
- **vLLM 0.19.x** — serves models with tensor parallelism across GPUs
- **Prefix caching enabled** — corpus and conversation prefixes cached in KV store
- **Chunked prefill enabled** — handles long prefixes without OOM

### GPU Allocation Strategy

```
GPU 0: Model A (e.g., Llama-70B for Flock queries)
GPU 1: Model B (e.g., Qwen3-235B shards for worker reasoning)  
GPU 2: Model B (continued)
GPU 3: Model B (continued)

OR (for large models):

GPU 0-3: Single large model with TP=4 (e.g., Scout 10M at 200GB BF16)
```

### Prefix Caching Economics

| Operation | Without Prefix Cache | With Prefix Cache |
|---|---|---|
| First Flock query (1.4MB corpus) | ~500K tokens processed | ~500K tokens processed |
| Second Flock query (same corpus) | ~500K tokens processed | ~50 tokens processed |
| 1000th Flock query (same corpus) | ~500K tokens × 1000 | ~50 tokens × 999 + 500K |
| **Total for 1000 queries** | **~500M tokens** | **~550K tokens** |

This is why Flock MUST be local, not commercial API: 500M tokens at $3/MTok = $1,500. Locally: $0 marginal cost after the first query.

---

## 10. Data Flow: Complete Wave Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│ WAVE N                                                           │
│                                                                  │
│ 1. INGEST (first wave only)                                      │
│    Corpus → disaggregate into atoms → store as `raw` rows        │
│    SHA256 fingerprint prevents re-ingestion                      │
│                                                                  │
│ 2. PREPARE DATA PACKAGES                                         │
│    Per angle: relevance-ranked atoms + rolling knowledge summary  │
│    + cross-domain connections flagged for this angle              │
│    + 20-30% off-angle raw data (deliberate misassignment)        │
│    + "FROM THE HIVE" RAG results from store                      │
│                                                                  │
│ 3. WORKER REASONING (parallel, tool-free)                        │
│    Each bee receives its data package, reasons freely             │
│    Hook observer captures transcript → stored as audit row        │
│                                                                  │
│ 4. CLONE REGISTRATION                                            │
│    Worker conversations registered with session proxy             │
│    vLLM prefix cache loads conversation KV states                 │
│                                                                  │
│ 5. FLOCK CATALOGUE (via cloned expert contexts)                  │
│    a. Extract claims from worker transcripts                     │
│    b. Disaggregate large findings into atoms                     │
│    c. Score relevance per angle (each clone scores its own)      │
│    d. Detect contradictions                                      │
│    e. Cluster and compress redundancy                            │
│    f. Run serendipity battery (contrarian, cross-angle bridge,   │
│       surprise scoring, consensus detection, diversity boost)    │
│    ALL results stored with parent_id lineage                     │
│                                                                  │
│ 6. CONVERGENCE CHECK                                             │
│    - New connection discovery rate                                │
│    - Worker agreement (Jaccard similarity)                        │
│    - Corpus delta (new external data?)                            │
│    - Hard ceiling: max_waves                                     │
│                                                                  │
│ 7. GENERATE ROLLING SUMMARIES → used in wave N+1 data packages  │
│                                                                  │
│ → WAVE N+1                                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Compaction: Decisions Are Data

When the store grows large (200K+ rows over 24 hours), compaction reduces noise while preserving the audit trail.

### Phase 1: Exact-Match Dedup (Pure SQL)

```sql
SELECT angle, fact, COUNT(*) FROM conditions
WHERE consider_for_use = TRUE
GROUP BY angle, fact HAVING COUNT(*) > 1
```

For each duplicate group: keep highest-confidence, mark others `consider_for_use = FALSE`, store a `dedup_decision` row.

### Phase 2: Semantic Dedup (LLM via Cloned Context)

1. Pre-cluster candidates by shared keywords within angle (union-find)
2. For each cluster, Flock `llm_filter` driven by the angle's cloned context: "are these the same claim?"
3. Store each decision as a row with full lineage

The domain expert's judgment drives dedup — the insulin specialist decides whether two insulin claims are truly identical.

---

## 12. Epistemological Limitations

### Convergence Bias

The gossip protocol is designed to synthesize, not to falsify. The "Resolution of Contradictions" sections almost always end with "Both are correct" or "Two sides of the same coin." When peer review systematically resolves contradictions into higher-order syntheses, the system risks constructing an elegant but unfalsifiable narrative.

**Mitigation (existing):** Templates 9 (Contrarian Challenge), 12 (Consensus Detector). **Mitigation (needed):** A dedicated adversarial validator bee whose explicit mandate is falsification.

### Evidence Hierarchy Porosity

The system currently treats forum logs, YouTube transcripts, and clinical literature with the same structural weight. Numerical precision from anonymous forum posts (`"246 ng/mL IGF-1"`) is stored alongside peer-reviewed data. The `source_type` field exists but is not systematically used for confidence weighting.

### Overfitting to Internal Consistency

The system is optimized for a specific input space (male, advanced, using specific equipment, with pharmaceutical-grade compounds). The moment variables change, the elegant equations may collapse. The model lacks:
- Pharmacogenomic variance (CYP450 polymorphisms, IRS-1 G972R, AR CAG repeats)
- mTORC2 feedback loops (currently only mTORC1 is modeled)
- Neuro-immune crosstalk (GDF-15, IL-6, peripheral → central fatigue cascade)
- Sulfur metabolism / catecholamine clearance (SULT1A3/PST-M, COMT pathways)

---

## 13. Corpus Architecture

### Sources (Current)

| Source | Type | Size | Purpose |
|---|---|---|---|
| Medical textbooks | Cleaned text | ~626K chars | Authoritative pharmacology reference |
| YouTube transcripts | 10% sample | ~338K chars | Practitioner knowledge (Milos Sarcev, MPMD, etc.) |
| Swarm findings | Accumulated | ~449K chars | Cross-domain connections from prior runs |
| Forum posts | Extracted | Variable | Real-world practitioner observations |
| Baseline corpus | Curated | ~12K chars | Core insulin/bodybuilding pharmacology reference |

### Corpus Loading into Flock

The corpus is loaded into the ConditionStore as `raw` rows. For Flock queries, the relevant subset is assembled into a single text prefix that gets cached by vLLM. For full-corpus queries (10M context models), the entire store can be serialized into a single prompt.

---

## 14. Implementation Status

### What Works (Tested)

- ConditionStore schema, reads, writes, compaction
- Flock SQL integration (DuckDB + LLM functions)
- Worker factory (tool-free bees with worldview prompts)
- Hook observer (transcript capture)
- Gossip protocol (multi-round peer exchange)
- Serendipity bridge (cross-angle connection detection)
- Rolling knowledge summaries
- Flock relevance scoring on vLLM with prefix caching (98% accuracy at 5.5 judgments/sec on H200)

### What Needs Testing

- Session proxy (clone registration + prefix forwarding)
- Cross-expert pollination via Flock chains
- Meta-expert composition
- Clone persistence across runs
- 10M context full-corpus Flock (Llama 4 Scout)
- Clone-transplant-Flock (bee → smarter model)

### Current Test Results

| Phase | Model | Result |
|---|---|---|
| 1A (Flock baseline) | Llama-3.3-70B-abliterated | **98% accuracy**, 848ms avg latency, 5.5/sec throughput |
| 1B (DeltaNet test) | Qwen3.5-35B-A3B-abliterated | **54% accuracy** — REJECTED (thinking model, 75× token overhead) |
| 1C (10M context) | Llama 4 Scout abliterated | In progress (vLLM loading issues with Llama 4 architecture) |
| 1D (Mamba-2) | Nemotron 3 Super Heretic FP8 | Pending (model downloading) |

---

## 15. File Map

```
MiroThinker/
├── apps/
│   ├── adk-agent/              # Core research agents (ADK framework)
│   │   ├── agents/             # Thinker, Maestro, Pipeline, Synthesiser
│   │   ├── models/             # DuckDB persistence, AtomicCondition schema
│   │   ├── aspects/            # Corpus refresh, heartbeat, output quality
│   │   └── callbacks/          # After-model, before-tool hooks
│   ├── strands-agent/          # OSINT research system + tools
│   │   └── corpus.py           # ConditionStore implementation
│   └── miroflow-agent/         # Interactive scaling framework
├── swarm/                      # GossipSwarm engine
│   ├── mcp_engine.py           # Orchestrator (wave lifecycle, Flock SQL)
│   ├── agent_worker.py         # Tool-free worker factory
│   ├── worker_observer.py      # Strands hook provider (transcript capture)
│   ├── session_proxy.py        # FastAPI proxy (clone → vLLM)
│   ├── data_package.py         # Structured research brief builder
│   ├── rag.py                  # Hive memory RAG scorer
│   ├── angles.py               # Angle detection + section assignment
│   ├── summaries.py            # Rolling knowledge summarizer
│   └── compactor.py            # Two-phase dedup engine
├── libs/miroflow-tools/        # MCP server manager, tool execution
├── docs/                       # Architecture documentation
│   ├── ARCHITECTURE.md         # This document
│   ├── STORE_ARCHITECTURE.md   # ConditionStore deep dive
│   ├── FLOCK_SERENDIPITY_ARCHITECTURE.md  # Flock + serendipity templates
│   ├── SWARM_CONVERSATION_ARCHITECTURE.md # Gossip protocol + thread discovery
│   ├── CLONED_CONTEXT_PATTERN.md          # Clone-transplant pattern
│   ├── SWARM_WAVE_ARCHITECTURE.md         # Complete wave lifecycle spec
│   ├── MODEL_REGISTRY.md       # 823-model catalog across 16 providers
│   ├── MODEL_USEFULNESS_ANALYSIS.md       # Architecture research per model family
│   ├── EMPLOYMENT_EXECUTION_PLAN.md       # Test execution plan
│   └── test_results/           # Phase-by-phase raw results (JSON + markdown)
└── scripts/h200_test/          # GPU cluster test runners
```

---

## 16. Glossary

| Term | Definition |
|---|---|
| **AtomicCondition** | Indivisible factual claim stored as a single database row |
| **Flock** | DuckDB extension enabling LLM calls (`llm_complete`/`llm_filter`) directly via SQL |
| **ConditionStore** | DuckDB-backed persistent store holding all findings, reasoning, and operations as a DAG |
| **Gossip Swarm** | Multi-agent synthesis via iterative peer-to-peer information exchange |
| **Bee / Worker** | Tool-free reasoning agent with a domain-specific worldview |
| **Clone** | A worker's conversation history registered with the session proxy for reuse |
| **Prefix Caching** | vLLM feature that caches KV states for repeated prompt prefixes |
| **Session Proxy** | FastAPI service that prepends cloned conversations to vLLM requests |
| **Data Package** | Structured research brief assembled by the orchestrator for each worker per wave |
| **Wave** | One complete cycle of: data packages → worker reasoning → Flock catalogue → convergence check |
| **Serendipity** | Unexpected AND relevant connections; the system's primary output beyond factual compilation |
| **Abliteration** | Removal of safety training at the weight level via orthogonalization |
| **Gradient-Flag Scoring** | Multi-dimensional quality assessment (trust, novelty, specificity, relevance) |
| **Lineage DAG** | Directed Acyclic Graph tracking provenance of every row via `parent_id` |
| **Thread** | A hidden connection between findings in different domains/locations |
| **Deliberate Misassignment** | Injecting 20-30% off-angle raw data into specialist bees to trigger thread discovery |
| **Cross-Angle Bridge** | Flock template that discovers unexpected connections between different specialist angles |
| **Contrarian Challenge** | Flock template that generates devil's advocate critique of consensus findings |
| **Meta-Expert** | Composition of multiple clone conversations into a single context |
| **Clone-Transplant** | Injecting a cheap model's conversation into a smarter model's prefix cache |
| **Dream Serendipity** | Clone-Transplant-Flock pattern: bee generates → transplant to smart model → Flock queries |

---

*This document is self-contained. An external reviewer needs no other context to evaluate the architecture.*
