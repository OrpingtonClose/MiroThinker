# ConditionStore Architecture: Full Audit Trail with Lineage DAG

## Core Principle

**The store holds everything. Every operation on data — reads, writes, transforms, dedup decisions, relevance judgments, disaggregation — is a row in the ConditionStore linked by `parent_id`.** Nothing is ever deleted. Nothing is ever truncated. The store is a complete, replayable audit trail of what every agent did with every piece of data across 24 hours of continuous operation.

## Why This Matters

A 24-hour continuous run against 150MB of YouTube transcripts produces 360–1440 runs, 200K–800K cumulative findings, and an unknowable number of cross-domain connections. Without a full audit trail:

- You can't trace a final-report claim back to the raw transcript sentence that produced it
- You can't measure which model architectures contributed genuinely novel findings
- You can't distinguish convergent discovery (two models independently reaching the same conclusion) from duplicated ingestion
- You can't audit whether a dedup operation incorrectly removed a unique finding

With the audit trail, every leaf in the final report has a provenance chain back to raw corpus, through every intermediate transformation.

## The Lineage DAG

Every row in the `conditions` table has a `parent_id` column. This creates a Directed Acyclic Graph (DAG) of data provenance:

```
Raw corpus paragraph (parent_id=NULL)
  ├── Disaggregated atom 1 (parent_id → raw paragraph)
  │     ├── Worker finding derived from atom 1 (parent_id → atom 1)
  │     │     └── Dedup decision: kept (parent_id → worker finding)
  │     └── Relevance score for angle X (parent_id → atom 1)
  ├── Disaggregated atom 2 (parent_id → raw paragraph)
  │     └── Worker finding derived from atom 2 (parent_id → atom 2)
  │           └── Cross-domain connection to atom 7 (parent_id → worker finding)
  └── Disaggregated atom 3 (parent_id → raw paragraph)
        └── Dedup decision: obsolete (parent_id → atom 3, related_id → atom 1)
```

### Row Types in the DAG

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
| `worker_transcript` | Full conversation transcript from a worker's reasoning session | `NULL` (audit record) |

### Key Columns for Lineage

```sql
parent_id       INTEGER   -- Direct parent in the DAG (what produced this row)
related_id      INTEGER   -- Secondary relationship (e.g., the "other" in a dedup pair)
source_model    TEXT      -- Which model architecture produced this row
source_run      TEXT      -- Which run produced this row (e.g., "run_20250416_120000")
source_type     TEXT      -- How this row was created (e.g., "worker_analysis", "corpus_ingestion", "compaction")
phase           TEXT      -- Swarm phase when created (e.g., "worker", "serendipity", "compact", "catalogue")
```

## Worker Architecture: Tool-Free Bees

### Workers Have No Tools

Workers are pure reasoning agents. They receive a **data package** — curated, angle-relevant material injected into their initial prompt — and reason over it. They don't search the store, they don't store findings, they don't call any tools. They just think.

The engine (orchestrator) prepares data packages before each wave:
1. Query the store for material relevant to the worker's angle
2. Include the latest rolling knowledge summary for the angle
3. Include any cross-domain connections flagged for the angle
4. Inject this package into the worker's system prompt or initial context

The worker reasons over the package and produces text. That's it.

### Why Tool-Free?

- **Context purity**: Tool calls consume context window space and change reasoning behavior. A worker reasoning freely about insulin pharmacokinetics shouldn't be interrupted by `search_corpus` return formatting.
- **Audit clarity**: When tools call the store, you can't distinguish the worker's reasoning from tool orchestration noise. With tool-free workers, the worker's output IS the reasoning — pure signal.
- **Architecture simplicity**: Workers don't need the Strands Agent tool-calling loop. They can be raw `llm_complete` calls with rich context, which is simpler and cheaper than a full Agent.

### Extracting Findings from Worker Reasoning

A **Strands hook observer** attached to each worker captures output without modifying context:

- **`AfterModelCallEvent`** — fires after every model response. `stop_response.message` contains the full response text.
- **`MessageAddedEvent`** — fires when any message is added to conversation. Captures the full dialogue.
- **`AfterInvocationEvent`** — fires when the worker finishes. `result` contains the final `AgentResult`.

The observer:
1. Captures the worker's complete conversation transcript (stored as `worker_transcript` row for audit)
2. The orchestrator later extracts claims from this transcript via the Flock-with-cloned-context pattern (see below)
3. Extracted claims are stored as `finding` rows with `parent_id` → transcript row

The worker never knows this is happening. Its context stays pristine.

## Cloned Context as Flock Backend: The Expert LLM Pattern

### The Core Insight

Each worker accumulates angle-specific expertise in its conversation history. This expertise is valuable beyond just the worker's reasoning output — it can drive store operations (disaggregation, relevance scoring, dedup judgments) with domain-specific intelligence.

But you don't modify the worker's context to do this. Instead, you **clone the worker's context** and use it as the LLM backend for Flock SQL queries.

### How It Works

Flock's `llm_complete`/`llm_filter` needs an LLM to drive it. Normally that's a generic endpoint. Instead, you point Flock at a **session proxy** that prepends the worker's conversation history to every request:

```
Orchestrator runs Flock SQL
    │
    ▼
DuckDB executes llm_filter("is this relevant to insulin timing?")
    │
    ▼
Flock sends request to session proxy
    │
    ▼
Session proxy prepends insulin worker's full conversation history
    │
    ▼
vLLM receives: [worker's conversation] + [Flock's question]
    │
    ▼
vLLM prefix cache: conversation already cached from prior calls → fast
    │
    ▼
Response: domain-expert judgment informed by full angle context
    │
    ▼
Orchestrator stores result as row with parent_id lineage
```

### The Session Proxy

A thin FastAPI service (~100 lines) that sits between Flock and vLLM:

```python
# Maintains per-worker conversation contexts
sessions: dict[str, list[dict]] = {}

@app.post("/v1/chat/completions")
async def completions(request: ChatRequest):
    session_id = request.headers.get("X-Session-Id")
    if session_id and session_id in sessions:
        # Prepend the worker's conversation as context
        messages = sessions[session_id] + request.messages
    else:
        messages = request.messages
    # Forward to vLLM
    return await vllm_client.chat(messages=messages, ...)
```

### vLLM Prefix Caching

vLLM has built-in prefix caching. When the proxy sends the worker's conversation as a prefix:
- **First call**: vLLM computes the full KV cache for the conversation (expensive)
- **Subsequent calls**: vLLM detects the same prefix, reuses the cached KV states (nearly free)

On the 8×H200, each GPU serves one model. The proxy maintains conversation histories for each worker. Flock calls route through the proxy, vLLM caches the KV states. No extra infrastructure needed.

### The Flow

After each research wave:

1. **Worker reasons** → produces text (tool-free)
2. **Hook observer captures** → stores transcript as audit row
3. **Orchestrator clones worker context** → registers conversation with session proxy
4. **Orchestrator runs Flock SQL** → `llm_filter`, `llm_complete` route through the cloned context
5. **Cloned context answers** with domain expertise:
   - "Is this finding relevant to my angle?" → relevance scoring
   - "Break this paragraph into atomic claims" → disaggregation
   - "Are these two findings the same claim?" → semantic dedup
6. **All results stored** as rows with `parent_id` lineage
7. **Real worker untouched** → continues in next wave with fresh data package

### Why This Pattern Is Powerful

- The worker's accumulated context becomes a **reusable expert resource** — not just for its own reasoning, but for driving intelligent store operations
- Each worker is the **relevance oracle for its own angle** — the insulin specialist scores insulin relevance, the hematology specialist scores hematology relevance
- The real worker's context is **never polluted** — the clone is disposable, the original continues unmodified
- **Self-improvement**: the audit trail of extracted claims vs. full conversation transcripts enables measuring extraction recall and improving over time

## Data Flow: No Truncation, Smart Retrieval

### The Problem with Truncation

Truncation (cutting off tool returns at N characters) destroys information and breaks the audit trail. If a data package was truncated, you cannot audit:
- What the worker could have seen but didn't
- Whether the truncated material contained the critical connection
- Whether a different truncation boundary would have changed the worker's conclusions

### The Solution: Disaggregation + Per-Angle Relevance Ranking

Instead of returning large blobs and truncating, the system:

1. **Disaggregates** large findings into atomic claims (each stored as a row with `parent_id` → source)
2. **Ranks by per-angle relevance** — scored by the cloned worker context via Flock (the domain expert)
3. **Returns complete atoms** — no individual result is ever cut off

```
┌─────────────────────────────────────────────────────┐
│                   ConditionStore                     │
│                                                     │
│  Raw paragraphs ──► Atoms (via orchestrator + Flock)│
│       │                    │                        │
│       │              ┌─────┴──────┐                 │
│       │              │ parent_id  │                 │
│       │              └─────┬──────┘                 │
│       │                    │                        │
│       ▼                    ▼                        │
│  ┌─────────┐    ┌──────────────────┐               │
│  │ finding  │    │ atom (row_type)  │               │
│  │ (blob)   │    │ - atomic claim   │               │
│  └─────────┘    │ - parent_id→blob │               │
│                  └──────────────────┘               │
└──────────────────────┬──────────────────────────────┘
                       │
        Orchestrator runs Flock SQL per angle,
        routed through cloned worker contexts
                       │
          ┌────────────┼────────────┐
          │            │            │
    Clone A        Clone B     Clone C
    (insulin       (hematology (ancillaries
     context)       context)    context)
          │            │            │
     Relevance    Relevance    Relevance
     scored by    scored by    scored by
     insulin      hematology   ancillaries
     expert       expert       expert
          │            │            │
     Top-N atoms  Top-N atoms  Top-N atoms
     (complete,   (complete,   (complete,
      no cutoff)   no cutoff)   no cutoff)
          │            │            │
          ▼            ▼            ▼
     Data package  Data package Data package
     for Worker A  for Worker B for Worker C
     (next wave)   (next wave)  (next wave)
```

## Compaction: Decisions Are Data

When the store grows large (200K+ rows over 24 hours), compaction reduces noise. But compaction decisions are themselves stored:

### Phase 1: Exact-Match Dedup (Pure SQL)

```sql
-- Find identical fact text within same angle
SELECT angle, fact, COUNT(*) as cnt
FROM conditions
WHERE consider_for_use = TRUE
GROUP BY angle, fact
HAVING COUNT(*) > 1
```

For each duplicate group:
- Keep the highest-confidence row active
- Mark others `consider_for_use = FALSE`
- Store a `dedup_decision` row: `parent_id` → obsoleted row, `related_id` → kept row, `obsolete_reason = 'exact_duplicate'`

### Phase 2: Semantic Dedup (LLM-Assisted via Cloned Context)

The orchestrator runs Flock SQL with the cloned worker context as the LLM backend:

1. Pre-cluster candidates by shared keywords within angle (union-find, no cartesian product)
2. For each cluster, Flock `llm_filter` driven by the angle's cloned context: "are these the same claim?"
3. Mark lower-confidence duplicates obsolete
4. Store each decision: `row_type='dedup_decision'`, `parent_id` → obsoleted, `related_id` → keeper, `obsolete_reason = 'semantic_duplicate'`

The domain expert's judgment drives dedup — the insulin specialist decides whether two insulin claims are truly identical, not a generic model.

## Corpus Fingerprinting

To prevent re-ingesting the same corpus across runs:

```sql
CREATE TABLE corpus_fingerprints (
    fingerprint TEXT PRIMARY KEY,   -- SHA256 of corpus content
    source TEXT,                     -- File path or identifier
    ingested_at TEXT,                -- ISO timestamp
    char_count INTEGER,              -- Size of corpus
    paragraph_count INTEGER          -- Number of paragraphs stored
)
```

Before ingestion, compute `SHA256(corpus_text)` and check `corpus_fingerprints`. If present, skip.

## Source Provenance

Every finding carries:

```sql
source_model TEXT   -- e.g., "huihui-ai/Qwen3.5-32B-abliterated"
source_run   TEXT   -- e.g., "run_20250416_120000"
```

This enables:
- **Epistemic diversity measurement**: How many unique claims per model?
- **Convergent discovery detection**: Two models storing the same claim independently = high confidence
- **Model contribution audit**: Which model architecture found the iron+trenbolone connection?

## Querying the DAG

### Trace a finding back to raw corpus

```sql
WITH RECURSIVE lineage AS (
    SELECT id, fact, parent_id, row_type, 0 as depth
    FROM conditions WHERE id = ?
    UNION ALL
    SELECT c.id, c.fact, c.parent_id, c.row_type, l.depth + 1
    FROM conditions c
    JOIN lineage l ON c.id = l.parent_id
)
SELECT * FROM lineage ORDER BY depth DESC;
```

### What did Model X uniquely find?

```sql
SELECT fact, confidence, angle
FROM conditions
WHERE source_model = 'huihui-ai/Qwen3.5-32B-abliterated'
  AND source_type = 'worker_analysis'
  AND consider_for_use = TRUE
  AND fact NOT IN (
      SELECT fact FROM conditions
      WHERE source_model != 'huihui-ai/Qwen3.5-32B-abliterated'
        AND consider_for_use = TRUE
  )
ORDER BY confidence DESC;
```

### Show all dedup decisions

```sql
SELECT d.id, d.obsolete_reason,
       obsoleted.fact as removed_claim,
       keeper.fact as kept_claim,
       d.source_model as judged_by
FROM conditions d
JOIN conditions obsoleted ON d.parent_id = obsoleted.id
JOIN conditions keeper ON d.related_id = keeper.id
WHERE d.row_type = 'dedup_decision'
ORDER BY d.created_at DESC;
```

### Cross-domain connections for a specific pair of angles

```sql
SELECT c.fact, c.confidence,
       a.fact as finding_a, b.fact as finding_b
FROM conditions c
JOIN conditions a ON c.parent_id = a.id
JOIN conditions b ON c.related_id = b.id
WHERE c.row_type = 'insight'
  AND a.angle LIKE '%insulin%'
  AND b.angle LIKE '%hematology%'
ORDER BY c.confidence DESC;
```

## Rolling Knowledge Summaries

Between waves, the engine generates condensed briefings per angle:

```sql
CREATE TABLE knowledge_summaries (
    id INTEGER PRIMARY KEY,
    angle TEXT NOT NULL,
    summary TEXT NOT NULL,
    finding_count INTEGER,
    created_at TEXT,
    run_number INTEGER
)
```

Summaries are injected into worker data packages so workers see compressed state instead of raw finding lists. The summaries are a **view** — the underlying findings remain intact in `conditions`.

## Design Invariants

1. **No truncation.** Ever. The orchestrator controls data package size via disaggregation + relevance ranking. No individual atom is ever cut off.
2. **No data modification.** Raw corpus, findings, atoms — all immutable once written. `consider_for_use = FALSE` hides but never deletes.
3. **Every operation is a row.** Dedup decisions, relevance scores, atomisation, conversation transcripts — all stored with `parent_id` lineage.
4. **`parent_id` connects everything.** Any row can be traced back through the DAG to its ultimate source.
5. **Workers are tool-free.** They receive data packages and reason. They don't interact with the store directly.
6. **Cloned context drives Flock.** The orchestrator uses cloned worker contexts as expert LLM backends for store operations. The real worker is never modified.
7. **The orchestrator runs all store operations.** Compaction, disaggregation, relevance scoring, data package preparation — all orchestrator-level, never worker-level.

## Implementation Files

| File | Responsibility |
|---|---|
| `apps/strands-agent/corpus.py` | `ConditionStore` — schema, reads, writes, compaction, fingerprinting |
| `swarm/mcp_engine.py` | Orchestrator — prepares data packages, runs Flock queries, calls compaction between waves |
| `swarm/agent_worker.py` | Factory for creating tool-free worker agents |
| `swarm/worker_observer.py` | Strands hook provider that captures worker conversation transcripts |
| `swarm/session_proxy.py` | FastAPI proxy that prepends cloned worker context for vLLM prefix caching |
| `scripts/h200_test/run_swarm_test.py` | CLI runner exposing all hyperparameters |
