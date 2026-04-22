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
| `tool_call` | Audit record of a tool invocation | `NULL` (standalone event) |
| `dedup_decision` | Record of a compaction judgment | Row that was judged |
| `relevance_score` | Per-angle relevance assessment | Row that was scored |
| `atom` | Disaggregated atomic claim from a larger blob | Source blob row |
| `similarity` | Detected similarity between two findings | One finding (`related_id` → other) |
| `contradiction` | Detected contradiction between findings | One finding (`related_id` → other) |
| `synthesis` | Merged/summarized knowledge briefing | Representative finding from angle |

### Key Columns for Lineage

```sql
parent_id       INTEGER   -- Direct parent in the DAG (what produced this row)
related_id      INTEGER   -- Secondary relationship (e.g., the "other" in a dedup pair)
source_model    TEXT      -- Which model architecture produced this row
source_run      TEXT      -- Which run produced this row (e.g., "run_20250416_120000")
source_type     TEXT      -- How this row was created (e.g., "worker_analysis", "corpus_ingestion", "compaction")
phase           TEXT      -- Swarm phase when created (e.g., "worker", "serendipity", "compact")
```

## Data Flow: No Truncation, Smart Retrieval

### The Problem with Truncation

Truncation (cutting off tool returns at N characters) destroys information and breaks the audit trail. If a worker's `search_corpus` call returns 15 results but the tool truncated 85 more, you cannot audit:
- What the worker could have seen but didn't
- Whether the truncated results contained the critical connection
- Whether a different truncation boundary would have changed the worker's conclusions

### The Solution: Disaggregation + Per-Angle Relevance Ranking

Instead of returning large blobs and truncating, the system:

1. **Disaggregates** large findings into atomic claims (each stored as a row with `parent_id` → source)
2. **Ranks by per-angle relevance** — each worker has a different angle, so the same query returns different results for different workers
3. **Returns complete atoms** — no individual result is ever cut off; the worker controls how many results via `max_results`

```
┌─────────────────────────────────────────────────────┐
│                   ConditionStore                     │
│                                                     │
│  Raw paragraphs ──► Atoms (via Flock query agent)   │
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
          ┌────────────┼────────────┐
          │            │            │
    Worker A       Worker B    Worker C
    (insulin)    (hematology) (ancillaries)
          │            │            │
     Relevance    Relevance    Relevance
     scored for   scored for   scored for
     insulin      hematology   ancillaries
          │            │            │
     Top-N atoms  Top-N atoms  Top-N atoms
     (complete,   (complete,   (complete,
      no cutoff)   no cutoff)   no cutoff)
```

### The Flock Query Agent

A dedicated agent **outside the swarm** that:

1. Reads raw findings/paragraphs from the store
2. Uses Flock `llm_complete` in SQL to break them into atomic claims
3. Stores each atom as a new row: `row_type='atom'`, `parent_id` → source
4. Scores relevance per-angle using Flock `llm_filter` or keyword overlap
5. Stores relevance scores as rows: `row_type='relevance_score'`, `parent_id` → atom

This agent runs **between waves** or on a schedule. It does not consume worker context — it has its own LLM calls via Flock.

### Worker Tool Behavior

When a worker calls `search_corpus(query, max_results=15)`:

1. SQL query fetches atoms + findings matching the query terms
2. Results ranked by relevance to **this worker's specific angle**
3. Top `max_results` returned — each result is a complete atomic claim, never truncated
4. The tool call itself is logged as a `tool_call` row in the store

The worker controls volume via `max_results`. The system controls quality via relevance ranking. No information is destroyed.

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

### Phase 2: Semantic Dedup (LLM-Assisted)

Called by an external agent (not swarm workers):

1. Pre-cluster candidates by shared keywords within angle (union-find, no cartesian product)
2. For each cluster, one LLM call: "which of these are the same claim?"
3. Mark lower-confidence duplicates obsolete
4. Store each decision: `row_type='dedup_decision'`, `parent_id` → obsoleted, `related_id` → keeper, `obsolete_reason = 'semantic_duplicate'`

The LLM's judgment is part of the audit trail. You can query: "show me every finding that was removed by semantic dedup and what it was compared against."

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

Before ingestion, compute `SHA256(corpus_text)` and check `corpus_fingerprints`. If present, skip. The fingerprint check itself can be logged.

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

Workers call `get_knowledge_briefing()` to see compressed state instead of reading every individual finding. The summaries are a **view** — the underlying findings remain intact in `conditions`.

## Design Invariants

1. **No truncation.** Ever. Workers control volume via `max_results`. The system controls quality via relevance ranking.
2. **No data modification.** Raw corpus, findings, atoms — all immutable once written. `consider_for_use = FALSE` hides but never deletes.
3. **Every operation is a row.** Tool calls, dedup decisions, relevance scores, atomisation — all stored with `parent_id` lineage.
4. **`parent_id` connects everything.** Any row can be traced back through the DAG to its ultimate source.
5. **Per-angle relevance.** The same query returns different results for different workers because relevance is scored against the worker's specific research angle.
6. **External agents handle store hygiene.** Compaction, disaggregation, and relevance scoring are done by agents outside the swarm. Workers only read and write findings.

## Implementation Files

| File | Responsibility |
|---|---|
| `apps/strands-agent/corpus.py` | `ConditionStore` — schema, reads, writes, compaction, fingerprinting |
| `swarm/worker_tools.py` | `@tool`-decorated functions workers use to interact with the store |
| `swarm/mcp_engine.py` | Orchestrator — calls compaction, rolling summaries between waves |
| `swarm/agent_worker.py` | Factory for creating Strands Agent workers with store tools |
| `scripts/h200_test/run_swarm_test.py` | CLI runner exposing all hyperparameters |
