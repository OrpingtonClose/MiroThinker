# Converged Depth Architecture

## The Universal Store Architecture — Depth, Understandability, and Intelligent External Data Selection

**Date:** 2026-04-26  
**Synthesis of:** 7 parallel subagent depth analyses + framework recommendation + refined master architecture  
**Status:** AUTHORITATIVE — this document is the architecture. Implement it.

---

## 1. The Core Insight

DEPTH and UNDERSTANDABILITY in analytical reports are not achieved by running more of the same queries. They are achieved by a **seven-pillar convergence**: (a) an expanded query wealth engine that fires 28 distinct query types — from surface-level VALIDATE to deep CAUSAL_TRACE, assumption excavation, and counterfactual probing — ensuring no dimension of a finding goes unexamined; (b) a semantic connection pipeline that detects, verifies, and stores relationships between findings as first-class data, enabling BRIDGE and SYNTHESIZE to operate on verified links rather than coarse cluster proximity; (c) an external data benefit assessment layer that treats every API call as an investment decision, scoring 13 distinct benefit signals, estimating cost in USD/tokens/latency, and selecting targets via UCB-greedy optimization so that external research is deployed only where internal evaluation has hit diminishing returns; (d) eight specialized curation agents that continuously transform raw store rows into actionable consumables — convergence signals for the orchestrator, briefing narratives for the operator, prefix-cache-optimized contexts for clones, and ranked contradiction digests for the rule engine — preventing context window overflow and information blindness; (e) a three-tier operator query interface that lets the human steer with structured templates, guided natural language, or freeform SQL, including cross-run temporal queries that reveal how findings evolve across sessions; (f) a reflexion loop that persists typed lessons across VM lifetimes via incremental Parquet deltas to B2, feeding back into clone context prefixes and scheduler rules so that the system does not relearn "Reddit is unreliable for medical claims" on every new run; and (g) a massive source ingestion pipeline that downloads, chunks, embeds, and atomizes books from Anna's Archive and beyond, tracking source utility per query embedding so that a 500-page textbook is never loaded whole into a worker's context but is instead retrieved via hybrid BM25+vector search as precisely relevant passages. These seven pillars are not independent features. They are a single converged system where query wealth generates findings, semantic connections weave them into a graph, external data fills gaps the graph exposes, curation surfaces the results to consumers, the operator steers when automation stalls, reflexion learns from every cycle, and massive sources feed the bottom of the funnel without overwhelming the top. That is how depth is manufactured.

---

## 2. Unified Actor Supervision Tree

```
OrchestratorActor (root)
│
├── Policy Layer (RuleEngineActor)
│   └── Rules: convergence-stuck trigger, cost-cap hard stop, operator override priority
│
├── SwarmSupervisor
│   ├── BeeActor (angle 1..N)
│   └── DiffusionSupervisor
│       ├── ManifestWorker
│       ├── ConfrontWorker
│       └── CorrectWorker
│
├── FlockSupervisor
│   └── CloneActor (angle 1..N)  ← vLLM prefix-cached
│
├── McpResearcherActor
│   ├── BraveToolActor
│   ├── ExaToolActor
│   ├── TavilyToolActor
│   ├── PubMedToolActor
│   └── AnnaArchiveToolActor  ← NEW
│
├── UserProxyActor
│   └── Interrupt buffer, question queue, resume signals
│
├── OperatorQueryEngine  ← NEW
│   ├── TemplateRegistry (Tier 1)
│   ├── QueryPlanner (Tier 2)
│   ├── MaestroSQLGate (Tier 3)
│   └── DigestGenerator
│
├── CurationSupervisor  ← NEW
│   ├── GlobalHealthCurator      → continuous
│   ├── AngleContextCurator      → on-demand + cached
│   ├── CloneContextCurator      → on-demand + cached
│   ├── ContradictionCurator     → continuous
│   ├── GapCurator               → continuous
│   ├── LessonCurator            → event-driven
│   ├── SourceQualityCurator     → periodic (every N rounds)
│   └── NarrativeCurator         → continuous
│
├── ReflexionActor  ← NEW
│   └── LessonStore (lessons table)
│
├── StoreSyncActor  ← NEW
│   └── Delta export + B2 snapshot every 60s
│
├── SemanticConnectionWorker  ← NEW
│   └── 3-stage pipeline: heuristic → embedding → LLM verification
│
└── SourceIngestionActor  ← NEW
    ├── SelectionWorker (metadata ranking)
    ├── DownloadWorker (streaming + resume)
    ├── ExtractionWorker (OCR if needed)
    └── ChunkEmbedWorker (DuckDB VSS)
```

**Placement rationale:**
- `CurationSupervisor` is a sibling of Swarm/Flock because it serves multiple consumers. If nested under SwarmSupervisor, Flock would need to ask Swarm for curated data — incorrect coupling.
- `ReflexionActor` is a sibling to UserProxyActor; it receives `FlockComplete` and `SwarmComplete` events and writes to the `lessons` table without blocking the pipeline.
- `StoreSyncActor` checkpoints the DuckDB file to B2 every 60 seconds and uploads a full base snapshot at run end, ensuring VM teardown does not erase intelligence.
- `SemanticConnectionWorker` runs as a background actor that consumes `StoreChangeEvent` messages from CurationSupervisor and produces `semantic_connections` rows.
- `SourceIngestionActor` handles the massive-source pipeline: book selection, download, extraction, chunking, embedding, and registration in `source_fingerprints` / `chunks`.

---

## 3. Converged Data Model

All schema additions are **additive only**. No columns are dropped. The append-only contract is preserved.

### 3.1 Core Table: `conditions` (Extended)

Existing columns plus these backward-compatible additions:

| Column | Type | Purpose |
|--------|------|---------|
| `run_number` | INTEGER | FK to `runs` table |
| `run_id` | TEXT | Cross-run federation key |
| `diffusion_pass` | INTEGER | Which diffusion pass |
| `diffusion_phase` | TEXT | `scaffold` \| `manifest` \| `confront` \| `correct` \| `stitch` |
| `section_angle` | TEXT | Diffusion section author |
| `convergence_status` | TEXT | `pending` \| `confirmed` \| `corrected` \| `converged` |
| `critique_target_id` | INTEGER | For critiques: which section row |
| `diffusion_report_id` | INTEGER | FK to final report row |
| `scores_json` | TEXT | Extensible gradient dimensions |
| `provenance_system` | TEXT | `swarm_extraction` \| `flock_bridge` \| `mcp_research` |
| `query_type` | TEXT | Which Flock query produced this row |
| `depth_evaluated` | BOOLEAN DEFAULT FALSE | Has CAUSAL_TRACE / ASSUMPTION_EXCAVATE run? |
| `understandability_evaluated` | BOOLEAN DEFAULT FALSE | Has TIER_SUMMARIZE / ANALOGY run? |

**Ghost columns to populate:** `cluster_id` (union-find clustering), `contradiction_flag` (contradiction detection scoring), `trust_score` (activated by SourceQualityCurator and EVIDENCE_MAP).

### 3.2 New Tables

```sql
-- Run registry
CREATE TABLE runs (
    run_number INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TEXT, ended_at TEXT,
    convergence_reason TEXT,
    total_queries INTEGER,
    total_cost_usd FLOAT,
    source_query TEXT,
    vm_id TEXT DEFAULT ''
);

-- Score history for temporal analysis
CREATE TABLE score_history (
    id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL,
    run_number INTEGER,
    evaluator_angle TEXT,
    query_type TEXT,
    old_confidence FLOAT, new_confidence FLOAT,
    old_novelty_score FLOAT, new_novelty_score FLOAT,
    old_specificity_score FLOAT, new_specificity_score FLOAT,
    old_relevance_score FLOAT, new_relevance_score FLOAT,
    old_actionability_score FLOAT, new_actionability_score FLOAT,
    old_fabrication_risk FLOAT, new_fabrication_risk FLOAT,
    magnitude FLOAT,
    evaluated_at TEXT
);

-- Lessons / reflexion persistence
CREATE TABLE lessons (
    id INTEGER PRIMARY KEY,
    lesson_type TEXT NOT NULL,  -- strategy_lesson | source_quality_lesson | model_behavior_lesson
                                -- | cost_lesson | query_efficiency_lesson | angle_quality_lesson
    fact TEXT NOT NULL,
    run_id TEXT NOT NULL,
    run_number INTEGER,
    angle TEXT DEFAULT '',
    query_type TEXT DEFAULT '',
    source_url TEXT DEFAULT '',
    source_type TEXT DEFAULT '',
    relevance_score FLOAT DEFAULT 0.5,
    confidence FLOAT DEFAULT 0.5,
    metadata JSON,
    parent_id INTEGER,
    created_at TEXT DEFAULT ''
);

-- Lesson application audit
CREATE TABLE lesson_applications (
    id INTEGER PRIMARY KEY,
    lesson_id INTEGER NOT NULL,
    target_run_id TEXT NOT NULL,
    target_actor TEXT DEFAULT '',
    application_method TEXT DEFAULT '',  -- clone_context_prefix | scheduler_rule | mcp_filtering
    applied_at TEXT DEFAULT ''
);

-- Semantic connections between findings
CREATE TABLE semantic_connections (
    id INTEGER PRIMARY KEY,
    source_condition_id INTEGER NOT NULL,
    target_condition_id INTEGER NOT NULL,
    connection_type TEXT NOT NULL,  -- see 10 types below
    detection_stage TEXT NOT NULL,  -- heuristic | embedding | llm_verified
    confidence FLOAT DEFAULT 0.0,
    explanation TEXT DEFAULT '',
    created_at TEXT DEFAULT '',
    verified_at TEXT DEFAULT ''
);

-- Source fingerprinting for massive ingestion
CREATE TABLE source_fingerprints (
    id INTEGER PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- annas_archive | libgen | zlibrary | open_library
    byte_sha256 TEXT,
    text_sha256 TEXT,
    text_simhash TEXT,
    isbn TEXT,
    doi TEXT,
    ol_key TEXT,
    content_hash TEXT,
    ingested_at TEXT,
    last_accessed_at TEXT,
    extraction_status TEXT DEFAULT 'pending',
    download_method TEXT,
    legal_tier TEXT,
    metadata_json TEXT DEFAULT '{}'
);

-- Chunk-level storage with embeddings
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    source_fingerprint_id INTEGER NOT NULL REFERENCES source_fingerprints(id),
    parent_condition_id INTEGER NOT NULL REFERENCES conditions(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    chapter_title TEXT DEFAULT '',
    section_title TEXT DEFAULT '',
    page_start INTEGER,
    page_end INTEGER,
    embedding FLOAT[768],
    token_count INTEGER,
    char_count INTEGER,
    created_at TEXT
);

-- Source utility tracking
CREATE TABLE source_utility_log (
    id INTEGER PRIMARY KEY,
    source_fingerprint_id INTEGER NOT NULL REFERENCES source_fingerprints(id),
    query_embedding FLOAT[768],
    query_text TEXT,
    angle TEXT,
    times_queried INTEGER DEFAULT 0,
    chunks_retrieved INTEGER DEFAULT 0,
    findings_produced INTEGER DEFAULT 0,
    avg_chunk_relevance FLOAT DEFAULT 0.0,
    utility_score FLOAT DEFAULT 0.5,
    utility_verdict TEXT DEFAULT 'pending',  -- pending | useful | marginal | useless | unprocessable
    last_queried_at TEXT,
    last_verdict_at TEXT,
    block_future_downloads BOOLEAN DEFAULT FALSE,
    block_reason TEXT
);

-- Source quality registry for learned preferences
CREATE TABLE source_quality_registry (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    authority_score FLOAT DEFAULT 0.5,
    avg_recency_score FLOAT DEFAULT 0.5,
    avg_finding_confidence FLOAT DEFAULT 0.5,
    fetch_count INTEGER DEFAULT 0,
    successful_fetch_count INTEGER DEFAULT 0,
    total_cost_usd FLOAT DEFAULT 0.0,
    total_info_gain_generated FLOAT DEFAULT 0.0,
    first_seen_at TEXT,
    last_seen_at TEXT,
    UNIQUE(domain, source_type)
);

-- Per-condition source linkage
CREATE TABLE condition_sources (
    condition_id INTEGER NOT NULL,
    source_registry_id INTEGER NOT NULL,
    source_url TEXT,
    extracted_fact TEXT,
    confidence_at_extraction FLOAT,
    PRIMARY KEY (condition_id, source_registry_id)
);

-- Condition embeddings for semantic search
CREATE TABLE condition_embeddings (
    condition_id INTEGER PRIMARY KEY REFERENCES conditions(id),
    embedding FLOAT[768],
    embedding_model TEXT DEFAULT '',
    created_at TEXT DEFAULT ''
);
```

### 3.3 Index Strategy

```sql
-- Covering index for Flock queries (PARAMOUNT — without this, the store is unusable at scale)
CREATE INDEX idx_findings_covering ON conditions (
    consider_for_use, row_type, score_version,
    confidence DESC, novelty_score, fabrication_risk,
    specificity_score, relevance_score, actionability_score
) INCLUDE (id, fact, angle, information_gain, evaluation_count, cluster_id);

-- Semantic connection queries
CREATE INDEX idx_semantic_conn_source ON semantic_connections(source_condition_id, connection_type);
CREATE INDEX idx_semantic_conn_target ON semantic_connections(target_condition_id, connection_type);
CREATE INDEX idx_semantic_conn_type ON semantic_connections(connection_type, confidence DESC);

-- Lessons fast retrieval
CREATE INDEX idx_lessons_actor_query ON lessons (
    lesson_type, angle, query_type, confidence DESC, relevance_score DESC
) INCLUDE (id, fact, metadata, run_id);
CREATE INDEX idx_lessons_temporal ON lessons (created_at DESC, lesson_type);
CREATE INDEX idx_lessons_run ON lessons (run_id, run_number);

-- Source tracking
CREATE INDEX idx_fp_text_simhash ON source_fingerprints(text_simhash);
CREATE INDEX idx_fp_isbn ON source_fingerprints(isbn);
CREATE INDEX idx_fp_content_hash ON source_fingerprints(content_hash);
CREATE INDEX idx_chunks_source ON chunks(source_fingerprint_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_condition_id);
CREATE INDEX idx_sul_verdict ON source_utility_log(utility_verdict, block_future_downloads);
CREATE INDEX idx_sul_source ON source_utility_log(source_fingerprint_id);

-- DuckDB VSS vector index for chunk retrieval
CREATE INDEX idx_chunks_embedding ON chunks USING HNSW (embedding) WITH (metric = 'cosine');
CREATE INDEX idx_condition_embeddings ON condition_embeddings USING HNSW (embedding) WITH (metric = 'cosine');
```

---

## 4. The Query Wealth Engine

### 4.1 The Complete Type Catalog (28 Types)

| Layer | Count | Types |
|-------|-------|-------|
| **Foundation** | 9 | VALIDATE, ADJUDICATE, VERIFY, ENRICH, GROUND, BRIDGE, CHALLENGE, SYNTHESIZE, AGGREGATE |
| **Depth** | 8 | CAUSAL_TRACE, ASSUMPTION_EXCAVATE, EVIDENCE_MAP, SCOPE, METHODOLOGY, TEMPORAL, ONTOLOGY, REPLICATION |
| **Understandability** | 4 | ANALOGY, TIER_SUMMARIZE, COUNTERFACTUAL, NARRATIVE_THREAD |
| **Meta** | 4 | META_PRODUCTIVITY, META_EXHAUSTION, META_COVERAGE, META_EFFECTIVENESS |
| **Composite** | 3 | DEEP_VALIDATE, RESOLVE_CONTRADICTION, SYNTHESIS_DEEPEN |
| **Total** | **28** | |

### 4.2 State-Driven Selection Decision Tree

The orchestrator allocates the per-round query budget according to store topology, not just prior magnitudes:

```
STORE STATE ANALYSIS (per round start)
│
├── High novelty + low confidence findings > threshold?
│   └── Allocate 25% to VALIDATE, 15% to DEEP_VALIDATE
│
├── Contradiction flags set?
│   ├── Simple binary → ADJUDICATE
│   └── Complex/multi-angle → RESOLVE_CONTRADICTION
│
├── High fabrication_risk findings?
│   └── Allocate 15% to VERIFY, 10% to REPLICATION
│
├── Low specificity + high relevance?
│   └── Allocate 10% to ENRICH, 5% to EVIDENCE_MAP
│
├── High actionability + unverified?
│   └── Allocate 10% to GROUND, 5% to EVIDENCE_MAP
│
├── Clusters with 3+ members?
│   ├── First synthesis → SYNTHESIZE
│   └── Prior synthesis exists → SYNTHESIS_DEEPEN
│
├── High-confidence single-angle findings?
│   └── Allocate 10% to CHALLENGE, 5% to COUNTERFACTUAL
│
├── Cross-angle cluster members?
│   └── Allocate 10% to BRIDGE, 5% to CAUSAL_TRACE
│
├── Single-source high-confidence findings?
│   └── Allocate 5% to REPLICATION, 5% to SOURCE_QUALITY
│
├── Older findings with newer conflicting evidence?
│   └── Allocate 5% to TEMPORAL
│
├── High-deviation cluster outliers?
│   └── Allocate 5% to ONTOLOGY
│
├── Information gain rate dropping?
│   └── META_COVERAGE audit
│
└── Round completion
    ├── META_PRODUCTIVITY analysis
    ├── META_EFFECTIVENESS analysis
    └── META_EXHAUSTION check
```

### 4.3 Reflexion-Informed Query Selection

The orchestrator maintains a `ReflexionState` dataclass that is updated after every round:

```python
@dataclass
class ReflexionState:
    exhausted_query_types: set[str]   # types with near-zero magnitude for 2+ rounds
    productive_pairs: list[tuple[str, str]]  # synergistic sequences
    breakthrough_findings: list[int]  # ONTOLOGY outlier_breakthrough IDs
    coverage_score_history: list[float]
```

**Rules:**
1. If a type is in `exhausted_query_types` for 2 rounds, demote its budget by 50% and promote its successor (e.g., VALIDATE → DEEP_VALIDATE).
2. If `productive_pairs` shows VALIDATE→CHALLENGE sequences produce high magnitude, fire CHALLENGE on freshly VALIDATEd findings even if they do not yet meet the `confidence > 0.8` threshold.
3. If `breakthrough_findings` is non-empty, boost ONTOLOGY, CAUSAL_TRACE, and SYNTHESIZE.
4. If `coverage_score_history` is flat, boost META_COVERAGE, AGGREGATE, and BRIDGE.

### 4.4 Semantic-Connection-Driven Triggering

When the `SemanticConnectionWorker` detects a verified connection, it emits a `ConnectionDetectedEvent` that the orchestrator uses to trigger targeted queries:

| Connection Type | Triggered Query |
|-----------------|-----------------|
| `causal_link` | CAUSAL_TRACE on both endpoints |
| `contradictory` | ADJUDICATE or RESOLVE_CONTRADICTION |
| `supporting` | REPLICATION or EVIDENCE_MAP |
| `generalizing` | SCOPE on the more specific endpoint |
| `analogous` | ANALOGY generation |

This replaces the coarse cluster-based BRIDGE trigger with precise, verified-link targeting.

### 4.5 External Data-Driven Query Injection

When the `McpResearcherActor` injects new `mcp_finding` rows, the orchestrator immediately overrides the next round's budget:

```python
if new_mcp_findings > 50:
    override = BudgetOverride(
        boost={"VALIDATE": 1.5, "VERIFY": 1.5, "TEMPORAL": 1.3},
        pause={"SYNTHESIZE"},  # wait until new findings are scored
        reason="MCP injection: validate before synthesize"
    )
```

---

## 5. External Data Selection Architecture

**This section is PARAMOUNT.** The way one chooses external data determines whether the system is an intelligent research engine or an expensive random sampler.

### 5.1 The 13 Benefit Signals

Each signal measures a distinct information deficit in the store. They are gradients, not binary flags.

| Signal | Condition | Weight | Meaning |
|--------|-----------|--------|---------|
| VALIDATE gap | `novelty > 0.6 AND confidence < 0.4` | 1.4× | Surprising but untrusted finding — external corroboration can flip it |
| VERIFY gap | `fabrication_risk > 0.5` | 1.3× | Smells like hallucination; needs ground-truth database |
| ADJUDICATE gap | `contradiction_flag = TRUE` | 1.35× | Blocks synthesis until resolved; highest priority |
| SWARM_INTENT gap | `row_type = 'research_target'` | 1.25× | Strategic priorities chosen by collective intelligence |
| WORKER_REQUEST gap | `expansion_gap != '' AND expansion_fulfilled = FALSE` | 1.2× | Precisely scoped research requests from the swarm |
| SOURCE_UPGRADE gap | `trust_score < 0.4 AND actionability > 0.6` | 1.1× | Prevents high-actionability garbage from reaching reports |
| ENRICH gap | `specificity < 0.4 AND relevance > 0.5` | 1.0× | Directionally correct but vague |
| GROUND gap | `actionability > 0.6 AND verification_status IS NULL` | 1.0× | Actionable but ungrounded |
| REPLICATION gap | `confidence > 0.7 AND source_type = 'single_study'` | 0.9× | Single-study risk |
| DEADLOCK gap | `evaluation_count >= 3 AND information_gain < 0.1` | 0.9× | Circular internal debate; external data breaks it |
| MECHANISM gap | Cross-angle high-relevance pair, no link | 0.85× | Missing mechanistic bridge |
| FRESHNESS gap | Old finding with newer conflicting evidence | 0.8× | May have been superseded |
| COVERAGE gap | Sparse cluster (weak evidential support) | 0.75× | Diffuse benefit across many findings |

### 5.2 Composite Benefit Score

```python
B_raw(c) = Σ (signal_i(c) × weight_i)

B_final(c) = B_raw(c) × saturation_penalty(c) × context_penalty(estimated_tokens)
           × convergence_boost(avg_magnitude_last_2_rounds)
```

Where:
- `saturation_penalty` = `1.0 / (1.0 + 0.3 * log(1 + external_fetch_count))`, multiplied by 0.5 if last fetch was < 2 rounds ago.
- `context_penalty` discounts benefit as the store approaches its context window limit (linear drop above 50%, steep drop above 80%).
- `convergence_boost` = 1.5 when `avg_magnitude < 0.02` for 2 rounds (external data is the only remaining lever).

### 5.3 Cost Estimation Model

Every candidate target receives a `FetchCost` estimate before any API call:

| Source | $/Query | Avg Tokens | Latency (p50) | Best For |
|--------|---------|------------|---------------|----------|
| Brave Search | $0.00 | 800–1,500 | 1.2s | Verification |
| Exa | $0.01 | 1,000–2,000 | 2.5s | Academic, precise |
| Tavily | $0.015 | 2,000–4,000 | 3.0s | Deep research |
| Perplexity | $0.05–0.20 | 3,000–8,000 | 4.0s | Complex synthesis |
| Semantic Scholar | $0.00 | 1,500–3,000 | 1.5s | Paper discovery |
| PubMed | $0.00 | 1,000–2,500 | 1.8s | Biomedical |
| Anna's Archive | $0.00 | 5,000–50,000 | 5.0s | Long-form books |
| Firecrawl | $0.01/page | 3,000–10,000/page | 3.5s | Deep crawl |

Cost estimation includes **context window interest**: each new `mcp_finding` row will appear in ~30% of future clone contexts, so its true token cost is `new_tokens * 0.3 * expected_future_rounds`.

### 5.4 UCB-Greedy Target Selection

```python
for target in candidates:
    cost_norm = geometric_mean(normalized_usd, normalized_tokens, normalized_time)
    exploration_bonus = alpha * sqrt(log(total_fetches + 1) / (fetch_count + 1))
    target.efficiency = (benefit_score + exploration_bonus) / (cost_norm + 0.01)

# Greedy knapsack selection with diversity floor
targets.sort(key=lambda t: t.efficiency, reverse=True)
selected = []
for t in targets:
    if len(selected) >= max_targets: break
    if t.fits_in(remaining_budget): selected.append(t)

# Serendipity floor: ensure >= 3 reason types represented
if len(reason_types_seen) < 3:
    force_include_highest_efficiency_from_third_type()
```

### 5.5 Dynamic Budget Allocation with ROI Feedback

```python
@dataclass
class ResearchBudget:
    usd: float
    tokens: int
    time_s: float

# Compute per cycle
usd_budget = min(hourly_cap / 6, avg_recent_spend * 1.5)
token_budget = int(context_headroom * 0.7)
time_budget = max(5.0, target_round_time - avg_flock_time)

# Adjust based on ROI
if last_round_roi > 10.0:   multiplier = 1.3
elif last_round_roi > 1.0:  multiplier = 1.0
elif last_round_roi > 0.1:  multiplier = 0.6
else:                       multiplier = 0.2  # near-zero ROI → mostly pause
```

### 5.6 Source Quality Registry with Learned Preferences

The `source_quality_registry` learns which APIs are high-leverage for which claim types:

```python
def source_efficiency_score(registry_row) -> float:
    roi = registry_row.total_info_gain_generated / max(registry_row.total_cost_usd, 0.01)
    return registry_row.authority_score * 0.5 + min(1.0, roi / 100) * 0.5
```

If arXiv consistently yields findings that survive CHALLENGE with high confidence, its efficiency score rises and the selector prefers it. If Reddit consistently yields findings that VERIFY flags as fabricated, its score falls.

### 5.7 Operator Override (Green / Yellow / Red)

| Tier | Cost Threshold | Action | Default |
|------|---------------|--------|---------|
| **Green** | <$0.05 | Auto-execute | Yes |
| **Yellow** | $0.05–$0.50 | Log intent, execute after 10s unless paused | Yes |
| **Red** | >$0.50 OR >5K tokens OR >10s latency | Halt, emit event, await explicit approve/reject | No |

Operator decisions are stored as `thought` rows with `source_type = 'operator_override'` so the system remembers: "Operator rejected Perplexity for this finding; use PubMed instead next time."

### 5.8 Three-Phase Feedback Loop

**Immediate (within 1 round):**
- Measure direct score deltas on parent conditions after MCP findings are evaluated.
- Track contradiction resolutions and gap fulfillments.

**Latent (2–3 rounds later):**
- Count synthesis rows whose `parent_ids` include an MCP finding.
- Count cluster growth and bridge insights enabled by the fetched data.

**ROI (after latent phase):**
```python
total_info_gain = sum(confidence_delta + fabrication_delta + specificity_delta)
                + sum(synthesis_enabled * 0.5)
                + sum(bridge_insights * 0.3)

roi_per_usd = total_info_gain / max(cost.usd, 0.01)
```

The `source_quality_registry` is updated with these ROI metrics, closing the learning loop.

### 5.9 Integration with Massive Source Ingestion

The external data selector treats Anna's Archive books as a special source class:
- **Pre-selection:** Metadata-only query with selection score (`0.35 * relevance + 0.20 * citations + 0.15 * recency + 0.15 * reliability_tier + 0.10 * size_efficiency + 0.05 * language`).
- **Budget constraints:** `max_books_per_query = 5`, `max_total_mb = 500`, `download_timeout = 300s`.
- **Utility prediction:** Before downloading, check `source_utility_log` for similar past queries (cosine similarity > 0.75 on query embedding). If predicted utility < 0.2, deprioritize.
- **Post-ingestion:** After swarm atomization, update `source_utility_log` with `findings_produced` and set `utility_verdict`. Three-strikes rule blocks future downloads for useless sources.

---

## 6. Semantic Connection Pipeline

The current BRIDGE query uses coarse `cluster_id` proximity. The semantic connection pipeline replaces this with verified, typed links between findings.

### 6.1 The 10 Connection Types

| Type | Definition | Detection Strategy |
|------|------------|-------------------|
| `causal_link` | A causes B (or contributes to B) | CAUSAL_TRACE agreement from 2+ angles |
| `contradictory` | A and B cannot both be true | ADJUDICATE verdict or high confidence delta |
| `supporting` | B strengthens A's evidential basis | EVIDENCE_MAP independent support overlap |
| `analogous` | A and B share structural similarity | ANALOGY generation match |
| `generalizing` | A is a special case of B | SCOPE cross-angle prediction validated |
| `specializing` | B is a special case of A | Inverse of generalizing |
| `methodological` | A and B share methodological basis | METHODOLOGY query match |
| `temporal` | B supersedes or refines A | TEMPORAL verdict = partially/fully_superseded |
| `statistical` | A and B correlate but mechanism unclear | Embedding similarity + no causal agreement |
| `compositional` | A and B combine to produce C | SYNTHESIZE parent_ids linkage |

### 6.2 Three-Stage Pipeline

**Stage 1: Heuristic Pre-Filter**

Goal: Reduce O(N²) candidate pairs to O(N * k) with cheap, deterministic rules.

```python
def heuristic_filter(store, max_candidates=1000):
    candidates = []
    # Rule 1: Term overlap (Jaccard > 0.2 on stemmed content words)
    # Rule 2: Cluster membership (same cluster_id, different angle)
    # Rule 3: Shared source_url or source_type
    # Rule 4: Temporal proximity (created within same round)
    # Rule 5: Contradiction flag pair
    return candidates[:max_candidates]
```

**Stage 2: Embedding Gate**

Goal: Filter heuristic candidates with vector similarity and angle diversity.

```python
def embedding_gate(candidate, threshold=0.72, min_angle_diversity=0.3):
    sim = cosine_similarity(source_embedding, target_embedding)
    if sim < threshold:
        return False
    # Angle diversity: if same angle, require higher similarity
    if source.angle == target.angle and sim < threshold + 0.1:
        return False
    return True
```

**Stage 3: LLM Verification**

Goal: Verify the connection type with high accuracy. **Use Python Flock with prefix caching.** Never use FlockMTL for this stage — the 167× cost increase is unacceptable for a high-volume verification task.

```python
# Prefix: shared system prompt for SemanticConnectionWorker
# Suffix: per-pair specific prompt

PROMPT = """
You are a semantic connection verifier. Given two findings, classify their relationship.

Finding A: "{fact_a}" (angle: {angle_a}, confidence: {conf_a})
Finding B: "{fact_b}" (angle: {angle_b}, confidence: {conf_b})

Classify EXACTLY ONE connection type from:
causal_link, contradictory, supporting, analogous, generalizing, specializing,
methodological, temporal, statistical, compositional, independent

Format:
CONNECTION_TYPE: [type]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [1 sentence]
"""
```

The `SemanticConnectionWorker` batches up to 32 verification prompts per vLLM prefix cache fill, maximizing throughput. Verified connections are stored in `semantic_connections` with `detection_stage = 'llm_verified'`.

### 6.3 Storage and Integration

Verified connections feed directly into query selection:
- `causal_link` → trigger CAUSAL_TRACE on both endpoints
- `contradictory` → trigger ADJUDICATE or RESOLVE_CONTRADICTION
- `supporting` → trigger REPLICATION or EVIDENCE_MAP
- `compositional` → trigger SYNTHESIS_DEEPEN

Connections are also surfaced to the operator via the `OperatorQueryEngine` template `SEMANTIC_GRAPH`:
```sql
SELECT sc.connection_type, sc.confidence, c1.fact as source, c2.fact as target
FROM semantic_connections sc
JOIN conditions c1 ON sc.source_condition_id = c1.id
JOIN conditions c2 ON sc.target_condition_id = c2.id
WHERE sc.detection_stage = 'llm_verified'
ORDER BY sc.confidence DESC;
```

---

## 7. Curation Architecture

### 7.1 The Eight Curators

| Curator | Consumer | Mode | Output |
|---------|----------|------|--------|
| **GlobalHealthCurator** | OrchestratorActor, RuleEngine | Continuous | `GlobalHealthSnapshot` (row counts, convergence trend, blockers, phase recommendation) |
| **AngleContextCurator** | SwarmSupervisor, BeeActors | On-demand + cached | `AngleContextBundle` (established, controversial, bridge-worthy, recent tiers) |
| **CloneContextCurator** | FlockSupervisor, CloneActors | On-demand + cached | `CloneContextCache` (prefix-cache-optimized context text) |
| **ContradictionCurator** | OrchestratorActor, RuleEngine, Operator | Continuous | `ContradictionDigest` (prioritized contradiction queue, stale detection) |
| **GapCurator** | OrchestratorActor, McpResearcherActor | Continuous | `GapDigest` (unified MCP research priority queue) |
| **LessonCurator** | All supervisors, Operator | Event-driven | `LessonDigest` (validated findings persisted as `lesson` rows) |
| **SourceQualityCurator** | RuleEngine, Flock | Periodic (every 5 rounds) | `SourceQualityReport` (activates dead `trust_score` column) |
| **NarrativeCurator** | Operator, UserProxyActor | Continuous | `OperatorBriefing` (human-readable progress narrative) |

### 7.2 Who Needs What

**OrchestratorActor** needs convergence signals and phase-transition triggers. The `GlobalHealthSnapshot` delivers a single message containing row counts, convergence trend, top-3 blockers, and recommended next phase — eliminating the need for 10+ ad-hoc SQL queries.

**SwarmSupervisor** needs relevant prior findings, not a brute-force dump. The `AngleContextBundle` delivers 100 highly relevant findings in four tiers: established (high confidence × information gain), controversial (contradiction_flag or high fabrication_risk), bridge-worthy (cross-angle with serendipity floor), and recent (last N rounds).

**FlockSupervisor** needs prefix-cache stability. The `CloneContextCurator` maintains a cache per angle with fine-grained invalidation: only rebuild when angle-specific rows change by >0.1 in score or new findings are admitted. This drops SQL queries from 640 per Flock session (8 clones × 4 queries × 20 rounds) to ~8 (one per angle) unless significant store changes occur.

**Rule Engine** needs predicate-ready facts. The `GlobalHealthSnapshot` and `ContradictionDigest` provide structured inputs for rules like "IF contradiction_backlog > 20 AND external_data_fetched_last_round = 0 THEN propose MCP research."

**Operator** needs decision-relevant language, not raw logs. The `NarrativeCurator` assembles: "Round 5: 47 findings evaluated. 3 contradictions resolved. 1 novel bridge found (dental health × trenbolone). Recommended research: PubMed search for insulin+mTOR mechanism."

### 7.3 Context Window Overflow Prevention

Curation is the primary defense against context window exhaustion:
- **Swarm bees:** `AngleContextBundle` caps at 100 items (configurable), with composite scoring ensuring only the most information-dense findings are included.
- **Flock clones:** `CloneContextCurator` caps at 40 items in 4 tiers, with token budget enforcement. If a tier exceeds its token allocation, items are truncated by `tanh(information_gain / 2)` ranking.
- **Operator:** `OperatorBriefing` is limited to 3 sentences of narrative + 3 alerts + 2 decisions_required. No raw row dumps.

---

## 8. Reflexion Loop Across VMs

### 8.1 Detect

The `ReflexionActor` analyzes post-phase metrics and stores typed lessons:

| Trigger | Lesson Type | Example Fact |
|---------|-------------|--------------|
| `convergence_score < threshold` for 3 rounds | `query_efficiency_lesson` | "Queries stopped moving scores; consider expanding MCP research" |
| `wasted_bridge_queries > 10%` | `strategy_lesson` | "Cross-angle bridging is noisy for this domain" |
| `gossip_info_gain` monotonically decreasing | `angle_quality_lesson` | "Worker angles converging on same claims; misassignment ratio too low" |
| MCP findings idle due to score_version lag | `model_behavior_lesson` | "Bootstrap score_version at round boundary, not session boundary" |
| Per-run cost vs. findings produced | `cost_lesson` | "Run 2 spent $4.20 on BRIDGE waste. Consider disabling BRIDGE." |
| VERIFY repeatedly downgrades a source | `source_quality_lesson` | "Reddit forum posts have fabrication_risk > 0.7 in 80% of VALIDATE queries" |

### 8.2 Store

Lessons are written to the `lessons` table via `LessonStore.record()`. The table is append-only, typed (`lesson_type`), and ranked by `confidence × relevance_score`.

### 8.3 VM Portability

The VM may be torn down at any moment. Lessons must survive.

**Within a run:** WAL-enabled DuckDB plus a 60-second `StoreSyncActor` delta export to local staging.

**At run end:** Upload full DuckDB as a base snapshot to B2, then truncate delta queue.

**On VM startup:** Download latest base snapshot. Stream in any newer deltas.

```python
class StoreSyncActor:
    async def _run(self):
        while True:
            await asyncio.sleep(60)
            await self._export_delta_to_parquet()
            await self._upload_to_b2()
```

### 8.4 Apply

| Injection Point | Mechanism | Example |
|-----------------|-----------|---------|
| Clone context prefix | Prepend top-N lessons to vLLM prefix before caching | "LESSON: BRIDGE on dental angles is low-yield; skip unless relevance > 0.8" |
| Scheduler rule engine | Adjust `FlockQueryManagerConfig` between rounds | If `cost_lesson` says VALIDATE is expensive and low-magnitude, reduce its budget share |
| MCP researcher target selection | Filter `research_target` rows | If `source_quality_lesson` flags `forum_post` as unreliable for medical claims, downgrade them |
| Human dashboard | FastAPI `/lessons` endpoint | Operator reviews lessons before launching Run 3 |

### 8.5 Operator `/lessons` Endpoint

```python
@app.get("/lessons")
async def get_lessons(
    lesson_type: str | None = None,
    angle: str | None = None,
    since: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 50,
):
    ...
```

Pre-flight checks:
1. `GET /lessons?lesson_type=cost_lesson&since=2026-04-20T00:00:00Z` → "Run 4 cost $18.50, 60% on BRIDGE waste."
2. `GET /lessons?lesson_type=angle_quality_lesson&angle=insulin_timing` → "insulin_timing consistently yields high information_gain. Prioritise it."
3. `GET /lessons?lesson_type=source_quality_lesson&min_confidence=0.8` → "Source reddit.com/r/bb produced 12 fabricated claims. Blacklist recommended."

---

## 9. Operator Query Interface

### 9.1 Three-Tier Design

**Tier 1: Structured Templates (default, <50ms)**

The operator types natural language. A lightweight rule-based classifier maps to pre-validated SQL templates. No LLM call required.

```sql
-- Template: CONTRADICTION_INVENTORY
SELECT c1.id, c1.fact, c2.fact, c1.confidence, c2.confidence, c1.angle, c2.angle
FROM conditions c1
JOIN conditions c2 ON c1.contradiction_partner = c2.id
WHERE c1.contradiction_flag = TRUE
  AND c1.run_number BETWEEN :run_start AND :run_end
ORDER BY ABS(c1.confidence - c2.confidence) DESC;
```

**Tier 2: Guided Natural Language (2–5s)**

For questions outside the template library, a lightweight query planner LLM decomposes the question into a multi-step plan of Tier 1 templates, executes them, and synthesizes a natural language response. The LLM never generates raw SQL — only template names and parameters.

**Tier 3: Freeform SQL / Maestro (variable latency)**

Prefixed with `/sql` or `/maestro`. Passed directly to DuckDB with destructive-action confirmation gates. Forbidden tokens: `DELETE`, `DROP`, `TRUNCATE`, `ALTER TABLE DROP`.

### 9.2 Cross-Run Querying

The `runs` table and `run_number` column on `conditions` enable temporal analysis:

```sql
-- Cross-run contradiction tracking
SELECT r.run_number, r.started_at, COUNT(*) as contradiction_count
FROM conditions c
JOIN runs r ON c.run_number = r.run_number
WHERE c.contradiction_flag = TRUE
  AND c.fact ILIKE '%insulin timing%'
  AND r.run_number BETWEEN (SELECT MAX(run_number) FROM runs) - 4
                       AND (SELECT MAX(run_number) FROM runs)
GROUP BY r.run_number, r.started_at
ORDER BY r.run_number;

-- Finding persistence across runs
SELECT fact, angle,
       COUNT(DISTINCT run_number) as runs_seen,
       AVG(confidence) as avg_confidence,
       MAX(confidence) - MIN(confidence) as confidence_volatility
FROM conditions
WHERE row_type = 'finding' AND consider_for_use = TRUE
  AND fact ILIKE '%trenbolone%'
GROUP BY fact, angle
HAVING COUNT(DISTINCT run_number) >= 2
ORDER BY confidence_volatility DESC;
```

### 9.3 Latency Design

| Action | Latency | Mechanism |
|--------|---------|-----------|
| Soft pause | 5–30s | Stop launching new workers, let in-flight finish, checkpoint state |
| Hard stop | <1s | Cancel everything, risk partial state |
| Injection (new angle) | 1 gossip round | Queue until current round completes |
| Tier 1 query | 10–100ms | Template + DuckDB execution |
| Tier 2 query | 2–5s | LLM planner + template chain |

---

## 10. Implementation Roadmap (6 Weeks)

| Week | Theme | Deliverables |
|------|-------|--------------|
| **Week 1** | Schema + P0 Blockers | Add `run_number`, `run_id`, diffusion columns to `conditions`; create `runs`, `score_history`, `lessons`, `lesson_applications` tables; add 20+ indices including `idx_findings_covering`; fix `GossipSwarm` to write `finding` rows with gradient flags; populate `cluster_id` and `contradiction_flag`; make diffusion queen persist intermediate artifacts |
| **Week 2** | Actor Scaffold + Curation MVP | Actor base class + supervision tree; `SwarmSupervisor`, `FlockSupervisor`, `McpResearcherActor`, `UserProxyActor`; `CurationSupervisor` with `GlobalHealthCurator`, `CloneContextCurator`, `NarrativeCurator`; scheduler daemon with priority queue and health monitor |
| **Week 3** | Query Wealth Expansion + Semantic Connections | Implement CAUSAL_TRACE, ASSUMPTION_EXCAVATE, TEMPORAL, REPLICATION (P0 depth types); create `semantic_connections` table; implement `SemanticConnectionWorker` with 3-stage pipeline (heuristic → embedding → LLM verification via Python Flock); wire verified connections into BRIDGE/SYNTHESIZE/AGGREGATE triggers |
| **Week 4** | External Data Benefit Assessment + Operator Override | `external_benefit.py` with 13 signals, cost estimation, UCB-greedy selector; `source_quality_registry` + `condition_sources` tables; green/yellow/red operator override tiers; three-phase feedback loop (immediate, latent, ROI); rule engine rules ER-1 (convergence-stuck), ER-2 (cost-cap), ER-3 (operator override) |
| **Week 5** | Reflexion Persistence + VM Sync | `ReflexionActor` + `LessonStore`; typed lesson detection from Flock/Swarm metrics; `StoreSyncActor` with 60s delta export to Parquet + B2 base snapshot; operator `/lessons` endpoint; lesson injection into clone context prefixes and scheduler rules |
| **Week 6** | Massive Sources Integration + Operator Query Interface | `SourceIngestionActor` with Anna's Archive pipeline; `source_fingerprints`, `chunks`, `source_utility_log` tables; hybrid BM25+vector retriever (DuckDB VSS); `OperatorQueryEngine` with 10 core Tier 1 templates, Tier 2 query planner, Tier 3 Maestro gate; cross-run query templates; end-to-end test: swarm→flock→external→semantic connection→reflexion→operator query |

---

## 11. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Swarm still writes `thought` not `finding` rows** | Medium | **Critical** | Week 1 deliverable: enforce `admit()` with `row_type='finding'` in `GossipSwarm.synthesize()`. Add integration test that asserts `COUNT(*) FROM conditions WHERE row_type='finding'` > 0 after swarm run. |
| **Ghost columns remain unpopulated** | Medium | **Critical** | Week 1 deliverable: port union-find clustering and contradiction detection from ADK algorithm battery. Add nightly assertion that `cluster_id >= 0` for >50% of findings. |
| **vLLM prefix cache broken by actor refactor** | Low | **Critical** | FlockQueryManager is untouched. Only the caller (`FlockSupervisor`) changes. `CloneContextCurator` preserves prefix stability via fine-grained invalidation. |
| **External research ROI is negative** | Medium | High | Three-phase feedback loop detects waste within 3 rounds. `adjust_budget_for_roi()` scales budget down to 0.2× if `roi_per_usd < 0.1`. Operator override allows manual pause. |
| **Curation becomes a bottleneck** | Medium | High | Async execution + caching + incremental updates. Measure latency per curator. Backpressure: drop non-critical `StoreChangeEvent` messages if queue > 1000. |
| **DuckDB file corruption during VM sync** | Medium | **Critical** | `CHECKPOINT` before upload. Keep last 3 base snapshots in B2 with versioning. Incremental Parquet deltas mean base snapshot + deltas can reconstruct state. |
| **Semantic connection pipeline too slow** | Medium | High | Stage 1 heuristic filter reduces candidates by >95%. Stage 2 embedding gate runs in batch on GPU. Stage 3 LLM verification uses vLLM prefix caching with 32-prompt batches. |
| **Massive source ingestion overwhelms storage** | Medium | High | Tiered storage: raw blobs to B2 cold tier after extraction. `max_chunks_per_source = 2000`. `source_utility_log` blocks useless sources after 3 strikes. |
| **Operator override rate >30%** | Medium | Medium | If auto-approval thresholds are too aggressive, the operator will constantly intervene. Monitor override rate dashboard KPI. Raise yellow tier threshold if >20%. |
| **Lesson schema explosion** | Low | Medium | Start with 6 typed lesson categories. Use `metadata` JSON for extensibility. Halflife decay (default 3 runs) prevents stale lessons from dominating. |
| **Single-lock serialization kills concurrency** | Medium | High | Extend `ConditionStore` to support multiple read connections. DuckDB handles read concurrency natively; the Python `RLock` is the bottleneck. Separate read and write locks. |
| **Store schema migration breaks templates** | Medium | Medium | Version templates with schema. Run validation suite against test database on every migration. |

---

## 12. Summary

This document converges seven parallel depth analyses into a single implementable architecture. The core principle: **DEPTH is manufactured, not discovered.** It is manufactured by (1) expanding the query battery from 9 to 28 types so no dimension of a finding goes unexamined, (2) weaving findings into a verified semantic graph rather than coarse clusters, (3) investing external data budget only where benefit exceeds cost and learning from every investment, (4) curating raw rows into actionable consumables for each consumer, (5) letting the operator steer with a three-tier query interface, (6) persisting lessons across VM lifetimes so the system compounds intelligence, and (7) ingesting massive sources without drowning the pipeline in noise.

Build the seven pillars in six weeks. Start with schema and P0 blockers. End with a system that does not merely validate claims — it traces causal chains, excavates assumptions, evaluates evidentiary structure, tests counterfactuals, narrates findings for different audiences, and learns from its own mistakes.

---

*End of converged architecture.*
