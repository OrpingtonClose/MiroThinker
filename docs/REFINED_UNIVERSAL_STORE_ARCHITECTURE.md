# Refined Universal Store Architecture

**Date:** 2026-04-26  
**Status:** Refined vision — replaces destructive swarm with additive extraction

---

## 1. The Universal Inbox Principle

> **Every byte that enters the system lands in the ConditionStore first, unchanged, with full lineage.**

No preprocessing. No truncation. No summarization. No filtering. The store is the system's memory. Everything else — swarm, Flock, synthesis — reads from it and writes back into it.

### 1.1 What Goes In (Unchanged)

| Source | What lands in Store | Row type | Lineage |
|--------|---------------------|----------|---------|
| **User attachment** (PDF, image, code, spreadsheet) | File blob reference + extracted text | `raw` | `parent_id = NULL` (root) |
| **External API call** (search, forum, YouTube, scrape) | Full response JSON/text | `raw` | `parent_id = NULL` |
| **Tool output** (transcript, web page, archive download) | Full tool output | `raw` | `parent_id = NULL` |
| **Agent reasoning output** (any agent, any layer) | Full reasoning text | `thought` | `parent_id = triggering_row` |
| **Swarm extraction** (worker output, gossip, finding) | Full worker output | `finding` | `parent_id = source_raw_row` |
| **Flock verdict** (VALIDATE, BRIDGE, CHALLENGE, etc.) | Full verdict + reasoning | `thought` | `parent_id = evaluated_finding` |
| **Swarm synthesis** (queen merge, hyper-reasoning mode) | Full synthesis text | `synthesis` | `parent_id = converged_findings` |
| **Flock query result** (hyperswarm simulation output) | Full query response | `thought` | `parent_id = query_context` |

**Nothing is ever dropped.** Every row is append-only. If something is garbage, it is flagged (`consider_for_use = FALSE`, `obsolete_reason = "garbage"`) — but the text remains.

---

## 2. The Swarm as Extraction Engine (Non-Destructive)

### 2.1 What the Swarm Does

The swarm's sole job: **read raw rows from the store and extract the maximum amount of processable information from them.**

It does NOT:
- Summarize (destruction)
- Compress (destruction)
- Filter out "irrelevant" material (destruction)
- Replace raw text with its own output (destruction)

It DOES:
- **Atomize**: Break large texts into atomic findings with confidence scores
- **Cross-reference**: Link findings to their raw sources (lineage)
- **Tag**: Apply gradient dimensions (confidence, novelty, specificity, fabrication_risk)
- **Surface**: Identify serendipitous connections, novel claims, contradictions
- **Preserve raw**: The original text stays in the store untouched

### 2.2 The Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL INBOX (ConditionStore)                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐  │
│  │ raw:pdf │ │ raw:api │ │ raw:tool│ │thought: │ │ synthesis:  │  │
│  │         │ │         │ │         │ │ agent   │ │ prev_report │  │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘  │
│       │           │           │           │              │         │
│       └───────────┴───────────┴───────────┴──────────────┘         │
│                              │                                       │
│                              ▼                                       │
│                    ┌─────────────────┐                               │
│                    │ SWARM EXTRACTION │                               │
│                    │  (bees at work)  │                               │
│                    └────────┬────────┘                               │
│                             │                                        │
│         ┌───────────────────┼───────────────────┐                    │
│         ▼                   ▼                   ▼                    │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐        │
│  │ finding:    │   │ finding:    │   │ finding:            │        │
│  │ atom_1      │   │ atom_2      │   │ serendipity_bridge  │        │
│  │ (confidence)│   │ (novelty)   │   │ (cross-domain)      │        │
│  └─────────────┘   └─────────────┘   └─────────────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Extraction vs. Destruction

| Operation | Old (destructive) | New (additive) |
|-----------|-------------------|----------------|
| PDF ingestion | Extract text, discard PDF | Store PDF blob + extracted text; both kept |
| Worker output | `parse_findings()` regex extracts 884 bullets from 2.1M chars, **throws rest away** | Full 2.1M char worker output stored as `finding` row; atomization creates *additional* `atom` rows, never replaces |
| Gossip round | Workers exchange 2K compressed summaries | Workers exchange *full* outputs; new `thought` rows created for each exchange |
| Queen merge | Produces one report, discards intermediate reasoning | Report is a `synthesis` row; every intermediate worker output remains in store |
| Flock verdict | Structured VERDICT/CONFIDENCE only | Full reasoning text stored; structured fields are *views* on the text |

**The store grows monotonically.** It never shrinks. Every extraction adds new rows; it never deletes or overwrites old ones.

---

## 3. The Flock as Reasoning Layer

### 3.1 Flock Operates on Swarm Output

The Flock does not touch raw data. It reasons over the **extracted findings** the swarm produced:

```
ConditionStore
  ├── raw rows (untouched by Flock)
  ├── finding rows (swarm extraction) ──► FLOCK INPUT
  ├── thought rows (agent/Flock reasoning) ──► FLOCK INPUT
  └── synthesis rows (swarm hyper-reasoning) ──► FLOCK INPUT
```

Flock queries are **hyperswarm simulations** via vLLM prefix caching:
- Each clone (angle expert) loads its accumulated context into prefix cache
- Hundreds of queries fire against the same prefix: VALIDATE, ADJUDICATE, BRIDGE, VERIFY, ENRICH, GROUND
- Every query response becomes a new `thought` row in the store

### 3.2 Flock Output Is External Data (To the Swarm)

```
Flock verdict ──► stored as `thought` row ──► swarm treats it as new raw input ──► swarm extracts from it
```

The Flock's reasoning is **raw material** for the next swarm cycle. The swarm atomizes Flock thoughts just like it atomizes forum posts.

---

## 4. Swarm Synthesis Mode = Diffusion Hyper-Reasoning

### 4.1 The Diffusion Pattern

The synthesis follows a **diffusion-like pattern**: the report starts as noise (raw worker outputs), then iteratively denoises through confrontation and correction passes until it converges to a coherent whole.

**Diffusion steps (as implemented in `swarm/queen.py:diffusion_queen_merge`):**

| Step | Action | Parallelism | Store artifact |
|------|--------|-------------|----------------|
| **0. Scaffold** | Generate structural outline from all worker outputs | 1 call | `thought:scaffold` row |
| **1. Manifest** | Each worker writes their section from their angle | N calls (one per angle) | `synthesis:section` rows per angle per pass |
| **2. Confront** | Reviewers from OTHER angles critique each section | N×K calls | `thought:critique` rows |
| **3. Correct** | Section writers revise based on confrontation feedback | N calls | `synthesis:section` rows (new versions) |
| **4. Converge?** | Compare corrected sections to previous pass | Compute only | No new rows (metadata update on existing rows) |
| **5. Stitch** | Final editorial assembly of converged sections | 1 call | `synthesis:final_report` row |

**Each pass produces new rows.** The store accumulates:
- Pass 1 sections, critiques, corrections
- Pass 2 sections, critiques, corrections
- Pass 3 sections, critiques, corrections
- ... until convergence or max_passes reached

**Why this matters:** Every intermediate "noisy" version is preserved. You can trace how a section evolved from its first draft through each confrontation to its final form. The diffusion is fully auditable.

### 4.2 Diffusion as Additive, Not Destructive

The old `queen_merge()` was single-shot: one call, one report, intermediate reasoning lost. The diffusion queen is iterative and every step lands in the store:

```
Old (destructive):
  worker_summaries ──► queen_merge() ──► final_report (1 row)
  Everything in between = lost

New (diffusion, additive):
  worker_summaries ──► scaffold ──► manifest ──► confront ──► correct ──► [converged?] ──► stitch ──► final_report
                         │           │            │            │              │              │
                    thought    synthesis    thought    synthesis       metadata      synthesis
                     row        rows         rows       rows          update          row
```

If convergence fails, you have all intermediate drafts to inspect. If a section is controversial, you have all critiques that shaped it. If the final report is wrong, you can trace exactly which confrontation was ignored.

### 4.3 Treated as External Data

The output of swarm synthesis is **not terminal**. Even the final `synthesis:final_report` row is fed back into the swarm as raw material:

```
Swarm synthesis mode
  → produces diffusion-refined report (synthesis row)
  → swarm ingests report text in next cycle
  → swarm extracts new findings from its own synthesis
  → Flock evaluates those findings
  → loop
```

This enables **recursive deepening**: a synthesis about insulin protocols is dissected to reveal gaps, contradictions, and new angles that weren't visible during the first extraction.

---

## 5. Smart External Data Fetchers

### 5.1 The Fetcher Layer

External tools are **data fetchers**, not thinkers. They are smartly wired with multiple options:

| Fetcher | Options | When to use |
|---------|---------|-------------|
| **Web search** | DuckDuckGo, Mojeek, Exa, Brave, Kagi | Uncensored-first, fall back to censored |
| **Forum mining** | Reddit, 4plebs, Warosu, Telegram, VK | Origin-country communities in native languages |
| **YouTube** | Transcript API, Apify scraper, bulk download | Channel-level harvest, comment mining |
| **Scientific** | PubMed, arXiv, OpenAlex, Crossref, Sci-Hub | Peer-reviewed sources |
| **Archives** | Anna's Archive, Internet Archive, LibGen | Books, documents, long-form material |
| **MCP tools** | 35+ servers (Firecrawl, Exa, Jina, etc.) | Structured extraction, deep crawls |

### 5.2 Fetcher → Store Path

```
Fetcher (any tool)
  → Full response lands in Store as `raw` row
  → Swarm extracts findings
  → Flock evaluates
  → Gaps detected → new fetcher queries triggered
  → loop
```

Fetchers are triggered by:
- User request ("download all videos from this channel")
- Swarm gap detection ("I need bloodwork data to resolve this contradiction")
- Flock flag ("this finding needs independent verification")
- Scheduled polling (forum monitors, alert-driven search)

---

## 6. Context Accumulation for Flock

### 6.1 The Swarm's End State

At the end of a swarm run (or continuously, in living mode), the swarm has accumulated:
- All worker reasoning traces
- All gossip exchanges
- All extracted findings with gradient flags
- All serendipity bridges
- All contradiction detections

**This accumulated context is the swarm's end state.** It is stored in the ConditionStore as a rich tapestry of `finding`, `thought`, `insight`, and `synthesis` rows.

### 6.2 End State → Flock Queries

The Flock uses the swarm's end state as **query material**:

```sql
-- Example: Flock query using swarm end state
SELECT fact, confidence, novelty, angle
FROM conditions
WHERE row_type IN ('finding', 'thought', 'insight')
  AND consider_for_use = TRUE
  AND angle = 'insulin_timing'
ORDER BY information_gain DESC;
```

The Flock's clone contexts are **built from this end state**:
- `build_clone_context_from_store()` assembles the worker's accumulated perspective
- This context is loaded into vLLM prefix cache
- Flock queries fire against it: "Is this finding consistent with the insulin timing angle?"

### 6.3 Cross-Run Federation

The end state persists across runs:
- Run 1: Swarm extracts findings about insulin → Flock validates → Store accumulates
- Run 2: Swarm starts with Run 1's end state already in store → extracts deeper findings → Flock validates with more context
- Run N: The store is a compound intelligence. Every run is smarter than the last.

---

## 7. Information Preservation Guarantee

### 7.1 The Anti-Destruction Contract

Every component in the system obeys this contract:

```
CONTRACT:
  1. Thou shalt not delete rows.
  2. Thou shalt not overwrite rows.
  3. Thou shalt not truncate text.
  4. Thou shalt not discard "irrelevant" material.
  5. New output SHALL be INSERTed as new rows with parent_id.
  6. "Irrelevant" material SHALL be flagged, not deleted.
  7. The original raw SHALL always be retrievable.
```

### 7.2 Row Types as Audit Trail

```sql
-- Every row in the system
create table conditions (
    id              integer primary key,
    row_type        text not null,  -- 'raw' | 'finding' | 'thought' | 'insight'
                                    -- | 'synthesis' | 'contradiction' | 'atom'
                                    -- | 'similarity' | 'verdict' | 'tool_output'
    fact            text not null,   -- the FULL text, never truncated
    parent_id       integer,         -- lineage: who produced this?
    source_url      text,            -- where did it come from?
    source_type     text,            -- 'forum' | 'youtube' | 'attachment' | 'api'
                                    -- | 'swarm_worker' | 'flock_verdict' | 'synthesis'
    angle           text,            -- which angle extracted this?
    confidence      float,           -- swarm-scored or flock-scored
    novelty         float,           -- information gain
    specificity     float,           -- how concrete
    fabrication_risk float,          -- hallucination risk
    consider_for_use boolean,        -- FALSE = flagged, not deleted
    obsolete_reason text,            -- why flagged
    metadata        json,            -- full structured data, not a summary
    created_at      timestamp
);
```

### 7.3 The Lineage DAG

Every final report can trace back to raw source:

```
final_report (synthesis:stitch)
  ├── pass_3_corrected (synthesis:section)
  │   ├── pass_3_critiques (thought:critique)
  │   │   └── reviewer_C_analysis (finding)
  │   │       └── raw_forum_post_1 (raw)
  │   └── pass_2_corrected (synthesis:section)
  │       ├── pass_2_critiques (thought:critique)
  │       │   └── reviewer_B_analysis (finding)
  │       │       └── raw_youtube_transcript (raw)
  │       └── pass_1_manifest (synthesis:section)
  │           ├── worker_A_output (finding)
  │           │   ├── raw_forum_post_1 (raw)
  │           │   └── raw_youtube_transcript (raw)
  │           └── scaffold (thought:scaffold)
  │               └── all_worker_outputs (finding)
  └── flock_verdicts (thought)
      ├── flock_VALIDATE_insulin (thought)
      │   └── finding_from_worker_A (finding)
      └── flock_BRIDGE_hematology (thought)
          └── finding_from_worker_B (finding)
```

**The diffusion leaves a full trail.** Every confrontation, every correction, every pass is a node in the DAG. A section that changed dramatically between Pass 1 and Pass 3 reveals a domain boundary that was initially misunderstood.

---

## 8. The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL WORLD                                      │
│  User │ Attachments │ APIs │ Forums │ YouTube │ Papers │ Archives │ Tools  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        UNIVERSAL INBOX (ConditionStore)                      │
│  All data lands here UNCHANGED. Full text preserved. Lineage tracked.        │
│  Row types: raw, tool_output, attachment_blob, agent_thought                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SWARM EXTRACTION ENGINE                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  BEES (workers): Atomize, cross-reference, tag, surface connections    │  │
│  │  GOSSIP: Exchange full outputs (not summaries), identify novel bridges │  │
│  │  SERENDIPITY: Detect unexpected but relevant cross-domain connections  │  │
│  │  DIFFUSION SYNTHESIS: Scaffold→Manifest→Confront→Correct→Stitch      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                       │
│  Output: finding, thought, insight, synthesis, contradiction rows            │
│  All INSERTed into Store. Raw rows untouched.                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SWARM END STATE ACCUMULATED                               │
│  Rich tapestry of extracted findings + reasoning traces + contradictions     │
│  This IS the accumulated context. It persists across runs.                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FLOCK HYPERSWARM                                     │
│  vLLM prefix caching loads clone contexts from swarm end state               │
│  Query battery: VALIDATE, ADJUDICATE, VERIFY, ENRICH, GROUND, BRIDGE         │
│  Each query = simulated expert agent evaluation                              │
│  Output: verdict rows (INSERTed into Store as `thought`)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLOCK OUTPUT → SWARM INPUT                                │
│  Flock verdicts are treated as external data                                 │
│  Swarm extracts new findings from Flock reasoning                            │
│  Recursive loop: swarm → flock → swarm → flock                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIFFUSION SYNTHESIS                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  PASS N: Manifest → Confront → Correct → Converge?                    │  │
│  │  Each pass: new section/critique rows INSERTed into Store             │  │
│  │  All intermediate drafts preserved (not overwritten)                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│                         FINAL STITCH → synthesis row                         │
│                         Fed back into swarm for recursive deepening          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Why This Fixes MiroThinker

### 9.1 The Original Disease

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `parse_findings()` throws away 99.95% of worker output | Destructive extraction | **Additive extraction**: full output stored, atoms are *additional* rows |
| Gossip compression kills diversity (92% → 6% info gain) | Workers exchange summaries | **Full-corpus gossip**: workers exchange complete outputs, stored as rows |
| Queen merge crashes silently | Single-shot, no intermediate recovery | **Diffusion queen**: scaffold→manifest→confront→correct→stitch, all rows stored |
| Flock queries fire against primitive prompts | No accumulated context | **Flock loads swarm end state** from store into prefix cache |
| Can't re-run swarm without re-running research | Research and swarm coupled | **Separate endpoints**: store is the coupling layer |
| Swarm bypassed with throwaway scripts | Friction too high | **Store as universal inbox**: everything flows through it naturally |

### 9.2 The Compound Intelligence Effect

Because the store grows monotonically:
- **Run 1:** 500 findings, 10 contradictions, 3 bridges
- **Run 2:** Starts with Run 1's store → extracts deeper findings → 800 findings, 25 contradictions, 12 bridges
- **Run N:** The store is a knowledge graph. The swarm is smarter because it has more material. The Flock is smarter because clones have richer contexts.

> *"The tenth session on a repository is meaningfully smarter than the first — and that intelligence is portable."* — Rookery DATALAKE.md

---

---

## 10. Brutal Critique — What's Broken Right Now

> *"The architecture is a beautiful vision with a brittle implementation."*

The following critique is grounded in a 4-subagent forensic audit of the actual codebase (`engine.py`, `queen.py`, `flock_query_manager.py`, `corpus.py`). It is not theoretical. These are the blockers that prevent the vision from working today.

---

### 10.1 P0 Blockers (Showstoppers)

#### A. Swarm → Store Is Broken

`GossipSwarm.synthesize()` in `engine.py` **never writes worker outputs as `finding` rows**. It only emits `LineageEntry` objects via `_emit()` → `emit()`, which store them as `thought`/`synthesis` rows with **no gradient flags** (`confidence`, `novelty_score`, `fabrication_risk`, etc.).

The `MCPSwarmEngine` in `mcp_engine.py` DOES write findings via agent tools (`store.admit(..., row_type="finding", confidence=...)`), but it is a **separate, incompatible class** that nothing uses in production. `run_swarm_test.py`, all integration tests, and all proxy adapters use `GossipSwarm`.

**Result:** The swarm runs entirely in memory. The store is bypassed. The Flock has nothing to evaluate.

#### B. `cluster_id` and `contradiction_flag` Are Ghost Columns

In `strands-agent/corpus.py` (the active store):
- `cluster_id` defaults to `-1` and is **never populated**. BRIDGE and SYNTHESIZE queries return zero rows.
- `contradiction_flag` defaults to `FALSE` and is **never populated**. ADJUDICATE queries return zero rows.

The old `adk-agent/models/corpus_store.py` had `detect_contradictions()` and clustering logic. The new `ConditionStore` has **zero clustering implementation**. Three of the nine Flock query types are dead code.

#### C. Diffusion Queen Is Store-Oblivious

`diffusion_queen_merge()` returns `(report, llm_calls)`. The intermediate scaffold, manifest sections, critiques, and corrected sections live only in Python heap memory. They are garbage-collected after the function returns. This directly violates the architecture's append-only guarantee for diffusion artifacts.

#### D. Zero User-Defined Indices

The `conditions` table has **no indices** beyond `PRIMARY KEY`. Every Flock query is a full table scan. At 1M rows, `select_queries()` would take 50-200ms per branch × 8 branches = 1-2 seconds just to select queries. At 10M rows, the system becomes unusable.

#### E. Single-Lock Serialization

```python
self._lock = threading.RLock()
```

Every read, write, lineage traversal, and health snapshot acquires this lock. Under async load with hundreds of concurrent Flock queries, the system spends more time waiting on `self._lock.acquire()` than executing SQL. DuckDB supports multiple read connections, but the `ConditionStore` multiplexes everything through a single connection behind a single Python lock.

---

### 10.2 P1 Blockers (Severe Friction)

#### F. No Scheduler, No Living Swarm

There is no `while True`, no cron, no file watcher, no store polling loop, no webhook endpoint. The "24-hour continuous operation" mentioned in docstrings is aspirational. Every run is a single-shot function call:

```python
result = await swarm.synthesize(corpus=corpus, query=query)
```

The recursive loop `swarm → flock → swarm → flock` is a diagram, not code.

#### G. Two Incompatible Swarm Engines

| Feature | `GossipSwarm` (`engine.py`) | `MCPSwarmEngine` (`mcp_engine.py`) |
|---------|----------------------------|-----------------------------------|
| Writes findings to store | **No** | Yes |
| Gossip rounds | Yes | No |
| Diffusion queen | Yes | No |
| Flock integration | No | Yes |
| Used in production | **Yes** | No |

You cannot get gossip/diffusion AND store-aware findings in the same run. This is why Devin and OpenHands bypassed the pipeline — the pieces don't fit together.

#### H. No One-Liner Entry Point

To run a full pipeline, you need:
1. Load corpus text
2. Initialize `ConditionStore`
3. Ingest corpus
4. Configure 3+ completion functions (worker, queen, serendipity)
5. Configure swarm config
6. Call `swarm.synthesize()`
7. Manually save results
8. Optionally run Flock (but only if you used `MCPSwarmEngine`)

There is no:
```python
report = await orchestrator.run(query="...")
```

---

### 10.3 P2 Blockers (Architectural Debt)

#### I. Append-Only vs. Functional Deletion

Rows with `consider_for_use = FALSE` are invisible to every consumer (`export_for_swarm`, `build_report`, `get_findings`, Flock queries). They occupy disk space and contribute to scan cost, but no query ever returns them. This is **soft deletion with extra storage cost** — not true preservation. The "Anti-Destruction Contract" is structurally incompatible with GDPR Article 17 (right to erasure).

#### J. Score Column Proliferation

The schema has 15+ FLOAT columns. Only 7 are actively updated by Flock. Seven more (`trust_score`, `duplication_score`, `composite_quality`, `information_density`, `cross_ref_boost`, `staleness_penalty`, `relationship_score`) are dead weight in `strands-agent`. Two (`scored_at`, `mcp_research_status`) are never written at all.

#### K. `parent_id` vs. `parent_ids` Duality

The schema carries two lineage mechanisms that do not interoperate:
- `parent_id`: Single integer FK, used by `admit()`, `ingest_raw()`
- `parent_ids`: JSON text array, used by `emit()`, `store_evaluation()`

`get_lineage_chain()` tries to bridge both. A row created by `admit()` with `parent_id=5` will never have `parent_ids` populated, so DAG traversals miss half the graph.

---

## 11. The Flock's Analytical Wealth

> *"Please do NOT underappreciate the wealth of analytics the flock queries allows."*

The Flock Query Manager is not merely a "run some queries" scheduler. It embodies several sophisticated information-theoretic and control-theoretic principles. Here is the deep dive.

---

### 11.1 Query Type Catalog (All 9 Types)

| Type | Trigger | Question Answered | New Rows | Score Delta |
|------|---------|-------------------|----------|-------------|
| **VALIDATE** | `novelty > 0.6 AND confidence < 0.4` | "Is this novel claim real?" | `evaluation` row | `confidence` ← extracted from response |
| **ADJUDICATE** | `contradiction_flag = TRUE` | "Which side is correct?" | `evaluation` row | **Asymmetric**: winner gets `eval_conf`, loser gets `1.0 - eval_conf` |
| **VERIFY** | `fabrication_risk > 0.4` | "Is this gym folklore or real evidence?" | `evaluation` row | `fabrication_risk` set to 0.1/0.5/0.9 based on verdict keyword |
| **ENRICH** | `specificity < 0.4 AND relevance > 0.5` | "Add concrete data to make this precise" | `evaluation` + **new `finding`** (if `ENRICHED_CLAIM` extracted) | `confidence` ← extracted |
| **GROUND** | `actionability > 0.6 AND verification_status = ''` | "What evidence supports this actionable finding?" | `evaluation` row | `confidence` ← extracted (but `verification_status` never updated) |
| **BRIDGE** | `cluster_id IN (clone's clusters)` or high relevance fallback | "How does this finding interact with your domain?" | `evaluation` + **new `insight`** (if connection found) | `confidence` ← extracted |
| **CHALLENGE** | `confidence > 0.8 AND angle != clone` | "What are the strongest objections?" | `evaluation` row | If `SURVIVES_CHALLENGE: no` → `confidence -= 0.3`; `partially` → `-= 0.15`; `yes` → no change |
| **SYNTHESIZE** | Cluster with 3+ findings | "What higher-order insight emerges from combining these?" | `evaluation` + **new `synthesis`** row | `confidence` ← extracted, applied to all cluster members |
| **AGGREGATE** | Once per round, after all clones | "What are the 5 highest-value external research directions?" | Up to 5 `research_target` rows | None (read-only strategic planning) |

---

### 11.2 The Information-Theoretic Machinery

#### A. Adaptive Budget Allocation (`compute_query_budget`)

This is a form of **Thompson sampling / upper-confidence-bound (UCB) allocation**:

1. Count eligible conditions per query type (SQL COUNT)
2. Allocate budget proportionally: `raw_share = (count / total_eligible) * total_budget`
3. Ensure every type gets at least 5 queries (exploration floor)
4. If `prior_type_magnitudes` provided, boost types that produced large score changes: `boost = 1.0 + 0.5 * (mag / max_mag)`
5. Normalize to not exceed total budget

**Why this matters:** In Round 1, all types get ~60 queries each. If ADJUDICATE produces massive score changes (resolving contradictions is high-impact) while GROUND produces near-zero changes, Round 2 shifts budget toward ADJUDICATE. The system *learns* which analytical operations are productive for the current store state.

#### B. Priority Decay (`compute_priority_decay`)

Formula: `decay = 1.0 / (1.0 + 0.3 * log(evaluation_count))`

- 0-1 evaluations: full priority
- 3 evaluations: ~30% reduction
- 10 evaluations: ~50% reduction
- 100 evaluations: ~80% reduction

This is a **logarithmic diminishing returns** model. The first evaluation of a finding is high-value (new perspective). The tenth is almost certainly redundant. The system progressively deprioritizes over-evaluated conditions, preventing endless circling around favorite findings.

#### C. The Serendipity Floor

For BRIDGE, SYNTHESIZE, and CHALLENGE queries:
```python
weight = max(relevance_score * query_relevance_boost, serendipity_floor)
```

With defaults (`boost=2.0`, `floor=0.4`):
- A finding with relevance 0.9 gets weight 1.8 (full boost)
- A finding with relevance 0.1 gets weight 0.4 (floor), NOT 0.2

**This is a diversity preservation mechanism.** Strict relevance filtering would kill cross-domain insights. A dental finding about trenbolone might have relevance 0.05 to a bodybuilding query. Without the floor, its BRIDGE priority would be ~0.065 — effectively zero. With the floor, it survives. *"Dentistry × trenbolone survives."*

#### D. Asymmetric Adjudication Scoring

When ADJUDICATE returns a verdict, winner and loser get *different* score deltas:

| Verdict | Winner Delta | Loser Delta |
|---------|-------------|-------------|
| side_a | `confidence = eval_conf` | `confidence = max(0.1, 1.0 - eval_conf)` |
| side_b | `confidence = max(0.1, 1.0 - eval_conf)` | `confidence = eval_conf` |
| both_valid | `confidence = min(1.0, eval_conf * 0.8)` | `confidence = min(1.0, eval_conf * 0.8)` |
| neither | `confidence = max(0.1, 0.5 - eval_conf * 0.3)` | `confidence = max(0.1, 0.5 - eval_conf * 0.3)` |

**Why asymmetric?** The evaluator's confidence is confidence *in the verdict*, not in both claims equally. If the evaluator is 90% confident that Side A is correct, Side A should get ~0.9 and Side B ~0.1. Symmetric averaging would give both ~0.5, erasing the adjudication's informational value.

#### E. Per-Type Magnitude Tracking

During each round, `round_type_magnitudes` accumulates total score magnitude per query type. This enables:
1. **Adaptive budget allocation** (described above)
2. **Convergence detection per type**: If VALIDATE magnitude drops to near-zero while ADJUDICATE magnitude stays high, validation is exhausted but contradictions still need work
3. **Offline analysis**: Post-hoc analysis of which query types were productive in which rounds

#### F. `information_gain` Accumulation

`information_gain` is the **cumulative absolute score change** a condition has experienced across all evaluations. It is initialized to `0.0` and incremented by `individual_mag` after every evaluation.

**How it's used:**
- Tier 1 of `build_clone_context_from_store()` selects `information_gain > 0.1` findings first
- Top findings in AGGREGATE state are ordered by `information_gain DESC`
- High `information_gain` means the finding is *controversial* or *evolving* — exactly the kind of context that makes clones interesting

**Limitation:** It only increases. There is no decay or reset. A finding evaluated 100 times in early rounds will permanently dominate rankings, even if it stabilized long ago.

#### G. Relevance-Weighted Convergence Detection

Convergence is measured as: `convergence_score = round_score_magnitude / round_queries`

Before summing, magnitude is multiplied by `(0.3 + 0.7 * relevance_score)`. This means:
- Score changes on irrelevant findings count less toward convergence
- The system converges when the *relevant* part of the store stabilizes, not when tangential noise stops moving

---

### 11.3 AGGREGATE as Strategic Research Planning

Once per round, after all clone turns, the AGGREGATE query:
1. Gathers top 30 findings by `information_gain`, top 15 gaps, top 10 contradictions, top 10 convergent themes
2. Asks the model: "Given EVERYTHING, produce exactly 5 highest-value external research directions"
3. The prompt explicitly ranks priorities:
   - (1) Resolve contradictions (highest value)
   - (2) Fill gaps where 2+ angles independently identified the same need
   - (3) Provide mechanistic evidence for convergent themes
   - (4) Address unexplored angles
   - (5) Ground high-novelty findings with authoritative sources
4. Research directions are stored as `research_target` rows with a +0.2 priority boost over individual gaps

**What makes it strategic:** It doesn't search randomly. It searches *targeted* based on the swarm's collective intelligence. It links directions back to specific gap IDs and contradiction IDs, creating traceability.

---

### 11.4 Analytical Ceiling

What Flock queries **CAN** do:
- Discover contradictions the swarm missed (via BRIDGE, CHALLENGE)
- Generate genuinely novel hypotheses (via SYNTHESIZE of cross-angle clusters)
- Identify weak sources (via VERIFY)
- Produce strategically prioritized research plans (via AGGREGATE)

What Flock queries **CANNOT** do:
- Find **false negatives** (claims the swarm never extracted)
- Discover mechanisms beyond the model's training data
- Perform **temporal analysis** ("has this finding been superseded?")
- Detect **systemic bias** in the swarm's extraction angle
- Verify claims against external databases (only against the model's parametric knowledge)

---

## 12. The Missing Query Battery

Seven query types that SHOULD exist but do not:

| Type | Trigger | Question | New Rows |
|------|---------|----------|----------|
| **TEMPORAL** | Older findings with newer conflicting evidence in same cluster | "Has this finding been superseded?" | `evaluation` + `insight` (evolution explanation) |
| **SOURCE_QUALITY** | `source_type` ∈ {forum, social} AND `confidence > 0.6` | "Does source quality justify confidence?" | `evaluation`; updates `trust_score` (currently dead column) |
| **SCOPE** | `specificity > 0.7 AND relevance > 0.7` from single angle | "Does this finding generalize beyond its domain?" | `insight` with `row_type = 'scope_extension'` |
| **REPLICATION** | `confidence > 0.7` with only one `source_url` | "Has this claim been independently replicated?" | `evaluation`; updates structured source count |
| **METHODOLOGY** | High `novelty` but low `specificity` | "What method produced this? What are its limitations?" | `evaluation` with methodological critique |
| **ONTOLOGY** | Finding that contradicts cluster consensus | "Is this outlier a breakthrough or an error?" | `evaluation` + `insight` if validated |
| **REDUNDANCY** | Finding with 5+ evaluations but near-zero `information_gain_rate` | "Is this finding still worth keeping in clone contexts?" | Flags for `consider_for_use` review |

These would raise the query battery from 9 to 16 types, covering temporal evolution, source epistemics, generalizability, replication, methodological critique, outlier detection, and redundancy pruning.

---

## 13. Data Model Reality & Proposed Evolution

### 13.1 Current Schema Audit Summary

The `conditions` table has **50 columns**. Of these:
- **18 are actively read/written** in `strands-agent`
- **14 are dead weight** (never read, never written, or default-only)
- **2 are ghost columns** (`cluster_id`, `contradiction_partner` — read heavily but never populated)
- **2 are schizophrenic** (`parent_id` INTEGER vs. `parent_ids` TEXT/JSON)

Full audit available in `subagent_data_model_refinement.md`.

### 13.2 Proposed Schema Evolution

#### New Columns (Backward-Compatible)

| Column | Type | Purpose |
|--------|------|---------|
| `run_number` | INTEGER | FK to `runs` table; enables cross-run queries |
| `diffusion_pass` | INTEGER | Which diffusion pass this row belongs to |
| `diffusion_phase` | TEXT | `scaffold` \| `manifest` \| `confront` \| `correct` \| `stitch` |
| `section_angle` | TEXT | For diffusion sections: which angle wrote it |
| `convergence_status` | TEXT | `pending` \| `confirmed` \| `corrected` \| `converged` |
| `critique_target_id` | INTEGER | For critiques: which section row they address |
| `diffusion_report_id` | INTEGER | FK to final report row |
| `scores_json` | TEXT | JSON blob for extensible gradient dimensions |
| `provenance_system` | TEXT | `swarm_extraction` \| `flock_bridge` \| `flock_validate` \| `mcp_research` |

#### New Tables

```sql
-- Run tracking
CREATE TABLE runs (
    run_number INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TEXT, ended_at TEXT,
    convergence_reason TEXT,
    total_queries INTEGER, source_query TEXT
);

-- Score history (temporal awareness)
CREATE TABLE score_history (
    id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL,
    run_number INTEGER, evaluator_angle TEXT, query_type TEXT,
    old_confidence FLOAT, new_confidence FLOAT,
    old_novelty_score FLOAT, new_novelty_score FLOAT,
    old_specificity_score FLOAT, new_specificity_score FLOAT,
    old_relevance_score FLOAT, new_relevance_score FLOAT,
    old_actionability_score FLOAT, new_actionability_score FLOAT,
    old_fabrication_risk FLOAT, new_fabrication_risk FLOAT,
    magnitude FLOAT, evaluated_at TEXT
);

-- Normalized evaluators (replaces JSON evaluator_angles)
CREATE TABLE condition_evaluators (
    condition_id INTEGER NOT NULL,
    evaluator_angle TEXT NOT NULL,
    evaluation_count INTEGER DEFAULT 1,
    first_evaluated_at TEXT, last_evaluated_at TEXT,
    PRIMARY KEY (condition_id, evaluator_angle)
);

-- Diffusion DAG edges
CREATE TABLE diffusion_edges (
    id INTEGER PRIMARY KEY,
    from_condition_id INTEGER NOT NULL,
    to_condition_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL,  -- 'critiques' | 'corrects' | 'replaces' | 'stitches'
    diffusion_pass INTEGER, created_at TEXT
);

-- Section convergence trajectory
CREATE TABLE section_convergence_log (
    id INTEGER PRIMARY KEY,
    section_angle TEXT NOT NULL,
    diffusion_pass INTEGER NOT NULL,
    jaccard_vs_prior FLOAT, chars_changed INTEGER,
    critiques_received INTEGER, convergence_status TEXT,
    created_at TEXT
);
```

#### Index Strategy (20+ Indices)

The single highest-impact optimization:
```sql
-- Covering index for the most common Flock query pattern
CREATE INDEX idx_findings_covering ON conditions (
    consider_for_use, row_type, score_version,
    confidence DESC, novelty_score, fabrication_risk,
    specificity_score, relevance_score, actionability_score
) INCLUDE (id, fact, angle, information_gain, evaluation_count, cluster_id);
```

Other critical indices:
- `idx_active_findings` — universal gate for all Flock queries
- `idx_angle_findings` — clone context building
- `idx_novelty_confidence` — VALIDATE queries
- `idx_fabrication_risk` — VERIFY queries
- `idx_specificity_relevance` — ENRICH queries
- `idx_contradictions` — ADJUDICATE queries
- `idx_cluster_members` — BRIDGE/SYNTHESIZE queries
- `idx_information_gain` — Tier 1 clone context
- `idx_phase` — gap analysis
- `idx_fact_angle` — compaction dedup

Full index list and migration SQL in `subagent_data_model_refinement.md` §7.2.

### 13.3 Score Column Bloat Reduction

**Keep as physical columns** (indexed, actively queried):
`confidence`, `novelty_score`, `specificity_score`, `relevance_score`, `actionability_score`, `fabrication_risk`, `information_gain`

**Move to `scores_json`** (dead/derivable/extensible):
`trust_score`, `duplication_score`, `composite_quality`, `information_density`, `cross_ref_boost`, `staleness_penalty`, `relationship_score`

**Remove entirely**:
`scored_at`, `mcp_research_status`

This reduces the schema surface area from 15+ floats to 7 floats + 1 JSON blob, without losing query performance on the active dimensions.

---

## 14. Integration Reality — The Glue List

### 14.1 What Exists vs. What Works

| Component | Exists? | Works in Production? | Missing Glue |
|-----------|---------|---------------------|-------------|
| ConditionStore schema | **Yes** | **Yes** | Schema is complete |
| Swarm worker synthesis (`GossipSwarm`) | **Yes** | **Partial** | Does NOT write `finding` rows to store |
| Gossip rounds | **Yes** | **Partial** | Outputs stay in Python dicts |
| Diffusion queen | **Yes** | **Partial** | Intermediate rows not stored |
| Flock query manager | **Yes** | **Partial** | Only wired in `MCPSwarmEngine` |
| Flock execution loop | **Yes** | **Partial** | Requires `score_version > 0` + bootstrap |
| Store → Flock trigger | **No** | **No** | No scheduler, daemon, or event trigger |
| Flock → Swarm trigger | **No** | **No** | After Flock returns, engine exits. No loop |
| MCP research interleaving | **Yes** | **Partial** | Blocks Flock round sequentially; loosely coupled to AGGREGATE |
| Living swarm scheduler | **No** | **No** | No daemon, cron, watcher, or polling loop |
| Health monitoring | **Partial** | **No** | Metrics written to store, but no dashboard reads them |
| Context Constructor (#239) | **Partial** | **Partial** | `build_clone_context_from_store()` exists but is scattered |
| vLLM prefix cache management | **No** | **No** | Relies on implicit caching; no warm/load/evict logic |
| Cross-run federation | **Partial** | **Partial** | Store persists to disk, but no orchestrator loads prior state |
| Unified one-liner entry point | **No** | **No** | Must wire 3+ configs, 3+ completion functions manually |

### 14.2 The 10-Item Build Priority List

| Priority | Item | Why | Complexity | Blocks |
|----------|------|-----|-----------|--------|
| **1** | **Unified Orchestrator** | One `run(query)` entry point. Eliminates "which swarm class?" friction. | Medium (2-3 days) | Adoption |
| **2** | **Swarm → Store Atomization Bridge** | `GossipSwarm` must write `finding` rows with gradient flags after each phase | Medium (2-3 days) | Flock, federation |
| **3** | **Clustering + Contradiction Detection** | Populate `cluster_id` and `contradiction_flag` in `strands-agent` | Medium (2-3 days) | BRIDGE, SYNTHESIZE, ADJUDICATE |
| **4** | **Diffusion Queen → Store Persistence** | Write scaffold/section/critique rows at each diffusion step | Low-Medium (1-2 days) | Auditability |
| **5** | **Index Strategy** | Create 20+ indices including `idx_findings_covering` | Low (1 day) | Scale |
| **6** | **Score History + Normalized Evaluators** | `score_history` table + `condition_evaluators` junction table | Medium (2-3 days) | Temporal queries |
| **7** | **Living Swarm Scheduler** | `asyncio` polling loop + webhook endpoint + backoff logic | High (5-7 days) | Real-time intelligence |
| **8** | **Context Constructor (#239)** | Dedicated module with token-budget assembly + prefix SHA tracking | Medium (3-4 days) | vLLM efficiency |
| **9** | **Flock → Swarm Feedback Wiring** | Re-ingest Flock `thought` rows as `raw` input for next swarm cycle | Medium (2-3 days) | Recursive deepening |
| **10** | **Dashboard / Observability** | FastAPI endpoint querying metric rows for live health display | Low (1-2 days) | Operational trust |

### 14.3 The One-Liner Vision

What the user experience should look like:

```python
from swarm.orchestrator import UniversalOrchestrator

orch = UniversalOrchestrator(db_path="research.duckdb")

# Simplest possible
report = await orch.run(query="insulin timing for muscle growth")

# With attachments and external research
report = await orch.run(
    query="...",
    attachments=["paper.pdf", "forum_posts.json"],
    enable_flock=True,
    enable_external_research=True,
)

# Living mode — continuous monitoring
await orch.run_living(
    query="...",
    poll_interval=300,  # check for new data every 5 minutes
    webhook_port=8080,  # accept push triggers
)
```

Everything else — store initialization, corpus ingestion, completion function routing, swarm execution, Flock evaluation, MCP research, diffusion synthesis, report generation — is internal.

---

## 15. Summary: Vision + Reality = Path Forward

### The Vision (Correct)
1. **Universal inbox**: Every byte enters the store unchanged
2. **Additive extraction**: Swarm atomizes without destroying originals
3. **Diffusion synthesis**: Scaffold → manifest → confront → correct → stitch, all preserved
4. **Flock hyperswarm**: Flag-driven mass evaluation with UCB allocation, priority decay, asymmetric adjudication
5. **Recursive loop**: swarm → flock → swarm → flock, compounding intelligence
6. **Append-only guarantee**: Nothing deleted, everything auditable

### The Reality (Brutal)
1. **Swarm writes `thought` rows, not `finding` rows** — Flock has nothing to evaluate
2. **`cluster_id` and `contradiction_flag` are never populated** — 3 of 9 Flock query types are dead code
3. **Diffusion queen returns strings** — intermediate reasoning is lost
4. **Zero indices** — every query is a full table scan
5. **Single-lock serialization** — async concurrency is killed
6. **No scheduler** — the recursive loop is a diagram, not running code
7. **Two incompatible swarm engines** — you can't get gossip AND store-aware findings in one run
8. **No one-liner entry point** — friction is why everything gets bypassed

### The Path Forward

**Build the glue, not more infrastructure.**

The store schema is complete. The swarm engine is sophisticated. The Flock is clever. The MCP researcher is competent. What is missing is the connective tissue: a unified orchestrator, a scheduler, store bridges that make the existing components actually talk to each other, and indices that make the store queryable at scale.

The architecture document was never wrong about the vision. It was wrong about the distance from vision to working system. That distance is approximately **10 prioritized build items, 3-4 weeks of focused glue work, and a willingness to stop building new abstractions until the old ones are wired together.**

---

## 16. Source Citations

| Concept | Source |
|---------|--------|
| Universal inbox / all data to store unchanged | User refinement (this conversation) |
| Swarm destroys information | Devin transcript, line 7933 |
| Swarm as extraction, not destruction | User refinement (this conversation) |
| Flock as hyperswarm via prefix caching | Devin subagent `3ad3df9d`, session.json |
| Synthesis mode as external data | User refinement (this conversation) |
| End state feeds Flock queries | User refinement (this conversation) |
| Smart fetchers with multiple options | Devin transcript, PR #83, issues #84-#85 |
| Living process / continuous loop | Devin transcript, lines 6681, 9126, 9393 |
| Data-bearing agents / swarm sovereignty | `docs/oblivious_thinker.txt.rtf` |
| Lineage DAG / append-only store | `docs/STORE_ARCHITECTURE.md` |
| Cross-run federation | `rookery/docs/DATALAKE.md` |
| Flock analytical wealth (UCB, decay, asymmetric scoring) | Subagent audit: `subagent_flock_analytics.md` |
| Architecture critique (scalability, failure modes, termination) | Subagent audit: `subagent_architecture_critique.md` |
| Data model refinement (ghost columns, indices, migration) | Subagent audit: `subagent_data_model_refinement.md` |
| Integration gaps (glue list, scheduler, one-liner) | Subagent audit: `subagent_integration_gaps.md` |
| Diffusion pattern | `swarm/queen.py:diffusion_queen_merge()`, Devin transcript line 25169 |

---

*This document is both vision and reality check. The first 9 sections describe the architecture as it should work. Sections 10-16 describe what is broken today and exactly what must be built to close the gap.*


## 17. Framework Decision: The Hybrid Path

*This section synthesizes 8 parallel subagent analyses into a single actionable decision. Full analysis: `FRAMEWORK_RECOMMENDATION.md`.*

### 17.1 The Question

Which framework should we build the Universal Store Architecture on?

- **strands-agent** — ReAct-style orchestrator, rich store, swarm bridge, 35+ MCP tools. Flock removed.
- **Google ADK** — Abandoned. Had superior algorithm battery but asyncio-blocking.
- **LangChain/LangGraph** — Native cycles, checkpointing, async streaming. Static topology.
- **DuckDB FlockMTL** — SQL-native LLM functions. No prefix caching, no adaptive budget.
- **Custom from scratch** — Full control. 12–20 weeks.

### 17.2 The Answer: No Single Framework

After auditing every candidate, the consensus is unanimous: **no single framework handles all requirements.** The correct path is a hybrid that uses each framework for what it does best, and builds custom code only where no framework fits.

**The Five Principles:**

1. **Start from working code.** ~75% of the vision is already in strands-agent. Evolving it takes 3–4 weeks. Custom from scratch takes 12–20.
2. **Preserve prefix caching.** vLLM prefix caching in `flock_query_manager.py` is the #1 cost optimization. FlockMTL abandons it (167× cost increase). Python Flock stays exactly as-is.
3. **The recursive loop is not a graph problem.** Dynamic spawning, nested recursion, heterogeneous parallelism, event-driven convergence, and user interrupts are all first-class in the actor model and poor fits for LangGraph, dataflow, or pure state machines.
4. **The Devin operator must become code.** 12 operator behaviors (probe→validate→decide, GPU waste detection, cost tracking, destructive action gates) require a custom rule engine. No framework implements these.
5. **Transplant, don't revive.** ADK-agent's 13-step algorithm battery, maestro SQL conductor, dashboard, and thought swarm are superior to strands-agent's current swarm. But ADK itself is abandoned due to asyncio blocking. Port the logic, not the framework.

### 17.3 The Hybrid Stack

```
┌─ USER INTERFACE (FastAPI, SSE, dashboard) ─────────────────┐
├─ UNIFIED ENTRY POINT: orch.run(query) ─────────────────────┤
├─ POLICY LAYER: 12-operator rule engine ────────────────────┤
├─ ACTOR SUPERVISION TREE (asyncio + Queue mailboxes) ───────┤
│   OrchestratorActor                                         │
│   ├── SwarmSupervisor → BeeActors + DiffusionSupervisor    │
│   ├── FlockSupervisor → CloneActors (prefix-cached)        │
│   ├── McpResearcherActor                                    │
│   └── UserProxyActor                                        │
├─ PLANNING AGENT: LangGraph (decisions only, NOT execution)  │
├─ EXECUTION: Python asyncio (existing in engine.py, FQM)    │
├─ DATA: ConditionStore + DuckDB + 20+ indices                │
│   FlockMTL used ONLY for embeddings / hybrid search         │
└─────────────────────────────────────────────────────────────┘
```

### 17.4 What Each Component Does

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| Foundation | **strands-agent** | Store, tools, swarm bridge, MCP integrations, FastAPI server (~75% already works) |
| Recursive loop | **Custom actor-event** | 9-state machine, priority queue, nested recursion, dynamic spawning, user interrupts |
| Planning | **LangGraph** | High-level decisions: "research more? synthesize? ask user?" |
| Policy | **Custom rule engine** | Convergence detection, cost caps, GPU waste, destructive action gates |
| Algorithms | **ADK transplant** | 13-step corpus battery, maestro SQL, dashboard, thought swarm |
| Embeddings | **FlockMTL** | Embedding generation, hybrid search only. NOT Flock core. |
| Execution | **asyncio** | Native Python event loop. Already used throughout. |

### 17.5 What We Rejected

| Rejected | Reason |
|----------|--------|
| Revive ADK | asyncio blocking, 6–10 weeks, high risk |
| FlockMTL for core | 167× cost increase, no adaptive budget, no MCP interleaving |
| LangGraph for recursive loop | static topology, cannot spawn sub-graphs, global recursion limit |
| Custom from scratch now | 12–20 weeks. Defer to months 4–6 if needed. |

### 17.6 Implementation Roadmap (4 Weeks)

**Week 1: P0 Blockers — Store Bridge + Indices**
- Fix `GossipSwarm.synthesize()` to write `finding` rows with gradient flags
- Populate `cluster_id` (union-find clustering) and `contradiction_flag`
- Add 20+ indices to ConditionStore
- Make diffusion queen persist intermediate artifacts

**Week 2: Actor Model + Scheduler**
- Actor base class (`asyncio.Queue` mailbox)
- Supervision tree: SwarmSupervisor, FlockSupervisor, McpResearcherActor, UserProxyActor
- Scheduler daemon with priority event queue and health monitor

**Week 3: ADK Algorithm Battery Transplant**
- Port 13-step corpus pipeline (scoring, clustering, contradiction, narrative chains)
- Port maestro SQL conductor
- Port real-time dashboard (SSE KPIs, HTML reports)
- Port thought swarm arbitration

**Week 4: Integration + One-Liner API**
- Unified orchestrator: `orch.run(query)`
- Rule engine MVP (7 must-have rules)
- Cross-run federation loader
- End-to-end test: full swarm→flock→external→reswarm cycle

### 17.7 The Longer View (Months 4–6)

After Phase 1 delivers a working recursive loop, Phase 2 revisits with real operational data:
- If asyncio actors suffice: keep the 350–450 LOC custom core
- If distributed execution needed: migrate supervision tree to Ray actors (1:1 mapping)
- If rule engine grows past 500 rules: migrate to Rete-algorithm engine
- Each layer is swappable without touching the others

### 17.8 One-Sentence Summary

> **Evolve strands-agent into a hybrid actor-event orchestrator with a transplanted ADK algorithm battery, keep LangGraph for planning only, preserve Python Flock for prefix caching, and deliver a working recursive loop in four weeks.**

---

*This document is both vision and reality check. Sections 1–9 describe the architecture as it should work. Sections 10–16 describe what is broken today. Section 17 describes the framework decision and implementation path to close the gap.*


## 18. Depth, Understandability, and Intelligent External Data Selection

*This section summarizes the converged architecture produced by 8 parallel subagents analyzing: reflexion persistence, operator querying, query wealth expansion, correlated semantic joins, external data benefit assessment, massive source ingestion, information curation, and cross-cutting synthesis. Full document: `CONVERGED_DEPTH_ARCHITECTURE.md` (5,901 words, 886 lines).*

### 18.1 The Core Insight

DEPTH and UNDERSTANDABILITY are not achieved by running more of the same queries. They are manufactured by seven converged pillars:

1. **Expanded query wealth** — 28 query types (9 foundation + 8 depth + 4 understandability + 4 meta + 3 composite)
2. **Semantic connections** — Verified, typed links between findings (causal, analogical, contradictory, supporting, etc.)
3. **Intelligent external data selection** — 13 benefit signals, cost estimation, UCB-greedy target selection
4. **Information curation** — 8 specialized curators transforming raw rows into actionable consumables
5. **Operator steering** — Three-tier query interface (structured / guided NL / freeform SQL)
6. **Reflexion persistence** — Typed lessons surviving VM teardown via delta sync to B2
7. **Massive source ingestion** — Anna's Archive pipeline with utility tracking and three-strikes blocking

### 18.2 Query Wealth Engine (28 Types)

| Layer | Types | Purpose |
|-------|-------|---------|
| **Foundation** (9) | VALIDATE, ADJUDICATE, VERIFY, ENRICH, GROUND, BRIDGE, CHALLENGE, SYNTHESIZE, AGGREGATE | Surface-level evaluation |
| **Depth** (8) | CAUSAL_TRACE, ASSUMPTION_EXCAVATE, EVIDENCE_MAP, SCOPE, METHODOLOGY, TEMPORAL, ONTOLOGY, REPLICATION | Mechanism, assumptions, evidence, boundaries |
| **Understandability** (4) | ANALOGY, TIER_SUMMARIZE, COUNTERFACTUAL, NARRATIVE_THREAD | Narrative, accessibility, counterfactuals |
| **Meta** (4) | META_PRODUCTIVITY, META_EXHAUSTION, META_COVERAGE, META_EFFECTIVENESS | Self-awareness and optimization |
| **Composite** (3) | DEEP_VALIDATE, RESOLVE_CONTRADICTION, SYNTHESIS_DEEPEN | Multi-step depth bundles |

Selection is state-driven: the orchestrator analyzes store topology (novelty, confidence, contradiction, fabrication risk, cluster density, actionability) and allocates budget to the types most likely to produce information gain. Reflexion (`ReflexionState`) demotes exhausted types and promotes synergistic sequences.

### 18.3 Semantic Connection Pipeline

A three-stage pipeline detects non-obvious relationships between findings:

- **Stage 1 (Heuristic):** Term overlap, cluster membership, shared source, temporal proximity → reduces O(n²) to O(n×k)
- **Stage 2 (Embedding):** Cosine similarity threshold (≥0.72), angle-diversity check → 3,000× reduction
- **Stage 3 (LLM):** Batched verification via Python Flock with vLLM prefix caching → 10k findings to ~2,000–5,000 verified pairs

10 connection types: causal_link, contradictory, supporting, analogous, generalizing, specializing, methodological, temporal, statistical, compositional. Stored in `semantic_connections` table. Trigger targeted BRIDGE, SYNTHESIZE, CAUSAL_TRACE queries.

### 18.4 External Data Selection (Paramount)

The way external data is chosen determines whether the system is intelligent or an expensive random sampler.

**13 benefit signals** measure information deficits: VALIDATE gap, VERIFY gap, ADJUDICATE gap, SWARM_INTENT gap, WORKER_REQUEST gap, SOURCE_UPGRADE gap, ENRICH gap, GROUND gap, REPLICATION gap, DEADLOCK gap, MECHANISM gap, FRESHNESS gap, COVERAGE gap.

**Composite benefit score:** `B_final = B_raw × saturation_penalty × context_penalty × convergence_boost`

**UCB-greedy target selection** with cost-aware knapsack constraints and serendipity floor (≥3 reason types represented).

**Cost estimation** covers 10+ APIs across USD, tokens, latency, and context-window interest dimensions.

**Dynamic budget allocation** scales with ROI: >10× ROI → 1.3× budget; <0.1× ROI → 0.2× budget.

**Operator override:** Green (<$0.05, auto), Yellow ($0.05–$0.50, 10s pause), Red (>$0.50 or >5K tokens or >10s, halt for approval).

**Three-phase feedback loop:** Immediate (score deltas), Latent (synthesis/bridge enablement), ROI (info_gain_per_usd).

### 18.5 Curation Architecture

Eight specialized curators under a `CurationSupervisor` actor:

| Curator | Consumer | Output |
|---------|----------|--------|
| GlobalHealthCurator | Orchestrator, RuleEngine | Convergence snapshot, phase recommendation |
| CloneContextCurator | FlockSupervisor | Prefix-cache-optimized clone contexts (drops 640 SQL queries to ~8) |
| AngleContextCurator | SwarmSupervisor | 100 top findings in 4 tiers (established, controversial, bridge-worthy, recent) |
| ContradictionCurator | Orchestrator, Operator | Prioritized contradiction queue |
| GapCurator | Orchestrator, McpResearcher | Unified MCP research priority queue |
| NarrativeCurator | Operator | Human-readable progress narrative |
| LessonCurator | All supervisors | Validated lessons for reflexion |
| SourceQualityCurator | RuleEngine, Flock | Source quality report (activates dead `trust_score` column) |

### 18.6 Reflexion Loop Across VMs

**Detect:** `ReflexionActor` analyzes post-phase metrics and stores typed lessons (`strategy_lesson`, `source_quality_lesson`, `model_behavior_lesson`, `cost_lesson`, `query_efficiency_lesson`, `angle_quality_lesson`).

**Store:** Append-only `lessons` table with `confidence × relevance_score` ranking.

**VM Portability:** `StoreSyncActor` exports deltas to Parquet every 60s + uploads base snapshot to B2 at run end. Lazy restore on VM startup.

**Apply:** Lessons inject into clone context prefixes, scheduler rules, MCP target filtering, and operator dashboard.

### 18.7 Operator Query Interface

Three-tier hybrid interface:
- **Tier 1 (Structured):** Natural language mapped to pre-validated SQL templates. <50ms. No LLM call.
- **Tier 2 (Guided NL):** Lightweight LLM planner decomposes question into template chain. 2–5s.
- **Tier 3 (Freeform SQL/Maestro):** `/sql` or `/maestro` prefix. Destructive-action gates. Forbidden: DELETE, DROP, TRUNCATE.

Cross-run querying via `runs` table and `run_number` column.

### 18.8 Expanded Actor Supervision Tree

The tree from §17.3 is extended with new actors:

```
OrchestratorActor
├── Policy Layer (RuleEngineActor)
├── SwarmSupervisor → BeeActors + DiffusionSupervisor
├── FlockSupervisor → CloneActors (prefix-cached)
├── McpResearcherActor → ToolActors + AnnaArchiveToolActor
├── UserProxyActor
├── OperatorQueryEngine ← NEW
├── CurationSupervisor ← NEW (8 curators)
├── ReflexionActor ← NEW
├── StoreSyncActor ← NEW
├── SemanticConnectionWorker ← NEW
└── SourceIngestionActor ← NEW
```

### 18.9 Implementation Roadmap (6 Weeks)

| Week | Focus |
|------|-------|
| **1** | Schema + P0 blockers (store bridge, indices, ghost columns) |
| **2** | Actor scaffold + curation MVP |
| **3** | Query wealth expansion + semantic connections table |
| **4** | External data benefit assessment + operator override |
| **5** | Reflexion persistence + VM sync |
| **6** | Massive sources + operator query interface + end-to-end test |

### 18.10 New Data Model Additions

Ten new tables beyond §13:
- `lessons`, `lesson_applications` — reflexion persistence
- `semantic_connections` — verified finding-to-finding relationships
- `source_fingerprints`, `chunks`, `source_utility_log` — massive source ingestion
- `source_quality_registry`, `condition_sources` — learned source preferences
- `condition_embeddings` — semantic search vectors

Full schema, indices, pseudocode, and risk matrix in `CONVERGED_DEPTH_ARCHITECTURE.md`.

---

*This document is both vision and reality check. Sections 1–9 describe the architecture as it should work. Sections 10–16 describe what is broken today. Section 17 describes the framework decision. Section 18 describes the depth/understandability/external-data convergence.*
