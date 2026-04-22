# Swarm Wave Architecture: Complete Implementation Specification

*Full wave-by-wave lifecycle for 24-hour continuous operation.*

---

## Overview

This document specifies the complete execution lifecycle of the swarm engine — from corpus ingestion through 1440 continuous runs, detailing every data flow, every component interaction, and every decision point. It supersedes the ad-hoc prompt-stuffing and tool-calling approaches with a unified architecture based on three primitives:

1. **Data packages** — structured research briefs assembled by the orchestrator and injected into tool-free workers
2. **Cloned contexts** — worker conversation histories registered with a session proxy and used as expert LLM backends for Flock SQL
3. **The audit trail DAG** — every operation stored as a row with `parent_id` lineage in the ConditionStore

The existing gossip engine's RAG ("FROM THE HIVE" pattern in `rag.py`) demonstrated that targeted cross-angle injection kindles productive reasoning in workers. This architecture generalizes that pattern: instead of keyword-scored hive memory, the orchestrator uses cloned worker contexts to score relevance with domain expertise, assembles multi-layered data packages, and delivers them as conversation-starting research briefs.

---

## System Components

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                  │
│  (swarm/mcp_engine.py — the brain)                                   │
│                                                                      │
│  Responsibilities:                                                   │
│  - Ingest corpus into ConditionStore                                 │
│  - Detect angles from user prompt                                    │
│  - Assemble data packages per worker per wave                        │
│  - Launch tool-free workers                                          │
│  - Capture worker transcripts via hook observer                      │
│  - Register cloned contexts with session proxy                       │
│  - Run Flock SQL for catalogue operations (disaggregation,           │
│    relevance scoring, dedup, cross-domain connections)               │
│  - Generate rolling knowledge summaries                              │
│  - Check convergence                                                 │
│  - Generate final report                                             │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐
  │ Workers  │  │ Session  │  │ ConditionStore│  │ Hook Observer    │
  │ (bees)   │  │ Proxy    │  │ (DuckDB)      │  │ (Strands hooks)  │
  │          │  │          │  │               │  │                  │
  │ Tool-free│  │ Prepends │  │ Full audit    │  │ AfterModelCall   │
  │ reasoning│  │ cloned   │  │ trail DAG     │  │ AfterInvocation  │
  │ agents   │  │ context  │  │ + Flock ext   │  │ MessageAdded     │
  └──────────┘  │ to vLLM  │  └───────────────┘  └──────────────────┘
                │ requests │
                └─────┬────┘
                      ▼
                ┌──────────┐
                │  vLLM    │
                │ (8×H200) │
                │ prefix   │
                │ caching  │
                └──────────┘
```

### Component Inventory

| Component | File | Purpose |
|---|---|---|
| Orchestrator | `swarm/mcp_engine.py` | Controls entire lifecycle, assembles data packages, runs Flock SQL |
| Worker factory | `swarm/agent_worker.py` | Creates tool-free worker agents with angle-specific system prompts |
| Hook observer | `swarm/worker_observer.py` | Strands hook provider capturing worker transcripts (read-only) |
| Session proxy | `swarm/session_proxy.py` | FastAPI service mapping model names to cloned conversations, forwarding to vLLM |
| ConditionStore | `apps/strands-agent/corpus.py` | DuckDB-backed store with Flock extension, full audit trail schema |
| Data package builder | `swarm/data_package.py` | Assembles structured research briefs per worker per wave |
| RAG scorer | `swarm/rag.py` | Keyword-based scoring (bootstrap); replaced by clone-scored Flock queries after wave 1 |
| Angle detector | `swarm/angles.py` | Prompt-driven angle extraction and section-angle assignment |
| Knowledge summarizer | `swarm/summaries.py` | Generates rolling per-angle knowledge summaries between waves |
| Compactor | `swarm/compactor.py` | Two-phase dedup (exact SQL + semantic via cloned context) |

---

## The Data Package

The data package is the single most important design element. It determines what the worker thinks about. A bad package produces generic summaries. A good package kindles deep, angle-specific reasoning with productive surprise.

### Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                       RESEARCH BRIEF                             │
│                Wave {N}, Angle: {angle_name}                     │
│                Worker: {worker_id}, Model: {model_name}          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 1  KNOWLEDGE STATE                                    ║   │
│  ║                                                          ║   │
│  ║  Rolling summary of what the swarm knows about this      ║   │
│  ║  angle so far. Not raw findings — distilled              ║   │
│  ║  understanding. Produced by the previous wave's          ║   │
│  ║  knowledge summarizer.                                    ║   │
│  ║                                                          ║   │
│  ║  Wave 1: Empty (first wave has no prior knowledge)       ║   │
│  ║  Wave 2+: "The swarm has established that insulin        ║   │
│  ║  sensitivity peaks 2-4h post-exercise, that GH           ║   │
│  ║  co-administration shifts the window by ~30min..."        ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 2  CORPUS MATERIAL                                    ║   │
│  ║                                                          ║   │
│  ║  Raw corpus excerpts relevant to this angle.             ║   │
│  ║                                                          ║   │
│  ║  Wave 1: Static section assignment (semantic scoring     ║   │
│  ║  of corpus sections against angles, as in current        ║   │
│  ║  engine.py). Workers get their assigned section.         ║   │
│  ║                                                          ║   │
│  ║  Wave 2+: Clone-scored corpus material. The              ║   │
│  ║  orchestrator asks the clone: "Which corpus excerpts     ║   │
│  ║  would deepen the worker's current analysis?" The        ║   │
│  ║  clone knows what the worker is thinking — it IS the     ║   │
│  ║  worker's context. Different from wave 1 assignment      ║   │
│  ║  because the clone may surface corpus passages that      ║   │
│  ║  weren't in the original section but are NOW relevant    ║   │
│  ║  given what the worker has learned.                      ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 3  FROM THE HIVE  (cross-angle RAG — the kindling)    ║   │
│  ║                                                          ║   │
│  ║  Findings from OTHER angles that relate to this          ║   │
│  ║  worker's current analysis.                              ║   │
│  ║                                                          ║   │
│  ║  Wave 1: Empty (no prior findings exist)                 ║   │
│  ║  Wave 2 (bootstrap): Keyword-scored RAG from rag.py.     ║   │
│  ║  Extract concepts from worker's wave-1 output, match     ║   │
│  ║  against other workers' outputs. This is the existing    ║   │
│  ║  "FROM THE HIVE" pattern that kindles conversation.      ║   │
│  ║                                                          ║   │
│  ║  Wave 3+: Clone-scored cross-angle RAG. The              ║   │
│  ║  orchestrator asks the clone: "Which findings from       ║   │
│  ║  OTHER angles would change this worker's analysis?"      ║   │
│  ║  The clone judges analytical relevance, not just         ║   │
│  ║  keyword overlap. Much more precise kindling.            ║   │
│  ║                                                          ║   │
│  ║  Framing: "These findings were retrieved because they    ║   │
│  ║  match concepts in your current analysis. Interpret      ║   │
│  ║  them through your {angle} lens — what do these          ║   │
│  ║  cross-domain findings MEAN in your domain?"             ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 4  CROSS-DOMAIN CONNECTIONS                           ║   │
│  ║                                                          ║   │
│  ║  Material that TWO OR MORE clones agree bridges          ║   │
│  ║  their domains. Output of cross-expert Flock queries.    ║   │
│  ║                                                          ║   │
│  ║  Wave 1-2: Empty (no clones yet, or clones too shallow)  ║   │
│  ║  Wave 3+: Orchestrator runs cross-clone chain:           ║   │
│  ║                                                          ║   │
│  ║    Step 1: Ask clone_insulin which hematology findings   ║   │
│  ║    have insulin implications                              ║   │
│  ║    Step 2: Ask clone_hematology to validate those as     ║   │
│  ║    genuine hematology                                     ║   │
│  ║    Intersection = validated cross-domain connection       ║   │
│  ║                                                          ║   │
│  ║  Stored as insight rows with parent_id → source          ║   │
│  ║  findings from each angle.                                ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 5  CHALLENGES                                         ║   │
│  ║                                                          ║   │
│  ║  Findings from other clones that CONTRADICT this         ║   │
│  ║  worker's current understanding. Framed as questions     ║   │
│  ║  to provoke deeper reasoning.                            ║   │
│  ║                                                          ║   │
│  ║  Wave 1-2: Empty                                         ║   │
│  ║  Wave 3+: Orchestrator asks each OTHER clone:            ║   │
│  ║  "What in your angle's findings contradicts {angle}?"    ║   │
│  ║                                                          ║   │
│  ║  Framing: "The hematology specialist found X, which      ║   │
│  ║  appears to conflict with your finding Y. What           ║   │
│  ║  mechanisms from {angle} explain this discrepancy?"       ║   │
│  ║                                                          ║   │
│  ║  Replaces the generic "contrarian challenge" template    ║   │
│  ║  (template 9) with expert-informed dissent.              ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 6  RESEARCH GAPS                                      ║   │
│  ║                                                          ║   │
│  ║  Topics where the store has no findings for this angle.  ║   │
│  ║  Extracted from the clone's analysis of store coverage.  ║   │
│  ║                                                          ║   │
│  ║  Wave 1: Implicit (the worker has the raw corpus)        ║   │
│  ║  Wave 2+: Orchestrator asks the clone:                   ║   │
│  ║  "What important aspects of {angle} have no findings     ║   │
│  ║  in the store yet?"                                      ║   │
│  ║                                                          ║   │
│  ║  Framing: "No analysis yet on: [gap list]. Consider      ║   │
│  ║  whether your corpus material contains evidence for      ║   │
│  ║  these gaps."                                            ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
│  ╔══════════════════════════════════════════════════════════╗   │
│  ║  § 7  PREVIOUS OUTPUT                                    ║   │
│  ║                                                          ║   │
│  ║  The worker's own previous wave output, for continuity.  ║   │
│  ║                                                          ║   │
│  ║  Wave 1: Empty (first wave)                              ║   │
│  ║  Wave 2+: The worker's full output from wave N-1.        ║   │
│  ║  Enables the worker to build on its prior reasoning      ║   │
│  ║  rather than starting from scratch each wave.            ║   │
│  ╚══════════════════════════════════════════════════════════╝   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Package Size Budget

The data package must fit in the worker's context window alongside its system prompt and response space. For a 32K token model:

| Component | Wave 1 Budget | Wave 2+ Budget | Notes |
|---|---|---|---|
| System prompt | ~500 tokens | ~500 tokens | Angle identity, rules, framing |
| § 1 Knowledge state | 0 | ~2,000 tokens | Compressed rolling summary |
| § 2 Corpus material | ~12,000 tokens | ~8,000 tokens | Decreases as knowledge state grows |
| § 3 From the hive | 0 | ~4,000 tokens | Cross-angle kindling |
| § 4 Cross-domain | 0 | ~2,000 tokens | Validated connections |
| § 5 Challenges | 0 | ~1,500 tokens | Contradictions as questions |
| § 6 Research gaps | 0 | ~500 tokens | Short gap list |
| § 7 Previous output | 0 | ~4,000 tokens | Continuity |
| **Response budget** | **~19,500 tokens** | **~13,500 tokens** | What the worker can generate |
| **Total** | **32,000 tokens** | **32,000 tokens** | |

Wave 1 gives the worker maximum response space because it has the most raw material to reason about. Later waves trade response space for richer context (knowledge state, cross-angle material, challenges).

For models with larger context windows (64K, 128K), the budgets scale proportionally — more room for deeper knowledge state and more cross-angle material.

---

## Wave Lifecycle: Blow-by-Blow

### Run Startup (Once Per 24-Hour Session)

```
1. CORPUS INGESTION
   │
   │  Input: 150MB YouTube transcript corpus
   │  
   │  a) Compute SHA256 fingerprint
   │  b) Check corpus_fingerprints table — skip if already ingested
   │  c) Stream corpus in chunks (not monolithic load):
   │     - Read 1MB chunk
   │     - Split into paragraphs
   │     - INSERT each paragraph as row_type='raw', parent_id=NULL
   │     - Fingerprint per chunk for resumability
   │  d) Store fingerprint in corpus_fingerprints
   │  
   │  Output: N raw corpus paragraphs in ConditionStore
   │  Audit: Each raw paragraph is a root node in the DAG

2. ANGLE DETECTION
   │
   │  Input: User's research prompt
   │  
   │  a) Extract required angles from the prompt via LLM
   │     (prompt-driven guarantee — angles emerge from the query,
   │      not preset and not LLM-guessed from corpus)
   │  b) Score corpus sections against angles (semantic assignment)
   │  c) Merge required + detected angles, cap at max_workers
   │  
   │  Output: N angles, each with assigned corpus sections
   │  Example: ["insulin timing & sensitivity", "hematological markers",
   │            "growth hormone protocols", "ancillary compounds",
   │            "training periodization"]

3. FLOCK SETUP
   │
   │  a) LOAD flock extension in DuckDB
   │  b) CREATE SECRET (TYPE OPENAI, API_KEY 'proxy-key',
   │                    BASE_URL 'http://localhost:18199/v1')
   │  c) CREATE MODEL('generic', 'default', 'openai')
   │     — generic model for wave 1 operations (no clones yet)
   │  
   │  Output: Flock ready, pointing at session proxy (which initially
   │          just passes through to vLLM without prepending context)

4. SESSION PROXY STARTUP
   │
   │  a) Launch FastAPI proxy on port 18199
   │  b) Configure vLLM backend endpoints (one per GPU)
   │  c) No sessions registered yet (empty session map)
   │  
   │  Output: Proxy running, routing all requests to vLLM directly
```

### Wave 1: Bootstrap (No Clones Available)

Wave 1 is special — there are no prior clones, no prior findings, no rolling summaries. The system must bootstrap from raw corpus alone.

```
WAVE 1
══════════════════════════════════════════════════════════════

Phase 1A: DATA PACKAGE ASSEMBLY (per worker)
─────────────────────────────────────────────
  │
  │  For each worker/angle:
  │  
  │  § 1 Knowledge state: EMPTY
  │  § 2 Corpus material: Assigned section from angle detection
  │      (semantic scoring of sections against angles, with
  │       deliberate misassignment for thread discovery —
  │       20-30% off-angle data injected)
  │  § 3 From the hive: EMPTY
  │  § 4 Cross-domain connections: EMPTY
  │  § 5 Challenges: EMPTY
  │  § 6 Research gaps: EMPTY
  │  § 7 Previous output: EMPTY
  │  
  │  Result: Workers get only their raw corpus section.
  │  This is identical to the current engine.py Phase 1 (Map).
  │  The magic happens in later waves.

Phase 1B: WORKER REASONING (parallel, tool-free)
─────────────────────────────────────────────────
  │
  │  For each worker (parallel, bounded by max_concurrency):
  │  
  │  a) Build system prompt:
  │     "You are a {angle} specialist. Everything you encounter,
  │      you interpret through the lens of {angle}..."
  │  b) Inject data package as the user message
  │  c) Worker reasons freely — no tools, no store interaction
  │     The worker produces its first analysis: findings,
  │     causal chains, predictions, cross-domain implications
  │  d) Worker output captured
  │  
  │  The worker's reasoning is the same quality as current
  │  engine.py Phase 1 output. The difference is what comes next.

Phase 1C: TRANSCRIPT CAPTURE (hook observer)
────────────────────────────────────────────
  │
  │  For each worker:
  │  
  │  a) AfterInvocationEvent fires → observer captures:
  │     - Full conversation: system prompt + data package + response
  │     - Model metadata: model name, token counts, latency
  │  b) Store as worker_transcript row:
  │     INSERT INTO conditions (
  │       row_type='worker_transcript',
  │       fact=<full conversation JSON>,
  │       angle=<worker's angle>,
  │       source_model=<model name>,
  │       source_run=<run identifier>,
  │       phase='wave_1'
  │     )
  │  
  │  Output: N transcript rows in the store (audit trail)

Phase 1D: CLONE REGISTRATION (orchestrator → session proxy)
───────────────────────────────────────────────────────────
  │
  │  For each worker:
  │  
  │  a) Extract conversation messages from transcript
  │  b) POST /sessions/{clone_id}
  │     Body: {messages: [system_prompt, data_package, response]}
  │  c) Session proxy stores the conversation
  │  d) CREATE MODEL('clone_{angle}', 'clone_{angle}', 'openai')
  │     in DuckDB (if not already created)
  │  
  │  Output: N cloned contexts registered. The session proxy now
  │  prepends the correct conversation when Flock sends requests
  │  with model_name='clone_{angle}'.
  │  
  │  vLLM prefix caching: The first Flock query through each
  │  clone will compute the full KV cache for the conversation.
  │  All subsequent queries reuse the cache (nearly free).

Phase 1E: CATALOGUE OPERATIONS (orchestrator via Flock + clones)
───────────────────────────────────────────────────────────────
  │
  │  Now that clones exist, the orchestrator runs store operations
  │  with domain expertise:
  │  
  │  E1. CLAIM EXTRACTION from worker transcripts
  │  ─────────────────────────────────────────────
  │  For each worker transcript:
  │    SELECT llm_complete(
  │      {'model_name': 'clone_{angle}'},
  │      {'prompt': 'Extract every factual claim from this
  │       analysis. Return each claim as a separate line.
  │       Preserve exact numbers and sources.
  │       Analysis: ' || transcript_text}
  │    )
  │  
  │  Each extracted claim → INSERT as row_type='finding',
  │  parent_id → transcript row, source_model, source_run, angle
  │  
  │  The clone extracts claims from ITS OWN reasoning. It knows
  │  which statements are factual claims vs. reasoning scaffolding
  │  because it has the full context of what it was thinking.
  │  
  │  E2. DISAGGREGATION of raw corpus paragraphs
  │  ────────────────────────────────────────────
  │  For large raw corpus paragraphs (>500 chars):
  │    SELECT id, fact,
  │      llm_complete(
  │        {'model_name': 'clone_{angle}'},
  │        {'prompt': 'Break this paragraph into atomic factual
  │         claims. Each claim should be independently
  │         understandable. Return one claim per line.
  │         Paragraph: ' || fact}
  │      ) as atoms
  │    FROM conditions
  │    WHERE row_type = 'raw'
  │      AND angle = '{angle}'
  │      AND LENGTH(fact) > 500
  │  
  │  Each atom → INSERT as row_type='atom',
  │  parent_id → raw paragraph, angle
  │  
  │  E3. RELEVANCE SCORING (per angle)
  │  ──────────────────────────────────
  │  For all atoms and findings not yet scored for this angle:
  │    SELECT id, fact,
  │      llm_filter(
  │        {'model_name': 'clone_{angle}'},
  │        fact,
  │        'Is this highly relevant to {angle}?'
  │      ) as relevant
  │    FROM conditions
  │    WHERE row_type IN ('atom', 'finding')
  │      AND consider_for_use = TRUE
  │  
  │  Each scoring judgment → INSERT as row_type='relevance_score',
  │  parent_id → scored row, angle, confidence = 1.0 or 0.0
  │  
  │  E4. INITIAL DEDUP (exact match only — wave 1 is too early
  │      for semantic dedup, clones are too shallow)
  │  ───────────────────────────────────────────────────────
  │  Pure SQL exact-match dedup within each angle.
  │  Dedup decisions stored as rows (see STORE_ARCHITECTURE.md).
  │  
  │  Output: Store now contains raw paragraphs, atoms, findings,
  │  relevance scores, dedup decisions — all linked by parent_id.
  │  The clones have been exercised, vLLM has cached their
  │  prefixes, and subsequent Flock calls will be fast.

Phase 1F: KNOWLEDGE SUMMARY GENERATION
──────────────────────────────────────
  │
  │  For each angle:
  │    SELECT fact, confidence FROM conditions
  │    WHERE angle = '{angle}'
  │      AND row_type = 'finding'
  │      AND consider_for_use = TRUE
  │    ORDER BY confidence DESC LIMIT 50
  │  
  │  Feed to clone:
  │    llm_complete(clone_{angle},
  │      'Synthesize these findings into a knowledge briefing.
  │       State what is established, what is uncertain, and what
  │       remains unknown. Be specific — preserve numbers and
  │       sources.')
  │  
  │  Store as knowledge_summary row for this angle + wave.
  │  This becomes § 1 of the next wave's data package.

Phase 1G: CONVERGENCE CHECK
───────────────────────────
  │
  │  Count new findings stored this wave (source_type='worker_analysis').
  │  Wave 1 always continues — there's no prior wave to compare to.
  │  
  │  → Proceed to Wave 2

═══════════════════════════════════════════════════════════════
```

### Wave 2: Bootstrap RAG (Keyword-Scored Hive Memory)

Wave 2 is the first wave where cross-angle material is available. Clones exist from wave 1 but are still shallow (one wave of context). The system uses a hybrid approach: keyword RAG for hive memory (proven to kindle), clone-scored for corpus and relevance.

```
WAVE 2
══════════════════════════════════════════════════════════════

Phase 2A: DATA PACKAGE ASSEMBLY (per worker)
─────────────────────────────────────────────
  │
  │  For each worker/angle:
  │  
  │  § 1 Knowledge state:
  │      Rolling summary from wave 1 (Phase 1F output).
  │      "The swarm established: [compressed findings]..."
  │  
  │  § 2 Corpus material (CLONE-SCORED):
  │      Orchestrator asks the wave-1 clone:
  │      
  │        SELECT id, fact FROM conditions
  │        WHERE row_type IN ('raw', 'atom')
  │          AND llm_filter(
  │            {'model_name': 'clone_{angle}'},
  │            fact,
  │            'Would this material deepen the analysis?'
  │          )
  │        ORDER BY confidence DESC
  │        LIMIT {corpus_budget}
  │      
  │      This may surface corpus passages from OUTSIDE the
  │      worker's original section — passages that are NOW
  │      relevant given what the worker learned in wave 1.
  │  
  │  § 3 From the hive (KEYWORD RAG — bootstrap):
  │      Use existing rag.py pattern:
  │      a) extract_concepts(worker's wave-1 output, top_k=15)
  │      b) query_hive(all_wave1_findings, concepts,
  │                    exclude_angle=this_angle, top_k=5)
  │      c) Format as "FROM THE HIVE" block with angle framing
  │      
  │      This is the EXISTING pattern from engine.py that the
  │      user confirmed "kindled the conversation in a very nice
  │      way." We use it here because clones are still shallow
  │      and keyword scoring is sufficient for initial kindling.
  │  
  │  § 4 Cross-domain connections: EMPTY
  │      (clones too shallow for reliable cross-expert queries)
  │  
  │  § 5 Challenges: EMPTY
  │      (too early — workers haven't established enough to
  │       meaningfully contradict)
  │  
  │  § 6 Research gaps:
  │      Extract from wave-1 output using gap markers
  │      (existing engine.py pattern — scan for "need more data",
  │       "unexplained", "insufficient evidence", etc.)
  │  
  │  § 7 Previous output:
  │      Worker's full wave-1 analysis for continuity.
  │  
  │  Result: Workers now have knowledge state + fresh corpus +
  │  cross-angle kindling + their own prior work. This is
  │  equivalent to the current gossip round 2 but structured.

Phase 2B: WORKER REASONING (parallel, tool-free)
─────────────────────────────────────────────────
  │  Same as Phase 1B but workers now have richer context.
  │  The "FROM THE HIVE" section kindles cross-domain reasoning.
  │  Workers produce deeper analysis informed by peer findings.

Phase 2C: TRANSCRIPT CAPTURE
─────────────────────────────
  │  Same as Phase 1C. New transcripts stored.

Phase 2D: CLONE UPDATE (not re-registration — APPEND)
─────────────────────────────────────────────────────
  │
  │  For each worker:
  │  
  │  a) Append wave-2 conversation to the clone's messages
  │     PUT /sessions/{clone_id}/append
  │     Body: {messages: [wave_2_package, wave_2_response]}
  │  b) vLLM prefix cache: The wave-1 prefix is still cached.
  │     The new conversation extends it. vLLM recomputes only
  │     the new tokens.
  │  
  │  The clone now has TWO waves of context — deeper expertise.
  │  Its relevance judgments in Phase 2E will be more nuanced
  │  than wave 1's.

Phase 2E: CATALOGUE OPERATIONS (same as 1E but better)
──────────────────────────────────────────────────────
  │
  │  Same operations as Phase 1E (claim extraction,
  │  disaggregation, relevance scoring, dedup) but:
  │  
  │  - Clones have 2 waves of context → better judgments
  │  - Semantic dedup now enabled (clones deep enough to judge
  │    whether two claims are "the same idea phrased differently")
  │  - Cross-angle relevance scoring: Each clone scores findings
  │    from OTHER angles for relevance to its own angle
  │    (preparation for wave 3's cross-domain connections)

Phase 2F: KNOWLEDGE SUMMARY UPDATE
───────────────────────────────────
  │  Same as Phase 1F but incorporates wave 2 findings.
  │  Summary grows richer but stays compressed.

Phase 2G: CONVERGENCE CHECK
───────────────────────────
  │  Compare new findings count to wave 1.
  │  If findings per wave dropping below threshold → convergence.
  │  Usually wave 2 INCREASES findings (kindling effect).
  │  → Proceed to Wave 3

═══════════════════════════════════════════════════════════════
```

### Wave 3+: Full Expert Mode (Clone-Scored Everything)

From wave 3 onward, clones have 2+ waves of accumulated context. The system operates at full capability.

```
WAVE 3+  (steady-state operation for remaining 24 hours)
══════════════════════════════════════════════════════════════

Phase NA: DATA PACKAGE ASSEMBLY (FULL — all 7 sections populated)
────────────────────────────────────────────────────────────────
  │
  │  § 1 Knowledge state: Rich rolling summary (2+ waves)
  │  
  │  § 2 Corpus material: Clone-scored (same as wave 2)
  │      But NOW the clone has deeper context, so it surfaces
  │      different material — passages that are relevant given
  │      the accumulated reasoning, not just the raw angle.
  │  
  │  § 3 From the hive (CLONE-SCORED — graduates from keyword):
  │      
  │      Orchestrator asks the clone via Flock:
  │        SELECT id, fact, angle FROM conditions
  │        WHERE angle != '{this_angle}'
  │          AND row_type = 'finding'
  │          AND consider_for_use = TRUE
  │          AND llm_filter(
  │            {'model_name': 'clone_{angle}'},
  │            fact,
  │            'Would this finding from another domain change
  │             or deepen this workers current analysis?'
  │          )
  │        ORDER BY confidence DESC
  │        LIMIT {hive_budget}
  │      
  │      The clone doesn't match keywords — it judges
  │      ANALYTICAL RELEVANCE. It knows what the worker is
  │      currently thinking about (it has the same context).
  │      It selects material that would genuinely kindle
  │      new reasoning, not just material that shares terms.
  │  
  │  § 4 Cross-domain connections (NEW — expert-driven):
  │      
  │      For each OTHER angle, orchestrator runs cross-clone chain:
  │      
  │        -- Step 1: Ask this worker's clone which findings from
  │        -- another angle have implications for this angle
  │        CREATE TEMP TABLE relevant_from_other AS
  │        SELECT id, fact FROM conditions
  │        WHERE angle = '{other_angle}'
  │          AND row_type = 'finding'
  │          AND consider_for_use = TRUE
  │          AND llm_filter(
  │            {'model_name': 'clone_{this_angle}'},
  │            fact,
  │            'Does this have implications for {this_angle}?'
  │          );
  │      
  │        -- Step 2: Ask the OTHER angle's clone to validate
  │        SELECT id, fact FROM relevant_from_other
  │        WHERE llm_filter(
  │          {'model_name': 'clone_{other_angle}'},
  │          fact,
  │          'Is this a substantive finding in {other_angle}?'
  │        );
  │      
  │      Intersection = validated cross-domain connection.
  │      Stored as insight rows with lineage.
  │      Top connections included in data package.
  │  
  │  § 5 Challenges (NEW — expert-informed dissent):
  │      
  │      For each OTHER angle's clone:
  │        llm_complete(clone_{other_angle},
  │          'Based on your domain expertise, what findings
  │           from {this_angle} are you skeptical about?
  │           What evidence from your domain contradicts them?')
  │      
  │      Formatted as questions:
  │      "The hematology specialist questions your finding that
  │       [X]. Their evidence: [Y]. What mechanisms from
  │       {angle} explain this discrepancy?"
  │  
  │  § 6 Research gaps:
  │      Clone-identified gaps:
  │        llm_complete(clone_{angle},
  │          'What important aspects of {angle} have no
  │           findings in the store yet? List specific gaps.')
  │  
  │  § 7 Previous output: Worker's wave N-1 analysis.

Phase NB: WORKER REASONING
───────────────────────────
  │  Workers reason with the richest possible context:
  │  knowledge state + fresh corpus + expert-scored hive memory
  │  + validated cross-domain connections + expert challenges
  │  + identified gaps + their own prior work.
  │  
  │  The data package is a structured research conversation
  │  that provokes deep, angle-specific reasoning.

Phase NC: TRANSCRIPT CAPTURE
─────────────────────────────
  │  Same as before.

Phase ND: CLONE UPDATE (with context management)
─────────────────────────────────────────────────
  │
  │  Context window management for the clone:
  │  
  │  If clone conversation < 80% of context window:
  │    Append wave N conversation to existing clone
  │  
  │  If clone conversation > 80% of context window:
  │    ROLLING DISTILLATION:
  │    a) Ask the clone to summarize its accumulated knowledge:
  │       "Synthesize everything you know about {angle} into
  │        a comprehensive but compressed briefing."
  │    b) Replace the full conversation with:
  │       [system_prompt, distilled_summary, latest_wave_output]
  │    c) Re-register with session proxy
  │    d) vLLM recomputes prefix cache (one-time cost)
  │    e) Store the distillation event as an audit row
  │  
  │  This ensures clones never overflow while preserving
  │  the accumulated expertise in compressed form.

Phase NE: CATALOGUE OPERATIONS (full suite)
───────────────────────────────────────────
  │
  │  All operations from Phase 1E, plus:
  │  
  │  E5. CROSS-EXPERT POLLINATION
  │  ────────────────────────────
  │  For each pair of angles (N×(N-1) queries):
  │    Ask clone_A: "Which of angle_B's findings have
  │    implications for your domain?"
  │  Store as insight rows with parent_id → source findings.
  │  
  │  E6. EXPERT DISAGREEMENT DETECTION
  │  ──────────────────────────────────
  │  For findings judged by multiple clones:
  │    Compare clone_A's relevance score vs clone_B's.
  │    Disagreement = interesting research target.
  │  Store as contradiction rows with parent_id → finding,
  │  related_id → the clone that disagrees.
  │  
  │  E7. META-EXPERT COMPOSITION (optional, high-value waves)
  │  ─────────────────────────────────────────────────────────
  │  For the most productive angle pairs (measured by
  │  cross-domain connections found):
  │    Compose meta-expert by concatenating two clone
  │    conversations:
  │      POST /sessions/meta_{angle_a}_{angle_b}
  │      Body: {messages: clone_a_msgs + clone_b_msgs}
  │    
  │    Ask the meta-expert questions that require both domains:
  │      "What interactions exist between {angle_a} findings
  │       and {angle_b} findings that neither specialist
  │       alone would identify?"
  │  
  │  Meta-expert insights stored as insight rows with
  │  parent_id → representative findings from both angles.

Phase NF: COMPACTION (periodic — every K waves)
───────────────────────────────────────────────
  │
  │  Not every wave. Run every K waves (e.g., K=5) or when
  │  store exceeds a row count threshold.
  │  
  │  Phase 1: Exact-match dedup (pure SQL, cheap)
  │  Phase 2: Semantic dedup (Flock + cloned contexts)
  │  
  │  Every decision stored as dedup_decision row with lineage.
  │  See STORE_ARCHITECTURE.md for details.
  │  
  │  Aggressive compaction is critical for 24-hour operation:
  │  Without it, store grows ~6000 rows/run → 8.6M rows over
  │  1440 runs. With compaction every 5 runs, growth rate drops
  │  to ~2000 net new rows/run → 2.9M rows over 1440 runs.

Phase NG: CONVERGENCE CHECK
───────────────────────────
  │
  │  Measure: New findings per wave (source_type='worker_analysis')
  │  
  │  If findings_per_wave < threshold for consecutive_waves:
  │    → Mark this run as converged
  │    → Generate final report for this run
  │    → Start next run (new wave counter, same persistent store)
  │  
  │  If max_waves reached:
  │    → Same as convergence
  │  
  │  Between runs: The store persists. The clones persist (or are
  │  reconstructed from stored transcripts). The next run's wave 1
  │  starts with the rolling knowledge summary from the previous
  │  run — continuity across the 24-hour session.

═══════════════════════════════════════════════════════════════
```

---

## Cross-Run Continuity (24-Hour Operation)

Each "run" is a complete swarm cycle (angle detection → waves → convergence → report). Over 24 hours, 360–1440 runs execute against the same persistent store. Here's how state carries across:

```
Run 1                    Run 2                    Run N
┌──────────┐            ┌──────────┐            ┌──────────┐
│ Wave 1-K │            │ Wave 1-K │            │ Wave 1-K │
│          │            │          │            │          │
│ Findings │─persist──► │ Findings │─persist──► │ Findings │
│ in store │            │ + prior  │            │ + all    │
│          │            │ findings │            │ prior    │
│ Clones   │─serialize─►│ Clones   │─serialize─►│ Clones   │
│ (wave K  │            │ (wave K  │            │ (wave K  │
│  context)│            │  context)│            │  context)│
│          │            │          │            │          │
│ Knowledge│─persist──► │ Knowledge│─persist──► │ Knowledge│
│ summaries│            │ summaries│            │ summaries│
└──────────┘            └──────────┘            └──────────┘
     │                       │                       │
     ▼                       ▼                       ▼
  Report 1               Report 2               Report N
```

### Clone Persistence Across Runs

At the end of Run M, each clone's conversation is already stored as `worker_transcript` rows in the ConditionStore. At the start of Run M+1:

```python
# Reconstruct clone from prior run's transcript
prior_transcript = store.query("""
    SELECT fact FROM conditions
    WHERE row_type = 'worker_transcript'
      AND angle = '{angle}'
      AND source_run = '{prior_run_id}'
    ORDER BY id DESC LIMIT 1
""")
messages = json.loads(prior_transcript)
proxy.register_session(f"clone_{angle}", messages=messages)
```

If the prior transcript exceeds the context window, use the rolling knowledge summary instead:

```python
# Compressed clone seed from knowledge summary
summary = store.query("""
    SELECT summary FROM knowledge_summaries
    WHERE angle = '{angle}'
    ORDER BY run_number DESC LIMIT 1
""")
messages = [
    {"role": "system", "content": f"You are a {angle} specialist..."},
    {"role": "assistant", "content": summary}
]
proxy.register_session(f"clone_{angle}", messages=messages)
```

Run M+1's wave 1 clones start with either full prior conversation (if fits) or compressed expertise (if not). Either way, the first wave of a new run already has expert-scored RAG and clone-driven data packages — no cold start.

### Store Growth Management

| Metric | Without Compaction | With Compaction (every 5 waves) |
|---|---|---|
| Rows per wave | ~6,000 | ~2,000 net new |
| Rows per run (5 waves) | ~30,000 | ~10,000 net new |
| Rows after 100 runs | ~3,000,000 | ~1,000,000 |
| Rows after 1440 runs | ~43,200,000 | ~14,400,000 |
| Estimated store size | ~25 GB | ~8 GB |

DuckDB handles these volumes well — it's designed for analytical workloads on datasets of this size. The `conditions` table should have indexes on `angle`, `row_type`, `consider_for_use`, and `parent_id` for efficient Flock queries.

---

## The RAG Graduation Path

The system starts with dumb RAG and graduates to expert RAG:

```
Wave 1: NO RAG
  │  Workers have only raw corpus.
  │  No cross-angle material available.
  │  This is the cold start.
  │
  ▼
Wave 2: KEYWORD RAG (existing rag.py — proven to kindle)
  │  extract_concepts() → score_relevance() → query_hive()
  │  "FROM THE HIVE" block injected into data package § 3.
  │  Keyword overlap scoring — imprecise but effective.
  │  This is what the user confirmed works well.
  │
  ▼
Wave 3+: CLONE-SCORED RAG (expert relevance judgment)
  │  Flock llm_filter with clone context → analytical relevance.
  │  The clone knows what the worker is thinking about.
  │  It selects material that would genuinely change the analysis,
  │  not just material that shares keywords.
  │  Much more precise kindling.
  │
  ▼
Wave 5+: CROSS-EXPERT RAG (multi-clone chain queries)
  │  Cross-clone Flock queries find validated connections.
  │  Material that TWO experts agree bridges their domains.
  │  This is discovery that no single worker could make.
  │
  ▼
Run 2+: PERSISTENT EXPERT RAG (clone carries prior run knowledge)
  │  Clones reconstructed from prior run transcripts.
  │  RAG scoring informed by ALL prior waves' expertise.
  │  The system gets better at kindling over time because
  │  the clones get deeper.
  │
  ▼
Run 100+: META-EXPERT RAG (composed multi-domain expertise)
     Meta-experts (concatenated clone pairs) score material
     that requires understanding of TWO domains simultaneously.
     Connections that individual clones would miss.
```

### Why Keyword RAG Stays for Wave 2

The user observed that keyword RAG "kindled the conversation in a very nice way." This means the current `rag.py` implementation already produces good results for initial cross-pollination. The clone at wave 1 has only one wave of context — it's not significantly better than keyword scoring for broad cross-angle matching.

The value of clone-scored RAG emerges at wave 3+ when the clone has enough accumulated context to make nuanced analytical relevance judgments. Prematurely switching to clone-scored RAG at wave 2 would add latency (Flock queries are slower than keyword scoring) without proportional quality improvement.

The hybrid approach: keyword RAG for the first cross-angle injection (fast, proven), clone-scored RAG once clones are deep enough to outperform keywords (wave 3+).

---

## The Serendipity Battery Integration

The existing 13-step Flock template battery (documented in `FLOCK_SERENDIPITY_ARCHITECTURE.md`) integrates directly with the cloned-context pattern. Each template that currently uses a generic `corpus_model` can route through angle-specific clones:

| Template | Generic (Current) | Clone-Scored (New) | When Available |
|---|---|---|---|
| 2. Score (trust, novelty) | Generic LLM scores | Clone scores its own angle's findings | Wave 1+ |
| 4. Detect Contradictions | Generic comparison | Two clones from different angles judge same finding | Wave 3+ |
| 5. Cluster | Generic grouping | Clone clusters with domain understanding | Wave 2+ |
| 6. Compress Redundancy | Generic dedup | Domain expert judges true redundancy | Wave 2+ |
| 9. Contrarian Challenge | Generic skeptic | OTHER angle's clone critiques (informed dissent) | Wave 3+ |
| 10. Cross-Angle Bridge | Generic polymath prompt | Cross-clone chain queries (Ramification 2) | Wave 3+ |
| 12. Consensus Detector | Generic counter-claims | Clone from DIFFERENT angle generates challenges | Wave 3+ |
| 13. Minority Amplifier | Generic boost | Clone identifies genuinely underrepresented findings | Wave 2+ |

Templates 1, 3, 7, 8, 11 are pure SQL — they don't use LLMs and need no modification.

The transition is seamless: change `{'model_name': 'corpus_model'}` to `{'model_name': 'clone_{angle}'}` in the SQL templates. The Flock extension doesn't know or care what's behind the model name — the session proxy handles the context prepending transparently.

---

## H200 Resource Allocation

### GPU Assignment (8×H200, 143 GB VRAM each)

```
GPU 0-4: Worker models (one per GPU)
  │  Each GPU serves one model architecture (e.g., Qwen3.5-32B,
  │  GLM-4-32B, etc.). Workers assigned to angles run on these.
  │  vLLM prefix caching stores cloned conversation KV states.
  │  
  │  Per GPU capacity:
  │    Model weights: ~64 GB (32B fp16)
  │    Available for KV cache: ~79 GB
  │    Max cached conversations: ~9 at 32K tokens
  │    Enough for: 2 worker conversations + 7 clone sessions

GPU 5-6: Catalogue operations (Flock queries)
  │  Dedicated to handling Flock SQL requests — disaggregation,
  │  relevance scoring, dedup, cross-domain queries.
  │  These GPUs get heavy burst traffic after each wave when
  │  catalogue operations run in parallel.
  │  
  │  Can serve a smaller/faster model optimized for classification
  │  (the Flock queries are mostly yes/no or short-answer).

GPU 7: Report generation + meta-experts
  │  Handles knowledge summary generation, final reports,
  │  and meta-expert composition queries.
  │  Intermittent usage — busy between runs, idle during waves.
```

### Timing Budget (Per Wave)

| Phase | Estimated Duration | Notes |
|---|---|---|
| Data package assembly | 30-60s | Flock queries for clone-scored material |
| Worker reasoning | 60-120s | Parallel across GPUs, bounded by slowest |
| Transcript capture | <1s | Hook fires synchronously |
| Clone registration | <1s | HTTP POST to proxy |
| Catalogue operations | 120-300s | Heaviest phase — many Flock queries |
| Knowledge summary | 30-60s | One LLM call per angle |
| Convergence check | <1s | Pure SQL |
| **Total per wave** | **~5-10 min** | |
| **Per run (5 waves)** | **~25-50 min** | |
| **Runs in 24 hours** | **~30-60 runs** | Conservative estimate |

Note: 360-1440 runs was based on faster per-run times with simpler architecture. With full catalogue operations, expect 30-60 runs in 24 hours — but each run produces much deeper, better-curated findings. Quality over quantity.

---

## Implementation Phases (Mapped to GitHub Issues)

### Phase 1: Foundation (Issues #183, #184, #190, #191, #192)
Remove truncation, make workers tool-free, clean up corpus fingerprinting, convergence fix, source provenance. These are prerequisites — mostly already implemented on PR #181.

### Phase 2: Hook Observer + Data Package (Issues #194, NEW)
Implement the Strands hook observer (`worker_observer.py`) and the data package builder (`data_package.py`). Workers receive structured research briefs instead of raw prompt stuffing.

### Phase 3: Session Proxy + Clone Registration (Issues #185, #197)
Build the session proxy (`session_proxy.py`), implement clone registration flow, configure Flock to route through the proxy with per-clone model names.

### Phase 4: Catalogue Operations (Issues #186, #196)
Implement claim extraction, disaggregation, relevance scoring, semantic dedup, and cross-domain connection discovery — all driven by cloned contexts via Flock SQL.

### Phase 5: RAG Graduation (Issue #183 extended)
Implement the keyword → clone-scored → cross-expert RAG graduation path. Integrate existing `rag.py` for wave 2 bootstrap, Flock-based scoring for wave 3+.

### Phase 6: Compaction + Store Management (Issue #186)
Two-phase compaction with audit trail. Periodic execution. Store growth monitoring.

### Phase 7: 24-Hour Continuity (Issues #187, #188, #189, #193, #195)
Cross-run clone persistence, rolling knowledge summaries as clone seeds, 150MB corpus streaming, API key safety, report generation from summaries.

### Phase 8: Serendipity Integration (Issue #196)
Route existing Flock template battery through cloned contexts. Transition from generic to expert-driven serendipity.

### Phase 9: QR Ontology + Graph-Structured Retrieval (NEW)
Expand `row_type` values to include QR entity types (theme, code, codeGroup, memo, researchQuestion). Implement relationship typing in clone catalogue operations. Rewrite data package assembly as graph traversal queries. Research question decomposition at run start. Theme emergence after wave 2+.

### Phase 10: Unleashed Clone Research (NEW)
Implement doubt extraction from worker transcripts. Build clone-with-tools agent factory. Trigger logic (high-uncertainty detection, gap density, convergence stall). Budget enforcement (max tool calls, max doubts per wave). Add § 8 FRESH EVIDENCE to data package structure.

---

## Qualitative Research Ontology: Pre-Baked Graph Structure

### The Problem with Untyped Edges

The ConditionStore is a graph that doesn't know it's a graph. It has `parent_id` and `related_id` edges, but they're semantically impoverished — a `parent_id` means "derived from" whether it's a finding derived from corpus, an atom derived from a finding, or a dedup decision about a finding. And `related_id` means "related to" generically. The graph exists structurally but has no vocabulary for what its edges MEAN.

Microsoft-style GraphRAG solves this by discovering the ontology at runtime — entity extraction, relationship extraction, community detection, hierarchical summaries. But this is expensive, error-prone, and redundant: we don't need to discover how research data relates because the qualitative research tradition already defined it decades ago.

### The Shortcut: Qualitative Research MCP Ontology

The [qualitative-research MCP server](https://github.com/tejpalvirk/qualitativeresearch) provides a pre-baked ontology for research knowledge graphs. Its entity and relationship types map directly to what the swarm needs:

#### Entity Types

| QR Entity | Swarm Equivalent | `row_type` Value | Description |
|---|---|---|---|
| `project` | Research query / run | `project` | The user's research prompt + metadata |
| `participant` | Worker (angle) | `worker` | Each angle-assigned bee |
| `interview` | Worker transcript | `worker_transcript` | Each wave's full reasoning (already exists) |
| `observation` | Raw corpus paragraph | `raw` | Ingested corpus material (already exists) |
| `document` | Corpus source | `document` | YouTube transcript, paper, etc. |
| `code` | Interpretive tag | `code` | Angle-specific label applied to data |
| `codeGroup` | Angle | `codeGroup` | The domain lens grouping related codes |
| `memo` | Clone's analytical reflection | `memo` | WHY a relationship was typed, analytical notes |
| `theme` | Emergent pattern | `theme` | Cross-finding pattern within an angle |
| `quote` | Extracted excerpt | `atom` | Atomic claim from corpus (already exists) |
| `researchQuestion` | Sub-question per angle | `researchQuestion` | Driving questions, gap tracking |
| `finding` | Worker finding | `finding` | Conclusion from reasoning (already exists) |
| `literature` | Source attribution | `source` | YouTube channel, timestamp, DOI |

#### Relationship Types (Typed Edges)

Relationships are stored as rows in the `conditions` table where `row_type` is the relationship type, `parent_id` points to the source entity, and `related_id` points to the target entity. The `fact` column holds the relationship description or justification.

| Relationship | Meaning | What It Enables |
|---|---|---|
| `supports` | Finding A provides evidence for Theme/Finding X | § 3 HIVE: retrieve supporting evidence from other angles |
| `contradicts` | Finding A conflicts with Finding/Theme B | § 5 CHALLENGES: retrieve contradictions as questions |
| `triangulates_with` | Two findings from different angles converge on same conclusion | § 4 CROSS-DOMAIN: validated multi-angle connections |
| `answers` | Finding addresses a research question | § 6 GAPS: find questions with no `answers` edge |
| `codes` | Interpretive tag applied to a data segment | Clone assigns angle-specific interpretive labels |
| `derived_from` | Provenance link | Existing `parent_id` — already implemented |
| `compares` | Explicit comparison between two findings | Cross-angle comparison stored as typed edge |
| `reflects_on` | Meta-analytical commentary about a finding | Clone's audit trail of analytical reasoning |
| `part_of` | Atom is part of a larger finding | Disaggregation lineage (supplements `parent_id`) |
| `contains` | Angle contains multiple themes | Hierarchical organization within angles |
| `precedes` | Temporal ordering | Sequence of analytical steps within a wave |

### How This Changes the RAG Architecture

**Before (flat retrieval):**
```
All findings → keyword or clone filter → top-K → data package
Clone answers: "Is this relevant?" (binary yes/no)
Scales: O(N) LLM calls where N = total findings
```

**After (graph-structured retrieval):**
```
Graph traversal along typed edges → structured retrieval → data package
Clone answers: "HOW does this relate?" (typed: supports/contradicts/triangulates/answers/...)
Scales: O(E) edge creation during catalogue, O(1) graph queries during retrieval
```

The clone's job shifts from **relevance oracle** (scanning all findings every wave) to **relationship typer** (typing each relationship once during catalogue, stored permanently as an edge). Data package assembly becomes pure SQL graph traversal — no LLM calls needed at retrieval time.

### Graph-Structured Data Package Assembly

Each data package section maps to a relationship type query:

```sql
-- § 3 FROM THE HIVE: findings from other angles that SUPPORT this worker's themes
SELECT f.id, f.fact, f.angle, r.fact as relationship_reason
FROM conditions f
JOIN conditions r ON r.parent_id = f.id
JOIN conditions t ON r.related_id = t.id
WHERE r.row_type = 'supports'
  AND t.row_type = 'theme'
  AND t.angle = '{this_angle}'
  AND f.angle != '{this_angle}'
ORDER BY f.confidence DESC
LIMIT {hive_budget};

-- § 4 CROSS-DOMAIN: findings that TRIANGULATE across angles
SELECT f1.fact as finding_a, f1.angle as angle_a,
       f2.fact as finding_b, f2.angle as angle_b,
       r.fact as connection_description
FROM conditions f1
JOIN conditions r ON r.parent_id = f1.id
JOIN conditions f2 ON r.related_id = f2.id
WHERE r.row_type = 'triangulates_with'
  AND (f1.angle = '{this_angle}' OR f2.angle = '{this_angle}')
ORDER BY r.confidence DESC
LIMIT {crossdomain_budget};

-- § 5 CHALLENGES: findings from other angles that CONTRADICT this worker's findings
SELECT f.fact, f.angle, r.fact as contradiction_reason,
       t.fact as contradicted_finding
FROM conditions f
JOIN conditions r ON r.parent_id = f.id
JOIN conditions t ON r.related_id = t.id
WHERE r.row_type = 'contradicts'
  AND t.angle = '{this_angle}'
  AND f.angle != '{this_angle}'
ORDER BY r.confidence DESC
LIMIT {challenge_budget};

-- § 6 GAPS: research questions with no answers
SELECT rq.fact as unanswered_question
FROM conditions rq
WHERE rq.row_type = 'researchQuestion'
  AND rq.angle = '{this_angle}'
  AND NOT EXISTS (
    SELECT 1 FROM conditions ans
    WHERE ans.row_type = 'answers'
      AND ans.related_id = rq.id
  );
```

No `llm_filter` calls. No scanning. The graph structure built during catalogue operations does the work at retrieval time.

### Clone Catalogue: From Binary Scoring to Relationship Typing

The clone's catalogue prompt changes from:

```
"Is this finding relevant to {angle}?"  → yes/no (binary, O(N) per wave)
```

To:

```
"Classify the relationship between this finding and {angle}'s themes.
 Options: supports, contradicts, triangulates_with, answers, compares,
 related_to, or none.
 If a relationship exists, explain WHY in one sentence."

→ typed edge + justification (stored as memo row with parent_id → relationship)
```

Each classification creates one edge in the graph. Over 24 hours, the graph accumulates hundreds of thousands of typed edges. Every retrieval query is a traversal — instant, no LLM needed.

### Theme Emergence

Themes are emergent patterns that cut across multiple findings within an angle. They don't exist in the raw corpus — they emerge from the worker's reasoning.

**Creation:** After wave 2+, the clone analyzes its worker's findings for recurring patterns:

```sql
SELECT llm_complete(
  {'model_name': 'clone_{angle}'},
  {'prompt': 'Review these findings and identify 3-5 emergent themes —
   recurring patterns or principles that connect multiple findings.
   For each theme, name it and list which findings support it.
   Findings: ' || GROUP_CONCAT(fact, '\n')}
) FROM conditions
WHERE row_type = 'finding'
  AND angle = '{angle}'
  AND consider_for_use = TRUE;
```

Each theme → INSERT as `row_type='theme'`, `angle`. Each finding-theme link → INSERT as `row_type='supports'`, `parent_id` → finding, `related_id` → theme.

Themes become the primary unit of knowledge organization. Workers in later waves reason about themes (not raw findings), and the data package presents cross-angle material in terms of theme relationships.

### Research Question Decomposition

At run start, the orchestrator decomposes the user's prompt into per-angle research questions:

```sql
-- User prompt: "Complete analysis of PED protocols"
-- Insulin angle gets:
INSERT INTO conditions (row_type, fact, angle) VALUES
  ('researchQuestion', 'What insulin timing protocols maximize nutrient partitioning?', 'insulin_timing'),
  ('researchQuestion', 'How does insulin sensitivity change with concurrent GH use?', 'insulin_timing'),
  ('researchQuestion', 'What are the dose-response curves for rapid-acting analogs?', 'insulin_timing');
```

Gap detection becomes trivial: find questions with no `answers` edge. As workers produce findings, the clone types `answers` relationships, gradually closing gaps and revealing which questions remain unresolved.

### Why Not Full GraphRAG

| GraphRAG Feature | Our Alternative | Why |
|---|---|---|
| Entity extraction (NER) | Clones are the entity resolvers | Clone knows "tren-e" = "trenbolone enanthate" from context |
| Relationship extraction | Clone catalogue with typed prompts | Domain expert types relationships, not generic NER |
| Community detection (Leiden) | Angles ARE the communities | Pre-defined by the research prompt |
| Hierarchical summaries | Rolling knowledge summaries per angle | Already in the architecture |
| Global search | Graph traversal over typed edges | Pure SQL, no LLM needed at query time |
| Local search | Clone-scored within-angle queries | Same Flock pattern, just scoped to angle |

GraphRAG's value is in the ontology, not the algorithm. We take the ontology (from qualitative research tradition) and skip the expensive runtime discovery.

---

## Unleashed Clones: Tool-Armed Expert Researchers

### The Pattern

Workers are tool-free bees — they receive data packages and reason. Their context stays pristine. But the CLONES of those workers carry the same accumulated expertise, the same doubts, the same "I wish I could verify..." moments. The worker's uncertainty is captured in its transcript.

What happens if you give the clone TOOLS?

Not catalogue tools (Flock SQL, relationship typing) — those are already in the architecture. RESEARCH tools. The same tools the strands-agent has: `search_corpus`, `get_peer_insights`, web search, API calls. The clone becomes an independent researcher that starts from the worker's exact state of knowledge and goes to resolve the worker's own doubts.

### Why This Is Powerful

The worker reasons: *"Trenbolone appears to increase hematocrit, but I'm uncertain about the dose-response curve. The corpus mentions 200mg and 400mg but not the intermediate range. I'd need bloodwork data from 300mg to confirm a linear relationship."*

This doubt is captured in the transcript. The clone has this exact context — it knows what's uncertain and WHY. When armed with tools, it doesn't start from scratch. It starts from a specific, well-articulated research question that emerged organically from deep domain reasoning.

A generic researcher would search "trenbolone hematocrit dose response." The unleashed clone searches for exactly the gap in its own understanding — the 300mg intermediate data point. It has the context to ask better questions because it has done the reasoning that revealed the gap.

### Architecture

```
Worker reasons (tool-free)
    │
    ▼
Hook observer captures transcript
    │
    ▼
Orchestrator registers clone with session proxy (normal flow)
    │
    ├──► Clone used as Flock backend (normal catalogue operations)
    │
    └──► Clone ALSO instantiated as a Strands Agent with tools
         │
         │  System prompt: "You are a {angle} specialist. You have
         │  been reasoning about {angle} and have specific doubts
         │  and uncertainties. You now have research tools. Go
         │  resolve your doubts. Every finding you discover must
         │  be stored."
         │
         │  Tools: search_corpus, web_search, get_peer_insights,
         │         store_finding, check_contradictions
         │
         │  Context: Full worker conversation history
         │  (same as the Flock clone, but now with tool access)
         │
         ▼
    Clone researches autonomously
         │
         │  - Searches for data it knows is missing
         │  - Verifies claims it was uncertain about
         │  - Follows up on cross-domain connections it noticed
         │  - Checks peer findings it found interesting
         │
         ▼
    All findings stored in ConditionStore with:
         - source_type = 'clone_research'
         - source_model = same as worker
         - angle = same as worker
         - parent_id → the transcript row that contained the doubt
         │
         ▼
    Clone discarded. Findings persist.
    Worker's next data package includes clone's discoveries.
```

### The Doubt Extraction Step

Before unleashing the clone, the orchestrator (or the clone itself via Flock) extracts the specific doubts from the worker's transcript:

```sql
SELECT llm_complete(
  {'model_name': 'clone_{angle}'},
  {'prompt': 'Review your reasoning transcript and list every
   uncertainty, doubt, unverified claim, and "I wish I knew..."
   moment. For each, state exactly what data would resolve it.
   Be specific — not "more research needed" but
   "bloodwork data showing hematocrit at 300mg/week trenbolone
   enanthate over 8+ weeks".'}
) as doubts;
```

Each doubt → INSERT as `row_type='researchQuestion'`, `parent_id` → transcript row. The clone then attacks these specific questions with tools.

### When to Unleash

Not every wave. Unleashing is expensive (tool calls, API calls, time). Trigger conditions:

1. **High-uncertainty wave**: Worker's output contains many hedged statements ("likely", "uncertain", "need data", "insufficient evidence"). The hook observer can flag this.
2. **Gap density**: Many `researchQuestion` rows with no `answers` edges for this angle.
3. **Convergence stall**: Information gain dropping but findings quality not high enough. Fresh research input needed.
4. **Cross-domain conflict**: A `contradicts` edge between this angle and another, where neither side has enough evidence. The clone can go find the deciding evidence.
5. **Scheduled**: Every N waves (e.g., every 3rd wave), regardless of triggers.

### The Feedback Loop

```
Worker reasons → doubts emerge → clone extracts doubts → clone researches
    → findings stored → next data package includes fresh evidence
    → worker reasons with resolved doubts → NEW doubts emerge → ...
```

Each cycle, the worker's uncertainty drives targeted research. The clone acts as the worker's research assistant — it knows exactly what the worker needs because it IS the worker (contextually).

### Relationship to Data Package

Clone research findings enter the data package differently from other material:

```
╔══════════════════════════════════════════════════════════╗
║  § 8  FRESH EVIDENCE (clone research results)            ║
║                                                          ║
║  Your clone-researcher investigated specific doubts from  ║
║  your previous analysis. Here's what it found:           ║
║                                                          ║
║  DOUBT: "Uncertain about 300mg tren-e hematocrit impact" ║
║  EVIDENCE: [clone's finding with source attribution]     ║
║                                                          ║
║  DOUBT: "GH timing relative to insulin window unclear"   ║
║  EVIDENCE: [clone's finding with source attribution]     ║
║                                                          ║
║  Integrate this evidence into your analysis. Where it    ║
║  confirms your prior reasoning, strengthen your claims.  ║
║  Where it contradicts, revise.                           ║
╚══════════════════════════════════════════════════════════╝
```

This is the most targeted data injection possible — evidence found specifically to resolve the worker's own stated doubts, by an entity that shares the worker's full analytical context.

### Context Isolation

Critical: The unleashed clone is a SEPARATE agent instance. It shares the worker's conversation as initialization context but runs independently. Its tool calls, search results, and intermediate reasoning DO NOT flow back into the real worker's context. Only the final findings (stored in the ConditionStore) reach the worker, mediated by the data package.

The worker remains tool-free. The clone does the dirty work.

### Harsh Counterpoints

**Cost**: Each unleashed clone runs a full Strands Agent loop with tool calls. 5 clones × 10 tool calls each × 30 waves = 1500 tool calls per run. At 60 runs over 24 hours, that's 90,000 tool calls. Not free.

**Diminishing returns**: After wave 5+, the clone's doubts become increasingly niche. "Need bloodwork data showing hematocrit at exactly 300mg/week over exactly 8 weeks" — this specific data may not exist anywhere. The clone wastes tool calls searching for unfindable data.

**Context divergence**: The clone starts from the worker's context but immediately diverges as it processes search results. After 10 tool calls, the clone's context is materially different from the worker's. Its relevance judgments may not match what the worker would actually find useful.

**Mitigation**: Budget the clone. Max 5 tool calls per unleash. Max 3 doubts investigated per wave. Prioritize doubts with the highest expected information gain (doubts referenced by multiple other findings, doubts at cross-domain boundaries).

---

## Invariants

1. **Workers are tool-free.** They receive data packages and reason. Period.
2. **The data package is a conversation, not a dump.** Framing matters as much as content. Material is presented with questions that provoke angle-specific reasoning.
3. **No truncation. Ever.** The orchestrator controls volume via relevance ranking and `max_results`. No individual atom is cut off.
4. **Every operation is a row.** Full audit trail DAG connected by `parent_id`.
5. **Clones are disposable copies, workers are sacred.** The real worker's context is never modified by external operations.
6. **RAG graduates, not replaces.** Keyword RAG for bootstrap (wave 2), graph-structured retrieval for steady state (wave 3+), cross-expert for deep discovery (wave 5+).
7. **The kindling pattern.** Cross-angle material is always framed with explicit instructions to interpret through the worker's lens. The framing activates domain expertise on foreign material.
8. **Compaction decisions are data.** Every dedup judgment stored with lineage — the system can audit its own store hygiene.
9. **Context window managed, not ignored.** Rolling distillation prevents clone overflow. Data package budgets respect model limits.
10. **The system improves over time.** Deeper clones → better data packages → deeper reasoning → deeper clones. The bootstrap loop is the engine of improvement.
11. **Typed edges, not generic links.** Every relationship in the graph has a type (supports, contradicts, triangulates_with, answers, etc.) borrowed from the qualitative research ontology. Retrieval is graph traversal, not scanning.
12. **Clones type relationships, not relevance.** The clone's catalogue work produces typed edges (stored once, queried forever), not binary relevance scores (computed per wave, discarded).
13. **Unleashed clones resolve doubts, not explore randomly.** When a clone gets tools, it attacks specific uncertainties from its worker's transcript — targeted research, not open-ended exploration.
