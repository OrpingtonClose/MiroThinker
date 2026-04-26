# Framework Recommendation: Universal Store Architecture

**Date:** 2026-04-26  
**Synthesis of:** 8 parallel subagent analyses (strands, LangGraph, custom, DuckDB FlockMTL, ADK, active orchestrator, recursive loop patterns, migration cost)  
**Decision:** No single framework. Hybrid evolution path.

---

## Executive Summary

After analyzing all five candidate frameworks and two cross-cutting concerns (active orchestrator requirements, recursive loop patterns), the verdict is unanimous:

> **Evolve `strands-agent` as the foundation. Build a lightweight actor-event orchestrator on top. Transplant ADK's algorithm battery. Use LangGraph only for planning. Adopt DuckDB FlockMTL for embeddings only.**

This is the only path that:
- Starts from ~75% working code (strands-agent)
- Preserves vLLM prefix caching (Flock cost optimization #1)
- Handles nested recursion, user interrupts, and dynamic spawning
- Can deliver a working recursive loop in **3–4 weeks**
- Keeps the door open to a fully custom core in months 4–6

---

## The Candidate Scorecard

| Framework | Fit Score | Role in Final Architecture | Blockers |
|-----------|-----------|---------------------------|----------|
| **strands-agent** | 7.5/10 | **Foundation** — store, tools, swarm bridge, MCP integrations, FastAPI server | Living loop 2/10, no state machine |
| **LangGraph** | 6/10 | **Planning agent only** — decision layer, DAG-shaped flows | Cannot spawn sub-graphs at runtime, static topology, HITL is blocking |
| **Custom actor-event** | 8.5/10 | **Recursive loop execution** — 9-state machine, priority queue, rule engine | 3–4 weeks dev cost |
| **ADK-agent (salvage)** | N/A | **Algorithm battery donor** — 13-step corpus pipeline, maestro, dashboard, thought swarm | Do NOT revive ADK (asyncio blocking) |
| **DuckDB FlockMTL** | 2.7/10 | **Augmentation only** — embeddings, hybrid search | 167× cost increase without prefix caching |

*Fit score reflects how well the framework handles the full Universal Store Architecture, not the quality of the framework itself.*

---

## The Five Principles of the Decision

### 1. Start From Working Code

The migration cost analysis found that **~75% of the Universal Store vision is already implemented in strands-agent**:

- Store layer: ~90% complete (ConditionStore, 1,800 LOC, 20+ proposed indices)
- Orchestrator: ~75% complete (AsyncTaskPool, swarm bridge, MCP tool dispatch)
- API/server: ~95% complete (FastAPI, streaming SSE, health checks)
- Missing: store bridges, scheduler, indices, unified entry point

A custom from-scratch build would take **12–20 weeks**. Reviving ADK would take **6–10 weeks** and carries the highest risk (it was abandoned for a reason). Evolving strands-agent takes **3–4 weeks**.

### 2. Preserve the Cost Optimization That Matters

The #1 cost optimization in the entire architecture is **vLLM prefix caching** in `flock_query_manager.py`. FlockMTL abandons this, causing a **167× increase in prompt token costs**. The Python FlockQueryManager must be kept exactly as-is. Any framework that forces migration away from it is disqualified.

### 3. The Recursive Loop Is Not a Graph Problem

LangGraph, agent graphs, and dataflow frameworks all assume:
- Static or near-static topology
- Homogeneous node types (mostly LLM calls)
- Shared monolithic state dict
- Global recursion limits

The Universal Store loop is:
- **Dynamically spawning** (helper agents for PDFs, new angles, research directions)
- **Heterogeneous parallelism** (vLLM prefix-cached inference + async HTTP + CPU-bound clustering)
- **Nested recursion** (outer swarm→flock loop, inner gossip loop, inner diffusion manifest→confront→correct loop)
- **Event-driven convergence** (scores, info_gain, gap detection — not iteration counts)
- **User-interruptible** at any layer

Only the **actor model** handles all five of these natively. The recommended hybrid (actor structure + event loop execution + rule engine policy) scored highest across every dimension of the recursive loop analysis.

### 4. The Devin Operator Must Become Code

The active orchestrator requirements codified **12 behaviors** the human operator performed in the Devin transcript:

1. Probe → validate → decide before any action
2. GPU waste detection and remediation
3. Subagent spawning for specialized tasks
4. Model censorship checks and rerouting
5. Cost tracking with real-time dashboards
6. Destructive action confirmation gates
7. Multi-agent consultation for ambiguous decisions
8. Economic guardrails (budget caps, rate limits)
9. Context accumulation across long sessions
10. Error recovery with fallback strategies
11. Real-time priority reordering
12. Cross-system health monitoring

These are not behaviors any existing framework implements. They require a **custom rule engine** sitting above the actor layer. This is why "use LangGraph for everything" fails — LangGraph has no concept of GPU waste, cost tracking, or destructive action gates.

### 5. Transplant, Don't Revive

ADK-agent was abandoned because Google ADK's asyncio handling blocked the event loop. But ADK-agent had **four superior components** that strands-agent lacks:

| Component | ADK File | What It Did | How to Transplant |
|-----------|----------|-------------|-------------------|
| Algorithm battery | `corpus_store.py` | 13-step pipeline: scoring, contradiction detection, union-find clustering, narrative chains, redundancy compression, quality gates | Port logic to new `strands-agent/corpus/` module, replace ADK callbacks with async functions |
| Maestro pattern | `maestro_sql.py` | Free-form SQL conductor with semantic templates | Port SQL template engine; use strands-agent's ConditionStore |
| Dashboard | `dashboard_sse.py` | Real-time SSE KPI streaming, HTML reports, `/corpus/stats` endpoint | Port to FastAPI; reuse strands-agent's existing SSE infrastructure |
| Thought swarm | `thought_swarm.py` | Parallel specialist thinkers, arbitration, cross-angle serendipity, multi-signal convergence | Port arbitration logic; use strands-agent's AsyncTaskPool |

**Verdict:** Do not import the ADK framework. Extract these four components' logic and rewrite them as async-native modules inside strands-agent.

---

## The Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                │
│         (FastAPI, SSE streaming, health checks, dashboard)           │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED ENTRY POINT                              │
│              `orch.run(query)` — one-liner API                       │
│         Accepts: user query, file uploads, interrupt signals         │
│         Returns: stream of OrchestratorEvent objects                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                     POLICY LAYER (Rule Engine)                       │
│  12 operator behaviors codified as IF-THEN rules on store state     │
│  Rules: probe→validate→decide, GPU waste, cost gates,               │
│         destructive action confirmation, convergence detection       │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                   ACTOR SUPERVISION TREE                             │
│                                                                      │
│  OrchestratorActor (root)                                            │
│  ├── SwarmSupervisor                                                 │
│  │   ├── BeeActor (angle 1)                                          │
│  │   ├── BeeActor (angle 2)                                          │
│  │   ├── ...                                                          │
│  │   └── DiffusionSupervisor                                         │
│  │       ├── ManifestWorker                                           │
│  │       ├── ConfrontWorker                                           │
│  │       └── CorrectWorker                                           │
│  ├── FlockSupervisor                                                 │
│  │   ├── CloneActor (angle 1)  ← vLLM prefix-cached                 │
│  │   ├── CloneActor (angle 2)                                        │
│  │   └── ...                                                          │
│  ├── McpResearcherActor                                              │
│  │   └── ToolActor (per MCP tool)                                    │
│  └── UserProxyActor                                                  │
│      └── Interrupt buffer, question queue, resume signals            │
│                                                                      │
│  Implementation: asyncio.Task + asyncio.Queue mailbox per actor      │
│  Supervision: parent catches child exceptions, restarts or escalates │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                    PLANNING AGENT (LangGraph)                        │
│  One LangGraph CompiledStateGraph for high-level decision-making:    │
│  "Should we do more research, synthesize, or ask the user?"          │
│  NOT used for: recursive loop execution, dynamic spawning,           │
│  swarm gossip rounds, Flock convergence detection                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                    EXECUTION SUBSTRATE (asyncio)                     │
│  Native Python asyncio event loop. Already used in:                  │
│  - engine.py (synthesize, gossip rounds)                             │
│  - flock_query_manager.py (clone queries, MCP interleaving)          │
│  - task_tools.py (AsyncTaskPool, concurrent tool dispatch)           │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER (DuckDB)                             │
│  ConditionStore (1,800 LOC) + proposed indices + migration SQL       │
│  FlockMTL used ONLY for: embedding generation, hybrid search         │
│  Python FlockQueryManager kept EXACTLY AS-IS for prefix caching      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap: Phase 1 (Weeks 1–4)

### Week 1: P0 Blockers — Store Bridge + Indices

**Goal:** Make the existing code actually populate the store correctly.

| Task | File(s) | LOC | Deliverable |
|------|---------|-----|-------------|
| Fix `GossipSwarm.synthesize()` bridge | `engine.py` | ~100 | Write `finding` rows (not `thought` rows) with `gradient_growth`, `gradient_decay`, `gradient_surprise` flags |
| Populate `cluster_id` | `corpus_store.py` (ported) | ~80 | Union-find clustering on semantic similarity; write `cluster_id` to `finding` rows |
| Populate `contradiction_flag` | `corpus_store.py` (ported) | ~60 | Contradiction detection scoring; flag `finding` rows above threshold |
| Add indices | migration SQL | ~30 | 20+ indices on `type`, `cluster_id`, `contradiction_flag`, `created_at`, `score`, etc. |
| Diffusion queen persistence | `queen.py` | ~80 | Write intermediate diffusion artifacts (manifest, confront, correct, stitch) as `finding` rows with `meta` JSON |

**End of Week 1:** The store contains real, queryable data. BRIDGE, SYNTHESIZE, and ADJUDICATE queries return non-zero rows.

### Week 2: Actor Model Scaffold + Scheduler

**Goal:** Replace the "diagram, not code" recursive loop with a running scheduler.

| Task | File(s) | LOC | Deliverable |
|------|---------|-----|-------------|
| Actor base class | `actors/base.py` | ~60 | `asyncio.Queue` mailbox, `send()`, `_run()` loop, graceful shutdown |
| Supervision tree | `actors/supervisor.py` | ~80 | Parent→child spawn, exception catching, restart policy (max 3 restarts) |
| SwarmSupervisor | `actors/swarm.py` | ~120 | Spawns BeeActors, collects results, emits `SwarmComplete` |
| FlockSupervisor | `actors/flock.py` | ~100 | Spawns CloneActors, manages prefix cache lifecycle, emits `FlockComplete` |
| McpResearcherActor | `actors/mcp.py` | ~80 | Queues MCP calls, respects rate limits, emits `ResearchComplete` |
| UserProxyActor | `actors/user.py` | ~60 | Buffers interrupts, handles pause/resume/inject |
| Scheduler daemon | `scheduler.py` | ~100 | Priority event queue, rule engine hook, health monitor, auto-pause on resource exhaustion |

**End of Week 2:** `python -m strands-agent.scheduler` starts a daemon that runs swarm→flock→swarm in a loop, pauses on user interrupt, and resumes on new data.

### Week 3: ADK Algorithm Battery Transplant

**Goal:** Port the four superior ADK components into strands-agent.

| Task | Source | Target | LOC | Deliverable |
|------|--------|--------|-----|-------------|
| Corpus algorithm battery | `adk-agent/corpus_store.py` | `strands-agent/corpus/` | ~600 | 13-step pipeline: scoring, clustering, contradiction, narrative chains, redundancy, quality gates |
| Maestro SQL conductor | `adk-agent/maestro_sql.py` | `strands-agent/maestro.py` | ~300 | Free-form SQL with semantic templates, runs against ConditionStore |
| Dashboard | `adk-agent/dashboard_sse.py` | `strands-agent/dashboard/` | ~400 | Real-time SSE KPIs, HTML reports, `/corpus/stats` endpoint |
| Thought swarm arbitration | `adk-agent/thought_swarm.py` | `strands-agent/swarm/` | ~300 | Parallel specialist thinkers, cross-angle serendipity, multi-signal convergence |

**End of Week 3:** The swarm is no longer a simple ReAct agent. It runs the full 13-step corpus pipeline with real-time observability.

### Week 4: Integration, Testing, Unified Entry Point

**Goal:** Glue everything together. One-liner API works end-to-end.

| Task | File(s) | LOC | Deliverable |
|------|---------|-----|-------------|
| Unified orchestrator | `orchestrator_universal.py` | ~200 | `orch.run(query)` — accepts query, returns event stream |
| Rule engine MVP | `rules.py` | ~150 | 7 must-have rules: convergence detection, external research trigger, user interrupt, destructive action gate, cost cap, GPU waste, auto-pause |
| Cross-run federation loader | `federation.py` | ~100 | Load `finding` rows from prior runs into current run context |
| End-to-end test | `tests/test_universal_loop.py` | ~200 | Full swarm→flock→external→reswarm cycle, assert store state |
| Documentation | `ARCHITECTURE.md`, `OPERATORS_GUIDE.md` | ~500 | How the orchestrator works, how to add rules, how to debug |

**End of Week 4:** The system is demonstrable. `orch.run("Analyze Q4 earnings reports")` triggers: swarm extraction → gossip convergence → Flock evaluation → gap detection → MCP research → new swarm cycle → final synthesis → dashboard report.

---

## What We Are NOT Doing

| Rejected Path | Why |
|---------------|-----|
| **Revive ADK-agent** | Abandoned due to asyncio event-loop blocking. High risk, 6–10 weeks, no guarantee the blocking issue is fixable. |
| **Rewrite Flock in FlockMTL** | 167× cost increase without vLLM prefix caching. Loses adaptive budget, asymmetric scoring, MCP interleaving. |
| **Use LangGraph for recursive loop** | Static topology, cannot spawn sub-graphs at runtime, global recursion limit shared across nesting levels, HITL is stop-the-world blocking. |
| **Build custom core from scratch now** | 12–20 weeks. We can get to a working recursive loop in 3–4 weeks by evolving strands-agent, then migrate to a custom core in months 4–6 if needed. |
| **CrewAI / AutoGen / other frameworks** | Not audited, but all share the same fundamental limitations as LangGraph for this architecture (static graphs, homogeneous agents, no nested recursion). |

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Actor model complexity exceeds team bandwidth | Medium | Start with 350–450 LOC of custom actor code, not a full framework. Use `asyncio` primitives everyone knows. |
| ADK transplant introduces bugs | Medium | Port one component at a time. Keep original ADK files as reference. Write tests before porting. |
| vLLM prefix caching broken by actor refactor | Low | FlockQueryManager is untouched. Only the *caller* (FlockSupervisor) changes. CloneActors still call `manager.query()` exactly as before. |
| Store schema migration is painful | Low | Migration SQL is additive only. No columns dropped. Rollback is `DROP TABLE` + recreate from backup. |
| Scheduler runs away (infinite loop) | Medium | Rule engine enforces max iterations, cost caps, and convergence thresholds. Health monitor auto-pauses on GPU/memory exhaustion. |
| User interrupts leave system in inconsistent state | Low | Cooperative cancellation: supervisor stops launching new workers, lets in-flight ones finish, stores partial results. Resume re-hydrates from store. |

---

## The Longer View (Months 4–6)

After Phase 1 delivers a working recursive loop, Phase 2 can revisit the framework decision with real operational data:

- **If the asyncio actor model proves sufficient:** Keep it. The 350–450 LOC custom core is maintainable.
- **If we need distributed execution:** Migrate to Ray actors. The supervision tree maps 1:1 to Ray's actor hierarchy.
- **If we need stronger static analysis:** Replace LangGraph planning agent with a custom PDDL solver or constraint programming layer.
- **If FlockMTL matures:** Re-evaluate embedding generation and hybrid search (still not Flock core).
- **If the rule engine grows beyond 500 rules:** Migrate to a proper Rete-algorithm engine (PyKnow, durable-rules).

The hybrid architecture is **designed to be evolvable**. Each layer is swappable without touching the others.

---

## One-Sentence Summary

> **Evolve strands-agent into a hybrid actor-event orchestrator with a transplanted ADK algorithm battery, keep LangGraph for planning only, preserve Python Flock for prefix caching, and deliver a working recursive loop in four weeks.**
