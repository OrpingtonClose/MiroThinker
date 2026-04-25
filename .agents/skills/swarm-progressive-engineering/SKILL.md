# Progressive Engineering Plan for Swarm Hardening

Knowledge about the cone-of-probability approach to the swarm backlog.

## Core Principle

Don't build a flat backlog. Build a **decision tree** where each stage is an instrumented experiment with pass/fail criteria that determine what comes next. Observability is the backbone — every stage emits metrics that inform the next stage's direction.

## Stage Dependency Chain

```
Stage 0: Observability Layer (#201) + merge fixes (#190, #191, #192)
    ↓ baseline metrics reveal dominant bottleneck
Stage 1: Fix top bottleneck (angles #187 or corpus streaming #188)
    ↓ did the fix improve metrics >20%?
Stage 2: Prove clone pattern (#202) — CRITICAL FORK
    ↓ clone precision > keyword RAG by ≥15%? If NO → pivot architecture
Stage 3: Tool-free workers + data packages (#184, #194, #203)
    ↓ A/B test: tool-free ≥ tool-calling on quality?
Stage 4: Graph ontology (#185, #198, #199)
    ↓ graph density > 0.5 edges/finding?
Stage 5: 24-hour scale (#193, #195, #188, #204)
    ↓ stable for 24h with steady quality?
Stage 6: On-demand only (#186, #200)
    ↓ only if metrics trigger (store >500K rows, high gap density)
```

## Key Decision Points

- **Stage 2 is the critical fork.** If clones don't outperform keyword RAG, everything from Stage 3 onward changes. Test this BEFORE building infrastructure.
- **Stage 3 A/B test.** Tool-free workers are theoretically cleaner but empirically unproven. Same corpus, same model, same angle — compare output quality.
- **Stage 6 items are demand-gated.** Compaction (#186) only when store >500K rows. Unleashed clones (#200) only when gap density is high.

## Closed/Redundant Issues

- #183 (truncation removal) — subsumed by #184 (tool-free workers) + #199 (graph assembly)
- #196 (cross-domain connections) — subsumed by #198 (QR ontology `triangulates_with` edges)
- #189 (API key safety) — irrelevant for local-only vLLM runs

## Architecture Docs

- `docs/STORE_ARCHITECTURE.md` — Audit trail DAG, tool-free workers, cloned context as Flock backend
- `docs/CLONED_CONTEXT_PATTERN.md` — 13 ramifications of expert persistence via KV cache
- `docs/SWARM_WAVE_ARCHITECTURE.md` — Wave-by-wave spec, data package structure, QR ontology, unleashed clones

## Cloned-Context Pattern (Key Innovation)

Workers are tool-free and receive structured data packages. Their conversation transcripts are captured by a Strands hook observer. The orchestrator registers these transcripts as named models in Flock (DuckDB LLM extension). Flock queries route through a vLLM session proxy that prepends the cloned worker context. vLLM prefix-caches the context so subsequent queries are fast.

This means:
- The clone has the worker's full reasoning loaded
- Clone judges analytical relevance, not keyword match
- RAG graduates from keyword → clone-scored → cross-expert → meta-expert over waves
- Expert persistence across runs via serialized conversation transcripts

## QR Ontology for Graph-Structured RAG

Instead of flat retrieval + llm_filter, use typed edges from the qualitative research tradition:
- `supports`, `contradicts`, `triangulates_with`, `answers`, `compares`, `reflects_on`, `codes`, `precedes`
- Entity types: `theme`, `code`, `codeGroup`, `memo`, `researchQuestion`, `quote`
- Data package assembly becomes graph traversal (O(1) SQL queries) instead of scanning (O(N) llm_filter)
- Clone's job shifts from "is this relevant?" to "HOW does this relate?"

## Metric Row Types

| Row Type | Emitted By | Contents |
|---|---|---|
| `wave_metric` | MCPSwarmEngine per wave | findings_new, elapsed_s, tool_calls, worker_results |
| `worker_metric` | Both engines per worker | angle, input/output chars, findings_stored, errors |
| `store_metric` | store_health_snapshot() | total/active/obsolete rows, rows by type/angle |
| `run_metric` | Both engines end of run | full run summary, phase_times, convergence_reason |
| `corpus_fingerprint` | ingest_raw() | SHA-256 hash in source_ref for dedup |
