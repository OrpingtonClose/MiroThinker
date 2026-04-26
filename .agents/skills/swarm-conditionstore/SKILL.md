# ConditionStore & Swarm Architecture

Knowledge about the DuckDB-backed ConditionStore and how the swarm engines interact with it.

## ConditionStore Location & Setup

- Implementation: `apps/strands-agent/corpus.py`
- In-memory by default (`:memory:`), file-backed via `ConditionStore(db_path="path.duckdb")`
- Thread-safe via `threading.Lock` (`self._lock`)
- Schema auto-migrates: `_ensure_lineage_columns()` runs `ALTER TABLE ADD COLUMN IF NOT EXISTS` idempotently in `__init__`

## Key Schema Columns

| Column | Purpose |
|---|---|
| `id` | Auto-incrementing primary key |
| `fact` | The actual content (finding text, metric JSON, etc.) |
| `row_type` | Discriminator: `finding`, `thought`, `insight`, `synthesis`, `raw`, `wave_metric`, `worker_metric`, `store_metric`, `run_metric`, `corpus_fingerprint` |
| `source_type` | Origin: `worker_analysis`, `researcher`, `observability`, `system`, `corpus_section` |
| `consider_for_use` | Boolean — FALSE excludes from research queries (metrics, raw, fingerprints) |
| `angle` | Research angle (e.g., `insulin_timing`, `hematology`) |
| `parent_id` | Lineage DAG edge → parent row |
| `related_id` | Cross-reference edge |
| `confidence` | 0.0–1.0 confidence score |
| `source_model` | Which LLM model produced this row |
| `source_run` | Run identifier (`run_YYYYMMDD_HHMMSS`) for cross-run queries |
| `iteration` | Wave/gossip round number |
| `phase` | Swarm phase name (`wave_1`, `gossip_round_2`, etc.) |

## Core Methods

- `admit()` — Insert a single condition row. Accepts `source_model`/`source_run` for provenance.
- `ingest_raw()` — Chunk-aware ingestion: raw → paragraph findings. SHA-256 fingerprint dedup.
- `emit_metric()` — Persist observability data as JSON in `fact` column with `consider_for_use=FALSE`.
- `store_health_snapshot()` — Query row distribution by type/angle, persist as `store_metric`.
- `get_findings()` — Research query: returns rows where `consider_for_use=TRUE`.
- `export_for_swarm()` — Export findings for swarm consumption.

## Two Swarm Engines

### MCPSwarmEngine (`swarm/mcp_engine.py`)
- Workers are Strands Agents with ConditionStore MCP tools
- Workers call `store_finding`, `search_corpus`, `get_peer_insights` etc.
- Convergence: counts worker-generated findings (not raw rows)
- Observability: emits `wave_metric`, `worker_metric`, `store_metric`, `run_metric`
- Run ID generated at start of `synthesize()`: `run_YYYYMMDD_HHMMSS`

### GossipSwarm (`swarm/engine.py`)
- Workers receive prompt-injected data (no tools)
- Gossip rounds with adaptive convergence (Jaccard similarity)
- Uses `lineage_store` (duck-typed ConditionStore) for persistence
- Observability: emits `run_metric` + `worker_metric` if lineage_store supports `emit_metric`

## Worker Tools (`swarm/worker_tools.py`)

- Factory pattern: `build_worker_tools(store, worker_angle, worker_id, phase, source_model, source_run)`
- Tools are closures over the store and worker identity
- `store_finding` uses `admit()` (not raw SQL) to propagate provenance
- Thread-safe finding counter per worker

## Testing

```bash
# Run observability tests
PYTHONPATH=apps/strands-agent:swarm:. python -m pytest tests/test_observability.py -v

# Run all swarm-related tests
PYTHONPATH=apps/strands-agent:swarm:. python -m pytest tests/ -v
```

## Metric Queries

```sql
-- Wave progression for a run
SELECT iteration, json_extract(fact, '$.findings_new') AS new_findings,
       json_extract(fact, '$.elapsed_s') AS seconds
FROM conditions WHERE row_type = 'wave_metric'
  AND source_run = 'run_20260416_120000'
ORDER BY iteration;

-- Cross-run comparison
SELECT source_run, json_extract(fact, '$.total_findings_stored') AS findings,
       json_extract(fact, '$.total_elapsed_s') AS elapsed
FROM conditions WHERE row_type = 'run_metric';

-- Store health over time
SELECT source_run, iteration,
       json_extract(fact, '$.total_rows') AS rows,
       json_extract(fact, '$.active_rows') AS active
FROM conditions WHERE row_type = 'store_metric';

-- Worker productivity
SELECT json_extract(fact, '$.angle') AS angle,
       json_extract(fact, '$.findings_stored') AS stored,
       json_extract(fact, '$.tool_calls') AS calls
FROM conditions WHERE row_type = 'worker_metric'
  AND source_run = 'run_20260416_120000';
```
