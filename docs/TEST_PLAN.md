# Test Plan: Validating Backlog Assumptions

> Every backlog issue rests on at least one untested assumption. This
> document maps each assumption to a concrete test, classifies tests by
> what they need to run, and defines pass/fail criteria.

## Assumption Registry

### A. Store & Observability (Stage 0)

| ID | Assumption | Backlog Issue | Test Type | Status |
|----|-----------|---------------|-----------|--------|
| A1 | `emit_metric()` persists correctly and excludes from research queries | #201 | Unit (mocked) | ✓ Covered by `test_observability.py` |
| A2 | `store_health_snapshot()` captures accurate distribution | #201 | Unit (mocked) | ✓ Covered |
| A3 | Corpus fingerprinting prevents re-ingestion of identical text | #190 | Unit (mocked) | ✓ Covered |
| A4 | Convergence counts only worker-generated findings, not raw ingestion | #191 | Unit (mocked) | ✓ `test_error_resilience.py::TestConvergenceExcludesRawIngestion` |
| A5 | Source provenance propagates through admit → store_finding → worker | #192 | Unit (mocked) | Partially covered |

### B. Error Resilience (PR #206)

| ID | Assumption | Test Type | Status |
|----|-----------|-----------|--------|
| B1 | Worker crash doesn't abort the wave — other workers' findings preserved | Unit (mocked agent) | ✓ `test_error_resilience.py::TestWorkerCrashIsolated` |
| B2 | Worker timeout fires and the wave continues with partial data | Unit (mocked hang) | ✓ `test_error_resilience.py::TestWorkerTimeoutFires` |
| B3 | Serendipity failure doesn't prevent report generation | Unit (mocked) | ✓ `test_error_resilience.py::TestSerendipityCrashDoesntBlockReport` |
| B4 | Report generation failure returns partial store summary | Unit (mocked) | ✓ `test_error_resilience.py::TestReportCrashReturnsPartial` |
| B5 | Angle detection failure falls back to section titles | Unit (mocked) | ✓ `test_error_resilience.py::TestAngleDetectionFallback` |
| B6 | Observability methods never crash the pipeline (emit_metric, health_snapshot) | Unit (mocked) | ⚠️ `test_error_resilience.py::TestEmitMetricFailureNonfatal` — **per-wave emit_metric is NOT wrapped; currently fatal. health_snapshot IS safe.** |
| B7 | WAL mode is enabled for file-backed DuckDB connections | Unit | ✓ `test_error_resilience.py::TestWALModeFileBacked` |

### C. Pipeline Mechanics (Stage 1–3)

| ID | Assumption | Backlog Issue | Test Type | Status |
|----|-----------|---------------|-----------|--------|
| C1 | Multiple workers writing to store concurrently don't corrupt data | #184 | Unit (threaded) | ✓ `test_error_resilience.py::TestConcurrentStoreWrites` |
| C2 | Compaction removes true duplicates without losing unique findings | #186 | Unit (mocked) | ✓ `test_error_resilience.py::TestCompactionPreservesUnique` |
| C3 | Convergence detection stops the run when growth rate drops | #191 | Integration (LLM) | ✓ `test_pipeline_integration.py::TestConvergenceBehavior` |
| C4 | Data package assembly (get_corpus_section + get_peer_insights + get_research_gaps) returns complete, non-overlapping data | #203 | Unit (mocked) | ✓ `test_pipeline_integration.py::TestDataPackageAssembly` |

### D. Quality Validation (Stage 2+, requires real LLM)

| ID | Assumption | Backlog Issue | Test Type | Status |
|----|-----------|---------------|-----------|--------|
| D1 | Angle detection produces orthogonal domain-specific angles, not "Part 1-5" | #187 | Integration (LLM) | ✓ `test_pipeline_integration.py::TestAngleDetectionQuality` |
| D2 | Workers explore their full corpus section (not stop at 3%) | — | Integration (LLM) | Partially covered by E2E smoke test |
| D3 | Workers store SPECIFIC evidence-backed findings, not summaries | — | Integration (LLM) | ✓ `test_pipeline_integration.py::TestFindingsAreSpecific` |
| D4 | Serendipity wave produces genuine cross-domain connections | — | Integration (LLM) | ✓ `test_pipeline_integration.py::TestSerendipityProducesConnections` |
| D5 | Report generation synthesizes store findings, not just the prompt | — | Integration (LLM) | ✓ `test_pipeline_integration.py::TestReportSynthesizesFindings` |
| D6 | Quality doesn't degrade over multiple waves | — | Integration (LLM) | ✓ `test_architecture_validation.py::TestLongRunSimulation` |
| D7 | Larger data packages (30K chars) produce more unique insights than 6K truncated | MODEL_SELECTION.md | Integration (LLM) | ✓ `test_architecture_validation.py::TestDataPackageSizeMatters` |

---

## Test Tiers

### Tier 1: Mechanical Resilience (no LLM, runs instantly)

Tests that validate the pipeline handles failures correctly. These
use mocked LLM responses and intentional failures.

**What we prove:** The pipeline won't crash during a 24h run due to
transient errors, timeouts, or corrupted responses.

Tests:
- `test_worker_crash_isolated` — inject Exception in one worker, verify
  other workers' findings are preserved in the store
- `test_worker_timeout_fires` — mock a worker that hangs forever,
  verify the wave completes within `worker_timeout_s + margin`
- `test_serendipity_crash_doesnt_block_report` — inject Exception in
  serendipity, verify report is still generated
- `test_report_crash_returns_partial` — inject Exception in report
  generation, verify partial summary returned
- `test_angle_detection_fallback` — inject Exception in LLM angle
  detection, verify section titles used as fallback
- `test_emit_metric_failure_nonfatal` — corrupt the DB connection
  for emit_metric, verify synthesize() still returns
- `test_wal_mode_file_backed` — create file-backed store, verify
  WAL journal mode is active
- `test_concurrent_store_writes` — 10 threads writing findings
  simultaneously, verify no data loss or corruption
- `test_convergence_excludes_raw_ingestion` — ingest corpus + admit
  worker findings, verify convergence count only sees worker findings
- `test_compaction_preserves_unique` — insert duplicates + uniques,
  run compact, verify uniques preserved and duplicates removed

### Tier 2: Pipeline Integration (needs real LLM endpoint)

Tests that validate the pipeline produces meaningful output end-to-end.
Uses OpenRouter/Groq/Together with a small model as the LLM backend.

**What we prove:** The pipeline actually works — workers explore corpus,
store findings, and produce a coherent report.

Tests:
- `test_full_pipeline_3_waves` — small corpus, 3 workers, 3 waves,
  verify: findings stored > 0, report generated, metrics populated
- `test_angle_detection_quality` — feed a multi-topic corpus, verify
  angles are domain-specific (not "Part 1", "Section 2")
- `test_worker_exploration_depth` — check that workers call
  `get_corpus_section` with increasing offsets (not just offset=0)
- `test_findings_are_specific` — verify stored findings contain
  numbers, dosages, or citations (not vague summaries)
- `test_convergence_behavior` — run 5 waves, verify findings_per_wave
  is decreasing (workers find less new stuff each wave)
- `test_serendipity_produces_connections` — verify serendipity worker
  stores findings that reference multiple angles

### Tier 3: Architecture Validation (needs real LLM, intensive)

Tests that validate the deeper architectural assumptions in the backlog.
These are expensive (many LLM calls) and take 10-30 minutes.

**What we prove:** The design decisions in the backlog are correct.

Tests:
- `test_data_package_size_matters` — run same corpus with
  max_return_chars=6000 vs 30000, compare unique insights ratio.
  **Pass:** 30K produces ≥15% more unique findings
- `test_multi_run_dedup` — run pipeline twice on same corpus, verify
  second run skips ingestion AND still finds new findings from
  cross-referencing
- `test_24h_simulation` — 10 sequential runs with same store, verify
  store doesn't corrupt and findings don't degrade (store growth
  monitored via health_snapshot)

---

## Test Infrastructure

### LLM Endpoint Options

Available API keys for integration tests:
- `OPENROUTER_API_KEY` — OpenRouter (55+ models, OpenAI-compatible)
- `GROQ_API_KEY` — Groq (fast inference, good for CI)
- `TOGETHER_API_KEY` — Together AI
- `FIREWORKS_API_KEY` — Fireworks AI

**Recommended for testing:** OpenRouter with a small, fast uncensored
model. The pipeline uses `OpenAIModel` from Strands which takes any
OpenAI-compatible endpoint.

```python
MCPSwarmConfig(
    api_base="https://openrouter.ai/api/v1",
    model="meta-llama/llama-3.1-8b-instruct",  # fast, cheap, uncensored enough for testing
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_tokens=2048,
    max_workers=3,
    max_waves=2,
)
```

### Test Corpus

A small but realistic multi-topic corpus for integration tests:

```python
TEST_CORPUS = """
## Insulin Timing and Nutrient Partitioning

Rapid-acting insulin (Humalog/NovoRapid) peaks at 60-90 minutes post-injection.
Pre-workout dosing at 4-6 IU with 10g dextrose per IU prevents hypoglycemia.
Humulin-R has a slower onset (30-60 min) and longer tail (6-8 hours).
Post-workout insulin drives amino acids into muscle tissue via GLUT4 upregulation.
Berberine at 500mg mimics some insulin-sensitizing effects via AMPK activation.

## Hematological Effects of Anabolic Compounds

Trenbolone acetate increases erythropoietin (EPO) production, raising hematocrit
by 15-20% over 8-12 weeks. At 400mg/week, hematocrit commonly reaches 52-54%.
Boldenone (EQ) at 300-600mg/week elevates RBC count through a different mechanism
— direct bone marrow stimulation rather than EPO.
Regular phlebotomy (500mL every 8 weeks) manages polycythemia.
Naringin (grapefruit extract) may modestly reduce hematocrit by 2-3 points.

## Growth Hormone and IGF-1 Cascade

Exogenous GH at 2-4 IU/day increases hepatic IGF-1 production within 6 hours.
Splitting doses (AM + pre-bed) mimics natural pulsatile secretion.
GH + insulin synergy: insulin prevents GH-induced insulin resistance while
GH amplifies insulin's anabolic effects through IGF-1 mediation.
MK-677 (ibutamoren) at 25mg/day raises GH by 40-60% but causes water retention.
CJC-1295/Ipamorelin combination provides more physiological GH release patterns.
"""
```

### Pass/Fail Criteria

Each test has explicit pass criteria. A test is NOT marked as passed
unless ALL criteria are met. Partial passes are logged as findings.

---

## Execution Order

1. **Run Tier 1 immediately** — no dependencies, validates error handling
2. **Run Tier 2 after Tier 1 passes** — validates pipeline mechanics
3. **Run Tier 3 only after Tier 2 confirms pipeline works** — validates
   architectural decisions
4. **Log all results** in a structured format for future reference

---

## What Each Test Validates in the Backlog

| Test | Validates | If FAILS, what changes |
|------|----------|----------------------|
| worker_crash_isolated | PR #206 hardening | Error handling is broken, fix before 24h run |
| worker_timeout_fires | PR #206 worker_timeout_s | Hung workers will block waves indefinitely |
| concurrent_store_writes | Thread safety assumption | Need store-level locking redesign |
| angle_detection_quality | #187 prompt-derived angles | Need preset angles, can't rely on LLM |
| findings_are_specific | Worker prompt quality | System prompt needs rework |
| convergence_behavior | #191 convergence detection | Threshold or counting logic is wrong |
| data_package_size_matters | MODEL_SELECTION.md token budget recs | 6K may be sufficient, don't need GLM-5.1's 200K |
| 24h_simulation | Overall architecture viability | Fundamental design issues |
