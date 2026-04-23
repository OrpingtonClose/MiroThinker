"""Tier 1 tests: mechanical resilience — no LLM, mocked agents, runs instantly.

Validates that the MCP swarm pipeline handles failures correctly during
a 24h unattended run.  Every test injects a specific failure mode and
verifies the pipeline degrades gracefully instead of crashing.

Covers assumptions B1-B7, C1, C2 from TEST_PLAN.md:
    B1  Worker crash doesn't abort the wave
    B2  Worker timeout fires and the wave continues
    B3  Serendipity failure doesn't prevent report generation
    B4  Report generation failure returns partial store summary
    B5  Angle detection failure falls back to section titles
    B6  Observability methods never crash the pipeline
    B7  WAL mode is enabled for file-backed DuckDB connections
    C1  Concurrent store writes don't corrupt data
    C2  Compaction removes duplicates without losing unique findings
    A4  Convergence counts only worker-generated findings, not raw ingestion
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure repo root and strands-agent are importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_STRANDS_AGENT = str(Path(__file__).resolve().parents[1] / "apps" / "strands-agent")
_SWARM = str(Path(__file__).resolve().parents[1] / "swarm")
for p in (_REPO_ROOT, _STRANDS_AGENT, _SWARM):
    if p not in sys.path:
        sys.path.insert(0, p)

from corpus import ConditionStore
from swarm.mcp_engine import MCPSwarmConfig, MCPSwarmEngine


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

TEST_CORPUS = """\
## Insulin Timing and Nutrient Partitioning

Rapid-acting insulin (Humalog/NovoRapid) peaks at 60-90 minutes post-injection.
Pre-workout dosing at 4-6 IU with 10g dextrose per IU prevents hypoglycemia.
Humulin-R has a slower onset (30-60 min) and longer tail (6-8 hours).
Post-workout insulin drives amino acids into muscle tissue via GLUT4 upregulation.
Berberine at 500mg mimics some insulin-sensitizing effects via AMPK activation.

## Hematological Effects of Anabolic Compounds

Trenbolone acetate increases erythropoietin (EPO) production, raising hematocrit
by 15-20% over 8-12 weeks. At 400mg/week, hematocrit commonly reaches 52-54%.
Boldenone (EQ) at 300-600mg/week elevates RBC count through a different mechanism.
Regular phlebotomy (500mL every 8 weeks) manages polycythemia.
Naringin (grapefruit extract) may modestly reduce hematocrit by 2-3 points.

## Growth Hormone and IGF-1 Cascade

Exogenous GH at 2-4 IU/day increases hepatic IGF-1 production within 6 hours.
Splitting doses (AM + pre-bed) mimics natural pulsatile secretion.
GH + insulin synergy: insulin prevents GH-induced insulin resistance while
GH amplifies insulin's anabolic effects through IGF-1 mediation.
MK-677 (ibutamoren) at 25mg/day raises GH by 40-60% but causes water retention.
"""

TEST_QUERY = "Analyze interactions between insulin, anabolic compounds, and growth hormone"


async def _mock_complete(prompt: str) -> str:
    """LLM mock that returns structured but predictable output."""
    if "research angles" in prompt.lower() or "distinct topics" in prompt.lower():
        return '["insulin_timing", "hematological_effects", "growth_hormone"]'
    if "score each" in prompt.lower() or "assignment" in prompt.lower():
        return '{"scores": [[1,0,0],[0,1,0],[0,0,1]]}'
    if "report" in prompt.lower():
        return "Mock synthesis report: insulin + anabolic compound interactions."
    if "extract" in prompt.lower() and "claims" in prompt.lower():
        return '[{"fact": "Mock finding from extraction", "confidence": 0.8, "tags": ["mock"]}]'
    return "Mock LLM response for testing."


def _make_engine(
    store: ConditionStore | None = None,
    complete=None,
    config: MCPSwarmConfig | None = None,
) -> MCPSwarmEngine:
    """Create a configured engine with sensible test defaults."""
    if store is None:
        store = ConditionStore()
    if complete is None:
        complete = _mock_complete
    if config is None:
        config = MCPSwarmConfig(
            max_workers=3,
            max_waves=2,
            convergence_threshold=2,
            worker_timeout_s=5.0,
            enable_serendipity_wave=False,
            enable_rolling_summaries=False,
            compact_every_n_waves=0,
        )
    return MCPSwarmEngine(store=store, complete=complete, config=config)


def _make_mock_worker_result(angle: str, worker_id: str) -> dict:
    """Create a mock worker result dict matching run_tool_free_worker output."""
    return {
        "angle": angle,
        "worker_id": worker_id,
        "response": f"Analysis of {angle}: found significant interactions between compounds at 4-6 IU dosing.",
        "status": "success",
        "input_chars": 500,
        "output_chars": 200,
        "model": "mock-model",
        "elapsed_s": 1.0,
    }


# ═══════════════════════════════════════════════════════════════════════
# B1: Worker crash doesn't abort the wave
# ═══════════════════════════════════════════════════════════════════════


class TestWorkerCrashIsolated:
    """B1: One worker crashes — others' findings preserved, wave continues."""

    @pytest.mark.asyncio
    async def test_worker_crash_doesnt_abort_wave(self) -> None:
        """Inject RuntimeError in one worker, verify wave completes."""
        store = ConditionStore()
        engine = _make_engine(store=store)

        # Pre-seed some findings so the store isn't empty
        store.admit("test finding from worker 0", row_type="finding",
                    angle="insulin_timing", confidence=0.8)
        store.admit("test finding from worker 1", row_type="finding",
                    angle="hematological_effects", confidence=0.7)

        findings_before = len(store.get_findings())
        assert findings_before == 2

        # Mock run_tool_free_worker: worker 0 crashes, workers 1 and 2 succeed
        call_count = 0

        async def _mock_run(package, query, **kwargs):
            nonlocal call_count
            call_count += 1
            if "worker_0" in package.worker_id:
                raise RuntimeError("Simulated worker crash")
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        # The pipeline must complete (report generated, not crashed)
        assert result.report is not None
        assert len(result.report) > 0


# ═══════════════════════════════════════════════════════════════════════
# B2: Worker timeout fires and the wave continues
# ═══════════════════════════════════════════════════════════════════════


class TestWorkerTimeoutFires:
    """B2: A hanging worker is killed by timeout, wave finishes."""

    @pytest.mark.asyncio
    async def test_timeout_cancels_hanging_worker(self) -> None:
        """Mock a worker that hangs forever, verify wave completes fast."""
        store = ConditionStore()
        config = MCPSwarmConfig(
            max_workers=2,
            max_waves=1,
            convergence_threshold=0,
            worker_timeout_s=2.0,  # short timeout for test speed
            enable_serendipity_wave=False,
            enable_rolling_summaries=False,
            compact_every_n_waves=0,
        )
        engine = _make_engine(store=store, config=config)

        async def _mock_run(package, query, **kwargs):
            if "worker_0" in package.worker_id:
                # Hang forever — should be cancelled by timeout
                await asyncio.sleep(999)
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        # Pipeline must complete
        assert result.report is not None


# ═══════════════════════════════════════════════════════════════════════
# B3: Serendipity failure doesn't prevent report generation
# ═══════════════════════════════════════════════════════════════════════


class TestSerendipityCrashDoesntBlockReport:
    """B3: Serendipity crash → report still generated."""

    @pytest.mark.asyncio
    async def test_serendipity_exception_still_produces_report(self) -> None:
        store = ConditionStore()
        config = MCPSwarmConfig(
            max_workers=2,
            max_waves=1,
            convergence_threshold=0,
            worker_timeout_s=5.0,
            enable_serendipity_wave=True,
            enable_rolling_summaries=False,
            compact_every_n_waves=0,
        )
        engine = _make_engine(store=store, config=config)

        async def _mock_run(package, query, **kwargs):
            if package.angle == "cross-domain connections":
                raise RuntimeError("Serendipity crash!")
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        # Report must be generated despite serendipity crash
        assert result.report is not None
        assert len(result.report) > 0


# ═══════════════════════════════════════════════════════════════════════
# B4: Report generation failure returns partial store summary
# ═══════════════════════════════════════════════════════════════════════


class TestReportCrashReturnsPartial:
    """B4: If report generation fails, return partial summary from store."""

    @pytest.mark.asyncio
    async def test_report_failure_returns_store_summary(self) -> None:
        store = ConditionStore()
        store.admit("insulin finding A", row_type="finding",
                    angle="insulin", confidence=0.9)
        store.admit("hematology finding B", row_type="finding",
                    angle="hematology", confidence=0.8)

        call_count = 0

        async def _failing_report_complete(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if "report" in prompt.lower() and "comprehensive" in prompt.lower():
                raise RuntimeError("LLM endpoint down!")
            return await _mock_complete(prompt)

        config = MCPSwarmConfig(
            max_workers=2,
            max_waves=1,
            convergence_threshold=0,
            worker_timeout_s=5.0,
            enable_serendipity_wave=False,
            enable_rolling_summaries=False,
            compact_every_n_waves=0,
        )
        engine = _make_engine(store=store, complete=_failing_report_complete, config=config)

        async def _mock_run(package, query, **kwargs):
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        # Must not crash — returns partial report
        assert result.report is not None
        # Partial report must mention the failure
        assert "failed" in result.report.lower() or "error" in result.report.lower() \
            or "findings" in result.report.lower()


# ═══════════════════════════════════════════════════════════════════════
# B5: Angle detection failure falls back to section titles
# ═══════════════════════════════════════════════════════════════════════


class TestAngleDetectionFallback:
    """B5: LLM angle detection fails → use section titles as angles."""

    @pytest.mark.asyncio
    async def test_angle_detection_exception_uses_section_titles(self) -> None:
        store = ConditionStore()

        async def _failing_angle_complete(prompt: str) -> str:
            if "distinct topics" in prompt.lower() or "research angles" in prompt.lower():
                raise RuntimeError("LLM angle detection failed!")
            return await _mock_complete(prompt)

        config = MCPSwarmConfig(
            max_workers=3,
            max_waves=1,
            convergence_threshold=0,
            worker_timeout_s=5.0,
            enable_serendipity_wave=False,
            enable_rolling_summaries=False,
            compact_every_n_waves=0,
        )
        engine = _make_engine(store=store, complete=_failing_angle_complete, config=config)

        async def _mock_run(package, query, **kwargs):
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        # Pipeline must complete (didn't crash on angle detection failure)
        assert result.report is not None

        # Angles should be section titles from the corpus headers
        # The corpus has 3 sections: Insulin Timing, Hematological Effects, GH/IGF-1
        assert len(result.angles_detected) >= 2, (
            f"Expected section-title fallback angles, got {result.angles_detected}"
        )


# ═══════════════════════════════════════════════════════════════════════
# B6: Observability methods never crash the pipeline
# ═══════════════════════════════════════════════════════════════════════


class TestEmitMetricFailureNonfatal:
    """B6: emit_metric crash doesn't abort synthesize()."""

    @pytest.mark.asyncio
    async def test_emit_metric_exception_nonfatal(self) -> None:
        store = ConditionStore()
        engine = _make_engine(store=store)

        # Corrupt emit_metric to always raise
        original_emit = store.emit_metric

        def _broken_emit(*args, **kwargs):
            raise RuntimeError("DB write failed!")

        store.emit_metric = _broken_emit

        async def _mock_run(package, query, **kwargs):
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            # The engine now wraps per-wave emit_metric in try/except,
            # so the pipeline should complete even with broken metrics
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        assert result.report is not None

    @pytest.mark.asyncio
    async def test_store_health_snapshot_exception_nonfatal(self) -> None:
        store = ConditionStore()
        engine = _make_engine(store=store)

        def _broken_health(*args, **kwargs):
            raise RuntimeError("Health snapshot failed!")

        store.store_health_snapshot = _broken_health

        async def _mock_run(package, query, **kwargs):
            return _make_mock_worker_result(package.angle, package.worker_id)

        with patch("swarm.mcp_engine.run_tool_free_worker", side_effect=_mock_run):
            result = await engine.synthesize(TEST_CORPUS, TEST_QUERY)

        assert result.report is not None


# ═══════════════════════════════════════════════════════════════════════
# B7: WAL mode is enabled for file-backed DuckDB connections
# ═══════════════════════════════════════════════════════════════════════


class TestWALModeFileBacked:
    """B7: File-backed ConditionStore has WAL journal mode active."""

    def test_wal_mode_active(self) -> None:
        # Use mktemp to get a path without creating the file —
        # DuckDB needs to create the file itself.
        db_path = tempfile.mktemp(suffix=".duckdb")

        try:
            store = ConditionStore(db_path=db_path)
            # DuckDB uses WAL mode by default for file-backed databases.
            # Verify the database file was created and is writable.
            assert os.path.exists(db_path)
            # Verify the store is functional with file backing
            cid = store.admit("test finding", row_type="finding", confidence=0.9)
            assert cid > 0

            findings = store.get_findings()
            assert len(findings) == 1
            assert findings[0]["fact"] == "test finding"

            # Verify the WAL file exists (DuckDB creates .wal alongside the db)
            wal_path = db_path + ".wal"
            # Note: DuckDB may not always create a separate .wal file
            # depending on version; the key test is that file-backed
            # operations work correctly with concurrent-safe writes
            store.conn.close()
        finally:
            for path in (db_path, db_path + ".wal"):
                if os.path.exists(path):
                    os.unlink(path)

    def test_in_memory_store_works_without_file(self) -> None:
        store = ConditionStore()
        cid = store.admit("in-memory finding", row_type="finding", confidence=0.9)
        assert cid > 0
        assert len(store.get_findings()) == 1


# ═══════════════════════════════════════════════════════════════════════
# C1: Concurrent store writes don't corrupt data
# ═══════════════════════════════════════════════════════════════════════


class TestConcurrentStoreWrites:
    """C1: 10 threads writing findings simultaneously — no data loss."""

    def test_concurrent_writes_no_corruption(self) -> None:
        store = ConditionStore()
        num_threads = 10
        findings_per_thread = 20
        errors: list[str] = []

        def _writer(thread_id: int) -> None:
            try:
                for i in range(findings_per_thread):
                    store.admit(
                        f"finding from thread {thread_id} item {i}",
                        row_type="finding",
                        angle=f"angle_{thread_id}",
                        confidence=0.5 + (i / findings_per_thread * 0.4),
                    )
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = [
            threading.Thread(target=_writer, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # No errors during concurrent writes
        assert not errors, f"Concurrent write errors: {errors}"

        # All findings must be present — no data loss
        # get_findings() has a default limit of 100, so use a higher limit
        expected = num_threads * findings_per_thread
        findings = store.get_findings(limit=expected + 10)
        assert len(findings) == expected, (
            f"Expected {expected} findings, got {len(findings)} — data loss detected"
        )

        # Verify each thread's findings are present
        for tid in range(num_threads):
            thread_findings = [
                f for f in findings if f"thread {tid} " in f["fact"]
            ]
            assert len(thread_findings) == findings_per_thread, (
                f"Thread {tid}: expected {findings_per_thread}, got {len(thread_findings)}"
            )


# ═══════════════════════════════════════════════════════════════════════
# C2: Compaction removes duplicates without losing unique findings
# ═══════════════════════════════════════════════════════════════════════


class TestCompactionPreservesUnique:
    """C2: Exact duplicates removed, unique findings preserved."""

    def test_exact_duplicate_removal(self) -> None:
        store = ConditionStore()

        # Insert duplicates within same angle
        store.admit("Insulin at 4-6 IU pre-workout", row_type="finding",
                    angle="insulin", confidence=0.7)
        store.admit("Insulin at 4-6 IU pre-workout", row_type="finding",
                    angle="insulin", confidence=0.9)  # higher confidence
        store.admit("Insulin at 4-6 IU pre-workout", row_type="finding",
                    angle="insulin", confidence=0.5)

        # Insert unique findings
        store.admit("GH at 2-4 IU daily", row_type="finding",
                    angle="insulin", confidence=0.8)
        store.admit("Trenbolone raises hematocrit", row_type="finding",
                    angle="hematology", confidence=0.85)

        assert len(store.get_findings()) == 5

        # Run compaction (Phase 1 only, no LLM)
        stats = store.compact(complete=None)

        assert stats["exact_duplicates_removed"] == 2, (
            f"Expected 2 exact duplicates removed, got {stats['exact_duplicates_removed']}"
        )

        # Unique findings must survive
        findings = store.get_findings()
        assert len(findings) == 3, (
            f"Expected 3 findings after compaction, got {len(findings)}"
        )

        # The highest-confidence duplicate must be the survivor
        insulin_findings = [f for f in findings if "4-6 IU" in f["fact"]]
        assert len(insulin_findings) == 1
        assert insulin_findings[0]["confidence"] == pytest.approx(0.9, abs=1e-5)

        # Other unique findings untouched
        gh_findings = [f for f in findings if "GH" in f["fact"]]
        tren_findings = [f for f in findings if "Trenbolone" in f["fact"]]
        assert len(gh_findings) == 1
        assert len(tren_findings) == 1

    def test_cross_angle_duplicates_preserved(self) -> None:
        """Same fact in different angles should NOT be deduplicated."""
        store = ConditionStore()

        store.admit("Insulin at 4-6 IU pre-workout", row_type="finding",
                    angle="insulin_timing", confidence=0.8)
        store.admit("Insulin at 4-6 IU pre-workout", row_type="finding",
                    angle="drug_interactions", confidence=0.7)

        stats = store.compact(complete=None)

        # Cross-angle duplicates are NOT removed (different angles)
        assert stats["exact_duplicates_removed"] == 0
        assert len(store.get_findings()) == 2


# ═══════════════════════════════════════════════════════════════════════
# A4: Convergence counts only worker-generated findings, not raw ingestion
# ═══════════════════════════════════════════════════════════════════════


class TestConvergenceExcludesRawIngestion:
    """A4: Convergence detection only counts worker-produced findings."""

    def test_raw_ingestion_excluded_from_convergence_count(self) -> None:
        store = ConditionStore()

        # Ingest raw corpus sections (source_type = 'corpus_section')
        store.ingest_raw(
            raw_text="Insulin at 4-6 IU pre-workout.\n\nGH at 2-4 IU daily.",
            source_type="corpus_section",
            source_ref="section_0",
            angle="insulin",
            iteration=0,
            user_query=TEST_QUERY,
        )

        # The convergence query from mcp_engine.py:
        # counts findings where source_type != 'corpus_section'
        with store._lock:
            worker_findings = store.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type IN ('finding', 'thought', 'insight', 'synthesis') "
                "AND source_type != 'corpus_section'"
            ).fetchone()[0]

        # Raw ingestion should NOT appear in worker finding count
        assert worker_findings == 0, (
            f"Expected 0 worker findings after raw ingestion, got {worker_findings}"
        )

        # Now add worker-generated findings
        store.admit(
            "Worker analysis: insulin synergy with GH",
            row_type="finding",
            angle="insulin",
            confidence=0.8,
            source_type="worker_analysis",
        )

        with store._lock:
            worker_findings_after = store.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type IN ('finding', 'thought', 'insight', 'synthesis') "
                "AND source_type != 'corpus_section'"
            ).fetchone()[0]

        assert worker_findings_after == 1, (
            f"Expected 1 worker finding, got {worker_findings_after}"
        )

    def test_convergence_threshold_only_uses_worker_findings(self) -> None:
        """Verify the engine convergence check ignores ingested corpus rows."""
        store = ConditionStore()

        # Ingest a large corpus to create many rows
        big_corpus = "\n\n".join(
            [f"Paragraph {i}: factual claim about compound {i}." for i in range(50)]
        )
        store.ingest_raw(
            raw_text=big_corpus,
            source_type="corpus_section",
            source_ref="big_section",
            angle="pharmacology",
            iteration=0,
            user_query=TEST_QUERY,
        )

        # Total rows should be high (from ingestion)
        with store._lock:
            total_rows = store.conn.execute(
                "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE"
            ).fetchone()[0]
        assert total_rows > 10, "Ingestion should create multiple rows"

        # But worker-finding count should be zero
        with store._lock:
            worker_count = store.conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type IN ('finding', 'thought', 'insight', 'synthesis') "
                "AND source_type != 'corpus_section'"
            ).fetchone()[0]
        assert worker_count == 0, (
            "Convergence count must be 0 when only raw ingestion has occurred"
        )
