"""Unit tests for the observability layer — metrics as ConditionStore rows.

Covers:
- emit_metric() persists JSON metric blobs with correct row_type and metadata
- store_health_snapshot() queries and persists store distribution
- Metric rows are excluded from research queries (consider_for_use = FALSE)
- source_model / source_run columns backfilled on older schemas
"""

from __future__ import annotations

import json

import pytest

from corpus import ConditionStore


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def store() -> ConditionStore:
    """In-memory ConditionStore for isolated tests."""
    return ConditionStore()


# ═══════════════════════════════════════════════════════════════════════
# emit_metric
# ═══════════════════════════════════════════════════════════════════════


class TestEmitMetric:
    """Verify emit_metric persists well-formed metric rows."""

    def test_basic_metric_stored(self, store: ConditionStore) -> None:
        data = {"findings_new": 12, "elapsed_s": 5.3}
        cid = store.emit_metric("wave_metric", data, iteration=2)

        row = store.conn.execute(
            "SELECT fact, row_type, source_type, consider_for_use, "
            "angle, iteration FROM conditions WHERE id = ?",
            [cid],
        ).fetchone()
        assert row is not None
        fact, row_type, source_type, consider, angle, iteration = row

        assert row_type == "wave_metric"
        assert source_type == "observability"
        assert consider is False  # excluded from research queries
        assert angle == "system"
        assert iteration == 2

        parsed = json.loads(fact)
        assert parsed["findings_new"] == 12
        assert parsed["elapsed_s"] == 5.3

    def test_source_model_and_run_persisted(self, store: ConditionStore) -> None:
        cid = store.emit_metric(
            "run_metric",
            {"total_workers": 5},
            source_model="qwen-32b",
            source_run="run_20260416_120000",
        )

        row = store.conn.execute(
            "SELECT source_model, source_run FROM conditions WHERE id = ?",
            [cid],
        ).fetchone()
        assert row is not None
        assert row[0] == "qwen-32b"
        assert row[1] == "run_20260416_120000"

    def test_custom_angle(self, store: ConditionStore) -> None:
        cid = store.emit_metric(
            "worker_metric",
            {"output_chars": 5000},
            angle="insulin_timing",
        )
        row = store.conn.execute(
            "SELECT angle FROM conditions WHERE id = ?", [cid],
        ).fetchone()
        assert row[0] == "insulin_timing"

    def test_parent_id_linked(self, store: ConditionStore) -> None:
        parent = store.admit("some finding", row_type="finding")
        cid = store.emit_metric(
            "worker_metric",
            {"quality": 0.8},
            parent_id=parent,
        )
        row = store.conn.execute(
            "SELECT parent_id FROM conditions WHERE id = ?", [cid],
        ).fetchone()
        assert row[0] == parent

    def test_metric_excluded_from_get_findings(self, store: ConditionStore) -> None:
        """Metric rows must not appear in research queries."""
        store.admit("real finding", row_type="finding", confidence=0.9)
        store.emit_metric("wave_metric", {"x": 1})

        findings = store.get_findings()
        assert len(findings) == 1
        assert findings[0]["fact"] == "real finding"

    def test_multiple_metrics_sequential_ids(self, store: ConditionStore) -> None:
        id1 = store.emit_metric("wave_metric", {"wave": 1})
        id2 = store.emit_metric("wave_metric", {"wave": 2})
        assert id2 == id1 + 1


# ═══════════════════════════════════════════════════════════════════════
# store_health_snapshot
# ═══════════════════════════════════════════════════════════════════════


class TestStoreHealthSnapshot:
    """Verify store_health_snapshot captures and persists distribution."""

    def test_empty_store_health(self, store: ConditionStore) -> None:
        health = store.store_health_snapshot(source_run="test_run")
        # The snapshot queries BEFORE inserting its own metric row,
        # so an empty store reports 0 rows.
        assert health["total_rows"] == 0
        assert health["active_rows"] == 0
        assert health["obsolete_rows"] == 0

    def test_populated_store_health(self, store: ConditionStore) -> None:
        store.admit("finding A", row_type="finding", angle="alpha")
        store.admit("finding B", row_type="finding", angle="alpha")
        store.admit("thought C", row_type="thought", angle="beta")
        store.ingest_raw("raw text here", source_type="corpus")

        health = store.store_health_snapshot(
            source_run="run_123", iteration=3,
        )

        # Raw ingestion creates 1 raw row + paragraph findings
        assert health["total_rows"] > 4
        assert health["active_rows"] >= 3  # at least the 3 admitted rows
        assert "finding" in health["rows_by_type"]
        assert "alpha" in health["rows_by_angle"]

        # Verify persisted as store_metric row
        rows = store.conn.execute(
            "SELECT fact, source_run, iteration FROM conditions "
            "WHERE row_type = 'store_metric'",
        ).fetchall()
        assert len(rows) == 1
        fact_json, run, iteration = rows[0]
        assert run == "run_123"
        assert iteration == 3
        parsed = json.loads(fact_json)
        assert parsed["total_rows"] == health["total_rows"]

    def test_health_snapshot_is_not_research_visible(
        self, store: ConditionStore,
    ) -> None:
        store.store_health_snapshot()
        findings = store.get_findings()
        for f in findings:
            assert f["row_type"] != "store_metric"


# ═══════════════════════════════════════════════════════════════════════
# Schema migration — source_model / source_run columns
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Corpus fingerprinting (#190)
# ═══════════════════════════════════════════════════════════════════════


class TestCorpusFingerprinting:
    """Verify SHA-256 fingerprint dedup on ingest_raw."""

    def test_first_ingest_succeeds(self, store: ConditionStore) -> None:
        ids = store.ingest_raw("Hello world paragraph one.\n\nParagraph two.")
        assert len(ids) > 0

    def test_duplicate_ingest_skipped(self, store: ConditionStore) -> None:
        text = "Unique corpus content for dedup test.\n\nSecond paragraph."
        ids1 = store.ingest_raw(text, source_type="corpus")
        ids2 = store.ingest_raw(text, source_type="corpus")
        assert len(ids1) > 0
        assert ids2 == []  # skipped

    def test_different_corpus_not_skipped(self, store: ConditionStore) -> None:
        ids1 = store.ingest_raw("Corpus A content.\n\nMore A.")
        ids2 = store.ingest_raw("Corpus B content.\n\nMore B.")
        assert len(ids1) > 0
        assert len(ids2) > 0

    def test_has_corpus_fingerprint(self, store: ConditionStore) -> None:
        text = "Check fingerprint existence."
        assert not store.has_corpus_fingerprint(text)
        store.ingest_raw(text)
        assert store.has_corpus_fingerprint(text)

    def test_fingerprint_not_in_research_queries(
        self, store: ConditionStore,
    ) -> None:
        store.ingest_raw("Some corpus text here.\n\nAnother paragraph.")
        findings = store.get_findings()
        for f in findings:
            assert "fingerprint" not in f.get("fact", "").lower() or \
                f.get("row_type") != "corpus_fingerprint"


# ═══════════════════════════════════════════════════════════════════════
# Schema migration — source_model / source_run columns
# ═══════════════════════════════════════════════════════════════════════


class TestSchemaMigration:
    """Verify that source_model and source_run columns exist after init."""

    def test_columns_exist(self, store: ConditionStore) -> None:
        """Both provenance columns should be queryable."""
        result = store.conn.execute(
            "SELECT source_model, source_run FROM conditions LIMIT 0"
        ).fetchall()
        # No error means columns exist
        assert result == []

    def test_admit_does_not_break_with_new_columns(
        self, store: ConditionStore,
    ) -> None:
        """Existing admit() still works — new columns default to empty."""
        cid = store.admit("test fact")
        row = store.conn.execute(
            "SELECT source_model, source_run FROM conditions WHERE id = ?",
            [cid],
        ).fetchone()
        assert row[0] == ""
        assert row[1] == ""
