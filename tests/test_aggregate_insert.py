"""Integration test for AGGREGATE INSERT into ConditionStore.

Validates that store_aggregate_directions() correctly inserts
research_target rows with the id column populated, preventing
the silent no-op regression where missing id caused all AGGREGATE
queries to fail silently.

Covers GitHub issue #252.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root and strands-agent are importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_STRANDS_AGENT = str(Path(__file__).resolve().parents[1] / "apps" / "strands-agent")
_SWARM = str(Path(__file__).resolve().parents[1] / "swarm")
for p in (_REPO_ROOT, _STRANDS_AGENT, _SWARM):
    if p not in sys.path:
        sys.path.insert(0, p)

from corpus import ConditionStore
from swarm.flock_query_manager import _store_aggregate_research_targets as store_aggregate_directions


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """Create an in-memory ConditionStore for testing."""
    return ConditionStore(db_path="")


# ── Tests ─────────────────────────────────────────────────────────────

class TestAggregateInsert:
    """Verify AGGREGATE INSERT produces valid research_target rows."""

    def test_insert_populates_id_column(self, store: ConditionStore) -> None:
        """The id column must be populated for every inserted row."""
        directions = [
            {
                "search_query": "insulin sensitivity fatty acid composition",
                "source_type": "pubmed",
                "why": "Cross-domain connection between insulin and lipid metabolism",
                "priority": 0.8,
            },
        ]

        stored = store_aggregate_directions(
            store=store,
            directions=directions,
            run_id="test_run_001",
            round_number=1,
        )

        assert stored == 1

        # Verify the row exists with a valid id
        rows = store.conn.execute(
            "SELECT id, fact, row_type, angle FROM conditions "
            "WHERE row_type = 'research_target'"
        ).fetchall()

        assert len(rows) == 1
        row = rows[0]
        assert row[0] is not None, "id column must not be NULL"
        assert row[0] > 0, "id must be a positive integer"
        assert "insulin sensitivity" in row[1], "fact should contain the search query"
        assert row[2] == "research_target"
        assert row[3] == "aggregate"

    def test_insert_multiple_directions(self, store: ConditionStore) -> None:
        """Multiple directions should each get unique ids."""
        directions = [
            {
                "search_query": "DHA neuroprotection trenbolone neurotoxicity",
                "source_type": "pubmed",
                "why": "Omega-3 may protect against tren-induced neuronal damage",
                "priority": 0.9,
            },
            {
                "search_query": "BPC-157 tendon healing dosage protocol",
                "source_type": "forum",
                "why": "Practitioner reports on peptide healing protocols",
                "priority": 0.7,
            },
            {
                "search_query": "growth hormone insulin resistance mechanism",
                "source_type": "pubmed",
                "why": "High-dose GH induces insulin resistance via specific pathways",
                "priority": 0.85,
            },
        ]

        stored = store_aggregate_directions(
            store=store,
            directions=directions,
            run_id="test_run_002",
            round_number=2,
        )

        assert stored == 3

        rows = store.conn.execute(
            "SELECT id, fact, expansion_gap, expansion_priority "
            "FROM conditions WHERE row_type = 'research_target' "
            "ORDER BY id"
        ).fetchall()

        assert len(rows) == 3

        # All ids must be unique and non-null
        ids = [r[0] for r in rows]
        assert all(i is not None for i in ids), "all ids must be non-null"
        assert len(set(ids)) == 3, "all ids must be unique"

        # expansion_gap should contain the search query
        for row in rows:
            assert row[2] is not None, "expansion_gap must be populated"
            assert len(row[2]) > 0, "expansion_gap must not be empty"

        # expansion_priority should be boosted (priority + 0.2, capped at 1.0)
        assert rows[0][3] == pytest.approx(1.0, abs=0.01)  # 0.9 + 0.2 = 1.1 -> capped at 1.0
        assert rows[1][3] == pytest.approx(0.9, abs=0.01)  # 0.7 + 0.2 = 0.9
        assert rows[2][3] == pytest.approx(1.0, abs=0.01)  # 0.85 + 0.2 = 1.05 -> capped at 1.0

    def test_insert_preserves_provenance(self, store: ConditionStore) -> None:
        """source_model and source_run must be populated for traceability."""
        directions = [
            {
                "search_query": "sulforaphane glutathione synthesis",
                "source_type": "pubmed",
                "why": "Methylation pathway support",
                "priority": 0.6,
            },
        ]

        store_aggregate_directions(
            store=store,
            directions=directions,
            run_id="provenance_test",
            round_number=3,
        )

        row = store.conn.execute(
            "SELECT source_model, source_run FROM conditions "
            "WHERE row_type = 'research_target'"
        ).fetchone()

        assert row is not None
        assert "flock_aggregate" in row[0], "source_model should identify flock aggregate"
        assert row[1] == "provenance_test", "source_run should match the run_id"

    def test_empty_directions_stores_nothing(self, store: ConditionStore) -> None:
        """Empty direction list should store zero rows without error."""
        stored = store_aggregate_directions(
            store=store,
            directions=[],
            run_id="empty_test",
            round_number=1,
        )

        assert stored == 0

        count = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE row_type = 'research_target'"
        ).fetchone()[0]
        assert count == 0

    def test_consider_for_use_is_true(self, store: ConditionStore) -> None:
        """Research targets must have consider_for_use=TRUE so MCP researcher picks them up."""
        directions = [
            {
                "search_query": "taurine cardiac protection anabolic steroids",
                "source_type": "pubmed",
                "why": "Cardioprotective amino acid in PED context",
                "priority": 0.75,
            },
        ]

        store_aggregate_directions(
            store=store,
            directions=directions,
            run_id="visibility_test",
            round_number=1,
        )

        row = store.conn.execute(
            "SELECT consider_for_use FROM conditions "
            "WHERE row_type = 'research_target'"
        ).fetchone()

        assert row is not None
        assert row[0] is True, "research targets must be visible to MCP researcher"
