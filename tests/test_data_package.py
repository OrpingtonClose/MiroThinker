"""Focused tests validating ConditionStore data-serving and DataPackage builder.

Backlog assumptions tested:
1. ConditionStore can serve data packages (filter by angle, row_type,
   confidence ordering, cross-angle queries, knowledge summaries).
2. build_data_packages produces correctly populated sections per wave.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root and strands-agent are importable
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
_STRANDS_AGENT = str(Path(__file__).resolve().parents[1] / "apps" / "strands-agent")
for p in (_REPO_ROOT, _STRANDS_AGENT):
    if p not in sys.path:
        sys.path.insert(0, p)

from corpus import AtomicCondition, ConditionStore
from swarm.angles import WorkerAssignment
from swarm.data_package import DataPackage, build_data_packages


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> ConditionStore:
    """In-memory ConditionStore seeded with 3 angles of test data."""
    s = ConditionStore(db_path="")

    # Angle 1: insulin_timing
    s.ingest_raw(
        raw_text=(
            "Rapid-acting insulin peaks at 60-90 min post-injection.\n\n"
            "Pre-workout insulin at 4-6 IU enhances glycogen supercompensation.\n\n"
            "Insulin sensitivity is highest in the morning fasted state."
        ),
        source_type="test",
        source_ref="corpus_insulin",
        angle="insulin_timing",
        iteration=0,
        user_query="insulin protocol",
    )

    # Angle 2: hematology
    s.ingest_raw(
        raw_text=(
            "Hematocrit above 54% increases thrombotic risk significantly.\n\n"
            "Therapeutic phlebotomy removes 450-500 mL per session.\n\n"
            "Ferritin below 30 ng/mL indicates depleted iron stores."
        ),
        source_type="test",
        source_ref="corpus_hematology",
        angle="hematology",
        iteration=0,
        user_query="blood work management",
    )

    # Angle 3: micronutrients
    s.ingest_raw(
        raw_text=(
            "Magnesium glycinate at 400 mg/day corrects intracellular depletion.\n\n"
            "Zinc at 30 mg/day supports testosterone via 5-alpha reductase.\n\n"
            "Vitamin D3 at 5000 IU/day maintains 60-80 ng/mL serum levels."
        ),
        source_type="test",
        source_ref="corpus_micronutrients",
        angle="micronutrients",
        iteration=0,
        user_query="micronutrient protocol",
    )

    # Add some non-finding rows for row_type filtering
    s.admit(
        fact="Insulin timing interacts with hematocrit via IGF-1 pathway",
        row_type="insight",
        angle="insulin_timing",
        confidence=0.85,
    )
    s.admit(
        fact="Contradicts: ferritin threshold should be 50 not 30",
        row_type="contradiction",
        angle="hematology",
        confidence=0.6,
        related_id=1,
    )
    s.admit(
        fact="What is the optimal zinc-to-copper ratio?",
        row_type="research_question",
        angle="micronutrients",
        confidence=0.3,
    )

    return s


# ===================================================================
# Part 1: ConditionStore can serve data packages
# ===================================================================

class TestConditionStoreDataServing:
    """Validate that ConditionStore supports the queries data packages need."""

    def test_filter_by_angle(self, store: ConditionStore) -> None:
        """Findings filtered by angle return only that angle's data."""
        insulin = store.get_findings(angle="insulin_timing")
        assert len(insulin) >= 3
        assert all(f["angle"] == "insulin_timing" for f in insulin)

        hema = store.get_findings(angle="hematology")
        assert len(hema) >= 3
        assert all(f["angle"] == "hematology" for f in hema)

        micro = store.get_findings(angle="micronutrients")
        assert len(micro) >= 3
        assert all(f["angle"] == "micronutrients" for f in micro)

    def test_filter_by_row_type(self, store: ConditionStore) -> None:
        """Different row_types are queryable via raw SQL on the store."""
        with store._lock:
            insights = store.conn.execute(
                "SELECT id, fact FROM conditions "
                "WHERE row_type = 'insight' AND consider_for_use = TRUE"
            ).fetchall()
        assert len(insights) >= 1
        assert any("IGF-1" in row[1] for row in insights)

        with store._lock:
            contradictions = store.conn.execute(
                "SELECT id, fact FROM conditions "
                "WHERE row_type = 'contradiction' AND consider_for_use = TRUE"
            ).fetchall()
        assert len(contradictions) >= 1

        with store._lock:
            questions = store.conn.execute(
                "SELECT id, fact FROM conditions "
                "WHERE row_type = 'research_question' AND consider_for_use = TRUE"
            ).fetchall()
        assert len(questions) >= 1
        assert any("zinc" in row[1].lower() for row in questions)

    def test_ordered_by_confidence(self, store: ConditionStore) -> None:
        """get_findings returns results ordered by confidence descending."""
        # Add findings with explicit varied confidence
        store.admit(fact="High-conf finding", angle="insulin_timing",
                    confidence=0.95, row_type="finding")
        store.admit(fact="Low-conf finding", angle="insulin_timing",
                    confidence=0.15, row_type="finding")
        store.admit(fact="Mid-conf finding", angle="insulin_timing",
                    confidence=0.55, row_type="finding")

        findings = store.get_findings(angle="insulin_timing")
        confidences = [f["confidence"] for f in findings]
        assert confidences == sorted(confidences, reverse=True), (
            f"Findings not ordered by confidence desc: {confidences}"
        )

    def test_cross_angle_queries(self, store: ConditionStore) -> None:
        """Querying findings from OTHER angles excludes the requesting angle."""
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact, angle, confidence
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND angle != 'insulin_timing'
                     AND row_type IN ('finding', 'thought', 'insight')
                   ORDER BY confidence DESC
                   LIMIT 15""",
            ).fetchall()
        assert len(rows) >= 6  # hematology + micronutrients findings
        angles_returned = {r[1] for r in rows}
        assert "insulin_timing" not in angles_returned
        assert "hematology" in angles_returned or "micronutrients" in angles_returned

    def test_store_summary_and_get_latest(self, store: ConditionStore) -> None:
        """store_summary persists and get_latest_summary retrieves it."""
        # Initially no summary
        assert store.get_latest_summary("insulin_timing") == ""

        store.store_summary(
            angle="insulin_timing",
            summary="Insulin peaks 60-90 min; pre-workout 4-6 IU optimal.",
            finding_count=3,
            run_number=1,
        )
        result = store.get_latest_summary("insulin_timing")
        assert "4-6 IU" in result

        # Store a second summary — get_latest returns the newest
        store.store_summary(
            angle="insulin_timing",
            summary="Updated: morning fasted insulin sensitivity confirmed.",
            finding_count=4,
            run_number=2,
        )
        latest = store.get_latest_summary("insulin_timing")
        assert "Updated" in latest
        assert "4-6 IU" not in latest  # old summary superseded

        # Other angles unaffected
        assert store.get_latest_summary("hematology") == ""

    def test_min_confidence_filter(self, store: ConditionStore) -> None:
        """get_findings respects min_confidence threshold."""
        store.admit(fact="Very low conf claim", angle="hematology",
                    confidence=0.1, row_type="finding")
        high_only = store.get_findings(angle="hematology", min_confidence=0.4)
        assert all(f["confidence"] >= 0.4 for f in high_only)

        all_findings = store.get_findings(angle="hematology", min_confidence=0.0)
        assert len(all_findings) >= len(high_only)


# ===================================================================
# Part 2: Data package builder works
# ===================================================================

def _make_assignments() -> list[WorkerAssignment]:
    """Create 3 mock WorkerAssignment objects for the 3 test angles."""
    return [
        WorkerAssignment(
            worker_id=0,
            angle="insulin_timing",
            raw_content=(
                "## Insulin Timing\n"
                "Rapid-acting insulin peaks at 60-90 min post-injection.\n"
                "Pre-workout insulin at 4-6 IU enhances glycogen supercompensation.\n"
                "Insulin sensitivity is highest in the morning fasted state."
            ),
        ),
        WorkerAssignment(
            worker_id=1,
            angle="hematology",
            raw_content=(
                "## Hematology\n"
                "Hematocrit above 54% increases thrombotic risk significantly.\n"
                "Therapeutic phlebotomy removes 450-500 mL per session.\n"
                "Ferritin below 30 ng/mL indicates depleted iron stores."
            ),
        ),
        WorkerAssignment(
            worker_id=2,
            angle="micronutrients",
            raw_content=(
                "## Micronutrients\n"
                "Magnesium glycinate at 400 mg/day corrects intracellular depletion.\n"
                "Zinc at 30 mg/day supports testosterone via 5-alpha reductase.\n"
                "Vitamin D3 at 5000 IU/day maintains 60-80 ng/mL serum levels."
            ),
        ),
    ]


class TestDataPackageBuilder:
    """Validate build_data_packages progressive enrichment per wave."""

    def test_wave1_only_corpus_material(self, store: ConditionStore) -> None:
        """Wave 1: only section 2 (corpus_material) is populated."""
        assignments = _make_assignments()
        packages = build_data_packages(
            store=store,
            assignments=assignments,
            wave=1,
            query="bodybuilding protocol",
        )
        assert len(packages) == 3

        for pkg in packages:
            assert pkg.corpus_material, "§2 corpus_material must be populated"
            assert not pkg.knowledge_state, "§1 must be empty in wave 1"
            assert not pkg.hive_findings, "§3 must be empty in wave 1"
            assert not pkg.cross_domain, "§4 must be empty in wave 1"
            assert not pkg.challenges, "§5 must be empty in wave 1"
            assert not pkg.research_gaps, "§6 must be empty in wave 1"
            assert not pkg.previous_output, "§7 must be empty in wave 1"

    def test_wave2_has_sections_1_2_3_6_7(self, store: ConditionStore) -> None:
        """Wave 2: sections 1, 2, 3, 6, 7 are populated."""
        # Seed the store with summaries so §1 is populated
        store.store_summary(
            angle="insulin_timing",
            summary="Insulin peaks 60-90 min; pre-workout 4-6 IU optimal.",
            finding_count=3,
            run_number=1,
        )
        store.store_summary(
            angle="hematology",
            summary="Hematocrit >54% is dangerous; phlebotomy standard is 450 mL.",
            finding_count=3,
            run_number=1,
        )
        store.store_summary(
            angle="micronutrients",
            summary="Mg 400 mg/d, Zn 30 mg/d, D3 5000 IU/d baseline.",
            finding_count=3,
            run_number=1,
        )

        assignments = _make_assignments()
        prior_outputs = {
            "insulin_timing": "Insulin at 4-6 IU pre-workout. Need more data on GH synergy. Uncertain about timing.",
            "hematology": "Hematocrit management. Further research on EPO interaction needed.",
            "micronutrients": "Baseline supplementation established. Gap in copper balance.",
        }

        packages = build_data_packages(
            store=store,
            assignments=assignments,
            wave=2,
            query="bodybuilding protocol",
            prior_outputs=prior_outputs,
        )
        assert len(packages) == 3

        for pkg in packages:
            assert pkg.corpus_material, "§2 must be populated in wave 2"
            assert pkg.knowledge_state, f"§1 must be populated in wave 2 for {pkg.angle}"
            # §3 hive_findings: populated from store cross-angle query
            assert pkg.hive_findings, f"§3 must be populated in wave 2 for {pkg.angle}"
            # §6 research_gaps: populated from gap markers in prior output
            assert pkg.research_gaps, f"§6 must be populated in wave 2 for {pkg.angle}"
            # §7 previous_output
            assert pkg.previous_output, "§7 must be populated in wave 2"

            # Sections 4 and 5 must still be empty in wave 2
            assert not pkg.cross_domain, "§4 must be empty in wave 2"
            assert not pkg.challenges, "§5 must be empty in wave 2"

    def test_wave3_all_sections_populated(self, store: ConditionStore) -> None:
        """Wave 3: all 7 sections populated."""
        # Seed summaries
        for angle in ("insulin_timing", "hematology", "micronutrients"):
            store.store_summary(
                angle=angle,
                summary=f"Rolling summary for {angle}.",
                finding_count=3,
                run_number=2,
            )

        # Add cross-domain insights and contradictions so §4/§5 have data
        store.admit(
            fact="Insulin timing affects hematocrit via EPO-IGF1 crosstalk",
            row_type="insight",
            angle="insulin_timing",
            confidence=0.8,
        )
        store.admit(
            fact="Micronutrient zinc modulates insulin receptor sensitivity",
            row_type="insight",
            angle="micronutrients",
            confidence=0.75,
        )
        # Contradiction targeting hematology
        target_id = store.admit(
            fact="Ferritin threshold 30 ng/mL is too conservative",
            row_type="finding",
            angle="hematology",
            confidence=0.7,
        )
        store.admit(
            fact="Contradicts ferritin threshold — should be 50 ng/mL minimum",
            row_type="contradiction",
            angle="micronutrients",
            confidence=0.65,
            related_id=target_id,
        )

        assignments = _make_assignments()
        prior_outputs = {
            "insulin_timing": "Insulin protocol established. Unclear interaction with iron status.",
            "hematology": "Blood markers tracked. Unresolved EPO interaction.",
            "micronutrients": "Supplementation baseline set. Missing data on absorption timing.",
        }

        packages = build_data_packages(
            store=store,
            assignments=assignments,
            wave=3,
            query="bodybuilding protocol",
            prior_outputs=prior_outputs,
        )
        assert len(packages) == 3

        for pkg in packages:
            assert pkg.corpus_material, f"§2 empty for {pkg.angle}"
            assert pkg.knowledge_state, f"§1 empty for {pkg.angle}"
            assert pkg.hive_findings, f"§3 empty for {pkg.angle}"
            assert pkg.previous_output, f"§7 empty for {pkg.angle}"

        # At least one package should have cross-domain and challenges
        has_cross_domain = any(pkg.cross_domain for pkg in packages)
        has_challenges = any(pkg.challenges for pkg in packages)
        assert has_cross_domain, "§4 cross_domain should be populated for at least one angle in wave 3"
        assert has_challenges, "§5 challenges should be populated for at least one angle in wave 3"

    def test_rendered_output_is_well_formatted(self, store: ConditionStore) -> None:
        """The rendered data package is a non-empty well-formatted string."""
        assignments = _make_assignments()
        packages = build_data_packages(
            store=store,
            assignments=assignments,
            wave=1,
            query="bodybuilding protocol",
        )

        for pkg in packages:
            rendered = pkg.render(query="bodybuilding protocol")
            assert isinstance(rendered, str)
            assert len(rendered) > 100, "Rendered output too short"
            assert "RESEARCH BRIEF" in rendered
            assert "§ 2  CORPUS MATERIAL" in rendered
            assert pkg.angle in rendered
            assert "bodybuilding protocol" in rendered

    def test_render_wave3_includes_all_section_headers(self, store: ConditionStore) -> None:
        """Wave 3 render includes all 7 section headers when data exists."""
        pkg = DataPackage(
            angle="insulin_timing",
            wave=3,
            worker_id="worker_0_wave_3",
            model="test-model",
            knowledge_state="Knowledge about insulin timing.",
            corpus_material="Raw corpus material here.",
            hive_findings="Cross-angle findings from hematology.",
            cross_domain="Validated cross-domain connections.",
            challenges="Expert-informed challenges.",
            research_gaps="Identified research gaps.",
            previous_output="Previous wave analysis.",
        )

        rendered = pkg.render(query="bodybuilding protocol")
        assert "§ 1  KNOWLEDGE STATE" in rendered
        assert "§ 2  CORPUS MATERIAL" in rendered
        assert "§ 3  FROM THE HIVE" in rendered
        assert "§ 4  CROSS-DOMAIN CONNECTIONS" in rendered
        assert "§ 5  CHALLENGES" in rendered
        assert "§ 6  RESEARCH GAPS" in rendered
        assert "§ 7  YOUR PREVIOUS ANALYSIS" in rendered

    def test_model_map_assigns_models(self, store: ConditionStore) -> None:
        """model_map correctly populates the model field on each package."""
        assignments = _make_assignments()
        model_map = {
            "insulin_timing": "gemma-4-27b",
            "hematology": "qwen3-8b",
            "micronutrients": "llama-3.3-70b",
        }
        packages = build_data_packages(
            store=store,
            assignments=assignments,
            wave=1,
            query="test",
            model_map=model_map,
        )
        for pkg in packages:
            assert pkg.model == model_map[pkg.angle]

    def test_worker_id_format(self, store: ConditionStore) -> None:
        """Worker IDs follow the worker_{id}_wave_{n} format."""
        assignments = _make_assignments()
        packages = build_data_packages(
            store=store,
            assignments=assignments,
            wave=2,
            query="test",
            prior_outputs={"insulin_timing": "prior", "hematology": "prior", "micronutrients": "prior"},
        )
        for pkg in packages:
            assert pkg.worker_id.startswith("worker_")
            assert "_wave_2" in pkg.worker_id
