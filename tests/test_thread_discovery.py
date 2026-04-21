"""Unit tests for thread-discovery mechanisms.

Covers three mechanisms designed to find connections between facts buried
in non-obvious locations:

1. Deliberate misassignment — off-angle raw data injected into each bee's
   slice so the bee's worldview activates on foreign data.
2. Swarm-internal RAG (hive memory) — bees query accumulated lineage
   entries between gossip rounds for targeted cross-angle findings.
3. Core interest worldview prompts — deep domain identity in worker
   system prompts that converts raw foreign data into thread-discovery.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from swarm.angles import WorkerAssignment, apply_misassignment
from swarm.lineage import LineageEntry
from swarm.rag import extract_concepts, query_hive, score_relevance
from swarm.worker import _build_synth_prompt, _build_gossip_prompt


# ═══════════════════════════════════════════════════════════════════════
# 1. Deliberate Misassignment
# ═══════════════════════════════════════════════════════════════════════


class TestApplyMisassignment:
    """Test off-angle data injection into worker slices."""

    @staticmethod
    def _make_assignments(n: int = 5) -> list[WorkerAssignment]:
        """Create N dummy assignments with realistic-length content."""
        angles = [
            "Molecular Biology", "Practitioner Findings",
            "Pharmacokinetics", "Safety Profile", "Endocrinology",
        ]
        # Each content block is >500 chars so 25% injection > 100 char threshold
        contents = [
            (
                "Iron sequestration via hepcidin regulation in hepatocytes. "
                "Ferritin elevated 340% in treated cohort. Transferrin saturation "
                "index dropped to 12% indicating functional iron deficiency. "
                "The hepcidin-ferroportin axis is the master regulator of systemic "
                "iron homeostasis. When hepcidin binds ferroportin on the "
                "basolateral membrane of enterocytes and macrophages, it triggers "
                "internalization and degradation, effectively blocking iron export "
                "into plasma. In the treated group, hepcidin mRNA was suppressed "
                "by 78%, leading to unchecked iron absorption from duodenal "
                "enterocytes and rapid mobilization from hepatic stores."
            ),
            (
                "My hematocrit went from 42 to 49.6 after 8 weeks on 400mg "
                "tren-e. Doc said donate blood. Bloodwork showed elevated RBC "
                "count and hemoglobin at 16.5 g/dL. Other guys on the forum "
                "reported similar patterns — hematocrit typically rises 15-20% "
                "on trenbolone cycles regardless of dose. One user posted labs "
                "showing ferritin dropped from 150 to 35 ng/mL during an 8-week "
                "cycle while hematocrit climbed steadily. Another noted that "
                "therapeutic phlebotomy brought hematocrit down but symptoms of "
                "fatigue persisted, suggesting the iron mobilization rather than "
                "the erythrocytosis itself was the primary issue."
            ),
            (
                "Trenbolone acetate half-life 48-72h. Peak plasma concentration "
                "at 6h post-injection. Hepatic first-pass metabolism negligible "
                "for injectable esters. Clearance via CYP3A4. The enanthate "
                "ester extends the release profile to approximately 7-10 days, "
                "with steady-state plasma levels achieved by week 3 of regular "
                "administration. Unlike testosterone, trenbolone does not undergo "
                "5-alpha reduction or aromatization, meaning its androgenic "
                "effects are mediated entirely through direct androgen receptor "
                "binding. Bioavailability studies in bovine models show 95% "
                "absorption from intramuscular depot within 14 days."
            ),
            (
                "Hepatotoxicity markers ALT/AST elevated 3x ULN. Cholesterol "
                "HDL suppressed to 15 mg/dL. Cardiovascular risk assessment "
                "shows LVH progression on echocardiogram. Iron overload in "
                "hepatocytes compounds the toxicity profile — when ferritin "
                "exceeds 500 ng/mL, Fenton chemistry generates hydroxyl radicals "
                "that damage mitochondrial membranes. Combined with the direct "
                "hepatocellular stress from trenbolone metabolites, this creates "
                "a synergistic injury pattern. Kidney function markers remained "
                "within normal limits in most subjects, though cystatin C showed "
                "a subtle upward trend suggesting early glomerular stress."
            ),
            (
                "Testosterone suppression to castrate levels within 72h. "
                "LH/FSH undetectable. Prolactin elevated 2x. Estradiol "
                "paradoxically low due to lack of aromatization substrate. "
                "The hypothalamic-pituitary-gonadal axis shows near-complete "
                "suppression within days of trenbolone administration, with "
                "recovery taking 4-8 weeks post-cessation depending on cycle "
                "length. Interestingly, IGF-1 levels remain elevated during "
                "suppression, suggesting hepatic GH receptor sensitization "
                "independent of gonadal status. Thyroid function markers TSH "
                "and T3/T4 were unaffected in most subjects, though a subset "
                "showed mildly elevated reverse T3 indicating stress response."
            ),
        ]
        return [
            WorkerAssignment(worker_id=i, angle=angles[i], raw_content=contents[i])
            for i in range(n)
        ]

    def test_misassignment_injects_off_angle_content(self):
        """Each worker should receive off-angle data from a distant worker."""
        assignments = self._make_assignments()
        original_contents = [a.raw_content for a in assignments]

        apply_misassignment(assignments, ratio=0.25)

        for i, a in enumerate(assignments):
            # Content should be longer (off-angle appended)
            assert len(a.raw_content) > len(original_contents[i])
            # Off-angle marker should be present
            assert "OFF-ANGLE DATA" in a.raw_content
            # Original content should still be there
            assert original_contents[i] in a.raw_content

    def test_misassignment_preserves_original_content(self):
        """On-angle content must not be removed or truncated."""
        assignments = self._make_assignments()
        original_contents = [a.raw_content for a in assignments]

        apply_misassignment(assignments, ratio=0.30)

        for i, a in enumerate(assignments):
            assert a.raw_content.startswith(original_contents[i])

    def test_misassignment_ratio_controls_injection_size(self):
        """Higher ratio → more off-angle content injected."""
        assignments_low = self._make_assignments()
        assignments_high = self._make_assignments()

        apply_misassignment(assignments_low, ratio=0.10)
        apply_misassignment(assignments_high, ratio=0.40)

        for low, high in zip(assignments_low, assignments_high):
            assert len(high.raw_content) >= len(low.raw_content)

    def test_misassignment_zero_ratio_is_noop(self):
        """Ratio of 0 should not inject anything."""
        assignments = self._make_assignments()
        original_contents = [a.raw_content for a in assignments]

        apply_misassignment(assignments, ratio=0.0)

        for i, a in enumerate(assignments):
            assert a.raw_content == original_contents[i]

    def test_misassignment_single_worker_is_noop(self):
        """Cannot misassign with fewer than 2 workers."""
        assignments = self._make_assignments(1)
        original = assignments[0].raw_content

        apply_misassignment(assignments, ratio=0.25)

        assert assignments[0].raw_content == original

    def test_misassignment_uses_score_matrix_for_distance(self):
        """With a score matrix, the most distant angle (lowest score) is selected."""
        assignments = self._make_assignments(3)

        # Score matrix: worker 0 is close to worker 1, far from worker 2
        score_matrix = [
            [10.0, 9.0, 1.0],   # worker 0: close to 1, far from 2
            [8.0, 10.0, 2.0],   # worker 1: close to 0, far from 2
            [1.0, 2.0, 10.0],   # worker 2: far from 0, far from 1
        ]

        apply_misassignment(assignments, score_matrix=score_matrix, ratio=0.25)

        # Worker 0 should get content from worker 2 (lowest score)
        assert assignments[2].angle in assignments[0].raw_content
        # Worker 2 should get content from worker 0 (lowest score)
        assert assignments[0].angle in assignments[2].raw_content

    def test_misassignment_positional_fallback(self):
        """Without a score matrix, uses maximum positional distance."""
        assignments = self._make_assignments(4)

        apply_misassignment(assignments, score_matrix=None, ratio=0.25)

        # Worker 0 should pair with worker 2 (N//2 = 2)
        assert assignments[2].angle in assignments[0].raw_content
        # Worker 1 should pair with worker 3
        assert assignments[3].angle in assignments[1].raw_content

    def test_misassignment_updates_char_count(self):
        """char_count should reflect the new (larger) raw_content."""
        assignments = self._make_assignments()

        apply_misassignment(assignments, ratio=0.25)

        for a in assignments:
            assert a.char_count == len(a.raw_content)

    def test_misassignment_cross_angle_content_is_raw(self):
        """Off-angle data should be raw content, not summaries."""
        assignments = self._make_assignments()
        # Worker 0 is Molecular Biology, should get raw data from a distant worker
        distant_content = assignments[2].raw_content  # positional N//2

        apply_misassignment(assignments, ratio=0.50)

        # The injected content should be a prefix of the distant worker's raw data
        off_angle_section = assignments[0].raw_content.split("OFF-ANGLE DATA")[1]
        # The actual raw content from the distant worker should appear
        expected_prefix = distant_content[:int(len(distant_content) * 0.50)]
        assert expected_prefix in off_angle_section


# ═══════════════════════════════════════════════════════════════════════
# 2. Swarm-Internal RAG (Hive Memory)
# ═══════════════════════════════════════════════════════════════════════


class TestExtractConcepts:
    """Test key concept extraction for RAG queries."""

    def test_extracts_frequent_terms(self):
        text = (
            "Iron sequestration via hepcidin regulation affects iron "
            "metabolism. Hepcidin controls iron absorption and iron "
            "recycling from macrophages."
        )
        concepts = extract_concepts(text, top_k=5)
        assert "iron" in concepts
        assert "hepcidin" in concepts

    def test_filters_stopwords(self):
        text = "The analysis of the data shows that the evidence indicates"
        concepts = extract_concepts(text, top_k=10)
        # Common words should be filtered out
        for stopword in ["the", "that", "shows"]:
            assert stopword not in concepts

    def test_empty_text_returns_empty(self):
        assert extract_concepts("") == []
        assert extract_concepts("   ") == []

    def test_respects_top_k(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
        concepts = extract_concepts(text, top_k=3)
        assert len(concepts) <= 3

    def test_returns_most_frequent_first(self):
        text = "iron iron iron hepcidin hepcidin ferritin"
        concepts = extract_concepts(text, top_k=3)
        assert concepts[0] == "iron"
        assert concepts[1] == "hepcidin"


class TestScoreRelevance:
    """Test keyword overlap scoring."""

    def test_exact_match_scores_high(self):
        score = score_relevance(
            ["iron", "hepcidin"],
            "Iron sequestration via hepcidin regulation",
        )
        assert score >= 2.0  # both concepts match

    def test_no_match_scores_zero(self):
        score = score_relevance(
            ["quantum", "photon"],
            "Iron sequestration via hepcidin regulation",
        )
        assert score == 0.0

    def test_partial_match(self):
        score = score_relevance(
            ["iron", "quantum"],
            "Iron metabolism is critical",
        )
        assert 0.5 <= score <= 1.5  # one match

    def test_empty_inputs(self):
        assert score_relevance([], "some text") == 0.0
        assert score_relevance(["iron"], "") == 0.0


class TestQueryHive:
    """Test targeted retrieval from accumulated lineage entries."""

    @staticmethod
    def _make_entries() -> list[LineageEntry]:
        """Create diverse lineage entries from different angles."""
        return [
            LineageEntry(
                entry_id="e1",
                phase="worker_synthesis",
                content=(
                    "Iron sequestration mechanism: hepcidin suppression leads "
                    "to increased iron absorption from hepatic stores. Ferritin "
                    "elevated 340% in treated cohort."
                ),
                angle="Molecular Biology",
            ),
            LineageEntry(
                entry_id="e2",
                phase="gossip_round_1",
                content=(
                    "Hematocrit went from 42 to 49.6 after 8 weeks on "
                    "trenbolone. Hemoglobin at 16.5 g/dL suggests significant "
                    "erythropoiesis upregulation."
                ),
                angle="Practitioner Findings",
            ),
            LineageEntry(
                entry_id="e3",
                phase="worker_synthesis",
                content=(
                    "Trenbolone acetate clearance via CYP3A4. Half-life "
                    "48-72h. No aromatization to estradiol."
                ),
                angle="Pharmacokinetics",
            ),
            LineageEntry(
                entry_id="e4",
                phase="gossip_round_1",
                content=(
                    "Hepatotoxicity markers elevated. ALT 3x ULN. Iron "
                    "overload compounds liver stress in concurrent users."
                ),
                angle="Safety Profile",
            ),
            LineageEntry(
                entry_id="e5",
                phase="corpus_analysis",
                content="Detected 5 sections, 5 angles",
                angle="",
            ),
        ]

    def test_query_excludes_same_angle(self):
        """Hive query should not return entries from the querying bee's own angle."""
        entries = self._make_entries()
        results = query_hive(
            entries=entries,
            concepts=["iron", "hepcidin", "ferritin"],
            exclude_angle="Molecular Biology",
            top_k=5,
            min_score=0.5,
        )
        for r in results:
            assert "Molecular Biology" not in r.split("]")[0]

    def test_query_returns_relevant_cross_angle_findings(self):
        """Should find iron-related findings from other angles."""
        entries = self._make_entries()
        results = query_hive(
            entries=entries,
            concepts=["iron", "hepatotoxicity", "liver"],
            exclude_angle="Molecular Biology",
            top_k=5,
            min_score=0.5,
        )
        assert len(results) > 0
        # Safety Profile entry mentions iron overload + liver
        found_safety = any("Safety Profile" in r for r in results)
        assert found_safety

    def test_query_skips_corpus_analysis(self):
        """corpus_analysis entries are metadata, not bee output."""
        entries = self._make_entries()
        results = query_hive(
            entries=entries,
            concepts=["sections", "angles", "detected"],
            exclude_angle="SomeAngle",
            top_k=5,
            min_score=0.5,
        )
        for r in results:
            assert "corpus_analysis" not in r.lower()

    def test_query_respects_top_k(self):
        entries = self._make_entries()
        results = query_hive(
            entries=entries,
            concepts=["iron", "hepcidin", "hematocrit", "hepatotoxicity"],
            exclude_angle="Endocrinology",
            top_k=2,
            min_score=0.5,
        )
        assert len(results) <= 2

    def test_query_empty_entries(self):
        results = query_hive(
            entries=[],
            concepts=["iron"],
            exclude_angle="Molecular Biology",
        )
        assert results == []

    def test_query_empty_concepts(self):
        entries = self._make_entries()
        results = query_hive(
            entries=entries,
            concepts=[],
            exclude_angle="Molecular Biology",
        )
        assert results == []

    def test_query_formats_results_with_attribution(self):
        """Results should include angle and phase for attribution."""
        entries = self._make_entries()
        results = query_hive(
            entries=entries,
            concepts=["iron", "hepcidin", "ferritin", "hepatotoxicity"],
            exclude_angle="Endocrinology",
            top_k=5,
            min_score=0.5,
        )
        for r in results:
            # Should be formatted as [angle — Phase] content
            assert r.startswith("[")
            assert "—" in r


# ═══════════════════════════════════════════════════════════════════════
# 3. Core Interest Worldview Prompts
# ═══════════════════════════════════════════════════════════════════════


class TestWorldviewSynthPrompt:
    """Test that synthesis prompts establish deep domain identity."""

    def test_prompt_establishes_specialist_identity(self):
        prompt = _build_synth_prompt(
            date="2025-01-15",
            angle="Molecular Biology",
            char_count=5000,
            section_content="Test content here",
            query="How does trenbolone affect iron metabolism?",
            max_chars=10000,
        )
        # Should establish specialist identity, not generic analyst
        assert "Molecular Biology specialist" in prompt
        assert "through the lens of Molecular Biology" in prompt

    def test_prompt_instructs_lens_based_reasoning(self):
        prompt = _build_synth_prompt(
            date="2025-01-15",
            angle="Pharmacokinetics",
            char_count=3000,
            section_content="Some PK data",
            query="What are the absorption characteristics?",
            max_chars=8000,
        )
        assert "EXPLAIN it through your domain" in prompt
        assert "mechanisms, frameworks, and first principles" in prompt

    def test_prompt_mentions_off_angle_data(self):
        """Prompt should instruct the bee to pay attention to off-angle data."""
        prompt = _build_synth_prompt(
            date="2025-01-15",
            angle="Safety Profile",
            char_count=4000,
            section_content="Safety data mixed with some PK data",
            query="What are the risks?",
            max_chars=10000,
        )
        assert "OFF-ANGLE" in prompt or "off-angle" in prompt or "OTHER DOMAINS" in prompt

    def test_prompt_uses_core_identity_label(self):
        """Should use 'CORE IDENTITY' instead of 'ASSIGNED ANGLE'."""
        prompt = _build_synth_prompt(
            date="2025-01-15",
            angle="Endocrinology",
            char_count=2000,
            section_content="Hormone data",
            query="What about hormones?",
            max_chars=8000,
        )
        assert "CORE IDENTITY" in prompt

    def test_prompt_requests_cross_domain_implications(self):
        prompt = _build_synth_prompt(
            date="2025-01-15",
            angle="Molecular Biology",
            char_count=5000,
            section_content="Test content",
            query="Test query",
            max_chars=10000,
        )
        assert "CROSS-DOMAIN IMPLICATIONS" in prompt or "IMPLICATIONS FOR OTHER" in prompt

    def test_prompt_requests_lens_based_analysis_output(self):
        """Final instruction should reference the angle's lens."""
        prompt = _build_synth_prompt(
            date="2025-01-15",
            angle="Practitioner Findings",
            char_count=3000,
            section_content="Clinical observations",
            query="What do practitioners observe?",
            max_chars=8000,
        )
        assert "Practitioner Findings lens" in prompt


class TestWorldviewGossipPrompt:
    """Test that gossip prompts maintain deep domain identity."""

    def test_gossip_establishes_specialist_identity(self):
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Molecular Biology",
            raw_section_block="",
            own_summary="My analysis of iron metabolism...",
            n_peers=3,
            peers_text="Peer 1: hematocrit data...",
            max_chars=10000,
        )
        assert "Molecular Biology specialist" in prompt
        assert "lens of Molecular Biology" in prompt

    def test_gossip_includes_hive_memory_when_provided(self):
        hive_data = (
            "[Safety Profile — Gossip Round 1] Iron overload compounds "
            "liver stress in concurrent users."
        )
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Molecular Biology",
            raw_section_block="",
            own_summary="My iron metabolism analysis...",
            n_peers=2,
            peers_text="Peer findings...",
            max_chars=10000,
            hive_memory=hive_data,
        )
        assert "FROM THE HIVE" in prompt
        assert hive_data in prompt
        assert "Molecular Biology lens" in prompt or "Molecular Biology" in prompt

    def test_gossip_no_hive_block_when_empty(self):
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Pharmacokinetics",
            raw_section_block="",
            own_summary="PK analysis...",
            n_peers=2,
            peers_text="Peer findings...",
            max_chars=10000,
            hive_memory="",
        )
        assert "FROM THE HIVE" not in prompt

    def test_gossip_uses_core_identity_label(self):
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Endocrinology",
            raw_section_block="",
            own_summary="Hormone analysis...",
            n_peers=2,
            peers_text="Peer findings...",
            max_chars=8000,
        )
        assert "CORE IDENTITY" in prompt

    def test_gossip_evidence_chain_references_angle(self):
        """Evidence chain instructions should reference the bee's domain."""
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Safety Profile",
            raw_section_block="",
            own_summary="Safety analysis...",
            n_peers=2,
            peers_text="Peer data...",
            max_chars=8000,
        )
        assert "Safety Profile" in prompt
        # Should appear multiple times — identity is woven throughout
        assert prompt.count("Safety Profile") >= 3

    def test_gossip_includes_delta_text_when_provided(self):
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Molecular Biology",
            raw_section_block="",
            own_summary="Analysis...",
            n_peers=2,
            peers_text="Peer data...",
            max_chars=10000,
            delta_text="New finding about hepcidin...",
        )
        assert "NEW EVIDENCE" in prompt
        assert "hepcidin" in prompt

    def test_gossip_includes_both_hive_and_delta(self):
        """When both hive memory and delta text are present, both should appear."""
        prompt = _build_gossip_prompt(
            date="2025-01-15",
            angle="Molecular Biology",
            raw_section_block="",
            own_summary="Analysis...",
            n_peers=2,
            peers_text="Peer data...",
            max_chars=10000,
            delta_text="New delta finding...",
            hive_memory="Hive memory finding...",
        )
        assert "FROM THE HIVE" in prompt
        assert "NEW EVIDENCE" in prompt
        assert "Hive memory finding" in prompt
        assert "New delta finding" in prompt


# ═══════════════════════════════════════════════════════════════════════
# 4. Integration: Engine Wiring
# ═══════════════════════════════════════════════════════════════════════


class TestEngineIntegration:
    """Verify the engine wires misassignment and hive memory correctly."""

    def test_swarm_config_has_thread_discovery_settings(self):
        from swarm.config import SwarmConfig
        config = SwarmConfig()
        # Misassignment defaults
        assert config.enable_misassignment is True
        assert config.misassignment_ratio == 0.25
        # Hive memory defaults
        assert config.enable_hive_memory is True
        assert config.hive_memory_top_k == 5

    def test_swarm_config_overrides_via_constructor(self):
        """Config knobs can be overridden via constructor kwargs."""
        from swarm.config import SwarmConfig
        config = SwarmConfig(
            enable_misassignment=False,
            misassignment_ratio=0.35,
            enable_hive_memory=False,
            hive_memory_top_k=10,
        )
        assert config.enable_misassignment is False
        assert config.misassignment_ratio == 0.35
        assert config.enable_hive_memory is False
        assert config.hive_memory_top_k == 10

    def test_gossip_swarm_initializes_lineage_cache(self):
        from swarm.engine import GossipSwarm

        async def mock_complete(prompt: str) -> str:
            return "mock response"

        swarm = GossipSwarm(complete=mock_complete)
        assert hasattr(swarm, "_lineage_entries")
        assert swarm._lineage_entries == []

    def test_emit_tracks_entries_locally(self):
        from swarm.engine import GossipSwarm

        async def mock_complete(prompt: str) -> str:
            return "mock response"

        swarm = GossipSwarm(complete=mock_complete)
        entry = LineageEntry(
            entry_id="test1",
            phase="worker_synthesis",
            content="Test content",
            angle="Molecular Biology",
        )
        swarm._emit(entry)
        assert len(swarm._lineage_entries) == 1
        assert swarm._lineage_entries[0].entry_id == "test1"
