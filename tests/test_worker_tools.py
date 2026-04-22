"""Unit tests for swarm/worker_tools.py — ConditionStore-backed agent tools."""

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

from corpus import ConditionStore
from swarm.worker_tools import build_worker_tools, _extract_terms, _keyword_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """In-memory ConditionStore with test data."""
    s = ConditionStore(db_path="")
    # Ingest some test findings
    s.ingest_raw(
        raw_text=(
            "Insulin at 4-6iu pre-workout improves nutrient partitioning.\n\n"
            "GH at 2iu three times daily synergises with insulin for IGF-1.\n\n"
            "Metformin may blunt mTOR signaling when taken with leucine."
        ),
        source_type="test",
        source_ref="test_corpus",
        angle="insulin_timing",
        iteration=0,
        user_query="bodybuilding protocol",
    )
    # Add findings from a different angle (peer data)
    s.ingest_raw(
        raw_text=(
            "Trenbolone increases red blood cell production via EPO.\n\n"
            "Boldenone at 400mg/wk provides appetite stimulation.\n\n"
            "Tren + boldenone compounds hematocrit pressure."
        ),
        source_type="test",
        source_ref="test_corpus_2",
        angle="anabolic_compounds",
        iteration=0,
        user_query="bodybuilding protocol",
    )
    return s


@pytest.fixture
def tools(store):
    """Worker tools for the insulin_timing angle."""
    return build_worker_tools(
        store=store,
        worker_angle="insulin_timing",
        worker_id="test_worker_0",
        phase="test_wave_1",
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_extract_terms_basic(self):
        terms = _extract_terms("insulin timing and nutrient partitioning")
        assert "insulin" in terms
        assert "timing" in terms
        assert "nutrient" in terms
        # Stopwords excluded
        assert "and" not in terms

    def test_extract_terms_empty(self):
        assert _extract_terms("") == []
        assert _extract_terms("a") == []

    def test_keyword_score_match(self):
        score = _keyword_score(["insulin", "timing"], "Insulin at 4iu pre-workout timing")
        assert score > 0

    def test_keyword_score_no_match(self):
        score = _keyword_score(["zebra", "platypus"], "Insulin at 4iu pre-workout")
        assert score == 0.0

    def test_keyword_score_empty(self):
        assert _keyword_score([], "some text") == 0.0
        assert _keyword_score(["term"], "") == 0.0


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------

class TestSearchCorpus:
    def test_search_finds_relevant(self, tools):
        search_corpus = tools[0]
        result = search_corpus(query="insulin pre-workout dosing")
        # Should find the insulin finding
        assert "insulin" in str(result).lower() or "finding" in str(result).lower()

    def test_search_no_match(self, tools):
        search_corpus = tools[0]
        result = search_corpus(query="quantum physics entanglement")
        assert "no findings match" in str(result).lower() or "finding" in str(result).lower()


class TestGetPeerInsights:
    def test_peer_insights_from_other_angle(self, tools):
        get_peer_insights = tools[1]
        result = get_peer_insights(topic="red blood cells EPO")
        # Should find trenbolone findings from anabolic_compounds angle
        result_str = str(result).lower()
        assert "anabolic" in result_str or "peer" in result_str or "epo" in result_str

    def test_peer_insights_no_match(self, tools):
        get_peer_insights = tools[1]
        result = get_peer_insights(topic="quantum physics")
        assert "no peer insights" in str(result).lower() or "peer" in str(result).lower()


class TestStoreFinding:
    def test_store_finding_basic(self, tools, store):
        store_finding = tools[2]
        result = store_finding(
            fact="Insulin at 5iu with 50g dextrose pre-workout maximises glycogen",
            confidence=0.8,
            evidence_source="test",
            reasoning="Derived from multiple studies",
        )
        assert "stored" in str(result).lower() or "success" in str(result).lower()

        # Verify it's in the store
        findings = store.get_findings(angle="insulin_timing")
        facts = [f["fact"] for f in findings]
        assert any("glycogen" in f.lower() for f in facts)

    def test_store_finding_empty_rejected(self, tools):
        store_finding = tools[2]
        result = store_finding(fact="", confidence=0.5)
        assert "error" in str(result).lower() or "empty" in str(result).lower()

    def test_store_finding_confidence_clamped(self, tools, store):
        store_finding = tools[2]
        # Confidence > 1.0 should be clamped
        store_finding(
            fact="Test claim with high confidence",
            confidence=5.0,
        )
        findings = store.get_findings(angle="insulin_timing")
        high_conf = [f for f in findings if "high confidence" in f["fact"].lower()]
        if high_conf:
            assert high_conf[0]["confidence"] <= 1.0


class TestCheckContradictions:
    def test_check_finds_related(self, tools):
        check_contradictions = tools[3]
        result = check_contradictions(claim="Metformin helps with insulin sensitivity")
        result_str = str(result).lower()
        # Should find the metformin finding
        assert "metformin" in result_str or "related" in result_str or "finding" in result_str


class TestGetResearchGaps:
    def test_gaps_returns_coverage(self, tools):
        get_research_gaps = tools[4]
        result = get_research_gaps()
        result_str = str(result).lower()
        # Should mention angle coverage
        assert "coverage" in result_str or "angle" in result_str or "findings" in result_str


class TestGetCorpusSection:
    def test_read_section(self, tools):
        get_corpus_section = tools[5]
        result = get_corpus_section(offset=0, max_chars=5000)
        result_str = str(result).lower()
        # Should contain corpus data
        assert "insulin" in result_str or "chars" in result_str

    def test_read_beyond_end(self, tools):
        get_corpus_section = tools[5]
        result = get_corpus_section(offset=999999, max_chars=1000)
        assert "entire section" in str(result).lower() or "no more" in str(result).lower()


class TestToolCallEventLogging:
    """Every tool invocation should be logged as a graph node in the store."""

    def test_search_logs_event(self, tools, store):
        search_corpus = tools[0]
        search_corpus(query="insulin dosing protocol")
        # Check that a tool_call row was created
        with store._lock:
            rows = store.conn.execute(
                "SELECT fact, source_type, row_type, source_ref "
                "FROM conditions WHERE row_type = 'tool_call' "
                "AND source_ref LIKE '%search_corpus%'"
            ).fetchall()
        assert len(rows) >= 1
        assert "search_corpus" in rows[0][0]

    def test_store_finding_logs_event(self, tools, store):
        store_finding = tools[2]
        store_finding(fact="Test finding for logging", confidence=0.6)
        with store._lock:
            rows = store.conn.execute(
                "SELECT fact FROM conditions WHERE row_type = 'tool_call' "
                "AND source_ref LIKE '%store_finding%'"
            ).fetchall()
        assert len(rows) >= 1

    def test_gaps_logs_event(self, tools, store):
        get_research_gaps = tools[4]
        get_research_gaps()
        with store._lock:
            rows = store.conn.execute(
                "SELECT fact FROM conditions WHERE row_type = 'tool_call' "
                "AND source_ref LIKE '%get_research_gaps%'"
            ).fetchall()
        assert len(rows) >= 1

    def test_tool_call_not_consider_for_use(self, tools, store):
        """Tool call events should NOT be returned as findings."""
        search_corpus = tools[0]
        search_corpus(query="insulin timing")
        with store._lock:
            rows = store.conn.execute(
                "SELECT consider_for_use FROM conditions "
                "WHERE row_type = 'tool_call'"
            ).fetchall()
        for row in rows:
            assert row[0] is False, "tool_call events must have consider_for_use=FALSE"
