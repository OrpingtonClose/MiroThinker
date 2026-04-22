# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Unit tests for swarm/research_organizer.py."""

from __future__ import annotations

import json
import pytest

from swarm.research_organizer import (
    ResearchNeed,
    ResearchOrganizerConfig,
    _extract_doubts_heuristic,
    count_uncertainty_signals,
    extract_doubts,
)


# ---------------------------------------------------------------------------
# Uncertainty signal detection
# ---------------------------------------------------------------------------

class TestCountUncertaintySignals:
    """Detect uncertainty/doubt signals in worker transcripts."""

    def test_detects_explicit_uncertainty(self) -> None:
        transcript = (
            "The dose-response curve is uncertain. I need data on "
            "the intermediate range. This claim is unverified."
        )
        count = count_uncertainty_signals(transcript)
        assert count >= 3

    def test_detects_hedging(self) -> None:
        transcript = "This might be related. Perhaps the mechanism involves..."
        count = count_uncertainty_signals(transcript)
        assert count >= 2

    def test_zero_for_confident_text(self) -> None:
        transcript = (
            "Insulin binds to the receptor. The half-life is 4 hours. "
            "Dosage is 10 IU pre-workout."
        )
        count = count_uncertainty_signals(transcript)
        assert count == 0

    def test_handles_empty_string(self) -> None:
        assert count_uncertainty_signals("") == 0


# ---------------------------------------------------------------------------
# Heuristic doubt extraction
# ---------------------------------------------------------------------------

class TestHeuristicDoubtExtraction:
    """Sentence-level doubt extraction when LLM fails."""

    def test_extracts_sentences_with_signals(self) -> None:
        transcript = (
            "Insulin timing is well-established at 15 minutes pre-workout. "
            "However, the GH interaction window is unclear and I need data "
            "to verify the 300mg intermediate range. "
            "Trenbolone effects on hematocrit are well-documented."
        )
        needs = _extract_doubts_heuristic(transcript, "GH_timing", "worker_1", max_doubts=5)
        assert len(needs) >= 1
        assert all(n.angle == "GH_timing" for n in needs)
        assert all(n.source_worker_id == "worker_1" for n in needs)

    def test_respects_max_doubts(self) -> None:
        transcript = (
            "This is uncertain. "
            "That is uncertain. "
            "Another uncertain claim. "
            "More uncertainty here. "
            "And more uncertain things."
        )
        needs = _extract_doubts_heuristic(transcript, "test", "w1", max_doubts=2)
        assert len(needs) <= 2

    def test_empty_transcript_returns_empty(self) -> None:
        needs = _extract_doubts_heuristic("", "test", "w1", max_doubts=3)
        assert needs == []

    def test_priority_scales_with_signal_count(self) -> None:
        # More signals = higher priority
        transcript = (
            "This is uncertain and unverified, I need data to verify."
        )
        needs = _extract_doubts_heuristic(transcript, "test", "w1", max_doubts=5)
        if needs:
            assert needs[0].priority > 0.3  # Should be elevated


# ---------------------------------------------------------------------------
# LLM-powered doubt extraction
# ---------------------------------------------------------------------------

class TestExtractDoubts:
    """LLM-powered doubt extraction from worker transcripts."""

    @pytest.mark.asyncio
    async def test_valid_json_extraction(self) -> None:
        llm_response = json.dumps([
            {
                "doubt": "Uncertain about 300mg tren hematocrit",
                "data_needed": "bloodwork data 300mg/week 8+ weeks",
                "priority": "HIGH",
            },
            {
                "doubt": "GH timing unclear relative to insulin",
                "data_needed": "pharmacokinetic study GH-insulin window",
                "priority": "MEDIUM",
            },
        ])

        async def mock_complete(prompt: str) -> str:
            return llm_response

        needs = await extract_doubts(
            transcript="some transcript with doubts",
            angle="hematology",
            worker_id="worker_3",
            complete=mock_complete,
            max_doubts=3,
        )
        assert len(needs) == 2
        assert needs[0].priority > needs[1].priority  # HIGH > MEDIUM
        assert needs[0].angle == "hematology"
        assert needs[0].source_worker_id == "worker_3"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_heuristic(self) -> None:
        async def mock_complete(prompt: str) -> str:
            raise RuntimeError("LLM down")

        needs = await extract_doubts(
            transcript="This is uncertain and I need data to verify the claim.",
            angle="pharmacology",
            worker_id="worker_1",
            complete=mock_complete,
            max_doubts=3,
        )
        # Should fall back to heuristic extraction
        assert isinstance(needs, list)

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(self) -> None:
        async def mock_complete(prompt: str) -> str:
            return "not valid json at all"

        needs = await extract_doubts(
            transcript="This is uncertain and I need data to verify.",
            angle="pharmacology",
            worker_id="w1",
            complete=mock_complete,
            max_doubts=3,
        )
        assert isinstance(needs, list)

    @pytest.mark.asyncio
    async def test_markdown_fenced_json(self) -> None:
        llm_response = "```json\n" + json.dumps([
            {"doubt": "test doubt", "data_needed": "test data", "priority": "LOW"},
        ]) + "\n```"

        async def mock_complete(prompt: str) -> str:
            return llm_response

        needs = await extract_doubts(
            transcript="transcript",
            angle="test",
            worker_id="w1",
            complete=mock_complete,
            max_doubts=3,
        )
        assert len(needs) == 1
        assert needs[0].priority == 0.3  # LOW maps to 0.3

    @pytest.mark.asyncio
    async def test_respects_max_doubts(self) -> None:
        llm_response = json.dumps([
            {"doubt": f"doubt {i}", "data_needed": f"data {i}", "priority": "MEDIUM"}
            for i in range(10)
        ])

        async def mock_complete(prompt: str) -> str:
            return llm_response

        needs = await extract_doubts(
            transcript="transcript",
            angle="test",
            worker_id="w1",
            complete=mock_complete,
            max_doubts=3,
        )
        assert len(needs) <= 3


# ---------------------------------------------------------------------------
# ResearchNeed data class
# ---------------------------------------------------------------------------

class TestResearchNeed:
    """ResearchNeed data class behavior."""

    def test_default_values(self) -> None:
        need = ResearchNeed(
            angle="test",
            doubt="something uncertain",
            data_needed="specific data",
        )
        assert need.priority == 0.5
        assert need.source_worker_id == ""

    def test_sorting_by_priority(self) -> None:
        needs = [
            ResearchNeed(angle="a", doubt="low", data_needed="d", priority=0.3),
            ResearchNeed(angle="b", doubt="high", data_needed="d", priority=0.9),
            ResearchNeed(angle="c", doubt="med", data_needed="d", priority=0.6),
        ]
        needs.sort(key=lambda n: n.priority, reverse=True)
        assert needs[0].doubt == "high"
        assert needs[1].doubt == "med"
        assert needs[2].doubt == "low"


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestResearchOrganizerConfig:
    """Configuration defaults."""

    def test_default_values(self) -> None:
        config = ResearchOrganizerConfig()
        assert config.max_doubts_per_worker == 3
        assert config.max_clones == 4
        assert config.max_searches_per_clone == 5
        assert config.max_extractions_per_clone == 2
        assert config.trigger_every_n_waves == 2
        assert config.trigger_uncertainty_threshold == 3
