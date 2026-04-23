# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Unit tests for swarm/corpus_builder.py."""

from __future__ import annotations

import json
import pytest

from swarm.corpus_builder import (
    CorpusBuilderConfig,
    QueryComprehension,
    _assemble_corpus,
    _fallback_comprehension,
    _select_diverse_urls,
    comprehend_query,
)


# ---------------------------------------------------------------------------
# QueryComprehension tests
# ---------------------------------------------------------------------------

class TestFallbackComprehension:
    """Heuristic fallback when LLM comprehension fails."""

    def test_extracts_words_longer_than_3_chars(self) -> None:
        comp = _fallback_comprehension("Milos Sarcev insulin protocol bodybuilding")
        assert "milos" in comp.entities
        assert "sarcev" in comp.entities
        assert "insulin" in comp.entities
        assert "protocol" in comp.entities
        assert "bodybuilding" in comp.entities

    def test_generates_search_queries(self) -> None:
        comp = _fallback_comprehension("insulin timing GH interaction")
        assert len(comp.search_queries) >= 2
        assert "insulin timing GH interaction" in comp.search_queries[0]

    def test_sub_questions_contain_original_query(self) -> None:
        q = "test query for research"
        comp = _fallback_comprehension(q)
        assert comp.sub_questions == [q]


class TestComprehendQuery:
    """LLM-powered query comprehension."""

    @pytest.mark.asyncio
    async def test_valid_json_response(self) -> None:
        llm_response = json.dumps({
            "entities": ["insulin", "GH", "trenbolone"],
            "domains": ["endocrinology", "pharmacology"],
            "sub_questions": ["How does insulin timing affect GH?"],
            "academic_signals": True,
            "search_queries": ["insulin GH timing bodybuilding"],
        })

        async def mock_complete(prompt: str) -> str:
            return llm_response

        comp = await comprehend_query("test query", mock_complete)
        assert comp.entities == ["insulin", "GH", "trenbolone"]
        assert comp.academic_signals is True
        assert len(comp.search_queries) == 1

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self) -> None:
        llm_response = "```json\n" + json.dumps({
            "entities": ["insulin"],
            "domains": ["pharmacology"],
            "sub_questions": ["How?"],
            "academic_signals": False,
            "search_queries": ["query1"],
        }) + "\n```"

        async def mock_complete(prompt: str) -> str:
            return llm_response

        comp = await comprehend_query("test", mock_complete)
        assert comp.entities == ["insulin"]

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(self) -> None:
        async def mock_complete(prompt: str) -> str:
            return "this is not json"

        comp = await comprehend_query("insulin protocol research", mock_complete)
        # Falls back to heuristic — should have entities from words
        assert len(comp.entities) > 0
        assert len(comp.search_queries) > 0

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self) -> None:
        async def mock_complete(prompt: str) -> str:
            raise RuntimeError("LLM down")

        comp = await comprehend_query("insulin protocol", mock_complete)
        assert len(comp.entities) > 0

    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self) -> None:
        async def mock_complete(prompt: str) -> str:
            return ""

        comp = await comprehend_query("insulin protocol", mock_complete)
        assert len(comp.search_queries) > 0


# ---------------------------------------------------------------------------
# URL selection tests
# ---------------------------------------------------------------------------

class TestSelectDiverseUrls:
    """URL selection for content extraction."""

    def test_deduplicates_same_domain(self) -> None:
        results = [
            {"url": "https://example.com/page1"},
            {"url": "https://example.com/page2"},
            {"url": "https://other.org/page1"},
        ]
        urls = _select_diverse_urls(results, max_count=5)
        domains = [u.split("/")[2] for u in urls]
        assert len(set(domains)) == len(domains)

    def test_prioritises_edu_gov_org(self) -> None:
        results = [
            {"url": "https://random-blog.com/post"},
            {"url": "https://nih.gov/research/insulin"},
            {"url": "https://spam.com/seo"},
            {"url": "https://harvard.edu/paper"},
        ]
        urls = _select_diverse_urls(results, max_count=3)
        # .gov and .edu should come first
        assert any("nih.gov" in u for u in urls[:2])
        assert any("harvard.edu" in u for u in urls[:2])

    def test_skips_social_media(self) -> None:
        results = [
            {"url": "https://facebook.com/post"},
            {"url": "https://twitter.com/thread"},
            {"url": "https://real-content.com/article"},
        ]
        urls = _select_diverse_urls(results, max_count=5)
        assert all("facebook.com" not in u for u in urls)
        assert all("twitter.com" not in u for u in urls)

    def test_respects_max_count(self) -> None:
        results = [
            {"url": f"https://site{i}.com/page"} for i in range(20)
        ]
        urls = _select_diverse_urls(results, max_count=5)
        assert len(urls) <= 5

    def test_handles_empty_results(self) -> None:
        urls = _select_diverse_urls([], max_count=5)
        assert urls == []

    def test_handles_missing_urls(self) -> None:
        results = [{"title": "no url"}, {"url": ""}, {"url": "https://valid.com/page"}]
        urls = _select_diverse_urls(results, max_count=5)
        assert len(urls) == 1


# ---------------------------------------------------------------------------
# Corpus assembly tests
# ---------------------------------------------------------------------------

class TestAssembleCorpus:
    """Corpus assembly from search results."""

    def test_includes_all_sections(self) -> None:
        corpus = _assemble_corpus(
            search_results=[
                {"title": "Result", "url": "https://x.com", "snippet": "A snippet", "source": "brave"},
            ],
            extracted_articles=["[Source: Article — url]\n\nFull article text here"],
            academic_results=[
                {"title": "Paper", "url": "https://arxiv.org/1234", "snippet": "Abstract", "source": "arxiv"},
            ],
            query="test query",
        )
        assert "RESEARCH CORPUS" in corpus
        assert "FULL ARTICLES" in corpus
        assert "ACADEMIC PAPERS" in corpus
        assert "SEARCH SNIPPETS" in corpus
        assert "Full article text here" in corpus

    def test_includes_inline_text_results(self) -> None:
        corpus = _assemble_corpus(
            search_results=[
                {
                    "title": "Exa Result",
                    "url": "https://exa.ai/result",
                    "snippet": "short",
                    "full_text": "A" * 600,  # > 500 chars
                    "source": "exa",
                },
            ],
            extracted_articles=[],
            academic_results=[],
            query="test",
        )
        assert "SEARCH RESULTS WITH CONTENT" in corpus

    def test_empty_results_produces_minimal_corpus(self) -> None:
        corpus = _assemble_corpus(
            search_results=[],
            extracted_articles=[],
            academic_results=[],
            query="test",
        )
        assert "RESEARCH CORPUS" in corpus
        assert len(corpus) > 0

    def test_query_in_header(self) -> None:
        corpus = _assemble_corpus(
            search_results=[],
            extracted_articles=[],
            academic_results=[],
            query="Milos Sarcev insulin protocol",
        )
        assert "Milos Sarcev insulin protocol" in corpus
