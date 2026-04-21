# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the KnowledgePlugin (cross-session learning)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knowledge_store import Insight, KnowledgeStore, reset_knowledge_store
from plugins.knowledge import KnowledgePlugin, _KNOWLEDGE_MARKER


@pytest.fixture
def store():
    """Create an in-memory KnowledgeStore for testing."""
    s = KnowledgeStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def plugin(store):
    """Create a KnowledgePlugin with in-memory store."""
    return KnowledgePlugin(store=store)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton is reset between tests."""
    reset_knowledge_store()
    yield
    reset_knowledge_store()


class TestKnowledgeInjection:
    """BeforeInvocationEvent: inject prior knowledge."""

    def test_no_injection_when_store_empty(self, plugin: KnowledgePlugin) -> None:
        """No knowledge marker injected when store has no matches."""
        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        plugin.inject_knowledge(event)

        # No knowledge marker should be present
        for msg in event.messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            assert _KNOWLEDGE_MARKER not in block.get("text", "")

    def test_injection_when_relevant_knowledge_exists(
        self, plugin: KnowledgePlugin, store: KnowledgeStore
    ) -> None:
        """Prior knowledge is injected when matching insights exist."""
        store.store_insight(Insight(
            fact="GLP-1 receptor agonists reduce HbA1c by 1.0-1.5%",
            topic="GLP-1 pharmacology",
            confidence=0.9,
        ))

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1 receptor agonists"}]},
        ]
        plugin.inject_knowledge(event)

        # Should have injected a knowledge message
        has_marker = False
        for msg in event.messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and _KNOWLEDGE_MARKER in block.get("text", ""):
                            has_marker = True
                            assert "GLP-1" in block["text"]
        assert has_marker

    def test_stale_knowledge_markers_stripped(
        self, plugin: KnowledgePlugin, store: KnowledgeStore
    ) -> None:
        """Old knowledge markers from previous turns are removed."""
        store.store_insight(Insight(
            fact="Existing fact about peptides",
            topic="peptides",
        ))

        # Simulate messages with an old knowledge marker
        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": f"{_KNOWLEDGE_MARKER}\nOld knowledge block"}]},
            {"role": "assistant", "content": [{"text": "Previous response"}]},
            {"role": "user", "content": [{"text": "new query about peptides"}]},
        ]
        plugin.inject_knowledge(event)

        # Count knowledge markers — should be exactly 1 (fresh injection)
        marker_count = 0
        for msg in event.messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and _KNOWLEDGE_MARKER in block.get("text", ""):
                            marker_count += 1
        assert marker_count <= 1

    def test_no_injection_on_empty_messages(self, plugin: KnowledgePlugin) -> None:
        """Plugin handles None/empty messages gracefully."""
        event = MagicMock()
        event.messages = None
        plugin.inject_knowledge(event)  # Should not raise

        event.messages = []
        plugin.inject_knowledge(event)  # Should not raise


class TestKnowledgeAccumulation:
    """AfterInvocationEvent: extract and store findings."""

    def test_facts_extracted_from_assistant_response(
        self, plugin: KnowledgePlugin, store: KnowledgeStore
    ) -> None:
        """Factual statements are extracted and stored."""
        # Set up the query context
        before_event = MagicMock()
        before_event.messages = [
            {"role": "user", "content": [{"text": "research GLP-1 receptor agonists"}]},
        ]
        plugin.inject_knowledge(before_event)

        # Simulate assistant response with factual claims
        after_event = MagicMock()
        after_event.messages = [
            {"role": "user", "content": [{"text": "research GLP-1 receptor agonists"}]},
            {"role": "assistant", "content": [{"text":
                "Semaglutide was approved by the FDA in 2017 for type 2 diabetes. "
                "In the SUSTAIN-6 trial, semaglutide reduced major cardiovascular events by 26%. "
                "The recommended starting dose is 0.25mg subcutaneous injection weekly."
            }]},
        ]
        plugin.accumulate_knowledge(after_event)

        # Should have stored some insights
        assert store.count_insights() > 0

    def test_meta_commentary_not_stored(
        self, plugin: KnowledgePlugin, store: KnowledgeStore
    ) -> None:
        """Agent meta-commentary (let me search, I'll look into) is filtered out."""
        before_event = MagicMock()
        before_event.messages = [
            {"role": "user", "content": [{"text": "find info about X"}]},
        ]
        plugin.inject_knowledge(before_event)

        after_event = MagicMock()
        after_event.messages = [
            {"role": "user", "content": [{"text": "find info about X"}]},
            {"role": "assistant", "content": [{"text":
                "Let me search for information about this topic. "
                "I'll look into the latest research on this. "
                "Based on my search, here are the results."
            }]},
        ]
        plugin.accumulate_knowledge(after_event)

        # Meta-commentary should be filtered — no insights stored
        assert store.count_insights() == 0

    def test_entities_extracted(
        self, plugin: KnowledgePlugin, store: KnowledgeStore
    ) -> None:
        """Named entities are extracted and tracked."""
        before_event = MagicMock()
        before_event.messages = [
            {"role": "user", "content": [{"text": "research BPC-157 peptide"}]},
        ]
        plugin.inject_knowledge(before_event)

        after_event = MagicMock()
        after_event.messages = [
            {"role": "user", "content": [{"text": "research BPC-157 peptide"}]},
            {"role": "assistant", "content": [{"text":
                "BPC-157 is a synthetic peptide studied by the FDA and WHO for "
                "its potential gastric protection properties."
            }]},
        ]
        plugin.accumulate_knowledge(after_event)

        assert store.count_entities() > 0

    def test_duplicate_facts_not_stored(
        self, plugin: KnowledgePlugin, store: KnowledgeStore
    ) -> None:
        """Near-duplicate facts from different conversations are deduplicated."""
        # Store initial fact
        store.store_insight(Insight(
            fact="Semaglutide was approved by the FDA in 2017 for type 2 diabetes",
        ))

        before_event = MagicMock()
        before_event.messages = [
            {"role": "user", "content": [{"text": "research semaglutide"}]},
        ]
        plugin.inject_knowledge(before_event)

        after_event = MagicMock()
        after_event.messages = [
            {"role": "user", "content": [{"text": "research semaglutide"}]},
            {"role": "assistant", "content": [{"text":
                "Semaglutide was approved by the FDA in 2017 for type 2 diabetes treatment."
            }]},
        ]
        plugin.accumulate_knowledge(after_event)

        # Should still be 1 (duplicate detected)
        assert store.count_insights() == 1


class TestExtractionHelpers:
    """Unit tests for static extraction methods."""

    def test_extract_facts_with_numbers(self) -> None:
        """Sentences with numbers are extracted as facts."""
        text = (
            "The study enrolled 3,297 participants across 16 countries. "
            "Results showed a 26% reduction in cardiovascular events. "
            "This is very interesting indeed."
        )
        facts = KnowledgePlugin._extract_facts(text)
        assert len(facts) >= 1
        assert any("3,297" in f[0] or "26%" in f[0] for f in facts)

    def test_extract_facts_filters_meta(self) -> None:
        """Meta-commentary sentences are filtered out."""
        text = "Let me search for this. I'll look into it. Here are the results."
        facts = KnowledgePlugin._extract_facts(text)
        assert len(facts) == 0

    def test_extract_facts_with_urls(self) -> None:
        """URLs are extracted as source and removed from fact text."""
        text = "The paper at https://pubmed.ncbi.nlm.nih.gov/12345 found 50% improvement in outcomes."
        facts = KnowledgePlugin._extract_facts(text)
        assert len(facts) >= 1
        if facts:
            fact_text, source_url = facts[0]
            assert "https://" not in fact_text
            assert "pubmed" in source_url

    def test_extract_entities_compounds(self) -> None:
        """Hyphenated compound names extracted."""
        text = "BPC-157 and GLP-1 are important peptides in the research."
        entities = KnowledgePlugin._extract_entities(text)
        names = [e[0] for e in entities]
        assert "BPC-157" in names
        assert "GLP-1" in names

    def test_extract_entities_organizations(self) -> None:
        """Uppercase abbreviations extracted as organizations."""
        text = "The FDA and WHO both issued guidelines on this topic."
        entities = KnowledgePlugin._extract_entities(text)
        names = [e[0] for e in entities]
        assert "FDA" in names
        assert "WHO" in names

    def test_infer_source_type(self) -> None:
        """Source types inferred correctly from URLs."""
        assert KnowledgePlugin._infer_source_type("https://pubmed.ncbi.nlm.nih.gov/123") == "academic"
        assert KnowledgePlugin._infer_source_type("https://arxiv.org/abs/2301.1234") == "preprint"
        assert KnowledgePlugin._infer_source_type("https://clinicaltrials.gov/ct2/show/NCT123") == "government"
        assert KnowledgePlugin._infer_source_type("https://reddit.com/r/science") == "forum"
        assert KnowledgePlugin._infer_source_type("") == "research"

    def test_infer_topic(self) -> None:
        """Topic inferred from query by removing stop words."""
        topic = KnowledgePlugin._infer_topic("find papers on GLP-1 pharmacokinetics")
        assert "papers" not in topic
        assert "find" not in topic
        # Should keep significant words
        assert "glp-1" in topic or "pharmacokinetics" in topic

    def test_is_knowledge_message(self) -> None:
        """Knowledge marker detection works."""
        msg_with = {"role": "user", "content": [{"text": f"{_KNOWLEDGE_MARKER}\nSome knowledge"}]}
        msg_without = {"role": "user", "content": [{"text": "Normal user message"}]}
        msg_assistant = {"role": "assistant", "content": [{"text": f"{_KNOWLEDGE_MARKER}"}]}

        assert KnowledgePlugin._is_knowledge_message(msg_with)
        assert not KnowledgePlugin._is_knowledge_message(msg_without)
        assert not KnowledgePlugin._is_knowledge_message(msg_assistant)
