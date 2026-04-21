# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the ToolRouterPlugin."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.domains import ACADEMIC, FORUM, GENERAL, PRACTITIONER
from plugins.tool_router import ToolRouterPlugin


class TestToolRouterPlugin:
    """Test ToolRouterPlugin query classification and guidance injection."""

    def setup_method(self) -> None:
        self.plugin = ToolRouterPlugin()

    def test_name(self) -> None:
        assert self.plugin.name == "tool-router"

    def test_extract_query_from_text_content(self) -> None:
        messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        query = ToolRouterPlugin._extract_query(messages)
        assert query == "find papers on GLP-1"

    def test_extract_query_from_string_content(self) -> None:
        messages = [
            {"role": "user", "content": "find papers on GLP-1"},
        ]
        query = ToolRouterPlugin._extract_query(messages)
        assert query == "find papers on GLP-1"

    def test_extract_query_picks_last_user_message(self) -> None:
        messages = [
            {"role": "user", "content": [{"text": "old query"}]},
            {"role": "assistant", "content": [{"text": "response"}]},
            {"role": "user", "content": [{"text": "new query"}]},
        ]
        query = ToolRouterPlugin._extract_query(messages)
        assert query == "new query"

    def test_extract_query_empty_messages(self) -> None:
        assert ToolRouterPlugin._extract_query(None) == ""
        assert ToolRouterPlugin._extract_query([]) == ""

    def test_get_recommended_tools_empty_before_classification(self) -> None:
        assert self.plugin.get_recommended_tools() == set()

    def test_get_recommended_tools_after_classification(self) -> None:
        """Simulate what route_tools does: classify and store the match."""
        from plugins.domains import classify_query

        self.plugin.last_match = classify_query("find PubMed papers on insulin")
        tools = self.plugin.get_recommended_tools()
        assert len(tools) > 0
        assert "search_pubmed" in tools or "openalex_search" in tools

    def test_route_tools_injects_guidance_message(self) -> None:
        """Test that route_tools prepends a guidance message."""
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1 pharmacokinetics"}]},
        ]

        self.plugin.route_tools(event)

        # Should have prepended a routing message
        assert len(event.messages) == 2
        first_msg = event.messages[0]
        assert first_msg["role"] == "user"
        text = first_msg["content"][0]["text"]
        assert "TOOL ROUTING" in text
        assert "ACADEMIC" in text.upper() or "openalex" in text.lower()

    def test_route_tools_sets_last_match(self) -> None:
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "MesoRx forum thread about tren"}]},
        ]

        self.plugin.route_tools(event)

        assert self.plugin.last_match is not None
        assert FORUM in self.plugin.last_match.domains or PRACTITIONER in self.plugin.last_match.domains

    def test_route_tools_no_messages_is_noop(self) -> None:
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = None

        # Should not raise
        self.plugin.route_tools(event)
        assert self.plugin.last_match is None

    def test_route_tools_general_query_still_injects(self) -> None:
        """Even general queries get routing guidance."""
        from unittest.mock import MagicMock

        event = MagicMock()
        event.messages = [
            {"role": "user", "content": [{"text": "what is the weather in London"}]},
        ]

        self.plugin.route_tools(event)
        assert self.plugin.last_match is not None
        # General domain still has guidance
        assert len(event.messages) == 2
