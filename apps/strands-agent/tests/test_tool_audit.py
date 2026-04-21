# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Tests for the ToolAuditPlugin."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plugins.tool_audit import ToolAuditPlugin
from plugins.tool_router import ToolRouterPlugin


class TestToolAuditPlugin:
    """Test ToolAuditPlugin tool usage tracking and resume logic."""

    def setup_method(self) -> None:
        self.router = ToolRouterPlugin()
        self.audit = ToolAuditPlugin(router=self.router)

    def test_name(self) -> None:
        assert self.audit.name == "tool-audit"

    def test_reset_clears_state(self) -> None:
        self.audit._tools_called.add("duckduckgo_search")
        self.audit._resume_count = 1
        self.audit._current_query = "test"

        self.audit.reset()

        assert len(self.audit._tools_called) == 0
        assert self.audit._resume_count == 0
        assert self.audit._current_query == ""

    def test_track_tool_call_records_tool_name(self) -> None:
        event = MagicMock()
        event.tool_use = {"name": "duckduckgo_search"}

        self.audit.track_tool_call(event)

        assert "duckduckgo_search" in self.audit._tools_called

    def test_track_tool_call_ignores_empty_name(self) -> None:
        event = MagicMock()
        event.tool_use = {"name": ""}

        self.audit.track_tool_call(event)

        assert len(self.audit._tools_called) == 0

    def test_audit_no_resume_when_recommended_tools_used(self) -> None:
        """No resume when the agent used at least one recommended tool."""
        from plugins.domains import classify_query

        # Simulate: router classified as academic
        self.router.last_match = classify_query("find papers on GLP-1 pharmacokinetics")

        # Agent used an academic tool
        self.audit._tools_called = {"openalex_search", "duckduckgo_search"}

        event = MagicMock()
        event.resume = None

        self.audit.audit_tool_usage(event)

        # Should NOT set resume
        assert event.resume is None

    def test_audit_triggers_resume_when_no_recommended_tools_used(self) -> None:
        """Resume triggered when zero recommended tools were used."""
        from plugins.domains import classify_query

        # Router classified as academic
        self.router.last_match = classify_query("find papers on GLP-1 pharmacokinetics")

        # Agent only used generic search — no academic tools
        self.audit._tools_called = {"duckduckgo_search", "jina_read_url"}
        self.audit._current_query = "find papers on GLP-1 pharmacokinetics"

        event = MagicMock()
        event.resume = None

        self.audit.audit_tool_usage(event)

        # Should set resume with nudge message
        assert event.resume is not None
        assert "specialized tools" in event.resume.lower() or "recommended" in event.resume.lower()

    def test_audit_respects_max_resumes(self) -> None:
        """No resume after max_resumes reached."""
        from plugins.domains import classify_query

        self.router.last_match = classify_query("find papers on GLP-1")
        self.audit._tools_called = {"duckduckgo_search"}
        self.audit._resume_count = 1  # Already resumed once (max_resumes=1)

        event = MagicMock()
        event.resume = None

        self.audit.audit_tool_usage(event)

        assert event.resume is None

    def test_audit_no_resume_when_no_tools_called(self) -> None:
        """No resume when no tools were called at all (simple chat)."""
        from plugins.domains import classify_query

        self.router.last_match = classify_query("find papers on GLP-1")
        self.audit._tools_called = set()  # No tools called at all

        event = MagicMock()
        event.resume = None

        self.audit.audit_tool_usage(event)

        assert event.resume is None

    def test_audit_without_router_uses_independent_classification(self) -> None:
        """When no router is provided, audit classifies independently."""
        audit_standalone = ToolAuditPlugin(router=None)
        audit_standalone._current_query = "find papers on GLP-1 pharmacokinetics"
        audit_standalone._tools_called = {"duckduckgo_search"}

        event = MagicMock()
        event.resume = None

        audit_standalone.audit_tool_usage(event)

        # Should still trigger resume via independent classification
        assert event.resume is not None

    def test_build_nudge_message_groups_by_category(self) -> None:
        missed = {
            "openalex_search",
            "openalex_citation_network",
            "search_pubmed",
            "pubmed_get_abstract",
            "check_retraction",
        }
        nudge = self.audit._build_nudge_message(missed)

        assert "Academic databases" in nudge or "PubMed" in nudge
        assert "Research integrity" in nudge or "retraction" in nudge.lower()

    def test_build_nudge_message_empty_set(self) -> None:
        assert self.audit._build_nudge_message(set()) == ""

    def test_track_invocation_start_resets_on_first_call(self) -> None:
        self.audit._tools_called = {"old_tool"}

        event = MagicMock()
        event.messages = [{"role": "user", "content": [{"text": "new query"}]}]

        self.audit.track_invocation_start(event)

        assert len(self.audit._tools_called) == 0
        assert self.audit._current_query == "new query"

    def test_track_invocation_start_skips_reset_on_same_query_resume(self) -> None:
        """Don't clear tools_called on resume with the same query."""
        self.audit._resume_count = 1
        self.audit._tools_called = {"kept_tool"}
        self.audit._current_query = "same query"

        event = MagicMock()
        event.messages = [{"role": "user", "content": [{"text": "same query"}]}]

        self.audit.track_invocation_start(event)

        assert "kept_tool" in self.audit._tools_called
        assert self.audit._resume_count == 1  # Not reset

    def test_track_invocation_start_resets_on_new_query_after_resume(self) -> None:
        """Reset state when a new query arrives on a reused agent after a resume."""
        self.audit._resume_count = 1
        self.audit._tools_called = {"stale_tool"}
        self.audit._current_query = "old query"

        event = MagicMock()
        event.messages = [{"role": "user", "content": [{"text": "completely new query"}]}]

        self.audit.track_invocation_start(event)

        assert len(self.audit._tools_called) == 0
        assert self.audit._resume_count == 0
        assert self.audit._current_query == "completely new query"

    def test_audit_works_after_new_query_resets_resume_count(self) -> None:
        """Verify audit is re-enabled after a new query resets the resume count."""
        from plugins.domains import classify_query

        # Simulate: first request triggered a resume
        self.audit._resume_count = 1
        self.audit._current_query = "old query"
        self.audit._tools_called = {"old_tool"}

        # New request arrives on the reused agent
        start_event = MagicMock()
        start_event.messages = [
            {"role": "user", "content": [{"text": "find papers on GLP-1"}]},
        ]
        self.audit.track_invocation_start(start_event)

        # Resume count should be reset
        assert self.audit._resume_count == 0

        # Now simulate the agent only using generic search
        self.router.last_match = classify_query("find papers on GLP-1")
        self.audit._tools_called = {"duckduckgo_search"}

        audit_event = MagicMock()
        audit_event.resume = None
        self.audit.audit_tool_usage(audit_event)

        # Audit should trigger resume (not be permanently disabled)
        assert audit_event.resume is not None
