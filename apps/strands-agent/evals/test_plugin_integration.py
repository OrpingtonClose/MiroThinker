# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Component integration evals — test budget, StreamCapture, and skills.

Verifies that budget_callback, StreamCapture, and the AgentSkills plugin
correctly interact with the Strands SDK event loop when attached to
an Agent instance via callback handlers.
"""

from __future__ import annotations

import queue
import threading
import time

from strands import Agent, tool

from evals.eval_collector import EvalCollectorPlugin
from evals.mock_model import (
    MockModel,
    simple_text_response,
    tool_call_response,
)


@tool
def search_tool(query: str) -> str:
    """Mock search tool that returns a fake result."""
    return f"Found 3 results for: {query}"


# ── budget_callback ──────────────────────────────────────────────────


class TestBudgetCallback:
    """Verify budget_callback counts tool calls and enforces limits."""

    def test_reset_clears_counters(self) -> None:
        from agent import _tool_call_count, reset_budget

        reset_budget()
        from agent import _tool_call_count as count_after

        assert count_after == 0

    def test_budget_counts_tool_call(self) -> None:
        import agent as a

        a.reset_budget()
        # Simulate a tool call event
        a.budget_callback(current_tool_use={"name": "search_tool", "toolUseId": "t-001"})
        assert a._tool_call_count == 1

    def test_budget_deduplicates_by_tool_use_id(self) -> None:
        import agent as a

        a.reset_budget()
        a.budget_callback(current_tool_use={"name": "search_tool", "toolUseId": "t-dup"})
        a.budget_callback(current_tool_use={"name": "search_tool", "toolUseId": "t-dup"})
        assert a._tool_call_count == 1

    def test_budget_ignores_events_without_tool(self) -> None:
        import agent as a

        a.reset_budget()
        a.budget_callback(data="some text")
        a.budget_callback(current_tool_use={})
        a.budget_callback(current_tool_use={"name": ""})
        assert a._tool_call_count == 0

    def test_budget_tracks_multiple_unique_tools(self) -> None:
        import agent as a

        a.reset_budget()
        a.budget_callback(current_tool_use={"name": "tool_a", "toolUseId": "t-001"})
        a.budget_callback(current_tool_use={"name": "tool_b", "toolUseId": "t-002"})
        a.budget_callback(current_tool_use={"name": "tool_c", "toolUseId": "t-003"})
        assert a._tool_call_count == 3


# ── StreamCapture ────────────────────────────────────────────────────


class TestStreamCapture:
    """Verify StreamCapture collects streaming tokens and tool events."""

    def test_captures_text_tokens(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        q = sc.activate()
        sc(data="Hello ")
        sc(data="world")
        sc.deactivate()

        tokens = []
        while True:
            item = q.get_nowait()
            if item is None:
                break
            tokens.append(item)

        assert len(tokens) == 2
        assert tokens[0] == ("text", "Hello ")
        assert tokens[1] == ("text", "world")

    def test_captures_reasoning_tokens(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        q = sc.activate()
        sc(reasoningText="Thinking step 1...")
        sc(reasoningText="Thinking step 2...")
        sc.deactivate()

        tokens = []
        while True:
            item = q.get_nowait()
            if item is None:
                break
            tokens.append(item)

        assert len(tokens) == 2
        assert tokens[0][0] == "thinking"
        assert tokens[1][0] == "thinking"

    def test_captures_tool_events(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        sc.activate()
        sc(current_tool_use={"name": "brave_search", "toolUseId": "t-001", "input": {"q": "test"}})
        sc.deactivate()

        assert len(sc.tool_events) == 1
        assert sc.tool_events[0]["tool"] == "brave_search"

    def test_deduplicates_tool_events(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        sc.activate()
        sc(current_tool_use={"name": "brave_search", "toolUseId": "t-dup"})
        sc(current_tool_use={"name": "brave_search", "toolUseId": "t-dup"})
        sc.deactivate()

        assert len(sc.tool_events) == 1

    def test_drops_tokens_when_not_active(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        # Don't activate — tokens should be silently dropped
        sc(data="dropped")
        sc(reasoningText="also dropped")
        assert len(sc.all_text) == 0

    def test_activate_clears_previous_state(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        q1 = sc.activate()
        sc(data="first session")
        sc.deactivate()

        q2 = sc.activate()
        assert len(sc.all_text) == 0
        assert len(sc.tool_events) == 0

    def test_response_text_tracks_data_only(self) -> None:
        from agent import StreamCapture

        sc = StreamCapture()
        sc.activate()
        sc(reasoningText="thinking")
        sc(data="answer")
        sc.deactivate()

        assert sc.response_text == ["answer"]
        assert sc.reasoning_text == ["thinking"]
        assert len(sc.all_text) == 2


# ── AgentSkills plugin ───────────────────────────────────────────────


class TestSkillsPlugin:
    """Verify AgentSkills plugin loads from the skills/ directory."""

    def test_skills_plugin_builder_returns_none_without_dir(self, tmp_path) -> None:
        """When skills dir doesn't exist, builder returns None."""
        import agent as a

        original = a._SKILLS_DIR
        try:
            a._SKILLS_DIR = tmp_path / "nonexistent"
            result = a._build_skills_plugin()
            assert result is None
        finally:
            a._SKILLS_DIR = original

    def test_skills_plugin_builder_returns_none_for_empty_dir(self, tmp_path) -> None:
        """When skills dir exists but has no valid skills, builder returns None."""
        import agent as a

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        original = a._SKILLS_DIR
        try:
            a._SKILLS_DIR = skills_dir
            result = a._build_skills_plugin()
            assert result is None
        finally:
            a._SKILLS_DIR = original


# ── Collector + Agent integration ────────────────────────────────────


class TestCollectorWithAgent:
    """Verify EvalCollectorPlugin works alongside callback handlers."""

    def test_collector_captures_with_mock_agent(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("search_tool", "t-001", {"query": "test"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[search_tool],
            plugins=[collector],
            callback_handler=None,
        )
        agent("Search for test")

        assert collector.total_tool_calls == 1
        assert collector.tool_names == ["search_tool"]

    def test_multiple_components_dont_interfere(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("search_tool", "t-001", {"query": "test"}),
            simple_text_response("Done"),
        ])
        agent = Agent(
            model=model,
            tools=[search_tool],
            plugins=[collector],
            callback_handler=None,
        )
        agent("Search")

        assert collector.total_tool_calls == 1
        assert collector.total_invocations == 1
