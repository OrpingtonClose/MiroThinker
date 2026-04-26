# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Agent metrics eval assertions.

Tests that ``AgentResult.metrics`` captures accurate execution data
including cycle counts, tool success/error rates, and timing.
"""

from __future__ import annotations

from strands import Agent, tool

from evals.eval_collector import EvalCollectorPlugin
from evals.mock_model import (
    MockModel,
    multi_tool_then_answer,
    reasoning_then_text,
    simple_text_response,
    tool_call_response,
)


@tool
def add_numbers(a: int, b: int) -> str:
    """Add two numbers together."""
    return str(a + b)


@tool
def failing_tool(query: str) -> str:
    """A tool that always fails."""
    msg = "intentional failure for testing"
    raise RuntimeError(msg)


class TestAgentMetricsBasic:
    """Verify AgentResult.metrics captures basic execution data."""

    def test_simple_query_metrics(self) -> None:
        model = MockModel([simple_text_response("4")])
        agent = Agent(model=model, callback_handler=None)
        result = agent("What is 2+2?")

        metrics = result.metrics
        assert metrics is not None
        summary = metrics.get_summary()
        assert "agent_invocations" in summary
        assert len(summary["agent_invocations"]) >= 1

    def test_tool_use_metrics(self) -> None:
        model = MockModel([
            tool_call_response("add_numbers", "tool-001", {"a": 2, "b": 3}),
            simple_text_response("The result is 5"),
        ])
        agent = Agent(model=model, tools=[add_numbers], callback_handler=None)
        result = agent("Add 2 and 3")

        metrics = result.metrics
        summary = metrics.get_summary()
        invocations = summary.get("agent_invocations", [])
        assert len(invocations) >= 1

        # At least 2 cycles: one for tool call, one for final answer
        total_cycles = sum(len(inv.get("cycles", [])) for inv in invocations)
        assert total_cycles >= 2

    def test_reasoning_plus_answer_metrics(self) -> None:
        model = MockModel([reasoning_then_text(
            reasoning="Let me think step by step...",
            answer="The answer is 42",
        )])
        agent = Agent(model=model, callback_handler=None)
        result = agent("Deep question")

        metrics = result.metrics
        summary = metrics.get_summary()
        assert len(summary.get("agent_invocations", [])) >= 1


class TestAgentMetricsMultiTool:
    """Verify metrics accuracy with multiple tool calls."""

    def test_multi_tool_cycle_count(self) -> None:
        messages = multi_tool_then_answer(
            tools=[
                ("add_numbers", "tool-001", {"a": 1, "b": 2}),
                ("add_numbers", "tool-002", {"a": 3, "b": 4}),
            ],
            answer="1+2=3, 3+4=7",
        )
        model = MockModel(messages)
        agent = Agent(model=model, tools=[add_numbers], callback_handler=None)
        result = agent("Add 1+2 and 3+4")

        summary = result.metrics.get_summary()
        invocations = summary.get("agent_invocations", [])
        total_cycles = sum(len(inv.get("cycles", [])) for inv in invocations)
        # 2 tool calls + 1 final answer = at least 3 cycles
        assert total_cycles >= 3

    def test_tool_error_tracked_in_metrics(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("failing_tool", "tool-001", {"query": "test"}),
            simple_text_response("The tool failed, but here is my answer"),
        ])
        agent = Agent(
            model=model,
            tools=[failing_tool],
            plugins=[collector],
            callback_handler=None,
        )
        result = agent("Try failing tool")

        # The collector should capture the tool error
        assert collector.total_tool_calls == 1
        assert collector.tool_calls[0].success is False
        assert "intentional failure" in (collector.tool_calls[0].error or "")


class TestAgentMetricsTiming:
    """Verify timing data in metrics is reasonable."""

    def test_invocation_duration_positive(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([simple_text_response("Quick answer")])
        agent = Agent(model=model, plugins=[collector], callback_handler=None)
        agent("Fast query")

        assert collector.invocations[0].duration > 0

    def test_model_call_duration_positive(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([simple_text_response("Quick")])
        agent = Agent(model=model, plugins=[collector], callback_handler=None)
        agent("Test")

        assert collector.model_calls[0].duration > 0

    def test_tool_call_duration_positive(self) -> None:
        collector = EvalCollectorPlugin()
        model = MockModel([
            tool_call_response("add_numbers", "tool-001", {"a": 1, "b": 1}),
            simple_text_response("2"),
        ])
        agent = Agent(
            model=model,
            tools=[add_numbers],
            plugins=[collector],
            callback_handler=None,
        )
        agent("Add 1+1")

        assert collector.tool_calls[0].duration > 0
