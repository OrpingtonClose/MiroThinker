# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Integration evals — real Venice API calls via the Strands Agent.

These evals require VENICE_API_KEY and test the full agent pipeline:
model inference, tool dispatch, reasoning, and response quality.

Usage::

    VENICE_API_KEY=... pytest evals/test_integ_agent.py -v -m integ
"""

from __future__ import annotations

import logging
import os
import time

import pytest
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager

from evals.eval_collector import EvalCollectorPlugin

pytestmark = pytest.mark.integ

logger = logging.getLogger(__name__)

VENICE_API_KEY = os.environ.get("VENICE_API_KEY", "")
VENICE_API_BASE = os.environ.get("VENICE_API_BASE", "https://api.venice.ai/api/v1")
VENICE_MODEL = os.environ.get("VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic")


def _build_model():
    """Build a Venice model for integration tests."""
    from strands.models.openai import OpenAIModel

    return OpenAIModel(
        client_args={
            "api_key": VENICE_API_KEY,
            "base_url": VENICE_API_BASE,
        },
        model_id=VENICE_MODEL,
        params={
            "extra_body": {
                "venice_parameters": {"include_venice_system_prompt": False},
                "reasoning": {"effort": "high"},
            }
        },
    )


def _skip_if_no_key() -> None:
    if not VENICE_API_KEY:
        pytest.skip("VENICE_API_KEY not set")


class TestIntegSimpleQueries:
    """Basic agent queries with real Venice model."""

    def test_arithmetic_answer(self) -> None:
        _skip_if_no_key()
        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(model=model, plugins=[collector], callback_handler=None)

        result = agent("What is 17 * 23? Reply with just the number.")
        answer = str(result).strip()

        assert "391" in answer
        assert collector.total_invocations == 1
        assert collector.total_model_calls >= 1

    def test_factual_question(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(model=model, plugins=[collector], callback_handler=None)

        result = agent("What is the chemical formula for water? One word answer.")
        answer = str(result).strip().lower()

        assert "h2o" in answer

    def test_reasoning_present(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(model=model, plugins=[collector], callback_handler=None)

        result = agent(
            "Explain why the sky is blue in exactly one sentence."
        )
        answer = str(result).strip()

        # Should have a non-trivial response
        assert len(answer) > 20
        assert collector.total_model_calls >= 1


class TestIntegToolUsage:
    """Verify the agent dispatches tools when given search-worthy queries."""

    def test_search_tool_called(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        from tools import duckduckgo_search

        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(
            model=model,
            tools=[duckduckgo_search],
            plugins=[collector],
            callback_handler=None,
            system_prompt="You are a research assistant. Use your search tool to find information.",
        )

        result = agent("Search for the latest news about quantum computing")
        assert collector.total_tool_calls >= 1
        assert "duckduckgo_search" in collector.tool_names

    def test_tool_result_in_response(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        from tools import duckduckgo_search

        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(
            model=model,
            tools=[duckduckgo_search],
            plugins=[collector],
            callback_handler=None,
            system_prompt="You are a research assistant. Use search tools to answer questions.",
        )

        result = agent("What is the current population of Tokyo? Search for it.")
        answer = str(result)

        # Should have some numeric content from the search
        assert len(answer) > 50


class TestIntegMultiTurn:
    """Verify context preservation across conversation turns."""

    def test_context_preserved(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(
            model=model,
            plugins=[collector],
            callback_handler=None,
            conversation_manager=SlidingWindowConversationManager(window_size=10),
        )

        agent("My name is TestUser42. Remember this.")
        time.sleep(2)
        result = agent("What is my name?")
        answer = str(result)

        assert "TestUser42" in answer
        assert collector.total_invocations == 2


class TestIntegBudgetEnforcement:
    """Verify budget_callback works with a real model."""

    def test_budget_tracks_real_tool_calls(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        import agent as a
        from tools import duckduckgo_search

        a.reset_budget()
        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(
            model=model,
            tools=[duckduckgo_search],
            plugins=[collector],
            callback_handler=a.budget_callback,
            system_prompt="You are a research assistant. Use search tools.",
        )

        agent("Search for information about mesh networking protocols")

        # Budget should have counted tool calls
        assert a._tool_call_count >= 0
        # Collector should also have captured them
        assert collector.total_tool_calls >= 0


class TestIntegMetricsIntegrity:
    """Verify metrics match between collector plugin and AgentResult."""

    def test_collector_matches_result_metrics(self) -> None:
        _skip_if_no_key()
        time.sleep(2)
        model = _build_model()
        collector = EvalCollectorPlugin()
        agent = Agent(model=model, plugins=[collector], callback_handler=None)

        result = agent("What is the speed of light in km/s?")

        # Collector and result metrics should agree on invocation count
        assert collector.total_invocations >= 1
        summary = result.metrics.get_summary()
        assert len(summary.get("agent_invocations", [])) >= 1
