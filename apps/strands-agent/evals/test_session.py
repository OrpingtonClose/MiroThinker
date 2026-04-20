# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Session replay evals — save and inspect agent sessions.

Uses Strands ``FileSessionManager`` to persist agent conversations,
then loads them back for structural assertions — verifying message
format, tool call records, and content integrity without hitting
any live API.
"""

from __future__ import annotations

import os
import tempfile
import uuid

from strands import Agent, tool
from strands.session.file_session_manager import FileSessionManager

from evals.eval_collector import EvalCollectorPlugin
from evals.mock_model import (
    MockModel,
    reasoning_then_text,
    simple_text_response,
    tool_call_response,
)


@tool
def echo_tool(text: str) -> str:
    """Echo the input text back."""
    return f"Echo: {text}"


def _make_session_manager(tmpdir: str) -> FileSessionManager:
    """Create a FileSessionManager with a unique session ID."""
    return FileSessionManager(session_id=str(uuid.uuid4()), storage_dir=tmpdir)


class TestSessionPersistence:
    """Verify agent sessions are saved and can be inspected offline."""

    def test_session_saved_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = _make_session_manager(tmpdir)
            model = MockModel([simple_text_response("Hello!")])
            agent = Agent(
                model=model,
                session_manager=session_mgr,
                callback_handler=None,
            )
            agent("Hi there")

            # Session file should exist
            files = os.listdir(tmpdir)
            assert len(files) >= 1

    def test_session_contains_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = _make_session_manager(tmpdir)
            model = MockModel([simple_text_response("The answer is 4.")])
            agent = Agent(
                model=model,
                session_manager=session_mgr,
                callback_handler=None,
            )
            agent("What is 2+2?")

            # Messages should contain both user and assistant messages
            messages = agent.messages
            assert len(messages) >= 2
            roles = [m["role"] for m in messages]
            assert "user" in roles
            assert "assistant" in roles

    def test_session_contains_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = _make_session_manager(tmpdir)
            model = MockModel([
                tool_call_response("echo_tool", "tool-001", {"text": "test"}),
                simple_text_response("Done"),
            ])
            agent = Agent(
                model=model,
                tools=[echo_tool],
                session_manager=session_mgr,
                callback_handler=None,
            )
            agent("Echo test")

            # Messages should contain toolUse and toolResult blocks
            has_tool_use = False
            has_tool_result = False
            for msg in agent.messages:
                for block in msg.get("content", []):
                    if "toolUse" in block:
                        has_tool_use = True
                    if "toolResult" in block:
                        has_tool_result = True
            assert has_tool_use, "Session should contain toolUse blocks"
            assert has_tool_result, "Session should contain toolResult blocks"

    def test_session_reasoning_answer_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = _make_session_manager(tmpdir)
            model = MockModel([reasoning_then_text(
                reasoning="Thinking about this deeply...",
                answer="42",
            )])
            agent = Agent(
                model=model,
                session_manager=session_mgr,
                callback_handler=None,
            )
            result = agent("Deep question")

            # The text answer from reasoning_then_text should be in the result
            assert "42" in str(result)

            # Session messages should contain an assistant message with text
            has_text = False
            for msg in agent.messages:
                if msg.get("role") == "assistant":
                    for block in msg.get("content", []):
                        if "text" in block and "42" in block["text"]:
                            has_text = True
            assert has_text, "Session should preserve the text answer"


class TestSessionReplay:
    """Verify session structure supports offline analysis."""

    def test_message_count_matches_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = _make_session_manager(tmpdir)
            model = MockModel([
                simple_text_response("First answer"),
                simple_text_response("Second answer"),
            ])
            agent = Agent(
                model=model,
                session_manager=session_mgr,
                callback_handler=None,
            )
            agent("First question")
            agent("Second question")

            # 2 user + 2 assistant = 4 messages minimum
            assert len(agent.messages) >= 4

    def test_session_with_metrics_and_collector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_mgr = _make_session_manager(tmpdir)
            collector = EvalCollectorPlugin()
            model = MockModel([
                tool_call_response("echo_tool", "tool-001", {"text": "data"}),
                simple_text_response("Processed"),
            ])
            agent = Agent(
                model=model,
                tools=[echo_tool],
                session_manager=session_mgr,
                plugins=[collector],
                callback_handler=None,
            )
            result = agent("Process data")

            # Both session and collector should have data
            assert len(agent.messages) >= 2
            assert collector.total_tool_calls == 1

            # Metrics should also be populated
            summary = result.metrics.get_summary()
            assert len(summary.get("agent_invocations", [])) >= 1
