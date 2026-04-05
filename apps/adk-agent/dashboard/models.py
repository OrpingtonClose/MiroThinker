# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Event data models for the ADK execution dashboard.

Defines structured event types covering all 8 MiroThinker algorithms,
agent lifecycle, LLM calls, and tool calls.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """All dashboard event types."""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    SUB_AGENT_START = "sub_agent_start"
    SUB_AGENT_END = "sub_agent_end"

    # LLM
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_END = "llm_call_end"

    # Tool
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"

    # Algorithm-specific events
    DEDUP_BLOCKED = "dedup_blocked"          # Algorithm 2
    DEDUP_ALLOWED = "dedup_allowed"          # Algorithm 2 (escape hatch)
    ARG_FIX_APPLIED = "arg_fix_applied"      # Algorithm 8
    BAD_RESULT_DETECTED = "bad_result"        # Algorithm 4
    CONTEXT_TRIMMED = "context_trimmed"       # Algorithm 5
    FORCE_END_TRIGGERED = "force_end"         # Algorithm 5 (context overflow)
    BOXED_EXTRACTED = "boxed_extracted"       # Algorithm 7
    RETRY_ATTEMPT = "retry_attempt"           # Algorithm 6
    FAILURE_SUMMARY = "failure_summary"       # Algorithm 6

    # Turn-level
    TURN_START = "turn_start"
    TURN_END = "turn_end"

    # Session-level
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class DashboardEvent:
    """A single dashboard event emitted by ADK callbacks or the main loop."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    agent_name: str = ""
    turn: int = 0
    attempt: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON / SSE transmission."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation with timing."""

    tool_name: str = ""
    call_id: str = ""
    agent_name: str = ""
    turn: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0
    arguments_summary: str = ""
    result_size_chars: int = 0
    error: str = ""
    was_dedup_blocked: bool = False
    was_bad_result: bool = False
    arg_fix_applied: bool = False

    def finish(self, result: str = "") -> None:
        self.end_time = time.time()
        self.duration_secs = round(self.end_time - self.start_time, 4)
        self.result_size_chars = len(result) if result else 0


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""

    call_id: str = ""
    agent_name: str = ""
    turn: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0
    prompt_tokens_est: int = 0
    completion_tokens_est: int = 0
    total_tokens_est: int = 0
    error: str = ""

    def finish(self, response_text: str = "") -> None:
        self.end_time = time.time()
        self.duration_secs = round(self.end_time - self.start_time, 4)
        if response_text:
            self.completion_tokens_est = max(1, len(response_text) // 4)
        self.total_tokens_est = self.prompt_tokens_est + self.completion_tokens_est


@dataclass
class RetryAttemptRecord:
    """Record of a context compression retry attempt."""

    attempt_number: int = 0
    max_attempts: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    duration_secs: float = 0.0
    failure_summary: str = ""
    answer_found: bool = False
    answer_source: str = ""  # "research", "summarization", "intermediate"

    def finish(self) -> None:
        self.end_time = time.time()
        self.duration_secs = round(self.end_time - self.start_time, 4)
