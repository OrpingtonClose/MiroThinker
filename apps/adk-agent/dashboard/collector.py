# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
ADK-aware metrics collector for the execution dashboard.

Adapted from deep-search-portal ``proxies/research_metrics.py``
``MetricsCollector`` class, extended with algorithm-specific event lists
that cover all 8 MiroThinker algorithms.

Usage::

    collector = DashboardCollector(session_id="abc", query="What is …")
    await collector.emit(DashboardEvent(event_type=EventType.TOOL_CALL_START, …))
    …
    metrics = collector.to_metrics_dict()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dashboard.models import (
    DashboardEvent,
    EventType,
    LLMCallRecord,
    RetryAttemptRecord,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)


class DashboardCollector:
    """Accumulates metrics during an ADK agent session.

    Provides:
    * An ``asyncio.Queue`` for real-time SSE streaming.
    * Typed lists for post-hoc analysis of each algorithm.
    * A ``to_metrics_dict()`` method for JSON serialisation.
    """

    def __init__(self, session_id: str, query: str) -> None:
        self.session_id = session_id
        self.query = query
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start_time = time.time()

        # SSE event queue — thread-safe so the uvicorn server thread
        # can safely read while the main event loop writes.
        self.event_queue: queue.Queue[Optional[DashboardEvent]] = queue.Queue()

        # Full event history
        self.events: list[DashboardEvent] = []

        # Running counters
        self.current_turn = 0
        self.current_attempt = 1
        self.total_tool_calls = 0
        self.total_llm_calls = 0
        self.tool_errors = 0
        self.llm_errors = 0

        # Algorithm-specific event lists
        self.dedup_blocks: list[dict[str, Any]] = []
        self.dedup_escapes: list[dict[str, Any]] = []
        self.arg_fixes: list[dict[str, Any]] = []
        self.bad_results: list[dict[str, Any]] = []
        self.context_trims: list[dict[str, Any]] = []
        self.force_ends: list[dict[str, Any]] = []
        self.boxed_answers: list[dict[str, Any]] = []
        self.retry_attempts: list[RetryAttemptRecord] = []

        # Detailed records
        self.tool_calls: list[ToolCallRecord] = []
        self.llm_calls: list[LLMCallRecord] = []

        # Active (in-flight) calls
        self._active_tool_calls: dict[str, ToolCallRecord] = {}
        self._active_llm_calls: dict[str, LLMCallRecord] = {}

        # Agent hierarchy tracking
        self.active_agents: list[str] = []

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    async def emit(self, event: DashboardEvent) -> None:
        """Push event to SSE queue and record in history (async)."""
        self._push(event)

    def emit_sync(self, event: DashboardEvent) -> None:
        """Push event synchronously — safe to call from non-async callbacks."""
        self._push(event)

    def _push(self, event: DashboardEvent) -> None:
        """Internal: append event and push to the thread-safe queue."""
        self.events.append(event)
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            logger.warning("Dashboard event queue full, dropping event")
        self._route_event(event)

    def _route_event(self, event: DashboardEvent) -> None:
        """Categorise event into the correct algorithm list."""
        et = event.event_type
        data = event.data

        if et == EventType.DEDUP_BLOCKED:
            self.dedup_blocks.append(data | {"timestamp": event.timestamp, "turn": event.turn})
        elif et == EventType.DEDUP_ALLOWED:
            self.dedup_escapes.append(data | {"timestamp": event.timestamp, "turn": event.turn})
        elif et == EventType.ARG_FIX_APPLIED:
            self.arg_fixes.append(data | {"timestamp": event.timestamp, "turn": event.turn})
        elif et == EventType.BAD_RESULT_DETECTED:
            self.bad_results.append(data | {"timestamp": event.timestamp, "turn": event.turn})
            self.tool_errors += 1
        elif et == EventType.CONTEXT_TRIMMED:
            self.context_trims.append(data | {"timestamp": event.timestamp, "turn": event.turn})
        elif et == EventType.FORCE_END_TRIGGERED:
            self.force_ends.append(data | {"timestamp": event.timestamp, "turn": event.turn})
        elif et == EventType.BOXED_EXTRACTED:
            self.boxed_answers.append(data | {"timestamp": event.timestamp, "turn": event.turn})
        elif et == EventType.TOOL_CALL_START:
            self.total_tool_calls += 1
            # Create a tracking record from the event data
            call_id = data.get("call_id") or f"tool_{self.total_tool_calls}"
            self.start_tool_call(
                call_id=call_id,
                tool_name=data.get("tool_name", ""),
                agent_name=event.agent_name,
                arguments=data.get("arguments_summary", ""),
            )
        elif et == EventType.TOOL_CALL_END:
            # Finalize the tracking record.  Match by call_id first; if
            # missing, fall back to the most recent active call with the
            # same tool_name to avoid mismatches when multiple tools are
            # in-flight concurrently.
            call_id = data.get("call_id")
            record = self._active_tool_calls.pop(call_id, None) if call_id else None
            if record is None:
                # Fallback: find by tool_name (pop the matching entry)
                tool_name = data.get("tool_name", "")
                for cid, rec in list(self._active_tool_calls.items()):
                    if rec.tool_name == tool_name:
                        record = self._active_tool_calls.pop(cid)
                        break
            if record:
                record.duration_secs = data.get("duration_secs", 0.0)
                record.result_size_chars = data.get("result_size_chars", 0)
                record.error = data.get("error", "")
                self.tool_calls.append(record)
        elif et == EventType.LLM_CALL_START:
            self.total_llm_calls += 1
            call_id = data.get("call_id") or f"llm_{self.total_llm_calls}"
            self.start_llm_call(
                call_id=call_id,
                agent_name=event.agent_name,
                prompt_tokens_est=data.get("estimated_prompt_tokens", 0),
            )
        elif et == EventType.LLM_CALL_END:
            call_id = data.get("call_id") or f"llm_{self.total_llm_calls}"
            record = self._active_llm_calls.pop(call_id, None)
            if record:
                record.completion_tokens_est = data.get("completion_tokens_est", 0)
                record.duration_secs = round(time.time() - record.start_time, 4)
                self.llm_calls.append(record)
        elif et == EventType.TURN_START:
            self.current_turn = event.turn
        elif et == EventType.AGENT_START:
            self.active_agents.append(event.agent_name)
        elif et == EventType.AGENT_END:
            if event.agent_name in self.active_agents:
                self.active_agents.remove(event.agent_name)

    # ------------------------------------------------------------------
    # Tool call tracking
    # ------------------------------------------------------------------

    def start_tool_call(
        self, call_id: str, tool_name: str, agent_name: str = "", arguments: str = ""
    ) -> ToolCallRecord:
        record = ToolCallRecord(
            tool_name=tool_name,
            call_id=call_id,
            agent_name=agent_name,
            turn=self.current_turn,
            start_time=time.time(),
            arguments_summary=arguments[:200] if arguments else "",
        )
        self._active_tool_calls[call_id] = record
        return record

    def end_tool_call(
        self, call_id: str, result: str = "", error: str = ""
    ) -> Optional[ToolCallRecord]:
        record = self._active_tool_calls.pop(call_id, None)
        if record:
            record.finish(result)
            record.error = error
            self.tool_calls.append(record)
        return record

    # ------------------------------------------------------------------
    # LLM call tracking
    # ------------------------------------------------------------------

    def start_llm_call(
        self, call_id: str, agent_name: str = "", prompt_tokens_est: int = 0
    ) -> LLMCallRecord:
        record = LLMCallRecord(
            call_id=call_id,
            agent_name=agent_name,
            turn=self.current_turn,
            start_time=time.time(),
            prompt_tokens_est=prompt_tokens_est,
        )
        self._active_llm_calls[call_id] = record
        return record

    def end_llm_call(
        self, call_id: str, response_text: str = "", error: str = ""
    ) -> Optional[LLMCallRecord]:
        record = self._active_llm_calls.pop(call_id, None)
        if record:
            record.finish(response_text)
            record.error = error
            self.llm_calls.append(record)
        return record

    # ------------------------------------------------------------------
    # Retry tracking
    # ------------------------------------------------------------------

    def start_retry(self, attempt: int, max_attempts: int) -> RetryAttemptRecord:
        record = RetryAttemptRecord(
            attempt_number=attempt,
            max_attempts=max_attempts,
            start_time=time.time(),
        )
        self.current_attempt = attempt
        self.retry_attempts.append(record)
        return record

    # ------------------------------------------------------------------
    # End session
    # ------------------------------------------------------------------

    async def end_session(self, final_answer: str = "", attempts_used: int = 0) -> None:
        """Signal that the session is over (sends sentinel to SSE queue)."""
        await self.emit(
            DashboardEvent(
                event_type=EventType.SESSION_END,
                data={"final_answer": final_answer, "attempts_used": attempts_used},
            )
        )
        self.event_queue.put_nowait(None)  # sentinel

    # ------------------------------------------------------------------
    # Metrics summary
    # ------------------------------------------------------------------

    def elapsed_secs(self) -> float:
        return round(time.time() - self._start_time, 2)

    def to_metrics_dict(self) -> dict[str, Any]:
        """Generate final metrics summary for the dashboard."""
        # Tool call summary
        tool_summary: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_duration": 0.0, "errors": 0, "total_result_chars": 0}
        )
        for tc in self.tool_calls:
            s = tool_summary[tc.tool_name]
            s["count"] += 1
            s["total_duration"] += tc.duration_secs
            s["total_result_chars"] += tc.result_size_chars
            if tc.error:
                s["errors"] += 1

        # LLM call summary
        total_prompt_tokens = sum(lc.prompt_tokens_est for lc in self.llm_calls)
        total_completion_tokens = sum(lc.completion_tokens_est for lc in self.llm_calls)
        avg_llm_duration = (
            sum(lc.duration_secs for lc in self.llm_calls) / len(self.llm_calls)
            if self.llm_calls
            else 0.0
        )

        return {
            "session_id": self.session_id,
            "query": self.query,
            "started_at": self.started_at,
            "elapsed_secs": self.elapsed_secs(),
            "kpi": {
                "turns": self.current_turn,
                "tool_calls": self.total_tool_calls,
                "tool_errors": self.tool_errors,
                "llm_calls": self.total_llm_calls,
                "llm_errors": self.llm_errors,
                "prompt_tokens_est": total_prompt_tokens,
                "completion_tokens_est": total_completion_tokens,
                "retry_attempts": len(self.retry_attempts),
                "intermediate_answers": len(self.boxed_answers),
                "elapsed_secs": self.elapsed_secs(),
            },
            "algorithms": {
                "dedup_blocks": self.dedup_blocks,
                "dedup_escapes": self.dedup_escapes,
                "arg_fixes": self.arg_fixes,
                "bad_results": self.bad_results,
                "context_trims": self.context_trims,
                "force_ends": self.force_ends,
                "boxed_answers": self.boxed_answers,
                "retry_attempts": [asdict(r) for r in self.retry_attempts],
            },
            "algorithm_stats": {
                "dedup_blocks_saved": len(self.dedup_blocks),
                "arg_fixes_applied": len(self.arg_fixes),
                "bad_results_caught": len(self.bad_results),
                "context_trims_performed": len(self.context_trims),
                "force_end_triggered": len(self.force_ends),
                "intermediate_answers_extracted": len(self.boxed_answers),
                "retry_attempts_used": len(self.retry_attempts),
            },
            "tool_calls": [asdict(tc) for tc in self.tool_calls],
            "tool_summary": dict(tool_summary),
            "llm_summary": {
                "total_calls": len(self.llm_calls),
                "total_prompt_tokens_est": total_prompt_tokens,
                "total_completion_tokens_est": total_completion_tokens,
                "avg_duration_secs": round(avg_llm_duration, 4),
            },
            "active_agents": self.active_agents,
            "events": [e.to_dict() for e in self.events],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str = "dashboard_logs") -> str:
        """Save metrics JSON to disk (like TaskLog.save())."""
        os.makedirs(output_dir, exist_ok=True)
        ts = self.started_at.replace(":", "-").replace(".", "-")
        filename = f"{output_dir}/dashboard_{self.session_id}_{ts}.json"
        data = self.to_metrics_dict()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Dashboard metrics saved to %s", filename)
        return filename

    @staticmethod
    def load(filepath: str) -> dict[str, Any]:
        """Load a saved metrics JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def list_saved_reports(output_dir: str = "dashboard_logs") -> list[str]:
        """List available saved report files."""
        p = Path(output_dir)
        if not p.exists():
            return []
        return sorted(str(f) for f in p.glob("dashboard_*.json"))
