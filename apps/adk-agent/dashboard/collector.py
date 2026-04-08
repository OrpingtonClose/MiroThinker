# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""PipelineCollector — thread-safe event accumulator for pipeline runs.

One collector is created per ``run_pipeline()`` invocation.  All ADK
callbacks write to it via ``dashboard.get_active_collector()``.  The
collector accumulates structured events, KPIs, and algorithm activity
and can produce:

* **snapshot()** — lightweight dict for SSE streaming (called every 500ms)
* **finalize()** — full dict for JSON dump + HTML report generation
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_DASHBOARD_LOGS_DIR = os.environ.get(
    "DASHBOARD_LOGS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard_logs"),
)


@dataclass
class _ToolCallRecord:
    tool_name: str
    agent: str
    args_summary: str
    start_time: float
    end_time: float = 0.0
    duration_secs: float = 0.0
    result_chars: int = 0
    error: str = ""
    was_dedup_blocked: bool = False
    arg_fix_applied: bool = False
    was_compressed: bool = False
    original_chars: int = 0


@dataclass
class _LLMCallRecord:
    agent: str
    start_time: float
    prompt_tokens_est: int = 0
    end_time: float = 0.0
    duration_secs: float = 0.0
    completion_tokens_est: int = 0


class PipelineCollector:
    """Thread-safe event collector for a single pipeline run."""

    def __init__(self, query: str, session_id: str) -> None:
        self._lock = threading.Lock()
        self.query = query
        self.session_id = session_id
        self.started_at = time.time()

        # Phase tracking
        self.phases: list[dict[str, Any]] = []
        self._current_phase: str = ""

        # Raw event stream
        self.events: list[dict[str, Any]] = []

        # Structured records
        self.tool_calls: list[_ToolCallRecord] = []
        self.llm_calls: list[_LLMCallRecord] = []
        self._pending_tool: dict[str, _ToolCallRecord] = {}  # tool_name → record
        self._pending_llm: dict[str, _LLMCallRecord] = {}    # agent → record

        # KPI counters
        self.total_adk_events = 0
        self.total_text_chars = 0
        self.total_reasoning_chars = 0

        # Algorithm counters
        self.dedup_blocks: list[dict] = []
        self.arg_fixes: list[dict] = []
        self.bad_results: list[dict] = []
        self.compressions: list[dict] = []
        self.keep_k_trims: list[dict] = []
        self.force_ends: list[dict] = []

        # Corpus tracking
        self.corpus_updates: list[dict] = []
        self.stall_events: list[dict] = []

        # Thinker escalation
        self.thinker_escalated = False
        self.thinker_escalate_time: float = 0.0

        # Finalization
        self._finalized = False
        self._finalized_at: float = 0.0

    # ── Phase lifecycle ───────────────────────────────────────────────

    def phase_start(self, phase: str, agent: str) -> None:
        with self._lock:
            self._current_phase = phase
            entry = {
                "phase": phase,
                "agent": agent,
                "start_time": time.time(),
                "end_time": 0.0,
                "outcome": "",
            }
            self.phases.append(entry)
            self._emit("phase_start", agent=agent, data={"phase": phase})

    def phase_end(self, phase: str, outcome: str) -> None:
        with self._lock:
            now = time.time()
            for p in reversed(self.phases):
                if p["phase"] == phase and p["end_time"] == 0.0:
                    p["end_time"] = now
                    p["outcome"] = outcome
                    break
            self._current_phase = ""
            self._emit("phase_end", data={"phase": phase, "outcome": outcome})

    # ── ADK event stream ──────────────────────────────────────────────

    def adk_event(
        self,
        agent: str = "",
        event_type: str = "",
        has_text: bool = False,
        text_len: int = 0,
        is_reasoning: bool = False,
        is_tool_call: bool = False,
        tool_name: str = "",
    ) -> None:
        with self._lock:
            self.total_adk_events += 1
            if is_reasoning:
                self.total_reasoning_chars += text_len
            elif has_text:
                self.total_text_chars += text_len
            self._emit(
                "adk_event",
                agent=agent,
                data={
                    "event_type": event_type,
                    "has_text": has_text,
                    "text_len": text_len,
                    "is_reasoning": is_reasoning,
                    "is_tool_call": is_tool_call,
                    "tool_name": tool_name,
                },
            )

    # ── Tool events ───────────────────────────────────────────────────

    def tool_start(self, tool_name: str, agent: str, args_summary: str) -> None:
        with self._lock:
            rec = _ToolCallRecord(
                tool_name=tool_name,
                agent=agent,
                args_summary=args_summary[:300],
                start_time=time.time(),
            )
            self._pending_tool[tool_name] = rec
            self._emit(
                "tool_call_start",
                agent=agent,
                data={
                    "tool_name": tool_name,
                    "args_summary": args_summary[:300],
                },
            )

    def tool_end(
        self,
        tool_name: str,
        agent: str,
        duration: float,
        result_chars: int,
        error: str = "",
    ) -> None:
        with self._lock:
            rec = self._pending_tool.pop(tool_name, None)
            if rec is None:
                rec = _ToolCallRecord(
                    tool_name=tool_name,
                    agent=agent,
                    args_summary="",
                    start_time=time.time() - duration,
                )
            rec.end_time = time.time()
            rec.duration_secs = duration
            rec.result_chars = result_chars
            rec.error = error
            self.tool_calls.append(rec)
            self._emit(
                "tool_call_end",
                agent=agent,
                data={
                    "tool_name": tool_name,
                    "duration_secs": round(duration, 3),
                    "result_chars": result_chars,
                    "error": error,
                },
            )

    def dedup_block(self, tool_name: str, query_key: str, consecutive: int) -> None:
        with self._lock:
            entry = {
                "tool_name": tool_name,
                "query_key": query_key,
                "consecutive": consecutive,
                "timestamp": time.time(),
            }
            self.dedup_blocks.append(entry)
            self._emit("dedup_block", data=entry)

    def arg_fix(self, tool_name: str, fixes: list[str]) -> None:
        with self._lock:
            entry = {
                "tool_name": tool_name,
                "fixes": fixes,
                "timestamp": time.time(),
            }
            self.arg_fixes.append(entry)
            self._emit("arg_fix", data=entry)

    def bad_result(self, tool_name: str, error: str) -> None:
        with self._lock:
            entry = {
                "tool_name": tool_name,
                "error": error[:500],
                "timestamp": time.time(),
            }
            self.bad_results.append(entry)
            self._emit("bad_result", data=entry)

    def compression(
        self, tool_name: str, original_chars: int, compressed_chars: int
    ) -> None:
        with self._lock:
            entry = {
                "tool_name": tool_name,
                "original_chars": original_chars,
                "compressed_chars": compressed_chars,
                "ratio": round(compressed_chars / max(original_chars, 1), 3),
                "timestamp": time.time(),
            }
            self.compressions.append(entry)
            # Tag the pending tool record too
            rec = self._pending_tool.get(tool_name)
            if rec:
                rec.was_compressed = True
                rec.original_chars = original_chars
            self._emit("compression", data=entry)

    # ── LLM events ────────────────────────────────────────────────────

    def llm_start(self, agent: str, prompt_tokens_est: int) -> None:
        with self._lock:
            rec = _LLMCallRecord(
                agent=agent,
                start_time=time.time(),
                prompt_tokens_est=prompt_tokens_est,
            )
            self._pending_llm[agent] = rec
            self._emit(
                "llm_call_start",
                agent=agent,
                data={"prompt_tokens_est": prompt_tokens_est},
            )

    def llm_end(self, agent: str, duration: float, completion_tokens_est: int) -> None:
        with self._lock:
            rec = self._pending_llm.pop(agent, None)
            if rec is None:
                rec = _LLMCallRecord(
                    agent=agent,
                    start_time=time.time() - duration,
                )
            rec.end_time = time.time()
            # Compute actual duration from timestamps when available,
            # rather than using the passed-in value (often 0.0).
            actual_duration = rec.end_time - rec.start_time if rec.start_time else duration
            rec.duration_secs = actual_duration
            rec.completion_tokens_est = completion_tokens_est
            self.llm_calls.append(rec)
            self._emit(
                "llm_call_end",
                agent=agent,
                data={
                    "duration_secs": round(actual_duration, 3),
                    "completion_tokens_est": completion_tokens_est,
                },
            )

    # ── Algorithm events ──────────────────────────────────────────────

    def keep_k_trim(self, kept: int, omitted: int, utilisation: float) -> None:
        with self._lock:
            entry = {
                "kept": kept,
                "omitted": omitted,
                "utilisation": round(utilisation, 3),
                "timestamp": time.time(),
            }
            self.keep_k_trims.append(entry)
            self._emit("keep_k_trim", data=entry)

    def force_end(self, estimated_tokens: int) -> None:
        with self._lock:
            entry = {
                "estimated_tokens": estimated_tokens,
                "timestamp": time.time(),
            }
            self.force_ends.append(entry)
            self._emit("force_end", data=entry)

    # ── Pipeline-specific events ──────────────────────────────────────

    def thinker_escalate(self) -> None:
        with self._lock:
            self.thinker_escalated = True
            self.thinker_escalate_time = time.time()
            self._emit("thinker_escalate")

    def corpus_update(self, admitted: int, total: int, iteration: int) -> None:
        with self._lock:
            entry = {
                "admitted": admitted,
                "total": total,
                "iteration": iteration,
                "timestamp": time.time(),
            }
            self.corpus_updates.append(entry)
            self._emit("corpus_update", data=entry)

    def stall_detected(self, agent: str, event_count: int, timeout: float) -> None:
        with self._lock:
            entry = {
                "agent": agent,
                "event_count": event_count,
                "timeout": timeout,
                "timestamp": time.time(),
            }
            self.stall_events.append(entry)
            self._emit("stall_detected", agent=agent, data=entry)

    # ── Snapshot (for SSE) ────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Lightweight snapshot for real-time SSE streaming."""
        with self._lock:
            elapsed = time.time() - self.started_at
            return {
                "session_id": self.session_id,
                "query": self.query[:200],
                "elapsed_secs": round(elapsed, 1),
                "current_phase": self._current_phase,
                "phases": [
                    {
                        "phase": p["phase"],
                        "agent": p["agent"],
                        "elapsed": round(
                            (p["end_time"] or time.time()) - p["start_time"], 1
                        ),
                        "outcome": p["outcome"],
                    }
                    for p in self.phases
                ],
                "kpi": {
                    "adk_events": self.total_adk_events,
                    "tool_calls": len(self.tool_calls),
                    "llm_calls": len(self.llm_calls),
                    "text_chars": self.total_text_chars,
                    "reasoning_chars": self.total_reasoning_chars,
                    "dedup_blocks": len(self.dedup_blocks),
                    "arg_fixes": len(self.arg_fixes),
                    "bad_results": len(self.bad_results),
                    "compressions": len(self.compressions),
                    "keep_k_trims": len(self.keep_k_trims),
                    "corpus_atoms": (
                        self.corpus_updates[-1]["total"]
                        if self.corpus_updates
                        else 0
                    ),
                },
                "thinker_escalated": self.thinker_escalated,
                "stalled": len(self.stall_events) > 0,
                "finalized": self._finalized,
                "event_count": len(self.events),
            }

    # ── Finalize ──────────────────────────────────────────────────────

    def finalize(self, result_text: str = "") -> dict[str, Any]:
        """Produce the full dashboard dict and save to disk."""
        with self._lock:
            self._finalized = True
            self._finalized_at = time.time()
            elapsed = self._finalized_at - self.started_at

            data = {
                "session_id": self.session_id,
                "query": self.query,
                "started_at": self.started_at,
                "finalized_at": self._finalized_at,
                "elapsed_secs": round(elapsed, 2),
                "result_length": len(result_text),
                "phases": self.phases,
                "kpi": {
                    "adk_events": self.total_adk_events,
                    "tool_calls": len(self.tool_calls),
                    "llm_calls": len(self.llm_calls),
                    "tool_errors": sum(1 for t in self.tool_calls if t.error),
                    "llm_errors": 0,
                    "text_chars": self.total_text_chars,
                    "reasoning_chars": self.total_reasoning_chars,
                    "prompt_tokens_est": sum(
                        r.prompt_tokens_est for r in self.llm_calls
                    ),
                    "completion_tokens_est": sum(
                        r.completion_tokens_est for r in self.llm_calls
                    ),
                    "elapsed_secs": round(elapsed, 2),
                },
                "algorithms": {
                    "dedup_blocks": self.dedup_blocks,
                    "arg_fixes": self.arg_fixes,
                    "bad_results": self.bad_results,
                    "compressions": self.compressions,
                    "keep_k_trims": self.keep_k_trims,
                    "force_ends": self.force_ends,
                },
                "tool_calls": [
                    {
                        "tool_name": t.tool_name,
                        "agent": t.agent,
                        "args_summary": t.args_summary,
                        "start_time": t.start_time,
                        "end_time": t.end_time,
                        "duration_secs": round(t.duration_secs, 3),
                        "result_chars": t.result_chars,
                        "error": t.error,
                        "was_dedup_blocked": t.was_dedup_blocked,
                        "arg_fix_applied": t.arg_fix_applied,
                        "was_compressed": t.was_compressed,
                        "original_chars": t.original_chars,
                    }
                    for t in self.tool_calls
                ],
                "tool_summary": self._build_tool_summary(),
                "llm_calls": [
                    {
                        "agent": r.agent,
                        "start_time": r.start_time,
                        "end_time": r.end_time,
                        "duration_secs": round(r.duration_secs, 3),
                        "prompt_tokens_est": r.prompt_tokens_est,
                        "completion_tokens_est": r.completion_tokens_est,
                    }
                    for r in self.llm_calls
                ],
                "corpus_updates": self.corpus_updates,
                "stall_events": self.stall_events,
                "thinker_escalated": self.thinker_escalated,
                "thinker_escalate_time": self.thinker_escalate_time,
                "events": self.events,
            }

            # Save to disk
            self._save(data)
            return data

    # ── Internal helpers ──────────────────────────────────────────────

    def _emit(
        self,
        event_type: str,
        agent: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append a raw event (must be called under self._lock)."""
        self.events.append(
            {
                "event_type": event_type,
                "timestamp": time.time(),
                "agent": agent,
                "phase": self._current_phase,
                "data": data or {},
            }
        )

    def _build_tool_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for t in self.tool_calls:
            if t.tool_name not in summary:
                summary[t.tool_name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "errors": 0,
                    "total_result_chars": 0,
                }
            s = summary[t.tool_name]
            s["count"] += 1
            s["total_duration"] = round(s["total_duration"] + t.duration_secs, 3)
            s["total_result_chars"] += t.result_chars
            if t.error:
                s["errors"] += 1
        return summary

    def _save(self, data: dict[str, Any]) -> None:
        """Write the finalized data to dashboard_logs/."""
        os.makedirs(_DASHBOARD_LOGS_DIR, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(self.started_at))
        filename = f"pipeline_{self.session_id[:8]}_{ts}.json"
        path = os.path.join(_DASHBOARD_LOGS_DIR, filename)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info("Dashboard data saved to %s", path)
        except Exception:
            logger.exception("Failed to save dashboard data to %s", path)
