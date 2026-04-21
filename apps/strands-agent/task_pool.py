# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Async task pool for parallel heterogeneous background work.

The orchestrator launches tasks via ``launch_*()`` tools. Each task
runs in its own thread (Strands agents are sync) with its own agent
instance. Results are auto-ingested into the shared ConditionStore so
downstream read-only tools (``query_corpus``, ``assess_coverage``, …)
pick them up automatically.

Task types:
- ``research``: fresh Strands researcher agent doing web / forum /
  academic search.
- ``harvest``: YouTube channel bulk download (Apify / yt-dlp cascade).
- ``gossip``: GossipSwarm synthesis on the current corpus.
- ``ingest``: placeholder for knowledge-engine ETL (future).

See ``MANIFEST.md`` sections 4–6 for the full design.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# TaskState
# ──────────────────────────────────────────────────────────────────────


_TERMINAL_STATUSES = frozenset({"complete", "failed", "cancelled"})


@dataclass
class TaskState:
    """Snapshot of a single background task inside the pool."""

    task_id: str
    task_type: str              # research | harvest | gossip | ingest
    description: str
    status: str = "pending"     # pending | running | complete | failed | cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    result_summary: str = ""
    error: str | None = None
    progress: str = ""
    ingested_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# AsyncTaskPool
# ──────────────────────────────────────────────────────────────────────


class AsyncTaskPool:
    """Owns a ThreadPoolExecutor + task registry for one job.

    One pool per ``/query/multi`` job. The pool:
    - spawns a fresh Strands researcher per research task (no shared
      conversation state, no shared budget);
    - runs harvest / gossip in background threads;
    - auto-ingests results into the job's ``ConditionStore``;
    - forwards task-lifecycle events to the job's SSE queue via
      ``event_emit``.

    The pool is **not** thread-safe across multiple callers racing to
    mutate its registry from the same thread, but it is safe for the
    common pattern: orchestrator thread calls ``launch_*`` / ``check_*``
    / ``await_*``, worker threads mutate their own ``TaskState`` only.
    """

    def __init__(
        self,
        store: Any,                                   # ConditionStore (avoid circular import)
        tools: list | None = None,
        job_cancel_event: threading.Event | None = None,
        event_emit: Callable[[dict[str, Any]], None] | None = None,
        max_concurrent: int = 4,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._store = store
        self._tools = list(tools or [])
        self._job_cancel_event = job_cancel_event
        self._event_emit = event_emit
        # Capture the asyncio loop at construction time so worker threads
        # can bridge events back into the loop via ``call_soon_threadsafe``
        # (asyncio.Queue is not thread-safe). ``_run_job`` constructs the
        # pool from the event loop; unit tests that call the pool
        # synchronously can pass ``loop=None``.
        if loop is not None:
            self._loop = loop
        else:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_concurrent)),
            thread_name_prefix="task-pool",
        )
        self._tasks: dict[str, TaskState] = {}
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._shutdown = False
        # Bridge thread that fans job_cancel_event -> every per-task
        # cancel event, including tasks registered after cancellation.
        self._cancel_bridge_started = False
        self._cancel_bridge_thread: threading.Thread | None = None
        if self._job_cancel_event is not None:
            self._start_cancel_bridge()

    # ── Event emission ──────────────────────────────────────────────

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._event_emit is None:
            return
        emit = self._event_emit
        loop = self._loop

        def _invoke() -> None:
            try:
                emit(payload)
            except Exception:
                logger.exception("task pool event_emit failed")

        # ``job.emit`` ultimately writes to an ``asyncio.Queue`` which is
        # NOT thread-safe. Worker threads must hop back onto the event
        # loop via ``call_soon_threadsafe`` before invoking the emit
        # callback; otherwise the awaiting ``Queue.get()`` may never be
        # woken and task lifecycle events would be silently lost.
        if loop is None or loop.is_closed():
            _invoke()
            return
        try:
            on_loop_thread = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop_thread = False
        if on_loop_thread:
            _invoke()
            return
        try:
            loop.call_soon_threadsafe(_invoke)
        except RuntimeError:
            # Loop already shut down; fall back to direct call.
            _invoke()

    # ── Cancel bridge ───────────────────────────────────────────────

    def _start_cancel_bridge(self) -> None:
        """Spawn a daemon thread that fans job cancellation out to tasks.

        Waits on ``job_cancel_event``. Once fired, every known per-task
        cancel event is set, and the bridge continues mirroring onto
        newly-registered task events for a short grace window so late
        launches also observe cancellation.
        """
        if self._cancel_bridge_started or self._job_cancel_event is None:
            return
        self._cancel_bridge_started = True

        def _bridge() -> None:
            job_ev = self._job_cancel_event
            if job_ev is None:
                return
            job_ev.wait()
            deadline = time.time() + 5.0
            while True:
                with self._lock:
                    events = list(self._cancel_events.values())
                for ev in events:
                    ev.set()
                if self._shutdown or time.time() >= deadline:
                    break
                time.sleep(0.05)

        thread = threading.Thread(
            target=_bridge,
            name="task-pool-cancel-bridge",
            daemon=True,
        )
        self._cancel_bridge_thread = thread
        thread.start()

    # ── Launch helpers ──────────────────────────────────────────────

    def _new_task_id(self, task_type: str) -> str:
        return f"task-{task_type}-{uuid.uuid4().hex[:10]}"

    def _register(self, task: TaskState) -> None:
        with self._lock:
            self._tasks[task.task_id] = task

    def _submit(
        self,
        task: TaskState,
        fn: Callable[..., Any],
        *args: Any,
    ) -> Future:
        """Submit a worker callable wrapped in status bookkeeping."""
        cancel_event = threading.Event()
        self._cancel_events[task.task_id] = cancel_event

        def _runner() -> Any:
            task.status = "running"
            self._emit({
                "type": "task_launched",
                "task_id": task.task_id,
                "task_type": task.task_type,
                "description": task.description,
            })
            try:
                return fn(task, cancel_event, *args)
            except Exception as exc:
                logger.exception(
                    "task_id=<%s> | task worker raised", task.task_id,
                )
                task.status = "failed"
                task.error = str(exc)
                task.finished_at = time.time()
                self._emit({
                    "type": "task_failed",
                    "task_id": task.task_id,
                    "error": task.error,
                })
                return None

        future = self._executor.submit(_runner)
        future.add_done_callback(
            lambda _fut, tid=task.task_id: self._finalise(tid),
        )
        self._futures[task.task_id] = future
        return future

    def _finalise(self, task_id: str) -> None:
        """Emit terminal event after a future settles."""
        task = self._tasks.get(task_id)
        if task is None:
            return
        if task.status in ("running", "pending"):
            # Worker didn't set a terminal status — treat as cancelled.
            # ``running`` covers the case where the executor was shut
            # down mid-flight; ``pending`` covers the case where
            # ``executor.shutdown(cancel_futures=True)`` dropped the
            # future before ``_runner`` ever started, so the status
            # field was never advanced past the dataclass default.
            task.status = "cancelled"
            task.finished_at = time.time()
        if task.status == "complete":
            self._emit({
                "type": "task_completed",
                "task_id": task.task_id,
                "ingested_count": task.ingested_count,
                "summary": task.result_summary,
            })
        elif task.status == "cancelled":
            self._emit({
                "type": "task_cancelled",
                "task_id": task.task_id,
            })
        # task_failed already emitted from _runner; avoid double-emit.

    # ── Public: launch ──────────────────────────────────────────────

    def launch_research(self, task_desc: str) -> str:
        """Launch a research task. Returns the task_id immediately.

        Spawns a **fresh** Strands researcher agent (never the singleton).
        The researcher's final response is auto-ingested into the corpus.
        """
        task = TaskState(
            task_id=self._new_task_id("research"),
            task_type="research",
            description=task_desc,
        )
        self._register(task)
        self._submit(task, self._run_research)
        return task.task_id

    def launch_harvest(
        self,
        channel: str,
        max_videos: int = 0,
        language: str = "en",
        include_comments: bool = True,
    ) -> str:
        """Launch a YouTube channel harvest. Returns task_id immediately."""
        task = TaskState(
            task_id=self._new_task_id("harvest"),
            task_type="harvest",
            description=f"harvest:{channel} max={max_videos} lang={language}",
        )
        self._register(task)
        self._submit(
            task, self._run_harvest,
            channel, max_videos, language, include_comments,
        )
        return task.task_id

    def launch_gossip(self, iteration: int = 0) -> str:
        """Launch a gossip synthesis task. Returns task_id immediately."""
        task = TaskState(
            task_id=self._new_task_id("gossip"),
            task_type="gossip",
            description=f"gossip:iter={iteration}",
        )
        self._register(task)
        self._submit(task, self._run_gossip, int(iteration))
        return task.task_id

    # ── Public: observation ─────────────────────────────────────────

    def check_tasks(self) -> list[dict[str, Any]]:
        """Return a status snapshot of every task the pool has seen."""
        with self._lock:
            return [t.to_dict() for t in self._tasks.values()]

    def await_tasks(
        self,
        task_ids: list[str],
        timeout: float = 600.0,
    ) -> list[dict[str, Any]]:
        """Block until every requested task reaches a terminal status."""
        deadline = time.time() + max(0.0, float(timeout))
        wanted = list(task_ids)
        futures: list[Future] = []
        with self._lock:
            for tid in wanted:
                fut = self._futures.get(tid)
                if fut is not None:
                    futures.append(fut)

        if futures:
            remaining = max(0.0, deadline - time.time())
            try:
                # Pass ``remaining`` directly. When the deadline has
                # already elapsed we want ``timeout=0.0`` (return
                # immediately), not ``timeout=None`` which would block
                # forever and deadlock the orchestrator tool thread.
                concurrent.futures.wait(futures, timeout=remaining)
            except Exception:
                logger.exception("await_tasks: concurrent.futures.wait failed")

        results: list[dict[str, Any]] = []
        with self._lock:
            for tid in wanted:
                task = self._tasks.get(tid)
                if task is None:
                    results.append({
                        "task_id": tid,
                        "status": "unknown",
                        "error": "task not found",
                    })
                else:
                    results.append(task.to_dict())
        return results

    def cancel_task(self, task_id: str) -> bool:
        """Signal cancellation for a single task."""
        ev = self._cancel_events.get(task_id)
        if ev is None:
            return False
        ev.set()
        future = self._futures.get(task_id)
        if future is not None:
            future.cancel()     # no-op once running, helps for pending
        return True

    def shutdown(self, drain_timeout: float = 30.0) -> None:
        """Signal cancellation to tasks and tear down the executor.

        Waits up to ``drain_timeout`` seconds for already-running workers
        to reach a terminal state before tearing down the pool. This
        lets workers that have finished their agent run but are still in
        the ``ingest_raw`` phase complete their writes to the shared
        ``ConditionStore`` before the caller (typically ``_run_job``)
        closes the store. Without this drain, ``store.close()`` races
        with in-flight ingest calls and silently drops results.
        """
        if self._shutdown:
            return
        self._shutdown = True
        # Signal cancellation so workers that are still running tool
        # calls / agent loops bail out quickly; workers that have moved
        # past the cancellation-sensitive phase (e.g. into ingest) will
        # ignore the flag and continue to completion.
        for ev in list(self._cancel_events.values()):
            ev.set()
        if self._job_cancel_event is not None:
            self._job_cancel_event.set()

        # Snapshot futures under the lock to avoid racing with
        # concurrent launch_* calls.
        with self._lock:
            pending = list(self._futures.values())

        if pending and drain_timeout > 0:
            try:
                concurrent.futures.wait(pending, timeout=drain_timeout)
            except Exception:
                logger.exception(
                    "task pool shutdown: drain wait failed",
                )

        try:
            # wait=True so any stragglers still in their finally / ingest
            # block get to finish rather than being hard-killed mid-write.
            # cancel_futures drops pending-but-not-started work.
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # cancel_futures only exists on Python >=3.9; we target 3.10+
            self._executor.shutdown(wait=True)
        except Exception:
            logger.exception("task pool shutdown: executor.shutdown failed")

    # ── Worker implementations ──────────────────────────────────────

    def _effective_cancel(
        self, task_cancel: threading.Event,
    ) -> threading.Event:
        """Union of per-task and job-level cancel events.

        Returns ``task_cancel`` itself but ensures it becomes set whenever
        the job-level event is set. Continuous job→task fan-out is
        handled by the cancel-bridge daemon thread started in
        ``__init__``; here we additionally mirror the current state
        synchronously for callers that race ahead of the bridge.
        """
        job_ev = self._job_cancel_event
        if job_ev is None:
            return task_cancel
        if job_ev.is_set():
            task_cancel.set()
        return task_cancel

    def _run_research(
        self,
        task: TaskState,
        task_cancel: threading.Event,
    ) -> None:
        """Background worker: run a fresh researcher on task.description."""
        from agent import ResearcherBudget, create_researcher_instance
        from jobs import JobCancelledError

        cancel_event = self._effective_cancel(task_cancel)
        budget = ResearcherBudget(cancel_flag=cancel_event)

        try:
            agent = create_researcher_instance(
                tools=self._tools,
                budget=budget,
            )
        except Exception as exc:
            task.status = "failed"
            task.error = f"failed to build researcher: {exc}"
            task.finished_at = time.time()
            self._emit({
                "type": "task_failed",
                "task_id": task.task_id,
                "error": task.error,
            })
            return

        raw_text = ""
        try:
            response = agent(task.description)
            raw_text = str(response)
        except JobCancelledError:
            task.status = "cancelled"
            task.finished_at = time.time()
            return
        except Exception as exc:
            logger.exception("research task failed")
            task.status = "failed"
            task.error = str(exc)
            task.finished_at = time.time()
            self._emit({
                "type": "task_failed",
                "task_id": task.task_id,
                "error": task.error,
            })
            return

        # Auto-ingest into ConditionStore.
        ingested_ids: list[Any] = []
        if raw_text.strip():
            try:
                ingested_ids = self._store.ingest_raw(
                    raw_text,
                    source_type="researcher",
                    source_ref=task.description[:200],
                )
            except Exception:
                logger.exception("research task ingest_raw failed")

        task.ingested_count = len(ingested_ids)
        task.result_summary = (
            f"ingested {task.ingested_count} findings "
            f"({len(raw_text)} chars of research text)"
        )
        task.status = "complete"
        task.finished_at = time.time()

    def _run_harvest(
        self,
        task: TaskState,
        task_cancel: threading.Event,
        channel: str,
        max_videos: int,
        language: str,
        include_comments: bool,
    ) -> None:
        """Background worker: run YouTube bulk harvest + ingest cache."""
        from youtube_tools import youtube_harvest_channel

        cancel_event = self._effective_cancel(task_cancel)

        summary_text = ""
        try:
            # youtube_harvest_channel is decorated with @tool — call the
            # underlying function directly by accessing ``__wrapped__``
            # when present, otherwise call the decorated object as-is.
            fn = getattr(youtube_harvest_channel, "__wrapped__", None)
            if fn is None:
                fn = youtube_harvest_channel
            summary_text = str(fn(
                channel=channel,
                max_videos=max_videos,
                language=language,
                include_comments=include_comments,
            ))
        except Exception as exc:
            logger.exception("harvest task failed")
            task.status = "failed"
            task.error = str(exc)
            task.finished_at = time.time()
            self._emit({
                "type": "task_failed",
                "task_id": task.task_id,
                "error": task.error,
            })
            return

        if cancel_event.is_set():
            task.status = "cancelled"
            task.finished_at = time.time()
            return

        # Ingest the harvest summary (transcripts + comments live in the
        # on-disk cache; the summary text contains enough context for the
        # researcher to query them via dedicated tools). Treat each
        # non-empty line of the summary as a finding stub; the atomizer
        # handles fine-grained extraction downstream.
        ingested_ids: list[Any] = []
        if summary_text.strip():
            try:
                ingested_ids = self._store.ingest_raw(
                    summary_text,
                    source_type="youtube_harvest",
                    source_ref=f"channel:{channel}",
                )
            except Exception:
                logger.exception("harvest task ingest_raw failed")

        task.ingested_count = len(ingested_ids)
        task.result_summary = (
            f"harvest summary ingested ({task.ingested_count} findings, "
            f"{len(summary_text)} chars)"
        )
        task.status = "complete"
        task.finished_at = time.time()

    def _run_gossip(
        self,
        task: TaskState,
        task_cancel: threading.Event,
        iteration: int,
    ) -> None:
        """Background worker: run GossipSwarm synthesis on current corpus."""
        from swarm_bridge import gossip_synthesize

        cancel_event = self._effective_cancel(task_cancel)

        try:
            corpus_text = self._store.export_for_swarm(min_confidence=0.0)
        except Exception as exc:
            task.status = "failed"
            task.error = f"export_for_swarm failed: {exc}"
            task.finished_at = time.time()
            self._emit({
                "type": "task_failed",
                "task_id": task.task_id,
                "error": task.error,
            })
            return

        if "(corpus is empty" in corpus_text:
            task.status = "complete"
            task.result_summary = "corpus empty — no gossip run"
            task.finished_at = time.time()
            return

        # Build corpus delta callback — lets the swarm pick up new
        # findings that producers ingested while gossip is running.
        from datetime import datetime, timezone

        watermark = datetime.now(timezone.utc).isoformat()

        async def _corpus_delta_fn() -> str:
            nonlocal watermark
            delta = self._store.export_delta(since=watermark)
            watermark = datetime.now(timezone.utc).isoformat()
            return delta

        # Wire up gap-driven research: when the swarm emits research
        # gaps, launch targeted research tasks automatically.
        async def _on_swarm_event(event: dict) -> None:
            if event.get("type") == "research_gap":
                for gap in event.get("gaps", [])[:5]:
                    try:
                        self.launch_research(
                            task_desc=f"TARGETED RESEARCH GAP: {gap}",
                        )
                    except Exception:
                        logger.warning("gap_text=<%s> | failed to launch gap research", gap[:80])

        # Export prior research (findings + thoughts from earlier iterations)
        # to feed into the swarm as additional context.
        try:
            prior_corpus = self._store.export_prior_research()
        except Exception:
            logger.warning("export_prior_research failed, continuing without prior corpus")
            prior_corpus = ""

        # We're in a worker thread — run the coroutine in a fresh loop.
        try:
            result = asyncio.run(
                gossip_synthesize(
                    corpus=corpus_text,
                    query=getattr(self._store, "user_query", "") or "",
                    cancel_event=None,   # thread cancellation only
                    corpus_delta_fn=_corpus_delta_fn,
                    on_event=_on_swarm_event,
                    lineage_store=self._store,  # persist all bee outputs
                    prior_corpus=prior_corpus,
                ),
            )
        except Exception as exc:
            logger.exception("gossip task failed")
            task.status = "failed"
            task.error = str(exc)
            task.finished_at = time.time()
            self._emit({
                "type": "task_failed",
                "task_id": task.task_id,
                "error": task.error,
            })
            return

        if cancel_event.is_set():
            task.status = "cancelled"
            task.finished_at = time.time()
            return

        metrics_dict: dict[str, Any] = {}
        m = getattr(result, "metrics", None)
        if m is not None:
            metrics_dict = {
                "info_gain": list(getattr(m, "gossip_info_gain", [])),
                "llm_calls": getattr(m, "total_llm_calls", 0),
                "elapsed_seconds": getattr(m, "total_elapsed_s", 0),
            }

        try:
            self._store.admit_synthesis(
                report=result.user_report,
                iteration=iteration,
                metrics=metrics_dict,
            )
        except Exception:
            logger.exception("gossip task admit_synthesis failed")

        task.ingested_count = 1
        task.result_summary = (
            f"synthesis: {len(result.user_report)} chars "
            f"(iteration={iteration}, metrics={json.dumps(metrics_dict)})"
        )
        task.status = "complete"
        task.finished_at = time.time()
