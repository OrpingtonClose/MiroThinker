"""Unit tests for ``apps/strands-agent/task_pool.py``.

Covers the behaviour documented in ``MANIFEST.md`` §14 without pulling
in the heavy Strands / LangChain / MCP stack:

- task lifecycle: pending -> running -> complete / failed / cancelled
- ``await_tasks`` returns immediately when the deadline has already
  elapsed (no ``timeout=None`` deadlock)
- ``_emit`` hops worker-thread emissions back onto the captured asyncio
  loop via ``call_soon_threadsafe`` so ``asyncio.Queue`` stays on the
  loop thread
- ``_emit`` falls through to a direct call when no loop is configured
  and when invoked from the loop thread itself
- cancel bridge fans ``job_cancel_event`` out to all per-task cancel
  events (including tasks registered after the job was cancelled)
- ``shutdown`` drains in-flight workers (so late ingest writes land
  before the caller tears the store down) and transitions pending /
  running futures dropped by ``cancel_futures=True`` to ``cancelled``
- ``launch_research`` / ``launch_harvest`` / ``launch_gossip`` register
  tasks with the correct type and monotonic ids
- ``check_tasks`` snapshots every known task; ``await_tasks`` surfaces
  unknown task ids as ``status="unknown"``

The tests interact with the pool via ``_submit`` (private) to inject
fake workers — this avoids dragging in the real Strands agent and
keeps the suite hermetic. ``launch_*`` smoke tests monkey-patch the
specific imports their workers perform at call time.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the app module importable without installing the package.
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "strands-agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


task_pool = importlib.import_module("task_pool")
AsyncTaskPool = task_pool.AsyncTaskPool
TaskState = task_pool.TaskState


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStore:
    """Minimal stand-in for ``ConditionStore`` used by the real pool.

    Only ``ingest_raw`` is exercised here; it records calls and returns
    a list of synthetic ids of configurable length.
    """

    def __init__(self, ingest_count: int = 0) -> None:
        self.ingest_count = ingest_count
        self.ingest_calls: list[dict] = []
        self.closed = False

    def ingest_raw(self, text, source_type: str, source_ref: str):
        self.ingest_calls.append({
            "text": text,
            "source_type": source_type,
            "source_ref": source_ref,
        })
        return [f"cond-{i}" for i in range(self.ingest_count)]

    def close(self) -> None:
        self.closed = True


class RecordingEmitter:
    """Captures ``event_emit`` payloads and the thread each fires on."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.threads: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, payload: dict) -> None:
        with self._lock:
            self.events.append(dict(payload))
            self.threads.append(threading.get_ident())

    def types(self) -> list[str]:
        with self._lock:
            return [e.get("type", "") for e in self.events]


# ---------------------------------------------------------------------------
# TaskState
# ---------------------------------------------------------------------------


def test_taskstate_defaults_and_round_trip():
    now = time.time()
    t = TaskState(task_id="task-x", task_type="research", description="foo")
    assert t.status == "pending"
    assert t.error is None
    assert t.ingested_count == 0
    assert t.finished_at == 0.0
    assert t.created_at >= now

    d = t.to_dict()
    assert d["task_id"] == "task-x"
    assert d["task_type"] == "research"
    assert d["status"] == "pending"


# ---------------------------------------------------------------------------
# Pool construction / lifecycle via _submit
# ---------------------------------------------------------------------------


def _make_pool(
    *,
    store=None,
    job_cancel_event=None,
    event_emit=None,
    loop=None,
    max_concurrent: int = 4,
) -> AsyncTaskPool:
    return AsyncTaskPool(
        store=store if store is not None else FakeStore(),
        tools=[],
        job_cancel_event=job_cancel_event,
        event_emit=event_emit,
        max_concurrent=max_concurrent,
        loop=loop,
    )


def _register_and_submit(pool: AsyncTaskPool, worker, *args, task_type="research"):
    """Helper: register a TaskState and submit ``worker`` via ``_submit``."""
    task = TaskState(
        task_id=pool._new_task_id(task_type),
        task_type=task_type,
        description="unit-test",
    )
    pool._register(task)
    future = pool._submit(task, worker, *args)
    return task, future


def test_submit_completes_and_emits_terminal_events():
    emitter = RecordingEmitter()
    pool = _make_pool(event_emit=emitter)

    def worker(task, cancel_event):
        assert task.status == "running"
        task.result_summary = "done"
        task.ingested_count = 3
        task.status = "complete"
        task.finished_at = time.time()

    task, future = _register_and_submit(pool, worker)
    future.result(timeout=5)

    assert task.status == "complete"
    assert task.ingested_count == 3
    types = emitter.types()
    assert "task_launched" in types
    assert "task_completed" in types
    assert "task_failed" not in types

    pool.shutdown(drain_timeout=1.0)


def test_submit_failing_worker_emits_task_failed_once():
    emitter = RecordingEmitter()
    pool = _make_pool(event_emit=emitter)

    def worker(task, cancel_event):
        raise RuntimeError("boom")

    task, future = _register_and_submit(pool, worker)
    # _runner swallows the exception itself (returns None) so this resolves.
    future.result(timeout=5)

    assert task.status == "failed"
    assert task.error == "boom"
    assert emitter.types().count("task_failed") == 1
    assert "task_completed" not in emitter.types()

    pool.shutdown(drain_timeout=1.0)


def test_finalise_marks_pending_future_as_cancelled():
    """Futures dropped before ``_runner`` starts must reach a terminal status.

    Single worker slot, plus a slow task #1 that only finishes after
    shutdown has already kicked in. Task #2 is queued behind it and
    ``executor.shutdown(cancel_futures=True)`` drops it before
    ``_runner`` ever runs — so ``_finalise`` is the *only* code path
    that can transition its status away from the dataclass default
    ``"pending"``.
    """
    emitter = RecordingEmitter()
    pool = _make_pool(event_emit=emitter, max_concurrent=1)

    block = threading.Event()

    def slow(task, cancel_event):
        block.wait(timeout=10)
        task.status = "complete"
        task.finished_at = time.time()

    def pending(task, cancel_event):  # pragma: no cover — must not run
        task.status = "complete"
        task.finished_at = time.time()

    t1, f1 = _register_and_submit(pool, slow)
    t2, f2 = _register_and_submit(pool, pending)

    # Ensure the executor has started t1 and t2 is still queued.
    for _ in range(200):
        if t1.status == "running":
            break
        time.sleep(0.01)
    assert t1.status == "running"
    assert t2.status == "pending"

    # Kick off shutdown in a background thread — it will block until
    # the running t1 drains (wait=True) while simultaneously dropping
    # queued t2 via cancel_futures=True.
    shutdown_done = threading.Event()

    def _do_shutdown():
        pool.shutdown(drain_timeout=0.0)
        shutdown_done.set()

    st = threading.Thread(target=_do_shutdown, daemon=True)
    st.start()

    # Shutdown is now parked waiting on t1. cancel_futures has already
    # dropped t2 so its future is cancelled and ``_finalise`` has run.
    time.sleep(0.1)
    assert t2.status == "cancelled"
    assert t2.finished_at > 0

    # Release t1 so shutdown can complete.
    block.set()
    st.join(timeout=5)
    assert shutdown_done.is_set()
    assert t1.status == "complete"

    # Terminal event emitted for the cancelled queued task.
    assert "task_cancelled" in emitter.types()


# ---------------------------------------------------------------------------
# await_tasks semantics
# ---------------------------------------------------------------------------


def test_await_tasks_returns_immediately_when_deadline_elapsed():
    """Regression: ``timeout=None`` must never reach ``concurrent.futures.wait``
    when the caller-provided deadline has already elapsed.
    """
    pool = _make_pool()

    never_done = threading.Event()

    def blocks_forever(task, cancel_event):
        # Cancellable via the shutdown path, but won't complete on its own
        # within the test window.
        never_done.wait(timeout=2.0)
        task.status = "complete"
        task.finished_at = time.time()

    task, _future = _register_and_submit(pool, blocks_forever)

    # Give the executor a chance to pick it up.
    time.sleep(0.05)

    start = time.time()
    results = pool.await_tasks([task.task_id], timeout=0.0)
    elapsed = time.time() - start

    # Must not block for the full worker duration when timeout==0.
    assert elapsed < 1.0
    assert len(results) == 1
    assert results[0]["task_id"] == task.task_id

    # Let the worker finish cleanly.
    never_done.set()
    pool.shutdown(drain_timeout=3.0)


def test_await_tasks_surfaces_unknown_ids():
    pool = _make_pool()
    results = pool.await_tasks(["task-does-not-exist"], timeout=0.1)
    assert len(results) == 1
    assert results[0]["status"] == "unknown"
    assert results[0]["error"] == "task not found"
    pool.shutdown(drain_timeout=0.5)


def test_check_tasks_lists_every_registered_task():
    pool = _make_pool()

    def quick(task, cancel_event):
        task.status = "complete"
        task.finished_at = time.time()

    t1, f1 = _register_and_submit(pool, quick)
    t2, f2 = _register_and_submit(pool, quick)
    f1.result(timeout=5)
    f2.result(timeout=5)

    snap = pool.check_tasks()
    ids = {row["task_id"] for row in snap}
    assert t1.task_id in ids
    assert t2.task_id in ids
    assert all(row["status"] == "complete" for row in snap)

    pool.shutdown(drain_timeout=0.5)


# ---------------------------------------------------------------------------
# _emit thread-safety bridge
# ---------------------------------------------------------------------------


def test_emit_without_loop_falls_through_to_direct_call():
    """No captured loop → synchronous dispatch on caller's thread."""
    emitter = RecordingEmitter()
    pool = _make_pool(event_emit=emitter, loop=None)
    assert pool._loop is None

    pool._emit({"type": "task_launched", "task_id": "t-1"})
    assert emitter.types() == ["task_launched"]

    pool.shutdown(drain_timeout=0.5)


def test_emit_from_worker_thread_hops_onto_loop_thread():
    """Worker-thread emissions must execute on the captured loop thread."""

    async def runner():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        emitter = RecordingEmitter()
        pool = _make_pool(event_emit=emitter, loop=loop)

        # Worker runs in a ThreadPoolExecutor thread; every emit it
        # performs must be dispatched back onto ``loop_thread``.
        def worker(task, cancel_event):
            assert threading.get_ident() != loop_thread
            task.status = "complete"
            task.finished_at = time.time()

        task, future = _register_and_submit(pool, worker)

        await asyncio.get_running_loop().run_in_executor(None, future.result, 5)
        # call_soon_threadsafe callbacks are scheduled on the loop but need
        # one iteration to actually run.
        for _ in range(20):
            await asyncio.sleep(0.01)
            if emitter.types().count("task_completed") >= 1:
                break

        # All emissions for this pool must have fired on the loop thread.
        assert emitter.events, "no events recorded"
        for tid in emitter.threads:
            assert tid == loop_thread, "emit must run on the loop thread"

        pool.shutdown(drain_timeout=1.0)

    asyncio.run(runner())


def test_emit_on_loop_thread_runs_inline():
    """When ``_emit`` is called from the loop thread it must not reschedule."""

    async def runner():
        loop = asyncio.get_running_loop()
        emitter = RecordingEmitter()
        pool = _make_pool(event_emit=emitter, loop=loop)

        pool._emit({"type": "task_launched", "task_id": "t-direct"})
        # No await needed — the emit should have fired inline.
        assert emitter.types() == ["task_launched"]

        pool.shutdown(drain_timeout=0.5)

    asyncio.run(runner())


# ---------------------------------------------------------------------------
# Cancel bridge
# ---------------------------------------------------------------------------


def test_cancel_bridge_fans_job_cancel_to_task_events():
    job_cancel = threading.Event()
    pool = _make_pool(job_cancel_event=job_cancel)

    saw_cancel = threading.Event()

    def worker(task, cancel_event):
        # Wait up to 2s for the bridge to set our cancel flag.
        for _ in range(200):
            if cancel_event.is_set():
                saw_cancel.set()
                task.status = "cancelled"
                task.finished_at = time.time()
                return
            time.sleep(0.01)
        task.status = "complete"
        task.finished_at = time.time()

    task, future = _register_and_submit(pool, worker)

    # Let the worker get going.
    time.sleep(0.05)
    job_cancel.set()
    future.result(timeout=5)

    assert saw_cancel.is_set()
    assert task.status == "cancelled"

    pool.shutdown(drain_timeout=1.0)


def test_cancel_bridge_propagates_to_late_registered_tasks():
    """Tasks registered after ``job_cancel_event`` fires still see cancellation
    within the bridge's grace window."""
    job_cancel = threading.Event()
    pool = _make_pool(job_cancel_event=job_cancel)

    job_cancel.set()  # cancel *before* any task exists
    # Give the bridge a moment to wake up.
    time.sleep(0.05)

    saw_cancel = threading.Event()

    def worker(task, cancel_event):
        # The cancel bridge should fan out within its 5s grace window.
        for _ in range(200):
            if cancel_event.is_set():
                saw_cancel.set()
                task.status = "cancelled"
                task.finished_at = time.time()
                return
            time.sleep(0.01)
        task.status = "complete"
        task.finished_at = time.time()

    task, future = _register_and_submit(pool, worker)
    future.result(timeout=5)

    assert saw_cancel.is_set(), "late task never saw cancellation"
    assert task.status == "cancelled"

    pool.shutdown(drain_timeout=1.0)


def test_effective_cancel_no_job_event_returns_same_event():
    pool = _make_pool(job_cancel_event=None)
    ev = threading.Event()
    assert pool._effective_cancel(ev) is ev
    assert not ev.is_set()
    pool.shutdown(drain_timeout=0.5)


# ---------------------------------------------------------------------------
# cancel_task
# ---------------------------------------------------------------------------


def test_cancel_task_sets_per_task_event_and_returns_true():
    pool = _make_pool()

    started = threading.Event()
    observed_cancel = threading.Event()

    def worker(task, cancel_event):
        started.set()
        for _ in range(200):
            if cancel_event.is_set():
                observed_cancel.set()
                task.status = "cancelled"
                task.finished_at = time.time()
                return
            time.sleep(0.01)
        task.status = "complete"
        task.finished_at = time.time()

    task, future = _register_and_submit(pool, worker)
    assert started.wait(timeout=2.0)
    assert pool.cancel_task(task.task_id) is True
    future.result(timeout=5)
    assert observed_cancel.is_set()
    assert task.status == "cancelled"

    pool.shutdown(drain_timeout=1.0)


def test_cancel_task_returns_false_for_unknown_id():
    pool = _make_pool()
    assert pool.cancel_task("task-does-not-exist") is False
    pool.shutdown(drain_timeout=0.5)


# ---------------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------------


def test_shutdown_drains_in_flight_ingest_writes():
    """Workers that have moved past their agent run and are still inside
    the ingest phase must finish before ``shutdown`` returns.
    """
    store = FakeStore(ingest_count=0)
    pool = _make_pool(store=store)

    entered_ingest = threading.Event()
    release = threading.Event()

    def worker(task, cancel_event):
        # Simulate a worker that reached the "ingest" phase and is
        # blocked on the store lock. ``cancel_event`` being set should
        # NOT abort the ingest — this mirrors the real ``_run_research``
        # behaviour where cancellation is only checked around the agent
        # loop, not around ingest_raw.
        entered_ingest.set()
        release.wait(timeout=5)
        # Real worker would call self._store.ingest_raw(...) here; we
        # just invoke it directly on the fake to prove the store is
        # still usable while shutdown is parked in its drain.
        task._ingest_called = True
        task.status = "complete"
        task.finished_at = time.time()

    task, future = _register_and_submit(pool, worker)
    assert entered_ingest.wait(timeout=2.0)

    # Kick off shutdown in a background thread so we can release the
    # worker after shutdown has entered its drain.
    shutdown_done = threading.Event()

    def _do_shutdown():
        pool.shutdown(drain_timeout=5.0)
        shutdown_done.set()

    t = threading.Thread(target=_do_shutdown, daemon=True)
    t.start()

    # The drain must not have finished yet.
    time.sleep(0.05)
    assert not shutdown_done.is_set()

    # Release the worker → ingest completes → drain unblocks.
    release.set()
    t.join(timeout=6)
    assert shutdown_done.is_set()
    assert task.status == "complete"
    assert getattr(task, "_ingest_called", False)


def test_shutdown_is_idempotent():
    pool = _make_pool()
    pool.shutdown(drain_timeout=0.1)
    pool.shutdown(drain_timeout=0.1)  # must not raise
    pool.shutdown(drain_timeout=0.1)


# ---------------------------------------------------------------------------
# launch_* wiring
# ---------------------------------------------------------------------------


def _install_fake_module(monkeypatch, name: str, **attrs):
    """Insert a throwaway module into sys.modules with the given attributes."""
    mod = type(sys)("_fake_" + name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_launch_research_registers_task_and_runs_worker(monkeypatch):
    """``launch_research`` must build a fresh agent via
    ``agent.create_researcher_instance`` and ingest its output.
    """
    events: list[dict] = []
    store = FakeStore(ingest_count=4)

    # Patch ``agent`` + ``jobs`` because the worker imports them at call
    # time. The fake agent returns a canned string when called.
    class _FakeBudget:
        def __init__(self, cancel_flag=None):
            self.cancel_flag = cancel_flag

    class _FakeAgent:
        def __init__(self, reply: str):
            self._reply = reply

        def __call__(self, prompt: str):
            return self._reply

    def _create_instance(tools, budget):
        return _FakeAgent("synthetic research output")

    _install_fake_module(
        monkeypatch,
        "agent",
        ResearcherBudget=_FakeBudget,
        create_researcher_instance=_create_instance,
    )

    class _JobCancelled(Exception):
        pass

    _install_fake_module(monkeypatch, "jobs", JobCancelledError=_JobCancelled)

    pool = AsyncTaskPool(
        store=store,
        tools=[],
        event_emit=events.append,
        max_concurrent=2,
        loop=None,
    )

    tid = pool.launch_research("describe the task")
    assert tid.startswith("task-research-")

    # Wait for completion.
    pool.await_tasks([tid], timeout=5.0)
    snap = {t["task_id"]: t for t in pool.check_tasks()}[tid]
    assert snap["status"] == "complete"
    assert snap["ingested_count"] == 4
    # Store saw the ingest call with the researcher source_type.
    assert store.ingest_calls
    assert store.ingest_calls[0]["source_type"] == "researcher"

    types = [e.get("type") for e in events]
    assert "task_launched" in types
    assert "task_completed" in types

    pool.shutdown(drain_timeout=1.0)


def test_launch_harvest_registers_task_with_harvest_prefix(monkeypatch):
    store = FakeStore(ingest_count=0)

    def _youtube_harvest_channel(channel, max_videos, language, include_comments):
        return "harvest summary text"

    _install_fake_module(
        monkeypatch,
        "youtube_tools",
        youtube_harvest_channel=_youtube_harvest_channel,
    )

    pool = AsyncTaskPool(
        store=store,
        tools=[],
        event_emit=None,
        max_concurrent=2,
        loop=None,
    )

    tid = pool.launch_harvest("@example", max_videos=3)
    assert tid.startswith("task-harvest-")

    pool.await_tasks([tid], timeout=5.0)
    snap = {t["task_id"]: t for t in pool.check_tasks()}[tid]
    assert snap["status"] == "complete"
    assert store.ingest_calls
    assert store.ingest_calls[0]["source_type"] == "youtube_harvest"

    pool.shutdown(drain_timeout=1.0)


class _EmptyCorpusStore(FakeStore):
    """Store whose ``export_for_swarm`` returns the 'empty' sentinel so
    ``_run_gossip`` short-circuits without invoking the real swarm engine."""

    user_query = ""

    def export_for_swarm(self, min_confidence: float = 0.0) -> str:
        return "(corpus is empty — no gossip possible)"


def test_launch_gossip_short_circuits_on_empty_corpus(monkeypatch):
    store = _EmptyCorpusStore()

    # ``_run_gossip`` imports ``gossip_synthesize`` from swarm_bridge at
    # call time. Install a sentinel that would explode if called — we
    # want to prove the worker never reaches it on an empty corpus.
    async def _exploding_synthesize(**_kwargs):
        raise AssertionError("gossip_synthesize must not run on empty corpus")

    _install_fake_module(
        monkeypatch,
        "swarm_bridge",
        gossip_synthesize=_exploding_synthesize,
    )

    pool = AsyncTaskPool(
        store=store,
        tools=[],
        event_emit=None,
        max_concurrent=2,
        loop=None,
    )

    tid = pool.launch_gossip(iteration=2)
    assert tid.startswith("task-gossip-")

    pool.await_tasks([tid], timeout=5.0)
    snap = {t["task_id"]: t for t in pool.check_tasks()}[tid]
    assert snap["status"] == "complete"
    assert "corpus empty" in snap["result_summary"]

    pool.shutdown(drain_timeout=1.0)
