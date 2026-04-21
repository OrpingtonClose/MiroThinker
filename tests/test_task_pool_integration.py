"""Integration tests for ``apps/strands-agent/task_pool.py`` against a
real ``ConditionStore`` (DuckDB, in-memory).

Covers the three integration scenarios documented in
``apps/strands-agent/MANIFEST.md`` §14:

- Launch 3 research tasks in parallel, verify all findings appear in
  the shared ``ConditionStore``.
- Launch a harvest task + research task simultaneously, verify both
  complete and both ingest into the store.
- Launch gossip after research, verify the synthesis row is stored.

The Strands / LangChain / MCP stack is still not pulled in — the real
``ConditionStore`` is used, but the per-task workers (researcher agent,
YouTube harvest, gossip swarm) are monkey-patched with deterministic
fakes so the suite runs in under a second and requires no credentials.
"""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "strands-agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


task_pool = importlib.import_module("task_pool")
corpus = importlib.import_module("corpus")


AsyncTaskPool = task_pool.AsyncTaskPool
ConditionStore = corpus.ConditionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_agent_module(monkeypatch, response_texts: dict[str, str]):
    """Install a fake ``agent`` module that returns a canned response for
    each task description (keyed by substring match).
    """

    class _FakeBudget:
        def __init__(self, cancel_flag=None):
            self.cancel_flag = cancel_flag

    class _FakeAgent:
        def __init__(self, table: dict[str, str]):
            self._table = table

        def __call__(self, prompt: str) -> str:
            for needle, reply in self._table.items():
                if needle in prompt:
                    return reply
            return "(fake agent: no canned reply matched) " + prompt

    def _create_instance(tools, budget):
        return _FakeAgent(response_texts)

    fake_agent = type(sys)("_fake_agent")
    fake_agent.ResearcherBudget = _FakeBudget
    fake_agent.create_researcher_instance = _create_instance
    monkeypatch.setitem(sys.modules, "agent", fake_agent)

    class _JobCancelled(Exception):
        pass

    fake_jobs = type(sys)("_fake_jobs")
    fake_jobs.JobCancelledError = _JobCancelled
    monkeypatch.setitem(sys.modules, "jobs", fake_jobs)


def _install_fake_youtube(monkeypatch, summary: str):
    """Install a fake ``youtube_tools`` module whose harvester returns
    ``summary`` synchronously.
    """
    def _harvest(channel, max_videos, language, include_comments):
        # Mirror the real decorated tool's behaviour: return a text blob.
        return (
            f"Harvest summary for {channel}\n\n"
            f"- max_videos={max_videos} language={language} "
            f"include_comments={include_comments}\n\n"
            f"{summary}"
        )

    fake_yt = type(sys)("_fake_youtube_tools")
    fake_yt.youtube_harvest_channel = _harvest
    monkeypatch.setitem(sys.modules, "youtube_tools", fake_yt)


def _install_fake_swarm(monkeypatch, synthesis_body: str, metrics=None):
    """Install a fake ``swarm_bridge`` module providing a deterministic
    async ``gossip_synthesize`` that returns a SynthResult-shaped object.
    """

    class _Metrics:
        def __init__(self) -> None:
            self.gossip_info_gain = [0.5, 0.7]
            self.total_llm_calls = 12
            self.total_elapsed_s = 1.5

    class _Result:
        def __init__(self, report: str) -> None:
            self.user_report = report
            self.metrics = _Metrics()

    async def _gossip_synthesize(corpus: str, query: str = "", cancel_event=None):
        return _Result(synthesis_body)

    fake_sb = type(sys)("_fake_swarm_bridge")
    fake_sb.gossip_synthesize = _gossip_synthesize
    monkeypatch.setitem(sys.modules, "swarm_bridge", fake_sb)


# ---------------------------------------------------------------------------
# Scenario 1 — three research tasks in parallel
# ---------------------------------------------------------------------------


def test_three_parallel_research_tasks_ingest_into_shared_store(monkeypatch):
    """§14 case 1 — three research tasks running concurrently all land
    their findings in the shared ``ConditionStore`` with distinct
    ``source_ref`` values and no cross-contamination.
    """
    _install_fake_agent_module(
        monkeypatch,
        {
            "alpha": (
                "ALPHA paragraph one about economic policy.\n\n"
                "ALPHA paragraph two about central banks.\n\n"
                "ALPHA paragraph three about inflation data."
            ),
            "beta": (
                "BETA paragraph one about climate metrics.\n\n"
                "BETA paragraph two about renewable energy.\n\n"
                "BETA paragraph three about carbon pricing."
            ),
            "gamma": (
                "GAMMA paragraph one about biomedical research.\n\n"
                "GAMMA paragraph two about clinical trials.\n\n"
                "GAMMA paragraph three about peer review."
            ),
        },
    )

    store = ConditionStore("")  # in-memory DuckDB
    events: list[dict] = []

    try:
        pool = AsyncTaskPool(
            store=store,
            tools=[],
            event_emit=events.append,
            max_concurrent=4,
            loop=None,
        )

        tids = [
            pool.launch_research("research alpha policy"),
            pool.launch_research("research beta climate"),
            pool.launch_research("research gamma biomed"),
        ]

        results = pool.await_tasks(tids, timeout=10.0)

        # All three tasks finished successfully.
        assert len(results) == 3
        for r in results:
            assert r["status"] == "complete", r
            assert r["ingested_count"] >= 3
            assert r["error"] in (None, "")

        # Every task got its own researcher instance — no shared budget
        # state, so ingested counts sum cleanly.
        total_findings = sum(r["ingested_count"] for r in results)
        assert total_findings == 9  # 3 paragraphs each × 3 tasks

        # Store snapshot: each alpha/beta/gamma paragraph appears exactly once.
        rows = store.conn.execute(
            "SELECT fact FROM conditions WHERE row_type = 'finding' ORDER BY id"
        ).fetchall()
        facts = [r[0] for r in rows]
        assert len(facts) == 9
        alpha_facts = [f for f in facts if "ALPHA" in f]
        beta_facts = [f for f in facts if "BETA" in f]
        gamma_facts = [f for f in facts if "GAMMA" in f]
        assert len(alpha_facts) == 3
        assert len(beta_facts) == 3
        assert len(gamma_facts) == 3

        # Every raw row has source_type='researcher'.
        raw_source_types = {
            r[0] for r in store.conn.execute(
                "SELECT DISTINCT source_type FROM conditions WHERE row_type = 'raw'"
            ).fetchall()
        }
        assert raw_source_types == {"researcher"}

        # Task lifecycle events: three launches + three completions.
        types = [e.get("type") for e in events]
        assert types.count("task_launched") == 3
        assert types.count("task_completed") == 3
        assert "task_failed" not in types

        # export_for_swarm surfaces every finding (not the '(empty)'
        # sentinel).
        exported = store.export_for_swarm()
        assert "(corpus is empty" not in exported
        for needle in ("ALPHA paragraph", "BETA paragraph", "GAMMA paragraph"):
            assert needle in exported

        pool.shutdown(drain_timeout=2.0)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Scenario 2 — harvest + research simultaneously
# ---------------------------------------------------------------------------


def test_harvest_and_research_run_simultaneously(monkeypatch):
    """§14 case 2 — a long-running harvest must not block a research
    task running in parallel, and both must ingest into the shared
    store with distinct ``source_type`` labels.
    """
    _install_fake_agent_module(
        monkeypatch,
        {
            "research": (
                "Research finding one on topic.\n\n"
                "Research finding two on topic."
            ),
        },
    )

    # Make the harvester deliberately slow so we can prove the research
    # task completes concurrently (not serially).
    harvest_started = threading.Event()
    harvest_release = threading.Event()

    def _slow_harvest(channel, max_videos, language, include_comments):
        harvest_started.set()
        # Block until the research task has completed (or timeout).
        harvest_release.wait(timeout=5.0)
        return (
            f"Harvest channel={channel} max_videos={max_videos}\n\n"
            f"Video transcript one snippet.\n\n"
            f"Video transcript two snippet."
        )

    fake_yt = type(sys)("_fake_youtube_tools")
    fake_yt.youtube_harvest_channel = _slow_harvest
    monkeypatch.setitem(sys.modules, "youtube_tools", fake_yt)

    store = ConditionStore("")

    try:
        pool = AsyncTaskPool(
            store=store,
            tools=[],
            event_emit=None,
            max_concurrent=4,
            loop=None,
        )

        harvest_id = pool.launch_harvest("@example-channel", max_videos=5)
        research_id = pool.launch_research("research this topic")

        # Wait for harvest to actually start.
        assert harvest_started.wait(timeout=2.0)

        # Research task should finish while harvest is still parked.
        research_result = pool.await_tasks([research_id], timeout=5.0)[0]
        assert research_result["status"] == "complete"
        assert research_result["ingested_count"] == 2

        # Harvest is still parked — prove it by checking status before
        # releasing it.
        mid_snapshot = {
            t["task_id"]: t for t in pool.check_tasks()
        }[harvest_id]
        assert mid_snapshot["status"] == "running"

        # Release the harvest and wait for it.
        harvest_release.set()
        harvest_result = pool.await_tasks([harvest_id], timeout=5.0)[0]
        assert harvest_result["status"] == "complete"
        assert harvest_result["ingested_count"] >= 1

        # Store now has findings with both source_types.
        counts = dict(store.conn.execute(
            "SELECT source_type, COUNT(*) FROM conditions "
            "WHERE row_type = 'finding' GROUP BY source_type"
        ).fetchall())
        assert counts.get("researcher", 0) == 2
        assert counts.get("youtube_harvest", 0) >= 1

        # source_ref for the harvest row encodes the channel.
        harvest_refs = {
            r[0] for r in store.conn.execute(
                "SELECT DISTINCT source_ref FROM conditions "
                "WHERE source_type = 'youtube_harvest'"
            ).fetchall()
        }
        assert any("@example-channel" in ref for ref in harvest_refs)

        pool.shutdown(drain_timeout=2.0)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Scenario 3 — gossip after research
# ---------------------------------------------------------------------------


def test_gossip_after_research_stores_synthesis(monkeypatch):
    """§14 case 3 — launching gossip after research tasks have ingested
    findings produces a synthesis row in the store.
    """
    _install_fake_agent_module(
        monkeypatch,
        {
            "seed": (
                "Seed finding one.\n\n"
                "Seed finding two.\n\n"
                "Seed finding three."
            ),
        },
    )
    _install_fake_swarm(monkeypatch, synthesis_body=(
        "SYNTHESIS REPORT\n"
        "================\n"
        "Combined view of the research corpus with three key insights."
    ))

    store = ConditionStore("")
    store.user_query = "integration-test query"
    events: list[dict] = []

    try:
        pool = AsyncTaskPool(
            store=store,
            tools=[],
            event_emit=events.append,
            max_concurrent=4,
            loop=None,
        )

        # Seed the corpus with research findings first.
        research_id = pool.launch_research("seed the corpus")
        r = pool.await_tasks([research_id], timeout=5.0)[0]
        assert r["status"] == "complete"
        assert r["ingested_count"] == 3

        # ``export_for_swarm`` must now surface non-empty findings so the
        # gossip worker does NOT short-circuit.
        exported = store.export_for_swarm()
        assert "(corpus is empty" not in exported
        assert "Seed finding" in exported

        # Launch gossip.
        gossip_id = pool.launch_gossip(iteration=1)
        gr = pool.await_tasks([gossip_id], timeout=5.0)[0]
        assert gr["status"] == "complete"
        # Gossip always records exactly one synthesis row.
        assert gr["ingested_count"] == 1
        assert "synthesis" in gr["result_summary"].lower()

        # Synthesis row is in the store.
        synth_rows = store.conn.execute(
            "SELECT fact, source_type, strategy, iteration "
            "FROM conditions WHERE row_type = 'synthesis'"
        ).fetchall()
        assert len(synth_rows) == 1
        fact, src_type, strategy, itr = synth_rows[0]
        assert "SYNTHESIS REPORT" in fact
        assert src_type == "gossip_swarm"
        assert itr == 1
        # Metrics serialised into strategy as JSON.
        assert "llm_calls" in strategy
        assert "info_gain" in strategy
        assert "elapsed_seconds" in strategy

        # Events include task_completed for both tasks.
        types = [e.get("type") for e in events]
        assert types.count("task_launched") == 2
        assert types.count("task_completed") == 2

        pool.shutdown(drain_timeout=2.0)
    finally:
        store.close()


def test_gossip_on_empty_corpus_short_circuits_without_calling_swarm(monkeypatch):
    """Sanity check complementing the unit test: when no research has
    run, ``launch_gossip`` completes without invoking the swarm and
    without writing a synthesis row.
    """

    async def _exploding_synthesize(**_kw):
        raise AssertionError("swarm must not be invoked on empty corpus")

    fake_sb = type(sys)("_fake_swarm_bridge")
    fake_sb.gossip_synthesize = _exploding_synthesize
    monkeypatch.setitem(sys.modules, "swarm_bridge", fake_sb)

    store = ConditionStore("")
    try:
        pool = AsyncTaskPool(
            store=store,
            tools=[],
            event_emit=None,
            max_concurrent=2,
            loop=None,
        )
        tid = pool.launch_gossip(iteration=0)
        res = pool.await_tasks([tid], timeout=5.0)[0]
        assert res["status"] == "complete"
        assert "corpus empty" in res["result_summary"]

        # No synthesis row was created.
        synth_count = store.conn.execute(
            "SELECT COUNT(*) FROM conditions WHERE row_type = 'synthesis'"
        ).fetchone()[0]
        assert synth_count == 0

        pool.shutdown(drain_timeout=1.0)
    finally:
        store.close()
