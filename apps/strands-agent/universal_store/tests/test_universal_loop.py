"""Comprehensive end-to-end tests for the Universal Store Architecture.

Every test is async, mocks all LLM / external dependencies, and asserts that
trace records are written so that "no silent anything" holds in test as well as
production.
"""

from __future__ import annotations

import asyncio
from typing import Any

import duckdb
import pytest

from universal_store.protocols import (
    Event,
    FetchCost,
    FlockComplete,
    FlockRoundComplete,
    McpResearchComplete,
    OperatorTier,
    OrchestratorPhase,
    ResearchBudget,
    ResearchTarget,
    StoreDelta,
    SwarmComplete,
)
from universal_store.schema import get_all_ddl
from universal_store.config import UnifiedConfig
from universal_store.trace import TraceStore
from universal_store.entrypoint import UniversalOrchestrator

from universal_store.actors.orchestrator import OrchestratorActor
from universal_store.actors.swarm import SwarmSupervisor
from universal_store.actors.flock import FlockSupervisor
from universal_store.actors.mcp import McpResearcherActor
from universal_store.actors.reflexion import LessonStore, ReflexionActor
from universal_store.actors.semantic import SemanticConnectionWorker
from universal_store.actors.curation import CloneContextCurator
from universal_store.scheduler import Scheduler
from universal_store.rules.engine import RuleContext, RuleEngine, RuleFired
from universal_store.corpus.battery import CorpusAlgorithmBattery

from universal_store.tests.fixtures import (
    MinimalDuckDBStore,
    MockActor,
    _strip_include_clauses,
    populated_store,
    minimal_config,
    sample_findings,
    sample_raw_rows,
    temp_duckdb,
    temp_trace_db,
    trace_store,
    reset_trace_store,
)


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_creation(
    temp_duckdb: str,
    trace_store: TraceStore,
) -> None:
    """Verify all tables and indices from ``get_all_ddl`` exist in a fresh database."""
    conn = duckdb.connect(temp_duckdb)

    # Base table must exist before ALTER statements run
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conditions (
            id INTEGER PRIMARY KEY,
            fact TEXT NOT NULL,
            source_url TEXT DEFAULT '',
            source_type TEXT DEFAULT '',
            source_ref TEXT DEFAULT '',
            row_type TEXT DEFAULT 'finding',
            parent_id INTEGER,
            related_id INTEGER,
            consider_for_use BOOLEAN DEFAULT TRUE,
            obsolete_reason TEXT DEFAULT '',
            angle TEXT DEFAULT '',
            strategy TEXT DEFAULT '',
            expansion_depth INTEGER DEFAULT 0,
            created_at TEXT DEFAULT '',
            iteration INTEGER DEFAULT 0,
            confidence FLOAT DEFAULT 0.5,
            trust_score FLOAT DEFAULT 0.5,
            novelty_score FLOAT DEFAULT 0.5,
            specificity_score FLOAT DEFAULT 0.5,
            relevance_score FLOAT DEFAULT 0.5,
            actionability_score FLOAT DEFAULT 0.5,
            duplication_score FLOAT DEFAULT -1.0,
            fabrication_risk FLOAT DEFAULT 0.0,
            verification_status TEXT DEFAULT '',
            scored_at TEXT DEFAULT '',
            score_version INTEGER DEFAULT 0,
            composite_quality FLOAT DEFAULT -1.0,
            information_density FLOAT DEFAULT -1.0,
            cross_ref_boost FLOAT DEFAULT 0.0,
            processing_status TEXT DEFAULT 'raw',
            expansion_tool TEXT DEFAULT 'none',
            expansion_hint TEXT DEFAULT '',
            expansion_fulfilled BOOLEAN DEFAULT FALSE,
            expansion_gap TEXT DEFAULT '',
            expansion_priority FLOAT DEFAULT 0.0,
            cluster_id INTEGER DEFAULT -1,
            cluster_rank INTEGER DEFAULT 0,
            contradiction_flag BOOLEAN DEFAULT FALSE,
            contradiction_partner INTEGER DEFAULT -1,
            staleness_penalty FLOAT DEFAULT 0.0,
            relationship_score FLOAT DEFAULT 0.0,
            phase TEXT DEFAULT '',
            parent_ids TEXT DEFAULT '',
            source_model TEXT DEFAULT '',
            source_run TEXT DEFAULT '',
            evaluation_count INTEGER DEFAULT 0,
            last_evaluated_at TEXT DEFAULT '',
            evaluator_angles TEXT DEFAULT '',
            mcp_research_status TEXT DEFAULT '',
            information_gain FLOAT DEFAULT 0.0
        )
        """
    )
    conn.execute(_strip_include_clauses(get_all_ddl()))
    conn.commit()

    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchdf()["table_name"].tolist()

    expected_tables = {
        "conditions",
        "runs",
        "score_history",
        "lessons",
        "lesson_applications",
        "semantic_connections",
        "source_fingerprints",
        "chunks",
        "source_utility_log",
        "source_quality_registry",
        "condition_sources",
        "condition_embeddings",
        "trace_records",
    }
    assert expected_tables.issubset(set(tables)), (
        f"Missing tables: {expected_tables - set(tables)}"
    )

    # DuckDB exposes indexes via duckdb_indexes()
    indices = conn.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE schema_name = 'main'"
    ).fetchdf()["index_name"].tolist()
    assert len(indices) > 0, "Expected at least one index to be created"

    conn.close()

    await trace_store.record(
        actor_id="test_schema_creation",
        event_type="schema_verified",
        payload={"tables": list(expected_tables), "index_count": len(indices)},
    )
    await trace_store._flush()
    records = await trace_store.query(event_type="schema_verified")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 2. Trace store records
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_store_records(trace_store: TraceStore) -> None:
    """Verify traces are written and queryable."""
    run_id = "trace-test-run"
    await trace_store.set_run(run_id)
    await trace_store.record(
        actor_id="test_actor",
        event_type="test_event",
        phase="test_phase",
        payload={"key": "value"},
        latency_ms=12.5,
    )
    await trace_store._flush()

    records = await trace_store.query(run_id=run_id, event_type="test_event")
    assert len(records) >= 1
    assert records[0]["actor_id"] == "test_actor"
    assert records[0]["phase"] == "test_phase"
    assert records[0]["latency_ms"] == pytest.approx(12.5)

    stats = await trace_store.get_stats(run_id)
    assert stats["total_events"] >= 1
    assert stats["actor_count"] >= 1


# ---------------------------------------------------------------------------
# 3. Orchestrator lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_lifecycle(
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify OrchestratorActor starts, phases transition, and stops cleanly."""
    orch = OrchestratorActor(actor_id="orchestrator_test", config=minimal_config)
    orch.start()
    await asyncio.sleep(0.1)

    health = await orch.health()
    assert health["running"] is True
    assert orch.phase == OrchestratorPhase.IDLE

    # IDLE → SWARMING (new raw data)
    await orch.send(StoreDelta(rows_added=1, row_types=["raw"]))
    await asyncio.sleep(0.2)
    assert orch.phase == OrchestratorPhase.SWARMING

    # SWARMING → FLOCKING
    await orch.send(SwarmComplete(findings=[1, 2], gaps=[]))
    await asyncio.sleep(0.2)
    assert orch.phase == OrchestratorPhase.FLOCKING

    # FLOCKING → FETCHING_EXTERNAL (directions present)
    await orch.send(FlockComplete(convergence_reason="test", directions=["d1"]))
    await asyncio.sleep(0.2)
    assert orch.phase == OrchestratorPhase.FETCHING_EXTERNAL

    # FETCHING_EXTERNAL → SWARMING (external research done)
    await orch.send(
        McpResearchComplete(findings_added=1, cost_usd=0.01, source_type="brave")
    )
    await asyncio.sleep(0.2)
    assert orch.phase == OrchestratorPhase.SWARMING

    # SWARMING → CONVERGED (no gaps remain)
    await orch.send(Event("ConvergenceDetected", {"layer": "test", "score": 0.001}))
    await asyncio.sleep(0.2)
    assert orch.phase == OrchestratorPhase.CONVERGED

    await orch.stop(graceful=True)
    await asyncio.sleep(0.1)
    health = await orch.health()
    assert health["running"] is False

    await trace_store._flush()
    records = await trace_store.query(actor_id="orchestrator_test")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 4. Swarm writes finding rows
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_swarm_writes_finding_rows(
    temp_duckdb: str,
    sample_raw_rows: list[dict],
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """CRITICAL: verify swarm extraction writes ``row_type='finding'``, not ``thought``."""
    store = MinimalDuckDBStore(temp_duckdb)
    raw_ids: list[int] = []
    for row in sample_raw_rows:
        rid = store.admit(**{k: v for k, v in row.items() if k != "id"})
        assert rid is not None
        raw_ids.append(rid)

    swarm = SwarmSupervisor(actor_id="swarm_test", store=store, config=minimal_config)
    swarm.start()
    await swarm.start_extraction(raw_ids, angles=["angle_a", "angle_b"])
    await asyncio.sleep(0.3)

    findings = store.conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE row_type = 'finding'"
    ).fetchone()[0]
    thoughts = store.conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE row_type = 'thought'"
    ).fetchone()[0]

    assert findings > 0, "Swarm must produce finding rows"
    assert thoughts == 0, "Swarm must NOT produce thought rows during extraction"

    await swarm.stop(graceful=True)
    store.close()

    await trace_store._flush()
    records = await trace_store.query(actor_id="swarm_test")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 5. Cluster id populated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cluster_id_populated(
    temp_duckdb: str,
    sample_findings: list[dict],
    trace_store: TraceStore,
) -> None:
    """Verify clustering populates ``cluster_id`` on condition rows."""
    store = MinimalDuckDBStore(temp_duckdb)
    for finding in sample_findings:
        store.admit(**{k: v for k, v in finding.items() if k != "id"})

    rows = store.get_findings(limit=500)
    clusters = await CorpusAlgorithmBattery.union_find_clustering(rows)
    assert len(clusters) > 0, "Union-find should produce at least one cluster mapping"

    for fid, cid in clusters.items():
        store.conn.execute(
            "UPDATE conditions SET cluster_id = ? WHERE id = ?", [cid, fid]
        )

    clustered = store.conn.execute(
        "SELECT DISTINCT cluster_id FROM conditions WHERE cluster_id > -1"
    ).fetchall()
    assert len(clustered) > 0, "cluster_id must be populated in the store"

    store.close()

    await trace_store._flush()
    records = await trace_store.query(event_type="union_find_clustering")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 6. Contradiction flag populated
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contradiction_flag_populated(
    temp_duckdb: str,
    trace_store: TraceStore,
) -> None:
    """Verify contradiction detection populates the ``contradiction_flag``."""
    store = MinimalDuckDBStore(temp_duckdb)
    store.admit(
        fact="Drug X cures disease Y with 95% efficacy in clinical trials.",
        confidence=0.95,
        row_type="finding",
        angle="medicine",
        source_type="clinical_trial",
    )
    store.admit(
        fact="Drug X shows no significant benefit over placebo for disease Y.",
        confidence=0.35,
        row_type="finding",
        angle="medicine",
        source_type="meta_analysis",
    )

    rows = store.get_findings(limit=500)
    contradictions = await CorpusAlgorithmBattery.detect_contradictions(rows)
    assert len(contradictions) > 0, "Should detect a contradiction pair"

    for a, b, _severity in contradictions:
        store.conn.execute(
            "UPDATE conditions SET contradiction_flag = TRUE WHERE id IN (?, ?)",
            [a, b],
        )

    flagged = store.conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE contradiction_flag = TRUE"
    ).fetchone()[0]
    assert flagged >= 2, "contradiction_flag must be populated"

    store.close()

    await trace_store._flush()
    records = await trace_store.query(event_type="detect_contradictions")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 7. Flock convergence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flock_convergence(
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify flock detects convergence and stops after 2 consecutive rounds."""
    flock = FlockSupervisor(actor_id="flock_test", config=minimal_config)
    flock.start()
    flock.start_flock(condition_ids=[1, 2], angles=["angle_a", "angle_b"])

    for round_num in range(3):
        for angle in ("angle_a", "angle_b"):
            event = FlockRoundComplete(
                round_num=round_num,
                convergence_score=0.001,
                directions=[angle],
            )
            await flock.send(event)
        await asyncio.sleep(0.05)

    assert flock._consecutive_converged >= 2, "Flock should have converged"
    health = await flock.health()
    assert health.get("consecutive_converged", 0) >= 2

    await flock.stop(graceful=True)

    await trace_store._flush()
    records = await trace_store.query(actor_id="flock_test")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 8. MCP benefit scoring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_benefit_scoring(
    temp_duckdb: str,
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify benefit scores are computed before fetch (cost estimation)."""
    store = MinimalDuckDBStore(temp_duckdb)
    mcp = McpResearcherActor(actor_id="mcp_benefit", store=store, config=minimal_config)

    gaps = [
        {
            "gap_id": "g1",
            "coverage_gap": 0.9,
            "recency_need": 0.8,
            "authority_need": 0.7,
            "preferred_source": "brave",
            "query": "quantum computing error rates",
            "reason_type": "coverage_gap",
            "confidence": 0.8,
        },
    ]

    # Direct unit check: benefit is computed while cost is still default
    targets = mcp.evaluate_benefit(gaps)
    assert len(targets) == len(gaps)
    assert all(t.benefit_score > 0 for t in targets)
    assert all(t.estimated_cost.total_cost_norm == 0.0 for t in targets), (
        "Benefit must be computed BEFORE cost estimation"
    )

    # Full flow trace ordering check
    mcp.start()
    await mcp.send(
        Event(
            "ResearchNeeded",
            {
                "gaps": gaps,
                "budget": {"usd": 5.0, "tokens": 10000, "time_s": 60.0},
            },
        )
    )
    await asyncio.sleep(0.3)
    await mcp.stop(graceful=True)

    await trace_store._flush()
    records = await trace_store.query(actor_id="mcp_benefit")
    event_types = [r["event_type"] for r in records]
    assert "benefit_evaluation_done" in event_types
    assert "cost_estimate" in event_types
    benefit_idx = event_types.index("benefit_evaluation_done")
    cost_idx = event_types.index("cost_estimate")
    assert benefit_idx < cost_idx, "Benefit evaluation must precede cost estimation"

    store.close()


# ---------------------------------------------------------------------------
# 9. Operator override red
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_operator_override_red(
    temp_duckdb: str,
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify red-tier targets halt for operator approval."""
    store = MinimalDuckDBStore(temp_duckdb)
    mcp = McpResearcherActor(actor_id="mcp_red", store=store, config=minimal_config)

    # Direct tier classification
    target = ResearchTarget(
        target_id="red_target",
        source_type="exa",
        query="expensive query",
        reason_type="depth_need",
        benefit_score=0.9,
        estimated_cost=FetchCost(usd=5.0, tokens=10000, latency_s=20.0),
    )
    tier = mcp.operator_override_check(target)
    assert tier == OperatorTier.RED, f"Expected RED tier, got {tier}"

    # Full flow: patch cost so the gap triggers red, then unblock
    mcp.estimate_cost = lambda _target: FetchCost(  # type: ignore[method-assign]
        usd=5.0, tokens=10000, latency_s=20.0
    )
    mcp.start()

    async def unblock() -> None:
        await asyncio.sleep(0.05)
        await mcp._operator_events.put(
            Event("OperatorDecision", {"target_id": "gap_red", "decision": "proceed"})
        )

    asyncio.create_task(unblock())
    await mcp.send(
        Event(
            "ResearchNeeded",
            {
                "gaps": [
                    {
                        "gap_id": "gap_red",
                        "coverage_gap": 0.9,
                        "preferred_source": "exa",
                        "query": "expensive query",
                        "reason_type": "depth_need",
                        "confidence": 0.9,
                    }
                ],
                "budget": {"usd": 10.0, "tokens": 20000, "time_s": 60.0},
            },
        )
    )
    await asyncio.sleep(0.3)

    await trace_store._flush()
    records = await trace_store.query(actor_id="mcp_red")
    event_types = {r["event_type"] for r in records}
    assert "operator_override_red" in event_types
    assert "red_tier_halt" in event_types

    await mcp.stop(graceful=True)
    store.close()


# ---------------------------------------------------------------------------
# 10. Reflexion lessons persisted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflexion_lessons_persisted(
    temp_duckdb: str,
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify lessons survive round completion (persisted to store)."""
    lesson_store = LessonStore(db_path=temp_duckdb)

    lesson_id = await lesson_store.record(
        {
            "lesson_type": "strategy_lesson",
            "fact": "Bridge queries waste exceeds 15% when angle diversity < 3.",
            "run_id": "run-123",
            "run_number": 1,
            "angle": "strategy",
            "query_type": "BRIDGE",
            "source_url": "",
            "source_type": "swarm",
            "relevance_score": 0.8,
            "confidence": 0.75,
            "metadata": {"waste_rate": 0.18},
            "halflife_runs": 3,
        }
    )
    assert lesson_id is not None and lesson_id > 0

    lessons = await lesson_store.query(run_id="run-123", min_confidence=0.7)
    assert any(l["id"] == lesson_id for l in lessons), (
        "Lesson must survive round completion"
    )

    # Also exercise the ReflexionActor event path
    reflexion = ReflexionActor(actor_id="reflexion_test", config=minimal_config)
    reflexion.start()
    await reflexion.send(
        SwarmComplete(findings=[1, 2], gaps=["low_info_gain"], run_id="run-123")
    )
    await asyncio.sleep(0.3)
    await reflexion.stop(graceful=True)

    await trace_store._flush()
    records = await trace_store.query(actor_id="reflexion_test")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 11. Semantic connection pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semantic_connection_pipeline(
    temp_duckdb: str,
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify the 3-stage pipeline (heuristic → embedding → LLM) produces connections."""
    store = MinimalDuckDBStore(temp_duckdb)

    # Seed findings that share terms / cluster to pass heuristic filter
    for i in range(4):
        store.admit(
            fact=(
                f"Quantum computing requires error correction "
                f"with surface codes iteration {i}."
            ),
            row_type="finding",
            angle="physics",
            confidence=0.8,
            source_type="paper",
            cluster_id=1,
        )

    # Dummy embeddings (identical vectors => cosine similarity = 1.0)
    for fid in range(1, 5):
        store.conn.execute(
            "INSERT INTO condition_embeddings (condition_id, embedding) VALUES (?, ?)",
            [fid, [0.1] * 768],
        )

    worker = SemanticConnectionWorker(
        actor_id="semantic_test", store=store, config=minimal_config
    )
    worker.start()
    await worker.send(StoreDelta(rows_added=4, row_types=["finding"]))
    await asyncio.sleep(0.4)

    rows = store.conn.execute("SELECT * FROM semantic_connections").fetchall()
    assert len(rows) > 0, "3-stage pipeline must produce semantic_connections rows"

    await worker.stop(graceful=True)
    store.close()

    await trace_store._flush()
    records = await trace_store.query(actor_id="semantic_test")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 12. Curation prevents overflow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_curation_prevents_overflow(
    temp_duckdb: str,
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify CloneContextCurator caps context items."""
    store = MinimalDuckDBStore(temp_duckdb)
    for i in range(50):
        store.admit(
            fact=f"Fact {i} for capped context.",
            row_type="finding",
            angle="test_angle",
            confidence=0.6,
        )

    curator = CloneContextCurator(
        actor_id="curator_test", store=store, config=minimal_config
    )
    curator.start()
    await curator.send(Event("BuildCloneContext", {"angle": "test_angle"}))
    await asyncio.sleep(0.3)

    cache = curator._cache.get("test_angle")
    assert cache is not None, "Curator should build a cache entry"
    # SQL LIMIT is max_items * 2; displayed buckets are further capped
    assert cache.item_count <= minimal_config.curation.clone_context_max_items * 2

    await curator.stop(graceful=True)
    store.close()

    await trace_store._flush()
    records = await trace_store.query(actor_id="curator_test")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 13. Scheduler routes events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scheduler_routes_events(
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify the scheduler priority queue routes events to registered actors."""
    scheduler = Scheduler(config=minimal_config)
    mock = MockActor("mock_target")
    mock.start()

    scheduler.register_actor("mock_target", mock)
    scheduler.register_route("TestEvent", "mock_target")
    await scheduler.start()
    await asyncio.sleep(0.1)

    event = Event("TestEvent", {"payload": 42})
    await scheduler.submit(event, priority=0)
    await asyncio.sleep(0.3)

    assert any(r.event_type == "TestEvent" for r in mock.received), (
        "Scheduler must route TestEvent to mock_target"
    )

    await scheduler.stop()
    await mock.stop()

    await trace_store._flush()
    records = await trace_store.query(actor_id="scheduler")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 14. Rule engine fires
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rule_engine_fires(
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Verify rules match and produce RuleFired events."""
    engine = RuleEngine.with_default_rules()

    ctx = RuleContext(
        current_phase=OrchestratorPhase.SWARMING,
        destructive_ops_pending=[
            {"op_type": "delete", "target": "all", "requires_confirmation": True}
        ],
        cost_accumulated_usd=100.0,
        budget=ResearchBudget(usd=1.0, tokens=100, time_s=10.0),
        high_value_findings=[{"finding_id": 1, "value_score": 0.95}],
        actor_crashes=[{"actor_id": "a1", "error": "boom"}],
    )

    events = await engine.evaluate(ctx)
    fired_types = {e.event_type for e in events}
    assert "RuleFired" in fired_types, "At least one rule must fire"

    # Verify trace
    await trace_store._flush()
    records = await trace_store.query(actor_id="rule_engine")
    assert len(records) > 0


# ---------------------------------------------------------------------------
# 15. End-to-end swarm→flock→external→reswarm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_end_to_end_swarm_flock_loop(
    temp_duckdb: str,
    sample_raw_rows: list[dict],
    minimal_config: UnifiedConfig,
    trace_store: TraceStore,
) -> None:
    """Full swarm→flock→external→reswarm cycle, assert store state."""
    store = MinimalDuckDBStore(temp_duckdb)

    # 1. Ingest raw rows
    raw_ids: list[int] = []
    for row in sample_raw_rows:
        rid = store.admit(**{k: v for k, v in row.items() if k != "id"})
        assert rid is not None
        raw_ids.append(rid)

    # 2. Swarm extraction
    swarm = SwarmSupervisor(actor_id="e2e_swarm", store=store, config=minimal_config)
    swarm.start()
    await swarm.start_extraction(raw_ids, angles=["physics", "engineering"])
    await asyncio.sleep(0.3)

    finding_count_after_swarm = store.conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE row_type = 'finding'"
    ).fetchone()[0]
    assert finding_count_after_swarm > 0, "Swarm must produce findings"

    # 3. Flock convergence
    flock = FlockSupervisor(actor_id="e2e_flock", config=minimal_config)
    flock.start()
    flock.start_flock(
        condition_ids=list(range(1, finding_count_after_swarm + 1)),
        angles=["physics", "engineering"],
    )
    for round_num in range(3):
        for angle in ("physics", "engineering"):
            await flock.send(
                FlockRoundComplete(
                    round_num=round_num,
                    convergence_score=0.001,
                    directions=[angle],
                )
            )
        await asyncio.sleep(0.05)
    assert flock._consecutive_converged >= 2, "Flock must converge"
    await flock.stop(graceful=True)

    # 4. External MCP research (mock)
    mcp = McpResearcherActor(actor_id="e2e_mcp", store=store, config=minimal_config)
    mcp.start()

    async def unblock_mcp() -> None:
        await asyncio.sleep(0.05)
        await mcp._operator_events.put(
            Event("OperatorDecision", {"target_id": "gap1", "decision": "proceed"})
        )

    asyncio.create_task(unblock_mcp())
    await mcp.send(
        Event(
            "ResearchNeeded",
            {
                "gaps": [
                    {
                        "gap_id": "gap1",
                        "coverage_gap": 0.9,
                        "preferred_source": "brave",
                        "query": "quantum error correction 2024",
                        "reason_type": "coverage_gap",
                        "confidence": 0.8,
                    }
                ],
                "budget": {"usd": 5.0, "tokens": 10000, "time_s": 60.0},
            },
        )
    )
    await asyncio.sleep(0.3)
    await mcp.stop(graceful=True)

    # 5. Re-swarm with new findings
    await swarm.start_extraction(raw_ids, angles=["physics", "engineering", "cs"])
    await asyncio.sleep(0.3)

    final_finding_count = store.conn.execute(
        "SELECT COUNT(*) FROM conditions WHERE row_type = 'finding'"
    ).fetchone()[0]
    assert final_finding_count > finding_count_after_swarm, (
        "Re-swarm must add more findings after external research"
    )

    # 6. Trace verification
    await trace_store._flush()
    all_records = await trace_store.query()
    actor_counts: dict[str, int] = {}
    for rec in all_records:
        aid = rec["actor_id"]
        actor_counts[aid] = actor_counts.get(aid, 0) + 1

    assert any("e2e_swarm" in a for a in actor_counts), "Trace must include swarm"
    assert any("e2e_flock" in a for a in actor_counts), "Trace must include flock"
    assert any("e2e_mcp" in a for a in actor_counts), "Trace must include mcp"

    await swarm.stop(graceful=True)
    store.close()
