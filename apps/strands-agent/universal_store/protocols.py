"""Shared protocols, dataclasses, and events for the Universal Store Architecture.

NO MODULE MAY IMPORT FROM OTHER universal_store MODULES EXCEPT protocols.py.
This is the single shared contract. Violating this rule creates circular dependencies.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum, auto
from typing import Any, AsyncIterator, Callable, Coroutine, Literal, Protocol, TypeVar

# ---------------------------------------------------------------------------
# Store types
# ---------------------------------------------------------------------------

class RowType(StrEnum):
    RAW = "raw"
    FINDING = "finding"
    THOUGHT = "thought"
    INSIGHT = "insight"
    SYNTHESIS = "synthesis"
    RESEARCH_TARGET = "research_target"
    LESSON = "lesson"
    CONNECTION = "connection"


class QueryType(StrEnum):
    # Foundation (9)
    VALIDATE = "VALIDATE"
    ADJUDICATE = "ADJUDICATE"
    VERIFY = "VERIFY"
    ENRICH = "ENRICH"
    GROUND = "GROUND"
    BRIDGE = "BRIDGE"
    CHALLENGE = "CHALLENGE"
    SYNTHESIZE = "SYNTHESIZE"
    AGGREGATE = "AGGREGATE"
    # Depth (8)
    CAUSAL_TRACE = "CAUSAL_TRACE"
    ASSUMPTION_EXCAVATE = "ASSUMPTION_EXCAVATE"
    EVIDENCE_MAP = "EVIDENCE_MAP"
    SCOPE = "SCOPE"
    METHODOLOGY = "METHODOLOGY"
    TEMPORAL = "TEMPORAL"
    ONTOLOGY = "ONTOLOGY"
    REPLICATION = "REPLICATION"
    # Understandability (4)
    ANALOGY = "ANALOGY"
    TIER_SUMMARIZE = "TIER_SUMMARIZE"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    NARRATIVE_THREAD = "NARRATIVE_THREAD"
    # Meta (4)
    META_PRODUCTIVITY = "META_PRODUCTIVITY"
    META_EXHAUSTION = "META_EXHAUSTION"
    META_COVERAGE = "META_COVERAGE"
    META_EFFECTIVENESS = "META_EFFECTIVENESS"
    # Composite (3)
    DEEP_VALIDATE = "DEEP_VALIDATE"
    RESOLVE_CONTRADICTION = "RESOLVE_CONTRADICTION"
    SYNTHESIS_DEEPEN = "SYNTHESIS_DEEPEN"


class ConnectionType(StrEnum):
    CAUSAL_LINK = "causal_link"
    CONTRADICTORY = "contradictory"
    SUPPORTING = "supporting"
    ANALOGOUS = "analogous"
    GENERALIZING = "generalizing"
    SPECIALIZING = "specializing"
    METHODOLOGICAL = "methodological"
    TEMPORAL = "temporal"
    STATISTICAL = "statistical"
    COMPOSITIONAL = "compositional"


class LessonType(StrEnum):
    STRATEGY = "strategy_lesson"
    SOURCE_QUALITY = "source_quality_lesson"
    MODEL_BEHAVIOR = "model_behavior_lesson"
    COST = "cost_lesson"
    QUERY_EFFICIENCY = "query_efficiency_lesson"
    ANGLE_QUALITY = "angle_quality_lesson"


class OrchestratorPhase(StrEnum):
    IDLE = "IDLE"
    INGESTING = "INGESTING"
    SWARMING = "SWARMING"
    FLOCKING = "FLOCKING"
    SYNTHESIZING = "SYNTHESIZING"
    FETCHING_EXTERNAL = "FETCHING_EXTERNAL"
    USER_INTERRUPTION = "USER_INTERRUPTION"
    CONVERGED = "CONVERGED"
    ERROR = "ERROR"


class OperatorTier(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# ---------------------------------------------------------------------------
# Events — every actor communicates via these
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Event:
    """Base event. All events are immutable and hashable for traceability."""
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_actor: str = ""
    trace_id: str = ""

    def with_source(self, actor_id: str) -> Event:
        return Event(
            event_type=self.event_type,
            payload=self.payload,
            timestamp=self.timestamp,
            source_actor=actor_id,
            trace_id=self.trace_id,
        )


class SwarmPhaseComplete(Event):
    def __init__(self, phase: str, metrics: dict, findings: list[int], **kw):
        super().__init__("SwarmPhaseComplete", {"phase": phase, "metrics": metrics, "findings": findings}, **kw)


class GossipRoundComplete(Event):
    def __init__(self, round_num: int, info_gain: float, gaps_found: int, **kw):
        super().__init__("GossipRoundComplete", {"round": round_num, "info_gain": info_gain, "gaps_found": gaps_found}, **kw)


class FlockRoundComplete(Event):
    def __init__(self, round_num: int, convergence_score: float, directions: list[str], **kw):
        super().__init__("FlockRoundComplete", {"round": round_num, "convergence_score": convergence_score, "directions": directions}, **kw)


class SwarmComplete(Event):
    def __init__(self, findings: list[int], gaps: list[str], **kw):
        super().__init__("SwarmComplete", {"findings": findings, "gaps": gaps}, **kw)


class FlockComplete(Event):
    def __init__(self, convergence_reason: str, directions: list[str], **kw):
        super().__init__("FlockComplete", {"convergence_reason": convergence_reason, "directions": directions}, **kw)


class McpResearchComplete(Event):
    def __init__(self, findings_added: int, cost_usd: float, source_type: str, **kw):
        super().__init__("McpResearchComplete", {"findings_added": findings_added, "cost_usd": cost_usd, "source_type": source_type}, **kw)


class StoreDelta(Event):
    def __init__(self, rows_added: int, row_types: list[str], **kw):
        super().__init__("StoreDelta", {"rows_added": rows_added, "row_types": row_types}, **kw)


class ConvergenceDetected(Event):
    def __init__(self, layer: str, score: float, **kw):
        super().__init__("ConvergenceDetected", {"layer": layer, "score": score}, **kw)


class UserInterrupt(Event):
    def __init__(self, message: str, action: Literal["pause", "stop", "inject"], **kw):
        super().__init__("UserInterrupt", {"message": message, "action": action}, **kw)


class ConnectionDetected(Event):
    def __init__(self, source_id: int, target_id: int, connection_type: str, confidence: float, **kw):
        super().__init__("ConnectionDetected", {"source_id": source_id, "target_id": target_id, "connection_type": connection_type, "confidence": confidence}, **kw)


class LessonRecorded(Event):
    def __init__(self, lesson_id: int, lesson_type: str, fact: str, **kw):
        super().__init__("LessonRecorded", {"lesson_id": lesson_id, "lesson_type": lesson_type, "fact": fact}, **kw)


class CuratorDigest(Event):
    def __init__(self, curator_name: str, digest_type: str, data: dict, **kw):
        super().__init__("CuratorDigest", {"curator_name": curator_name, "digest_type": digest_type, "data": data}, **kw)


class HealthCheck(Event):
    def __init__(self, actor_id: str, status: Literal["healthy", "degraded", "failed"], memory_mb: float, **kw):
        super().__init__("HealthCheck", {"actor_id": actor_id, "status": status, "memory_mb": memory_mb}, **kw)


# ---------------------------------------------------------------------------
# Actor protocol
# ---------------------------------------------------------------------------

class ActorProtocol(Protocol):
    """Every actor in the supervision tree implements this."""

    actor_id: str
    mailbox: asyncio.Queue[Event]

    async def send(self, event: Event) -> None: ...
    async def _run(self) -> None: ...
    async def stop(self, graceful: bool = True) -> None: ...


# ---------------------------------------------------------------------------
# Store protocol
# ---------------------------------------------------------------------------

class StoreProtocol(Protocol):
    """Abstract interface for the universal store."""

    async def execute(self, sql: str, params: tuple | None = None) -> list[dict]: ...
    async def insert(self, table: str, row: dict) -> int: ...
    async def query(self, sql: str, params: tuple | None = None) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Reflexion state
# ---------------------------------------------------------------------------

@dataclass
class ReflexionState:
    exhausted_query_types: set[str] = field(default_factory=set)
    productive_pairs: list[tuple[str, str]] = field(default_factory=list)
    breakthrough_findings: list[int] = field(default_factory=list)
    coverage_score_history: list[float] = field(default_factory=list)
    source_blacklist: set[str] = field(default_factory=set)
    angle_boosts: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# External data models
# ---------------------------------------------------------------------------

@dataclass
class FetchCost:
    usd: float = 0.0
    tokens: int = 0
    latency_s: float = 0.0
    context_window_interest: float = 0.0  # estimated future token cost

    @property
    def total_cost_norm(self) -> float:
        import math
        return math.pow(self.usd * max(self.tokens, 1) * max(self.latency_s, 0.1), 1.0 / 3.0)


@dataclass
class ResearchTarget:
    target_id: str
    source_type: str
    query: str
    reason_type: str
    benefit_score: float
    estimated_cost: FetchCost
    confidence: float = 0.5


@dataclass
class ResearchBudget:
    usd: float
    tokens: int
    time_s: float


@dataclass
class BudgetOverride:
    boost: dict[str, float] = field(default_factory=dict)
    pause: set[str] = field(default_factory=set)
    reason: str = ""


# ---------------------------------------------------------------------------
# Curation models
# ---------------------------------------------------------------------------

@dataclass
class GlobalHealthSnapshot:
    total_rows: int
    finding_count: int
    contradiction_count: int
    gap_count: int
    convergence_trend: list[float]
    top_blockers: list[str]
    recommended_phase: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CloneContextCache:
    angle: str
    context_text: str
    token_count: int
    item_count: int
    invalidated_at: str | None = None


@dataclass
class AngleContextBundle:
    established: list[dict]
    controversial: list[dict]
    bridge_worthy: list[dict]
    recent: list[dict]


@dataclass
class ContradictionDigest:
    contradictions: list[dict]
    stale_count: int
    priority_queue: list[dict]


@dataclass
class GapDigest:
    gaps: list[dict]
    mcp_priority_queue: list[dict]


@dataclass
class OperatorBriefing:
    narrative: str
    alerts: list[str]
    decisions_required: list[str]


# ---------------------------------------------------------------------------
# Orchestrator event stream
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorEvent:
    """Emitted to the user/UI from orch.run(query)."""
    phase: OrchestratorPhase
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = ""


class OrchestratorEventStream(Protocol):
    async def events(self) -> AsyncIterator[OrchestratorEvent]: ...


# ---------------------------------------------------------------------------
# Trace record (single container schema)
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    """Every observable action in the system becomes one of these."""
    trace_id: str
    run_id: str
    actor_id: str
    event_type: str
    phase: str
    payload_json: str
    timestamp: str
    latency_ms: float = 0.0
    error: str = ""
    stack_trace: str = ""
