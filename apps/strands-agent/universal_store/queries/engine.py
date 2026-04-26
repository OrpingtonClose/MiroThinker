"""Query Wealth Engine — state-driven selection of the 28-query analytical battery.

Pure selection logic. No query execution. No actor imports.
Every selection, allocation, and composite build is traced via TraceStore.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any

from universal_store.protocols import BudgetOverride, QueryType, ReflexionState
from universal_store.trace import TraceStore
from universal_store.config import UnifiedConfig


# ---------------------------------------------------------------------------
# Query type metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QueryTypeDef:
    """Metadata for a single query type in the 28-type battery.

    Attributes:
        name: The QueryType enum member name.
        purpose: Human-readable description of what this query does.
        cost_weight: Relative cost multiplier (1.0 = baseline).
        depth_category: One of foundation, depth, understandability, meta, composite.
        applicable_when: Predicate describing when this type is relevant.
    """

    name: str
    purpose: str
    cost_weight: float
    depth_category: str
    applicable_when: str


# Complete 28-type catalog aligned with QueryType enum.
QUERY_TYPE_DEFINITIONS: tuple[QueryTypeDef, ...] = (
    # Foundation (9)
    QueryTypeDef("VALIDATE", "Confirm novelty claims under uncertainty", 1.0, "foundation", "novelty > 0.6 AND confidence < 0.4"),
    QueryTypeDef("ADJUDICATE", "Resolve contradictions between findings", 1.1, "foundation", "contradiction_flag = TRUE"),
    QueryTypeDef("VERIFY", "Check fabrication risk against ground truth", 1.0, "foundation", "fabrication_risk > 0.4"),
    QueryTypeDef("ENRICH", "Add concrete precision to vague findings", 1.0, "foundation", "specificity < 0.4 AND relevance > 0.5"),
    QueryTypeDef("GROUND", "Anchor actionable claims in evidence", 1.0, "foundation", "actionability > 0.6 AND verification_status IS NULL"),
    QueryTypeDef("BRIDGE", "Connect findings across angles/clusters", 1.2, "foundation", "cross-angle or cross-cluster membership"),
    QueryTypeDef("CHALLENGE", "Stress-test high-confidence findings", 1.1, "foundation", "confidence > 0.8"),
    QueryTypeDef("SYNTHESIZE", "Produce higher-order insights from clusters", 1.3, "foundation", "cluster with 3+ members"),
    QueryTypeDef("AGGREGATE", "Strategic research planning", 1.0, "foundation", "once per round after clones"),
    # Depth (8)
    QueryTypeDef("CAUSAL_TRACE", "Trace causal chains between findings", 1.4, "depth", "cross-angle causal candidates"),
    QueryTypeDef("ASSUMPTION_EXCAVATE", "Surface hidden assumptions", 1.3, "depth", "any validated finding"),
    QueryTypeDef("EVIDENCE_MAP", "Map evidentiary structure", 1.2, "depth", "findings with overlapping sources"),
    QueryTypeDef("SCOPE", "Evaluate generalizability beyond domain", 1.2, "depth", "specificity > 0.7 AND relevance > 0.7"),
    QueryTypeDef("METHODOLOGY", "Critique methodological basis", 1.2, "depth", "high novelty, low specificity"),
    QueryTypeDef("TEMPORAL", "Detect superseded findings", 1.2, "depth", "older findings with newer conflicting evidence"),
    QueryTypeDef("ONTOLOGY", "Classify outlier breakthrough vs error", 1.3, "depth", "high-deviation cluster outlier"),
    QueryTypeDef("REPLICATION", "Assess independent replication", 1.1, "depth", "single-source high-confidence finding"),
    # Understandability (4)
    QueryTypeDef("ANALOGY", "Generate structural analogies", 1.0, "understandability", "cross-domain structural similarity"),
    QueryTypeDef("TIER_SUMMARIZE", "Summarize for audience tier", 0.8, "understandability", "dense cluster or synthesis"),
    QueryTypeDef("COUNTERFACTUAL", "Probe counterfactual implications", 1.2, "understandability", "high-confidence causal claim"),
    QueryTypeDef("NARRATIVE_THREAD", "Weave findings into narrative", 1.0, "understandability", "synthesis or cluster ready for reporting"),
    # Meta (4)
    QueryTypeDef("META_PRODUCTIVITY", "Analyze per-type yield", 0.6, "meta", "round completion"),
    QueryTypeDef("META_EXHAUSTION", "Detect exhausted query types", 0.6, "meta", "round completion"),
    QueryTypeDef("META_COVERAGE", "Audit coverage gaps", 0.6, "meta", "information gain rate dropping"),
    QueryTypeDef("META_EFFECTIVENESS", "Measure round effectiveness", 0.6, "meta", "round completion"),
    # Composite (3)
    QueryTypeDef("DEEP_VALIDATE", "Thorough validation: validate + excavate + map", 2.5, "composite", "high-stakes novel claim"),
    QueryTypeDef("RESOLVE_CONTRADICTION", "Deep contradiction resolution", 2.8, "composite", "complex multi-angle contradiction"),
    QueryTypeDef("SYNTHESIS_DEEPEN", "Deep synthesis with causal probing", 2.6, "composite", "prior synthesis exists, deepen"),
)

# Lookup helpers
_QUERY_DEF_BY_NAME: dict[str, QueryTypeDef] = {d.name: d for d in QUERY_TYPE_DEFINITIONS}
_QUERY_DEF_BY_TYPE: dict[QueryType, QueryTypeDef] = {
    QueryType(d.name): d for d in QUERY_TYPE_DEFINITIONS
}

# Successor map for exhausted-type promotion.
_EXHAUSTED_SUCCESSOR: dict[QueryType, QueryType] = {
    QueryType.VALIDATE: QueryType.DEEP_VALIDATE,
    QueryType.ADJUDICATE: QueryType.RESOLVE_CONTRADICTION,
    QueryType.SYNTHESIZE: QueryType.SYNTHESIS_DEEPEN,
    QueryType.VERIFY: QueryType.REPLICATION,
    QueryType.CHALLENGE: QueryType.COUNTERFACTUAL,
    QueryType.BRIDGE: QueryType.CAUSAL_TRACE,
    QueryType.ENRICH: QueryType.EVIDENCE_MAP,
    QueryType.GROUND: QueryType.EVIDENCE_MAP,
    QueryType.AGGREGATE: QueryType.META_COVERAGE,
}


# ---------------------------------------------------------------------------
# Tracing helper
# ---------------------------------------------------------------------------

async def _trace(
    actor_id: str,
    event_type: str,
    phase: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget trace record; never raise."""
    try:
        store = await TraceStore.get()
        await store.record(
            actor_id=actor_id,
            event_type=event_type,
            phase=phase,
            payload=payload or {},
        )
    except Exception:
        # Trace failures must not break selection logic.
        pass


# ---------------------------------------------------------------------------
# QuerySelector
# ---------------------------------------------------------------------------

class QuerySelector:
    """State-driven selector for the 28-query analytical battery.

    Implements the decision tree from CONVERGED_DEPTH_ARCHITECTURE.md §4.2
    with reflexion-informed overrides from §4.3.
    """

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig.from_env()
        self._actor_id = "QuerySelector"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def select(
        self,
        store_state: dict[str, Any],
        reflexion: ReflexionState,
        budget: dict[str, Any],
    ) -> list[QueryType]:
        """Select query types for the current round.

        Args:
            store_state: Snapshot of store topology. Expected keys mirror
                the decision-tree predicates (e.g. ``high_novelty_low_confidence``,
                ``contradiction_flags``, ``clusters_with_3_plus``, etc.).
            reflexion: Mutable reflexion state updated each round.
            budget: Budget descriptor. May contain ``max_queries``,
                ``budget_override`` (BudgetOverride), ``round_number``,
                ``new_mcp_findings``, etc.

        Returns:
            Ordered list of QueryType values to execute this round.
        """
        phase = "selection"
        await _trace(
            self._actor_id,
            "query_selector_start",
            phase,
            payload={
                "store_state_keys": list(store_state.keys()),
                "exhausted_types": sorted(reflexion.exhausted_query_types),
                "productive_pairs": reflexion.productive_pairs,
                "breakthrough_count": len(reflexion.breakthrough_findings),
                "coverage_history_length": len(reflexion.coverage_score_history),
            },
        )

        # 1. Base scores for all 28 types
        scores: dict[QueryType, float] = {
            QueryType(d.name): 1.0 for d in QUERY_TYPE_DEFINITIONS
        }

        # 2. State-driven decision tree (§4.2)
        self._apply_decision_tree(scores, store_state)

        # 3. Reflexion-informed overrides (§4.3)
        self._apply_exhaustion_rules(scores, reflexion)
        self._apply_productive_pair_rules(scores, reflexion)
        self._apply_breakthrough_boost(scores, reflexion)
        self._apply_coverage_plateau_boost(scores, reflexion)

        # 4. External-data-driven injection (§4.5)
        self._apply_mcp_injection_rules(scores, store_state, budget)

        # 5. BudgetOverride boost / pause
        override = budget.get("budget_override")
        if isinstance(override, BudgetOverride):
            for qt_name, boost in override.boost.items():
                try:
                    qt = QueryType(qt_name)
                    scores[qt] = scores.get(qt, 0.0) * boost
                except ValueError:
                    continue
            for qt_name in override.pause:
                try:
                    qt = QueryType(qt_name)
                    scores[qt] = 0.0
                except ValueError:
                    continue
            await _trace(
                self._actor_id,
                "budget_override_applied",
                phase,
                payload={
                    "boosts": override.boost,
                    "pauses": sorted(override.pause),
                    "reason": override.reason,
                },
            )

        # 6. Drop zero / negative scores and sort descending
        positive = {qt: sc for qt, sc in scores.items() if sc > 0.0}
        if not positive:
            # Fallback: at least run META_COVERAGE to diagnose emptiness.
            positive = {QueryType.META_COVERAGE: 1.0}

        ordered = sorted(positive.items(), key=lambda x: x[1], reverse=True)

        # 7. Respect max_queries ceiling if provided
        max_queries = budget.get("max_queries")
        if isinstance(max_queries, int) and max_queries > 0:
            ordered = ordered[:max_queries]

        result = [qt for qt, _ in ordered]

        await _trace(
            self._actor_id,
            "query_selector_end",
            phase,
            payload={
                "selected": [qt.value for qt in result],
                "scores": {qt.value: round(sc, 4) for qt, sc in ordered},
            },
        )
        return result

    # ------------------------------------------------------------------
    # Decision tree (§4.2)
    # ------------------------------------------------------------------

    def _apply_decision_tree(
        self,
        scores: dict[QueryType, float],
        state: dict[str, Any],
    ) -> None:
        """Apply the state-driven decision tree to modify *scores* in-place."""

        # High novelty + low confidence → VALIDATE / DEEP_VALIDATE
        if state.get("high_novelty_low_confidence", 0) > 0:
            scores[QueryType.VALIDATE] = scores.get(QueryType.VALIDATE, 0.0) + 2.5
            scores[QueryType.DEEP_VALIDATE] = scores.get(QueryType.DEEP_VALIDATE, 0.0) + 1.5

        # Contradiction flags → ADJUDICATE / RESOLVE_CONTRADICTION
        contradiction_flags = state.get("contradiction_flags", 0)
        if contradiction_flags > 0:
            if contradiction_flags == 1:
                scores[QueryType.ADJUDICATE] = scores.get(QueryType.ADJUDICATE, 0.0) + 2.0
            else:
                scores[QueryType.RESOLVE_CONTRADICTION] = scores.get(QueryType.RESOLVE_CONTRADICTION, 0.0) + 2.5

        # High fabrication risk → VERIFY / REPLICATION
        if state.get("high_fabrication_risk", 0) > 0:
            scores[QueryType.VERIFY] = scores.get(QueryType.VERIFY, 0.0) + 1.5
            scores[QueryType.REPLICATION] = scores.get(QueryType.REPLICATION, 0.0) + 1.0

        # Low specificity + high relevance → ENRICH / EVIDENCE_MAP
        if state.get("low_specificity_high_relevance", 0) > 0:
            scores[QueryType.ENRICH] = scores.get(QueryType.ENRICH, 0.0) + 1.0
            scores[QueryType.EVIDENCE_MAP] = scores.get(QueryType.EVIDENCE_MAP, 0.0) + 0.5

        # High actionability + unverified → GROUND / EVIDENCE_MAP
        if state.get("high_actionability_unverified", 0) > 0:
            scores[QueryType.GROUND] = scores.get(QueryType.GROUND, 0.0) + 1.0
            scores[QueryType.EVIDENCE_MAP] = scores.get(QueryType.EVIDENCE_MAP, 0.0) + 0.5

        # Clusters with 3+ members → SYNTHESIZE / SYNTHESIS_DEEPEN
        if state.get("clusters_with_3_plus", 0) > 0:
            if state.get("prior_synthesis_exists"):
                scores[QueryType.SYNTHESIS_DEEPEN] = scores.get(QueryType.SYNTHESIS_DEEPEN, 0.0) + 2.0
            else:
                scores[QueryType.SYNTHESIZE] = scores.get(QueryType.SYNTHESIZE, 0.0) + 2.0

        # High-confidence single-angle findings → CHALLENGE / COUNTERFACTUAL
        if state.get("high_confidence_single_angle", 0) > 0:
            scores[QueryType.CHALLENGE] = scores.get(QueryType.CHALLENGE, 0.0) + 1.0
            scores[QueryType.COUNTERFACTUAL] = scores.get(QueryType.COUNTERFACTUAL, 0.0) + 0.5

        # Cross-angle cluster members → BRIDGE / CAUSAL_TRACE
        if state.get("cross_angle_cluster_members", 0) > 0:
            scores[QueryType.BRIDGE] = scores.get(QueryType.BRIDGE, 0.0) + 1.0
            scores[QueryType.CAUSAL_TRACE] = scores.get(QueryType.CAUSAL_TRACE, 0.0) + 0.5

        # Single-source high-confidence → REPLICATION / SOURCE_QUALITY proxy
        if state.get("single_source_high_confidence", 0) > 0:
            scores[QueryType.REPLICATION] = scores.get(QueryType.REPLICATION, 0.0) + 0.5
            scores[QueryType.VERIFY] = scores.get(QueryType.VERIFY, 0.0) + 0.5

        # Older findings with newer conflicting evidence → TEMPORAL
        if state.get("older_findings_newer_conflicting", 0) > 0:
            scores[QueryType.TEMPORAL] = scores.get(QueryType.TEMPORAL, 0.0) + 0.5

        # High-deviation cluster outliers → ONTOLOGY
        if state.get("high_deviation_outliers", 0) > 0:
            scores[QueryType.ONTOLOGY] = scores.get(QueryType.ONTOLOGY, 0.0) + 0.5

        # Information gain rate dropping → META_COVERAGE audit
        igr = state.get("information_gain_rate")
        if isinstance(igr, (int, float)) and igr < 0.05:
            scores[QueryType.META_COVERAGE] = scores.get(QueryType.META_COVERAGE, 0.0) + 1.0

        # Round-completion meta types always get a small baseline so they fire
        # when no stronger signal dominates.
        scores[QueryType.META_PRODUCTIVITY] = scores.get(QueryType.META_PRODUCTIVITY, 0.0) + 0.2
        scores[QueryType.META_EFFECTIVENESS] = scores.get(QueryType.META_EFFECTIVENESS, 0.0) + 0.2
        scores[QueryType.META_EXHAUSTION] = scores.get(QueryType.META_EXHAUSTION, 0.0) + 0.2

    # ------------------------------------------------------------------
    # Reflexion rules (§4.3)
    # ------------------------------------------------------------------

    def _apply_exhaustion_rules(
        self,
        scores: dict[QueryType, float],
        reflexion: ReflexionState,
    ) -> None:
        """Demote exhausted types by 50% and promote their successors."""
        for type_name in reflexion.exhausted_query_types:
            try:
                qt = QueryType(type_name)
            except ValueError:
                continue
            scores[qt] = scores.get(qt, 0.0) * 0.5
            successor = _EXHAUSTED_SUCCESSOR.get(qt)
            if successor is not None:
                scores[successor] = scores.get(successor, 0.0) * 1.3

    def _apply_productive_pair_rules(
        self,
        scores: dict[QueryType, float],
        reflexion: ReflexionState,
    ) -> None:
        """Boost successors of known productive sequences."""
        for first, second in reflexion.productive_pairs:
            try:
                successor = QueryType(second)
            except ValueError:
                continue
            scores[successor] = scores.get(successor, 0.0) * 1.25

    def _apply_breakthrough_boost(
        self,
        scores: dict[QueryType, float],
        reflexion: ReflexionState,
    ) -> None:
        """If breakthrough findings exist, boost deep analytical types."""
        if reflexion.breakthrough_findings:
            scores[QueryType.ONTOLOGY] = scores.get(QueryType.ONTOLOGY, 0.0) * 1.5
            scores[QueryType.CAUSAL_TRACE] = scores.get(QueryType.CAUSAL_TRACE, 0.0) * 1.4
            scores[QueryType.SYNTHESIZE] = scores.get(QueryType.SYNTHESIZE, 0.0) * 1.3

    def _apply_coverage_plateau_boost(
        self,
        scores: dict[QueryType, float],
        reflexion: ReflexionState,
    ) -> None:
        """If coverage scores are flat, boost exploratory meta types."""
        history = reflexion.coverage_score_history
        if len(history) >= 3:
            recent = history[-3:]
            if max(recent) - min(recent) < 0.02:
                scores[QueryType.META_COVERAGE] = scores.get(QueryType.META_COVERAGE, 0.0) * 1.5
                scores[QueryType.AGGREGATE] = scores.get(QueryType.AGGREGATE, 0.0) * 1.3
                scores[QueryType.BRIDGE] = scores.get(QueryType.BRIDGE, 0.0) * 1.2

    # ------------------------------------------------------------------
    # MCP injection rules (§4.5)
    # ------------------------------------------------------------------

    def _apply_mcp_injection_rules(
        self,
        scores: dict[QueryType, float],
        state: dict[str, Any],
        budget: dict[str, Any],
    ) -> None:
        """Override next-round budget when MCP injects fresh findings."""
        new_mcp = budget.get("new_mcp_findings", state.get("new_mcp_findings", 0))
        if isinstance(new_mcp, int) and new_mcp > 50:
            scores[QueryType.VALIDATE] = scores.get(QueryType.VALIDATE, 0.0) * 1.5
            scores[QueryType.VERIFY] = scores.get(QueryType.VERIFY, 0.0) * 1.5
            scores[QueryType.TEMPORAL] = scores.get(QueryType.TEMPORAL, 0.0) * 1.3
            scores[QueryType.SYNTHESIZE] = scores.get(QueryType.SYNTHESIZE, 0.0) * 0.1  # paused


# ---------------------------------------------------------------------------
# QueryBudgetAllocator
# ---------------------------------------------------------------------------

class QueryBudgetAllocator:
    """Allocate per-round budget across selected query types.

    Uses UCB-inspired allocation, logarithmic priority decay, and a
    serendipity floor for cross-domain preservation.
    """

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig.from_env()
        self._actor_id = "QueryBudgetAllocator"

    async def allocate(
        self,
        selected: list[QueryType],
        store_state: dict[str, Any],
        reflexion: ReflexionState,
        budget: dict[str, Any],
    ) -> dict[QueryType, int]:
        """Return the number of query slots allocated per selected type.

        Args:
            selected: Types chosen by QuerySelector (ordered by priority).
            store_state: Store topology snapshot.
            reflexion: Reflexion state with historical performance.
            budget: Budget descriptor containing at least ``max_queries``.
                May also contain ``ucb_alpha`` override.

        Returns:
            Mapping from QueryType to allocated query count (≥ 0).
        """
        phase = "allocation"
        await _trace(
            self._actor_id,
            "budget_allocator_start",
            phase,
            payload={
                "selected": [qt.value for qt in selected],
                "budget_keys": list(budget.keys()),
            },
        )

        max_queries = int(budget.get("max_queries", 100))
        if max_queries <= 0:
            await _trace(self._actor_id, "budget_allocator_empty", phase, payload={"reason": "max_queries <= 0"})
            return {}

        # Historical performance per type (fallback to uniform)
        type_magnitudes: dict[str, float] = store_state.get("type_magnitudes", {})
        type_selection_counts: dict[str, int] = store_state.get("type_selection_counts", {})

        ucb_alpha = budget.get("ucb_alpha", self.config.flock.ucb_alpha)
        total_rounds = sum(type_selection_counts.values()) or 1

        # Compute raw UCB scores
        raw_scores: dict[QueryType, float] = {}
        for qt in selected:
            name = qt.value
            mag = type_magnitudes.get(name, 0.5)
            count = type_selection_counts.get(name, 0)

            # UCB exploration bonus
            exploration = ucb_alpha * math.sqrt(
                math.log(total_rounds + 1) / (count + 1)
            )
            ucb_score = mag + exploration

            # Priority decay based on evaluation count of eligible conditions
            eligible_count = store_state.get(f"eligible_{name.lower()}", 0)
            decay = self._compute_priority_decay(eligible_count)
            decayed = ucb_score * decay

            # Serendipity floor for bridge / synthesize / challenge
            if qt in (QueryType.BRIDGE, QueryType.SYNTHESIZE, QueryType.CHALLENGE):
                serendipity_floor = self.config.flock.serendipity_floor
                decayed = max(decayed, serendipity_floor)

            raw_scores[qt] = max(decayed, 0.0)

        # Normalise to max_queries
        total_raw = sum(raw_scores.values()) or 1.0
        allocations: dict[QueryType, int] = {}
        remaining = max_queries

        # First pass: proportional floor
        for qt in selected:
            share = (raw_scores[qt] / total_raw) * max_queries
            allocations[qt] = max(0, int(share))
            remaining -= allocations[qt]

        # Second pass: distribute remainder by descending fractional remainder
        fractions = [
            (qt, (raw_scores[qt] / total_raw) * max_queries - allocations[qt])
            for qt in selected
        ]
        fractions.sort(key=lambda x: x[1], reverse=True)
        for qt, _ in fractions:
            if remaining <= 0:
                break
            allocations[qt] += 1
            remaining -= 1

        # Ensure every selected type gets at least 1 query (exploration floor)
        for qt in selected:
            if allocations.get(qt, 0) == 0 and max_queries >= len(selected):
                # Re-steal from the highest-allocation type if needed
                donor = max(allocations, key=lambda k: allocations[k])
                if allocations[donor] > 1:
                    allocations[donor] -= 1
                    allocations[qt] = 1

        await _trace(
            self._actor_id,
            "budget_allocator_end",
            phase,
            payload={
                "allocations": {qt.value: c for qt, c in allocations.items()},
                "total_allocated": sum(allocations.values()),
                "max_queries": max_queries,
            },
        )
        return allocations

    @staticmethod
    def _compute_priority_decay(eligible_count: int) -> float:
        """Logarithmic diminishing-returns decay.

        Formula: ``1.0 / (1.0 + 0.3 * log(1 + eligible_count))``.
        """
        return 1.0 / (1.0 + 0.3 * math.log(1 + max(eligible_count, 0)))


# ---------------------------------------------------------------------------
# CompositeQueryBuilder
# ---------------------------------------------------------------------------

class CompositeQueryBuilder:
    """Decompose composite queries into their constituent atomic queries.

    Composite queries are virtual types that expand into a coordinated
    sequence of foundation/depth/understandability queries.
    """

    _DECOMPOSITIONS: dict[QueryType, list[QueryType]] = {
        QueryType.DEEP_VALIDATE: [
            QueryType.VALIDATE,
            QueryType.ASSUMPTION_EXCAVATE,
            QueryType.EVIDENCE_MAP,
        ],
        QueryType.RESOLVE_CONTRADICTION: [
            QueryType.ADJUDICATE,
            QueryType.CAUSAL_TRACE,
            QueryType.ASSUMPTION_EXCAVATE,
        ],
        QueryType.SYNTHESIS_DEEPEN: [
            QueryType.SYNTHESIZE,
            QueryType.COUNTERFACTUAL,
            QueryType.CAUSAL_TRACE,
        ],
    }

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig.from_env()
        self._actor_id = "CompositeQueryBuilder"

    async def build(self, query_type: QueryType) -> list[QueryType]:
        """Expand a composite query into its atomic sub-queries.

        Args:
            query_type: The composite type to expand.

        Returns:
            Ordered list of atomic QueryType values. If *query_type* is not
            composite, returns a singleton list containing it unchanged.
        """
        phase = "composite_build"
        await _trace(
            self._actor_id,
            "composite_build_start",
            phase,
            payload={"input": query_type.value},
        )

        atoms = self._DECOMPOSITIONS.get(query_type, [query_type])
        result = list(atoms)

        await _trace(
            self._actor_id,
            "composite_build_end",
            phase,
            payload={
                "input": query_type.value,
                "expanded": [qt.value for qt in result],
                "is_composite": query_type in self._DECOMPOSITIONS,
            },
        )
        return result

    @classmethod
    def is_composite(cls, query_type: QueryType) -> bool:
        """Return True if *query_type* is a composite query."""
        return query_type in cls._DECOMPOSITIONS

    @classmethod
    def supported_composites(cls) -> list[QueryType]:
        """Return the list of composite query types supported."""
        return list(cls._DECOMPOSITIONS.keys())


# ---------------------------------------------------------------------------
# Convenience orchestrator helper
# ---------------------------------------------------------------------------

class QueryWealthEngine:
    """Facade that wires Selector → Allocator → Builder with tracing.

    This is the single entry-point recommended for the orchestrator.
    """

    def __init__(self, config: UnifiedConfig | None = None) -> None:
        self.config = config or UnifiedConfig.from_env()
        self.selector = QuerySelector(self.config)
        self.allocator = QueryBudgetAllocator(self.config)
        self.builder = CompositeQueryBuilder(self.config)
        self._actor_id = "QueryWealthEngine"

    async def plan_round(
        self,
        store_state: dict[str, Any],
        reflexion: ReflexionState,
        budget: dict[str, Any],
    ) -> dict[QueryType, list[QueryType]]:
        """Plan one full round: select, allocate, and expand composites.

        Returns:
            Mapping from each allocated query type to its expanded atom list.
            For non-composite types the value is a singleton list.
        """
        await _trace(
            self._actor_id,
            "plan_round_start",
            "planning",
            payload={"budget": budget, "store_keys": list(store_state.keys())},
        )

        selected = await self.selector.select(store_state, reflexion, budget)
        allocations = await self.allocator.allocate(selected, store_state, reflexion, budget)

        plan: dict[QueryType, list[QueryType]] = {}
        for qt, count in allocations.items():
            atoms = await self.builder.build(qt)
            plan[qt] = atoms
            if count > 1 and not CompositeQueryBuilder.is_composite(qt):
                # Duplicate singletons to reflect count (composites stay single entries)
                plan[qt] = atoms * count

        await _trace(
            self._actor_id,
            "plan_round_end",
            "planning",
            payload={
                "plan": {
                    qt.value: [a.value for a in atoms] for qt, atoms in plan.items()
                },
            },
        )
        return plan


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "QueryTypeDef",
    "QUERY_TYPE_DEFINITIONS",
    "QuerySelector",
    "QueryBudgetAllocator",
    "CompositeQueryBuilder",
    "QueryWealthEngine",
]
