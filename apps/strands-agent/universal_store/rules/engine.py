"""Rule Engine — codifies 12 operator behaviors as IF-THEN rules.

The RuleEngine sits above the actors, emitting ``RuleFired`` events that
actors consume.  It is stateless and deterministic: given the same
:class:`RuleContext` it always produces the same events.

Design constraints
------------------
* No imports from actor modules (breaks layering).
* Evaluation never blocks on I/O — condition and action callables must be
  pure/synchronous functions operating on the context snapshot.
* Every rule evaluation, firing, and conflict is traced via :class:`TraceStore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from universal_store.protocols import (
    BudgetOverride,
    Event,
    OperatorTier,
    OrchestratorPhase,
    ResearchBudget,
)
from universal_store.trace import TraceStore
from universal_store.config import UnifiedConfig


# ---------------------------------------------------------------------------
# RuleFired event
# ---------------------------------------------------------------------------

class RuleFired(Event):
    """Emitted by :class:`RuleEngine` when a rule matches and its action fires."""

    def __init__(
        self,
        rule_name: str,
        action_event: Event,
        context_summary: dict[str, Any] | None = None,
        **kw: Any,
    ):
        payload = {
            "rule_name": rule_name,
            "action_event_type": action_event.event_type,
            "action_payload": action_event.payload,
            "context_summary": context_summary or {},
        }
        super().__init__("RuleFired", payload, **kw)


# ---------------------------------------------------------------------------
# RuleContext
# ---------------------------------------------------------------------------

@dataclass
class RuleContext:
    """Immutable-style snapshot of world state used during rule evaluation.

    Actors assemble this context from their local view of the universe and
    pass it to :meth:`RuleEngine.evaluate`.
    """

    store_state: dict[str, Any] = field(default_factory=dict)
    actor_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_phase: OrchestratorPhase = OrchestratorPhase.IDLE
    round_number: int = 0
    session_rounds: int = 0
    last_event: Event | None = None
    budget: ResearchBudget = field(default_factory=lambda: ResearchBudget(0.0, 0, 0.0))
    budget_override: BudgetOverride = field(default_factory=BudgetOverride)
    cost_accumulated_usd: float = 0.0
    cost_accumulated_tokens: int = 0
    gpu_idle_s: float = 0.0
    destructive_ops_pending: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    model_blocks: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_decisions: list[dict[str, Any]] = field(default_factory=list)
    actor_crashes: list[dict[str, Any]] = field(default_factory=list)
    health_check_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    high_value_findings: list[dict[str, Any]] = field(default_factory=list)
    config: UnifiedConfig = field(default_factory=UnifiedConfig)
    run_id: str = ""


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """Single IF-THEN rule.

    Attributes
    ----------
    name:
        Human-readable identifier (also used for tracing).
    condition_fn:
        Synchronous predicate ``(RuleContext) -> bool``.  Must not block.
    action_fn:
        Synchronous factory ``(RuleContext) -> Event``.  Must not block.
    salience:
        Priority for conflict resolution (higher wins).
    specificity:
        Tie-breaker when two rules have equal salience (higher wins).
    conflicts_with:
        Set of rule *names* that this rule mutually excludes.  If a rule with
        higher salience in this set has already fired, this rule is suppressed.
    """

    name: str
    condition_fn: Callable[[RuleContext], bool]
    action_fn: Callable[[RuleContext], Event]
    salience: float = 1.0
    specificity: float = 1.0
    conflicts_with: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Built-in rule implementations
# ---------------------------------------------------------------------------

# -- R1: probe_validate_decide ---------------------------------------------


def _r1_condition(ctx: RuleContext) -> bool:
    """Before any destructive action, probe store state."""
    return len(ctx.destructive_ops_pending) > 0


def _r1_action(ctx: RuleContext) -> Event:
    op = ctx.destructive_ops_pending[0]
    return Event(
        "ProbeValidateDecide",
        {
            "op_type": op.get("op_type", "unknown"),
            "target": op.get("target", ""),
            "reason": "Probe store state before destructive action",
        },
    )


# -- R2: gpu_waste_detection ------------------------------------------------

_ACTIVE_PHASES = {
    OrchestratorPhase.SWARMING,
    OrchestratorPhase.FLOCKING,
    OrchestratorPhase.SYNTHESIZING,
    OrchestratorPhase.FETCHING_EXTERNAL,
}


def _r2_condition(ctx: RuleContext) -> bool:
    """If GPU idle > 30 s during active phase, alert."""
    return ctx.current_phase in _ACTIVE_PHASES and ctx.gpu_idle_s > 30.0


def _r2_action(ctx: RuleContext) -> Event:
    return Event(
        "GpuWasteAlert",
        {"gpu_idle_s": ctx.gpu_idle_s, "phase": ctx.current_phase.value},
    )


# -- R3: subagent_spawn -----------------------------------------------------


def _r3_condition(ctx: RuleContext) -> bool:
    """If gap requires specialized skill, spawn helper."""
    if not ctx.gaps:
        return False
    for gap in ctx.gaps:
        if gap.startswith("skill:") or gap.startswith("specialized:"):
            return True
    return False


def _r3_action(ctx: RuleContext) -> Event:
    gap = next(
        g for g in ctx.gaps if g.startswith("skill:") or g.startswith("specialized:")
    )
    return Event(
        "SubagentSpawn",
        {"gap": gap, "reason": "Specialized skill required to close gap"},
    )


# -- R4: model_censorship_check ---------------------------------------------


def _r4_condition(ctx: RuleContext) -> bool:
    """If response blocked, reroute to backup model."""
    return any(block.get("blocked", False) for block in ctx.model_blocks)


def _r4_action(ctx: RuleContext) -> Event:
    block = next(b for b in ctx.model_blocks if b.get("blocked", False))
    return Event(
        "ModelReroute",
        {
            "original_model": block.get("model", "unknown"),
            "reason": "Response blocked by censorship filter",
            "fallback_model": block.get("fallback_model", "backup-model"),
        },
    )


# -- R5: cost_tracking ------------------------------------------------------


def _r5_condition(ctx: RuleContext) -> bool:
    """Accumulate cost per run; alert if > budget."""
    usd_exceeded = ctx.cost_accumulated_usd > ctx.budget.usd > 0
    token_exceeded = ctx.cost_accumulated_tokens > ctx.budget.tokens > 0
    return usd_exceeded or token_exceeded


def _r5_action(ctx: RuleContext) -> Event:
    return Event(
        "BudgetAlert",
        {
            "cost_usd": ctx.cost_accumulated_usd,
            "cost_tokens": ctx.cost_accumulated_tokens,
            "budget_usd": ctx.budget.usd,
            "budget_tokens": ctx.budget.tokens,
            "reason": "Accumulated cost exceeds budget",
        },
    )


# -- R6: destructive_action_gate --------------------------------------------


def _r6_condition(ctx: RuleContext) -> bool:
    """Require operator confirmation for destructive ops."""
    return any(
        op.get("requires_confirmation", True) for op in ctx.destructive_ops_pending
    )


def _r6_action(ctx: RuleContext) -> Event:
    op = next(
        o for o in ctx.destructive_ops_pending if o.get("requires_confirmation", True)
    )
    return Event(
        "OperatorConfirmationRequired",
        {
            "op_type": op.get("op_type", "unknown"),
            "target": op.get("target", ""),
            "reason": "Destructive action gated pending operator approval",
        },
    )


# -- R7: multi_agent_consultation -------------------------------------------


def _r7_condition(ctx: RuleContext) -> bool:
    """If ambiguous decision, consult 3+ angles."""
    return len(ctx.ambiguous_decisions) > 0


def _r7_action(ctx: RuleContext) -> Event:
    decision = ctx.ambiguous_decisions[0]
    return Event(
        "MultiAgentConsultation",
        {
            "decision_id": decision.get("decision_id", ""),
            "proposed_action": decision.get("proposed_action", ""),
            "required_angles": 3,
            "reason": "Ambiguous decision requires multi-angle consultation",
        },
    )


# -- R8: economic_guardrails ------------------------------------------------


def _r8_condition(ctx: RuleContext) -> bool:
    """Enforce budget caps and rate limits."""
    budget = ctx.budget
    if budget.usd <= 0 and budget.tokens <= 0:
        return False
    usd_ratio = ctx.cost_accumulated_usd / max(budget.usd, 1)
    token_ratio = ctx.cost_accumulated_tokens / max(budget.tokens, 1)
    return usd_ratio >= 0.9 or token_ratio >= 0.9


def _r8_action(ctx: RuleContext) -> Event:
    return Event(
        "EconomicGuardrailTriggered",
        {
            "cost_usd": ctx.cost_accumulated_usd,
            "cost_tokens": ctx.cost_accumulated_tokens,
            "budget_usd": ctx.budget.usd,
            "budget_tokens": ctx.budget.tokens,
            "reason": "Economic guardrail: approaching or exceeding budget cap",
        },
    )


# -- R9: context_accumulation -----------------------------------------------


def _r9_condition(ctx: RuleContext) -> bool:
    """If session > 100 rounds, summarize and checkpoint."""
    return ctx.session_rounds > 100


def _r9_action(ctx: RuleContext) -> Event:
    return Event(
        "ContextSummarizeCheckpoint",
        {
            "session_rounds": ctx.session_rounds,
            "reason": "Session exceeded 100 rounds; summarize and checkpoint context",
        },
    )


# -- R10: error_recovery ----------------------------------------------------


def _r10_condition(ctx: RuleContext) -> bool:
    """On actor crash, attempt restart with fallback strategy."""
    return len(ctx.actor_crashes) > 0


def _r10_action(ctx: RuleContext) -> Event:
    crash = ctx.actor_crashes[0]
    return Event(
        "ActorRestartFallback",
        {
            "actor_id": crash.get("actor_id", ""),
            "error": crash.get("error", ""),
            "fallback_strategy": crash.get("fallback_strategy", "one_for_one_restart"),
            "reason": "Actor crash detected; initiate fallback restart",
        },
    )


# -- R11: priority_reorder --------------------------------------------------


def _r11_condition(ctx: RuleContext) -> bool:
    """If high-value finding detected, boost priority."""
    return len(ctx.high_value_findings) > 0


def _r11_action(ctx: RuleContext) -> Event:
    finding = ctx.high_value_findings[0]
    return Event(
        "PriorityBoost",
        {
            "finding_id": finding.get("finding_id", ""),
            "value_score": finding.get("value_score", 0.0),
            "reason": "High-value finding detected; boost processing priority",
        },
    )


# -- R12: health_monitor ----------------------------------------------------


def _r12_condition(ctx: RuleContext) -> bool:
    """If any actor unhealthy for > 3 checks, escalate."""
    for checks in ctx.health_check_history.values():
        unhealthy = [c for c in checks if c.get("status") != "healthy"]
        if len(unhealthy) > 3:
            return True
    return False


def _r12_action(ctx: RuleContext) -> Event:
    for actor_id, checks in ctx.health_check_history.items():
        unhealthy = [c for c in checks if c.get("status") != "healthy"]
        if len(unhealthy) > 3:
            return Event(
                "HealthEscalation",
                {
                    "actor_id": actor_id,
                    "unhealthy_checks": len(unhealthy),
                    "latest_status": checks[-1].get("status", "unknown"),
                    "reason": "Actor unhealthy for more than 3 consecutive checks",
                },
            )
    return Event(
        "HealthEscalation",
        {"actor_id": "unknown", "reason": "Health escalation triggered"},
    )


# ---------------------------------------------------------------------------
# RuleEngine
# ---------------------------------------------------------------------------

class RuleEngine:
    """Stateful container for rules.  Evaluates a :class:`RuleContext` and
    returns a list of :class:`RuleFired` events.

    Conflict resolution happens automatically: if two fired rules declare each
    other in ``conflicts_with``, the one with higher *salience* wins; on a
    tie, higher *specificity* wins.
    """

    def __init__(self, rules: list[Rule] | None = None):
        self._rules: list[Rule] = list(rules) if rules is not None else []

    @classmethod
    def with_default_rules(cls) -> "RuleEngine":
        """Factory that pre-loads the 12 canonical operator rules."""
        return cls(
            rules=[
                Rule(
                    name="probe_validate_decide",
                    condition_fn=_r1_condition,
                    action_fn=_r1_action,
                    salience=5.0,
                    specificity=3.0,
                ),
                Rule(
                    name="gpu_waste_detection",
                    condition_fn=_r2_condition,
                    action_fn=_r2_action,
                    salience=4.0,
                    specificity=4.0,
                ),
                Rule(
                    name="subagent_spawn",
                    condition_fn=_r3_condition,
                    action_fn=_r3_action,
                    salience=3.0,
                    specificity=3.0,
                ),
                Rule(
                    name="model_censorship_check",
                    condition_fn=_r4_condition,
                    action_fn=_r4_action,
                    salience=6.0,
                    specificity=4.0,
                ),
                Rule(
                    name="cost_tracking",
                    condition_fn=_r5_condition,
                    action_fn=_r5_action,
                    salience=5.0,
                    specificity=2.0,
                    conflicts_with={"economic_guardrails"},
                ),
                Rule(
                    name="destructive_action_gate",
                    condition_fn=_r6_condition,
                    action_fn=_r6_action,
                    salience=7.0,
                    specificity=4.0,
                ),
                Rule(
                    name="multi_agent_consultation",
                    condition_fn=_r7_condition,
                    action_fn=_r7_action,
                    salience=3.0,
                    specificity=3.0,
                ),
                Rule(
                    name="economic_guardrails",
                    condition_fn=_r8_condition,
                    action_fn=_r8_action,
                    salience=8.0,
                    specificity=3.0,
                    conflicts_with={"cost_tracking"},
                ),
                Rule(
                    name="context_accumulation",
                    condition_fn=_r9_condition,
                    action_fn=_r9_action,
                    salience=2.0,
                    specificity=2.0,
                ),
                Rule(
                    name="error_recovery",
                    condition_fn=_r10_condition,
                    action_fn=_r10_action,
                    salience=9.0,
                    specificity=4.0,
                ),
                Rule(
                    name="priority_reorder",
                    condition_fn=_r11_condition,
                    action_fn=_r11_action,
                    salience=3.0,
                    specificity=3.0,
                ),
                Rule(
                    name="health_monitor",
                    condition_fn=_r12_condition,
                    action_fn=_r12_action,
                    salience=8.0,
                    specificity=4.0,
                ),
            ]
        )

    def add_rule(self, rule: Rule) -> None:
        """Register an additional rule."""
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name.  Returns ``True`` if found."""
        for idx, r in enumerate(self._rules):
            if r.name == name:
                self._rules.pop(idx)
                return True
        return False

    async def evaluate(self, context: RuleContext) -> list[Event]:
        """Evaluate all rules against *context* and return fired events.

        The method is ``async`` solely so that trace records can be written
        to the non-blocking :class:`TraceStore` queue.  No rule condition or
        action is permitted to perform blocking I/O.
        """
        trace = await TraceStore.get()
        await trace.record(
            actor_id="rule_engine",
            event_type="rule_evaluation_start",
            phase=context.current_phase.value,
            payload={
                "run_id": context.run_id,
                "round": context.round_number,
                "session_rounds": context.session_rounds,
                "rule_count": len(self._rules),
            },
        )

        fired: list[tuple[Rule, Event]] = []

        for rule in self._rules:
            matched = False
            try:
                matched = rule.condition_fn(context)
            except Exception as exc:  # noqa: BLE001
                await trace.record(
                    actor_id="rule_engine",
                    event_type="rule_condition_error",
                    phase=context.current_phase.value,
                    payload={"rule": rule.name, "error": str(exc)},
                )
                continue

            await trace.record(
                actor_id="rule_engine",
                event_type="rule_evaluated",
                phase=context.current_phase.value,
                payload={"rule": rule.name, "matched": matched},
            )

            if not matched:
                continue

            try:
                action_event = rule.action_fn(context)
            except Exception as exc:  # noqa: BLE001
                await trace.record(
                    actor_id="rule_engine",
                    event_type="rule_action_error",
                    phase=context.current_phase.value,
                    payload={"rule": rule.name, "error": str(exc)},
                )
                continue

            fired.append((rule, action_event))
            await trace.record(
                actor_id="rule_engine",
                event_type="rule_fired",
                phase=context.current_phase.value,
                payload={
                    "rule": rule.name,
                    "event_type": action_event.event_type,
                    "salience": rule.salience,
                    "specificity": rule.specificity,
                },
            )

        # Conflict resolution
        resolved = self._resolve_conflicts(fired)

        kept_names = {r.name for r, _ in resolved}
        for rule, event in fired:
            if rule.name not in kept_names:
                await trace.record(
                    actor_id="rule_engine",
                    event_type="rule_conflict_suppressed",
                    phase=context.current_phase.value,
                    payload={
                        "rule": rule.name,
                        "event_type": event.event_type,
                        "reason": "conflict_with_higher_salience_rule",
                    },
                )

        # Wrap resolved events in RuleFired envelopes
        result: list[Event] = []
        for rule, action_event in resolved:
            result.append(
                RuleFired(
                    rule_name=rule.name,
                    action_event=action_event,
                    context_summary={
                        "phase": context.current_phase.value,
                        "round": context.round_number,
                        "run_id": context.run_id,
                    },
                )
            )

        await trace.record(
            actor_id="rule_engine",
            event_type="rule_evaluation_end",
            phase=context.current_phase.value,
            payload={
                "run_id": context.run_id,
                "evaluated": len(self._rules),
                "matched": len(fired),
                "fired_after_conflicts": len(result),
            },
        )

        return result

    def _resolve_conflicts(
        self, fired: list[tuple[Rule, Event]]
    ) -> list[tuple[Rule, Event]]:
        """Keep only non-conflicting rules, preferring higher salience."""
        # Sort by salience DESC, then specificity DESC
        sorted_fired = sorted(
            fired,
            key=lambda pair: (-pair[0].salience, -pair[0].specificity),
        )

        kept: list[tuple[Rule, Event]] = []
        kept_names: set[str] = set()

        for rule, event in sorted_fired:
            if rule.conflicts_with & kept_names:
                continue
            kept.append((rule, event))
            kept_names.add(rule.name)

        return kept
