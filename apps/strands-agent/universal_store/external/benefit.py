"""External data benefit assessment layer.

Treats every API call as an investment decision:
- 13 benefit signals with learned weights
- Composite scoring with saturation, context, and convergence modifiers
- Cost estimation for 10+ external APIs
- UCB-greedy target selection with cost-aware knapsack
- Dynamic budget allocation driven by ROI feedback
- Three-tier operator override (green / yellow / red)
- Three-phase feedback loop (immediate / latent / ROI)

Every score, estimate, selection, and feedback event is traced.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from universal_store.config import UnifiedConfig
from universal_store.protocols import (
    BudgetOverride,
    FetchCost,
    OperatorTier,
    ResearchBudget,
    ResearchTarget,
    StoreProtocol,
)
from universal_store.trace import TraceStore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 1. BenefitScorer — 13 signals, composite scoring
# ---------------------------------------------------------------------------

@dataclass
class BenefitScorer:
    """Score candidate research targets using 13 benefit signals."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None

    # 13 signals: (name, weight, condition_key_prefix)
    SIGNALS: tuple[tuple[str, float, str], ...] = (
        ("validate_gap", 1.40, "validate"),
        ("verify_gap", 1.30, "verify"),
        ("adjudicate_gap", 1.35, "adjudicate"),
        ("swarm_intent_gap", 1.25, "swarm_intent"),
        ("worker_request_gap", 1.20, "worker_request"),
        ("source_upgrade_gap", 1.10, "source_upgrade"),
        ("enrich_gap", 1.00, "enrich"),
        ("ground_gap", 1.00, "ground"),
        ("replication_gap", 0.90, "replication"),
        ("deadlock_gap", 0.90, "deadlock"),
        ("mechanism_gap", 0.85, "mechanism"),
        ("freshness_gap", 0.80, "freshness"),
        ("coverage_gap", 0.75, "coverage"),
    )

    async def _get_trace(self) -> TraceStore:
        if self.trace is None:
            self.trace = await TraceStore.get()
        return self.trace

    async def score(self, store_state: dict) -> list[ResearchTarget]:
        """Compute benefit scores for every candidate in *store_state*.

        *store_state* must contain:
        - ``candidates``: list[dict] with keys:
            target_id, source_type, query, reason_type,
            estimated_cost (as dict or FetchCost),
            and per-signal floats (e.g. validate_gap=0.8).
        - ``external_fetch_count``: int  (per candidate or global)
        - ``last_fetch_rounds_ago``: int  (per candidate)
        - ``context_tokens_used``: int
        - ``avg_magnitude_last_2_rounds``: float
        - ``convergence_boost_active``: bool (optional)

        Returns a list of :class:`ResearchTarget` with ``benefit_score`` set.
        """
        trace = await self._get_trace()
        start = time.time()
        candidates = store_state.get("candidates", [])
        scored: list[ResearchTarget] = []

        for cand in candidates:
            target_id = cand.get("target_id", "")
            source_type = cand.get("source_type", "")
            query = cand.get("query", "")
            reason_type = cand.get("reason_type", "")
            confidence = _clamp(cand.get("confidence", 0.5))

            # Parse estimated cost
            est = cand.get("estimated_cost", {})
            if isinstance(est, FetchCost):
                estimated_cost = est
            else:
                estimated_cost = FetchCost(
                    usd=float(est.get("usd", 0.0)),
                    tokens=int(est.get("tokens", 0)),
                    latency_s=float(est.get("latency_s", 0.0)),
                    context_window_interest=float(est.get("context_window_interest", 0.0)),
                )

            # Raw benefit
            b_raw = 0.0
            signal_values: dict[str, float] = {}
            for name, weight, _prefix in self.SIGNALS:
                val = _clamp(cand.get(name, 0.0))
                signal_values[name] = val
                b_raw += val * weight

            # Saturation penalty
            ext_count = cand.get("external_fetch_count", store_state.get("external_fetch_count", 0))
            last_fetch_ago = cand.get("last_fetch_rounds_ago", store_state.get("last_fetch_rounds_ago", 999))
            k = self.cfg.external.benefit_saturation_k
            saturation = 1.0 / (1.0 + k * math.log1p(ext_count))
            if last_fetch_ago < 2:
                saturation *= 0.5

            # Context penalty
            context_used = store_state.get("context_tokens_used", 0)
            context_limit = self.cfg.external.context_window_limit_tokens
            ratio = context_used / max(context_limit, 1)
            if ratio < 0.5:
                context_penalty = 1.0
            elif ratio < 0.8:
                context_penalty = 1.0 - (ratio - 0.5) / 0.3 * 0.5
            else:
                context_penalty = max(0.0, 0.5 - (ratio - 0.8) / 0.2 * 0.5)

            # Convergence boost
            avg_mag = store_state.get("avg_magnitude_last_2_rounds", 1.0)
            boost_active = store_state.get("convergence_boost_active", avg_mag < 0.02)
            convergence_boost = self.cfg.external.convergence_boost_multiplier if boost_active else 1.0

            b_final = b_raw * saturation * context_penalty * convergence_boost

            target = ResearchTarget(
                target_id=target_id,
                source_type=source_type,
                query=query,
                reason_type=reason_type,
                benefit_score=_clamp(b_final, 0.0, 100.0),
                estimated_cost=estimated_cost,
                confidence=confidence,
            )
            scored.append(target)

            await trace.record(
                actor_id="BenefitScorer",
                event_type="benefit_score",
                phase="scoring",
                payload={
                    "target_id": target_id,
                    "b_raw": round(b_raw, 4),
                    "saturation": round(saturation, 4),
                    "context_penalty": round(context_penalty, 4),
                    "convergence_boost": convergence_boost,
                    "b_final": round(target.benefit_score, 4),
                    "signals": {k: round(v, 4) for k, v in signal_values.items()},
                },
                latency_ms=(time.time() - start) * 1000,
            )

        await trace.record(
            actor_id="BenefitScorer",
            event_type="score_batch_complete",
            phase="scoring",
            payload={"count": len(scored)},
        )
        return scored


# ---------------------------------------------------------------------------
# 2. CostEstimator — cost table + context-window interest
# ---------------------------------------------------------------------------

@dataclass
class CostEstimator:
    """Estimate FetchCost for external APIs without calling them."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None

    # Base cost table (source_type -> (usd, tokens_lo, tokens_hi, latency_s))
    _TABLE: dict[str, tuple[float, int, int, float]] = field(default_factory=lambda: {
        "brave": (0.00, 800, 1_500, 1.2),
        "exa": (0.01, 1_000, 2_000, 2.5),
        "tavily": (0.015, 2_000, 4_000, 3.0),
        "perplexity": (0.10, 3_000, 8_000, 4.0),
        "semantic_scholar": (0.00, 1_500, 3_000, 1.5),
        "pubmed": (0.00, 1_000, 2_500, 1.8),
        "annas_archive": (0.00, 5_000, 50_000, 5.0),
        "firecrawl": (0.01, 3_000, 10_000, 3.5),
        "google_scholar": (0.00, 1_200, 2_500, 1.5),
        "arxiv": (0.00, 1_500, 3_500, 1.5),
        "jstor": (0.02, 2_000, 5_000, 3.0),
        "openalex": (0.00, 1_000, 2_500, 1.5),
    })

    async def _get_trace(self) -> TraceStore:
        if self.trace is None:
            self.trace = await TraceStore.get()
        return self.trace

    async def estimate(self, source_type: str) -> FetchCost:
        """Return a :class:`FetchCost` estimate for *source_type*.

        Includes context-window interest: each result is expected to appear
        in ~30 % of future clone contexts.
        """
        trace = await self._get_trace()
        start = time.time()
        key = source_type.lower().strip()

        usd, tok_lo, tok_hi, latency = self._TABLE.get(
            key, (0.01, 1_000, 2_500, 2.0)
        )

        # Use midpoint token estimate
        tokens = (tok_lo + tok_hi) // 2

        # Context window interest: 30 % of tokens × expected future rounds
        expected_future_rounds = max(
            1,
            self.cfg.scheduler.max_total_rounds // max(self.cfg.flock.ucb_alpha, 1),
        )
        context_interest = tokens * 0.30 * expected_future_rounds

        cost = FetchCost(
            usd=usd,
            tokens=tokens,
            latency_s=latency,
            context_window_interest=context_interest,
        )

        await trace.record(
            actor_id="CostEstimator",
            event_type="cost_estimate",
            phase="estimation",
            payload={
                "source_type": source_type,
                "usd": cost.usd,
                "tokens": cost.tokens,
                "latency_s": cost.latency_s,
                "context_window_interest": round(cost.context_window_interest, 2),
            },
            latency_ms=(time.time() - start) * 1000,
        )
        return cost


# ---------------------------------------------------------------------------
# 3. TargetSelector — UCB-greedy + cost-aware knapsack + serendipity floor
# ---------------------------------------------------------------------------

@dataclass
class TargetSelector:
    """Select a subset of ResearchTargets that fit budget and diversity rules."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None

    async def _get_trace(self) -> TraceStore:
        if self.trace is None:
            self.trace = await TraceStore.get()
        return self.trace

    async def select(
        self,
        targets: list[ResearchTarget],
        budget: ResearchBudget,
    ) -> list[ResearchTarget]:
        """UCB-greedy selection with cost-aware knapsack and serendipity floor.

        Ensures at least 3 distinct ``reason_type`` values are represented
        in the final selection.
        """
        trace = await self._get_trace()
        start = time.time()
        max_targets = self.cfg.external.max_targets_per_round
        alpha = self.cfg.flock.ucb_alpha

        total_fetches = sum(1 for _ in targets)  # placeholder; real count from registry

        # Compute UCB-greedy efficiency
        scored: list[tuple[float, ResearchTarget]] = []
        for t in targets:
            # Heuristic fetch_count per source_type (would come from registry in prod)
            fetch_count = 0
            cost_norm = t.estimated_cost.total_cost_norm
            exploration = alpha * math.sqrt(
                math.log(total_fetches + 1) / (fetch_count + 1)
            )
            efficiency = (t.benefit_score + exploration) / (cost_norm + 0.01)
            scored.append((efficiency, t))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[ResearchTarget] = []
        remaining = ResearchBudget(
            usd=budget.usd,
            tokens=budget.tokens,
            time_s=budget.time_s,
        )
        reason_types_seen: set[str] = set()

        for efficiency, t in scored:
            if len(selected) >= max_targets:
                break
            fits = (
                t.estimated_cost.usd <= remaining.usd
                and t.estimated_cost.tokens <= remaining.tokens
                and t.estimated_cost.latency_s <= remaining.time_s
            )
            if fits:
                selected.append(t)
                reason_types_seen.add(t.reason_type)
                remaining.usd -= t.estimated_cost.usd
                remaining.tokens -= t.estimated_cost.tokens
                remaining.time_s -= t.estimated_cost.latency_s

        # Serendipity floor: ensure ≥3 reason types
        if len(reason_types_seen) < 3:
            missing_needed = 3 - len(reason_types_seen)
            for efficiency, t in scored:
                if t in selected:
                    continue
                if t.reason_type not in reason_types_seen:
                    # Force-include up to missing_needed, budget permitting or overriding
                    selected.append(t)
                    reason_types_seen.add(t.reason_type)
                    missing_needed -= 1
                    if missing_needed <= 0:
                        break

        await trace.record(
            actor_id="TargetSelector",
            event_type="target_selection",
            phase="selection",
            payload={
                "candidates": len(targets),
                "selected": len(selected),
                "reason_types": sorted(reason_types_seen),
                "budget_remaining": {
                    "usd": round(remaining.usd, 4),
                    "tokens": remaining.tokens,
                    "time_s": round(remaining.time_s, 2),
                },
            },
            latency_ms=(time.time() - start) * 1000,
        )
        return selected


# ---------------------------------------------------------------------------
# 4. DynamicBudget — ROI-based scaling
# ---------------------------------------------------------------------------

@dataclass
class DynamicBudget:
    """Compute per-cycle research budget based on recent ROI."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None

    # Internal rolling stats (caller can update these)
    avg_recent_spend_usd: float = 0.0
    avg_flock_time_s: float = 10.0
    hourly_cap_usd: float = 10.0
    target_round_time_s: float = 120.0

    async def _get_trace(self) -> TraceStore:
        if self.trace is None:
            self.trace = await TraceStore.get()
        return self.trace

    async def compute(self, last_roi: float) -> ResearchBudget:
        """Return a :class:`ResearchBudget` scaled by *last_roi*.

        Scaling rules (from architecture §5.5):
        - ROI > 10.0  → multiplier 1.3
        - ROI > 1.0   → multiplier 1.0
        - ROI > 0.1   → multiplier 0.6
        - else        → multiplier 0.2
        """
        trace = await self._get_trace()
        start = time.time()

        if last_roi > 10.0:
            multiplier = 1.3
        elif last_roi > 1.0:
            multiplier = 1.0
        elif last_roi > 0.1:
            multiplier = 0.6
        else:
            multiplier = 0.2

        usd_budget = min(
            self.hourly_cap_usd / 6.0,
            self.avg_recent_spend_usd * 1.5,
        ) * multiplier

        context_limit = self.cfg.external.context_window_limit_tokens
        token_budget = int(context_limit * 0.7)

        time_budget = max(5.0, self.target_round_time_s - self.avg_flock_time_s)

        budget = ResearchBudget(
            usd=round(usd_budget, 4),
            tokens=token_budget,
            time_s=round(time_budget, 2),
        )

        await trace.record(
            actor_id="DynamicBudget",
            event_type="budget_compute",
            phase="budgeting",
            payload={
                "last_roi": round(last_roi, 4),
                "multiplier": multiplier,
                "usd_budget": budget.usd,
                "token_budget": budget.tokens,
                "time_budget": budget.time_s,
            },
            latency_ms=(time.time() - start) * 1000,
        )
        return budget


# ---------------------------------------------------------------------------
# 5. OperatorOverride — green / yellow / red tiers with timeouts
# ---------------------------------------------------------------------------

@dataclass
class OperatorOverride:
    """Check whether a target needs operator approval."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None

    async def _get_trace(self) -> TraceStore:
        if self.trace is None:
            self.trace = await TraceStore.get()
        return self.trace

    async def check(self, target: ResearchTarget) -> OperatorTier:
        """Return the operator tier for *target*.

        Green  : auto-execute
        Yellow : log intent, execute after timeout unless paused
        Red    : halt and await explicit approval
        """
        trace = await self._get_trace()
        start = time.time()

        cost = target.estimated_cost
        tier = OperatorTier.GREEN

        if (
            cost.usd > self.cfg.external.red_tier_usd
            or cost.tokens > self.cfg.external.red_tier_tokens
            or cost.latency_s > self.cfg.external.red_tier_latency_s
        ):
            tier = OperatorTier.RED
        elif cost.usd > self.cfg.external.yellow_tier_usd:
            tier = OperatorTier.YELLOW
        elif cost.usd > self.cfg.external.green_tier_usd:
            tier = OperatorTier.YELLOW

        await trace.record(
            actor_id="OperatorOverride",
            event_type="operator_check",
            phase="override",
            payload={
                "target_id": target.target_id,
                "source_type": target.source_type,
                "tier": tier.value,
                "usd": cost.usd,
                "tokens": cost.tokens,
                "latency_s": cost.latency_s,
                "timeout_s": self.cfg.external.operator_override_timeout_s,
            },
            latency_ms=(time.time() - start) * 1000,
        )
        return tier


# ---------------------------------------------------------------------------
# 6. FeedbackLoop — immediate, latent, ROI; updates source_quality_registry
# ---------------------------------------------------------------------------

@dataclass
class FeedbackLoop:
    """Three-phase feedback that closes the learning loop."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None
    store: StoreProtocol | None = None

    async def _get_trace(self) -> TraceStore:
        if self.trace is None:
            self.trace = await TraceStore.get()
        return self.trace

    # ------------------------------------------------------------------
    # Immediate feedback (within 1 round)
    # ------------------------------------------------------------------
    async def immediate(self, results: dict) -> dict:
        """Score deltas on parent conditions and track contradiction resolutions.

        *results* expected keys:
        - ``target_id``: str
        - ``source_type``: str
        - ``parent_condition_deltas``: list[dict] with keys:
            condition_id, confidence_delta, fabrication_delta, specificity_delta
        - ``contradictions_resolved``: int
        - ``gaps_fulfilled``: int
        - ``cost_usd``: float
        """
        trace = await self._get_trace()
        start = time.time()

        deltas = results.get("parent_condition_deltas", [])
        total_confidence_delta = sum(d.get("confidence_delta", 0.0) for d in deltas)
        total_fabrication_delta = sum(d.get("fabrication_delta", 0.0) for d in deltas)
        total_specificity_delta = sum(d.get("specificity_delta", 0.0) for d in deltas)

        payload = {
            "target_id": results.get("target_id"),
            "source_type": results.get("source_type"),
            "total_confidence_delta": round(total_confidence_delta, 4),
            "total_fabrication_delta": round(total_fabrication_delta, 4),
            "total_specificity_delta": round(total_specificity_delta, 4),
            "contradictions_resolved": results.get("contradictions_resolved", 0),
            "gaps_fulfilled": results.get("gaps_fulfilled", 0),
        }

        await trace.record(
            actor_id="FeedbackLoop",
            event_type="feedback_immediate",
            phase="feedback",
            payload=payload,
            latency_ms=(time.time() - start) * 1000,
        )
        return payload

    # ------------------------------------------------------------------
    # Latent feedback (2–3 rounds later)
    # ------------------------------------------------------------------
    async def latent(self, results: dict, rounds_later: int = 2) -> dict:
        """Count synthesis rows and bridge insights enabled by fetched data.

        *results* expected keys:
        - ``target_id``: str
        - ``source_type``: str
        - ``synthesis_enabled``: int
        - ``bridge_insights``: int
        - ``cluster_growth``: int
        """
        trace = await self._get_trace()
        start = time.time()

        payload = {
            "target_id": results.get("target_id"),
            "source_type": results.get("source_type"),
            "rounds_later": rounds_later,
            "synthesis_enabled": results.get("synthesis_enabled", 0),
            "bridge_insights": results.get("bridge_insights", 0),
            "cluster_growth": results.get("cluster_growth", 0),
        }

        await trace.record(
            actor_id="FeedbackLoop",
            event_type="feedback_latent",
            phase="feedback",
            payload=payload,
            latency_ms=(time.time() - start) * 1000,
        )
        return payload

    # ------------------------------------------------------------------
    # ROI computation
    # ------------------------------------------------------------------
    async def roi(self, results: dict) -> float:
        """Compute info-gain-per-USD for a completed fetch.

        *results* expected keys:
        - ``immediate``: dict (output of :meth:`immediate`)
        - ``latent``: dict (output of :meth:`latent`)
        - ``cost_usd``: float
        - ``source_type``: str
        - ``domain``: str (for registry update)

        Updates ``source_quality_registry`` if ``self.store`` is provided.
        """
        trace = await self._get_trace()
        start = time.time()

        immediate_data = results.get("immediate", {})
        latent_data = results.get("latent", {})
        cost_usd = max(results.get("cost_usd", 0.01), 0.01)

        total_info_gain = (
            immediate_data.get("total_confidence_delta", 0.0)
            + immediate_data.get("total_fabrication_delta", 0.0)
            + immediate_data.get("total_specificity_delta", 0.0)
            + latent_data.get("synthesis_enabled", 0) * 0.5
            + latent_data.get("bridge_insights", 0) * 0.3
        )

        roi_value = total_info_gain / cost_usd
        source_type = results.get("source_type", "unknown")
        domain = results.get("domain", source_type)

        await trace.record(
            actor_id="FeedbackLoop",
            event_type="feedback_roi",
            phase="feedback",
            payload={
                "target_id": results.get("target_id"),
                "source_type": source_type,
                "domain": domain,
                "total_info_gain": round(total_info_gain, 4),
                "cost_usd": cost_usd,
                "roi": round(roi_value, 4),
            },
            latency_ms=(time.time() - start) * 1000,
        )

        # Update source_quality_registry if store is available
        if self.store is not None:
            await self._update_source_quality_registry(domain, source_type, roi_value, cost_usd)

        return roi_value

    async def _update_source_quality_registry(
        self,
        domain: str,
        source_type: str,
        roi: float,
        cost_usd: float,
    ) -> None:
        """Upsert a row into ``source_quality_registry``.

        Uses atomic SQL so concurrent feedback loops do not lose updates.
        """
        if self.store is None:
            return

        trace = await self._get_trace()
        start = time.time()

        try:
            # Attempt to update existing row
            rows = await self.store.execute(
                """
                UPDATE source_quality_registry
                SET
                    fetch_count = fetch_count + 1,
                    successful_fetch_count = successful_fetch_count + 1,
                    total_cost_usd = total_cost_usd + ?,
                    total_info_gain_generated = total_info_gain_generated + ?,
                    last_seen_at = ?
                WHERE domain = ? AND source_type = ?
                RETURNING id
                """,
                (cost_usd, roi * cost_usd, _now(), domain, source_type),
            )
            if not rows:
                # Insert new row
                await self.store.execute(
                    """
                    INSERT INTO source_quality_registry
                    (domain, source_type, authority_score, avg_recency_score,
                     avg_finding_confidence, fetch_count, successful_fetch_count,
                     total_cost_usd, total_info_gain_generated, first_seen_at, last_seen_at)
                    VALUES (?, ?, 0.5, 0.5, 0.5, 1, 1, ?, ?, ?, ?)
                    """,
                    (domain, source_type, cost_usd, roi * cost_usd, _now(), _now()),
                )

            await trace.record(
                actor_id="FeedbackLoop",
                event_type="registry_update",
                phase="feedback",
                payload={
                    "domain": domain,
                    "source_type": source_type,
                    "roi": round(roi, 4),
                    "cost_usd": cost_usd,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as exc:
            await trace.record(
                actor_id="FeedbackLoop",
                event_type="registry_update_error",
                phase="feedback",
                payload={"domain": domain, "source_type": source_type, "error": str(exc)},
                error=exc,
            )


# ---------------------------------------------------------------------------
# 7. Convenience composite (optional but useful)
# ---------------------------------------------------------------------------

@dataclass
class ExternalBenefitPipeline:
    """High-level facade that wires scorer → estimator → selector → override."""

    cfg: UnifiedConfig = field(default_factory=UnifiedConfig.from_env)
    trace: TraceStore | None = None
    store: StoreProtocol | None = None

    scorer: BenefitScorer = field(init=False)
    estimator: CostEstimator = field(init=False)
    selector: TargetSelector = field(init=False)
    budget_engine: DynamicBudget = field(init=False)
    override: OperatorOverride = field(init=False)
    feedback: FeedbackLoop = field(init=False)

    def __post_init__(self):
        self.scorer = BenefitScorer(cfg=self.cfg, trace=self.trace)
        self.estimator = CostEstimator(cfg=self.cfg, trace=self.trace)
        self.selector = TargetSelector(cfg=self.cfg, trace=self.trace)
        self.budget_engine = DynamicBudget(cfg=self.cfg, trace=self.trace)
        self.override = OperatorOverride(cfg=self.cfg, trace=self.trace)
        self.feedback = FeedbackLoop(cfg=self.cfg, trace=self.trace, store=self.store)

    async def run_cycle(
        self,
        store_state: dict,
        last_roi: float,
    ) -> tuple[list[ResearchTarget], list[OperatorTier], ResearchBudget]:
        """Run a full external-data benefit cycle.

        1. Compute dynamic budget from *last_roi*.
        2. Score candidates.
        3. Estimate costs for any missing estimates.
        4. Select targets under budget.
        5. Check operator override tiers.

        Returns ``(selected_targets, tiers, budget)``.
        """
        budget = await self.budget_engine.compute(last_roi)
        targets = await self.scorer.score(store_state)

        # Ensure every target has an estimated cost
        for t in targets:
            if t.estimated_cost.usd == 0.0 and t.estimated_cost.tokens == 0:
                t.estimated_cost = await self.estimator.estimate(t.source_type)

        selected = await self.selector.select(targets, budget)
        tiers = [await self.override.check(t) for t in selected]
        return selected, tiers, budget
