"""MCP Researcher Actor — external data fetching via MCP tools.

The `McpResearcherActor` receives `ResearchNeeded` events, scores benefit,
estimates cost, selects targets via UCB-greedy, executes fetches through
per-tool `ToolActor` children, and emits `McpResearchComplete`.

No external APIs are called directly; all tool interactions are placeholder
async functions suitable for testing and future wiring.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from typing import Any

from universal_store.actors.base import Actor
from universal_store.config import UnifiedConfig
from universal_store.protocols import (
    Event,
    FetchCost,
    McpResearchComplete,
    OperatorTier,
    ResearchBudget,
    ResearchTarget,
    StoreProtocol,
)
from universal_store.trace import TraceStore, trace_block

# ---------------------------------------------------------------------------
# Placeholder async tool functions — never call real APIs
# ---------------------------------------------------------------------------

async def _placeholder_brave_search(query: str) -> list[dict[str, Any]]:
    await asyncio.sleep(0.01)
    return [{"url": "https://example.com/brave", "snippet": f"Brave result for {query}"}]


async def _placeholder_exa_search(query: str) -> list[dict[str, Any]]:
    await asyncio.sleep(0.01)
    return [{"url": "https://example.com/exa", "snippet": f"Exa result for {query}"}]


async def _placeholder_tavily_search(query: str) -> list[dict[str, Any]]:
    await asyncio.sleep(0.01)
    return [{"url": "https://example.com/tavily", "snippet": f"Tavily result for {query}"}]


async def _placeholder_pubmed_search(query: str) -> list[dict[str, Any]]:
    await asyncio.sleep(0.01)
    return [{"url": "https://example.com/pubmed", "snippet": f"PubMed result for {query}"}]


async def _placeholder_annas_archive_search(query: str) -> list[dict[str, Any]]:
    await asyncio.sleep(0.01)
    return [{"url": "https://example.com/anna", "snippet": f"AnnaArchive result for {query}"}]


_TOOL_DISPATCH: dict[str, Any] = {
    "brave": _placeholder_brave_search,
    "exa": _placeholder_exa_search,
    "tavily": _placeholder_tavily_search,
    "pubmed": _placeholder_pubmed_search,
    "anna_archive": _placeholder_annas_archive_search,
}

# ---------------------------------------------------------------------------
# ToolActor — one per MCP tool target
# ---------------------------------------------------------------------------

class ToolActor(Actor):
    """Lightweight actor that executes a single MCP tool fetch.

    Parameters
    ----------
    actor_id: Unique identifier (should include tool name).
    source_type: MCP tool key (e.g. ``"brave"``, ``"pubmed"``).
    query: The search query to execute.
    """

    def __init__(self, actor_id: str, source_type: str, query: str):
        super().__init__(actor_id)
        self.source_type = source_type
        self.query = query
        self.result: list[dict[str, Any]] | None = None
        self.error: Exception | None = None

    async def _run(self) -> None:
        trace = await TraceStore.get()
        async with trace_block(self.actor_id, "tool_fetch", self.source_type, {"query": self.query}):
            fn = _TOOL_DISPATCH.get(self.source_type)
            if fn is None:
                self.error = ValueError(f"Unknown source_type: {self.source_type}")
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="tool_fetch_unknown_source",
                    error=self.error,
                )
                return
            try:
                self.result = await fn(self.query)
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="tool_fetch_success",
                    payload={"source_type": self.source_type, "results": len(self.result)},
                )
            except Exception as exc:
                self.error = exc
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="tool_fetch_error",
                    error=exc,
                )


# ---------------------------------------------------------------------------
# McpResearcherActor
# ---------------------------------------------------------------------------

class McpResearcherActor(Actor):
    """Handles external data fetching via MCP tools.

    Receives ``ResearchNeeded`` events, evaluates gaps, selects targets under
    budget, performs operator-tier override checks, spawns ``ToolActor``
    children, persists findings to the store, and emits
    ``McpResearchComplete``.

    Parameters
    ----------
    actor_id: Unique identifier for this actor.
    store: A ``StoreProtocol`` implementation (e.g. DuckDB store).
    config: ``UnifiedConfig`` instance; falls back to ``from_env()`` if omitted.
    """

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig.from_env()
        self._paused: bool = False
        self._operator_events: asyncio.Queue[Event] = asyncio.Queue()
        # UCB state: maps target_id -> {count, avg_reward}
        self._ucb_state: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main loop: read ``ResearchNeeded`` events from mailbox."""
        trace = await TraceStore.get()
        await trace.record(actor_id=self.actor_id, event_type="mcp_researcher_started")

        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if event.event_type == "ResearchNeeded":
                async with trace_block(self.actor_id, "research_cycle", "", event.payload):
                    await self._handle_research_needed(event)
            elif event.event_type == "OperatorDecision":
                await self._operator_events.put(event)
            elif event.event_type == "Pause":
                self._paused = True
                await trace.record(actor_id=self.actor_id, event_type="mcp_researcher_paused")
            elif event.event_type == "Resume":
                self._paused = False
                await trace.record(actor_id=self.actor_id, event_type="mcp_researcher_resumed")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_research_needed(self, event: Event) -> None:
        """Orchestrate a full research cycle for a single ``ResearchNeeded``."""
        trace = await TraceStore.get()
        payload = event.payload
        gaps: list[dict] = payload.get("gaps", [])
        budget_dict: dict = payload.get("budget", {})
        budget = ResearchBudget(
            usd=budget_dict.get("usd", self.config.external.green_tier_usd * 5),
            tokens=budget_dict.get("tokens", 10_000),
            time_s=budget_dict.get("time_s", 60.0),
        )

        await trace.record(
            actor_id=self.actor_id,
            event_type="research_needed_received",
            payload={"gap_count": len(gaps), "budget": budget.__dict__},
        )

        if not gaps:
            await self._emit_complete(0, 0.0, "none")
            return

        # 1. Evaluate benefit
        targets = self.evaluate_benefit(gaps)
        await trace.record(
            actor_id=self.actor_id,
            event_type="benefit_evaluation_done",
            payload={"targets": len(targets)},
        )

        # 2. Estimate cost (traced inside method)
        for t in targets:
            t.estimated_cost = await self._traced_estimate_cost(t)

        # 3. Select targets under budget
        selected = self.select_targets(targets, budget)
        await trace.record(
            actor_id=self.actor_id,
            event_type="targets_selected",
            payload={"selected": len(selected), "target_ids": [t.target_id for t in selected]},
        )

        if not selected:
            await self._emit_complete(0, 0.0, "none")
            return

        # 4. Operator override + execution
        findings_added = 0
        total_cost_usd = 0.0
        source_type_set: set[str] = set()

        for target in selected:
            tier = await self._traced_operator_override_check(target)

            if tier == OperatorTier.RED:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="operator_override_red",
                    payload={"target_id": target.target_id, "reason": "cost/latency exceeds red tier"},
                )
                await self._halt_and_await_operator(target)
                # After resuming, re-evaluate
                tier = await self._traced_operator_override_check(target)
                if tier == OperatorTier.RED:
                    continue

            if tier == OperatorTier.YELLOW:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="operator_override_yellow",
                    payload={"target_id": target.target_id, "wait_s": self.config.external.operator_override_timeout_s},
                )
                await self._log_intent_and_wait(target)
                if self._paused:
                    await self._halt_and_await_operator(target)

            # Green (or yellow after wait) -> execute
            result, cost = await self._execute_target(target)
            total_cost_usd += cost.usd
            source_type_set.add(target.source_type)
            if result:
                row_id = await self._write_finding(target, result)
                if row_id is not None:
                    findings_added += 1

        await self._emit_complete(
            findings_added,
            round(total_cost_usd, 6),
            ",".join(sorted(source_type_set)) if source_type_set else "none",
        )

    # ------------------------------------------------------------------
    # 1. Benefit evaluation
    # ------------------------------------------------------------------

    def evaluate_benefit(self, gaps: list[dict]) -> list[ResearchTarget]:
        """Score each gap with 13 benefit signals and return ranked targets.

        The 13 signals are heuristic dimensions that estimate how much a gap
        would benefit from external data. Weights are placeholders and can be
        tuned via reflexion lessons.
        """
        targets: list[ResearchTarget] = []
        for gap in gaps:
            signals = self._compute_signals(gap)
            # Weighted sum (placeholder weights)
            weights = [
                0.15,  # coverage_gap
                0.12,  # recency_need
                0.10,  # authority_need
                0.10,  # contradiction_risk
                0.08,  # novelty_potential
                0.08,  # depth_need
                0.08,  # methodological_need
                0.07,  # statistical_need
                0.07,  # temporal_need
                0.06,  # analogy_potential
                0.05,  # narrative_need
                0.04,  # replication_need
                0.00,  # serendipity_boost (added after selection)
            ]
            score = sum(s * w for s, w in zip(signals, weights))

            target_id = gap.get("gap_id", str(uuid.uuid4())[:8])
            target = ResearchTarget(
                target_id=target_id,
                source_type=gap.get("preferred_source", "brave"),
                query=gap.get("query", ""),
                reason_type=gap.get("reason_type", "coverage_gap"),
                benefit_score=min(score, 1.0),
                estimated_cost=FetchCost(),  # populated later
                confidence=gap.get("confidence", 0.5),
            )
            targets.append(target)
        return targets

    @staticmethod
    def _compute_signals(gap: dict) -> list[float]:
        """Return 13 normalized benefit signals for a single gap."""
        return [
            float(gap.get("coverage_gap", 0.5)),
            float(gap.get("recency_need", 0.5)),
            float(gap.get("authority_need", 0.5)),
            float(gap.get("contradiction_risk", 0.5)),
            float(gap.get("novelty_potential", 0.5)),
            float(gap.get("depth_need", 0.5)),
            float(gap.get("methodological_need", 0.5)),
            float(gap.get("statistical_need", 0.5)),
            float(gap.get("temporal_need", 0.5)),
            float(gap.get("analogy_potential", 0.5)),
            float(gap.get("narrative_need", 0.5)),
            float(gap.get("replication_need", 0.5)),
            float(gap.get("serendipity_boost", 0.0)),
        ]

    # ------------------------------------------------------------------
    # 2. Cost estimation
    # ------------------------------------------------------------------

    async def _traced_estimate_cost(self, target: ResearchTarget) -> FetchCost:
        trace = await TraceStore.get()
        async with trace_block(self.actor_id, "estimate_cost", target.source_type, {"target_id": target.target_id}):
            cost = self.estimate_cost(target)
            await trace.record(
                actor_id=self.actor_id,
                event_type="cost_estimate",
                payload={
                    "target_id": target.target_id,
                    "usd": cost.usd,
                    "tokens": cost.tokens,
                    "latency_s": cost.latency_s,
                },
            )
            return cost

    def estimate_cost(self, target: ResearchTarget) -> FetchCost:
        """Placeholder cost model per MCP API.

        Costs are derived from historical averages per source type.
        """
        source = target.source_type.lower()
        if source == "brave":
            return FetchCost(usd=0.01, tokens=500, latency_s=0.5, context_window_interest=0.02)
        if source == "exa":
            return FetchCost(usd=0.03, tokens=800, latency_s=0.8, context_window_interest=0.03)
        if source == "tavily":
            return FetchCost(usd=0.02, tokens=600, latency_s=0.6, context_window_interest=0.025)
        if source == "pubmed":
            return FetchCost(usd=0.00, tokens=300, latency_s=1.2, context_window_interest=0.01)
        if source == "anna_archive":
            return FetchCost(usd=0.00, tokens=200, latency_s=2.0, context_window_interest=0.015)
        return FetchCost(usd=0.05, tokens=500, latency_s=1.0, context_window_interest=0.02)

    # ------------------------------------------------------------------
    # 3. Target selection (UCB-greedy + knapsack + serendipity floor)
    # ------------------------------------------------------------------

    def select_targets(
        self,
        targets: list[ResearchTarget],
        budget: ResearchBudget,
    ) -> list[ResearchTarget]:
        """Select a subset of targets via UCB-greedy under knapsack constraints.

        Enforces a serendipity floor: at least 3 distinct ``reason_type``
        values must be represented in the final selection.
        """
        if not targets:
            return []

        # Normalise each target's value/cost ratio with UCB bonus
        scored: list[tuple[float, ResearchTarget]] = []
        total_pulls = sum(self._ucb_state.get(t.target_id, {}).get("count", 0) for t in targets) or 1

        for t in targets:
            state = self._ucb_state.setdefault(t.target_id, {"count": 0, "avg_reward": 0.0})
            count = state["count"]
            avg_reward = state["avg_reward"]
            # UCB1 bonus
            bonus = math.sqrt(2 * math.log(total_pulls) / max(count, 1))
            # Value = benefit / normalised cost
            cost_norm = t.estimated_cost.total_cost_norm if t.estimated_cost.total_cost_norm > 0 else 0.1
            ucb_score = (avg_reward + bonus) * (t.benefit_score / cost_norm)
            scored.append((ucb_score, t))

        # Greedy knapsack sort by descending UCB score
        scored.sort(key=lambda x: x[0], reverse=True)

        selected: list[ResearchTarget] = []
        spent_usd = 0.0
        spent_tokens = 0
        spent_time = 0.0
        reason_types: set[str] = set()

        for _, t in scored:
            c = t.estimated_cost
            if (
                spent_usd + c.usd <= budget.usd
                and spent_tokens + c.tokens <= budget.tokens
                and spent_time + c.latency_s <= budget.time_s
            ):
                selected.append(t)
                spent_usd += c.usd
                spent_tokens += c.tokens
                spent_time += c.latency_s
                reason_types.add(t.reason_type)

        # Serendipity floor: ensure >= 3 reason types
        if len(reason_types) < 3:
            for _, t in scored:
                if t not in selected and t.reason_type not in reason_types:
                    c = t.estimated_cost
                    if (
                        spent_usd + c.usd <= budget.usd
                        and spent_tokens + c.tokens <= budget.tokens
                        and spent_time + c.latency_s <= budget.time_s
                    ):
                        selected.append(t)
                        spent_usd += c.usd
                        spent_tokens += c.tokens
                        spent_time += c.latency_s
                        reason_types.add(t.reason_type)
                    if len(reason_types) >= 3:
                        break

        return selected

    # ------------------------------------------------------------------
    # 4. Operator override check
    # ------------------------------------------------------------------

    async def _traced_operator_override_check(self, target: ResearchTarget) -> OperatorTier:
        trace = await TraceStore.get()
        async with trace_block(self.actor_id, "operator_override_check", target.source_type, {"target_id": target.target_id}):
            tier = self.operator_override_check(target)
            await trace.record(
                actor_id=self.actor_id,
                event_type="operator_override",
                payload={
                    "target_id": target.target_id,
                    "tier": tier.value,
                    "cost_usd": target.estimated_cost.usd,
                },
            )
            return tier

    def operator_override_check(self, target: ResearchTarget) -> OperatorTier:
        """Classify a target into green / yellow / red tiers.

        Thresholds are driven by ``ExternalDataConfig``.
        """
        c = target.estimated_cost
        ext = self.config.external

        if c.usd >= ext.red_tier_usd or c.tokens >= ext.red_tier_tokens or c.latency_s >= ext.red_tier_latency_s:
            return OperatorTier.RED
        if c.usd >= ext.yellow_tier_usd:
            return OperatorTier.YELLOW
        return OperatorTier.GREEN

    async def _log_intent_and_wait(self, target: ResearchTarget) -> None:
        """Yellow tier: log intent and wait briefly unless paused."""
        trace = await TraceStore.get()
        await trace.record(
            actor_id=self.actor_id,
            event_type="yellow_tier_wait_start",
            payload={"target_id": target.target_id, "wait_s": self.config.external.operator_override_timeout_s},
        )
        try:
            await asyncio.wait_for(
                asyncio.sleep(self.config.external.operator_override_timeout_s),
                timeout=self.config.external.operator_override_timeout_s + 1.0,
            )
        except asyncio.TimeoutError:
            pass
        await trace.record(
            actor_id=self.actor_id,
            event_type="yellow_tier_wait_end",
            payload={"target_id": target.target_id},
        )

    async def _halt_and_await_operator(self, target: ResearchTarget) -> None:
        """Red tier: emit a halting event and block until operator decides."""
        trace = await TraceStore.get()
        halt_event = Event(
            "OperatorHaltRequired",
            {
                "target_id": target.target_id,
                "source_type": target.source_type,
                "query": target.query,
                "estimated_cost": target.estimated_cost.__dict__,
                "reason": "Red tier threshold exceeded",
            },
        )
        await self.send_to_parent(halt_event)
        await trace.record(
            actor_id=self.actor_id,
            event_type="red_tier_halt",
            payload={"target_id": target.target_id},
        )

        # Block until we receive an OperatorDecision or Resume
        while not self._shutdown:
            try:
                decision = await asyncio.wait_for(self._operator_events.get(), timeout=2.0)
                if decision.event_type == "OperatorDecision":
                    if decision.payload.get("target_id") == target.target_id:
                        await trace.record(
                            actor_id=self.actor_id,
                            event_type="operator_decision_received",
                            payload=decision.payload,
                        )
                        break
                elif decision.event_type == "Resume":
                    self._paused = False
                    break
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # 5. Execution
    # ------------------------------------------------------------------

    async def _execute_target(self, target: ResearchTarget) -> tuple[list[dict[str, Any]] | None, FetchCost]:
        """Spawn a ``ToolActor``, run it, and return results with actual cost."""
        trace = await TraceStore.get()
        child_id = f"{self.actor_id}_tool_{target.target_id}_{uuid.uuid4().hex[:6]}"
        tool = ToolActor(child_id, target.source_type, target.query)
        self.spawn_child(tool)

        start = time.time()
        await trace.record(
            actor_id=self.actor_id,
            event_type="tool_spawned",
            payload={"child_id": child_id, "target_id": target.target_id},
        )

        # Wait for the child task to finish (ToolActor._run exits after one fetch)
        if tool._task:
            try:
                await asyncio.wait_for(tool._task, timeout=30.0)
            except asyncio.TimeoutError:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="tool_timeout",
                    payload={"child_id": child_id},
                )
                await tool.stop(graceful=False)
                return None, FetchCost(usd=0.0, latency_s=time.time() - start)

        latency = time.time() - start
        # Synthesise actual cost (placeholder: same as estimate)
        actual_cost = FetchCost(
            usd=target.estimated_cost.usd,
            tokens=target.estimated_cost.tokens,
            latency_s=latency,
            context_window_interest=target.estimated_cost.context_window_interest,
        )

        # Update UCB state
        state = self._ucb_state.setdefault(target.target_id, {"count": 0, "avg_reward": 0.0})
        reward = 1.0 if (tool.result and len(tool.result) > 0) else 0.0
        state["count"] += 1
        state["avg_reward"] += (reward - state["avg_reward"]) / state["count"]

        await trace.record(
            actor_id=self.actor_id,
            event_type="tool_result",
            payload={
                "child_id": child_id,
                "target_id": target.target_id,
                "result_count": len(tool.result) if tool.result else 0,
                "latency_s": round(latency, 3),
            },
        )

        if tool.error:
            await trace.record(
                actor_id=self.actor_id,
                event_type="tool_error_propagated",
                error=tool.error,
                payload={"child_id": child_id},
            )

        return tool.result, actual_cost

    # ------------------------------------------------------------------
    # 6. Store write
    # ------------------------------------------------------------------

    async def _write_finding(
        self,
        target: ResearchTarget,
        result: list[dict[str, Any]],
    ) -> int | None:
        """Write fetched findings to the store as ``row_type='finding'``.

        Returns the inserted row id, or ``None`` on failure.
        """
        trace = await TraceStore.get()
        # Flatten result snippets into a single fact text
        snippets = [r.get("snippet", str(r)) for r in result if isinstance(r, dict)]
        fact_text = "\n".join(snippets)[:4000]

        row = {
            "fact": fact_text,
            "row_type": "finding",
            "provenance_system": "mcp_research",
            "source_type": target.source_type,
            "query_type": target.reason_type,
            "confidence": target.confidence,
            "score_version": 1,
            "information_gain": target.benefit_score,
            "evaluation_count": 1,
        }

        try:
            row_id = await self.store.insert("conditions", row)
            await trace.record(
                actor_id=self.actor_id,
                event_type="finding_written",
                payload={"row_id": row_id, "target_id": target.target_id},
            )
            return row_id
        except Exception as exc:
            await trace.record(
                actor_id=self.actor_id,
                event_type="finding_write_failed",
                error=exc,
                payload={"target_id": target.target_id},
            )
            return None

    # ------------------------------------------------------------------
    # 7. Completion event
    # ------------------------------------------------------------------

    async def _emit_complete(self, findings_added: int, cost_usd: float, source_type: str) -> None:
        trace = await TraceStore.get()
        event = McpResearchComplete(
            findings_added=findings_added,
            cost_usd=cost_usd,
            source_type=source_type,
        )
        await self.send_to_parent(event)
        await trace.record(
            actor_id=self.actor_id,
            event_type="mcp_research_complete_emitted",
            payload={"findings_added": findings_added, "cost_usd": cost_usd, "source_type": source_type},
        )
