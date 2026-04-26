"""Flock evaluation — multi-angle clone supervision with prefix caching and convergence detection.

A Flock is a set of :class:`CloneActor` instances, each exploring a different
analytical angle.  The :class:`FlockSupervisor` coordinates rounds, detects
convergence, and allocates budgets via UCB with priority decay and a
serendipity floor.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any, Callable

from universal_store.actors.base import Actor
from universal_store.actors.supervisor import Supervisor
from universal_store.protocols import (
    BudgetOverride,
    Event,
    FlockComplete,
    FlockRoundComplete,
    QueryType,
)
from universal_store.trace import trace_block, TraceStore
from universal_store.config import UnifiedConfig


# ---------------------------------------------------------------------------
# Placeholder helpers — replace with real implementations
# ---------------------------------------------------------------------------

def build_clone_context_from_store(angle: str, condition_ids: list[int]) -> str:
    """Build a prefix-cached context string from store findings for this angle.

    Placeholder: in production this queries the universal store and formats
    findings into a vLLM-prefix-cachable context block.
    """
    return f"[Context for angle='{angle}' conditions={condition_ids}]"


def bootstrap_score_version() -> None:
    """Re-bootstrap the scoring model version for the current round.

    **CRITICAL:** must be called once per round, not once per session.
    Placeholder: in production this refreshes the scoring backend.
    """
    pass


# ---------------------------------------------------------------------------
# CloneActor — one per angle, maintains prefix-cached context
# ---------------------------------------------------------------------------

class CloneActor(Actor):
    """An actor that explores a single analytical angle within a Flock.

    Each :class:`CloneActor` maintains its own prefix-cached vLLM context and
    runs independent rounds of querying.  Results are emitted back to the
    :class:`FlockSupervisor` as :class:`FlockRoundComplete` events.
    """

    def __init__(
        self,
        actor_id: str,
        angle: str,
        condition_ids: list[int],
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.angle = angle
        self.condition_ids = condition_ids
        self.config = config or UnifiedConfig.from_env()
        self.round_num = 0
        self.query_fn: Callable[..., Any] | None = None  # injection point

        # Prefix-cache state
        self._context_cache: str | None = None
        self._last_store_hash: str = ""

        # Budget / pause state received from supervisor
        self._budget_override: BudgetOverride | None = None

    # -- public API ----------------------------------------------------------

    def set_query_fn(self, fn: Callable[..., Any]) -> None:
        """Inject the real query function (used by tests or wiring code)."""
        self.query_fn = fn

    # -- main loop -----------------------------------------------------------

    async def _run(self) -> None:
        """Run rounds until shutdown.

        Each iteration:

        1. Drain mailbox for budget overrides / pause signals.
        2. Build (or reuse) prefix-cached context.
        3. Bootstrap score version **per round**.
        4. Fire queries.
        5. Emit :class:`FlockRoundComplete` to parent supervisor.
        """
        while not self._shutdown:
            async with trace_block(
                self.actor_id,
                "clone_round",
                str(self.round_num),
                payload={"angle": self.angle, "round": self.round_num},
            ):
                # 1. Process control messages
                await self._drain_mailbox()

                # 2. Build context with prefix-cache preservation
                context = await self._build_context()

                # 3. Bootstrap score version — PER ROUND (critical bug fix)
                bootstrap_score_version()

                # 4. Execute queries for this round
                query_results = await self._execute_queries(context)

                # 5. Compute local convergence score
                convergence_score = self._compute_convergence_score(query_results)

                # 6. Emit round completion
                event = FlockRoundComplete(
                    round_num=self.round_num,
                    convergence_score=convergence_score,
                    directions=[self.angle],
                )
                await self.send_to_parent(event)

                trace = await self._ensure_trace()
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="clone_round_complete",
                    payload={
                        "round": self.round_num,
                        "angle": self.angle,
                        "convergence_score": convergence_score,
                        "result_count": len(query_results),
                    },
                )

                self.round_num += 1

                # Cooperative yield so the event loop stays responsive
                await asyncio.sleep(0.001)

    # -- mailbox handling -----------------------------------------------------

    async def _drain_mailbox(self) -> None:
        """Non-blocking consumption of control events."""
        while True:
            try:
                event = self.mailbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._handle_control_event(event)

    async def _handle_control_event(self, event: Event) -> None:
        """Apply budget overrides or pause signals from the supervisor."""
        trace = await self._ensure_trace()
        if event.event_type == "BudgetOverride":
            payload = event.payload.get("override")
            if isinstance(payload, BudgetOverride):
                self._budget_override = payload
            else:
                # Allow dict form as a fallback
                self._budget_override = BudgetOverride(
                    boost=event.payload.get("boost", {}),
                    pause=set(event.payload.get("pause", [])),
                    reason=event.payload.get("reason", ""),
                )
            await trace.record(
                actor_id=self.actor_id,
                event_type="clone_budget_override_applied",
                payload={"angle": self.angle, "override": event.payload},
            )
        elif event.event_type == "PauseClone":
            duration = event.payload.get("duration_s", 1.0)
            await asyncio.sleep(duration)

    # -- context building (prefix-cache aware) --------------------------------

    async def _build_context(self) -> str:
        """Return a context string, leveraging the prefix cache when possible."""
        trace = await self._ensure_trace()
        async with trace_block(
            self.actor_id,
            "build_context",
            str(self.round_num),
            payload={"angle": self.angle},
        ):
            current_hash = await self._compute_store_hash()
            if (
                self._context_cache is not None
                and current_hash == self._last_store_hash
            ):
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="context_cache_hit",
                    payload={"angle": self.angle, "round": self.round_num},
                )
                return self._context_cache

            # Cache miss or store changed — rebuild
            context = build_clone_context_from_store(self.angle, self.condition_ids)
            self._context_cache = context
            self._last_store_hash = current_hash

            await trace.record(
                actor_id=self.actor_id,
                event_type="context_cache_miss",
                payload={
                    "angle": self.angle,
                    "round": self.round_num,
                    "new_hash": current_hash,
                    "context_length": len(context),
                },
            )
            return context

    async def _compute_store_hash(self) -> str:
        """Placeholder fingerprint of the relevant store slice.

        In production this would run a lightweight COUNT / checksum query
        against the rows matching ``condition_ids`` so that the prefix cache
        is only invalidated when the store changes materially.
        """
        data = f"{self.condition_ids}:{self.angle}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    # -- query execution ------------------------------------------------------

    async def _execute_queries(self, context: str) -> list[dict[str, Any]]:
        """Fire one or more queries for this round and return raw results."""
        query_types = self._select_query_types()
        results: list[dict[str, Any]] = []
        for qt in query_types:
            result = await self._fire_query(qt, context)
            results.append(result)
        return results

    async def _fire_query(self, query_type: QueryType, context: str) -> dict[str, Any]:
        """Placeholder query execution.

        In the real implementation this delegates to
        ``flock_query_manager.query()``.  We do **not** import that module
        here to keep the dependency graph clean.
        """
        trace = await self._ensure_trace()
        async with trace_block(
            self.actor_id,
            "fire_query",
            f"{query_type.value}",
            payload={"angle": self.angle, "round": self.round_num},
        ):
            # Placeholder result shape
            result: dict[str, Any] = {
                "query_type": query_type.value,
                "score": 0.0,
                "result": {},
                "tokens_used": len(context) // 4,
            }

            if self.query_fn is not None:
                # If a real function was injected (e.g. in tests), call it
                try:
                    result = await self.query_fn(query_type, context)
                except Exception as exc:
                    await trace.record(
                        actor_id=self.actor_id,
                        event_type="query_fn_error",
                        error=exc,
                        payload={"query_type": query_type.value},
                    )
                    raise

            await trace.record(
                actor_id=self.actor_id,
                event_type="query_fired",
                payload={
                    "query_type": query_type.value,
                    "round": self.round_num,
                    "angle": self.angle,
                },
            )
            return result

    def _select_query_types(self) -> list[QueryType]:
        """Choose which query types to run this round.

        Respects budget override pauses and applies angle-specific heuristics.
        """
        # Base set — in production this would be angle-dependent
        base_types = [QueryType.SYNTHESIZE, QueryType.VALIDATE]

        if self._budget_override:
            base_types = [
                qt for qt in base_types
                if qt.value not in self._budget_override.pause
            ]
            # Apply boosts by repeating high-priority types
            for qt_str, boost in self._budget_override.boost.items():
                try:
                    qt = QueryType(qt_str)
                    repeats = max(1, int(boost * 2))
                    base_types.extend([qt] * repeats)
                except ValueError:
                    continue

        return base_types

    def _compute_convergence_score(self, results: list[dict[str, Any]]) -> float:
        """Derive a scalar convergence score from query results."""
        if not results:
            return 0.0
        scores = [r.get("score", 0.0) for r in results]
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# FlockSupervisor — manages clones, detects convergence, allocates budget
# ---------------------------------------------------------------------------

class FlockSupervisor(Supervisor):
    """Supervisor for a Flock of :class:`CloneActor` instances.

    Responsibilities:

    1. Spawn one :class:`CloneActor` per analytical angle.
    2. Aggregate :class:`FlockRoundComplete` events.
    3. Detect convergence (``avg_magnitude < threshold`` for 2 rounds).
    4. Allocate budgets via UCB with priority decay and serendipity floor.
    5. Emit :class:`FlockComplete` when converged.
    """

    def __init__(
        self,
        actor_id: str,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.config = config or UnifiedConfig.from_env()
        self._clone_angles: list[str] = []
        self._condition_ids: list[int] = []
        self._round_reports: dict[int, list[Event]] = {}
        self._convergence_history: list[float] = []
        self._consecutive_converged = 0
        # UCB bookkeeping
        self._clone_scores: dict[str, list[float]] = {}
        self._clone_plays: dict[str, int] = {}
        self._flock_started = False

    # -- lifecycle ------------------------------------------------------------

    def start_flock(self, condition_ids: list[int], angles: list[str]) -> None:
        """Spawn a :class:`CloneActor` for every angle.

        Args:
            condition_ids: Row IDs in the store that define the problem space.
            angles: Analytical angles (one clone per angle).
        """
        self._clone_angles = angles
        self._condition_ids = condition_ids

        for angle in angles:
            clone_id = self._clone_id(angle)
            # Register factory so the supervisor can restart crashed clones
            self.register_child(
                clone_id,
                lambda a=angle, cid=condition_ids, cfg=self.config: CloneActor(
                    self._clone_id(a), a, cid, cfg
                ),
            )
            # Spawn immediately
            clone = CloneActor(clone_id, angle, condition_ids, self.config)
            self.spawn_child(clone)

            # Init UCB state
            self._clone_scores[angle] = []
            self._clone_plays[angle] = 0

        self._flock_started = True

    @staticmethod
    def _clone_id(angle: str) -> str:
        """Sanitise an angle into a valid actor ID."""
        safe = angle.replace(" ", "_").replace("/", "_").lower()
        return f"clone-{safe}"

    # -- event routing --------------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        """Route Flock events; delegate everything else to the base supervisor."""
        async with trace_block(
            self.actor_id,
            "handle_event",
            event.event_type,
            payload={"source": event.source_actor},
        ):
            if event.event_type == "FlockRoundComplete":
                await self._on_round_complete(event)
            else:
                await super()._handle_event(event)

    async def _on_round_complete(self, event: Event) -> None:
        """Record a clone's round result and check for round-level convergence."""
        trace = await TraceStore.get()
        round_num: int = event.payload.get("round", 0)
        angle: str = (event.payload.get("directions") or [""])[0]
        score: float = event.payload.get("convergence_score", 0.0)

        await trace.record(
            actor_id=self.actor_id,
            event_type="flock_round_report",
            payload={
                "round": round_num,
                "angle": angle,
                "score": score,
                "source_actor": event.source_actor,
            },
        )

        # Store report
        self._round_reports.setdefault(round_num, []).append(event)

        # Update per-clone stats for UCB
        self._clone_scores[angle].append(score)
        self._clone_plays[angle] += 1

        # When every clone has reported for this round, evaluate convergence
        reports = self._round_reports[round_num]
        if len(reports) >= len(self._clone_angles):
            async with trace_block(
                self.actor_id,
                "convergence_check",
                str(round_num),
            ):
                await self._evaluate_round(round_num)

    # -- convergence ----------------------------------------------------------

    async def _evaluate_round(self, round_num: int) -> None:
        """Check whether the flock has converged after a completed round."""
        trace = await TraceStore.get()
        reports = self._round_reports[round_num]
        scores = [
            r.payload.get("convergence_score", 0.0) for r in reports
        ]

        avg_magnitude = sum(abs(s) for s in scores) / len(scores) if scores else 1.0
        self._convergence_history.append(avg_magnitude)

        threshold = self.config.flock.magnitude_convergence_threshold

        await trace.record(
            actor_id=self.actor_id,
            event_type="convergence_check",
            payload={
                "round": round_num,
                "avg_magnitude": avg_magnitude,
                "threshold": threshold,
                "scores": scores,
            },
        )

        if avg_magnitude < threshold:
            self._consecutive_converged += 1
            await trace.record(
                actor_id=self.actor_id,
                event_type="convergence_round_passed",
                payload={
                    "round": round_num,
                    "consecutive_converged": self._consecutive_converged,
                },
            )
        else:
            self._consecutive_converged = 0
            await trace.record(
                actor_id=self.actor_id,
                event_type="convergence_round_failed",
                payload={"round": round_num, "avg_magnitude": avg_magnitude},
            )

        if self._consecutive_converged >= 2:
            directions = [
                (r.payload.get("directions") or [""])[0] for r in reports
            ]
            completion = FlockComplete(
                convergence_reason=(
                    f"avg_magnitude < {threshold} for 2 consecutive rounds"
                ),
                directions=directions,
            )
            await self.send_to_parent(completion)
            await trace.record(
                actor_id=self.actor_id,
                event_type="flock_converged",
                payload={
                    "round": round_num,
                    "avg_magnitude": avg_magnitude,
                    "directions": directions,
                },
            )
            return

        # Not converged yet — allocate budgets for the next round
        await self._allocate_budgets(round_num + 1)

    # -- budget allocation (UCB + decay + floor) ------------------------------

    async def _allocate_budgets(self, round_num: int) -> None:
        """Compute UCB-based budget allocation and push to each clone."""
        trace = await TraceStore.get()
        total_clones = len(self._clone_angles)
        if total_clones == 0:
            return

        ucb_values: dict[str, float] = {}
        log_total = math.log(max(round_num, 1))

        for angle in self._clone_angles:
            plays = self._clone_plays.get(angle, 0)
            scores = self._clone_scores.get(angle, [])
            if plays == 0 or not scores:
                # Unexplored clones get infinite priority so they are tried
                ucb_values[angle] = float("inf")
            else:
                avg_score = sum(scores) / len(scores)
                exploration = math.sqrt(2 * log_total / plays)
                ucb_values[angle] = (
                    avg_score + self.config.flock.ucb_alpha * exploration
                )

        # Normalise to [0, 1] after applying decay and floor
        max_ucb = max(ucb_values.values()) if ucb_values else 1.0
        if max_ucb == 0 or math.isinf(max_ucb):
            max_ucb = 1.0

        allocations: dict[str, float] = {}
        for angle in self._clone_angles:
            raw = ucb_values[angle]
            if math.isinf(raw):
                normalised = 1.0
            else:
                normalised = raw / max_ucb

            # Priority decay: older angles lose priority over time
            decayed = normalised * (
                (1 - self.config.flock.priority_decay_rate) ** round_num
            )

            # Serendipity floor: every angle receives a minimum share
            floor = self.config.flock.serendipity_floor / total_clones
            allocations[angle] = max(floor, decayed)

        # Push allocations to clones
        for angle, alloc in allocations.items():
            clone_id = self._clone_id(angle)
            clone = self._children.get(clone_id)
            if clone is None:
                continue

            override = BudgetOverride(
                boost={angle: alloc},
                reason=f"UCB allocation round {round_num}",
            )
            await clone.send(
                Event(
                    "BudgetOverride",
                    {"override": override, "allocation": alloc},
                )
            )

        await trace.record(
            actor_id=self.actor_id,
            event_type="budget_allocated",
            payload={"round": round_num, "allocations": allocations},
        )

    # -- health ---------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Return flock health including convergence state."""
        base = await super().health()
        base.update(
            {
                "flock_started": self._flock_started,
                "clone_count": len(self._clone_angles),
                "rounds_completed": len(self._convergence_history),
                "convergence_history": self._convergence_history[-5:],
                "consecutive_converged": self._consecutive_converged,
            }
        )
        return base
