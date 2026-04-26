"""Curation actors — continuously transform raw store rows into actionable consumables.

The :class:`CurationSupervisor` manages eight specialized curator actors:

1. :class:`GlobalHealthCurator` — continuous, emits ``GlobalHealthSnapshot``
2. :class:`CloneContextCurator` — on-demand + cached, emits ``CloneContextCache``
3. :class:`AngleContextCurator` — on-demand + cached, emits ``AngleContextBundle``
4. :class:`ContradictionCurator` — continuous, emits ``ContradictionDigest``
5. :class:`GapCurator` — continuous, emits ``GapDigest``
6. :class:`NarrativeCurator` — continuous, emits ``OperatorBriefing``
7. :class:`LessonCurator` — event-driven, emits ``CuratorDigest``
8. :class:`SourceQualityCurator` — periodic (every *N* rounds), emits source quality report

All curators inherit from :class:`Actor` and offload blocking store work to
``asyncio.to_thread`` when the concrete store does not natively implement
coroutine-based ``StoreProtocol`` methods.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

from universal_store.actors.base import Actor
from universal_store.actors.supervisor import Supervisor
from universal_store.protocols import (
    AngleContextBundle,
    CloneContextCache,
    ContradictionDigest,
    CuratorDigest,
    Event,
    GapDigest,
    GlobalHealthSnapshot,
    OperatorBriefing,
    StoreProtocol,
)
from universal_store.config import UnifiedConfig
from universal_store.trace import trace_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _store_query(store: StoreProtocol, sql: str, params: tuple | None = None) -> list[dict]:
    """Invoke ``store.query`` without blocking the event loop.

    If the store implements ``query`` as a coroutine function we ``await`` it
    directly; otherwise we defer to ``asyncio.to_thread``.
    """
    if inspect.iscoroutinefunction(store.query):
        return await store.query(sql, params)
    return await asyncio.to_thread(store.query, sql, params)


async def _store_execute(store: StoreProtocol, sql: str, params: tuple | None = None) -> list[dict]:
    """Invoke ``store.execute`` without blocking the event loop."""
    if inspect.iscoroutinefunction(store.execute):
        return await store.execute(sql, params)
    return await asyncio.to_thread(store.execute, sql, params)


def _as_dict(row: dict | tuple, columns: list[str] | None = None) -> dict[str, Any]:
    """Normalise a store result row to a ``dict``."""
    if isinstance(row, dict):
        return dict(row)
    if columns is not None:
        return {col: val for col, val in zip(columns, row)}
    raise ValueError("Cannot normalise row to dict without column names")


# ---------------------------------------------------------------------------
# Base curator mixin
# ---------------------------------------------------------------------------

class _CuratorActor(Actor):
    """Minimal base that injects ``store`` and ``config`` into every curator."""

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self._store = store
        self._config = config or UnifiedConfig()


# ---------------------------------------------------------------------------
# 1. GlobalHealthCurator
# ---------------------------------------------------------------------------

class GlobalHealthCurator(_CuratorActor):
    """Continuous curator that emits a :class:`GlobalHealthSnapshot` every *N* seconds."""

    async def _run(self) -> None:
        interval = self._config.curation.global_health_interval_s
        await asyncio.sleep(interval)

        async with trace_block(
            self.actor_id, "global_health_cycle", phase="curation", payload={"interval_s": interval}
        ):
            total_rows = await _store_query(self._store, "SELECT COUNT(*) AS cnt FROM conditions")
            finding_count = await _store_query(
                self._store, "SELECT COUNT(*) AS cnt FROM conditions WHERE row_type = 'finding'"
            )
            contradiction_count = await _store_query(
                self._store,
                "SELECT COUNT(*) AS cnt FROM conditions WHERE contradiction_flag = TRUE",
            )
            gap_count = await _store_query(
                self._store,
                "SELECT COUNT(*) AS cnt FROM conditions WHERE row_type = 'research_target' OR expansion_gap != ''",
            )

            blockers = await _store_query(
                self._store,
                """
                SELECT fact, confidence, contradiction_flag, expansion_gap
                FROM conditions
                WHERE consider_for_use = TRUE
                  AND (contradiction_flag = TRUE OR expansion_gap != '' OR confidence < 0.3)
                ORDER BY confidence ASC, contradiction_flag DESC
                LIMIT 5
                """,
            )

            top_blockers: list[str] = []
            for b in blockers:
                d = _as_dict(b, ["fact", "confidence", "contradiction_flag", "expansion_gap"])
                prefix = "[CONTRADICTS] " if d.get("contradiction_flag") else ""
                top_blockers.append(
                    f"{prefix}{d.get('fact', '')[:120]} (conf={d.get('confidence', 0):.2f})"
                )

            total = total_rows[0].get("cnt", 0) if total_rows else 0
            findings = finding_count[0].get("cnt", 0) if finding_count else 0
            contradictions = contradiction_count[0].get("cnt", 0) if contradiction_count else 0
            gaps = gap_count[0].get("cnt", 0) if gap_count else 0

            snapshot = GlobalHealthSnapshot(
                total_rows=total,
                finding_count=findings,
                contradiction_count=contradictions,
                gap_count=gaps,
                convergence_trend=[],
                top_blockers=top_blockers,
                recommended_phase="SWARMING" if gaps > contradictions else "SYNTHESIZING",
            )

            await self.send_to_parent(
                Event("GlobalHealthSnapshot", {"snapshot": snapshot.__dict__})
            )


# ---------------------------------------------------------------------------
# 2. CloneContextCurator
# ---------------------------------------------------------------------------

class CloneContextCurator(_CuratorActor):
    """On-demand curator that builds tiered clone context for a given angle.

    Triggered by ``BuildCloneContext`` events.  Results are cached in-memory
    keyed by angle.
    """

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id, store, config)
        self._cache: dict[str, CloneContextCache] = {}

    async def _run(self) -> None:
        try:
            event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return

        if event.event_type != "BuildCloneContext":
            return

        angle = event.payload.get("angle", "")
        if not angle:
            return

        async with trace_block(
            self.actor_id,
            "clone_context_cycle",
            phase="curation",
            payload={"angle": angle},
        ):
            # Cache hit — return immediately
            if angle in self._cache:
                await self.send_to_parent(
                    Event("CloneContextCache", {"cache": self._cache[angle].__dict__})
                )
                return

            max_items = self._config.curation.clone_context_max_items

            rows = await _store_query(
                self._store,
                """
                SELECT id, fact, confidence, contradiction_flag, verification_status,
                       created_at, row_type
                FROM conditions
                WHERE angle = ? AND consider_for_use = TRUE
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (angle, max_items * 2),
            )

            established: list[dict] = []
            controversial: list[dict] = []
            bridge_worthy: list[dict] = []
            recent: list[dict] = []

            for row in rows:
                d = _as_dict(
                    row,
                    [
                        "id",
                        "fact",
                        "confidence",
                        "contradiction_flag",
                        "verification_status",
                        "created_at",
                        "row_type",
                    ],
                )
                if d.get("contradiction_flag"):
                    controversial.append(d)
                elif d.get("verification_status") == "verified" and d.get("confidence", 0) > 0.7:
                    established.append(d)
                elif d.get("confidence", 0) > 0.4:
                    bridge_worthy.append(d)
                else:
                    recent.append(d)

            # Assemble readable context text
            parts: list[str] = []
            limit_per_bucket = max_items // 4

            if established:
                parts.append("=== ESTABLISHED ===")
                for item in established[:limit_per_bucket]:
                    parts.append(f"- {item.get('fact', '')}")
            if controversial:
                parts.append("=== CONTROVERSIAL ===")
                for item in controversial[:limit_per_bucket]:
                    parts.append(f"- {item.get('fact', '')}")
            if bridge_worthy:
                parts.append("=== BRIDGE-WORTHY ===")
                for item in bridge_worthy[:limit_per_bucket]:
                    parts.append(f"- {item.get('fact', '')}")
            if recent:
                parts.append("=== RECENT ===")
                for item in recent[:limit_per_bucket]:
                    parts.append(f"- {item.get('fact', '')}")

            context_text = "\n".join(parts)
            token_count = len(context_text.split())

            cache_entry = CloneContextCache(
                angle=angle,
                context_text=context_text,
                token_count=token_count,
                item_count=len(rows),
            )
            self._cache[angle] = cache_entry

            await self.send_to_parent(
                Event("CloneContextCache", {"cache": cache_entry.__dict__})
            )


# ---------------------------------------------------------------------------
# 3. AngleContextCurator
# ---------------------------------------------------------------------------

class AngleContextCurator(_CuratorActor):
    """On-demand curator that bundles swarm-bee context, capped at 100 items.

    Triggered by ``BuildAngleContext`` events.
    """

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id, store, config)
        self._cache: dict[str, AngleContextBundle] = {}

    async def _run(self) -> None:
        try:
            event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return

        if event.event_type != "BuildAngleContext":
            return

        angle = event.payload.get("angle", "")
        if not angle:
            return

        async with trace_block(
            self.actor_id,
            "angle_context_cycle",
            phase="curation",
            payload={"angle": angle},
        ):
            if angle in self._cache:
                bundle = self._cache[angle]
                await self.send_to_parent(
                    Event("AngleContextBundle", {"bundle": bundle.__dict__})
                )
                return

            cap = self._config.curation.angle_bundle_max_items

            rows = await _store_query(
                self._store,
                """
                SELECT id, fact, confidence, contradiction_flag, verification_status,
                       created_at, row_type, angle
                FROM conditions
                WHERE angle = ? AND consider_for_use = TRUE
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (angle, cap),
            )

            established: list[dict] = []
            controversial: list[dict] = []
            bridge_worthy: list[dict] = []
            recent: list[dict] = []

            for row in rows:
                d = _as_dict(
                    row,
                    [
                        "id",
                        "fact",
                        "confidence",
                        "contradiction_flag",
                        "verification_status",
                        "created_at",
                        "row_type",
                        "angle",
                    ],
                )
                if d.get("contradiction_flag"):
                    controversial.append(d)
                elif d.get("verification_status") == "verified" and d.get("confidence", 0) > 0.7:
                    established.append(d)
                elif d.get("confidence", 0) > 0.4:
                    bridge_worthy.append(d)
                else:
                    recent.append(d)

            bundle = AngleContextBundle(
                established=established[:25],
                controversial=controversial[:25],
                bridge_worthy=bridge_worthy[:25],
                recent=recent[:25],
            )
            self._cache[angle] = bundle

            await self.send_to_parent(
                Event("AngleContextBundle", {"bundle": bundle.__dict__})
            )


# ---------------------------------------------------------------------------
# 4. ContradictionCurator
# ---------------------------------------------------------------------------

class ContradictionCurator(_CuratorActor):
    """Continuous curator that ranks contradictions by confidence delta."""

    async def _run(self) -> None:
        interval = self._config.curation.contradiction_digest_interval_s
        await asyncio.sleep(interval)

        async with trace_block(
            self.actor_id,
            "contradiction_cycle",
            phase="curation",
            payload={"interval_s": interval},
        ):
            rows = await _store_query(
                self._store,
                """
                SELECT id, fact, confidence, contradiction_partner,
                       staleness_penalty, created_at
                FROM conditions
                WHERE contradiction_flag = TRUE AND consider_for_use = TRUE
                ORDER BY confidence DESC
                """,
            )

            contradictions: list[dict] = []
            stale_count = 0
            for row in rows:
                d = _as_dict(
                    row,
                    [
                        "id",
                        "fact",
                        "confidence",
                        "contradiction_partner",
                        "staleness_penalty",
                        "created_at",
                    ],
                )
                if d.get("staleness_penalty", 0) > 0.3:
                    stale_count += 1
                contradictions.append(d)

            priority_queue = sorted(
                contradictions,
                key=lambda x: x.get("confidence", 0.5)
                * (1 - x.get("staleness_penalty", 0)),
                reverse=True,
            )

            digest = ContradictionDigest(
                contradictions=contradictions[:50],
                stale_count=stale_count,
                priority_queue=priority_queue[:20],
            )

            await self.send_to_parent(
                Event("ContradictionDigest", {"digest": digest.__dict__})
            )


# ---------------------------------------------------------------------------
# 5. GapCurator
# ---------------------------------------------------------------------------

class GapCurator(_CuratorActor):
    """Continuous curator that emits an MCP-prioritised gap digest."""

    async def _run(self) -> None:
        interval = self._config.curation.gap_digest_interval_s
        await asyncio.sleep(interval)

        async with trace_block(
            self.actor_id,
            "gap_cycle",
            phase="curation",
            payload={"interval_s": interval},
        ):
            rows = await _store_query(
                self._store,
                """
                SELECT id, fact, expansion_gap, expansion_priority, angle,
                       confidence, created_at
                FROM conditions
                WHERE (row_type = 'research_target' OR expansion_gap != '')
                  AND consider_for_use = TRUE
                ORDER BY expansion_priority DESC
                """,
            )

            gaps = [
                _as_dict(
                    row,
                    [
                        "id",
                        "fact",
                        "expansion_gap",
                        "expansion_priority",
                        "angle",
                        "confidence",
                        "created_at",
                    ],
                )
                for row in rows
            ]

            mcp_priority_queue = sorted(
                gaps,
                key=lambda x: x.get("expansion_priority", 0) * x.get("confidence", 0.5),
                reverse=True,
            )

            digest = GapDigest(
                gaps=gaps[:100],
                mcp_priority_queue=mcp_priority_queue[:20],
            )

            await self.send_to_parent(
                Event("GapDigest", {"digest": digest.__dict__})
            )


# ---------------------------------------------------------------------------
# 6. NarrativeCurator
# ---------------------------------------------------------------------------

class NarrativeCurator(_CuratorActor):
    """Continuous curator that assembles a human-readable operator briefing."""

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id, store, config)
        self._cycle_count = 0

    async def _run(self) -> None:
        interval = self._config.curation.narrative_interval_s
        await asyncio.sleep(interval)
        self._cycle_count += 1

        async with trace_block(
            self.actor_id,
            "narrative_cycle",
            phase="curation",
            payload={"cycle": self._cycle_count},
        ):
            finding_res = await _store_query(
                self._store,
                "SELECT COUNT(*) AS cnt FROM conditions WHERE row_type = 'finding'",
            )
            resolved_res = await _store_query(
                self._store,
                "SELECT COUNT(*) AS cnt FROM conditions WHERE contradiction_flag = TRUE AND verification_status = 'resolved'",
            )
            bridge_res = await _store_query(
                self._store,
                "SELECT COUNT(*) AS cnt FROM semantic_connections WHERE connection_type = 'bridge'",
            )

            findings = finding_res[0].get("cnt", 0) if finding_res else 0
            resolved = resolved_res[0].get("cnt", 0) if resolved_res else 0
            bridges = bridge_res[0].get("cnt", 0) if bridge_res else 0

            narrative = (
                f"Round {self._cycle_count}: {findings} findings, "
                f"{resolved} contradictions resolved, {bridges} novel bridges. "
            )

            recommendations: list[str] = []
            if resolved > 0:
                recommendations.append("Continue contradiction resolution.")
            if bridges < 3:
                recommendations.append("Focus on bridge-building between angles.")
            if findings > 50 and bridges == 0:
                recommendations.append(
                    "High finding count but no bridges — investigate semantic connections."
                )

            recommended = (
                recommendations[0]
                if recommendations
                else "Maintain current strategy."
            )
            narrative += f"Recommended: {recommended}"

            alerts: list[str] = []
            if resolved > 5:
                alerts.append(f"{resolved} active contradictions require attention.")
            if findings > 100 and bridges < 2:
                alerts.append("Corpus is large but poorly connected.")

            briefing = OperatorBriefing(
                narrative=narrative,
                alerts=alerts[: self._config.curation.operator_briefing_max_alerts],
                decisions_required=recommendations[
                    : self._config.curation.operator_briefing_max_decisions
                ],
            )

            await self.send_to_parent(
                Event("OperatorBriefing", {"briefing": briefing.__dict__})
            )


# ---------------------------------------------------------------------------
# 7. LessonCurator
# ---------------------------------------------------------------------------

class LessonCurator(_CuratorActor):
    """Event-driven curator that validates and emits a digest on ``LessonRecorded``."""

    async def _run(self) -> None:
        try:
            event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return

        if event.event_type != "LessonRecorded":
            return

        lesson_id = event.payload.get("lesson_id")
        lesson_type = event.payload.get("lesson_type", "")
        fact = event.payload.get("fact", "")

        async with trace_block(
            self.actor_id,
            "lesson_cycle",
            phase="curation",
            payload={"lesson_id": lesson_id, "lesson_type": lesson_type},
        ):
            rows = await _store_query(
                self._store,
                "SELECT id, confidence, relevance_score FROM lessons WHERE id = ?",
                (lesson_id,),
            )

            valid = False
            if rows:
                d = _as_dict(row=rows[0], columns=["id", "confidence", "relevance_score"])
                if d.get("confidence", 0) >= self._config.reflexion.min_lesson_confidence:
                    valid = True

            digest = CuratorDigest(
                curator_name=self.actor_id,
                digest_type="lesson",
                data={
                    "lesson_id": lesson_id,
                    "lesson_type": lesson_type,
                    "fact": fact,
                    "validated": valid,
                },
            )

            await self.send_to_parent(
                Event("CuratorDigest", {"digest": digest.__dict__})
            )


# ---------------------------------------------------------------------------
# 8. SourceQualityCurator
# ---------------------------------------------------------------------------

class SourceQualityCurator(_CuratorActor):
    """Periodic curator that emits a source quality report every *N* rounds.

    Listens for ``GossipRoundComplete``, ``FlockRoundComplete`` and
    ``SwarmPhaseComplete`` events to count rounds.
    """

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id, store, config)
        self._round_counter = 0

    async def _run(self) -> None:
        try:
            event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
        except asyncio.TimeoutError:
            return

        if event.event_type not in (
            "GossipRoundComplete",
            "FlockRoundComplete",
            "SwarmPhaseComplete",
        ):
            return

        self._round_counter += 1
        interval = self._config.curation.source_quality_interval_rounds
        if self._round_counter < interval:
            return

        self._round_counter = 0

        async with trace_block(
            self.actor_id,
            "source_quality_cycle",
            phase="curation",
            payload={"interval_rounds": interval},
        ):
            rows = await _store_query(
                self._store,
                """
                SELECT domain, source_type, authority_score, avg_recency_score,
                       avg_finding_confidence, fetch_count, successful_fetch_count,
                       total_cost_usd, total_info_gain_generated
                FROM source_quality_registry
                ORDER BY authority_score DESC
                """,
            )

            sources = [
                _as_dict(
                    row,
                    [
                        "domain",
                        "source_type",
                        "authority_score",
                        "avg_recency_score",
                        "avg_finding_confidence",
                        "fetch_count",
                        "successful_fetch_count",
                        "total_cost_usd",
                        "total_info_gain_generated",
                    ],
                )
                for row in rows
            ]

            avg_authority = (
                sum(s.get("authority_score", 0) for s in sources) / max(len(sources), 1)
            )

            report = {
                "sources": sources,
                "total_domains": len(sources),
                "avg_authority": round(avg_authority, 3),
            }

            digest = CuratorDigest(
                curator_name=self.actor_id,
                digest_type="source_quality",
                data=report,
            )

            await self.send_to_parent(
                Event("CuratorDigest", {"digest": digest.__dict__})
            )


# ---------------------------------------------------------------------------
# CurationSupervisor
# ---------------------------------------------------------------------------

class CurationSupervisor(Supervisor):
    """Supervises the eight specialised curators in the curation layer.

    The supervisor registers factory functions for each child so that crashed
    curators can be restarted automatically.  It also routes domain-specific
    events to the appropriate curator mailboxes.
    """

    def __init__(
        self,
        actor_id: str = "curation_supervisor",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
        strategy: str = "restart",
        max_restarts: int = 3,
        restart_window_s: float = 60.0,
    ):
        super().__init__(
            actor_id,
            strategy=strategy,
            max_restarts=max_restarts,
            restart_window_s=restart_window_s,
        )
        self._store = store
        self._config = config or UnifiedConfig()
        self._register_curators()

    def _register_curators(self) -> None:
        """Register the eight curator factories."""
        self.register_child(
            "global_health",
            lambda: GlobalHealthCurator("global_health", self._store, self._config),
        )
        self.register_child(
            "clone_context",
            lambda: CloneContextCurator("clone_context", self._store, self._config),
        )
        self.register_child(
            "angle_context",
            lambda: AngleContextCurator("angle_context", self._store, self._config),
        )
        self.register_child(
            "contradiction",
            lambda: ContradictionCurator("contradiction", self._store, self._config),
        )
        self.register_child(
            "gap", lambda: GapCurator("gap", self._store, self._config)
        )
        self.register_child(
            "narrative",
            lambda: NarrativeCurator("narrative", self._store, self._config),
        )
        self.register_child(
            "lesson", lambda: LessonCurator("lesson", self._store, self._config)
        )
        self.register_child(
            "source_quality",
            lambda: SourceQualityCurator("source_quality", self._store, self._config),
        )

    async def _handle_event(self, event: Event) -> None:
        """Route incoming events to the correct curator mailbox."""
        if event.event_type == "BuildCloneContext":
            child = self._children.get("clone_context")
            if child:
                await child.send(event)
        elif event.event_type == "BuildAngleContext":
            child = self._children.get("angle_context")
            if child:
                await child.send(event)
        elif event.event_type == "LessonRecorded":
            child = self._children.get("lesson")
            if child:
                await child.send(event)
        elif event.event_type in (
            "GossipRoundComplete",
            "FlockRoundComplete",
            "SwarmPhaseComplete",
        ):
            child = self._children.get("source_quality")
            if child:
                await child.send(event)
            child = self._children.get("narrative")
            if child:
                await child.send(event)
        else:
            # Broadcast generic events to all children
            await self.broadcast_to_children(event)
