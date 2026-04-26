"""Reflexion actor — analyzes post-phase metrics, detects anomalies,
stores typed lessons, and makes them available for future runs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from universal_store.actors.base import Actor
from universal_store.config import UnifiedConfig
from universal_store.protocols import (
    Event,
    FlockComplete,
    LessonRecorded,
    LessonType,
    McpResearchComplete,
    ReflexionState,
    SwarmComplete,
)
from universal_store.trace import TraceStore, trace_block


class LessonStore:
    """Helper class for lesson CRUD. Not an actor.

    Wraps the ``lessons`` and ``lesson_applications`` tables with
    non-blocking async operations.
    """

    def __init__(self, db_path: str = "") -> None:
        cfg = UnifiedConfig.from_env()
        self.db_path = db_path or cfg.store.db_path
        self._conn: Any | None = None
        self._lock = asyncio.Lock()

    async def _ensure_conn(self) -> Any:
        """Lazy-init DuckDB connection and ensure tables exist."""
        if self._conn is None:
            import duckdb

            self._conn = duckdb.connect(self.db_path)
            from universal_store.schema import (
                LESSON_APPLICATIONS_TABLE,
                LESSONS_TABLE,
            )

            self._conn.execute(LESSONS_TABLE)
            self._conn.execute(LESSON_APPLICATIONS_TABLE)
        return self._conn

    async def record(self, lesson: dict) -> int:
        """Insert a lesson into the ``lessons`` table and return its ``id``.

        Args:
            lesson: Dict matching the ``lessons`` schema.

        Returns:
            The generated ``lesson_id``.
        """
        async with trace_block(
            "LessonStore",
            "lesson_record",
            payload={"lesson_type": lesson.get("lesson_type")},
        ):
            conn = await self._ensure_conn()
            cols = [
                "lesson_type",
                "fact",
                "run_id",
                "run_number",
                "angle",
                "query_type",
                "source_url",
                "source_type",
                "relevance_score",
                "confidence",
                "metadata",
                "halflife_runs",
            ]
            placeholders = ", ".join(["?"] * len(cols))
            values: list[Any] = []
            for c in cols:
                if c == "metadata":
                    values.append(json.dumps(lesson.get(c, {})))
                else:
                    values.append(lesson.get(c, "" if c != "run_number" else 0))

            row = conn.execute(
                f"INSERT INTO lessons ({', '.join(cols)}) "
                f"VALUES ({placeholders}) RETURNING id",
                values,
            ).fetchone()
            return row[0]

    async def query(
        self,
        lesson_type: str | None = None,
        angle: str | None = None,
        since: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Query lessons with optional filters.

        Args:
            lesson_type: Filter by ``LessonType`` value.
            angle: Filter by angle string.
            since: ISO timestamp lower bound (inclusive).
            min_confidence: Minimum confidence score.
            limit: Max rows to return.

        Returns:
            List of lesson dicts ordered by ``created_at DESC``.
        """
        async with trace_block(
            "LessonStore",
            "lesson_query",
            payload={
                "lesson_type": lesson_type,
                "angle": angle,
                "min_confidence": min_confidence,
                "limit": limit,
            },
        ):
            conn = await self._ensure_conn()
            conditions = ["1=1"]
            params: list[Any] = []
            if lesson_type:
                conditions.append("lesson_type = ?")
                params.append(lesson_type)
            if angle:
                conditions.append("angle = ?")
                params.append(angle)
            if since:
                conditions.append("created_at >= ?")
                params.append(since)
            if min_confidence > 0.0:
                conditions.append("confidence >= ?")
                params.append(min_confidence)

            sql = (
                f"SELECT * FROM lessons WHERE {' AND '.join(conditions)} "
                "ORDER BY created_at DESC LIMIT ?"
            )
            params.append(limit)
            df = conn.execute(sql, params).fetchdf()
            return df.to_dict("records")

    async def apply_to_run(
        self, lesson_id: int, target_run_id: str, method: str = ""
    ) -> int:
        """Record that a lesson was applied to a specific run.

        Args:
            lesson_id: The lesson to apply.
            target_run_id: The run receiving the lesson.
            method: How the lesson was applied (e.g. ``angle_boost``).

        Returns:
            The generated ``lesson_applications.id``.
        """
        async with trace_block(
            "LessonStore",
            "lesson_apply",
            payload={"lesson_id": lesson_id, "target_run_id": target_run_id, "method": method},
        ):
            conn = await self._ensure_conn()
            row = conn.execute(
                "INSERT INTO lesson_applications (lesson_id, target_run_id, application_method) "
                "VALUES (?, ?, ?) RETURNING id",
                [lesson_id, target_run_id, method],
            ).fetchone()
            return row[0]


class ReflexionActor(Actor):
    """Analyzes post-phase metrics, detects anomalies, stores typed lessons.

    The actor listens for completion events from Flock, Swarm, and MCP
    phases, converts anomalous patterns into typed ``Lesson`` rows, and
    maintains an in-memory ``ReflexionState`` that can be queried by the
    orchestrator before future runs.
    """

    def __init__(
        self,
        actor_id: str = "reflexion",
        config: UnifiedConfig | None = None,
    ) -> None:
        super().__init__(actor_id)
        self.config = config or UnifiedConfig.from_env()
        self.lesson_store = LessonStore(self.config.store.db_path)
        self.state = ReflexionState()

    async def _run(self) -> None:
        """Main loop: read completion events, detect lessons, emit recordings."""
        trace = await self._ensure_trace()
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if event.event_type not in (
                "FlockComplete",
                "SwarmComplete",
                "McpResearchComplete",
            ):
                continue

            async with trace_block(
                self.actor_id,
                "detect_lessons",
                event.event_type,
                payload={"source_actor": event.source_actor},
            ):
                lessons = await self.detect_lessons(event)

            for lesson in lessons:
                async with trace_block(
                    self.actor_id,
                    "store_lesson",
                    lesson.get("lesson_type", ""),
                    payload={"fact_preview": lesson.get("fact", "")[:200]},
                ):
                    lesson_id = await self.lesson_store.record(lesson)

                recorded = LessonRecorded(
                    lesson_id=lesson_id,
                    lesson_type=lesson["lesson_type"],
                    fact=lesson["fact"],
                )
                await self.send_to_parent(recorded)

            async with trace_block(
                self.actor_id,
                "update_reflexion_state",
                event.event_type,
                payload={"lesson_count": len(lessons)},
            ):
                self._update_state(event, lessons)

    async def detect_lessons(self, event: Event) -> list[dict]:
        """Analyze an event and return candidate lesson dicts.

        Detection heuristics:
        - ``FlockComplete``: convergence_score stuck below threshold.
        - ``SwarmComplete``: wasted bridge queries, decreasing gossip info_gain,
          VERIFY source downgrades.
        - ``McpResearchComplete``: idle findings (possible score_version lag),
          high cost-per-finding.

        Args:
            event: A completion event (FlockComplete, SwarmComplete, or
                McpResearchComplete).

        Returns:
            List of lesson dicts ready for ``LessonStore.record``.
        """
        trace = await self._ensure_trace()
        lessons: list[dict] = []
        run_id = event.payload.get("run_id", "") or getattr(trace, "_run_id", "")

        if event.event_type == "FlockComplete":
            lessons.extend(await self._detect_flock_lessons(event, trace, run_id))
        elif event.event_type == "SwarmComplete":
            lessons.extend(await self._detect_swarm_lessons(event, trace, run_id))
        elif event.event_type == "McpResearchComplete":
            lessons.extend(await self._detect_mcp_lessons(event, run_id))

        return lessons

    async def _detect_flock_lessons(
        self, event: Event, trace: TraceStore, run_id: str
    ) -> list[dict]:
        """Detect query-efficiency lessons from Flock history."""
        lessons: list[dict] = []
        threshold = self.config.scheduler.convergence_threshold

        flock_rounds = await trace.query(
            event_type="FlockRoundComplete",
            run_id=run_id,
            limit=20,
        )
        scores: list[float] = []
        for row in sorted(flock_rounds, key=lambda r: r.get("timestamp", "")):
            payload = json.loads(row.get("payload_json", "{}"))
            scores.append(payload.get("convergence_score", 0.0))

        if len(scores) >= self.config.scheduler.max_convergence_stuck_rounds:
            recent = scores[-self.config.scheduler.max_convergence_stuck_rounds :]
            if all(s < threshold for s in recent):
                lessons.append(
                    {
                        "lesson_type": LessonType.QUERY_EFFICIENCY,
                        "fact": (
                            f"Flock convergence score remained below {threshold} "
                            f"for {len(recent)} consecutive rounds"
                        ),
                        "run_id": run_id,
                        "run_number": 0,
                        "angle": "",
                        "query_type": "",
                        "source_url": "",
                        "source_type": "",
                        "relevance_score": 0.7,
                        "confidence": 0.6,
                        "metadata": {
                            "last_scores": recent,
                            "threshold": threshold,
                        },
                        "halflife_runs": self.config.reflexion.halflife_runs_default,
                    }
                )

        return lessons

    async def _detect_swarm_lessons(
        self, event: Event, trace: TraceStore, run_id: str
    ) -> list[dict]:
        """Detect strategy, angle-quality, and source-quality lessons."""
        lessons: list[dict] = []

        # 1. wasted_bridge_queries > 10%
        swarm_phases = await trace.query(
            event_type="SwarmPhaseComplete",
            run_id=run_id,
            limit=50,
        )
        total_bridge = 0
        wasted_bridge = 0
        for row in swarm_phases:
            payload = json.loads(row.get("payload_json", "{}"))
            metrics = payload.get("metrics", {})
            if payload.get("phase") == "BRIDGE" or metrics.get("phase") == "BRIDGE":
                total_bridge += metrics.get("total_queries", 0)
                wasted_bridge += metrics.get("wasted_queries", 0)

        if total_bridge > 0 and (wasted_bridge / total_bridge) > 0.10:
            waste_rate = wasted_bridge / total_bridge
            lessons.append(
                {
                    "lesson_type": LessonType.STRATEGY,
                    "fact": (
                        f"Bridge query waste rate was {waste_rate:.1%} "
                        f"({wasted_bridge}/{total_bridge})"
                    ),
                    "run_id": run_id,
                    "run_number": 0,
                    "angle": "",
                    "query_type": "BRIDGE",
                    "source_url": "",
                    "source_type": "",
                    "relevance_score": 0.6,
                    "confidence": min(0.5 + waste_rate, 0.95),
                    "metadata": {
                        "wasted_bridge": wasted_bridge,
                        "total_bridge": total_bridge,
                    },
                    "halflife_runs": self.config.reflexion.halflife_runs_default,
                }
            )

        # 2. gossip_info_gain decreasing
        gossip_rounds = await trace.query(
            event_type="GossipRoundComplete",
            run_id=run_id,
            limit=20,
        )
        info_gains: list[float] = []
        for row in sorted(gossip_rounds, key=lambda r: r.get("timestamp", "")):
            payload = json.loads(row.get("payload_json", "{}"))
            info_gains.append(payload.get("info_gain", 0.0))

        if len(info_gains) >= 3:
            decreasing = all(
                info_gains[i] > info_gains[i + 1]
                for i in range(len(info_gains) - 1)
            )
            if decreasing:
                lessons.append(
                    {
                        "lesson_type": LessonType.ANGLE_QUALITY,
                        "fact": (
                            "Gossip info_gain decreased monotonically over "
                            "recent rounds"
                        ),
                        "run_id": run_id,
                        "run_number": 0,
                        "angle": "",
                        "query_type": "",
                        "source_url": "",
                        "source_type": "",
                        "relevance_score": 0.6,
                        "confidence": 0.55,
                        "metadata": {"info_gain_series": info_gains},
                        "halflife_runs": self.config.reflexion.halflife_runs_default,
                    }
                )

        # 3. VERIFY downgrades source
        verify_events = await trace.query(
            event_type="source_downgraded",
            run_id=run_id,
            limit=10,
        )
        if not verify_events:
            # Fallback: scan recent trace records for downgrade keywords
            recent = await trace.query(run_id=run_id, limit=200)
            verify_events = [
                row
                for row in recent
                if "downgrade" in str(row.get("payload_json", "")).lower()
                or "verify" in str(row.get("event_type", "")).lower()
            ]

        for row in verify_events[:1]:
            payload = json.loads(row.get("payload_json", "{}"))
            source = payload.get("source_type", payload.get("source", "unknown"))
            lessons.append(
                {
                    "lesson_type": LessonType.SOURCE_QUALITY,
                    "fact": f"Source quality downgraded for {source}",
                    "run_id": run_id,
                    "run_number": 0,
                    "angle": "",
                    "query_type": "VERIFY",
                    "source_url": payload.get("source_url", ""),
                    "source_type": source,
                    "relevance_score": 0.7,
                    "confidence": 0.6,
                    "metadata": {
                        "downgrade_reason": payload.get("reason", "unknown")
                    },
                    "halflife_runs": self.config.reflexion.halflife_runs_default,
                }
            )

        return lessons

    async def _detect_mcp_lessons(self, event: Event, run_id: str) -> list[dict]:
        """Detect model-behavior and cost lessons from MCP completion."""
        lessons: list[dict] = []
        findings_added = event.payload.get("findings_added", 0)
        cost_usd = event.payload.get("cost_usd", 0.0)
        source_type = event.payload.get("source_type", "unknown")

        # MCP findings idle due to score_version lag
        if findings_added == 0 and cost_usd > 0.0:
            lessons.append(
                {
                    "lesson_type": LessonType.MODEL_BEHAVIOR,
                    "fact": (
                        f"MCP research produced zero findings despite "
                        f"${cost_usd:.2f} cost — possible score_version lag"
                    ),
                    "run_id": run_id,
                    "run_number": 0,
                    "angle": "",
                    "query_type": "",
                    "source_url": "",
                    "source_type": source_type,
                    "relevance_score": 0.6,
                    "confidence": 0.6,
                    "metadata": {
                        "cost_usd": cost_usd,
                        "findings_added": findings_added,
                    },
                    "halflife_runs": self.config.reflexion.halflife_runs_default,
                }
            )

        # per-run cost vs findings
        cost_per_finding = cost_usd / max(findings_added, 1)
        if cost_per_finding > 0.50 or (findings_added == 0 and cost_usd > 0.10):
            lessons.append(
                {
                    "lesson_type": LessonType.COST,
                    "fact": (
                        f"High cost-per-finding: ${cost_per_finding:.2f} "
                        f"for {findings_added} findings"
                    ),
                    "run_id": run_id,
                    "run_number": 0,
                    "angle": "",
                    "query_type": "",
                    "source_url": "",
                    "source_type": source_type,
                    "relevance_score": 0.5,
                    "confidence": 0.6,
                    "metadata": {
                        "cost_usd": cost_usd,
                        "findings_added": findings_added,
                        "cost_per_finding": cost_per_finding,
                    },
                    "halflife_runs": self.config.reflexion.halflife_runs_default,
                }
            )

        return lessons

    def _update_state(self, event: Event, lessons: list[dict]) -> None:
        """Update the in-memory ``ReflexionState`` after detection.

        Args:
            event: The completion event that triggered detection.
            lessons: The lessons that were detected (already stored).
        """
        if event.event_type == "FlockComplete":
            directions = event.payload.get("directions", [])
            for d in directions:
                self.state.productive_pairs.append((d, "flock"))

        elif event.event_type == "SwarmComplete":
            findings = event.payload.get("findings", [])
            self.state.breakthrough_findings.extend(findings)
            gaps = event.payload.get("gaps", [])
            coverage = len(findings) / max(len(findings) + len(gaps), 1)
            self.state.coverage_score_history.append(coverage)

        elif event.event_type == "McpResearchComplete":
            source_type = event.payload.get("source_type", "")
            if source_type:
                findings = event.payload.get("findings_added", 0)
                current = self.state.angle_boosts.get(source_type, 0.0)
                self.state.angle_boosts[source_type] = current + (findings * 0.1)

        for lesson in lessons:
            ltype = lesson.get("lesson_type", "")
            qtype = lesson.get("query_type", "")
            if ltype == LessonType.QUERY_EFFICIENCY and qtype:
                self.state.exhausted_query_types.add(qtype)
            elif ltype == LessonType.SOURCE_QUALITY:
                st = lesson.get("source_type", "")
                if st:
                    self.state.source_blacklist.add(st)

    async def health(self) -> dict[str, Any]:
        """Return health metrics including ReflexionState summary."""
        base = await super().health()
        base["reflexion"] = {
            "exhausted_query_types": list(self.state.exhausted_query_types),
            "productive_pairs_count": len(self.state.productive_pairs),
            "breakthrough_findings_count": len(self.state.breakthrough_findings),
            "coverage_score_history_length": len(self.state.coverage_score_history),
            "source_blacklist_count": len(self.state.source_blacklist),
            "angle_boosts": self.state.angle_boosts,
        }
        return base
