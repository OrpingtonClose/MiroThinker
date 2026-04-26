"""Swarm actor orchestration — map → gossip → synthesis pipeline.

This module implements the full swarm lifecycle:

1. **Map** (`start_extraction`) — spawn ``BeeActor`` instances, one per research
   angle. Each bee extracts atomic findings from its assigned raw rows and
   writes them to the store as ``row_type='finding'`` with gradient flags.

2. **Gossip** (`start_gossip`) — bees exchange and refine findings across
   angles. The supervisor emits ``GossipRoundComplete`` after each round.
   When ``info_gain < threshold`` for two consecutive rounds, the supervisor
   emits ``SwarmComplete`` and shuts down gracefully.

3. **Synthesis** (`start_synthesis`) — spawn a ``DiffusionSupervisor`` that
   orchestrates a ``ManifestWorker → ConfrontWorker → CorrectWorker``
   pipeline. Intermediate artifacts are persisted as ``finding`` rows tagged
   with ``diffusion_phase``. The loop converges after a configurable number
   of passes.

All store operations are executed via :func:`asyncio.to_thread` so the
async event loop is never blocked. Every phase is wrapped in
:func:`trace_block` and every write is recorded in the trace store.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from universal_store.actors.base import Actor
from universal_store.actors.supervisor import Supervisor
from universal_store.config import UnifiedConfig
from universal_store.protocols import (
    Event,
    GossipRoundComplete,
    RowType,
    StoreProtocol,
    SwarmComplete,
    SwarmPhaseComplete,
)
from universal_store.trace import TraceStore, trace_block, traced


# ---------------------------------------------------------------------------
# BeeActor — one per angle, extracts findings from raw rows
# ---------------------------------------------------------------------------

class BeeActor(Actor):
    """Worker bee that extracts and refines findings for a single angle.

    Args:
        actor_id: Unique identifier for this bee.
        angle: The research angle this bee is responsible for.
        raw_condition_ids: IDs of raw rows assigned to this bee.
        store: The corpus store (typically ``ConditionStore``).
        config: Unified configuration object.
    """

    def __init__(
        self,
        actor_id: str,
        angle: str,
        raw_condition_ids: list[int],
        store: Any,
        config: UnifiedConfig,
    ) -> None:
        super().__init__(actor_id)
        self.angle = angle
        self.raw_condition_ids = raw_condition_ids
        self.store = store
        self.config = config
        self._findings: list[int] = []
        self._gossip_findings: list[int] = []

    # -- public lifecycle --------------------------------------------------

    async def extract(self) -> list[int]:
        """Public entry-point called by ``SwarmSupervisor`` via ``gather``."""
        return await self._do_extraction()

    async def gossip_round(self, round_num: int) -> list[int]:
        """Public entry-point for a single gossip round."""
        return await self._do_gossip_round(round_num)

    # -- internal loop -----------------------------------------------------

    async def _run(self) -> None:
        """Mailbox loop — handles async events when not driven directly."""
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if event.event_type == "extract":
                await self._do_extraction()
            elif event.event_type == "gossip":
                payload = event.payload or {}
                await self._do_gossip_round(payload.get("round", 0))
            elif event.event_type == "stop":
                break

    # -- extraction --------------------------------------------------------

    @traced("bee_actor", "extraction", "map")
    async def _do_extraction(self) -> list[int]:
        """Extract findings from assigned raw rows and persist them."""
        async with trace_block(self.actor_id, "bee_extraction", self.angle):
            raw_rows = await self._fetch_raw_rows()
            new_ids: list[int] = []
            for row in raw_rows:
                fid = await self._extract_and_write(row)
                if fid is not None:
                    new_ids.append(fid)
            self._findings.extend(new_ids)

            await self.send_to_parent(
                SwarmPhaseComplete(
                    phase="extraction",
                    metrics={"angle": self.angle, "findings_count": len(new_ids)},
                    findings=new_ids,
                )
            )
            return new_ids

    async def _fetch_raw_rows(self) -> list[dict[str, Any]]:
        """Fetch raw ``conditions`` rows assigned to this bee."""
        if not self.raw_condition_ids:
            return []

        def _query() -> list[dict[str, Any]]:
            placeholders = ",".join("?" * len(self.raw_condition_ids))
            sql = (
                f"SELECT id, fact, source_url, source_type, confidence, "
                f"angle, row_type, strategy, created_at "
                f"FROM conditions WHERE id IN ({placeholders}) AND row_type = 'raw'"
            )
            result = self.store.conn.execute(sql, self.raw_condition_ids)
            cols = [desc[0] for desc in result.description]
            return [dict(zip(cols, row)) for row in result.fetchall()]

        return await asyncio.to_thread(_query)

    async def _extract_and_write(self, row: dict[str, Any]) -> int | None:
        """Turn a single raw row into a ``finding`` with gradient flags."""
        trace = await TraceStore.get()
        raw_fact = row.get("fact", "")
        if not raw_fact:
            return None

        # Heuristic atomisation: first paragraph / first 500 chars
        paragraphs = [p.strip() for p in raw_fact.split("\n") if p.strip()]
        finding_text = paragraphs[0][:500] if paragraphs else raw_fact[:500]

        # Gradient flags — heuristics based on text shape
        words = finding_text.split()
        word_count = len(words)
        unique_words = len(set(w.lower() for w in words))
        gradient_growth = round(min(1.0, word_count / 100.0), 4)
        gradient_decay = 0.0 if "not" in finding_text.lower() else 0.1
        gradient_surprise = round(min(1.0, unique_words / max(word_count, 1)), 4)

        strategy = json.dumps({
            "gradient_growth": gradient_growth,
            "gradient_decay": gradient_decay,
            "gradient_surprise": gradient_surprise,
            "parent_raw_id": row.get("id"),
            "extraction_method": "bee_heuristic",
        })

        async with trace_block(self.actor_id, "store_write", "finding"):
            cid = await asyncio.to_thread(
                self.store.admit,
                fact=finding_text,
                row_type=str(RowType.FINDING),
                angle=self.angle,
                source_type="bee_extraction",
                source_url=row.get("source_url", ""),
                confidence=float(row.get("confidence", 0.5)) * 0.9,
                strategy=strategy,
                parent_id=row.get("id"),
            )

        if cid:
            await trace.record(
                actor_id=self.actor_id,
                event_type="finding_admitted",
                payload={
                    "finding_id": cid,
                    "angle": self.angle,
                    "gradient_growth": gradient_growth,
                    "gradient_decay": gradient_decay,
                    "gradient_surprise": gradient_surprise,
                },
            )
        return cid

    # -- gossip ------------------------------------------------------------

    @traced("bee_actor", "gossip", "gossip")
    async def _do_gossip_round(self, round_num: int) -> list[int]:
        """Exchange findings with peer angles and write refined variants."""
        async with trace_block(
            self.actor_id, f"gossip_round_{round_num}", self.angle
        ):
            peer_findings = await self._fetch_peer_findings()
            refined: list[int] = []
            for peer in peer_findings[:20]:  # cap to avoid explosion
                fid = await self._refine_finding(peer, round_num)
                if fid is not None:
                    refined.append(fid)
            self._gossip_findings.extend(refined)
            return refined

    async def _fetch_peer_findings(self) -> list[dict[str, Any]]:
        """Return active findings from *other* angles."""

        def _query() -> list[dict[str, Any]]:
            rows = self.store.conn.execute(
                """SELECT id, fact, confidence, angle, strategy, created_at
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND angle != ?
                   ORDER BY confidence DESC
                   LIMIT 50""",
                [self.angle],
            ).fetchall()
            cols = ["id", "fact", "confidence", "angle", "strategy", "created_at"]
            return [dict(zip(cols, row)) for row in rows]

        return await asyncio.to_thread(_query)

    async def _refine_finding(
        self, peer: dict[str, Any], round_num: int
    ) -> int | None:
        """Create a cross-angle refinement of a peer finding."""
        trace = await TraceStore.get()
        peer_text = peer.get("fact", "")
        if not peer_text:
            return None

        refined_text = f"[{self.angle} perspective] {peer_text[:400]}"

        strategy = json.dumps({
            "refined_from": peer.get("id"),
            "refined_by": self.angle,
            "gossip_round": round_num,
            "gradient_growth": 0.5,
            "gradient_decay": 0.0,
            "gradient_surprise": 0.3,
        })

        async with trace_block(self.actor_id, "store_write", "finding"):
            cid = await asyncio.to_thread(
                self.store.admit,
                fact=refined_text,
                row_type=str(RowType.FINDING),
                angle=self.angle,
                source_type="bee_gossip",
                confidence=float(peer.get("confidence", 0.5)) * 0.95,
                strategy=strategy,
                parent_id=peer.get("id"),
            )

        if cid:
            await trace.record(
                actor_id=self.actor_id,
                event_type="gossip_finding_admitted",
                payload={"finding_id": cid, "round": round_num},
            )
        return cid


# ---------------------------------------------------------------------------
# DiffusionSupervisor — manifest → confront → correct → converge
# ---------------------------------------------------------------------------

class DiffusionSupervisor(Supervisor):
    """Synthesis supervisor that iterates manifest→confront→correct.

    Intermediate artifacts are written as ``finding`` rows with the
    ``diffusion_phase`` tag stored in the ``phase`` column.
    """

    def __init__(
        self,
        actor_id: str,
        store: Any,
        config: UnifiedConfig | None = None,
    ) -> None:
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig.from_env()
        self._converged = False
        self._diffusion_pass = 0
        self._pipeline_classes: list[type[Actor]] = [
            ManifestWorker,
            ConfrontWorker,
            CorrectWorker,
        ]

    async def _run(self) -> None:
        """Supervisor loop: handle pipeline commands and child crashes."""
        trace = await TraceStore.get()
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=5.0)
            except asyncio.TimeoutError:
                await self._health_check_all()
                continue

            await trace.record(
                actor_id=self.actor_id,
                event_type="diffusion_received",
                payload={"event_type": event.event_type},
            )

            if event.event_type == "start_pipeline":
                await self._run_pipeline()
            elif event.event_type == "actor_crashed":
                await self._handle_child_crash(
                    event.payload.get("actor_id", event.source_actor)
                )

    async def _run_pipeline(self) -> None:
        """Run the manifest→confront→correct loop until convergence."""
        max_passes = self.config.swarm.max_gossip_rounds

        while not self._converged and not self._shutdown:
            self._diffusion_pass += 1
            if self._diffusion_pass > max_passes:
                break

            pass_label = f"pass_{self._diffusion_pass}"
            async with trace_block(self.actor_id, "diffusion_pass", pass_label):
                workers: list[Actor] = []
                for cls in self._pipeline_classes:
                    worker_id = (
                        f"{self.actor_id}:{cls.__name__.lower()}"
                        f":{self._diffusion_pass}"
                    )
                    worker = cls(
                        actor_id=worker_id,
                        store=self.store,
                        config=self.config,
                        diffusion_pass=self._diffusion_pass,
                    )
                    workers.append(worker)
                    self.spawn_child(worker)

                # Run sequentially: manifest → confront → correct
                for worker in workers:
                    await worker.send(
                        Event("run", {"diffusion_pass": self._diffusion_pass})
                    )
                    await self._await_worker(worker)

                # Simple convergence heuristic: converge after 3 passes
                if self._diffusion_pass >= 3:
                    self._converged = True

        async with trace_block(
            self.actor_id, "diffusion_converged", f"passes_{self._diffusion_pass}"
        ):
            await self.send(
                Event(
                    "diffusion_converged",
                    {
                        "passes": self._diffusion_pass,
                        "converged": self._converged,
                    },
                )
            )

    async def _await_worker(self, worker: Actor, timeout_s: float = 60.0) -> None:
        """Poll worker health until it finishes or times out."""
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            if self._shutdown:
                break
            health = await worker.health()
            if not health.get("running"):
                break
            await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Pipeline workers — Manifest, Confront, Correct
# ---------------------------------------------------------------------------

class ManifestWorker(Actor):
    """Produces a structured manifest of current findings."""

    def __init__(
        self,
        actor_id: str,
        store: Any,
        config: UnifiedConfig,
        diffusion_pass: int = 0,
    ) -> None:
        super().__init__(actor_id)
        self.store = store
        self.config = config
        self.diffusion_pass = diffusion_pass

    async def _run(self) -> None:
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=30.0)
            except asyncio.TimeoutError:
                return
            if event.event_type == "run":
                await self._manifest()
                return

    @traced("manifest_worker", "manifest", "synthesis")
    async def _manifest(self) -> None:
        async with trace_block(self.actor_id, "manifest", f"pass_{self.diffusion_pass}"):
            findings = await self._fetch_findings()
            manifest_text = self._build_manifest(findings)
            await self._write_artifact(
                fact=manifest_text,
                diffusion_phase=f"manifest_pass_{self.diffusion_pass}",
            )
            await self.send_to_parent(
                Event(
                    "worker_complete",
                    {"worker": "manifest", "pass": self.diffusion_pass},
                )
            )

    async def _fetch_findings(self) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            return self.store.get_findings(min_confidence=0.0, limit=500)

        return await asyncio.to_thread(_query)

    def _build_manifest(self, findings: list[dict[str, Any]]) -> str:
        lines = [
            f"=== MANIFEST pass {self.diffusion_pass} "
            f"({len(findings)} findings) ==="
        ]
        for f in findings[:50]:
            lines.append(
                f"  [{f.get('angle', '')}] {f.get('fact', '')[:200]}"
            )
        return "\n".join(lines)

    async def _write_artifact(self, fact: str, diffusion_phase: str) -> int | None:
        trace = await TraceStore.get()
        strategy = json.dumps({
            "diffusion_phase": diffusion_phase,
            "worker_type": "manifest",
            "pass": self.diffusion_pass,
        })

        async with trace_block(self.actor_id, "store_write", "finding"):
            cid = await asyncio.to_thread(
                self.store.admit,
                fact=fact,
                row_type=str(RowType.FINDING),
                angle="synthesis",
                source_type="diffusion_manifest",
                confidence=0.75,
                strategy=strategy,
                phase=diffusion_phase,
            )

        if cid:
            await trace.record(
                actor_id=self.actor_id,
                event_type="manifest_admitted",
                payload={"finding_id": cid, "phase": diffusion_phase},
            )
        return cid


class ConfrontWorker(Actor):
    """Surfaces contradictions between findings from different angles."""

    def __init__(
        self,
        actor_id: str,
        store: Any,
        config: UnifiedConfig,
        diffusion_pass: int = 0,
    ) -> None:
        super().__init__(actor_id)
        self.store = store
        self.config = config
        self.diffusion_pass = diffusion_pass

    async def _run(self) -> None:
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=30.0)
            except asyncio.TimeoutError:
                return
            if event.event_type == "run":
                await self._confront()
                return

    @traced("confront_worker", "confront", "synthesis")
    async def _confront(self) -> None:
        async with trace_block(
            self.actor_id, "confront", f"pass_{self.diffusion_pass}"
        ):
            findings = await self._fetch_findings()
            confront_text = self._build_confrontation(findings)
            await self._write_artifact(
                fact=confront_text,
                diffusion_phase=f"confront_pass_{self.diffusion_pass}",
            )
            await self.send_to_parent(
                Event(
                    "worker_complete",
                    {"worker": "confront", "pass": self.diffusion_pass},
                )
            )

    async def _fetch_findings(self) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            return self.store.get_findings(min_confidence=0.0, limit=500)

        return await asyncio.to_thread(_query)

    def _build_confrontation(self, findings: list[dict[str, Any]]) -> str:
        lines = [f"=== CONFRONTATION pass {self.diffusion_pass} ==="]
        by_angle: dict[str, list[dict[str, Any]]] = {}
        for f in findings:
            by_angle.setdefault(f.get("angle", ""), []).append(f)

        angles = [a for a in by_angle.keys() if a]
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                a_finds = by_angle[angles[i]]
                b_finds = by_angle[angles[j]]
                if a_finds and b_finds:
                    lines.append(f"\n-- {angles[i]} vs {angles[j]} --")
                    lines.append(
                        f"  A: {a_finds[0].get('fact', '')[:150]}"
                    )
                    lines.append(
                        f"  B: {b_finds[0].get('fact', '')[:150]}"
                    )
        return "\n".join(lines)

    async def _write_artifact(self, fact: str, diffusion_phase: str) -> int | None:
        trace = await TraceStore.get()
        strategy = json.dumps({
            "diffusion_phase": diffusion_phase,
            "worker_type": "confront",
            "pass": self.diffusion_pass,
        })

        async with trace_block(self.actor_id, "store_write", "finding"):
            cid = await asyncio.to_thread(
                self.store.admit,
                fact=fact,
                row_type=str(RowType.FINDING),
                angle="synthesis",
                source_type="diffusion_confront",
                confidence=0.6,
                strategy=strategy,
                phase=diffusion_phase,
            )

        if cid:
            await trace.record(
                actor_id=self.actor_id,
                event_type="confront_admitted",
                payload={"finding_id": cid, "phase": diffusion_phase},
            )
        return cid


class CorrectWorker(Actor):
    """Produces corrections / resolutions for surfaced contradictions."""

    def __init__(
        self,
        actor_id: str,
        store: Any,
        config: UnifiedConfig,
        diffusion_pass: int = 0,
    ) -> None:
        super().__init__(actor_id)
        self.store = store
        self.config = config
        self.diffusion_pass = diffusion_pass

    async def _run(self) -> None:
        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=30.0)
            except asyncio.TimeoutError:
                return
            if event.event_type == "run":
                await self._correct()
                return

    @traced("correct_worker", "correct", "synthesis")
    async def _correct(self) -> None:
        async with trace_block(
            self.actor_id, "correct", f"pass_{self.diffusion_pass}"
        ):
            confrontations = await self._fetch_confrontations()
            corrections = self._build_corrections(confrontations)
            for corr_text in corrections:
                await self._write_artifact(
                    fact=corr_text,
                    diffusion_phase=f"correct_pass_{self.diffusion_pass}",
                )
            await self.send_to_parent(
                Event(
                    "worker_complete",
                    {"worker": "correct", "pass": self.diffusion_pass},
                )
            )

    async def _fetch_confrontations(self) -> list[dict[str, Any]]:
        def _query() -> list[dict[str, Any]]:
            rows = self.store.conn.execute(
                """SELECT id, fact, confidence, strategy
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                     AND source_type = 'diffusion_confront'
                     AND phase LIKE 'confront_pass_%'
                   ORDER BY id DESC
                   LIMIT 10""",
            ).fetchall()
            cols = ["id", "fact", "confidence", "strategy"]
            return [dict(zip(cols, row)) for row in rows]

        return await asyncio.to_thread(_query)

    def _build_corrections(
        self, confrontations: list[dict[str, Any]]
    ) -> list[str]:
        if not confrontations:
            return [
                f"[Correction pass {self.diffusion_pass}] "
                f"No contradictions to resolve."
            ]
        corrections = []
        for c in confrontations:
            text = (
                f"[Correction pass {self.diffusion_pass}] "
                f"Resolved: {c.get('fact', '')[:200]}"
            )
            corrections.append(text)
        return corrections

    async def _write_artifact(self, fact: str, diffusion_phase: str) -> int | None:
        trace = await TraceStore.get()
        strategy = json.dumps({
            "diffusion_phase": diffusion_phase,
            "worker_type": "correct",
            "pass": self.diffusion_pass,
        })

        async with trace_block(self.actor_id, "store_write", "finding"):
            cid = await asyncio.to_thread(
                self.store.admit,
                fact=fact,
                row_type=str(RowType.FINDING),
                angle="synthesis",
                source_type="diffusion_correct",
                confidence=0.8,
                strategy=strategy,
                phase=diffusion_phase,
            )

        if cid:
            await trace.record(
                actor_id=self.actor_id,
                event_type="correct_admitted",
                payload={"finding_id": cid, "phase": diffusion_phase},
            )
        return cid


# ---------------------------------------------------------------------------
# SwarmSupervisor — top-level orchestrator: map → gossip → synthesis
# ---------------------------------------------------------------------------

class SwarmSupervisor(Supervisor):
    """Manages the full swarm lifecycle.

    Phases:
        1. **Extraction** — spawn ``BeeActor`` instances (one per angle).
        2. **Gossip** — run rounds until ``info_gain`` drops below threshold
           for two consecutive rounds.
        3. **Synthesis** — spawn ``DiffusionSupervisor`` to run the
           manifest→confront→correct pipeline.

    All phase boundaries and store writes are traced. Parallel work is
    coordinated via :func:`asyncio.gather` and bounded by an
    :class:`asyncio.Semaphore`.
    """

    def __init__(
        self,
        actor_id: str,
        store: Any,
        config: UnifiedConfig | None = None,
    ) -> None:
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig.from_env()
        self._findings: list[int] = []
        self._gaps: list[str] = []
        self._gossip_round = 0
        self._info_gain_history: list[float] = []
        self._bee_actors: dict[str, BeeActor] = {}
        self._sem: asyncio.Semaphore = asyncio.Semaphore(
            self.config.swarm.max_workers_per_phase
        )
        self._diffusion_supervisor: DiffusionSupervisor | None = None
        self._phase: str = "idle"

    # -- phase 1: extraction -----------------------------------------------

    async def start_extraction(
        self, raw_condition_ids: list[int], angles: list[str]
    ) -> None:
        """Spawn ``BeeActor`` instances and run parallel extraction.

        Each bee receives a roughly equal subset of ``raw_condition_ids``.
        """
        async with trace_block(self.actor_id, "start_extraction", ""):
            if not angles:
                return

            chunk_size = max(1, len(raw_condition_ids) // len(angles))
            for i, angle in enumerate(angles):
                start = i * chunk_size
                end = (
                    start + chunk_size
                    if i < len(angles) - 1
                    else len(raw_condition_ids)
                )
                subset = raw_condition_ids[start:end]

                bee = BeeActor(
                    actor_id=f"bee:{angle}",
                    angle=angle,
                    raw_condition_ids=subset,
                    store=self.store,
                    config=self.config,
                )
                self._bee_actors[angle] = bee
                self.spawn_child(bee)

            async def _run_extract(bee: BeeActor) -> list[int]:
                async with self._sem:
                    return await bee.extract()

            await asyncio.gather(
                *[_run_extract(bee) for bee in self._bee_actors.values()]
            )
            self._phase = "extraction"

    # -- phase 2: gossip ---------------------------------------------------

    async def start_gossip(self) -> None:
        """Run gossip rounds until information gain converges.

        Emits ``GossipRoundComplete`` after every round. If ``info_gain`` is
        below the configured threshold for two consecutive rounds (and the
        minimum round count has been met), emits ``SwarmComplete`` and
        initiates graceful shutdown.
        """
        async with trace_block(self.actor_id, "start_gossip", ""):
            threshold = self.config.swarm.gossip_info_gain_threshold
            max_rounds = self.config.swarm.max_gossip_rounds
            min_rounds = self.config.swarm.min_gossip_rounds

            for round_num in range(1, max_rounds + 1):
                if self._shutdown:
                    break

                self._gossip_round = round_num
                async with trace_block(
                    self.actor_id, "gossip_round", f"round_{round_num}"
                ):
                    async def _run_gossip(bee: BeeActor) -> list[int]:
                        async with self._sem:
                            return await bee.gossip_round(round_num)

                    results = await asyncio.gather(
                        *[_run_gossip(bee) for bee in self._bee_actors.values()]
                    )
                    flat = [fid for sub in results for fid in sub]
                    new_findings = [fid for fid in flat if fid not in self._findings]
                    self._findings.extend(new_findings)

                    info_gain = len(new_findings) / max(len(self._findings), 1)
                    self._info_gain_history.append(info_gain)

                    gaps: list[str] = []
                    if info_gain < threshold:
                        gaps.append(f"low_info_gain_round_{round_num}")

                    event = GossipRoundComplete(
                        round_num=round_num,
                        info_gain=info_gain,
                        gaps_found=len(gaps),
                    )
                    await self.send(event)

                    # Convergence check
                    if round_num >= min_rounds and len(self._info_gain_history) >= 2:
                        last_two = self._info_gain_history[-2:]
                        if all(g < threshold for g in last_two):
                            self._gaps = gaps
                            await self.send(
                                SwarmComplete(
                                    findings=self._findings, gaps=self._gaps
                                )
                            )
                            await self.stop(graceful=True)
                            return

            # Max rounds reached without convergence
            await self.send(
                SwarmComplete(findings=self._findings, gaps=self._gaps)
            )
            await self.stop(graceful=True)

    # -- phase 3: synthesis ------------------------------------------------

    async def start_synthesis(self) -> None:
        """Spawn ``DiffusionSupervisor`` with the manifest→confront→correct pipeline."""
        async with trace_block(self.actor_id, "start_synthesis", ""):
            self._diffusion_supervisor = DiffusionSupervisor(
                actor_id=f"{self.actor_id}:diffusion",
                store=self.store,
                config=self.config,
            )
            self.spawn_child(self._diffusion_supervisor)
            await self._diffusion_supervisor.send(Event("start_pipeline"))
            self._phase = "synthesis"

    # -- event routing -----------------------------------------------------

    async def _handle_event(self, event: Event) -> None:
        """Route child events and update aggregate state."""
        if event.event_type == "SwarmPhaseComplete":
            phase = event.payload.get("phase")
            if phase == "extraction":
                self._findings.extend(event.payload.get("findings", []))
        elif event.event_type == "GossipRoundComplete":
            pass  # handled inline in start_gossip
        elif event.event_type == "SwarmComplete":
            pass
        elif event.event_type == "diffusion_converged":
            pass
