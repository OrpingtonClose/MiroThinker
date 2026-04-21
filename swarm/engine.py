# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""GossipSwarm — the main orchestrator for parallel corpus synthesis.

Architecture (information-asymmetric conversation):

    OPERATING PRINCIPLE: No worker hosts the whole corpus.
    Each worker sees a SLICE of the findings relevant to its angle.
    Cross-domain insights emerge from the conversation between workers,
    not from individual analysis of the full dataset.

    ┌──────────────────────────────────────────────────────────┐
    │                    QUEEN (merge)                          │
    │   Reads all conversation-refined summaries + serendipity  │
    │   insights → produces final unified synthesis             │
    └──────────┬──────────┬──────────┬──────────┬─────────────┘
               │          │          │          │
    ┌──────────▼──┐ ┌─────▼─────┐ ┌──▼────────┐ │
    │  Worker A   │ │ Worker B  │ │ Worker C  │ ...
    │  Angle 1    │ │ Angle 2   │ │ Angle 3   │
    │  SLICE of   │ │ SLICE of  │ │ SLICE of  │
    │  corpus     │ │ corpus    │ │ corpus    │
    └──────┬──────┘ └──────┬────┘ └──────┬────┘
           │               │             │
           └───────────────┼─────────────┘
                    Conversation Round(s)
              (each worker reads peers'
               summaries + own slice,
               discovers connections across
               domains it couldn't see alone)
                           │
               ┌───────────▼───────────┐
               │  Serendipity Bridge   │
               │  Polymath connector   │
               │  finds cross-angle    │
               │  surprises            │
               └───────────────────────┘

Phases:
  0. Corpus Analysis — detect sections, extract angles, assign workers
  1. Map — each worker synthesizes its section (parallel)
  2. Gossip — workers read peers + own raw data, refine (parallel, adaptive)
     Multiple rounds with round-specific prompts:
       Round 1: Incorporate peer findings
       Round 2: Resolve contradictions
       Round 3: Final definitive synthesis
  3. Serendipity — polymath finds cross-angle surprises (sequential)
  4. Queen Merge — combines everything into final synthesis (sequential)

Per-phase model selection:
  Workers and gossip use ``worker_complete`` (fast, uncensored, parallel).
  Queen and knowledge report use ``queen_complete`` (best writer available).
  Serendipity uses ``serendipity_complete`` (best cross-domain reasoner).
  All default to the single ``complete`` function if per-phase callables
  are not provided.

Lineage tracking:
  If ``config.lineage_store`` is set, every phase emits a ``LineageEntry``
  with parent pointers forming a DAG from final reports back to raw sections.

Quality manifest:
  If ``config.enable_quality_manifest`` is True (default), a computed
  provenance footer is appended to both reports.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from swarm.angles import (
    WorkerAssignment,
    _parse_findings,
    assign_workers,
    detect_angles_via_llm,
    detect_sections,
    distribute_findings_to_angles,
    score_section_angle_pairs,
    _prepare_sections,
)
from swarm.config import CompleteFn, SwarmConfig
from swarm.convergence import check_convergence, information_gain, select_diverse_peers
from swarm.lineage import LineageEntry
from swarm.quality_manifest import SwarmQualityManifest
from swarm.queen import build_knowledge_report, queen_merge, queen_lucidity_pass
from swarm.serendipity import find_serendipitous_connections
from swarm.topologies.collision import CollisionStatement
from swarm.topologies.coordinator import coordinate_all_angles
from swarm.topologies.hierarchy import LeafCluster, cluster_findings
from swarm.topologies.mesh import BridgeWorker, run_mesh_rounds
from swarm.topologies.serendipity_panel import run_serendipity_panel
from swarm.worker import worker_gossip_refine, worker_synthesize

logger = logging.getLogger(__name__)


def _lid() -> str:
    """Generate a short unique lineage entry ID."""
    return uuid.uuid4().hex[:12]


@dataclass
class SwarmMetrics:
    """Telemetry from a swarm synthesis run."""

    total_llm_calls: int = 0
    total_workers: int = 0
    gossip_rounds_executed: int = 0
    gossip_rounds_configured: int = 0
    gossip_converged_early: bool = False
    serendipity_produced: bool = False
    phase_times: dict[str, float] = field(default_factory=dict)
    worker_input_chars: list[int] = field(default_factory=list)
    worker_output_chars: list[int] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    gossip_info_gain: list[float] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    # Three-tier topology metrics
    leaf_clusters: int = 0
    bridge_workers: int = 0
    bridge_rounds_executed: int = 0
    collisions_produced: int = 0
    panel_insights_produced: int = 0


@dataclass
class SwarmResult:
    """Result of a gossip swarm synthesis.

    Contains two reports:
    - user_report: Concise narrative synthesis (queen merge, 3000-6000 words)
    - knowledge_report: Full-length structured report preserving all findings

    The legacy ``synthesis`` attribute is an alias for ``user_report``.
    """

    synthesis: str
    knowledge_report: str = ""
    metrics: SwarmMetrics = field(default_factory=SwarmMetrics)
    worker_summaries: dict[str, str] = field(default_factory=dict)
    serendipity_insights: str = ""
    angles_detected: list[str] = field(default_factory=list)
    quality_manifest: SwarmQualityManifest | None = None
    collision_statements: list[CollisionStatement] = field(default_factory=list)

    @property
    def user_report(self) -> str:
        """Alias for synthesis — the concise end-user narrative."""
        return self.synthesis


class GossipSwarm:
    """Parallel corpus synthesis engine with gossip refinement.

    Model-agnostic: accepts any async callable that takes a prompt string
    and returns a completion string. Works with Venice API, local Ollama,
    the proxy layer, or any other LLM backend.

    Per-phase model selection allows different models for workers (fast,
    uncensored), queen merge (best writer), and serendipity bridge (best
    cross-domain reasoner).  An uncensored model with equivalent capability
    is strictly better than its censored counterpart.

    Lineage tracking: pass ``config=SwarmConfig(lineage_store=store)`` to
    record every phase output with parent pointers forming a DAG.

    Usage:
        async def fast_local(prompt: str) -> str:
            # Ollama Gemma-4-uncensored on H200
            ...

        async def best_writer(prompt: str) -> str:
            # Qwen-Claude-Opus or remote Claude
            ...

        swarm = GossipSwarm(
            complete=fast_local,
            queen_complete=best_writer,
            config=SwarmConfig(gossip_rounds=3),
        )
        result = await swarm.synthesize(corpus="...", query="...")
        print(result.user_report)
        print(result.knowledge_report)
    """

    def __init__(
        self,
        complete: CompleteFn,
        worker_complete: CompleteFn | None = None,
        queen_complete: CompleteFn | None = None,
        serendipity_complete: CompleteFn | None = None,
        config: SwarmConfig | None = None,
    ) -> None:
        self.complete = complete
        self.worker_complete = worker_complete or complete
        self.queen_complete = queen_complete or complete
        self.serendipity_complete = serendipity_complete or complete
        self.config = config or SwarmConfig()

    def _emit(self, entry: LineageEntry) -> None:
        """Emit a lineage entry if a store is configured."""
        store = self.config.lineage_store
        if store is not None:
            store.emit(entry)

    async def synthesize(
        self,
        corpus: str,
        query: str,
        on_event: "Callable[[dict], Awaitable[None]] | None" = None,
        cancel_event: "asyncio.Event | None" = None,
    ) -> SwarmResult:
        """Run the full gossip swarm pipeline on the given corpus.

        Args:
            corpus: The full text corpus to synthesize.
            query: The user's research query.
            on_event: Optional async callback for streaming progress events.
                Called with structured dicts at each phase boundary and
                gossip round completion.
            cancel_event: Optional asyncio.Event checked between phases
                and gossip rounds.  If set, returns a partial result.

        Returns:
            SwarmResult with the final synthesis, metrics, and intermediate data.

        Notes:
            If ``config.corpus_delta_fn`` is set, the engine calls it between
            gossip rounds to pick up new findings from external producers.
            The delta text is injected into each worker's next gossip prompt.
        """
        # Dispatch to hierarchical pipeline if enabled
        if self.config.enable_hierarchy:
            return await self._synthesize_hierarchical(
                corpus, query, on_event, cancel_event,
            )

        t0 = time.monotonic()
        metrics = SwarmMetrics()
        metrics.gossip_rounds_configured = self.config.gossip_rounds
        config = self.config

        async def _emit_event(event: dict) -> None:
            """Emit a progress event if a callback is configured."""
            if on_event is not None:
                try:
                    await on_event(event)
                except Exception:
                    logger.debug("on_event callback failed, continuing")

        def _is_cancelled() -> bool:
            """Check if cancellation has been requested."""
            return cancel_event is not None and cancel_event.is_set()

        # ── Phase 0: Corpus Analysis ─────────────────────────────────
        phase_start = time.monotonic()
        sections = detect_sections(corpus)
        logger.info(
            "corpus_sections=<%d>, corpus_chars=<%d> | corpus analysis complete",
            len(sections), len(corpus),
        )

        # Detect angles via LLM (or fall back to section titles)
        angles = await detect_angles_via_llm(
            corpus, query, self.complete,
            max_angles=config.max_workers,
        )
        metrics.total_llm_calls += 1 if angles else 0

        if not angles:
            angles = [s.title for s in sections[:config.max_workers]]

        # ── Angle-driven worker creation ──────────────────────────
        #
        # OPERATING PRINCIPLE: no worker hosts the whole corpus.
        # Each worker gets a SLICE of the findings, assigned by the
        # LLM based on angle relevance.  Information asymmetry is
        # what makes the conversation between workers productive —
        # Worker A knows things Worker B doesn't, and vice versa.
        # Cross-domain insights (e.g. the iron-trenbolone link)
        # emerge from the conversation, not from individual analysis.
        #
        # When the corpus has fewer structural sections than detected
        # angles (common for flat text from export_for_swarm()), we
        # parse the corpus into individual findings and distribute
        # them across angles via LLM.  When enough sections exist,
        # the existing semantic/keyword assignment handles it.
        if len(sections) < len(angles):
            findings = _parse_findings(corpus)
            if len(findings) >= len(angles):
                # Distribute findings across angles — each worker
                # gets a different slice of the corpus
                angle_to_findings = await distribute_findings_to_angles(
                    findings, angles, self.complete,
                )
                metrics.total_llm_calls += 1

                assignments = []
                for i, angle in enumerate(angles):
                    finding_indices = angle_to_findings.get(angle, [])
                    # Build this worker's slice from its assigned findings
                    slice_lines = [findings[idx][1] for idx in finding_indices]
                    worker_content = "\n".join(slice_lines)
                    assignments.append(
                        WorkerAssignment(
                            worker_id=i,
                            angle=angle,
                            raw_content=worker_content,
                        )
                    )

                logger.info(
                    "sections=<%d>, angles=<%d>, findings=<%d> | "
                    "distributed findings across angle workers: %s",
                    len(sections), len(angles), len(findings),
                    {a: len(idxs) for a, idxs in angle_to_findings.items()},
                )
            else:
                # Very few findings — split evenly with round-robin
                logger.info(
                    "sections=<%d>, angles=<%d>, findings=<%d> | "
                    "too few findings for LLM distribution, using round-robin",
                    len(sections), len(angles), len(findings),
                )
                buckets: dict[int, list[str]] = {i: [] for i in range(len(angles))}
                for idx, (_, text) in enumerate(findings):
                    buckets[idx % len(angles)].append(text)
                assignments = [
                    WorkerAssignment(
                        worker_id=i,
                        angle=angle,
                        raw_content="\n".join(buckets[i]),
                    )
                    for i, angle in enumerate(angles)
                ]
        else:
            # Enough sections — use semantic or keyword assignment
            score_matrix = None
            if config.enable_semantic_assignment and angles and len(sections) >= 2:
                prepared = _prepare_sections(
                    list(sections), config.max_workers, config.max_section_chars,
                )
                if prepared and len(prepared) >= 2:
                    try:
                        score_matrix = await score_section_angle_pairs(
                            prepared, angles, self.complete,
                        )
                        metrics.total_llm_calls += 1
                        if score_matrix is not None:
                            logger.info(
                                "sections=<%d>, angles=<%d> | semantic scoring complete",
                                len(prepared), len(angles),
                            )
                        else:
                            logger.info(
                                "sections=<%d>, angles=<%d> | semantic scoring "
                                "returned None, using keyword fallback",
                                len(prepared), len(angles),
                            )
                    except Exception as exc:
                        logger.warning(
                            "error=<%s> | semantic scoring failed, falling back to keyword",
                            exc,
                        )

            assignments = assign_workers(
                sections, angles,
                max_workers=config.max_workers,
                max_section_chars=config.max_section_chars,
                score_matrix=score_matrix,
            )

        if not assignments:
            return SwarmResult(
                synthesis="No content found in corpus to synthesize.",
                metrics=metrics,
            )

        metrics.total_workers = len(assignments)
        metrics.worker_input_chars = [a.char_count for a in assignments]
        metrics.phase_times["corpus_analysis"] = time.monotonic() - phase_start

        logger.info(
            "workers=<%d>, angles=<%s> | worker assignment complete",
            len(assignments), [a.angle for a in assignments],
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "corpus_analysis",
            "sections": len(sections),
            "angles": [a.angle for a in assignments],
            "workers": len(assignments),
            "elapsed_s": round(metrics.phase_times["corpus_analysis"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(synthesis="[cancelled]", metrics=metrics)

        # Lineage: emit corpus analysis entries
        corpus_entry_id = _lid()
        self._emit(LineageEntry(
            entry_id=corpus_entry_id,
            phase="corpus_analysis",
            content=f"Detected {len(sections)} sections, {len(angles)} angles",
            metadata={
                "corpus_chars": len(corpus),
                "section_count": len(sections),
                "angle_count": len(angles),
                "query": query[:200],
            },
        ))

        # ── Phase 1: Map (parallel worker synthesis) ─────────────────
        phase_start = time.monotonic()
        sem = asyncio.Semaphore(config.max_concurrency)
        worker_entry_ids: dict[int, str] = {}

        async def _bounded_synthesize(assignment: WorkerAssignment) -> None:
            async with sem:
                try:
                    assignment.summary = await worker_synthesize(
                        angle=assignment.angle,
                        section_content=assignment.raw_content,
                        query=query,
                        max_chars=config.max_summary_chars,
                        complete_fn=self.worker_complete,
                    )
                except Exception:
                    logger.warning(
                        "worker_id=<%d>, angle=<%s> | worker synthesis failed, using raw truncation",
                        assignment.worker_id, assignment.angle,
                    )
                    assignment.summary = assignment.raw_content[:config.max_summary_chars]
                    metrics.degradations.append(
                        f"Worker {assignment.angle} synthesis failed, used raw truncation"
                    )

        await asyncio.gather(*[_bounded_synthesize(a) for a in assignments])
        metrics.total_llm_calls += len(assignments)
        metrics.phase_times["map"] = time.monotonic() - phase_start

        # Lineage: emit worker synthesis entries
        for a in assignments:
            eid = _lid()
            worker_entry_ids[a.worker_id] = eid
            self._emit(LineageEntry(
                entry_id=eid,
                phase="worker_synthesis",
                angle=a.angle,
                content=a.summary,
                parent_ids=(corpus_entry_id,),
                metadata={
                    "worker_id": a.worker_id,
                    "input_chars": a.char_count,
                    "output_chars": len(a.summary),
                },
            ))

        logger.info(
            "map_phase_s=<%.1f>, workers=<%d> | map phase complete",
            metrics.phase_times["map"], len(assignments),
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "map_complete",
            "workers": len(assignments),
            "elapsed_s": round(metrics.phase_times["map"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            worker_summaries = {a.angle: a.summary for a in assignments}
            return SwarmResult(
                synthesis="[cancelled after map phase]",
                metrics=metrics,
                worker_summaries=worker_summaries,
                angles_detected=[a.angle for a in assignments],
            )

        # ── Phase 2: Gossip Rounds (parallel, adaptive) ─────────────
        phase_start = time.monotonic()
        rounds_executed = 0
        delta_text = ""  # corpus delta injection text for next round

        for gossip_round in range(1, config.max_gossip_rounds + 1):
            # Save previous summaries for convergence check
            for a in assignments:
                a.prev_summary = a.summary

            gossip_failures = 0

            # Get round-specific prompt modifier (if any)
            round_prompt = config.round_prompts.get(gossip_round, "")

            # Capture delta_text for this round (closure-safe)
            _round_delta = delta_text

            async def _bounded_gossip(
                assignment: WorkerAssignment,
                _round_prompt: str = round_prompt,
                _delta: str = _round_delta,
            ) -> None:
                nonlocal gossip_failures
                async with sem:
                    peer_sums = [
                        other.summary for other in assignments
                        if other.worker_id != assignment.worker_id and other.summary
                    ]
                    # Diversity-Aware Retention (DAR): select maximally
                    # disagreeing peers to reduce echo-chamber effects.
                    # From "Hear Both Sides" (arXiv 2603.20640v1).
                    if config.enable_diversity_aware_gossip and assignment.summary:
                        peer_sums = select_diverse_peers(
                            own_summary=assignment.summary,
                            peer_summaries=peer_sums,
                            top_k=config.dar_top_k,
                        )
                    # Full-corpus gossip: pass raw section content
                    raw = assignment.raw_content if config.enable_full_corpus_gossip else None

                    try:
                        assignment.summary = await worker_gossip_refine(
                            angle=assignment.angle,
                            own_summary=assignment.summary,
                            peer_summaries=peer_sums,
                            raw_section=raw,
                            query=query,
                            max_chars=config.max_summary_chars,
                            complete_fn=self.worker_complete,
                            round_prompt=_round_prompt,
                            delta_text=_delta,
                        )
                    except Exception:
                        gossip_failures += 1
                        logger.warning(
                            "worker_id=<%d>, angle=<%s>, round=<%d> | gossip refinement failed, keeping previous summary",
                            assignment.worker_id, assignment.angle, gossip_round,
                        )
                        metrics.degradations.append(
                            f"Worker {assignment.angle} gossip round {gossip_round} failed"
                        )

            await asyncio.gather(*[_bounded_gossip(a) for a in assignments])
            metrics.total_llm_calls += len(assignments)
            rounds_executed = gossip_round

            # Measure information gain this round
            current = [a.summary for a in assignments]
            previous = [a.prev_summary for a in assignments]
            gain = information_gain(current, previous)
            metrics.gossip_info_gain.append(gain)

            # Lineage: emit gossip round entries
            for a in assignments:
                prev_eid = worker_entry_ids[a.worker_id]
                new_eid = _lid()
                self._emit(LineageEntry(
                    entry_id=new_eid,
                    phase=f"gossip_round_{gossip_round}",
                    angle=a.angle,
                    content=a.summary,
                    parent_ids=(prev_eid,),
                    metadata={
                        "round": gossip_round,
                        "info_gain": round(gain, 4),
                        "worker_id": a.worker_id,
                        "output_chars": len(a.summary),
                        "failures_this_round": gossip_failures,
                    },
                ))
                worker_entry_ids[a.worker_id] = new_eid

            logger.info(
                "gossip_round=<%d>, failures=<%d>, info_gain=<%.3f> | gossip round complete",
                gossip_round, gossip_failures, gain,
            )

            await _emit_event({
                "type": "gossip_round",
                "round": gossip_round,
                "total_rounds": config.gossip_rounds,
                "info_gain": round(gain, 4),
                "failures": gossip_failures,
                "elapsed_s": round(time.monotonic() - phase_start, 1),
            })

            # Check cancellation between gossip rounds
            if _is_cancelled():
                metrics.gossip_rounds_executed = rounds_executed
                metrics.phase_times["gossip"] = time.monotonic() - phase_start
                metrics.total_elapsed_s = time.monotonic() - t0
                worker_summaries = {a.angle: a.summary for a in assignments}
                return SwarmResult(
                    synthesis=f"[cancelled after gossip round {gossip_round}]",
                    metrics=metrics,
                    worker_summaries=worker_summaries,
                    angles_detected=[a.angle for a in assignments],
                )

            # ── Gap extraction: scan worker outputs for research gaps ──
            gap_markers = [
                "need more data", "cannot resolve without", "need data on",
                "unexplained", "need bloodwork", "need practitioner",
                "requires further", "insufficient evidence", "data gap",
                "missing data", "no source found", "unresolved question",
            ]
            gaps_found: list[str] = []
            for a in assignments:
                for line in a.summary.split("\n"):
                    low = line.lower()
                    if any(marker in low for marker in gap_markers):
                        cleaned = line.strip()
                        if cleaned and len(cleaned) > 20:
                            gaps_found.append(cleaned)
            if gaps_found:
                await _emit_event({
                    "type": "research_gap",
                    "gaps": gaps_found[:10],
                    "round": gossip_round,
                })
                logger.info(
                    "gossip_round=<%d>, gaps=<%d> | research gaps emitted",
                    gossip_round, len(gaps_found),
                )

            # ── Corpus delta: fetch new findings from external producers ──
            delta_text = ""
            if config.corpus_delta_fn is not None:
                try:
                    delta_text = await config.corpus_delta_fn()
                    if delta_text:
                        logger.info(
                            "gossip_round=<%d>, delta_chars=<%d> | new findings injected for next round",
                            gossip_round, len(delta_text),
                        )
                        await _emit_event({
                            "type": "corpus_delta",
                            "round": gossip_round,
                            "delta_chars": len(delta_text),
                        })
                except Exception:
                    logger.warning(
                        "gossip_round=<%d> | corpus_delta_fn call failed, continuing without delta",
                        gossip_round,
                    )
                    delta_text = ""

            # Adaptive stopping: check convergence
            # Only after min_gossip_rounds (ensure workers get enough cross-referencing)
            # Skip if ANY failures — failed workers have unchanged summaries which
            # inflate Jaccard similarity and cause false convergence detection
            if (
                config.enable_adaptive_rounds
                and gossip_round >= config.min_gossip_rounds
                and gossip_round < config.gossip_rounds
            ):
                if gossip_failures > 0:
                    logger.warning(
                        "gossip_round=<%d>, failures=<%d> | skipping convergence check due to worker failures",
                        gossip_round, gossip_failures,
                    )
                else:
                    workers_converged = check_convergence(
                        current, previous, config.convergence_threshold,
                    )
                    # Live mode: convergence requires BOTH workers agree
                    # AND no new data arrived (convergence = silence)
                    new_data_arrived = bool(delta_text)
                    if workers_converged and not new_data_arrived:
                        logger.info(
                            "gossip_round=<%d> | convergence detected (workers + no new data), stopping early",
                            gossip_round,
                        )
                        metrics.gossip_converged_early = True
                        break
                    elif workers_converged and new_data_arrived:
                        logger.info(
                            "gossip_round=<%d> | workers converged but new data arrived, continuing",
                            gossip_round,
                        )

            # Hard ceiling: never exceed max_gossip_rounds
            if gossip_round >= config.max_gossip_rounds:
                logger.info(
                    "gossip_round=<%d>, max=<%d> | hard ceiling reached, stopping",
                    gossip_round, config.max_gossip_rounds,
                )
                break

        metrics.gossip_rounds_executed = rounds_executed
        metrics.phase_times["gossip"] = time.monotonic() - phase_start
        metrics.worker_output_chars = [len(a.summary) for a in assignments]

        # Build worker summaries dict for downstream use
        worker_summaries = {a.angle: a.summary for a in assignments}

        # ── Phase 3: Serendipity Bridge ──────────────────────────────
        # Wrapped in try/except: serendipity is optional bonus insight and
        # must not crash the pipeline, discarding expensive phases 0-2.
        serendipity_insights = ""
        serendipity_entry_id = ""
        if config.enable_serendipity and len(assignments) >= 2:
            phase_start = time.monotonic()
            try:
                serendipity_insights = await find_serendipitous_connections(
                    worker_summaries=worker_summaries,
                    query=query,
                    complete_fn=self.serendipity_complete,
                )
            except Exception:
                logger.warning("serendipity bridge failed, continuing without it")
                serendipity_insights = ""
                metrics.degradations.append("Serendipity bridge failed")
            metrics.total_llm_calls += 1
            metrics.serendipity_produced = bool(serendipity_insights)
            metrics.phase_times["serendipity"] = time.monotonic() - phase_start

            # Lineage: emit serendipity entry
            if serendipity_insights:
                serendipity_entry_id = _lid()
                self._emit(LineageEntry(
                    entry_id=serendipity_entry_id,
                    phase="serendipity",
                    content=serendipity_insights,
                    parent_ids=tuple(worker_entry_ids.values()),
                    metadata={"output_chars": len(serendipity_insights)},
                ))

            logger.info(
                "serendipity_chars=<%d> | serendipity bridge complete",
                len(serendipity_insights),
            )

            await _emit_event({
                "type": "swarm_phase",
                "phase": "serendipity",
                "chars": len(serendipity_insights),
                "elapsed_s": round(metrics.phase_times.get("serendipity", 0), 1),
            })

        # Check cancellation before queen merge
        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(
                synthesis="[cancelled before queen merge]",
                metrics=metrics,
                worker_summaries=worker_summaries,
                serendipity_insights=serendipity_insights,
                angles_detected=[a.angle for a in assignments],
            )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "queen_merge_start",
        })

        # ── Phase 4: Dual Output (parallel) ──────────────────────────
        # Build both reports simultaneously — they share the same inputs
        # but produce different outputs (knowledge = full, user = concise).
        phase_start = time.monotonic()

        knowledge_task = build_knowledge_report(
            worker_summaries=worker_summaries,
            query=query,
            complete_fn=self.queen_complete,
            serendipity_insights=serendipity_insights,
            corpus_chars=len(corpus),
            gossip_rounds=rounds_executed,
            converged_early=metrics.gossip_converged_early,
        )
        user_task = queen_merge(
            worker_summaries=worker_summaries,
            query=query,
            complete_fn=self.queen_complete,
            serendipity_insights=serendipity_insights,
            max_summary_chars=config.max_summary_chars,
        )

        knowledge_report, user_report = await asyncio.gather(
            knowledge_task, user_task,
        )
        metrics.total_llm_calls += 2  # one for knowledge exec summary, one for queen
        metrics.phase_times["queen_merge"] = time.monotonic() - phase_start
        metrics.total_elapsed_s = time.monotonic() - t0

        # Build quality manifest
        manifest = SwarmQualityManifest(
            corpus_chars=len(corpus),
            section_count=len(sections),
            worker_count=len(assignments),
            angles=[a.angle for a in assignments],
            gossip_rounds_executed=rounds_executed,
            gossip_rounds_configured=config.gossip_rounds,
            gossip_converged_early=metrics.gossip_converged_early,
            gossip_info_gain=list(metrics.gossip_info_gain),
            serendipity_produced=metrics.serendipity_produced,
            phase_times_s=dict(metrics.phase_times),
            total_llm_calls=metrics.total_llm_calls,
            total_elapsed_s=metrics.total_elapsed_s,
            worker_input_chars=list(metrics.worker_input_chars),
            worker_output_chars=list(metrics.worker_output_chars),
            degradations=list(metrics.degradations),
        )

        # Append quality manifest to reports
        if config.enable_quality_manifest:
            manifest_md = manifest.to_markdown()
            knowledge_report = knowledge_report + "\n\n" + manifest_md
            user_report = user_report + "\n\n" + manifest_md

        # Lineage: emit final report entries
        queen_parent_ids = list(worker_entry_ids.values())
        if serendipity_entry_id:
            queen_parent_ids.append(serendipity_entry_id)

        self._emit(LineageEntry(
            entry_id=_lid(),
            phase="queen_merge",
            content=user_report,
            parent_ids=tuple(queen_parent_ids),
            metadata={
                "output_chars": len(user_report),
                "report_type": "user",
            },
        ))
        self._emit(LineageEntry(
            entry_id=_lid(),
            phase="knowledge_report",
            content=knowledge_report,
            parent_ids=tuple(queen_parent_ids),
            metadata={
                "output_chars": len(knowledge_report),
                "report_type": "knowledge",
            },
        ))

        await _emit_event({
            "type": "swarm_phase",
            "phase": "queen_merge_complete",
            "user_chars": len(user_report),
            "knowledge_chars": len(knowledge_report),
            "elapsed_s": round(metrics.phase_times["queen_merge"], 1),
        })

        logger.info(
            "llm_calls=<%d>, elapsed_s=<%.1f>, workers=<%d>, gossip_rounds=<%d>, "
            "converged_early=<%s>, serendipity=<%s>, "
            "knowledge_chars=<%d>, user_chars=<%d>, "
            "info_gain_per_round=<%s>, degradations=<%d> | swarm synthesis complete",
            metrics.total_llm_calls, metrics.total_elapsed_s,
            metrics.total_workers, metrics.gossip_rounds_executed,
            metrics.gossip_converged_early, metrics.serendipity_produced,
            len(knowledge_report), len(user_report),
            [f"{g:.3f}" for g in metrics.gossip_info_gain],
            len(metrics.degradations),
        )

        return SwarmResult(
            synthesis=user_report,
            knowledge_report=knowledge_report,
            metrics=metrics,
            worker_summaries=worker_summaries,
            serendipity_insights=serendipity_insights,
            angles_detected=[a.angle for a in assignments],
            quality_manifest=manifest,
        )

    # ══════════════════════════════════════════════════════════════════
    # THREE-TIER HIERARCHICAL PIPELINE
    # ══════════════════════════════════════════════════════════════════
    #
    # Architecture:
    #
    #   Leaf Workers (tiny slices, ~15-25 findings)
    #       ↕ summaries only
    #   Angle Coordinators (one per top-level angle)
    #       ↕ condensed summaries only
    #   Cross-Angle Bridge Workers (mesh topology, CollisionStatements)
    #       ↕ collision statements only
    #   Serendipity Panel (concurrent polymaths, different lenses)
    #       ↕ top collisions + summaries only
    #   Queen (editor, never sees raw data)
    #
    # No agent at any tier ever holds the full corpus.

    async def _synthesize_hierarchical(
        self,
        corpus: str,
        query: str,
        on_event: "Callable[[dict], Awaitable[None]] | None" = None,
        cancel_event: "asyncio.Event | None" = None,
    ) -> SwarmResult:
        """Three-tier hierarchical pipeline for large-corpus synthesis.

        Phases:
          H0. Corpus analysis — detect angles, parse findings, sub-cluster
          H1. Leaf workers — each synthesizes a tiny slice (parallel)
          H2. Leaf gossip — leaf workers within same angle gossip (parallel)
          H3. Coordinators — aggregate leaf summaries per angle (parallel)
          H4. Bridge mesh — cross-angle conversation via CollisionStatements
          H5. Serendipity panel — concurrent polymaths with different lenses
          H6. Queen merge + lucidity pass — final narrative
        """
        t0 = time.monotonic()
        metrics = SwarmMetrics()
        config = self.config

        async def _emit_event(event: dict) -> None:
            if on_event is not None:
                try:
                    await on_event(event)
                except Exception:
                    pass

        def _is_cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        # ── H0: Corpus Analysis + Sub-clustering ─────────────────────
        phase_start = time.monotonic()

        # Detect angles
        angles = await detect_angles_via_llm(
            corpus, query, self.complete,
            max_angles=config.max_workers,
        )
        metrics.total_llm_calls += 1

        if not angles:
            sections = detect_sections(corpus)
            angles = [s.title for s in sections[:config.max_workers]]

        if not angles:
            return SwarmResult(
                synthesis="No content found in corpus to synthesize.",
                metrics=metrics,
            )

        # Parse corpus into individual findings
        findings = _parse_findings(corpus)

        if len(findings) < 2:
            # Too few findings for hierarchy — fall back to flat
            logger.info(
                "findings=<%d> | too few for hierarchy, falling back to flat pipeline",
                len(findings),
            )
            # Temporarily disable hierarchy to avoid recursion
            config.enable_hierarchy = False
            try:
                return await self.synthesize(corpus, query, on_event, cancel_event)
            finally:
                config.enable_hierarchy = True

        # Distribute findings across angles
        angle_to_findings = await distribute_findings_to_angles(
            findings, angles, self.complete,
        )
        metrics.total_llm_calls += 1

        # Sub-cluster each angle's findings into leaf clusters
        all_leaf_clusters: dict[str, list[LeafCluster]] = {}
        total_leaves = 0

        for i, angle in enumerate(angles):
            finding_indices = angle_to_findings.get(angle, [])
            angle_findings = [(findings[idx][0], findings[idx][1]) for idx in finding_indices]

            if not angle_findings:
                continue

            leaves = cluster_findings(
                findings=angle_findings,
                angle=angle,
                angle_index=i,
                max_leaf_size=config.max_leaf_size,
                min_leaf_size=config.min_leaf_size,
            )
            all_leaf_clusters[angle] = leaves
            total_leaves += len(leaves)

        metrics.leaf_clusters = total_leaves
        metrics.phase_times["h0_analysis"] = time.monotonic() - phase_start

        logger.info(
            "angles=<%d>, findings=<%d>, leaf_clusters=<%d> | "
            "hierarchical clustering complete",
            len(angles), len(findings), total_leaves,
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h0_analysis",
            "angles": angles,
            "findings": len(findings),
            "leaf_clusters": total_leaves,
            "elapsed_s": round(metrics.phase_times["h0_analysis"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(synthesis="[cancelled]", metrics=metrics)

        # Lineage: corpus analysis
        corpus_entry_id = _lid()
        self._emit(LineageEntry(
            entry_id=corpus_entry_id,
            phase="h0_analysis",
            content=f"Detected {len(angles)} angles, {len(findings)} findings, "
                    f"{total_leaves} leaf clusters",
            metadata={
                "corpus_chars": len(corpus),
                "angles": angles,
                "findings_count": len(findings),
                "leaf_clusters": total_leaves,
            },
        ))

        # ── H1: Leaf Workers (parallel synthesis of tiny slices) ─────
        phase_start = time.monotonic()
        sem = asyncio.Semaphore(config.max_concurrency)
        leaf_entry_ids: dict[str, str] = {}  # cluster_id -> lineage id

        # Collect all leaf worker tasks
        leaf_summaries: dict[str, dict[str, str]] = {
            angle: {} for angle in all_leaf_clusters
        }

        async def _leaf_synthesize(
            angle: str, cluster: LeafCluster,
        ) -> None:
            async with sem:
                try:
                    summary = await worker_synthesize(
                        angle=f"{angle} [{cluster.cluster_id}]",
                        section_content=cluster.raw_content,
                        query=query,
                        max_chars=config.max_summary_chars,
                        complete_fn=self.worker_complete,
                    )
                    leaf_summaries[angle][cluster.cluster_id] = summary
                except Exception:
                    logger.warning(
                        "cluster=<%s>, angle=<%s> | leaf synthesis failed, using truncation",
                        cluster.cluster_id, angle,
                    )
                    leaf_summaries[angle][cluster.cluster_id] = (
                        cluster.raw_content[:config.max_summary_chars]
                    )
                    metrics.degradations.append(
                        f"Leaf {cluster.cluster_id} synthesis failed"
                    )

        tasks = []
        for angle, clusters in all_leaf_clusters.items():
            for cluster in clusters:
                tasks.append(_leaf_synthesize(angle, cluster))

        await asyncio.gather(*tasks)
        metrics.total_llm_calls += total_leaves
        metrics.total_workers = total_leaves
        metrics.phase_times["h1_leaf_synth"] = time.monotonic() - phase_start

        # Lineage: leaf synthesis
        for angle, clusters in all_leaf_clusters.items():
            for cluster in clusters:
                eid = _lid()
                leaf_entry_ids[cluster.cluster_id] = eid
                summary = leaf_summaries.get(angle, {}).get(cluster.cluster_id, "")
                self._emit(LineageEntry(
                    entry_id=eid,
                    phase="h1_leaf_synth",
                    angle=angle,
                    content=summary,
                    parent_ids=(corpus_entry_id,),
                    metadata={
                        "cluster_id": cluster.cluster_id,
                        "input_findings": cluster.size,
                        "input_chars": cluster.char_count,
                        "output_chars": len(summary),
                    },
                ))

        logger.info(
            "leaf_workers=<%d>, elapsed_s=<%.1f> | leaf synthesis complete",
            total_leaves, metrics.phase_times["h1_leaf_synth"],
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h1_leaf_synth",
            "leaf_workers": total_leaves,
            "elapsed_s": round(metrics.phase_times["h1_leaf_synth"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(
                synthesis="[cancelled after leaf synthesis]",
                metrics=metrics,
                angles_detected=angles,
            )

        # ── H2: Leaf Gossip (within-angle conversation) ──────────────
        # Leaf workers within the same angle exchange summaries to find
        # intra-angle connections before coordinators aggregate
        phase_start = time.monotonic()
        gossip_rounds_executed = 0

        for gossip_round in range(1, config.gossip_rounds + 1):
            round_prompt = config.round_prompts.get(gossip_round, "")
            round_tasks = []

            for angle, clusters in all_leaf_clusters.items():
                if len(clusters) <= 1:
                    continue  # single leaf — no intra-angle gossip needed

                for cluster in clusters:
                    cid = cluster.cluster_id
                    own_summary = leaf_summaries[angle].get(cid, "")

                    # Peer summaries from other leaf clusters in same angle
                    peer_sums = [
                        leaf_summaries[angle][other_c.cluster_id]
                        for other_c in clusters
                        if other_c.cluster_id != cid
                        and leaf_summaries[angle].get(other_c.cluster_id)
                    ]

                    if not peer_sums:
                        continue

                    async def _leaf_gossip(
                        _angle: str = angle,
                        _cid: str = cid,
                        _own: str = own_summary,
                        _peers: list[str] = peer_sums,
                        _raw: str = cluster.raw_content,
                        _rp: str = round_prompt,
                    ) -> None:
                        async with sem:
                            try:
                                refined = await worker_gossip_refine(
                                    angle=f"{_angle} [{_cid}]",
                                    own_summary=_own,
                                    peer_summaries=_peers,
                                    raw_section=_raw if config.enable_full_corpus_gossip else None,
                                    query=query,
                                    max_chars=config.max_summary_chars,
                                    complete_fn=self.worker_complete,
                                    round_prompt=_rp,
                                )
                                leaf_summaries[_angle][_cid] = refined
                            except Exception:
                                logger.warning(
                                    "cluster=<%s>, round=<%d> | leaf gossip failed",
                                    _cid, gossip_round,
                                )

                    round_tasks.append(_leaf_gossip())

            if round_tasks:
                await asyncio.gather(*round_tasks)
                metrics.total_llm_calls += len(round_tasks)

            gossip_rounds_executed = gossip_round

        metrics.gossip_rounds_executed = gossip_rounds_executed
        metrics.phase_times["h2_leaf_gossip"] = time.monotonic() - phase_start

        logger.info(
            "leaf_gossip_rounds=<%d>, elapsed_s=<%.1f> | leaf gossip complete",
            gossip_rounds_executed, metrics.phase_times["h2_leaf_gossip"],
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h2_leaf_gossip",
            "rounds": gossip_rounds_executed,
            "elapsed_s": round(metrics.phase_times["h2_leaf_gossip"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(
                synthesis="[cancelled after leaf gossip]",
                metrics=metrics,
                angles_detected=angles,
            )

        # ── H3: Angle Coordinators (aggregate leaf summaries) ────────
        phase_start = time.monotonic()

        angle_summaries = await coordinate_all_angles(
            angle_leaf_summaries=leaf_summaries,
            query=query,
            max_chars=config.coordinator_max_chars,
            complete_fn=self.worker_complete,
            max_concurrency=config.max_concurrency,
        )
        # Count LLM calls: one per angle with >1 leaf, zero for single-leaf
        coord_calls = sum(
            1 for angle in all_leaf_clusters
            if len(all_leaf_clusters[angle]) > 1
        )
        metrics.total_llm_calls += coord_calls
        metrics.phase_times["h3_coordinators"] = time.monotonic() - phase_start

        # Lineage: coordinator entries
        coord_entry_ids: dict[str, str] = {}
        for angle, summary in angle_summaries.items():
            eid = _lid()
            coord_entry_ids[angle] = eid
            parent_ids = tuple(
                leaf_entry_ids[c.cluster_id]
                for c in all_leaf_clusters.get(angle, [])
                if c.cluster_id in leaf_entry_ids
            )
            self._emit(LineageEntry(
                entry_id=eid,
                phase="h3_coordinator",
                angle=angle,
                content=summary,
                parent_ids=parent_ids,
                metadata={
                    "leaf_count": len(all_leaf_clusters.get(angle, [])),
                    "output_chars": len(summary),
                },
            ))

        logger.info(
            "coordinators=<%d>, elapsed_s=<%.1f> | coordinator aggregation complete",
            len(angle_summaries), metrics.phase_times["h3_coordinators"],
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h3_coordinators",
            "angles": list(angle_summaries.keys()),
            "elapsed_s": round(metrics.phase_times["h3_coordinators"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(
                synthesis="[cancelled after coordinators]",
                metrics=metrics,
                worker_summaries=angle_summaries,
                angles_detected=angles,
            )

        # ── H4: Bridge Workers (cross-angle mesh conversation) ───────
        phase_start = time.monotonic()

        # Create one bridge worker per angle
        bridge_workers_list: list[BridgeWorker] = []
        for angle, summary in angle_summaries.items():
            bw = BridgeWorker(
                name=f"Bridge-{angle}",
                angle=angle,
                angle_summary=summary,
            )
            bridge_workers_list.append(bw)

        metrics.bridge_workers = len(bridge_workers_list)

        # Run mesh rounds
        all_collisions: list[CollisionStatement] = []
        if len(bridge_workers_list) >= 2:
            all_collisions = await run_mesh_rounds(
                bridge_workers=bridge_workers_list,
                complete_fn=self.worker_complete,
                num_rounds=config.bridge_rounds,
                max_concurrency=config.max_concurrency,
                peer_summaries=angle_summaries,
                on_event=on_event,
            )
            # Each bridge worker makes one LLM call per round
            metrics.total_llm_calls += (
                len(bridge_workers_list) * config.bridge_rounds
            )

        metrics.collisions_produced = len(all_collisions)
        metrics.phase_times["h4_bridge_mesh"] = time.monotonic() - phase_start

        # Lineage: bridge mesh
        bridge_entry_id = ""
        if all_collisions:
            bridge_entry_id = _lid()
            collision_summary = "\n".join(
                c.to_prompt_line() for c in all_collisions[:20]
            )
            self._emit(LineageEntry(
                entry_id=bridge_entry_id,
                phase="h4_bridge_mesh",
                content=collision_summary,
                parent_ids=tuple(coord_entry_ids.values()),
                metadata={
                    "bridge_workers": len(bridge_workers_list),
                    "rounds": config.bridge_rounds,
                    "collisions": len(all_collisions),
                    "top_score": all_collisions[0].serendipity_score if all_collisions else 0,
                },
            ))

        logger.info(
            "bridge_workers=<%d>, rounds=<%d>, collisions=<%d>, "
            "elapsed_s=<%.1f> | bridge mesh complete",
            len(bridge_workers_list), config.bridge_rounds,
            len(all_collisions), metrics.phase_times["h4_bridge_mesh"],
        )

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h4_bridge_mesh",
            "bridge_workers": len(bridge_workers_list),
            "collisions": len(all_collisions),
            "elapsed_s": round(metrics.phase_times["h4_bridge_mesh"], 1),
        })

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(
                synthesis="[cancelled after bridge mesh]",
                metrics=metrics,
                worker_summaries=angle_summaries,
                angles_detected=angles,
                collision_statements=all_collisions,
            )

        # ── H5: Serendipity Panel (concurrent polymaths) ─────────────
        serendipity_insights = ""
        serendipity_entry_id = ""

        if config.enable_serendipity_panel and len(angle_summaries) >= 2:
            phase_start = time.monotonic()
            try:
                serendipity_insights = await run_serendipity_panel(
                    angle_summaries=angle_summaries,
                    collisions=all_collisions,
                    query=query,
                    complete_fn=self.serendipity_complete,
                    max_concurrency=config.max_concurrency,
                    max_panel_size=config.serendipity_panel_size,
                )
            except Exception:
                logger.warning("serendipity panel failed, continuing without it")
                metrics.degradations.append("Serendipity panel failed")

            panel_calls = min(config.serendipity_panel_size, 5)
            metrics.total_llm_calls += panel_calls
            metrics.serendipity_produced = bool(serendipity_insights)
            metrics.panel_insights_produced = len(serendipity_insights)
            metrics.phase_times["h5_serendipity"] = time.monotonic() - phase_start

            if serendipity_insights:
                serendipity_entry_id = _lid()
                parent_ids = list(coord_entry_ids.values())
                if bridge_entry_id:
                    parent_ids.append(bridge_entry_id)
                self._emit(LineageEntry(
                    entry_id=serendipity_entry_id,
                    phase="h5_serendipity",
                    content=serendipity_insights,
                    parent_ids=tuple(parent_ids),
                    metadata={
                        "panel_size": config.serendipity_panel_size,
                        "output_chars": len(serendipity_insights),
                    },
                ))

            logger.info(
                "panel_size=<%d>, insights_chars=<%d>, elapsed_s=<%.1f> | "
                "serendipity panel complete",
                config.serendipity_panel_size, len(serendipity_insights),
                metrics.phase_times.get("h5_serendipity", 0),
            )

            await _emit_event({
                "type": "swarm_phase",
                "phase": "h5_serendipity",
                "panel_size": config.serendipity_panel_size,
                "insights_chars": len(serendipity_insights),
                "elapsed_s": round(metrics.phase_times.get("h5_serendipity", 0), 1),
            })
        elif config.enable_serendipity and len(angle_summaries) >= 2:
            # Fall back to single polymath if panel disabled
            phase_start = time.monotonic()
            try:
                serendipity_insights = await find_serendipitous_connections(
                    worker_summaries=angle_summaries,
                    query=query,
                    complete_fn=self.serendipity_complete,
                )
            except Exception:
                logger.warning("serendipity bridge failed, continuing without it")
            metrics.total_llm_calls += 1
            metrics.serendipity_produced = bool(serendipity_insights)
            metrics.phase_times["h5_serendipity"] = time.monotonic() - phase_start

        if _is_cancelled():
            metrics.total_elapsed_s = time.monotonic() - t0
            return SwarmResult(
                synthesis="[cancelled before queen merge]",
                metrics=metrics,
                worker_summaries=angle_summaries,
                serendipity_insights=serendipity_insights,
                angles_detected=angles,
                collision_statements=all_collisions,
            )

        # ── H6: Queen Merge + Lucidity Pass ──────────────────────────
        phase_start = time.monotonic()

        # Prepare collision text for the queen
        collision_text = ""
        if all_collisions:
            top = sorted(all_collisions, key=lambda c: c.serendipity_score, reverse=True)[:15]
            collision_text = (
                "\n\nCROSS-DOMAIN COLLISION INSIGHTS (from bridge worker conversation):\n"
                + "\n".join(c.to_prompt_line() for c in top)
            )

        # Build enriched worker summaries with collision context
        enriched_summaries = {}
        for angle, summary in angle_summaries.items():
            enriched_summaries[angle] = summary + collision_text

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h6_queen_start",
        })

        # Dual output: knowledge report + user report (parallel)
        knowledge_task = build_knowledge_report(
            worker_summaries=enriched_summaries,
            query=query,
            complete_fn=self.queen_complete,
            serendipity_insights=serendipity_insights,
            corpus_chars=len(corpus),
            gossip_rounds=gossip_rounds_executed,
            converged_early=False,
        )
        user_task = queen_merge(
            worker_summaries=enriched_summaries,
            query=query,
            complete_fn=self.queen_complete,
            serendipity_insights=serendipity_insights,
            max_summary_chars=config.max_summary_chars,
        )

        knowledge_report, user_report = await asyncio.gather(
            knowledge_task, user_task,
        )
        metrics.total_llm_calls += 2

        # Lucidity pass: multi-pass editing for clarity
        if config.enable_queen_lucidity_pass and user_report:
            try:
                user_report = await queen_lucidity_pass(
                    report=user_report,
                    query=query,
                    complete_fn=self.queen_complete,
                )
                metrics.total_llm_calls += 1
            except Exception:
                logger.warning("queen lucidity pass failed, using unpolished report")
                metrics.degradations.append("Queen lucidity pass failed")

        metrics.phase_times["h6_queen"] = time.monotonic() - phase_start
        metrics.total_elapsed_s = time.monotonic() - t0

        # Build quality manifest
        manifest = SwarmQualityManifest(
            corpus_chars=len(corpus),
            section_count=len(angles),
            worker_count=total_leaves,
            angles=angles,
            gossip_rounds_executed=gossip_rounds_executed,
            gossip_rounds_configured=config.gossip_rounds,
            gossip_converged_early=False,
            gossip_info_gain=list(metrics.gossip_info_gain),
            serendipity_produced=metrics.serendipity_produced,
            phase_times_s=dict(metrics.phase_times),
            total_llm_calls=metrics.total_llm_calls,
            total_elapsed_s=metrics.total_elapsed_s,
            worker_input_chars=[],
            worker_output_chars=[len(s) for s in angle_summaries.values()],
            degradations=list(metrics.degradations),
        )

        if config.enable_quality_manifest:
            manifest_md = manifest.to_markdown()
            knowledge_report = knowledge_report + "\n\n" + manifest_md
            user_report = user_report + "\n\n" + manifest_md

        # Lineage: queen entries
        queen_parent_ids = list(coord_entry_ids.values())
        if bridge_entry_id:
            queen_parent_ids.append(bridge_entry_id)
        if serendipity_entry_id:
            queen_parent_ids.append(serendipity_entry_id)

        self._emit(LineageEntry(
            entry_id=_lid(),
            phase="h6_queen",
            content=user_report,
            parent_ids=tuple(queen_parent_ids),
            metadata={
                "output_chars": len(user_report),
                "report_type": "user",
                "lucidity_pass": config.enable_queen_lucidity_pass,
            },
        ))
        self._emit(LineageEntry(
            entry_id=_lid(),
            phase="h6_knowledge",
            content=knowledge_report,
            parent_ids=tuple(queen_parent_ids),
            metadata={
                "output_chars": len(knowledge_report),
                "report_type": "knowledge",
            },
        ))

        await _emit_event({
            "type": "swarm_phase",
            "phase": "h6_queen_complete",
            "user_chars": len(user_report),
            "knowledge_chars": len(knowledge_report),
            "elapsed_s": round(metrics.phase_times["h6_queen"], 1),
        })

        logger.info(
            "topology=<hierarchical>, llm_calls=<%d>, elapsed_s=<%.1f>, "
            "angles=<%d>, leaves=<%d>, bridges=<%d>, collisions=<%d>, "
            "serendipity=<%s>, knowledge_chars=<%d>, user_chars=<%d>, "
            "degradations=<%d> | hierarchical synthesis complete",
            metrics.total_llm_calls, metrics.total_elapsed_s,
            len(angles), total_leaves, len(bridge_workers_list),
            len(all_collisions), metrics.serendipity_produced,
            len(knowledge_report), len(user_report),
            len(metrics.degradations),
        )

        return SwarmResult(
            synthesis=user_report,
            knowledge_report=knowledge_report,
            metrics=metrics,
            worker_summaries=angle_summaries,
            serendipity_insights=serendipity_insights,
            angles_detected=angles,
            quality_manifest=manifest,
            collision_statements=all_collisions,
        )
