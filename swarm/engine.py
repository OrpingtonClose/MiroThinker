# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""GossipSwarm — the main orchestrator for parallel corpus synthesis.

Architecture (angle-based gossip with full-corpus access):

    ┌──────────────────────────────────────────────────────────┐
    │                    QUEEN (merge)                          │
    │   Reads all gossip-refined summaries + serendipity        │
    │   insights → produces final unified synthesis             │
    └──────────┬──────────┬──────────┬──────────┬─────────────┘
               │          │          │          │
    ┌──────────▼──┐ ┌─────▼─────┐ ┌──▼────────┐ │
    │  Worker A   │ │ Worker B  │ │ Worker C  │ ...
    │  Angle 1    │ │ Angle 2   │ │ Angle 3   │
    │  + raw data │ │ + raw data│ │ + raw data│
    └──────┬──────┘ └──────┬────┘ └──────┬────┘
           │               │             │
           └───────────────┼─────────────┘
                    Gossip Round(s)
              (each worker reads peers'
               summaries + own raw section,
               refines with cross-references)
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
from datetime import datetime, timezone

from swarm.angles import (
    WorkerAssignment,
    apply_misassignment,
    assign_workers,
    detect_angles_via_llm,
    detect_sections,
    extract_required_angles,
    merge_angles,
    score_section_angle_pairs,
    _prepare_sections,
)
from swarm.config import CompleteFn, SwarmConfig
from swarm.convergence import check_convergence, information_gain, select_diverse_peers
from swarm.lineage import LineageEntry
from swarm.quality_manifest import SwarmQualityManifest
from swarm.queen import build_knowledge_report, diffusion_queen_merge, queen_merge
from swarm.rag import extract_concepts, query_hive
from swarm.serendipity import find_serendipitous_connections
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
        # Local lineage entry cache for swarm-internal RAG (hive memory).
        # Reset at the start of each synthesize() call.
        self._lineage_entries: list[LineageEntry] = []

    def _emit(self, entry: LineageEntry) -> None:
        """Emit a lineage entry if a store is configured.

        Also tracks entries locally for swarm-internal RAG (hive memory).
        """
        self._lineage_entries.append(entry)
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
        t0 = time.monotonic()
        metrics = SwarmMetrics()
        metrics.gossip_rounds_configured = self.config.gossip_rounds
        config = self.config

        # Reset local lineage cache for this run
        self._lineage_entries = []

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

        # Extract required angles from the user's query (prompt-driven
        # guarantee).  These angles are always included regardless of what
        # the corpus contains — prevents underrepresented topics from being
        # absorbed into dominant ones.
        required_angles = list(config.required_angles)
        if not required_angles:
            required_angles = await extract_required_angles(
                query, self.complete,
            )
            metrics.total_llm_calls += 1

        if required_angles:
            logger.info(
                "required_angles=<%s> | prompt-driven angle guarantee active",
                required_angles,
            )

        # Detect angles via LLM (or fall back to section titles)
        detected_angles = await detect_angles_via_llm(
            corpus, query, self.complete,
            max_angles=config.max_workers,
        )
        metrics.total_llm_calls += 1 if detected_angles else 0

        # Merge: required angles first, then detected angles fill remaining
        angles = merge_angles(
            detected=detected_angles or [s.title for s in sections[:config.max_workers]],
            required=required_angles,
            max_angles=config.max_workers,
        )

        if not angles:
            angles = [s.title for s in sections[:config.max_workers]]

        # Semantic assignment: score section-angle pairs via LLM
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

        # Assign workers
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

        # ── Deliberate Misassignment ─────────────────────────────────
        # Inject off-angle raw data into each bee's slice.  The off-angle
        # portion (20-30%) is where thread discovery happens: the bee's
        # worldview activates on foreign data that other specialists
        # would have overlooked.
        misassignment_applied = False
        if config.enable_misassignment and len(assignments) >= 2:
            apply_misassignment(
                assignments,
                score_matrix=score_matrix,
                ratio=config.misassignment_ratio,
            )
            misassignment_applied = True
            logger.info(
                "misassignment_ratio=<%.2f>, workers=<%d> | "
                "off-angle data injected for thread discovery",
                config.misassignment_ratio, len(assignments),
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
                "query": query,
            },
        ))

        # ── Phase 1: Map (parallel worker synthesis) ─────────────────
        phase_start = time.monotonic()
        sem = asyncio.Semaphore(config.max_concurrency)
        worker_entry_ids: dict[int, str] = {}

        _has_off_angle = misassignment_applied

        async def _bounded_synthesize(assignment: WorkerAssignment) -> None:
            async with sem:
                try:
                    assignment.summary = await worker_synthesize(
                        angle=assignment.angle,
                        section_content=assignment.raw_content,
                        query=query,
                        max_chars=config.max_summary_chars,
                        complete_fn=self.worker_complete,
                        has_off_angle_data=_has_off_angle,
                    )
                except Exception:
                    logger.warning(
                        "worker_id=<%d>, angle=<%s> | worker synthesis failed, using raw content",
                        assignment.worker_id, assignment.angle,
                    )
                    assignment.summary = assignment.raw_content[:config.max_summary_chars]
                    metrics.degradations.append(
                        f"Worker {assignment.angle} synthesis failed, used raw content"
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

            # ── Hive Memory RAG ──────────────────────────────────────
            # Starting from round 2, query accumulated lineage entries
            # for findings relevant to each bee's current analysis.
            # Round 1 has no prior output to query against.
            hive_memory_map: dict[int, str] = {}
            if (
                config.enable_hive_memory
                and gossip_round >= 2
                and self._lineage_entries
            ):
                for a in assignments:
                    if not a.summary:
                        continue
                    concepts = extract_concepts(a.summary, top_k=15)
                    if not concepts:
                        continue
                    hive_results = query_hive(
                        entries=self._lineage_entries,
                        concepts=concepts,
                        exclude_angle=a.angle,
                        top_k=config.hive_memory_top_k,
                    )
                    if hive_results:
                        hive_memory_map[a.worker_id] = "\n\n".join(hive_results)
                if hive_memory_map:
                    logger.info(
                        "gossip_round=<%d>, hive_hits=<%d> | "
                        "hive memory injected for %d workers",
                        gossip_round, len(hive_memory_map),
                        len(hive_memory_map),
                    )

            async def _bounded_gossip(
                assignment: WorkerAssignment,
                _round_prompt: str = round_prompt,
                _delta: str = _round_delta,
                _hive_map: dict[int, str] = hive_memory_map,
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

                    # Hive memory for this worker (from RAG query above)
                    hive_mem = _hive_map.get(assignment.worker_id, "")

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
                            hive_memory=hive_mem,
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
                    "gaps": gaps_found,
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
            serendipity_llm_calls = 0
            try:
                serendipity_insights, serendipity_llm_calls = await find_serendipitous_connections(
                    worker_summaries=worker_summaries,
                    query=query,
                    complete_fn=self.serendipity_complete,
                )
            except Exception:
                logger.warning("serendipity bridge failed, continuing without it")
                serendipity_insights = ""
                metrics.degradations.append("Serendipity bridge failed")
            metrics.total_llm_calls += serendipity_llm_calls
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
                "serendipity_chars=<%d> | two-pass serendipity bridge complete",
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
        if config.enable_diffusion_queen:
            user_task = diffusion_queen_merge(
                worker_summaries=worker_summaries,
                query=query,
                complete_fn=self.queen_complete,
                serendipity_insights=serendipity_insights,
                max_passes=config.diffusion_max_passes,
                convergence_threshold=config.convergence_threshold,
                reviewers_per_section=config.diffusion_reviewers_per_section,
            )
        else:
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
        # LLM calls: 1 for knowledge exec summary + N for diffusion (or 1 for single-shot)
        diffusion_calls = (
            1  # scaffold
            + len(worker_summaries) * config.diffusion_max_passes  # manifest
            + len(worker_summaries) * config.diffusion_reviewers_per_section
            * config.diffusion_max_passes  # confront
            + len(worker_summaries) * config.diffusion_max_passes  # correct
            + 1  # stitch
        ) if config.enable_diffusion_queen else 1
        metrics.total_llm_calls += 1 + diffusion_calls
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

        # ── Observability: persist metrics to lineage store ──────────
        store = self.config.lineage_store
        if store is not None and hasattr(store, "emit_metric"):
            run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            store.emit_metric(
                "run_metric",
                {
                    "total_llm_calls": metrics.total_llm_calls,
                    "total_workers": metrics.total_workers,
                    "gossip_rounds_executed": metrics.gossip_rounds_executed,
                    "gossip_rounds_configured": metrics.gossip_rounds_configured,
                    "gossip_converged_early": metrics.gossip_converged_early,
                    "serendipity_produced": metrics.serendipity_produced,
                    "phase_times": dict(metrics.phase_times),
                    "total_elapsed_s": round(metrics.total_elapsed_s, 1),
                    "worker_input_chars": list(metrics.worker_input_chars),
                    "worker_output_chars": list(metrics.worker_output_chars),
                    "gossip_info_gain": [
                        round(g, 4) for g in metrics.gossip_info_gain
                    ],
                    "degradations": list(metrics.degradations),
                    "corpus_chars": len(corpus),
                    "angles": [a.angle for a in assignments],
                },
                source_run=run_id,
            )
            # Per-worker metrics
            for a in assignments:
                store.emit_metric(
                    "worker_metric",
                    {
                        "angle": a.angle,
                        "input_chars": a.char_count,
                        "output_chars": len(a.summary),
                    },
                    angle=a.angle,
                    source_run=run_id,
                )
            # Store health snapshot
            if hasattr(store, "store_health_snapshot"):
                store.store_health_snapshot(source_run=run_id)

        return SwarmResult(
            synthesis=user_report,
            knowledge_report=knowledge_report,
            metrics=metrics,
            worker_summaries=worker_summaries,
            serendipity_insights=serendipity_insights,
            angles_detected=[a.angle for a in assignments],
            quality_manifest=manifest,
        )
