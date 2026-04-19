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
  3. Serendipity — polymath finds cross-angle surprises (sequential)
  4. Queen Merge — combines everything into final synthesis (sequential)

With parallel execution: phases 1+2 are embarrassingly parallel.
Only phases 3+4 must be sequential.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from swarm.angles import (
    WorkerAssignment,
    assign_workers,
    detect_angles_via_llm,
    detect_sections,
)
from swarm.config import CompleteFn, SwarmConfig
from swarm.convergence import check_convergence
from swarm.queen import queen_merge
from swarm.serendipity import find_serendipitous_connections
from swarm.worker import worker_gossip_refine, worker_synthesize

logger = logging.getLogger(__name__)


@dataclass
class SwarmMetrics:
    """Telemetry from a swarm synthesis run."""

    total_llm_calls: int = 0
    total_workers: int = 0
    gossip_rounds_executed: int = 0
    gossip_converged_early: bool = False
    serendipity_produced: bool = False
    phase_times: dict[str, float] = field(default_factory=dict)
    worker_input_chars: list[int] = field(default_factory=list)
    worker_output_chars: list[int] = field(default_factory=list)
    total_elapsed_s: float = 0.0


@dataclass
class SwarmResult:
    """Result of a gossip swarm synthesis."""

    synthesis: str
    metrics: SwarmMetrics
    worker_summaries: dict[str, str] = field(default_factory=dict)
    serendipity_insights: str = ""
    angles_detected: list[str] = field(default_factory=list)


class GossipSwarm:
    """Parallel corpus synthesis engine with gossip refinement.

    Model-agnostic: accepts any async callable that takes a prompt string
    and returns a completion string. Works with Venice API, local Ollama,
    the proxy layer, or any other LLM backend.

    Usage:
        async def my_llm(prompt: str) -> str:
            # call your LLM here
            return response

        swarm = GossipSwarm(complete=my_llm)
        result = await swarm.synthesize(corpus="...", query="...")
        print(result.synthesis)
    """

    def __init__(
        self,
        complete: CompleteFn,
        config: SwarmConfig | None = None,
    ) -> None:
        self.complete = complete
        self.config = config or SwarmConfig()

    async def synthesize(
        self,
        corpus: str,
        query: str,
    ) -> SwarmResult:
        """Run the full gossip swarm pipeline on the given corpus.

        Args:
            corpus: The full text corpus to synthesize.
            query: The user's research query.

        Returns:
            SwarmResult with the final synthesis, metrics, and intermediate data.
        """
        t0 = time.monotonic()
        metrics = SwarmMetrics()
        config = self.config

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

        # Assign workers
        assignments = assign_workers(
            sections, angles,
            max_workers=config.max_workers,
            max_section_chars=config.max_section_chars,
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

        # ── Phase 1: Map (parallel worker synthesis) ─────────────────
        phase_start = time.monotonic()
        sem = asyncio.Semaphore(config.max_workers)

        async def _bounded_synthesize(assignment: WorkerAssignment) -> None:
            async with sem:
                assignment.summary = await worker_synthesize(
                    angle=assignment.angle,
                    section_content=assignment.raw_content,
                    query=query,
                    max_chars=config.max_summary_chars,
                    complete_fn=self.complete,
                )

        await asyncio.gather(*[_bounded_synthesize(a) for a in assignments])
        metrics.total_llm_calls += len(assignments)
        metrics.phase_times["map"] = time.monotonic() - phase_start

        logger.info(
            "map_phase_s=<%.1f>, workers=<%d> | map phase complete",
            metrics.phase_times["map"], len(assignments),
        )

        # ── Phase 2: Gossip Rounds (parallel, adaptive) ─────────────
        phase_start = time.monotonic()
        rounds_executed = 0

        for gossip_round in range(1, config.gossip_rounds + 1):
            # Save previous summaries for convergence check
            for a in assignments:
                a.prev_summary = a.summary

            async def _bounded_gossip(assignment: WorkerAssignment) -> None:
                async with sem:
                    peer_sums = [
                        other.summary for other in assignments
                        if other.worker_id != assignment.worker_id and other.summary
                    ]
                    # Full-corpus gossip: pass raw section content
                    raw = assignment.raw_content if config.enable_full_corpus_gossip else None

                    assignment.summary = await worker_gossip_refine(
                        angle=assignment.angle,
                        own_summary=assignment.summary,
                        peer_summaries=peer_sums,
                        raw_section=raw,
                        query=query,
                        max_chars=config.max_summary_chars,
                        complete_fn=self.complete,
                    )

            await asyncio.gather(*[_bounded_gossip(a) for a in assignments])
            metrics.total_llm_calls += len(assignments)
            rounds_executed = gossip_round

            logger.info(
                "gossip_round=<%d> | gossip round complete",
                gossip_round,
            )

            # Adaptive stopping: check convergence
            if config.enable_adaptive_rounds and gossip_round < config.gossip_rounds:
                current = [a.summary for a in assignments]
                previous = [a.prev_summary for a in assignments]
                if check_convergence(current, previous, config.convergence_threshold):
                    logger.info(
                        "gossip_round=<%d> | convergence detected, stopping early",
                        gossip_round,
                    )
                    metrics.gossip_converged_early = True
                    break

        metrics.gossip_rounds_executed = rounds_executed
        metrics.phase_times["gossip"] = time.monotonic() - phase_start
        metrics.worker_output_chars = [len(a.summary) for a in assignments]

        # Build worker summaries dict for downstream use
        worker_summaries = {a.angle: a.summary for a in assignments}

        # ── Phase 3: Serendipity Bridge ──────────────────────────────
        serendipity_insights = ""
        if config.enable_serendipity and len(assignments) >= 2:
            phase_start = time.monotonic()
            serendipity_insights = await find_serendipitous_connections(
                worker_summaries=worker_summaries,
                query=query,
                complete_fn=self.complete,
            )
            metrics.total_llm_calls += 1
            metrics.serendipity_produced = bool(serendipity_insights)
            metrics.phase_times["serendipity"] = time.monotonic() - phase_start

            logger.info(
                "serendipity_chars=<%d> | serendipity bridge complete",
                len(serendipity_insights),
            )

        # ── Phase 4: Queen Merge ─────────────────────────────────────
        phase_start = time.monotonic()
        final_synthesis = await queen_merge(
            worker_summaries=worker_summaries,
            query=query,
            complete_fn=self.complete,
            serendipity_insights=serendipity_insights,
            max_summary_chars=config.max_summary_chars,
        )
        metrics.total_llm_calls += 1
        metrics.phase_times["queen_merge"] = time.monotonic() - phase_start
        metrics.total_elapsed_s = time.monotonic() - t0

        logger.info(
            "llm_calls=<%d>, elapsed_s=<%.1f>, workers=<%d>, gossip_rounds=<%d>, "
            "converged_early=<%s>, serendipity=<%s> | swarm synthesis complete",
            metrics.total_llm_calls, metrics.total_elapsed_s,
            metrics.total_workers, metrics.gossip_rounds_executed,
            metrics.gossip_converged_early, metrics.serendipity_produced,
        )

        return SwarmResult(
            synthesis=final_synthesis,
            metrics=metrics,
            worker_summaries=worker_summaries,
            serendipity_insights=serendipity_insights,
            angles_detected=[a.angle for a in assignments],
        )
