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
from swarm.convergence import check_convergence, information_gain
from swarm.queen import build_knowledge_report, queen_merge
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
    gossip_info_gain: list[float] = field(default_factory=list)


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
        sem = asyncio.Semaphore(config.max_concurrency)

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

            gossip_failures = 0

            # Get round-specific prompt modifier (if any)
            round_prompt = config.round_prompts.get(gossip_round, "")

            async def _bounded_gossip(
                assignment: WorkerAssignment,
                _round_prompt: str = round_prompt,
            ) -> None:
                nonlocal gossip_failures
                async with sem:
                    peer_sums = [
                        other.summary for other in assignments
                        if other.worker_id != assignment.worker_id and other.summary
                    ]
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
                        )
                    except Exception:
                        gossip_failures += 1
                        logger.warning(
                            "worker_id=<%d>, angle=<%s>, round=<%d> | gossip refinement failed, keeping previous summary",
                            assignment.worker_id, assignment.angle, gossip_round,
                        )

            await asyncio.gather(*[_bounded_gossip(a) for a in assignments])
            metrics.total_llm_calls += len(assignments)
            rounds_executed = gossip_round

            # Measure information gain this round
            current = [a.summary for a in assignments]
            previous = [a.prev_summary for a in assignments]
            gain = information_gain(current, previous)
            metrics.gossip_info_gain.append(gain)

            logger.info(
                "gossip_round=<%d>, failures=<%d>, info_gain=<%.3f> | gossip round complete",
                gossip_round, gossip_failures, gain,
            )

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
        # Wrapped in try/except: serendipity is optional bonus insight and
        # must not crash the pipeline, discarding expensive phases 0-2.
        serendipity_insights = ""
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
            metrics.total_llm_calls += 1
            metrics.serendipity_produced = bool(serendipity_insights)
            metrics.phase_times["serendipity"] = time.monotonic() - phase_start

            logger.info(
                "serendipity_chars=<%d> | serendipity bridge complete",
                len(serendipity_insights),
            )

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

        logger.info(
            "llm_calls=<%d>, elapsed_s=<%.1f>, workers=<%d>, gossip_rounds=<%d>, "
            "converged_early=<%s>, serendipity=<%s>, "
            "knowledge_chars=<%d>, user_chars=<%d>, "
            "info_gain_per_round=<%s> | swarm synthesis complete",
            metrics.total_llm_calls, metrics.total_elapsed_s,
            metrics.total_workers, metrics.gossip_rounds_executed,
            metrics.gossip_converged_early, metrics.serendipity_produced,
            len(knowledge_report), len(user_report),
            [f"{g:.3f}" for g in metrics.gossip_info_gain],
        )

        return SwarmResult(
            synthesis=user_report,
            knowledge_report=knowledge_report,
            metrics=metrics,
            worker_summaries=worker_summaries,
            serendipity_insights=serendipity_insights,
            angles_detected=[a.angle for a in assignments],
        )
