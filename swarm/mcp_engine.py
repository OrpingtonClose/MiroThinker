# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MCP-driven swarm engine — workers are agents with ConditionStore tools.

Replaces the single-shot prompt-stuffing architecture with agent-workers
that explore the corpus via tool calls.  Each worker is a Strands Agent
with tools to search, discover peer findings, and store its analysis.

24-hour continuous operation:
    - Corpus fingerprinting: skip re-ingestion if corpus already in store
    - Token budgets: every tool call return is capped to prevent overflow
    - Source provenance: every finding tagged with model + run number
    - Store compaction: periodic deduplication to prevent unbounded growth
    - Rolling summarization: knowledge briefings compress prior runs
    - Report generation with token budget: prompt capped regardless of
      store size

Architecture:

    ┌─────────────────────────────────────────────────┐
    │              MCPSwarmEngine                       │
    │  0. Check corpus fingerprint (skip if seen)     │
    │  1. Ingest corpus into ConditionStore            │
    │  2. Create N agent-workers (one per angle)       │
    │  3. Run workers in parallel waves                │
    │  4. Check convergence (store growth rate)        │
    │  5. Compact store (periodic dedup)               │
    │  6. Generate rolling summaries                   │
    │  7. Generate report from store                   │
    └──────────────┬──────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Worker A    Worker B    Worker C   ...
    (Agent)     (Agent)     (Agent)
        │          │          │
        └──────────┼──────────┘
                   ▼
           ConditionStore (DuckDB)
           Shared research database
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from swarm.agent_worker import create_worker_agent, run_worker_agent
from swarm.angles import (
    WorkerAssignment,
    assign_workers,
    detect_angles_via_llm,
    detect_sections,
    extract_required_angles,
    merge_angles,
)

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)


@dataclass
class MCPSwarmMetrics:
    """Telemetry from an MCP swarm run."""

    total_workers: int = 0
    total_waves: int = 0
    total_findings_stored: int = 0
    total_tool_calls: int = 0
    findings_per_wave: list[int] = field(default_factory=list)
    phase_times: dict[str, float] = field(default_factory=dict)
    total_elapsed_s: float = 0.0
    convergence_reason: str = ""
    worker_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MCPSwarmResult:
    """Result of an MCP swarm synthesis."""

    report: str
    metrics: MCPSwarmMetrics = field(default_factory=MCPSwarmMetrics)
    angles_detected: list[str] = field(default_factory=list)


@dataclass
class MCPSwarmConfig:
    """Configuration for the MCP swarm engine.

    Attributes:
        max_workers: Maximum number of parallel worker agents.
        max_waves: Maximum worker waves before stopping.
        convergence_threshold: Stop if new findings per wave drops below this.
        api_base: vLLM or OpenAI-compatible API endpoint.
        model: Model identifier for the endpoint.
        api_key: API key (usually not needed for local vLLM).
        max_tokens: Max tokens per worker LLM response.
        temperature: Sampling temperature for workers.
        required_angles: Angles that must be covered regardless of corpus.
        report_max_tokens: Max tokens for the report generation call.
        enable_serendipity_wave: Run a final cross-domain discovery wave.
        source_model: Model name for provenance tracking.
        source_run: Run identifier (e.g. "run_042") for provenance.
        max_return_chars: Hard ceiling on characters any tool call returns.
        compact_every_n_waves: Run store compaction after this many waves.
        enable_rolling_summaries: Generate knowledge briefings after waves.
        report_max_chars: Max prompt chars for report generation.
        worker_timeout_s: Per-worker timeout in seconds (default 600).
            Workers that exceed this are cancelled so the wave can proceed.
        enable_flock_evaluation: Run mass Flock evaluation rounds after
            worker waves.  Fires thousands of flag-driven queries against
            cached clone perspectives to simulate a large swarm.
        flock_max_rounds: Maximum Flock evaluation rounds.
        flock_max_queries_per_round: Maximum queries per clone per round.
        flock_batch_size: Parallel query batch size for Flock evaluation.
        enable_mcp_research: Run MCP-powered external data acquisition
            between waves.  Uses gradient flags to identify gaps and
            fans out across available search APIs.
    """

    max_workers: int = 7
    max_waves: int = 3
    convergence_threshold: int = 5
    api_base: str = "http://localhost:8000/v1"
    model: str = "default"
    api_key: str = "not-needed"
    max_tokens: int = 4096
    temperature: float = 0.3
    required_angles: list[str] = field(default_factory=list)
    report_max_tokens: int = 8192
    enable_serendipity_wave: bool = True
    source_model: str = ""
    source_run: str = ""
    max_return_chars: int = 6000
    compact_every_n_waves: int = 3
    enable_rolling_summaries: bool = True
    report_max_chars: int = 24000
    worker_timeout_s: float = 600.0
    enable_flock_evaluation: bool = True
    flock_max_rounds: int = 10
    flock_max_queries_per_round: int = 500
    flock_batch_size: int = 20
    enable_mcp_research: bool = True


class MCPSwarmEngine:
    """Swarm engine where workers are Strands Agents with ConditionStore tools.

    Usage:
        engine = MCPSwarmEngine(
            store=my_condition_store,
            complete=my_llm_fn,
            config=MCPSwarmConfig(max_workers=7),
        )
        result = await engine.synthesize(corpus="...", query="...")
        print(result.report)
    """

    def __init__(
        self,
        store: "ConditionStore",
        complete: Callable[[str], Awaitable[str]],
        config: MCPSwarmConfig | None = None,
    ) -> None:
        self.store = store
        self.complete = complete
        self.config = config or MCPSwarmConfig()

    async def synthesize(
        self,
        corpus: str,
        query: str,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ) -> MCPSwarmResult:
        """Run the MCP swarm pipeline.

        0. Check corpus fingerprint (skip re-ingestion if already seen)
        1. Ingest corpus into ConditionStore (only if new)
        2. Detect angles and assign sections
        3. Run agent-workers in parallel waves
        4. Check convergence between waves
        5. Compact store periodically
        6. Generate rolling summaries
        7. Generate report from store

        Args:
            corpus: Full text corpus to synthesize.
            query: The user's research query.
            on_event: Optional async callback for progress events.

        Returns:
            MCPSwarmResult with report and metrics.
        """
        t0 = time.monotonic()
        metrics = MCPSwarmMetrics()
        config = self.config
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        async def _emit(event: dict) -> None:
            if on_event is not None:
                try:
                    await on_event(event)
                except Exception:
                    pass

        # ── Phase 0: Corpus ingestion (with fingerprint check) ───────
        phase_start = time.monotonic()

        # Compute corpus fingerprint to prevent re-ingestion
        corpus_hash = hashlib.sha256(corpus.encode()).hexdigest()[:16]
        corpus_already_ingested = self.store.has_corpus_hash(corpus_hash)

        sections = detect_sections(corpus)
        logger.info(
            "sections=<%d>, corpus_chars=<%d>, already_ingested=<%s> | corpus analysis complete",
            len(sections), len(corpus), corpus_already_ingested,
        )

        # Detect angles (always needed for worker assignment)
        # Wrapped defensively — if LLM-based angle detection fails,
        # fall back to section titles so the run continues.
        required_angles = list(config.required_angles)
        try:
            if not required_angles:
                required_angles = await extract_required_angles(
                    query, self.complete,
                )

            detected_angles = await detect_angles_via_llm(
                corpus, query, self.complete,
                max_angles=config.max_workers,
            )
        except Exception as exc:
            logger.warning(
                "error=<%s> | angle detection failed, falling back to section titles",
                exc,
            )
            detected_angles = None

        angles = merge_angles(
            detected=detected_angles or [s.title for s in sections[:config.max_workers]],
            required=required_angles,
            max_angles=config.max_workers,
        )

        if not angles:
            angles = [s.title for s in sections[:config.max_workers]]

        # Assign sections to angles
        assignments = assign_workers(
            sections, angles,
            max_workers=config.max_workers,
        )

        # Only ingest if corpus hasn't been seen before
        if not corpus_already_ingested:
            for a in assignments:
                self.store.ingest_raw(
                    raw_text=a.raw_content,
                    source_type="corpus_section",
                    source_ref=f"section_{a.worker_id}",
                    angle=a.angle,
                    iteration=0,
                    user_query=query,
                )
            # Register fingerprint so future runs skip ingestion
            paragraph_count = sum(
                len(a.raw_content.split("\n\n")) for a in assignments
            )
            self.store.register_corpus_hash(
                fingerprint=corpus_hash,
                source=f"corpus_{len(corpus)}_chars",
                char_count=len(corpus),
                paragraph_count=paragraph_count,
            )
            logger.info(
                "corpus_hash=<%s>, paragraphs=<%d> | corpus ingested and fingerprinted",
                corpus_hash, paragraph_count,
            )
        else:
            logger.info(
                "corpus_hash=<%s> | corpus already in store, skipping re-ingestion",
                corpus_hash,
            )

        metrics.total_workers = len(assignments)
        metrics.phase_times["ingestion"] = time.monotonic() - phase_start

        await _emit({
            "type": "swarm_phase",
            "phase": "ingestion_complete",
            "workers": len(assignments),
            "angles": [a.angle for a in assignments],
            "corpus_skipped": corpus_already_ingested,
        })

        logger.info(
            "workers=<%d>, angles=<%s> | starting agent waves",
            len(assignments), [a.angle for a in assignments],
        )

        # ── Phase 1-N: Worker waves ──────────────────────────────────
        # Accumulate worker outputs per angle across waves.
        # Used by Flock evaluation phase to build clone contexts.
        prior_outputs: dict[str, str] = {}

        wave = 0
        for wave in range(1, config.max_waves + 1):
            phase_start = time.monotonic()
            phase = f"wave_{wave}"

            # Count worker-generated findings before this wave (#191)
            # Excludes raw ingestion rows — only counts findings/thoughts/insights
            # produced by worker agents, not corpus ingestion.
            with self.store._lock:
                findings_before = self.store.conn.execute(
                    "SELECT COUNT(*) FROM conditions "
                    "WHERE consider_for_use = TRUE "
                    "AND row_type IN ('finding', 'thought', 'insight', 'synthesis') "
                    "AND source_type != 'corpus_section'"
                ).fetchone()[0]

            await _emit({
                "type": "swarm_phase",
                "phase": f"wave_{wave}_start",
                "wave": wave,
            })

            # Create and run agent-workers in parallel
            agents = []
            for a in assignments:
                agent = create_worker_agent(
                    store=self.store,
                    angle=a.angle,
                    worker_id=f"worker_{a.worker_id}_wave_{wave}",
                    query=query,
                    api_base=config.api_base,
                    model=config.model,
                    api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    phase=phase,
                    max_return_chars=config.max_return_chars,
                    source_model=config.source_model or config.model,
                    source_run=config.source_run or run_id,
                )
                agents.append((agent, a))

            # Run all workers in parallel with timeout protection.
            # A hung worker (e.g. dead LLM connection) must not block the
            # entire wave.  asyncio.wait_for raises TimeoutError which
            # asyncio.gather captures via return_exceptions=True.
            timeout = config.worker_timeout_s
            tasks = [
                asyncio.wait_for(
                    run_worker_agent(
                        agent=agent,
                        angle=a.angle,
                        worker_id=f"worker_{a.worker_id}_wave_{wave}",
                        query=query,
                    ),
                    timeout=timeout,
                )
                for agent, a in agents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results — exceptions (including TimeoutError) are
            # logged and skipped so the wave continues with partial data.
            wave_tool_calls = 0
            for idx, r in enumerate(results):
                if isinstance(r, asyncio.TimeoutError):
                    worker_angle = assignments[idx].angle if idx < len(assignments) else "unknown"
                    logger.warning(
                        "wave=<%d>, angle=<%s>, timeout_s=<%.0f> | worker timed out",
                        wave, worker_angle, timeout,
                    )
                    continue
                if isinstance(r, Exception):
                    logger.warning("wave=<%d> | worker failed: %s", wave, r)
                    continue
                if isinstance(r, dict):
                    metrics.worker_results.append(r)
                    wave_tool_calls += r.get("tool_calls", 0)
                    # Accumulate worker output for Flock clone contexts.
                    # Each wave's response is appended so the clone holds
                    # the full reasoning chain across all waves.
                    angle_key = r.get("angle", "")
                    response_text = r.get("response", "")
                    if angle_key and response_text:
                        prior_outputs[angle_key] = (
                            prior_outputs.get(angle_key, "") + "\n" + response_text
                        ).strip()

            metrics.total_tool_calls += wave_tool_calls

            # Count worker-generated findings after this wave (#191)
            with self.store._lock:
                findings_after = self.store.conn.execute(
                    "SELECT COUNT(*) FROM conditions "
                    "WHERE consider_for_use = TRUE "
                    "AND row_type IN ('finding', 'thought', 'insight', 'synthesis') "
                    "AND source_type != 'corpus_section'"
                ).fetchone()[0]

            new_findings = findings_after - findings_before
            metrics.findings_per_wave.append(new_findings)

            # Track total active findings (all types)
            with self.store._lock:
                total_active = self.store.conn.execute(
                    "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE"
                ).fetchone()[0]
            metrics.total_findings_stored = total_active

            wave_time = time.monotonic() - phase_start
            metrics.phase_times[f"wave_{wave}"] = wave_time

            await _emit({
                "type": "swarm_phase",
                "phase": f"wave_{wave}_complete",
                "wave": wave,
                "new_findings": new_findings,
                "total_findings": total_active,
                "elapsed_s": round(wave_time, 1),
            })

            logger.info(
                "wave=<%d>, new_findings=<%d>, total=<%d>, elapsed_s=<%.1f> | wave complete",
                wave, new_findings, total_active, wave_time,
            )

            # Periodic compaction to prevent unbounded store growth
            if (config.compact_every_n_waves > 0
                    and wave % config.compact_every_n_waves == 0):
                compact_start = time.monotonic()
                try:
                    stats = self.store.compact(complete=self.complete)
                except Exception as exc:
                    logger.warning(
                        "wave=<%d>, error=<%s> | compaction failed, continuing",
                        wave, exc,
                    )
                    stats = {"exact_duplicates_removed": 0, "semantic_duplicates_removed": 0}
                compact_time = time.monotonic() - compact_start
                total_removed = (
                    stats.get("exact_duplicates_removed", 0)
                    + stats.get("semantic_duplicates_removed", 0)
                )
                logger.info(
                    "wave=<%d>, exact_dupes=<%d>, semantic_dupes=<%d>, compact_time=<%.1f>s | compaction complete",
                    wave,
                    stats.get("exact_duplicates_removed", 0),
                    stats.get("semantic_duplicates_removed", 0),
                    compact_time,
                )
                metrics.phase_times[f"compact_wave_{wave}"] = compact_time

            # ── Emit wave metric ─────────────────────────────────
            self.store.emit_metric(
                "wave_metric",
                {
                    "findings_new": new_findings,
                    "findings_total": total_active,
                    "tool_calls": wave_tool_calls,
                    "elapsed_s": round(wave_time, 1),
                    "workers": len(assignments),
                    "worker_results": [
                        r for r in results
                        if isinstance(r, dict)
                    ],
                },
                source_model=config.model,
                source_run=run_id,
                iteration=wave,
            )

            # ── Emit per-worker metrics ──────────────────────────
            for r in results:
                if not isinstance(r, dict):
                    continue
                self.store.emit_metric(
                    "worker_metric",
                    {
                        "worker_id": r.get("worker_id", ""),
                        "angle": r.get("angle", ""),
                        "tool_calls": r.get("tool_calls", 0),
                        "findings_stored": r.get("findings_stored", 0),
                        "output_chars": r.get("output_chars", 0),
                        "error": str(r.get("error", "")),
                    },
                    angle=r.get("angle", "unknown"),
                    source_model=config.model,
                    source_run=run_id,
                    iteration=wave,
                )

            # Convergence check: stop if too few new findings
            if new_findings < config.convergence_threshold:
                metrics.convergence_reason = (
                    f"converged at wave {wave}: {new_findings} new findings "
                    f"< threshold {config.convergence_threshold}"
                )
                logger.info(
                    "wave=<%d>, new_findings=<%d>, threshold=<%d> | convergence detected",
                    wave, new_findings, config.convergence_threshold,
                )
                break
        else:
            metrics.convergence_reason = f"max waves ({config.max_waves}) reached"

        metrics.total_waves = wave if config.max_waves > 0 else 0

        # ── Serendipity wave (optional) ──────────────────────────────
        # Wrapped defensively — serendipity is a bonus phase.  A crash
        # here must not abort the run or prevent report generation.
        if config.enable_serendipity_wave and len(assignments) >= 2:
            phase_start = time.monotonic()
            try:
                # Build angle list for targeted cross-domain search
                angle_list = ", ".join(a.angle for a in assignments)

                serendipity_agent = create_worker_agent(
                    store=self.store,
                    angle="cross-domain connections",
                    worker_id="serendipity",
                    query=query,
                    api_base=config.api_base,
                    model=config.model,
                    api_key=config.api_key,
                    max_tokens=config.max_tokens,
                    temperature=0.5,  # slightly higher for creativity
                    phase="serendipity",
                    max_return_chars=config.max_return_chars,
                    source_model=config.source_model or config.model,
                    source_run=config.source_run or run_id,
                )

                serendipity_result = await asyncio.wait_for(
                    run_worker_agent(
                        agent=serendipity_agent,
                        angle="cross-domain connections",
                        worker_id="serendipity",
                        query=(
                            f"Your job is to find CROSS-DOMAIN CONNECTIONS that no single "
                            f"specialist would see. The research angles are: {angle_list}. "
                            f"Use find_connections to discover interactions between angle pairs. "
                            f"Use get_peer_insights and search_corpus to find where different "
                            f"specialists' findings interact, compound, or contradict in "
                            f"unexpected ways. Look for: compound interactions (e.g. iron + "
                            f"trenbolone hematocrit), nutrient-drug interactions, timing "
                            f"dependencies, dose-response relationships that span multiple "
                            f"domains. Store every cross-domain connection as a finding with "
                            f"store_finding. Research query: {query}"
                        ),
                    ),
                    timeout=config.worker_timeout_s,
                )
                metrics.worker_results.append(serendipity_result)
            except Exception as exc:
                logger.warning(
                    "error=<%s> | serendipity wave failed, continuing to report",
                    exc,
                )
            metrics.phase_times["serendipity"] = time.monotonic() - phase_start

            await _emit({
                "type": "swarm_phase",
                "phase": "serendipity_complete",
                "elapsed_s": round(metrics.phase_times["serendipity"], 1),
            })

        # ── Flock evaluation phase ────────────────────────────────────
        # Mass flag-driven queries against cached clone perspectives.
        # Each worker's accumulated reasoning becomes a clone context.
        # Thousands of queries simulate an incredibly large swarm.
        if config.enable_flock_evaluation and prior_outputs:
            phase_start = time.monotonic()
            await _emit({
                "type": "swarm_phase",
                "phase": "flock_evaluation_start",
            })

            try:
                from swarm.flock_query_manager import (
                    CloneContext,
                    FlockQueryManager,
                    FlockQueryManagerConfig,
                )

                # Build clone contexts from worker transcripts
                clones = [
                    CloneContext(
                        angle=angle,
                        context_summary=output,
                        context_tokens=len(output) // 3,
                        wave=wave,
                        worker_id=f"clone_{angle}",
                    )
                    for angle, output in prior_outputs.items()
                    if output
                ]

                if clones:
                    flock_config = FlockQueryManagerConfig(
                        max_rounds=config.flock_max_rounds,
                        max_queries_per_round=config.flock_max_queries_per_round,
                        batch_size=config.flock_batch_size,
                    )

                    # Wire interleaved MCP research into the Flock loop
                    # so external data is acquired between evaluation rounds
                    mcp_research_fn = None
                    if config.enable_mcp_research:
                        from swarm.mcp_researcher import run_mcp_research_round

                        async def _interleaved_research(rid: str) -> int:
                            result = await run_mcp_research_round(
                                store=self.store,
                                run_id=rid,
                                complete=self.complete,
                            )
                            return result.findings_stored

                        mcp_research_fn = _interleaved_research

                    flock_manager = FlockQueryManager(
                        store=self.store,
                        complete=self.complete,
                        config=flock_config,
                        mcp_research_fn=mcp_research_fn,
                    )
                    flock_result = await flock_manager.run(
                        clones=clones,
                        run_id=config.source_run or run_id,
                        on_event=on_event,
                    )

                    metrics.phase_times["flock_evaluation"] = time.monotonic() - phase_start
                    logger.info(
                        "flock_queries=<%d>, flock_evals=<%d>, flock_new=<%d>, "
                        "elapsed_s=<%.1f>, reason=<%s> | flock evaluation complete",
                        flock_result.total_queries,
                        flock_result.total_evaluations,
                        flock_result.total_new_findings,
                        flock_result.elapsed_s,
                        flock_result.convergence_reason,
                    )

                    await _emit({
                        "type": "swarm_phase",
                        "phase": "flock_evaluation_complete",
                        "total_queries": flock_result.total_queries,
                        "total_evaluations": flock_result.total_evaluations,
                        "total_new_findings": flock_result.total_new_findings,
                        "convergence_reason": flock_result.convergence_reason,
                        "elapsed_s": round(flock_result.elapsed_s, 1),
                    })
                else:
                    logger.info("no clone contexts available for flock evaluation")
            except Exception as exc:
                logger.warning(
                    "error=<%s> | flock evaluation failed, continuing to report",
                    exc,
                )
                metrics.phase_times["flock_evaluation"] = time.monotonic() - phase_start

        # ── MCP research phase ────────────────────────────────────────
        # Flag-driven external data acquisition.  Reads ConditionStore
        # gradient flags to identify what's missing, fans out across
        # available search APIs, writes results back as mcp_finding rows.
        if config.enable_mcp_research:
            phase_start = time.monotonic()
            await _emit({
                "type": "swarm_phase",
                "phase": "mcp_research_start",
            })

            try:
                from swarm.mcp_researcher import run_mcp_research_round

                research_metrics = await run_mcp_research_round(
                    store=self.store,
                    run_id=config.source_run or run_id,
                    complete=self.complete,
                )

                metrics.phase_times["mcp_research"] = time.monotonic() - phase_start
                logger.info(
                    "research_targets=<%d>, api_calls=<%d>, findings_stored=<%d>, "
                    "elapsed_s=<%.1f> | MCP research complete",
                    research_metrics.targets_researched,
                    research_metrics.api_calls_made,
                    research_metrics.findings_stored,
                    research_metrics.elapsed_s,
                )

                await _emit({
                    "type": "swarm_phase",
                    "phase": "mcp_research_complete",
                    "targets_researched": research_metrics.targets_researched,
                    "findings_stored": research_metrics.findings_stored,
                    "apis_used": research_metrics.apis_used,
                    "elapsed_s": round(research_metrics.elapsed_s, 1),
                })
            except Exception as exc:
                logger.warning(
                    "error=<%s> | MCP research failed, continuing to report",
                    exc,
                )
                metrics.phase_times["mcp_research"] = time.monotonic() - phase_start

        # ── Rolling summaries (optional) ─────────────────────────────
        # Summaries are nice-to-have.  A failure must not block report.
        if config.enable_rolling_summaries:
            phase_start = time.monotonic()
            try:
                await self._generate_rolling_summaries(assignments)
            except Exception as exc:
                logger.warning(
                    "error=<%s> | rolling summaries failed, continuing to report",
                    exc,
                )
            metrics.phase_times["summaries"] = time.monotonic() - phase_start

        # ── Report generation ────────────────────────────────────────
        # Report generation is critical but should not crash the entire
        # run — return a partial report on failure so metrics and store
        # data are still available.
        phase_start = time.monotonic()
        try:
            report = await self._generate_report(query, assignments)
        except Exception as exc:
            logger.error(
                "error=<%s> | report generation failed, returning store summary",
                exc,
            )
            report = (
                f"(report generation failed: {exc})\n\n"
                f"Store contains {metrics.total_findings_stored} findings "
                f"across {len(assignments)} angles after {metrics.total_waves} waves."
            )
        metrics.phase_times["report"] = time.monotonic() - phase_start
        metrics.total_elapsed_s = time.monotonic() - t0

        await _emit({
            "type": "swarm_phase",
            "phase": "complete",
            "total_elapsed_s": round(metrics.total_elapsed_s, 1),
            "total_findings": metrics.total_findings_stored,
        })

        # ── Run-level observability ──────────────────────────────────
        # Observability is best-effort — never let it crash the return.
        try:
            self.store.store_health_snapshot(
                source_run=run_id,
                iteration=metrics.total_waves,
            )

            self.store.emit_metric(
                "run_metric",
                {
                    "total_workers": metrics.total_workers,
                    "total_waves": metrics.total_waves,
                    "total_findings_stored": metrics.total_findings_stored,
                    "total_tool_calls": metrics.total_tool_calls,
                    "findings_per_wave": metrics.findings_per_wave,
                    "phase_times": metrics.phase_times,
                    "total_elapsed_s": round(metrics.total_elapsed_s, 1),
                    "convergence_reason": metrics.convergence_reason,
                    "angles": [a.angle for a in assignments],
                    "corpus_chars": len(corpus),
                },
                source_model=config.model,
                source_run=run_id,
            )

            logger.info(
                "run_id=<%s>, total_elapsed_s=<%.1f> | observability metrics persisted",
                run_id, metrics.total_elapsed_s,
            )
        except Exception as exc:
            logger.warning(
                "run_id=<%s>, error=<%s> | failed to persist run metrics",
                run_id, exc,
            )

        return MCPSwarmResult(
            report=report,
            metrics=metrics,
            angles_detected=[a.angle for a in assignments],
        )

    async def _generate_rolling_summaries(
        self,
        assignments: list[WorkerAssignment],
    ) -> None:
        """Generate knowledge briefings per angle for future workers.

        Compresses the current angle-level findings into a short summary
        and stores it in the knowledge_summaries table.  Workers in
        subsequent runs can call get_knowledge_briefing() to see a
        condensed view of all accumulated knowledge without reading
        every individual finding.
        """
        # Get current run number from store stats
        stats = self.store.get_store_stats()
        run_number = len(stats.get("models_seen", [])) or 1

        for a in assignments:
            with self.store._lock:
                count = self.store.conn.execute(
                    """SELECT COUNT(*) FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = ?
                         AND row_type IN ('finding', 'thought', 'insight')""",
                    [a.angle],
                ).fetchone()[0]

            if count == 0:
                continue

            # Get top findings for this angle (highest confidence)
            with self.store._lock:
                rows = self.store.conn.execute(
                    """SELECT fact, confidence
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = ?
                         AND row_type IN ('finding', 'thought', 'insight')
                       ORDER BY confidence DESC
                       LIMIT 15""",
                    [a.angle],
                ).fetchall()

            if not rows:
                continue

            # Build a condensed summary from top findings
            summary_parts = [f"[{a.angle}] {count} total findings. Top claims:"]
            for fact, conf in rows:
                summary_parts.append(f"- (conf={conf:.2f}) {fact[:200]}")

            summary = "\n".join(summary_parts)[:2000]

            self.store.store_summary(
                angle=a.angle,
                summary=summary,
                finding_count=count,
                run_number=run_number,
            )

        logger.info(
            "angles=<%d>, run_number=<%d> | rolling summaries generated",
            len(assignments), run_number,
        )

    async def _generate_report(
        self,
        query: str,
        assignments: list[WorkerAssignment],
    ) -> str:
        """Generate a report by querying the store, not by queen merge.

        Retrieves top findings per angle from the store and asks the LLM
        to compose a narrative from structured data.  The prompt is capped
        at report_max_chars to prevent context overflow even with 600K+
        findings in the store.

        Prioritizes worker-generated insights over raw corpus paragraphs.
        """
        max_chars = self.config.report_max_chars
        # Reserve space for the prompt framing
        framing_budget = 800
        findings_budget = max_chars - framing_budget

        # Gather top findings per angle, preferring worker analysis
        sections = []
        chars_used = 0
        for a in assignments:
            if chars_used >= findings_budget:
                break

            with self.store._lock:
                rows = self.store.conn.execute(
                    """SELECT fact, confidence, source_url, source_type
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = ?
                         AND row_type IN ('finding', 'thought', 'insight',
                                          'evaluation', 'mcp_finding', 'synthesis')
                       ORDER BY
                         CASE WHEN source_type = 'worker_analysis' THEN 0
                              WHEN source_type = 'flock_evaluation' THEN 1
                              WHEN source_type = 'mcp_research' THEN 2
                              ELSE 3 END,
                         confidence DESC
                       LIMIT 30""",
                    [a.angle],
                ).fetchall()

            if rows:
                findings = []
                section_chars = 0
                for fact, conf, src_url, src_type in rows:
                    src = f" ({src_url})" if src_url else ""
                    tag = " [worker]" if src_type == "worker_analysis" else ""
                    line = f"  [conf={conf:.2f}]{tag} {fact}{src}"
                    if chars_used + section_chars + len(line) > findings_budget:
                        break
                    findings.append(line)
                    section_chars += len(line) + 1

                if findings:
                    header = f"=== {a.angle.upper()} ({len(findings)} findings) ===\n"
                    section = header + "\n".join(findings)
                    sections.append(section)
                    chars_used += len(section) + 2

        # Get cross-domain findings
        if chars_used < findings_budget:
            with self.store._lock:
                cross_rows = self.store.conn.execute(
                    """SELECT fact, confidence, angle
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = 'cross-domain connections'
                         AND row_type IN ('finding', 'thought', 'insight')
                       ORDER BY confidence DESC
                       LIMIT 20""",
                ).fetchall()

            if cross_rows:
                cross_findings = []
                for fact, conf, angle in cross_rows:
                    line = f"  [conf={conf:.2f}] {fact}"
                    if chars_used + len(line) > findings_budget:
                        break
                    cross_findings.append(line)
                    chars_used += len(line) + 1

                if cross_findings:
                    sections.append(
                        f"=== CROSS-DOMAIN CONNECTIONS ({len(cross_findings)} findings) ===\n"
                        + "\n".join(cross_findings)
                    )

        if not sections:
            return "(no findings to report — store is empty)"

        # Get store stats for context
        stats = self.store.get_store_stats()

        store_text = "\n\n".join(sections)

        prompt = (
            f"You are writing a comprehensive research report.\n\n"
            f"RESEARCH QUERY: {query}\n\n"
            f"Store contains {stats['total_rows']} total entries across "
            f"{len(stats.get('by_angle', {}))} angles. "
            f"Below are the highest-confidence findings from "
            f"{len(assignments)} specialist researchers.\n\n"
            f"{store_text}\n\n"
            f"Write a comprehensive, practitioner-grade report that:\n"
            f"1. Synthesizes these findings into a coherent narrative\n"
            f"2. Highlights cross-domain connections and compound interactions\n"
            f"3. Preserves exact numbers, dosages, and citations\n"
            f"4. Organizes by practical phases (conservative → radical)\n"
            f"5. Explains mechanisms, not just protocols\n"
            f"6. No disclaimers or safety warnings\n\n"
            f"Write the full report:"
        )

        return await self.complete(prompt)
