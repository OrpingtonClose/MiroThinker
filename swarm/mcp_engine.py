# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MCP swarm engine — tool-free workers with structured data packages.

The orchestrator builds a 7-section data package per worker per wave,
dispatches tool-free workers (simple LLM calls), extracts findings
from their reasoning output, and stores everything in the ConditionStore.

Workers never touch the store directly.  The orchestrator is the sole
writer.  Workers are pure reasoning agents — they receive material and
produce text.

24-hour continuous operation:
    - Corpus fingerprinting: skip re-ingestion if corpus already in store
    - Source provenance: every finding tagged with model + run number
    - Store compaction: periodic deduplication to prevent unbounded growth
    - Rolling summarization: knowledge briefings compress prior runs
    - Worker transcripts: full audit trail for clone reconstruction
    - Per-worker model assignment: different models per angle

Architecture:

    ┌─────────────────────────────────────────────────┐
    │              MCPSwarmEngine                      │
    │  0. Check corpus fingerprint (skip if seen)     │
    │  1. Ingest corpus into ConditionStore           │
    │  2. Detect angles from query + corpus           │
    │  3. Build data packages (§1-§7)                 │
    │  4. Dispatch tool-free workers in parallel       │
    │  5. Extract findings from worker output          │
    │  6. Store transcripts + findings                 │
    │  7. Check convergence                           │
    │  8. Compact store (periodic dedup)              │
    │  9. Generate rolling summaries                  │
    │ 10. Generate report from store                  │
    └──────────────┬──────────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Worker A    Worker B    Worker C   ...
    (tool-free) (tool-free) (tool-free)
    No store    No store    No store
    access      access      access
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from swarm.agent_worker import run_tool_free_worker
from swarm.angles import (
    WorkerAssignment,
    assign_workers,
    detect_angles_via_llm,
    detect_sections,
    extract_required_angles,
    merge_angles,
)
from swarm.data_package import build_data_packages
from swarm.finding_extractor import (
    extract_findings_llm,
    store_extracted_findings,
    store_worker_transcript,
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
        model: Default model identifier for the endpoint.
        model_map: Per-angle model assignment. Maps angle name to model
            identifier.  Workers whose angle is not in the map use the
            default ``model``.
        api_key: API key (usually not needed for local vLLM).
        max_tokens: Max tokens per worker LLM response.
        temperature: Sampling temperature for workers.
        required_angles: Angles that must be covered regardless of corpus.
        report_max_tokens: Max tokens for the report generation call.
        enable_serendipity_wave: Run a final cross-domain discovery wave.
        source_model: Model name for provenance tracking.
        source_run: Run identifier (e.g. "run_042") for provenance.
        compact_every_n_waves: Run store compaction after this many waves.
        enable_rolling_summaries: Generate knowledge briefings after waves.
        worker_timeout_s: Per-worker timeout in seconds (default 600).
            Workers that exceed this are cancelled so the wave can proceed.
    """

    max_workers: int = 7
    max_waves: int = 3
    convergence_threshold: int = 5
    api_base: str = "http://localhost:8000/v1"
    model: str = "default"
    model_map: dict[str, str] = field(default_factory=dict)
    api_key: str = "not-needed"
    max_tokens: int = 8192
    temperature: float = 0.3
    required_angles: list[str] = field(default_factory=list)
    report_max_tokens: int = 8192
    enable_serendipity_wave: bool = True
    source_model: str = ""
    source_run: str = ""
    compact_every_n_waves: int = 3
    enable_rolling_summaries: bool = True
    worker_timeout_s: float = 600.0


class MCPSwarmEngine:
    """Swarm engine with tool-free workers and structured data packages.

    Workers receive curated data packages and produce reasoning text.
    The orchestrator extracts findings from the text and stores them.
    No tools, no store access for workers.

    Usage:
        engine = MCPSwarmEngine(
            store=my_condition_store,
            complete=my_llm_fn,
            config=MCPSwarmConfig(
                max_workers=7,
                model_map={
                    "insulin_timing": "kimi-linear-48b",
                    "hematology": "glm-5.1",
                },
            ),
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
        """Run the MCP swarm pipeline with tool-free workers.

        0. Check corpus fingerprint (skip re-ingestion if already seen)
        1. Ingest corpus into ConditionStore (only if new)
        2. Detect angles and assign sections
        3. For each wave:
           a. Build data packages (§1-§7) per worker
           b. Dispatch tool-free workers in parallel
           c. Extract findings from worker output
           d. Store transcripts + findings
           e. Check convergence
        4. Compact store periodically
        5. Generate rolling summaries
        6. Run serendipity wave (tool-free)
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

        corpus_hash = hashlib.sha256(corpus.encode()).hexdigest()[:16]
        corpus_already_ingested = self.store.has_corpus_hash(corpus_hash)

        sections = detect_sections(corpus)
        logger.info(
            "sections=<%d>, corpus_chars=<%d>, already_ingested=<%s> | corpus analysis complete",
            len(sections), len(corpus), corpus_already_ingested,
        )

        # Detect angles (always needed for worker assignment)
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
            "workers=<%d>, angles=<%s> | starting tool-free worker waves",
            len(assignments), [a.angle for a in assignments],
        )

        # ── Phase 1-N: Worker waves (tool-free) ─────────────────────
        wave = 0
        prior_outputs: dict[str, str] = {}

        for wave in range(1, config.max_waves + 1):
            phase_start = time.monotonic()

            await _emit({
                "type": "swarm_phase",
                "phase": f"wave_{wave}_start",
                "wave": wave,
            })

            # ── Phase A: Build data packages ─────────────────────
            pkg_start = time.monotonic()
            packages = build_data_packages(
                store=self.store,
                assignments=assignments,
                wave=wave,
                query=query,
                prior_outputs=prior_outputs,
                model_map=config.model_map,
            )
            metrics.phase_times[f"wave_{wave}_pkg"] = time.monotonic() - pkg_start

            # ── Phase B: Dispatch tool-free workers in parallel ──
            timeout = config.worker_timeout_s
            tasks = [
                asyncio.wait_for(
                    run_tool_free_worker(
                        package=pkg,
                        query=query,
                        api_base=config.api_base,
                        model=config.model_map.get(pkg.angle, config.model),
                        api_key=config.api_key,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    ),
                    timeout=timeout,
                )
                for pkg in packages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # ── Phase C: Process results ─────────────────────────
            wave_findings_count = 0
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
                if not isinstance(r, dict):
                    continue

                metrics.worker_results.append(r)
                angle = r.get("angle", "")
                worker_id = r.get("worker_id", "")
                response = r.get("response", "")
                model_used = r.get("model", config.model)

                if not response or r.get("status") != "success":
                    continue

                # Save worker output for next wave's §7
                prior_outputs[angle] = response

                # ── Phase D: Store transcript ────────────────────
                try:
                    store_worker_transcript(
                        store=self.store,
                        worker_id=worker_id,
                        angle=angle,
                        transcript=response,
                        source_model=model_used,
                        source_run=config.source_run or run_id,
                        iteration=wave,
                    )
                except Exception as exc:
                    logger.warning(
                        "worker_id=<%s>, error=<%s> | transcript storage failed",
                        worker_id, exc,
                    )

                # ── Phase E: Extract findings from reasoning ─────
                try:
                    findings = await extract_findings_llm(
                        worker_output=response,
                        angle=angle,
                        query=query,
                        complete=self.complete,
                    )
                    stored = store_extracted_findings(
                        store=self.store,
                        findings=findings,
                        source_model=model_used,
                        source_run=config.source_run or run_id,
                        iteration=wave,
                    )
                    wave_findings_count += stored
                except Exception as exc:
                    logger.warning(
                        "worker_id=<%s>, error=<%s> | finding extraction failed",
                        worker_id, exc,
                    )

            metrics.findings_per_wave.append(wave_findings_count)

            # Track total active findings
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
                "new_findings": wave_findings_count,
                "total_findings": total_active,
                "elapsed_s": round(wave_time, 1),
            })

            logger.info(
                "wave=<%d>, new_findings=<%d>, total=<%d>, elapsed_s=<%.1f> | wave complete",
                wave, wave_findings_count, total_active, wave_time,
            )

            # Periodic compaction
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
                logger.info(
                    "wave=<%d>, exact_dupes=<%d>, semantic_dupes=<%d>, compact_time=<%.1f>s | compaction complete",
                    wave,
                    stats.get("exact_duplicates_removed", 0),
                    stats.get("semantic_duplicates_removed", 0),
                    compact_time,
                )
                metrics.phase_times[f"compact_wave_{wave}"] = compact_time

            # ── Emit wave metric ─────────────────────────────────
            try:
                self.store.emit_metric(
                    "wave_metric",
                    {
                        "findings_new": wave_findings_count,
                        "findings_total": total_active,
                        "tool_calls": 0,
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
            except Exception as exc:
                logger.warning(
                    "wave=<%d>, error=<%s> | wave metric emission failed",
                    wave, exc,
                )

            # ── Emit per-worker metrics ──────────────────────────
            for r in results:
                if not isinstance(r, dict):
                    continue
                try:
                    self.store.emit_metric(
                        "worker_metric",
                        {
                            "worker_id": r.get("worker_id", ""),
                            "angle": r.get("angle", ""),
                            "tool_calls": 0,
                            "input_chars": r.get("input_chars", 0),
                            "output_chars": r.get("output_chars", 0),
                            "model": r.get("model", ""),
                            "elapsed_s": r.get("elapsed_s", 0),
                            "error": str(r.get("error", "")),
                        },
                        angle=r.get("angle", "unknown"),
                        source_model=r.get("model", config.model),
                        source_run=run_id,
                        iteration=wave,
                    )
                except Exception as exc:
                    logger.warning(
                        "wave=<%d>, error=<%s> | worker metric emission failed",
                        wave, exc,
                    )

            # Convergence check
            if wave_findings_count < config.convergence_threshold:
                metrics.convergence_reason = (
                    f"converged at wave {wave}: {wave_findings_count} new findings "
                    f"< threshold {config.convergence_threshold}"
                )
                logger.info(
                    "wave=<%d>, new_findings=<%d>, threshold=<%d> | convergence detected",
                    wave, wave_findings_count, config.convergence_threshold,
                )
                break
        else:
            metrics.convergence_reason = f"max waves ({config.max_waves}) reached"

        metrics.total_waves = wave if config.max_waves > 0 else 0

        # ── Serendipity wave (tool-free) ─────────────────────────────
        if config.enable_serendipity_wave and len(assignments) >= 2:
            phase_start = time.monotonic()
            try:
                await self._run_serendipity_wave(
                    assignments=assignments,
                    query=query,
                    run_id=run_id,
                    metrics=metrics,
                    prior_outputs=prior_outputs,
                )
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

        # ── Rolling summaries ────────────────────────────────────────
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
                    "total_tool_calls": 0,
                    "findings_per_wave": metrics.findings_per_wave,
                    "phase_times": metrics.phase_times,
                    "total_elapsed_s": round(metrics.total_elapsed_s, 1),
                    "convergence_reason": metrics.convergence_reason,
                    "angles": [a.angle for a in assignments],
                    "corpus_chars": len(corpus),
                    "model_map": config.model_map,
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

    async def _run_serendipity_wave(
        self,
        assignments: list[WorkerAssignment],
        query: str,
        run_id: str,
        metrics: MCPSwarmMetrics,
        prior_outputs: dict[str, str],
    ) -> None:
        """Run a tool-free serendipity wave for cross-domain discovery.

        Gathers top findings from all angles and asks a cross-domain
        specialist to find connections that no single specialist would see.

        Args:
            assignments: Worker assignments (for angle list).
            query: The user's research query.
            run_id: Current run identifier.
            metrics: Metrics object to update.
            prior_outputs: Map of angle → latest worker output.
        """
        config = self.config
        angle_list = ", ".join(a.angle for a in assignments)

        # Build a cross-domain data package from all angle summaries
        from swarm.data_package import DataPackage

        summaries: list[str] = []
        for a in assignments:
            output = prior_outputs.get(a.angle, "")
            if output:
                # Take first ~2000 chars of each angle's latest output
                summaries.append(
                    f"=== {a.angle.upper()} ===\n{output[:2000]}"
                )

        if not summaries:
            logger.info("no worker outputs available for serendipity wave")
            return

        serendipity_pkg = DataPackage(
            angle="cross-domain connections",
            wave=0,
            worker_id="serendipity",
            model=config.model_map.get("cross-domain connections", config.model),
            corpus_material="\n\n".join(summaries),
        )

        serendipity_result = await asyncio.wait_for(
            run_tool_free_worker(
                package=serendipity_pkg,
                query=(
                    f"Find CROSS-DOMAIN CONNECTIONS that no single specialist "
                    f"would see. The research angles are: {angle_list}. "
                    f"Look for: compound interactions, nutrient-drug interactions, "
                    f"timing dependencies, dose-response relationships that span "
                    f"multiple domains, mechanistic convergences, framework "
                    f"transfers. Research query: {query}"
                ),
                api_base=config.api_base,
                model=config.model_map.get("cross-domain connections", config.model),
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                temperature=0.5,
            ),
            timeout=config.worker_timeout_s,
        )

        metrics.worker_results.append(serendipity_result)

        if serendipity_result.get("status") == "success":
            response = serendipity_result.get("response", "")
            if response:
                # Store transcript — use wave after last regular wave
                serendipity_iteration = metrics.total_waves + 1
                store_worker_transcript(
                    store=self.store,
                    worker_id="serendipity",
                    angle="cross-domain connections",
                    transcript=response,
                    source_model=serendipity_result.get("model", config.model),
                    source_run=config.source_run or run_id,
                    iteration=serendipity_iteration,
                )

                # Extract findings
                findings = await extract_findings_llm(
                    worker_output=response,
                    angle="cross-domain connections",
                    query=query,
                    complete=self.complete,
                )
                store_extracted_findings(
                    store=self.store,
                    findings=findings,
                    source_model=serendipity_result.get("model", config.model),
                    source_run=config.source_run or run_id,
                    iteration=serendipity_iteration,
                )

    async def _generate_rolling_summaries(
        self,
        assignments: list[WorkerAssignment],
    ) -> None:
        """Generate knowledge briefings per angle for future workers.

        Compresses the current angle-level findings into a short summary
        and stores it in the knowledge_summaries table.  Workers in
        subsequent waves can see this via §1 (KNOWLEDGE STATE) in their
        data package.
        """
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
        """Generate a report from the store's accumulated findings.

        Retrieves top findings per angle and asks the LLM to compose a
        comprehensive practitioner-grade report.
        """
        # Gather top findings per angle, preferring worker analysis
        sections = []
        for a in assignments:
            with self.store._lock:
                rows = self.store.conn.execute(
                    """SELECT fact, confidence, source_url, source_type
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = ?
                         AND row_type IN ('finding', 'thought', 'insight')
                       ORDER BY
                         CASE WHEN source_type = 'worker_analysis' THEN 0 ELSE 1 END,
                         confidence DESC
                       LIMIT 30""",
                    [a.angle],
                ).fetchall()

            if rows:
                findings = []
                for fact, conf, src_url, src_type in rows:
                    src = f" ({src_url})" if src_url else ""
                    tag = " [worker]" if src_type == "worker_analysis" else ""
                    line = f"  [conf={conf:.2f}]{tag} {fact}{src}"
                    findings.append(line)

                if findings:
                    header = f"=== {a.angle.upper()} ({len(findings)} findings) ===\n"
                    section = header + "\n".join(findings)
                    sections.append(section)

        # Get cross-domain findings
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
                cross_findings.append(line)

            if cross_findings:
                sections.append(
                    f"=== CROSS-DOMAIN CONNECTIONS ({len(cross_findings)} findings) ===\n"
                    + "\n".join(cross_findings)
                )

        if not sections:
            return "(no findings to report — store is empty)"

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
