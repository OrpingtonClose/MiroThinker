# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MCP-driven swarm engine — workers are agents with ConditionStore tools.

Replaces the single-shot prompt-stuffing architecture with agent-workers
that explore the corpus via tool calls.  Each worker is a Strands Agent
with tools to search, discover peer findings, and store its analysis.

Key differences from the original GossipSwarm:
- **No context window constraint**: Workers pull data on demand via tools.
  A 32K context model can process a 10MB corpus — it just makes more
  tool calls.
- **No queen merge**: Workers write findings directly to the store.
  The final report is a separate task that queries the store.
- **No explicit gossip rounds**: Workers discover peer findings via
  get_peer_insights tool.  Cross-pollination happens organically as
  workers store findings that become visible to other workers' searches.
- **Convergence via store**: The engine runs workers in waves.  After
  each wave, it checks whether new findings are still being stored.
  When the store stops growing, the swarm has converged.

Architecture:

    ┌─────────────────────────────────────────────────┐
    │              MCPSwarmEngine                       │
    │  1. Ingest corpus into ConditionStore            │
    │  2. Create N agent-workers (one per angle)       │
    │  3. Run workers in parallel waves                │
    │  4. Check convergence (store growth rate)        │
    │  5. Generate report from store                   │
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
import logging
import time
from dataclasses import dataclass, field
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

        1. Ingest corpus into ConditionStore
        2. Detect angles and assign sections
        3. Run agent-workers in parallel waves
        4. Check convergence between waves
        5. Generate report from store

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

        async def _emit(event: dict) -> None:
            if on_event is not None:
                try:
                    await on_event(event)
                except Exception:
                    pass

        # ── Phase 0: Corpus ingestion ────────────────────────────────
        phase_start = time.monotonic()
        sections = detect_sections(corpus)
        logger.info(
            "sections=<%d>, corpus_chars=<%d> | corpus analysis complete",
            len(sections), len(corpus),
        )

        # Ingest each section as raw data with angle attribution
        required_angles = list(config.required_angles)
        if not required_angles:
            required_angles = await extract_required_angles(
                query, self.complete,
            )

        detected_angles = await detect_angles_via_llm(
            corpus, query, self.complete,
            max_angles=config.max_workers,
        )

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

        # Ingest assigned sections into the store
        for a in assignments:
            self.store.ingest_raw(
                raw_text=a.raw_content,
                source_type="corpus_section",
                source_ref=f"section_{a.worker_id}",
                angle=a.angle,
                iteration=0,
                user_query=query,
            )

        metrics.total_workers = len(assignments)
        metrics.phase_times["ingestion"] = time.monotonic() - phase_start

        await _emit({
            "type": "swarm_phase",
            "phase": "ingestion_complete",
            "workers": len(assignments),
            "angles": [a.angle for a in assignments],
        })

        logger.info(
            "workers=<%d>, angles=<%s> | corpus ingested, starting agent waves",
            len(assignments), [a.angle for a in assignments],
        )

        # ── Phase 1-N: Worker waves ──────────────────────────────────
        # Each wave: create agent-workers, run in parallel, count new findings.
        # Stop when findings per wave drops below convergence threshold.
        for wave in range(1, config.max_waves + 1):
            phase_start = time.monotonic()
            phase = f"wave_{wave}"

            # Count findings before this wave
            with self.store._lock:
                findings_before = self.store.conn.execute(
                    "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE"
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
                )
                agents.append((agent, a))

            # Run all workers in parallel
            tasks = [
                run_worker_agent(
                    agent=agent,
                    angle=a.angle,
                    worker_id=f"worker_{a.worker_id}_wave_{wave}",
                    query=query,
                )
                for agent, a in agents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            wave_tool_calls = 0
            for r in results:
                if isinstance(r, Exception):
                    logger.warning("wave=<%d> | worker failed: %s", wave, r)
                    continue
                if isinstance(r, dict):
                    metrics.worker_results.append(r)
                    wave_tool_calls += r.get("tool_calls", 0)

            metrics.total_tool_calls += wave_tool_calls

            # Count findings after this wave
            with self.store._lock:
                findings_after = self.store.conn.execute(
                    "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE"
                ).fetchone()[0]

            new_findings = findings_after - findings_before
            metrics.findings_per_wave.append(new_findings)
            metrics.total_findings_stored = findings_after

            wave_time = time.monotonic() - phase_start
            metrics.phase_times[f"wave_{wave}"] = wave_time

            await _emit({
                "type": "swarm_phase",
                "phase": f"wave_{wave}_complete",
                "wave": wave,
                "new_findings": new_findings,
                "total_findings": findings_after,
                "elapsed_s": round(wave_time, 1),
            })

            logger.info(
                "wave=<%d>, new_findings=<%d>, total=<%d>, elapsed_s=<%.1f> | wave complete",
                wave, new_findings, findings_after, wave_time,
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

        metrics.total_waves = wave

        # ── Serendipity wave (optional) ──────────────────────────────
        if config.enable_serendipity_wave and len(assignments) >= 2:
            phase_start = time.monotonic()

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
            )

            serendipity_result = await run_worker_agent(
                agent=serendipity_agent,
                angle="cross-domain connections",
                worker_id="serendipity",
                query=(
                    f"Your job is to find CROSS-DOMAIN CONNECTIONS that no single "
                    f"specialist would see. Use get_peer_insights and search_corpus "
                    f"to find where different specialists' findings interact, compound, "
                    f"or contradict in unexpected ways. Look for: compound interactions, "
                    f"nutrient-drug interactions, timing dependencies, dose-response "
                    f"relationships that span multiple domains. Store every cross-domain "
                    f"connection as a finding. Research query: {query}"
                ),
            )
            metrics.worker_results.append(serendipity_result)
            metrics.phase_times["serendipity"] = time.monotonic() - phase_start

            await _emit({
                "type": "swarm_phase",
                "phase": "serendipity_complete",
                "elapsed_s": round(metrics.phase_times["serendipity"], 1),
            })

        # ── Report generation ────────────────────────────────────────
        phase_start = time.monotonic()
        report = await self._generate_report(query, assignments)
        metrics.phase_times["report"] = time.monotonic() - phase_start
        metrics.total_elapsed_s = time.monotonic() - t0

        await _emit({
            "type": "swarm_phase",
            "phase": "complete",
            "total_elapsed_s": round(metrics.total_elapsed_s, 1),
            "total_findings": metrics.total_findings_stored,
        })

        return MCPSwarmResult(
            report=report,
            metrics=metrics,
            angles_detected=[a.angle for a in assignments],
        )

    async def _generate_report(
        self,
        query: str,
        assignments: list[WorkerAssignment],
    ) -> str:
        """Generate a report by querying the store, not by queen merge.

        Retrieves top findings per angle from the store and asks the LLM
        to compose a narrative from structured data — much smaller prompt
        than the old queen merge which received all worker outputs.
        """
        # Gather top findings per angle
        sections = []
        for a in assignments:
            with self.store._lock:
                rows = self.store.conn.execute(
                    """SELECT fact, confidence, source_url
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND angle = ?
                         AND row_type IN ('finding', 'thought', 'insight')
                       ORDER BY confidence DESC
                       LIMIT 30""",
                    [a.angle],
                ).fetchall()

            if rows:
                findings = []
                for fact, conf, src_url in rows:
                    src = f" ({src_url})" if src_url else ""
                    findings.append(f"  [conf={conf:.2f}] {fact}{src}")
                sections.append(
                    f"=== {a.angle.upper()} ({len(rows)} findings) ===\n"
                    + "\n".join(findings)
                )

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
                cross_findings.append(f"  [conf={conf:.2f}] {fact}")
            sections.append(
                f"=== CROSS-DOMAIN CONNECTIONS ({len(cross_rows)} findings) ===\n"
                + "\n".join(cross_findings)
            )

        if not sections:
            return "(no findings to report — store is empty)"

        store_text = "\n\n".join(sections)

        prompt = (
            f"You are writing a comprehensive research report.\n\n"
            f"RESEARCH QUERY: {query}\n\n"
            f"Below are the key findings from {len(assignments)} specialist "
            f"researchers, organized by domain. Each finding has a confidence "
            f"score and source attribution.\n\n"
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
