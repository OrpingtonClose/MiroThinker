# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MaestroBlock -- free-form Flock conductor phase.

Wraps the maestro's after_agent_callback logic as a fenced block:
- Preserves maestro reasoning as thought lineage
- Drains pending search results
- Kicks off safety-net scoring
- Refreshes state with current corpus for the thinker
- Runs periodic synthesis and thought swarm cycle
- Advances the iteration counter
"""

from __future__ import annotations

import asyncio
import logging
import os

from models.pipeline_block import (
    BlockContext, BlockCriticality, BlockResult, ParamSpec,
    PipelineBlock,
)

logger = logging.getLogger(__name__)


class MaestroBlock(PipelineBlock):
    """Free-form Flock conductor: organises corpus via SQL."""

    name = "maestro"
    needs_corpus = True
    criticality = BlockCriticality.BEST_EFFORT
    is_looped = True

    input_specs = [
        ParamSpec(
            key="corpus_summary_for_maestro",
            expected_type=str,
            required=False,
            description="Compact structural summary for maestro orientation",
        ),
        ParamSpec(
            key="research_strategy",
            expected_type=str,
            required=False,
            description="Thinker's strategy for this iteration",
        ),
    ]
    output_specs = [
        ParamSpec(
            key="research_findings",
            expected_type=str,
            required=False,
            description="Updated corpus briefing after maestro",
        ),
    ]

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Post-maestro processing: thought preservation, scoring, swarm, state refresh.

        NOTE: The actual maestro LLM call is handled by ADK's Agent.
        This block runs as the after_agent_callback wrapper.
        """
        from callbacks.condition_manager import (
            _drain_search_queue,
            _safe_corpus_write,
            _swarm_tasks,
            run_swarm_synthesis,
        )

        state = ctx.state
        corpus = ctx.corpus
        if corpus is None:
            return BlockResult(
                metrics={"skipped": True, "reason": "no_corpus"},
                diagnosis="Maestro block skipped: no corpus available",
            )

        corpus_key = state.get("_corpus_key", "default")
        iteration = state.get("_corpus_iteration", 0)
        user_query = state.get("user_query", "")

        # ── THOUGHT LINEAGE: Preserve maestro's reasoning ─────────────
        # The maestro's text output would otherwise be overwritten when
        # we refresh state["research_findings"].  Store it as an immutable
        # thought row so the full reasoning chain is preserved.
        # Uses _safe_corpus_write to acquire the per-corpus async lock,
        # preventing races with background swarm tasks.
        maestro_output = state.get("research_findings", "")
        if maestro_output and maestro_output.strip():
            try:
                from callbacks.condition_manager import _safe_corpus_write
                _corpus = corpus
                _output = maestro_output
                _iter = iteration
                await _safe_corpus_write(
                    corpus_key,
                    lambda: _corpus.admit_thought(
                        reasoning=_output,
                        angle="maestro_reasoning",
                        strategy=f"maestro_iteration_{_iter}",
                        iteration=_iter,
                    ),
                )
                logger.info(
                    "Maestro reasoning preserved: %d chars at iteration %d",
                    len(maestro_output), iteration,
                )
            except Exception:
                logger.warning(
                    "Failed to preserve maestro reasoning (non-fatal)",
                    exc_info=True,
                )

        # Drain remaining queued search results
        _drain_search_queue(state)

        # Snapshot corpus state BEFORE background thread starts
        # (DuckDB connections are not thread-safe)
        thinker_briefing = corpus.format_for_thinker(current_iteration=iteration)

        # Devil's advocate injection
        try:
            from callbacks.condition_manager import _maybe_inject_devils_advocate
            thinker_briefing = _maybe_inject_devils_advocate(
                thinker_briefing, corpus, iteration,
            )
        except Exception:
            pass

        # Pre-compute ALL corpus-derived state BEFORE the background thread
        # starts.  DuckDB connections are not thread-safe, so every read must
        # happen here on the main thread.  CorpusRefreshAspect skips keys
        # that are already in state_updates, so setting them here prevents
        # any post-block DuckDB access.
        expansion_text = ""
        try:
            expansion_targets = corpus.get_expansion_targets()
            if expansion_targets:
                lines = ["=== ENRICHMENT TASKS (from corpus analysis) ==="]
                for t in expansion_targets:
                    lines.append(
                        f"- Finding [{t['id']}] needs enrichment via "
                        f"{t['strategy']}: {t['hint']}"
                    )
                lines.append("=== END ENRICHMENT TASKS ===")
                expansion_text = "\n".join(lines)
        except Exception:
            logger.debug("Expansion targets computation failed (non-fatal)")

        state_updates = {
            "research_findings": thinker_briefing,
            "corpus_for_synthesis": corpus.format_for_synthesiser(),
            "corpus_summary_for_maestro": corpus.format_summary_for_maestro(),
            "_expansion_targets": expansion_text,
        }

        # Emit corpus stats to dashboard
        if ctx.collector is not None:
            try:
                total = corpus.count()
                ctx.collector.corpus_update(0, total, iteration)
                ctx.collector.emit_event("maestro_complete", data={
                    "total_conditions": total,
                    "iteration": iteration,
                })
            except Exception:
                pass

        # Safety-net scoring under async lock
        def _background_scoring() -> None:
            try:
                scored = corpus.score_new_conditions(user_query)
                if scored:
                    logger.info("Maestro safety-net: scored %d conditions", scored)
                deduped = corpus.compute_duplications()
                if deduped:
                    logger.info("Maestro safety-net: deduped %d pairs", deduped)
                if scored:
                    corpus.compute_composite_quality()
            except Exception:
                logger.warning("Maestro safety-net scoring failed", exc_info=True)

        try:
            await _safe_corpus_write(corpus_key, _background_scoring)
        except Exception:
            logger.warning("Async safety-net scoring failed", exc_info=True)

        # Periodic synthesis under async lock
        synthesis_interval = int(os.environ.get("SYNTHESIS_INTERVAL", "1"))
        if iteration > 0 and iteration % synthesis_interval == 0:
            try:
                swarm_report = run_swarm_synthesis(state)
                if swarm_report and swarm_report.strip():
                    await _safe_corpus_write(
                        corpus_key,
                        lambda: corpus.admit_thought(
                            reasoning=swarm_report,
                            angle="periodic_synthesis",
                            strategy="periodic_synthesis_report",
                            iteration=iteration,
                        ),
                    )
            except Exception:
                logger.warning("Periodic synthesis failed (non-fatal)", exc_info=True)

        # Thought swarm cycle (async task under lock)
        try:
            from tools.swarm_thinkers import run_swarm_cycle

            swarm_state_snapshot = {
                "_corpus_iteration": iteration,
                "user_query": state.get("user_query", ""),
                "research_strategy": state.get("research_strategy", ""),
            }

            async def _async_bg_swarm() -> None:
                try:
                    swarm_ids = await _safe_corpus_write(
                        corpus_key,
                        lambda: run_swarm_cycle(swarm_state_snapshot, corpus),
                    )
                    if swarm_ids:
                        logger.info(
                            "Background swarm produced %d new thoughts at iter %d",
                            len(swarm_ids), iteration,
                        )
                except Exception:
                    logger.warning("Background swarm cycle failed (non-fatal)", exc_info=True)

            task = asyncio.create_task(
                _async_bg_swarm(),
                name=f"swarm-iter-{iteration}",
            )
            _swarm_tasks[corpus_key] = task
        except Exception:
            logger.warning("Swarm dispatch failed (non-fatal)", exc_info=True)

        # Advance iteration
        state_updates["_corpus_iteration"] = iteration + 1

        return BlockResult(
            metrics={
                "iteration": iteration,
                "corpus_key": corpus_key,
            },
            state_updates=state_updates,
        )
