# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""MaestroBlock -- free-form Flock conductor phase.

Wraps the maestro's after_agent_callback logic as a fenced block:
- Drains pending search results
- Kicks off safety-net scoring in a background thread
- Refreshes state with current corpus for the thinker
- Runs periodic synthesis and thought swarm cycle
- Advances the iteration counter
"""

from __future__ import annotations

import logging
import os
import threading

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
            key="research_findings",
            expected_type=str,
            required=False,
            description="Current corpus briefing",
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
        """Post-maestro processing: scoring, swarm, state refresh.

        NOTE: The actual maestro LLM call is handled by ADK's Agent.
        This block runs as the after_agent_callback wrapper.
        """
        from callbacks.condition_manager import (
            _drain_search_queue,
            _scoring_lock,
            _scoring_threads,
            run_swarm_synthesis,
        )

        state = ctx.state
        corpus = ctx.corpus
        if corpus is None:
            return BlockResult(
                metrics={"skipped": True, "reason": "no_corpus"},
                diagnosis="Maestro block skipped: no corpus available",
            )

        # Drain remaining queued search results
        _drain_search_queue(state)

        # Safety-net scoring in background thread
        user_query = state.get("user_query", "")
        iteration = state.get("_corpus_iteration", 0)

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

        state_updates = {
            "research_findings": thinker_briefing,
            "corpus_for_synthesis": corpus.format_for_synthesiser(),
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

        # Background scoring thread
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

        corpus_key = state.get("_corpus_key", "default")
        with _scoring_lock:
            old_t = _scoring_threads.get(corpus_key)
            if old_t is not None and old_t.is_alive():
                logger.warning(
                    "Previous scoring thread still alive (key=%s) — "
                    "skipping safety-net scoring",
                    corpus_key,
                )
            else:
                t = threading.Thread(
                    target=_background_scoring,
                    daemon=True,
                    name=f"maestro-safety-scoring-{corpus_key}",
                )
                _scoring_threads[corpus_key] = t
                t.start()

        # Periodic synthesis
        synthesis_interval = int(os.environ.get("SYNTHESIS_INTERVAL", "1"))
        if iteration > 0 and iteration % synthesis_interval == 0:
            try:
                swarm_report = run_swarm_synthesis(state)
                if swarm_report and swarm_report.strip():
                    corpus.admit_thought(
                        reasoning=swarm_report,
                        angle="periodic_synthesis",
                        strategy="periodic_synthesis_report",
                        iteration=iteration,
                    )
            except Exception:
                logger.warning("Periodic synthesis failed (non-fatal)", exc_info=True)

        # Thought swarm cycle (background)
        try:
            from tools.swarm_thinkers import run_swarm_cycle

            swarm_state_snapshot = {
                "_corpus_iteration": iteration,
                "user_query": state.get("user_query", ""),
                "research_strategy": state.get("research_strategy", ""),
            }

            def _bg_swarm() -> None:
                try:
                    swarm_ids = run_swarm_cycle(swarm_state_snapshot, corpus)
                    if swarm_ids:
                        logger.info(
                            "Background swarm produced %d new thoughts at iter %d",
                            len(swarm_ids), iteration,
                        )
                except Exception:
                    logger.warning("Background swarm cycle failed (non-fatal)", exc_info=True)

            swarm_thread = threading.Thread(
                target=_bg_swarm, daemon=True, name=f"swarm-iter-{iteration}",
            )
            swarm_thread.start()
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
