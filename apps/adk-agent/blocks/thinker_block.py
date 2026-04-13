# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ThinkerBlock -- pure reasoning phase (no tools, no corpus mutation).

The thinker reads the enriched corpus briefing, reasons about gaps
and emerging narratives, and outputs a research strategy for the
next iteration.  Signals EVIDENCE_SUFFICIENT to exit the loop.

This block wraps the thinker's after_agent_callback logic:
- Strategy tracking across iterations
- Convergence detection
- EVIDENCE_SUFFICIENT escalation
"""

from __future__ import annotations

import logging

from models.pipeline_block import (
    BlockContext, BlockCriticality, BlockResult, ParamSpec,
    PipelineBlock, RoutingHint,
)

logger = logging.getLogger(__name__)

_SENTINEL = "EVIDENCE_SUFFICIENT"


class ThinkerBlock(PipelineBlock):
    """Strategy phase: pure reasoning over the corpus briefing."""

    name = "thinker"
    needs_corpus = False
    criticality = BlockCriticality.CRITICAL
    is_looped = True

    input_specs = [
        ParamSpec(
            key="research_findings",
            expected_type=str,
            validator=lambda v: bool(v and v.strip()),
            description="Corpus briefing for the thinker",
        ),
    ]
    output_specs = [
        ParamSpec(
            key="research_strategy",
            expected_type=str,
            required=False,
            description="Thinker's research strategy output",
        ),
    ]

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Process thinker output: track strategy, detect convergence.

        NOTE: The actual LLM call is handled by ADK's Agent.  This block
        runs as the after_agent_callback wrapper — it processes the
        thinker's output (already in state) and decides routing.
        """
        state = ctx.state
        strategy = state.get("research_strategy", "")
        iteration = state.get("_corpus_iteration", 0)
        metrics: dict = {
            "strategy_length": len(strategy),
            "iteration": iteration,
        }

        # Track strategies for iteration context injection
        if strategy and strategy.strip():
            summary = strategy[:500]
            prev = state.get("_prev_thinker_strategies", "")
            separator = f"\n--- Iteration {iteration} strategy ---\n"
            new_history = prev + separator + summary
            if len(new_history) > 2000:
                new_history = new_history[-2000:]
            state["_prev_thinker_strategies"] = new_history

        # Count extractable queries for metrics
        try:
            from tools.search_executor import extract_search_queries
            user_query = state.get("user_query", "")
            queries = extract_search_queries(strategy, user_query=user_query)
            metrics["extractable_queries"] = len(queries)
        except Exception:
            metrics["extractable_queries"] = 0

        # Convergence detection
        routing = RoutingHint.CONTINUE
        if strategy and _SENTINEL not in strategy:
            prev_strategy = state.get("_last_thinker_strategy", "")
            if prev_strategy and _strategies_converged(strategy, prev_strategy):
                logger.info(
                    "Convergence detected — thinker strategy is repeating. Escalating."
                )
                routing = RoutingHint.ESCALATE
                metrics["convergence_detected"] = True

                if ctx.collector is not None:
                    try:
                        ctx.collector.emit_event("convergence_detected", data={
                            "iteration": iteration,
                        })
                    except Exception:
                        pass

            state["_last_thinker_strategy"] = strategy

        # EVIDENCE_SUFFICIENT sentinel
        if _SENTINEL in strategy:
            logger.info("Thinker signalled EVIDENCE_SUFFICIENT — escalating")
            routing = RoutingHint.ESCALATE
            metrics["evidence_sufficient"] = True

            if ctx.collector is not None:
                try:
                    ctx.collector.thinker_escalate()
                except Exception:
                    pass

        return BlockResult(
            metrics=metrics,
            routing=routing,
        )


def _strategies_converged(current: str, previous: str) -> bool:
    """Detect if two strategies are substantively the same.

    Uses keyword overlap as a lightweight convergence signal.
    """
    import re as _re

    def _keywords(text: str) -> set[str]:
        words = set(_re.findall(r'\b[a-z]{4,}\b', text.lower()))
        stops = {
            "this", "that", "with", "from", "have", "been", "will",
            "would", "could", "should", "about", "which", "their",
            "there", "these", "those", "then", "than", "when", "what",
            "search", "find", "look", "also", "more", "most", "some",
            "into", "each", "such", "much", "very", "just", "only",
        }
        return words - stops

    curr_kw = _keywords(current)
    prev_kw = _keywords(previous)
    if not curr_kw or len(curr_kw) < 5:
        return False
    overlap = len(curr_kw & prev_kw) / len(curr_kw)
    return overlap > 0.80
