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

        # Count extractable queries for metrics (regex-only — fast, no LLM).
        # The full LLM dissolution happens later in SearchExecutorBlock;
        # we only need a rough count here for observability.
        try:
            from tools.search_executor import _regex_extract_queries
            queries = _regex_extract_queries(strategy)
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

    Primary path: LLM-powered semantic comparison that understands
    whether the thinker is making genuine intellectual progress or
    repeating the same ideas with different words.

    Falls back to keyword overlap if LLM is unavailable.
    """
    if not current or not previous:
        return False

    # Primary: LLM-powered convergence check
    try:
        from utils.flock_proxy import get_flock_proxy_url
        import json as _json
        import urllib.request

        proxy_url = get_flock_proxy_url()
        if proxy_url:
            prompt = (
                "Compare these two research strategies from consecutive "
                "iterations.  Is the second one making GENUINE NEW "
                "INTELLECTUAL PROGRESS, or is it repeating the same ideas "
                "(possibly with different vocabulary)?\n\n"
                "Signs of real progress: new research angles, deeper "
                "questions, new connections, refined hypotheses.\n"
                "Signs of repetition: same questions rephrased, same "
                "angles with minor rewording, no new intellectual content.\n\n"
                "Return ONLY one word: PROGRESSING or CONVERGED\n\n"
                f"PREVIOUS STRATEGY:\n{previous[:1500]}\n\n"
                f"CURRENT STRATEGY:\n{current[:1500]}"
            )
            body = _json.dumps({
                "model": "flock-model",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 16,
                "temperature": 0.1,
            }).encode()
            req = urllib.request.Request(
                f"{proxy_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            choices = data.get("choices", [])
            answer = (choices[0]["message"]["content"] if choices else "").strip().strip(".,!?\n").upper()
            if answer == "CONVERGED":
                logger.info("LLM convergence check: CONVERGED")
                return True
            if answer in ("PROGRESSING", "PROGRESS"):
                logger.info("LLM convergence check: PROGRESSING")
                return False
    except Exception:
        pass  # fall through to keyword heuristic

    # Fallback: keyword overlap
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
