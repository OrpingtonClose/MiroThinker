# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ThinkerBlock -- pure reasoning phase with thought lineage.

The thinker reads the enriched corpus briefing, reasons about gaps
and emerging narratives, and outputs a research strategy for the
next iteration.  Signals EVIDENCE_SUFFICIENT to exit the loop.

This block handles:
- Thought admission to DuckDB (preserving thinker reasoning lineage)
- Strategy tracking across iterations
- Convergence detection (delegates to convergence module when available)
- EVIDENCE_SUFFICIENT escalation
"""

from __future__ import annotations

import asyncio
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
    needs_corpus = True
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
        """Process thinker output: admit thought, track strategy, detect convergence.

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

        # ── THOUGHT LINEAGE: Write thinker thought to DuckDB ──
        # Preserves the full reasoning chain as immutable thought rows.
        # Uses _safe_corpus_write to acquire the per-corpus async lock,
        # preventing races with background swarm tasks.
        corpus_key = state.get("_corpus_key", "default")
        if strategy and strategy.strip() and ctx.corpus is not None:
            try:
                from callbacks.condition_manager import _safe_corpus_write
                _corpus = ctx.corpus
                _strategy = strategy
                _iter = iteration
                await _safe_corpus_write(
                    corpus_key,
                    lambda: _corpus.admit_thought(
                        reasoning=_strategy,
                        angle="thinker_reasoning",
                        strategy=f"thinker_iteration_{_iter}",
                        iteration=_iter,
                    ),
                )
                logger.info(
                    "Thinker thought admitted: %d chars at iteration %d",
                    len(strategy), iteration,
                )
                metrics["thought_admitted"] = True
            except Exception:
                logger.warning(
                    "Failed to admit thinker thought — will retry in search_executor",
                    exc_info=True,
                )
                # Deferred admission fallback
                state["_pending_thinker_thought"] = {
                    "reasoning": strategy,
                    "angle": "thinker_reasoning",
                    "strategy": f"thinker_iteration_{iteration}",
                    "iteration": iteration,
                }
                metrics["thought_admitted"] = False

        # Track strategies for iteration context injection
        if strategy and strategy.strip():
            prev = state.get("_prev_thinker_strategies", "")
            separator = f"\n--- Iteration {iteration} strategy ---\n"
            state["_prev_thinker_strategies"] = prev + separator + strategy

        # Count extractable queries for metrics (regex-only — fast, no LLM).
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
            if prev_strategy:
                converged = False
                # Try corpus-delta convergence (Phase 6) first
                try:
                    from models.convergence import check_convergence
                    converged = check_convergence(ctx.corpus, iteration, strategy, prev_strategy)
                except ImportError:
                    converged = _strategies_converged(strategy, prev_strategy)
                except Exception:
                    converged = _strategies_converged(strategy, prev_strategy)

                if converged:
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
                f"PREVIOUS STRATEGY:\n{previous}\n\n"
                f"CURRENT STRATEGY:\n{current}"
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
