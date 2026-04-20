"""Corpus inspection tools for the deepagents orchestrator.

These tools give the orchestrator intelligence about corpus state,
replacing truncated string passing between research and gossip phases.

The orchestrator uses these to decide:
- What to research next (gap analysis)
- When to trigger gossip synthesis (coverage assessment)
- When to stop (sufficient coverage)
- What contradictions need resolution

All tools read from the active ConditionStore via contextvars
(per-asyncio-task isolation for concurrent jobs).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Per-task ConditionStore — set by _run_job before invoking orchestrator.
# Each asyncio task (job) gets its own store via contextvars.
_current_store: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_current_store", default=None,
)


def set_current_store(store: Any) -> None:
    """Set the ConditionStore for the current asyncio task context."""
    _current_store.set(store)


def _get_store() -> Any:
    """Get the active ConditionStore for the current context."""
    store = _current_store.get()
    if store is None:
        raise RuntimeError("corpus_tools: no active ConditionStore for this context")
    return store


# ------------------------------------------------------------------
# Tools exposed to the orchestrator via create_deep_agent(tools=[...])
# ------------------------------------------------------------------


def query_corpus(
    min_confidence: float = 0.0,
    limit: int = 50,
    angle: str = "",
) -> str:
    """Query the current research corpus from ConditionStore.

    Returns conditions matching the filters, formatted as structured text
    with fact, source, confidence, and verification status per row.

    Use this to understand what's been gathered so far before deciding
    what to research next or whether to trigger gossip synthesis.

    Args:
        min_confidence: Minimum confidence threshold (0.0-1.0).
        limit: Maximum number of results to return.
        angle: Filter by research angle (empty = all angles).

    Returns:
        Structured text listing matching conditions with metadata.
    """
    store = _get_store()
    findings = store.get_findings(
        min_confidence=min_confidence,
        angle=angle,
        limit=limit,
    )
    if not findings:
        return "(no findings in corpus yet)"

    lines = [f"=== CORPUS: {len(findings)} findings ==="]
    for f in findings:
        src = f" ({f['source_url']})" if f.get("source_url") else ""
        conf = f" [conf={f['confidence']:.2f}]"
        vstatus = f" [{f['verification_status']}]" if f.get("verification_status") else ""
        lines.append(f"[#{f['id']}]{conf}{vstatus} {f['fact']}{src}")

    return "\n".join(lines)


def assess_coverage(topic: str) -> str:
    """Assess how well a specific topic is covered in the corpus.

    Returns coverage metrics: total conditions, confidence distribution,
    identified gaps, and suggested research directions.

    Use this to decide whether more research is needed on a topic
    or whether the corpus has sufficient depth for synthesis.

    Args:
        topic: The topic to assess coverage for.

    Returns:
        Structured coverage assessment with metrics and gaps.
    """
    store = _get_store()

    total = store.count()
    by_type = store.count_by_type()
    findings = store.get_findings(limit=500)

    topic_lower = topic.lower()
    relevant = [
        f for f in findings
        if topic_lower in f.get("fact", "").lower()
    ]

    if relevant:
        confs = [f["confidence"] for f in relevant]
        avg_conf = sum(confs) / len(confs)
        low_conf_count = sum(1 for c in confs if c < 0.4)
        high_conf_count = sum(1 for c in confs if c >= 0.7)
    else:
        avg_conf = 0.0
        low_conf_count = 0
        high_conf_count = 0

    hints = store.get_expansion_hints()
    topic_hints = [
        h for h in hints
        if topic_lower in h.get("expansion_gap", "").lower()
        or topic_lower in h.get("fact", "").lower()
    ]

    contradictions = store.get_contradictions()
    topic_contradictions = [
        c for c in contradictions
        if topic_lower in c["claim_a"]["fact"].lower()
        or topic_lower in c["claim_b"]["fact"].lower()
    ]

    lines = [
        f"=== COVERAGE ASSESSMENT: {topic} ===",
        f"Total corpus: {total} conditions ({by_type})",
        f"Topic-relevant findings: {len(relevant)}",
        f"Avg confidence: {avg_conf:.2f}",
        f"High confidence (>=0.7): {high_conf_count}",
        f"Low confidence (<0.4): {low_conf_count}",
        f"Expansion hints: {len(topic_hints)}",
        f"Contradictions: {len(topic_contradictions)}",
    ]

    if len(relevant) < 5:
        lines.append("\n** SPARSE COVERAGE — more research needed **")
    elif avg_conf < 0.5:
        lines.append("\n** LOW CONFIDENCE — verification research needed **")
    elif low_conf_count > len(relevant) * 0.3:
        lines.append("\n** MANY UNVERIFIED CLAIMS — targeted verification needed **")
    else:
        lines.append("\n** ADEQUATE COVERAGE — ready for synthesis **")

    if topic_hints:
        lines.append("\nExpansion gaps:")
        for h in topic_hints[:5]:
            lines.append(f"  - {h['expansion_gap']} (priority={h['expansion_priority']:.2f})")

    if topic_contradictions:
        lines.append("\nContradictions to resolve:")
        for c in topic_contradictions[:3]:
            lines.append(f"  - {c['claim_a']['fact'][:100]}")
            lines.append(f"    vs: {c['claim_b']['fact'][:100]}")

    return "\n".join(lines)


def get_gap_analysis() -> str:
    """Produce structured gap analysis from the current corpus state.

    Queries for: low-confidence claims, unverified/speculative conditions,
    unfulfilled expansion hints, contradictions, and synthesis highlights.

    Use this after gossip synthesis to decide what the researcher
    should focus on next.

    Returns:
        Structured gap analysis text.
    """
    store = _get_store()
    syntheses = store.get_all_syntheses()
    latest_iteration = max((s["iteration"] for s in syntheses), default=0)
    return store.query_gaps(user_query="", iteration=latest_iteration)


def trigger_gossip(iteration: int = 0) -> str:
    """Run gossip swarm synthesis on the current corpus.

    Exports the corpus from ConditionStore as structured text,
    runs the 6-worker gossip swarm with 3 rounds, and stores
    the full synthesis back as row_type='synthesis'.

    No truncation. Full report text stored and returned.

    Args:
        iteration: Current iteration number for tracking.

    Returns:
        The full gossip synthesis report and quality metrics.
    """
    import concurrent.futures

    from swarm_bridge import gossip_synthesize

    store = _get_store()
    corpus_text = store.export_for_swarm(min_confidence=0.0)

    if "(corpus is empty" in corpus_text:
        return "Cannot run gossip synthesis — corpus is empty. Run research first."

    # gossip_synthesize is async; run it in a fresh event loop in a thread
    # because we're called from a sync tool context inside LangGraph
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                asyncio.run,
                gossip_synthesize(corpus=corpus_text, query=store.user_query),
            ).result()
    else:
        result = asyncio.run(
            gossip_synthesize(corpus=corpus_text, query=store.user_query),
        )

    # Store full synthesis in ConditionStore (NO truncation)
    metrics_dict: dict[str, Any] = {}
    if hasattr(result, "metrics"):
        m = result.metrics
        metrics_dict = {
            "info_gain": list(getattr(m, "gossip_info_gain", [])),
            "llm_calls": getattr(m, "total_llm_calls", 0),
            "elapsed_seconds": getattr(m, "total_elapsed_s", 0),
        }

    store.admit_synthesis(
        report=result.user_report,
        iteration=iteration,
        metrics=metrics_dict,
    )

    # Also store knowledge report if available (was previously discarded)
    # Uses row_type='knowledge_report' to avoid polluting synthesis queries
    # (get_synthesis and get_all_syntheses filter on row_type='synthesis')
    knowledge_report = getattr(result, "knowledge_report", "")
    if knowledge_report and knowledge_report.strip():
        strategy = json.dumps(metrics_dict) if metrics_dict else ""
        store.admit(
            fact=knowledge_report,
            source_type="knowledge_report",
            row_type="knowledge_report",
            strategy=strategy,
            iteration=iteration,
            confidence=0.8,
        )

    lines = [
        f"=== GOSSIP SYNTHESIS (iteration {iteration}) ===",
        f"Corpus size: {store.count()} conditions",
        f"User report: {len(result.user_report)} chars",
        f"Knowledge report: {len(knowledge_report)} chars",
    ]
    if metrics_dict.get("info_gain"):
        lines.append(f"Info gain per round: {metrics_dict['info_gain']}")
    if metrics_dict.get("llm_calls"):
        lines.append(f"LLM calls: {metrics_dict['llm_calls']}")

    lines.append(f"\n{result.user_report}")
    return "\n".join(lines)


def build_report(include_sources: bool = True) -> str:
    """Build a complete report from all synthesis rows in the corpus.

    Queries ConditionStore for all synthesis reports plus high-confidence
    findings. No truncation. Full text.

    Use this when research is complete and you want to produce the
    final output for the user.

    Args:
        include_sources: Whether to append a source catalogue.

    Returns:
        Full report text compiled from corpus state.
    """
    store = _get_store()
    return store.build_report(user_query="", include_sources=include_sources)
