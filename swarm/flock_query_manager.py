# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Flock Query Manager — flag-driven mass evaluation against cached vLLM clones.

Instead of running hundreds of parallel agents, the swarm is SIMULATED
through mass Flock queries fired at a small number of vLLM-cached clones.
Each clone represents one perspective (a bee's accumulated reasoning).
Clones take turns: load one perspective's context, fire thousands of
evaluations through it, evict, load the next.

The intelligence lives in QUERY SCHEDULING, not in agent count:

    ConditionStore state (gradient flags)
        → QueryManager selects highest-value questions
        → fires at cached clone
        → results flow back into ConditionStore
        → flags update
        → repeat

Query types are driven by gradient flag combinations:

    high novelty + low confidence  → VALIDATE: "Is this novel claim real?"
    contradiction_flag = TRUE      → ADJUDICATE: "Which side is correct?"
    high fabrication_risk           → VERIFY: "Cross-check against your knowledge"
    low specificity + high relevance → ENRICH: "Add specific data/citations"
    high actionability + unverified → GROUND: "What evidence supports this?"
    cross-angle findings            → BRIDGE: "What's the interaction between A and B?"

Architecture:

    ┌──────────────────────────────────────────────────┐
    │              FlockQueryManager                     │
    │  1. Read ConditionStore state + gradient flags    │
    │  2. Select highest information-gain queries       │
    │  3. Group queries by target clone (perspective)   │
    │  4. Load clone context into vLLM prefix cache     │
    │  5. Fire query battery against cached clone       │
    │  6. Parse results → new ConditionStore rows       │
    │  7. Update flags on evaluated conditions          │
    │  8. Check convergence (diminishing returns?)      │
    │  9. Next clone / next round                       │
    └──────────────────────────────────────────────────┘

The ConditionStore IS the swarm's memory.  Every evaluation result
becomes a new row.  The gradient flags on those rows drive the NEXT
round of query selection.  Scale comes from query volume and smart
flag-based selection, not from more clones or more VRAM.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)


def _get_store_lock(store: "ConditionStore") -> Any:
    """Return the store's write lock, supporting both CorpusStore and ConditionStore.

    CorpusStore (adk-agent) uses ``_write_lock`` while ConditionStore
    (strands-agent) uses ``_lock``.  This helper transparently picks
    whichever is available.
    """
    return getattr(store, "_write_lock", getattr(store, "_lock", None))


# ---------------------------------------------------------------------------
# Query types — each driven by specific flag combinations
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    """Types of Flock evaluation queries, each triggered by flag state."""

    VALIDATE = "validate"
    """High novelty + low confidence → confirm or refute novel claim."""

    ADJUDICATE = "adjudicate"
    """Contradiction flag set → determine which side is correct."""

    VERIFY = "verify"
    """High fabrication risk → cross-check against clone's knowledge."""

    ENRICH = "enrich"
    """Low specificity + high relevance → add concrete data/citations."""

    GROUND = "ground"
    """High actionability + unverified → find supporting evidence."""

    BRIDGE = "bridge"
    """Cross-angle findings → discover interaction effects."""

    CHALLENGE = "challenge"
    """High confidence from one angle → stress-test from another."""

    SYNTHESIZE = "synthesize"
    """Multiple related findings → combine into higher-order insight."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FlockQuery:
    """A single evaluation query to fire at a cached clone.

    Attributes:
        query_type: The type of evaluation being performed.
        prompt: The full prompt to send to the clone.
        target_condition_ids: Condition IDs being evaluated.
        source_angle: The angle/perspective of the clone evaluating this.
        priority: Information gain estimate (higher = evaluate first).
        metadata: Additional context for result processing.
    """

    query_type: QueryType
    prompt: str
    target_condition_ids: list[int]
    source_angle: str = ""
    priority: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlockEvaluation:
    """Result of a single Flock evaluation query.

    Attributes:
        query_type: What kind of evaluation was performed.
        condition_ids_evaluated: Which conditions were assessed.
        evaluator_angle: The perspective that evaluated.
        verdict: The clone's judgment (free text).
        score_delta: Suggested score adjustments per flag.
        new_findings: Any new findings discovered during evaluation.
        elapsed_s: Wall-clock time for this evaluation.
    """

    query_type: QueryType
    condition_ids_evaluated: list[int]
    evaluator_angle: str
    verdict: str
    score_delta: dict[str, float] = field(default_factory=dict)
    new_findings: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class CloneContext:
    """A cached clone perspective in vLLM.

    Each clone can run on a different model — they take turns, never
    simultaneous, so the VRAM constraint is per-clone not aggregate.
    This enables a multi-model roster: Ling for deep scientific
    reasoning, Qwen3.6 for speed, a smaller model for bulk, etc.

    Attributes:
        angle: The research angle this clone represents.
        context_summary: Compressed reasoning summary for this perspective.
        context_tokens: Estimated token count of the cached context.
        wave: Which wave this clone's reasoning comes from.
        worker_id: The original worker ID.
        model_id: Model identifier for this clone (e.g. ``Ling-2.5-1T``,
            ``Qwen3-235B-A22B``).  When set, the caller should route
            queries for this clone to the appropriate vLLM endpoint.
            Empty string means use the default model.
        base_url: Optional vLLM endpoint URL for this clone's model.
            Allows different clones to target different vLLM instances
            (e.g. one on TP=8 for a 1T model, another on TP=1 for 27B).
        model_kwargs: Extra model parameters for this clone (e.g.
            ``{"temperature": 0.1}`` for high-precision evaluation vs
            ``{"temperature": 0.7}`` for creative bridging).
    """

    angle: str
    context_summary: str
    context_tokens: int = 0
    wave: int = 0
    worker_id: str = ""
    model_id: str = ""
    base_url: str = ""
    model_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRoundMetrics:
    """Metrics from one round of query evaluation.

    Attributes:
        round_number: Which evaluation round this was.
        clone_angle: Which perspective was loaded.
        queries_fired: Total queries in this round.
        queries_by_type: Breakdown by query type.
        new_evaluations: Evaluation rows created.
        new_findings: New findings discovered.
        elapsed_s: Wall-clock time for the round.
        convergence_score: How much new information was gained (0-1).
    """

    round_number: int = 0
    clone_angle: str = ""
    queries_fired: int = 0
    queries_by_type: dict[str, int] = field(default_factory=dict)
    new_evaluations: int = 0
    new_findings: int = 0
    elapsed_s: float = 0.0
    convergence_score: float = 1.0


@dataclass
class FlockSwarmResult:
    """Result of a full Flock swarm simulation.

    Attributes:
        total_queries: Total queries fired across all rounds.
        total_evaluations: Total evaluation rows created.
        total_new_findings: New findings discovered.
        rounds: Per-round metrics.
        convergence_reason: Why the swarm stopped.
        elapsed_s: Total wall-clock time.
    """

    total_queries: int = 0
    total_evaluations: int = 0
    total_new_findings: int = 0
    rounds: list[QueryRoundMetrics] = field(default_factory=list)
    convergence_reason: str = ""
    elapsed_s: float = 0.0


@dataclass
class FlockQueryManagerConfig:
    """Configuration for the Flock Query Manager.

    Attributes:
        max_rounds: Maximum evaluation rounds before stopping.
        max_queries_per_round: Maximum queries per clone per round.
        batch_size: How many queries to fire in parallel.
        convergence_threshold: Stop if new information per round drops
            below this fraction of total queries.
        novelty_confidence_gap: Minimum gap between novelty and confidence
            to trigger a VALIDATE query (novelty - confidence > gap).
        fabrication_risk_floor: Minimum fabrication_risk to trigger VERIFY.
        specificity_ceiling: Maximum specificity_score for ENRICH queries.
        contradiction_boost: Priority multiplier for contradiction queries.
        cross_angle_min_relevance: Minimum relevance for BRIDGE queries.
        enable_synthesis: Whether to run SYNTHESIZE queries.
        enable_challenge: Whether to run CHALLENGE queries.
    """

    max_rounds: int = 10
    max_queries_per_round: int = 500
    batch_size: int = 20
    convergence_threshold: float = 0.02
    novelty_confidence_gap: float = 0.2
    fabrication_risk_floor: float = 0.4
    specificity_ceiling: float = 0.4
    contradiction_boost: float = 2.0
    cross_angle_min_relevance: float = 0.3
    enable_synthesis: bool = True
    enable_challenge: bool = True


# ---------------------------------------------------------------------------
# Query selection — the intelligence layer
# ---------------------------------------------------------------------------

def select_queries(
    store: "ConditionStore",
    clone: CloneContext,
    config: FlockQueryManagerConfig,
    round_number: int,
) -> list[FlockQuery]:
    """Select the highest information-gain queries for a given clone.

    Reads the ConditionStore's gradient flags and selects queries that
    maximize the information gained from evaluating against this clone's
    perspective.  Queries are sorted by priority (descending).

    The flag-driven selection rules:
        VALIDATE:  novelty > 0.6 AND confidence < 0.4
        ADJUDICATE: contradiction_flag = TRUE
        VERIFY:    fabrication_risk > threshold
        ENRICH:    specificity < threshold AND relevance > 0.5
        GROUND:    actionability > 0.6 AND verification_status = ''
        BRIDGE:    findings from OTHER angles with relevance > threshold
        CHALLENGE: confidence > 0.8 from a different angle
        SYNTHESIZE: cluster of related findings (same cluster_id)

    Args:
        store: The ConditionStore to query.
        clone: The current clone perspective.
        config: Query manager configuration.
        round_number: Current evaluation round (for dedup).

    Returns:
        Sorted list of FlockQuery objects, highest priority first.
    """
    queries: list[FlockQuery] = []
    clone_angle = clone.angle
    lock = _get_store_lock(store)

    # --- VALIDATE: high novelty, low confidence ---
    try:
        with lock:
            validate_rows = store.conn.execute(
                "SELECT id, fact, novelty_score, confidence, angle "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND novelty_score > 0.6 "
                "AND confidence < 0.4 "
                "AND score_version > 0 "
                "ORDER BY (novelty_score - confidence) DESC "
                "LIMIT ?",
                [config.max_queries_per_round // 6],
            ).fetchall()
        for cid, fact, novelty, conf, angle in validate_rows:
            priority = (novelty - conf) * 1.5
            queries.append(FlockQuery(
                query_type=QueryType.VALIDATE,
                prompt=_build_validate_prompt(fact, angle, clone_angle),
                target_condition_ids=[cid],
                source_angle=clone_angle,
                priority=priority,
                metadata={"original_angle": angle, "novelty": novelty, "confidence": conf},
            ))
    except Exception as exc:
        logger.warning("error=<%s> | VALIDATE query selection failed", exc)

    # --- ADJUDICATE: contradictions ---
    try:
        with lock:
            contra_rows = store.conn.execute(
                "SELECT c.id, c.fact, c.angle, c.confidence, "
                "       c.contradiction_partner, c2.fact, c2.angle "
                "FROM conditions c "
                "LEFT JOIN conditions c2 ON c.contradiction_partner = c2.id "
                "WHERE c.consider_for_use = TRUE "
                "AND c.contradiction_flag = TRUE "
                "AND c.row_type = 'finding' "
                "AND c.score_version > 0 "
                "ORDER BY c.confidence DESC "
                "LIMIT ?",
                [config.max_queries_per_round // 6],
            ).fetchall()
        for cid, fact, angle, conf, partner_id, partner_fact, partner_angle in contra_rows:
            if partner_fact is None:
                continue
            priority = 0.8 * config.contradiction_boost
            queries.append(FlockQuery(
                query_type=QueryType.ADJUDICATE,
                prompt=_build_adjudicate_prompt(
                    fact, angle, partner_fact, partner_angle, clone_angle,
                ),
                target_condition_ids=[cid, partner_id],
                source_angle=clone_angle,
                priority=priority,
                metadata={"side_a_angle": angle, "side_b_angle": partner_angle},
            ))
    except Exception as exc:
        logger.warning("error=<%s> | ADJUDICATE query selection failed", exc)

    # --- VERIFY: high fabrication risk ---
    try:
        with lock:
            verify_rows = store.conn.execute(
                "SELECT id, fact, fabrication_risk, angle, source_url "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND fabrication_risk > ? "
                "AND score_version > 0 "
                "ORDER BY fabrication_risk DESC "
                "LIMIT ?",
                [config.fabrication_risk_floor, config.max_queries_per_round // 6],
            ).fetchall()
        for cid, fact, fab_risk, angle, source_url in verify_rows:
            priority = fab_risk * 1.2
            queries.append(FlockQuery(
                query_type=QueryType.VERIFY,
                prompt=_build_verify_prompt(fact, angle, source_url, clone_angle),
                target_condition_ids=[cid],
                source_angle=clone_angle,
                priority=priority,
                metadata={"fabrication_risk": fab_risk, "original_angle": angle},
            ))
    except Exception as exc:
        logger.warning("error=<%s> | VERIFY query selection failed", exc)

    # --- ENRICH: low specificity, high relevance ---
    try:
        with lock:
            enrich_rows = store.conn.execute(
                "SELECT id, fact, specificity_score, relevance_score, angle "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND specificity_score < ? "
                "AND relevance_score > 0.5 "
                "AND score_version > 0 "
                "ORDER BY (relevance_score - specificity_score) DESC "
                "LIMIT ?",
                [config.specificity_ceiling, config.max_queries_per_round // 6],
            ).fetchall()
        for cid, fact, spec, rel, angle in enrich_rows:
            priority = (rel - spec) * 1.0
            queries.append(FlockQuery(
                query_type=QueryType.ENRICH,
                prompt=_build_enrich_prompt(fact, angle, clone_angle),
                target_condition_ids=[cid],
                source_angle=clone_angle,
                priority=priority,
                metadata={"specificity": spec, "relevance": rel, "original_angle": angle},
            ))
    except Exception as exc:
        logger.warning("error=<%s> | ENRICH query selection failed", exc)

    # --- GROUND: high actionability, unverified ---
    try:
        with lock:
            ground_rows = store.conn.execute(
                "SELECT id, fact, actionability_score, angle "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND actionability_score > 0.6 "
                "AND (verification_status = '' OR verification_status IS NULL) "
                "AND score_version > 0 "
                "ORDER BY actionability_score DESC "
                "LIMIT ?",
                [config.max_queries_per_round // 6],
            ).fetchall()
        for cid, fact, action_score, angle in ground_rows:
            priority = action_score * 0.9
            queries.append(FlockQuery(
                query_type=QueryType.GROUND,
                prompt=_build_ground_prompt(fact, angle, clone_angle),
                target_condition_ids=[cid],
                source_angle=clone_angle,
                priority=priority,
                metadata={"actionability": action_score, "original_angle": angle},
            ))
    except Exception as exc:
        logger.warning("error=<%s> | GROUND query selection failed", exc)

    # --- BRIDGE: cross-angle findings ---
    try:
        with lock:
            bridge_rows = store.conn.execute(
                "SELECT id, fact, angle, relevance_score, novelty_score "
                "FROM conditions "
                "WHERE consider_for_use = TRUE "
                "AND row_type = 'finding' "
                "AND angle != ? "
                "AND relevance_score > ? "
                "AND score_version > 0 "
                "ORDER BY (novelty_score * relevance_score) DESC "
                "LIMIT ?",
                [clone_angle, config.cross_angle_min_relevance,
                 config.max_queries_per_round // 6],
            ).fetchall()
        for cid, fact, angle, rel, nov in bridge_rows:
            priority = nov * rel * 1.3
            queries.append(FlockQuery(
                query_type=QueryType.BRIDGE,
                prompt=_build_bridge_prompt(fact, angle, clone_angle),
                target_condition_ids=[cid],
                source_angle=clone_angle,
                priority=priority,
                metadata={"from_angle": angle, "relevance": rel, "novelty": nov},
            ))
    except Exception as exc:
        logger.warning("error=<%s> | BRIDGE query selection failed", exc)

    # Sort by priority descending, cap at max
    queries.sort(key=lambda q: q.priority, reverse=True)
    return queries[:config.max_queries_per_round]


# ---------------------------------------------------------------------------
# Prompt builders — short, focused evaluation prompts
# ---------------------------------------------------------------------------

def _build_validate_prompt(fact: str, origin_angle: str, evaluator_angle: str) -> str:
    return (
        f"EVALUATE from your {evaluator_angle} expertise.\n"
        f"A researcher studying {origin_angle} claims:\n"
        f'"{fact}"\n\n'
        f"Is this claim valid? Rate confidence 0.0-1.0 and explain briefly.\n"
        f"Format: CONFIDENCE: X.X\nVERDICT: [supported/refuted/insufficient_evidence]\n"
        f"REASONING: [1-2 sentences]"
    )


def _build_adjudicate_prompt(
    fact_a: str, angle_a: str,
    fact_b: str, angle_b: str,
    evaluator_angle: str,
) -> str:
    return (
        f"ADJUDICATE from your {evaluator_angle} expertise.\n"
        f"Two findings contradict each other:\n\n"
        f"SIDE A ({angle_a}):\n\"{fact_a}\"\n\n"
        f"SIDE B ({angle_b}):\n\"{fact_b}\"\n\n"
        f"Which side does the evidence support? Or is this a false contradiction "
        f"(both can be true under different conditions)?\n"
        f"Format: VERDICT: [side_a/side_b/both_valid/neither]\n"
        f"CONFIDENCE: X.X\nREASONING: [1-2 sentences]\n"
        f"CONDITIONS: [under what conditions each holds, if both_valid]"
    )


def _build_verify_prompt(
    fact: str, angle: str, source_url: str, evaluator_angle: str,
) -> str:
    source_text = source_url if source_url else "(no source cited)"
    return (
        f"VERIFY from your {evaluator_angle} expertise.\n"
        f"This finding has been flagged as potentially fabricated:\n"
        f'"{fact}"\n'
        f"Source: {source_text}\n\n"
        f"Does this match your knowledge? Are the specific claims "
        f"(names, numbers, studies cited) real?\n"
        f"Format: VERDICT: [verified/likely_fabricated/partially_true/unverifiable]\n"
        f"CONFIDENCE: X.X\nREASONING: [1-2 sentences]"
    )


def _build_enrich_prompt(fact: str, origin_angle: str, evaluator_angle: str) -> str:
    return (
        f"ENRICH from your {evaluator_angle} expertise.\n"
        f"This finding from {origin_angle} is relevant but lacks specifics:\n"
        f'"{fact}"\n\n'
        f"Add concrete data: specific numbers, dosages, study names, "
        f"mechanisms, or citations that make this claim precise and testable.\n"
        f"Format: ENRICHED_CLAIM: [the claim with added specifics]\n"
        f"ADDED_DATA: [list each specific data point you added]\n"
        f"SOURCES: [any sources for the added data]"
    )


def _build_ground_prompt(fact: str, origin_angle: str, evaluator_angle: str) -> str:
    return (
        f"GROUND from your {evaluator_angle} expertise.\n"
        f"This actionable finding needs evidence grounding:\n"
        f'"{fact}"\n\n'
        f"What specific evidence supports or refutes this? Cite mechanisms, "
        f"studies, or established principles.\n"
        f"Format: EVIDENCE_FOR: [supporting evidence]\n"
        f"EVIDENCE_AGAINST: [contradicting evidence]\n"
        f"NET_ASSESSMENT: [supported/contested/unsupported]\n"
        f"CONFIDENCE: X.X"
    )


def _build_bridge_prompt(fact: str, origin_angle: str, evaluator_angle: str) -> str:
    return (
        f"BRIDGE from your {evaluator_angle} expertise.\n"
        f"This finding comes from {origin_angle}:\n"
        f'"{fact}"\n\n'
        f"How does this interact with your {evaluator_angle} domain? "
        f"Look for: mechanistic links, compounding effects, contradictions, "
        f"or shared upstream causes.\n"
        f"Format: INTERACTION: [describe the cross-domain connection]\n"
        f"TYPE: [amplifies/contradicts/shares_mechanism/independent]\n"
        f"IMPLICATION: [what this means for the research]\n"
        f"CONFIDENCE: X.X"
    )


def _build_synthesize_prompt(
    facts: list[str], angles: list[str], evaluator_angle: str,
) -> str:
    findings_block = "\n".join(
        f"  [{a}] {f}" for f, a in zip(facts, angles)
    )
    return (
        f"SYNTHESIZE from your {evaluator_angle} expertise.\n"
        f"These related findings span multiple angles:\n"
        f"{findings_block}\n\n"
        f"What higher-order insight emerges from combining them? "
        f"What do they collectively imply that none implies alone?\n"
        f"Format: SYNTHESIS: [the emergent insight]\n"
        f"MECHANISM: [the underlying mechanism connecting them]\n"
        f"PREDICTION: [what this predicts that could be tested]\n"
        f"CONFIDENCE: X.X"
    )


def _build_challenge_prompt(
    fact: str, origin_angle: str, evaluator_angle: str,
) -> str:
    return (
        f"CHALLENGE from your {evaluator_angle} expertise.\n"
        f"This high-confidence finding from {origin_angle} needs stress-testing:\n"
        f'"{fact}"\n\n'
        f"What are the strongest objections? Under what conditions does this "
        f"break down? What edge cases or confounders are being ignored?\n"
        f"Format: OBJECTIONS: [strongest counter-arguments]\n"
        f"FAILURE_CONDITIONS: [when this claim breaks down]\n"
        f"CONFOUNDERS: [ignored variables]\n"
        f"SURVIVES_CHALLENGE: [yes/partially/no]\n"
        f"CONFIDENCE: X.X"
    )


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_evaluation_result(
    raw_response: str,
    query: FlockQuery,
    elapsed_s: float,
) -> FlockEvaluation:
    """Parse a clone's response into a structured FlockEvaluation.

    Extracts CONFIDENCE, VERDICT, and any new findings from the response.
    Tolerant of format variations — extracts what it can.

    Args:
        raw_response: The clone's raw text response.
        query: The original query that produced this response.
        elapsed_s: Time taken for this evaluation.

    Returns:
        Structured FlockEvaluation.
    """
    import re

    verdict = raw_response.strip()
    score_delta: dict[str, float] = {}
    new_findings: list[dict[str, Any]] = []

    # Extract CONFIDENCE if present
    conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", raw_response, re.IGNORECASE)
    if conf_match:
        try:
            conf_val = max(0.0, min(1.0, float(conf_match.group(1))))
            score_delta["confidence"] = conf_val
        except ValueError:
            pass

    # Extract VERDICT
    verdict_match = re.search(
        r"VERDICT:\s*(\S+)", raw_response, re.IGNORECASE,
    )
    if verdict_match:
        verdict = verdict_match.group(1).strip()

    # For ENRICH queries, extract the enriched claim as a new finding
    if query.query_type == QueryType.ENRICH:
        enriched_match = re.search(
            r"ENRICHED_CLAIM:\s*(.+?)(?:\n[A-Z]|\Z)",
            raw_response, re.IGNORECASE | re.DOTALL,
        )
        if enriched_match:
            enriched_text = enriched_match.group(1).strip()
            if len(enriched_text) > 30:
                new_findings.append({
                    "fact": enriched_text,
                    "row_type": "finding",
                    "source_type": "flock_enrichment",
                    "angle": query.source_angle,
                    "confidence": score_delta.get("confidence", 0.6),
                })

    # For BRIDGE queries, extract interaction as a new finding
    if query.query_type == QueryType.BRIDGE:
        interaction_match = re.search(
            r"INTERACTION:\s*(.+?)(?:\n[A-Z]|\Z)",
            raw_response, re.IGNORECASE | re.DOTALL,
        )
        if interaction_match:
            interaction_text = interaction_match.group(1).strip()
            if len(interaction_text) > 30:
                new_findings.append({
                    "fact": interaction_text,
                    "row_type": "insight",
                    "source_type": "flock_bridge",
                    "angle": query.source_angle,
                    "confidence": score_delta.get("confidence", 0.5),
                })

    # For SYNTHESIZE queries, extract synthesis as a new finding
    if query.query_type == QueryType.SYNTHESIZE:
        synth_match = re.search(
            r"SYNTHESIS:\s*(.+?)(?:\n[A-Z]|\Z)",
            raw_response, re.IGNORECASE | re.DOTALL,
        )
        if synth_match:
            synth_text = synth_match.group(1).strip()
            if len(synth_text) > 30:
                new_findings.append({
                    "fact": synth_text,
                    "row_type": "synthesis",
                    "source_type": "flock_synthesis",
                    "angle": query.source_angle,
                    "confidence": score_delta.get("confidence", 0.5),
                })

    # For VERIFY, adjust fabrication_risk based on structured verdict only.
    # When verdict_match fails, verdict is the entire raw response —
    # substring matching against free text would produce false positives
    # (e.g. "could not be verified" matching "verified").
    if query.query_type == QueryType.VERIFY and verdict_match:
        verdict_lower = verdict.lower()
        if verdict_lower == "verified":
            score_delta["fabrication_risk"] = 0.1
        elif "fabricated" in verdict_lower:
            score_delta["fabrication_risk"] = 0.9
        elif "partially" in verdict_lower:
            score_delta["fabrication_risk"] = 0.5

    # For CHALLENGE, adjust confidence if challenge succeeds
    if query.query_type == QueryType.CHALLENGE:
        survives_match = re.search(
            r"SURVIVES_CHALLENGE:\s*(\S+)", raw_response, re.IGNORECASE,
        )
        if survives_match:
            survives = survives_match.group(1).lower()
            if survives == "no":
                score_delta["confidence"] = max(
                    0.2, score_delta.get("confidence", 0.5) - 0.3,
                )
            elif survives == "partially":
                score_delta["confidence"] = max(
                    0.3, score_delta.get("confidence", 0.5) - 0.15,
                )

    return FlockEvaluation(
        query_type=query.query_type,
        condition_ids_evaluated=query.target_condition_ids,
        evaluator_angle=query.source_angle,
        verdict=verdict,
        score_delta=score_delta,
        new_findings=new_findings,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Store integration — write evaluation results back
# ---------------------------------------------------------------------------

def store_evaluation(
    store: "ConditionStore",
    evaluation: FlockEvaluation,
    query: FlockQuery,
    run_id: str,
    round_number: int,
) -> tuple[int, float]:
    """Write an evaluation result into the ConditionStore.

    Creates a new 'evaluation' row linking back to the evaluated
    conditions, and optionally updates gradient flags on the
    evaluated conditions based on the evaluation's score_delta.

    Args:
        store: The ConditionStore.
        evaluation: The evaluation result.
        query: The original query.
        run_id: Current run identifier.
        round_number: Current evaluation round.

    Returns:
        Tuple of (rows_created, score_magnitude) where rows_created
        is the count of new rows (1 evaluation + any new findings)
        and score_magnitude is the total absolute change across all
        updated flags (used for convergence detection).
    """
    import json

    rows_created = 0
    score_magnitude = 0.0

    # 1. Create the evaluation row itself
    eval_fact = (
        f"[{evaluation.query_type.value}] "
        f"Evaluator: {evaluation.evaluator_angle} | "
        f"Verdict: {evaluation.verdict[:200]}"
    )
    metadata = {
        "query_type": evaluation.query_type.value,
        "target_ids": evaluation.condition_ids_evaluated,
        "score_delta": evaluation.score_delta,
        "round": round_number,
        "elapsed_s": evaluation.elapsed_s,
    }

    with _get_store_lock(store):
        cid = store._next_id
        store._next_id += 1
        store.conn.execute(
            """INSERT INTO conditions
               (id, fact, source_type, source_ref, row_type,
                consider_for_use, angle, strategy,
                created_at, phase, parent_ids)
               VALUES (?, ?, 'flock_evaluation', ?, 'evaluation', TRUE,
                       ?, ?, ?, 'flock_round', ?)""",
            [
                cid,
                eval_fact,
                f"flock_round_{round_number}",
                evaluation.evaluator_angle,
                json.dumps(metadata),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(evaluation.condition_ids_evaluated),
            ],
        )
        rows_created += 1

    # 2. Apply score deltas to evaluated conditions
    if evaluation.score_delta:
        if (
            evaluation.query_type == QueryType.ADJUDICATE
            and len(evaluation.condition_ids_evaluated) == 2
        ):
            # Asymmetric application: the evaluator's confidence is
            # confidence in the verdict, not in both claims equally.
            # side_a = condition_ids_evaluated[0]
            # side_b = condition_ids_evaluated[1]
            verdict_lower = evaluation.verdict.lower()
            eval_conf = evaluation.score_delta.get("confidence", 0.5)
            if verdict_lower == "side_a":
                winner_delta = {"confidence": eval_conf}
                loser_delta = {"confidence": max(0.1, 1.0 - eval_conf)}
            elif verdict_lower == "side_b":
                winner_delta = {"confidence": max(0.1, 1.0 - eval_conf)}
                loser_delta = {"confidence": eval_conf}
            elif verdict_lower == "both_valid":
                # Both sides are conditionally true — moderate boost
                winner_delta = {"confidence": min(1.0, eval_conf * 0.8)}
                loser_delta = {"confidence": min(1.0, eval_conf * 0.8)}
            else:
                # "neither" or unrecognised — penalise both
                winner_delta = {"confidence": max(0.1, 0.5 - eval_conf * 0.3)}
                loser_delta = {"confidence": max(0.1, 0.5 - eval_conf * 0.3)}
            score_magnitude += _apply_score_delta(
                store, evaluation.condition_ids_evaluated[0], winner_delta,
            )
            score_magnitude += _apply_score_delta(
                store, evaluation.condition_ids_evaluated[1], loser_delta,
            )
        else:
            for target_id in evaluation.condition_ids_evaluated:
                score_magnitude += _apply_score_delta(
                    store, target_id, evaluation.score_delta,
                )

    # 3. Store any new findings generated during evaluation
    for finding in evaluation.new_findings:
        with _get_store_lock(store):
            fid = store._next_id
            store._next_id += 1
            store.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_type, row_type,
                    consider_for_use, angle, confidence,
                    created_at, phase, parent_ids)
                   VALUES (?, ?, ?, ?, TRUE, ?, ?, ?, 'flock_discovery', ?)""",
                [
                    fid,
                    finding["fact"],
                    finding.get("source_type", "flock_evaluation"),
                    finding.get("row_type", "finding"),
                    finding.get("angle", evaluation.evaluator_angle),
                    finding.get("confidence", 0.5),
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(evaluation.condition_ids_evaluated),
                ],
            )
            rows_created += 1

    return rows_created, score_magnitude


def _apply_score_delta(
    store: "ConditionStore",
    condition_id: int,
    delta: dict[str, float],
) -> float:
    """Apply score adjustments from an evaluation to a condition.

    Uses weighted averaging: new_score = 0.7 * old + 0.3 * evaluation.
    This prevents a single evaluation from dominating the score.
    All changed flags are written in a single UPDATE statement and
    score_version is incremented exactly once per evaluation, matching
    the pattern in corpus_store.py ``_score_single``.

    Args:
        store: The ConditionStore.
        condition_id: Which condition to update.
        delta: Map of flag name → evaluation's score for that flag.

    Returns:
        Total absolute magnitude of score changes across all flags.
        As scores stabilize across rounds this approaches zero, which
        is what drives convergence detection.
    """
    flag_columns = {
        "confidence", "trust_score", "novelty_score",
        "specificity_score", "relevance_score", "actionability_score",
        "fabrication_risk",
    }

    # Filter to valid flags
    applicable = {k: v for k, v in delta.items() if k in flag_columns}
    if not applicable:
        return 0.0

    total_magnitude = 0.0
    try:
        with _get_store_lock(store):
            # Read all current values in one query
            cols = ", ".join(applicable.keys())
            old_row = store.conn.execute(
                f"SELECT {cols} FROM conditions WHERE id = ?",
                [condition_id],
            ).fetchone()
            if not old_row:
                return 0.0

            # Compute new values and accumulate magnitude
            set_clauses: list[str] = []
            params: list[float] = []
            for i, (flag_name, eval_score) in enumerate(applicable.items()):
                old_value = old_row[i] if old_row[i] is not None else 0.5
                new_value = old_value * 0.7 + eval_score * 0.3
                total_magnitude += abs(new_value - old_value)
                set_clauses.append(f"{flag_name} = ?")
                params.append(new_value)

            # Single UPDATE: all flags + one score_version increment
            set_clauses.append("score_version = score_version + 1")
            params.append(condition_id)
            store.conn.execute(
                f"UPDATE conditions SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )
    except Exception as exc:
        logger.warning(
            "condition_id=<%d>, error=<%s> | score delta application failed",
            condition_id, exc,
        )

    return total_magnitude


# ---------------------------------------------------------------------------
# The main loop — orchestrates rounds of evaluation
# ---------------------------------------------------------------------------

class FlockQueryManager:
    """Orchestrates mass Flock queries to simulate an incredibly large swarm.

    The swarm isn't hundreds of agents — it's thousands of QUERIES fired
    at a small number of cached vLLM clones, scheduled by gradient flags.

    Usage:
        manager = FlockQueryManager(
            store=condition_store,
            complete=my_llm_fn,
            config=FlockQueryManagerConfig(max_rounds=10),
        )
        result = await manager.run(
            clones=[clone_a, clone_b, clone_c],
            run_id="run_001",
        )

    Multi-model usage (clones take turns, never simultaneous):
        async def route_to_model(clone: CloneContext) -> Callable:
            return make_vllm_client(clone.base_url, clone.model_id)

        manager = FlockQueryManager(
            store=condition_store,
            complete=default_llm_fn,        # fallback
            complete_for_clone=route_to_model,  # per-clone routing
        )
        result = await manager.run(clones=[
            CloneContext(angle="lipid_metabolism", model_id="Ling-2.5-1T",
                         base_url="http://gpu1:8000/v1"),
            CloneContext(angle="mTORC1_signaling", model_id="Qwen3.6-27B",
                         base_url="http://gpu2:8000/v1"),
        ], run_id="run_001")
    """

    def __init__(
        self,
        store: "ConditionStore",
        complete: Callable[[str], Awaitable[str]],
        config: FlockQueryManagerConfig | None = None,
        complete_for_clone: Callable[[CloneContext], Awaitable[Callable[[str], Awaitable[str]]]] | None = None,
    ) -> None:
        self.store = store
        self.complete = complete
        self.config = config or FlockQueryManagerConfig()
        self._complete_for_clone = complete_for_clone

    async def run(
        self,
        clones: list[CloneContext],
        run_id: str,
        on_event: Callable[[dict], Awaitable[None]] | None = None,
    ) -> FlockSwarmResult:
        """Run the full Flock swarm simulation.

        For each round:
          1. Iterate through clones (taking turns)
          2. Select highest-value queries based on flag state
          3. Fire queries in batches
          4. Parse results and write back to ConditionStore
          5. Check convergence

        Args:
            clones: List of cached clone perspectives.
            run_id: Run identifier for provenance.
            on_event: Optional progress callback.

        Returns:
            FlockSwarmResult with complete metrics.
        """
        t0 = time.monotonic()
        result = FlockSwarmResult()
        store = self.store

        async def _emit(event: dict) -> None:
            if on_event:
                try:
                    await on_event(event)
                except Exception:
                    pass

        # Bootstrap: promote unscored findings so query filters can match.
        # ConditionStore defaults score_version to 0 and only
        # _apply_score_delta increments it — but _apply_score_delta is
        # only reachable AFTER queries are selected.  Without this step
        # all queries require score_version > 0 and nothing ever matches.
        try:
            lock = _get_store_lock(store)
            with lock:
                bootstrapped = store.conn.execute(
                    "UPDATE conditions "
                    "SET score_version = 1 "
                    "WHERE score_version = 0 "
                    "AND row_type = 'finding' "
                    "AND consider_for_use = TRUE"
                ).rowcount
            if bootstrapped:
                logger.info(
                    "bootstrapped=<%d> | promoted unscored findings to score_version=1",
                    bootstrapped,
                )
        except Exception as exc:
            logger.warning("error=<%s> | bootstrap scoring failed", exc)

        for round_num in range(1, self.config.max_rounds + 1):
            round_start = time.monotonic()
            round_queries = 0
            round_evaluations = 0
            round_new_findings = 0
            round_score_magnitude = 0.0

            await _emit({
                "type": "flock_round_start",
                "round": round_num,
                "clones": [c.angle for c in clones],
            })

            # Each clone takes a turn — clones are never simultaneous,
            # so each can use a different model (evict one, load the next)
            for clone in clones:
                clone_start = time.monotonic()
                clone_queries = 0

                # Resolve per-clone completion function (model routing)
                clone_complete = self.complete
                if self._complete_for_clone and clone.model_id:
                    try:
                        clone_complete = await self._complete_for_clone(clone)
                        logger.info(
                            "round=<%d>, clone=<%s>, model=<%s> | using per-clone model",
                            round_num, clone.angle, clone.model_id,
                        )
                    except Exception as exc:
                        logger.warning(
                            "round=<%d>, clone=<%s>, model=<%s>, error=<%s> | "
                            "per-clone model resolution failed, using default",
                            round_num, clone.angle, clone.model_id, exc,
                        )

                await _emit({
                    "type": "flock_clone_start",
                    "round": round_num,
                    "clone_angle": clone.angle,
                    "model_id": clone.model_id or "default",
                })

                # Select queries based on current flag state
                queries = select_queries(
                    self.store, clone, self.config, round_num,
                )

                if not queries:
                    logger.info(
                        "round=<%d>, clone=<%s> | no queries selected, clone exhausted",
                        round_num, clone.angle,
                    )
                    continue

                # Fire queries in batches
                for batch_start in range(0, len(queries), self.config.batch_size):
                    batch = queries[batch_start:batch_start + self.config.batch_size]
                    batch_results = await self._fire_batch(
                        batch, clone, clone_complete,
                    )

                    for query, (response, elapsed) in zip(batch, batch_results):
                        if not response:
                            continue

                        evaluation = _parse_evaluation_result(
                            response, query, elapsed,
                        )
                        rows, magnitude = store_evaluation(
                            self.store, evaluation, query,
                            run_id, round_num,
                        )
                        round_evaluations += 1
                        round_new_findings += len(evaluation.new_findings)
                        round_score_magnitude += magnitude
                        round_queries += 1
                        clone_queries += 1

                clone_time = time.monotonic() - clone_start
                logger.info(
                    "round=<%d>, clone=<%s>, model=<%s>, queries=<%d>, elapsed_s=<%.1f> | clone turn complete",
                    round_num, clone.angle, clone.model_id or "default", clone_queries, clone_time,
                )

                await _emit({
                    "type": "flock_clone_complete",
                    "round": round_num,
                    "clone_angle": clone.angle,
                    "model_id": clone.model_id or "default",
                    "queries": clone_queries,
                    "elapsed_s": round(clone_time, 1),
                })

            # Round metrics
            round_time = time.monotonic() - round_start
            # Convergence = average absolute score change per query.
            # As the ConditionStore stabilises across rounds, evaluations
            # keep running but the blended scores barely move.  The
            # magnitude tracks that: early rounds shift scores by large
            # amounts; later rounds produce near-zero deltas.
            avg_magnitude = (
                round_score_magnitude / max(round_queries, 1)
            )
            convergence_score = avg_magnitude

            round_metrics = QueryRoundMetrics(
                round_number=round_num,
                queries_fired=round_queries,
                new_evaluations=round_evaluations,
                new_findings=round_new_findings,
                elapsed_s=round_time,
                convergence_score=convergence_score,
            )
            result.rounds.append(round_metrics)
            result.total_queries += round_queries
            result.total_evaluations += round_evaluations
            result.total_new_findings += round_new_findings

            await _emit({
                "type": "flock_round_complete",
                "round": round_num,
                "queries": round_queries,
                "evaluations": round_evaluations,
                "score_magnitude": round(round_score_magnitude, 4),
                "new_findings": round_new_findings,
                "convergence_score": round(convergence_score, 4),
                "elapsed_s": round(round_time, 1),
            })

            logger.info(
                "round=<%d>, queries=<%d>, evals=<%d>, score_magnitude=<%.4f>, "
                "new_findings=<%d>, convergence=<%.4f>, elapsed_s=<%.1f> | round complete",
                round_num, round_queries, round_evaluations, round_score_magnitude,
                round_new_findings, convergence_score, round_time,
            )

            # Convergence check
            if round_queries == 0:
                result.convergence_reason = "no_queries_available"
                break

            if convergence_score < self.config.convergence_threshold:
                result.convergence_reason = (
                    f"convergence_threshold_reached "
                    f"(score={convergence_score:.3f} < "
                    f"threshold={self.config.convergence_threshold})"
                )
                break

        if not result.convergence_reason:
            result.convergence_reason = f"max_rounds_reached ({self.config.max_rounds})"

        result.elapsed_s = time.monotonic() - t0

        logger.info(
            "total_queries=<%d>, total_evals=<%d>, total_new=<%d>, "
            "elapsed_s=<%.1f>, reason=<%s> | flock swarm complete",
            result.total_queries, result.total_evaluations,
            result.total_new_findings, result.elapsed_s,
            result.convergence_reason,
        )

        return result

    async def _fire_batch(
        self,
        queries: list[FlockQuery],
        clone: CloneContext,
        complete_fn: Callable[[str], Awaitable[str]] | None = None,
    ) -> list[tuple[str, float]]:
        """Fire a batch of queries against the Flock engine.

        Each query is prepended with the clone's context summary so the
        model evaluates from that perspective.  Queries run in parallel.

        Args:
            queries: Batch of queries to fire.
            clone: The current clone perspective.
            complete_fn: Completion callable for this clone.  When
                ``None``, falls back to ``self.complete``.  This allows
                different clones to target different models/endpoints.

        Returns:
            List of (response_text, elapsed_seconds) tuples.
        """
        _complete = complete_fn or self.complete

        async def _single(query: FlockQuery) -> tuple[str, float]:
            # Prepend clone context to the query prompt
            full_prompt = (
                f"You are an expert researcher with deep knowledge in "
                f"{clone.angle}. Your accumulated analysis:\n\n"
                f"{clone.context_summary[:50000]}\n\n"
                f"{'═' * 40}\n"
                f"EVALUATION TASK:\n"
                f"{query.prompt}"
            )
            t0 = time.monotonic()
            try:
                response = await _complete(full_prompt)
                elapsed = time.monotonic() - t0
                return (response, elapsed)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "query_type=<%s>, model=<%s>, error=<%s>, elapsed_s=<%.1f> | query failed",
                    query.query_type.value, clone.model_id or "default", exc, elapsed,
                )
                return ("", elapsed)

        results = await asyncio.gather(
            *[_single(q) for q in queries],
            return_exceptions=False,
        )
        return list(results)
