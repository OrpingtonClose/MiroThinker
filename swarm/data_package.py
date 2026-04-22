# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Data package builder — 7-section structured research briefs.

The data package is the single most important design element. It determines
what the worker thinks about. A bad package produces generic summaries.
A good package kindles deep, angle-specific reasoning with productive surprise.

Architecture reference: docs/SWARM_WAVE_ARCHITECTURE.md § "The Data Package"

Structure:
    § 1  KNOWLEDGE STATE      — rolling summary of accumulated angle knowledge
    § 2  CORPUS MATERIAL      — raw corpus excerpts relevant to this angle
    § 3  FROM THE HIVE        — findings from OTHER angles (cross-angle RAG)
    § 4  CROSS-DOMAIN CONNECTIONS — material validated by multiple experts
    § 5  CHALLENGES           — expert-informed dissent from other angles
    § 6  RESEARCH GAPS        — identified gaps and unanswered questions
    § 7  PREVIOUS OUTPUT      — worker's prior wave analysis for continuity
    § 8  FRESH EVIDENCE       — clone research resolving worker doubts

Wave behavior:
    Wave 1: Only §2 populated (bootstrap from raw corpus)
    Wave 2: §1, §2, §3 (keyword RAG), §6, §7 populated
    Wave 3+: All 7 sections populated (full expert mode)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from swarm.rag import extract_concepts, query_hive

if TYPE_CHECKING:
    from corpus import ConditionStore
    from swarm.lineage import LineageEntry

logger = logging.getLogger(__name__)


@dataclass
class DataPackage:
    """Structured research brief for a tool-free worker.

    Attributes:
        angle: The research angle this package is built for.
        wave: Current wave number.
        worker_id: Identifier for the worker receiving this package.
        model: Model name assigned to this worker.
        knowledge_state: §1 — rolling summary of accumulated knowledge.
        corpus_material: §2 — raw corpus excerpts for this angle.
        hive_findings: §3 — cross-angle findings from other workers.
        cross_domain: §4 — validated cross-domain connections.
        challenges: §5 — expert-informed dissent from other angles.
        research_gaps: §6 — identified gaps and unanswered questions.
        previous_output: §7 — worker's prior wave analysis.
        fresh_evidence: §8 — clone research resolving worker doubts.
    """

    angle: str
    wave: int
    worker_id: str
    model: str = ""
    knowledge_state: str = ""
    corpus_material: str = ""
    hive_findings: str = ""
    cross_domain: str = ""
    challenges: str = ""
    research_gaps: str = ""
    previous_output: str = ""
    fresh_evidence: str = ""

    def render(self, query: str) -> str:
        """Render the data package as a structured prompt for the worker.

        Args:
            query: The user's research query.

        Returns:
            Formatted multi-section research brief string.
        """
        sections: list[str] = []

        sections.append(
            f"{'═' * 60}\n"
            f"  RESEARCH BRIEF\n"
            f"  Wave {self.wave}, Angle: {self.angle}\n"
            f"  Worker: {self.worker_id}, Model: {self.model}\n"
            f"{'═' * 60}\n"
        )

        sections.append(f"RESEARCH QUERY: {query}\n")

        if self.knowledge_state:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 1  KNOWLEDGE STATE\n"
                f"{'─' * 40}\n"
                f"What the swarm has established about {self.angle} so far:\n\n"
                f"{self.knowledge_state}\n"
            )

        if self.corpus_material:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 2  CORPUS MATERIAL\n"
                f"{'─' * 40}\n"
                f"Raw source material relevant to {self.angle}:\n\n"
                f"{self.corpus_material}\n"
            )

        if self.hive_findings:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 3  FROM THE HIVE\n"
                f"{'─' * 40}\n"
                f"Findings from OTHER research angles that relate to your "
                f"current analysis. Interpret them through your {self.angle} "
                f"lens — what do these cross-domain findings MEAN in your "
                f"domain?\n\n"
                f"{self.hive_findings}\n"
            )

        if self.cross_domain:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 4  CROSS-DOMAIN CONNECTIONS\n"
                f"{'─' * 40}\n"
                f"Material that multiple experts agree bridges their "
                f"domains:\n\n"
                f"{self.cross_domain}\n"
            )

        if self.challenges:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 5  CHALLENGES\n"
                f"{'─' * 40}\n"
                f"Other specialists question some of your findings. "
                f"Address these challenges with evidence from your "
                f"domain:\n\n"
                f"{self.challenges}\n"
            )

        if self.research_gaps:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 6  RESEARCH GAPS\n"
                f"{'─' * 40}\n"
                f"Known gaps in the current understanding of "
                f"{self.angle}:\n\n"
                f"{self.research_gaps}\n"
            )

        if self.previous_output:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 7  YOUR PREVIOUS ANALYSIS\n"
                f"{'─' * 40}\n"
                f"Your analysis from the previous wave. Build on this — "
                f"deepen, correct, and extend:\n\n"
                f"{self.previous_output}\n"
            )

        if self.fresh_evidence:
            sections.append(
                f"{'─' * 40}\n"
                f"§ 8  FRESH EVIDENCE (clone research results)\n"
                f"{'─' * 40}\n"
                f"{self.fresh_evidence}\n"
            )

        sections.append(
            f"\n{'═' * 60}\n"
            f"INSTRUCTIONS: You are a {self.angle} specialist. Analyze "
            f"ALL material above through your {self.angle} lens. "
            f"Produce specific, evidence-backed findings. Preserve exact "
            f"numbers, dosages, and citations. When cross-domain material "
            f"connects to your domain, explain the mechanism. No "
            f"disclaimers, no moral commentary. Practitioner language.\n"
            f"{'═' * 60}"
        )

        return "\n".join(sections)


def build_data_packages(
    store: "ConditionStore",
    assignments: list[Any],
    wave: int,
    query: str,
    *,
    prior_outputs: dict[str, str] | None = None,
    model_map: dict[str, str] | None = None,
    default_model: str = "",
    lineage_entries: list["LineageEntry"] | None = None,
) -> list[DataPackage]:
    """Build data packages for all workers in a wave.

    Follows the progressive enrichment described in
    SWARM_WAVE_ARCHITECTURE.md:
        Wave 1: Only §2 (corpus material)
        Wave 2: §1, §2, §3 (keyword RAG), §6, §7
        Wave 3+: All 7 sections

    Args:
        store: The shared ConditionStore.
        assignments: List of WorkerAssignment objects from angle detection.
        wave: Current wave number (1-based).
        query: The user's research query.
        prior_outputs: Map of angle → previous wave's worker output text.
        model_map: Map of angle → model name for per-worker model assignment.
        default_model: Fallback model name when angle is not in model_map.
        lineage_entries: Accumulated lineage entries for hive RAG queries.

    Returns:
        List of DataPackage objects, one per worker.
    """
    prior_outputs = prior_outputs or {}
    model_map = model_map or {}

    packages: list[DataPackage] = []

    for a in assignments:
        angle = a.angle
        worker_id = f"worker_{a.worker_id}_wave_{wave}"
        model = model_map.get(angle, default_model)

        pkg = DataPackage(
            angle=angle,
            wave=wave,
            worker_id=worker_id,
            model=model,
        )

        # § 2: CORPUS MATERIAL — always present (raw section assignment)
        pkg.corpus_material = a.raw_content

        if wave >= 2:
            # § 1: KNOWLEDGE STATE — rolling summary from store
            pkg.knowledge_state = _get_knowledge_state(store, angle)

            # § 3: FROM THE HIVE — cross-angle RAG
            if lineage_entries:
                pkg.hive_findings = _get_hive_findings(
                    lineage_entries, angle, prior_outputs.get(angle, ""),
                )
            else:
                pkg.hive_findings = _get_hive_findings_from_store(store, angle)

            # § 6: RESEARCH GAPS
            pkg.research_gaps = _get_research_gaps(
                store, angle, prior_outputs.get(angle, ""),
            )

            # § 7: PREVIOUS OUTPUT
            pkg.previous_output = prior_outputs.get(angle, "")

        if wave >= 2:
            # § 8: FRESH EVIDENCE — clone research from previous wave
            pkg.fresh_evidence = _get_fresh_evidence(store, angle, wave)

        if wave >= 3:
            # § 4: CROSS-DOMAIN CONNECTIONS
            pkg.cross_domain = _get_cross_domain_connections(store, angle)

            # § 5: CHALLENGES
            pkg.challenges = _get_challenges(store, angle)

        logger.info(
            "angle=<%s>, wave=<%d>, sections_populated=<%d> | data package assembled",
            angle, wave,
            sum(1 for s in [
                pkg.knowledge_state, pkg.corpus_material, pkg.hive_findings,
                pkg.cross_domain, pkg.challenges, pkg.research_gaps,
                pkg.previous_output, pkg.fresh_evidence,
            ] if s),
        )

        packages.append(pkg)

    return packages


def _get_knowledge_state(store: "ConditionStore", angle: str) -> str:
    """Retrieve the rolling knowledge summary for an angle.

    Args:
        store: The shared ConditionStore.
        angle: Research angle to retrieve summary for.

    Returns:
        Knowledge summary string, or empty if none exists.
    """
    try:
        summary = store.get_latest_summary(angle)
        if summary:
            return summary
    except Exception as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | failed to retrieve knowledge summary",
            angle, exc,
        )

    # Fallback: build a quick summary from top findings
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact, confidence
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND angle = ?
                     AND row_type IN ('finding', 'thought', 'insight')
                   ORDER BY confidence DESC
                   LIMIT 20""",
                [angle],
            ).fetchall()

        if rows:
            parts = [f"Established findings for {angle}:"]
            for fact, conf in rows:
                parts.append(f"- [{conf:.1f}] {fact}")
            return "\n".join(parts)
    except Exception as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | failed to build fallback knowledge state",
            angle, exc,
        )

    return ""


def _get_hive_findings(
    lineage_entries: list["LineageEntry"],
    angle: str,
    prior_output: str,
) -> str:
    """Get cross-angle RAG findings using keyword scoring.

    Extracts concepts from the worker's prior output and queries the
    hive for relevant findings from other angles.

    Args:
        lineage_entries: All lineage entries from prior waves.
        angle: This worker's angle (excluded from results).
        prior_output: This worker's previous wave output.

    Returns:
        Formatted cross-angle findings string.
    """
    if not prior_output:
        return ""

    concepts = extract_concepts(prior_output, top_k=15)
    if not concepts:
        return ""

    hive_results = query_hive(
        entries=lineage_entries,
        concepts=concepts,
        exclude_angle=angle,
        top_k=10,
        min_score=1.0,
    )

    if not hive_results:
        return ""

    return "\n\n".join(hive_results)


def _get_hive_findings_from_store(
    store: "ConditionStore",
    angle: str,
) -> str:
    """Get cross-angle findings directly from the ConditionStore.

    Used when lineage_entries are not available. Queries the store
    for findings from other angles.

    Args:
        store: The shared ConditionStore.
        angle: This worker's angle (excluded from results).

    Returns:
        Formatted cross-angle findings string.
    """
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact, angle, confidence
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND angle != ?
                     AND row_type IN ('finding', 'thought', 'insight')
                     AND source_type != 'corpus_section'
                   ORDER BY confidence DESC
                   LIMIT 15""",
                [angle],
            ).fetchall()

        if not rows:
            return ""

        parts: list[str] = []
        for fact, src_angle, conf in rows:
            parts.append(f"[{src_angle}] {fact}")

        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | failed to query store for hive findings",
            angle, exc,
        )
        return ""


def _get_research_gaps(
    store: "ConditionStore",
    angle: str,
    prior_output: str,
) -> str:
    """Identify research gaps for an angle.

    Extracts gap markers from prior output and queries the store for
    unresolved research questions.

    Args:
        store: The shared ConditionStore.
        angle: Research angle.
        prior_output: Worker's previous wave output.

    Returns:
        Formatted research gaps string.
    """
    gaps: list[str] = []

    # Extract gap markers from prior output
    if prior_output:
        gap_markers = [
            "need more data", "insufficient evidence", "unexplained",
            "unclear", "unknown", "uncertain", "no evidence",
            "further research", "not enough", "gap in",
            "missing data", "unresolved", "contradictory",
        ]
        lines = prior_output.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(marker in line_lower for marker in gap_markers):
                gaps.append(f"- From prior analysis: {line.strip()}")

    # Check store for research question rows
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact FROM conditions
                   WHERE row_type = 'research_question'
                     AND angle = ?
                     AND consider_for_use = TRUE
                   ORDER BY id DESC
                   LIMIT 10""",
                [angle],
            ).fetchall()

        for (fact,) in rows:
            gaps.append(f"- Open question: {fact}")
    except Exception as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | failed to query research questions",
            angle, exc,
        )

    return "\n".join(gaps) if gaps else ""


def _get_cross_domain_connections(
    store: "ConditionStore",
    angle: str,
) -> str:
    """Get validated cross-domain connections for an angle.

    Queries the store for insight rows that bridge this angle
    with other domains.

    Args:
        store: The shared ConditionStore.
        angle: Research angle.

    Returns:
        Formatted cross-domain connections string.
    """
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact, angle FROM conditions
                   WHERE row_type = 'insight'
                     AND consider_for_use = TRUE
                     AND (angle = ? OR fact LIKE '%' || replace(replace(?, '%', '\%'), '_', '\_') || '%' ESCAPE '\')
                   ORDER BY confidence DESC
                   LIMIT 10""",
                [angle, angle],
            ).fetchall()

        if not rows:
            return ""

        parts: list[str] = []
        for fact, src_angle in rows:
            parts.append(f"[{src_angle}] {fact}")

        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | failed to query cross-domain connections",
            angle, exc,
        )
        return ""


def _get_challenges(
    store: "ConditionStore",
    angle: str,
) -> str:
    """Get expert-informed challenges for an angle.

    Queries the store for contradiction rows where other angles
    disagree with this angle's findings.

    Args:
        store: The shared ConditionStore.
        angle: Research angle.

    Returns:
        Formatted challenges string.
    """
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT c.fact, c.angle, target.fact as target_fact
                   FROM conditions c
                   LEFT JOIN conditions target ON c.related_id = target.id
                   WHERE c.row_type = 'contradiction'
                     AND c.consider_for_use = TRUE
                     AND (target.angle = ? OR c.angle = ?)
                   ORDER BY c.id DESC
                   LIMIT 5""",
                [angle, angle],
            ).fetchall()

        if not rows:
            return ""

        parts: list[str] = []
        for fact, src_angle, target_fact in rows:
            challenge = f"The {src_angle} specialist questions: {fact}"
            if target_fact:
                challenge += f"\n  Regarding your finding: {target_fact}"
            parts.append(challenge)

        return "\n\n".join(parts)
    except Exception as exc:
        logger.debug(
            "angle=<%s>, error=<%s> | failed to query challenges",
            angle, exc,
        )
        return ""


def _get_fresh_evidence(
    store: "ConditionStore",
    angle: str,
    wave: int,
) -> str:
    """Get fresh evidence from clone research for this angle.

    Retrieves clone_research findings stored during the previous wave's
    Research Organizer run.  These are presented as doubt-resolution
    pairs in §8 of the data package.

    Args:
        store: The shared ConditionStore.
        angle: Research angle.
        wave: Current wave number (looks for evidence from wave-1).

    Returns:
        Formatted fresh evidence string, or empty if no clone
        research exists for this angle.
    """
    try:
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact, confidence, user_query, source_ref
                   FROM conditions
                   WHERE source_type = 'clone_research'
                     AND angle = ?
                     AND iteration = ?
                   ORDER BY confidence DESC
                   LIMIT 10""",
                [angle, wave - 1],
            ).fetchall()
    except Exception as exc:
        logger.debug(
            "angle=<%s>, wave=<%d>, error=<%s> | "
            "failed to retrieve clone research findings",
            angle, wave, exc,
        )
        return ""

    if not rows:
        return ""

    lines = [
        "Your clone-researcher investigated specific doubts from your "
        "previous analysis. Here's what it found:\n"
    ]

    for row in rows:
        fact = row[0]
        confidence = row[1]
        doubt = row[2] or "unspecified doubt"
        source = row[3] or "clone research"
        conf_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
        lines.append(f"DOUBT: {doubt}")
        lines.append(f"EVIDENCE ({conf_label} confidence): {fact}")
        lines.append(f"SOURCE: {source}")
        lines.append("")

    lines.append(
        "Integrate this evidence into your analysis. Where it confirms "
        "your prior reasoning, strengthen your claims. Where it "
        "contradicts, revise."
    )

    return "\n".join(lines)
