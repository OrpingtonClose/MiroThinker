# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Thought swarm orchestration — parallel specialist thinkers, arbitration,
recursive thought splitting, and topology management.

This module implements the "hard" layer of the thought swarm architecture:

  1. **Specialist spawning**: Multiple parallel LLM calls, each focused on
     a different research angle extracted from the thinker's strategy.
     Results are persisted as ``row_type='thought'`` rows in DuckDB.

  2. **Thought arbitration**: When competing specialists produce conflicting
     conclusions for the same angle, an arbitration call evaluates evidence
     quality and produces a verdict thought.

  3. **Expansion-depth splitting**: Broad thoughts containing multiple
     sub-claims are decomposed into child thoughts at deeper expansion
     depths, enabling recursive specialisation.

  4. **Swarm topology (SwarmRouter)**: Parses the thinker's structured
     angle output, manages specialist concurrency, routes corpus subsets
     to the right specialists, and detects convergence.

All specialist work runs as HTTP LLM calls via ``CorpusStore._http_complete``
(the same proven pattern used by gossip synthesis).  Thread safety for
concurrent writes is handled by ``CorpusStore._write_lock``.

This module is called from ``maestro_condition_callback`` — no pipeline
structure changes are required.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env vars)
# ---------------------------------------------------------------------------

# Maximum parallel specialist thinkers per iteration
_MAX_SPECIALISTS = int(os.environ.get("SWARM_SPECIALISTS", "3"))

# Maximum recursive thought splitting depth
_MAX_THOUGHT_DEPTH = int(os.environ.get("MAX_THOUGHT_DEPTH", "3"))

# Minimum corpus findings before specialists are spawned
_MIN_FINDINGS_FOR_SWARM = int(os.environ.get("MIN_FINDINGS_FOR_SWARM", "5"))

# Toggle entire swarm on/off
_SWARM_ENABLED = os.environ.get("SWARM_ENABLED", "1") == "1"

# Minimum chars for a specialist output to be worth storing
_MIN_THOUGHT_CHARS = 100


# ═══════════════════════════════════════════════════════════════════════
# 1. Angle extraction from thinker strategy
# ═══════════════════════════════════════════════════════════════════════

def extract_angles(strategy_text: str) -> list[str]:
    """Extract specialist angles from the thinker's strategy output.

    The thinker is instructed to include a structured block::

        SPECIALIST_ANGLES: [angle1, angle2, angle3]

    Falls back to heuristic extraction from section headings if no
    structured block is found.
    """
    if not strategy_text:
        return []

    # Try structured format first
    m = re.search(
        r"SPECIALIST_ANGLES:\s*\[([^\]]+)\]",
        strategy_text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1)
        angles = [a.strip().strip("'\"") for a in raw.split(",")]
        return [a for a in angles if a and len(a) > 2][:_MAX_SPECIALISTS]

    # Fallback: extract from numbered angle items
    angle_items = re.findall(
        r"(?:angle|focus|investigate|explore)\s*(?:\d+)?[:\-]\s*(.+)",
        strategy_text,
        re.IGNORECASE,
    )
    if angle_items:
        return [a.strip()[:80] for a in angle_items][:_MAX_SPECIALISTS]

    return []


# ═══════════════════════════════════════════════════════════════════════
# 2. Specialist thinker spawning
# ═══════════════════════════════════════════════════════════════════════

def _build_specialist_prompt(
    angle: str,
    corpus_summary: str,
    prior_thoughts: list[dict],
    user_query: str,
) -> str:
    """Build the prompt for a specialist thinker focused on *angle*.

    The specialist sees:
    - The user's original query
    - A summary of the corpus (findings relevant to their angle)
    - Any prior thoughts from this angle (to build on, not repeat)
    """
    prior_section = ""
    if prior_thoughts:
        prior_lines = []
        for t in prior_thoughts[-3:]:  # last 3 thoughts for this angle
            prior_lines.append(
                f"  [thought #{t['id']}, depth={t['expansion_depth']}]: "
                f"{t['fact'][:500]}"
            )
        prior_section = (
            "\n\nPRIOR ANALYSIS FOR THIS ANGLE (build on this, do NOT repeat):\n"
            + "\n".join(prior_lines)
        )

    return f"""\
You are a specialist research analyst assigned to a specific angle of \
investigation. Your job is to provide DEEP, FOCUSED analysis on your \
assigned angle — going beyond what a generalist would see.

USER QUERY: {user_query}

YOUR ASSIGNED ANGLE: {angle}

CORPUS SUMMARY (relevant findings):
{corpus_summary[:6000]}
{prior_section}

INSTRUCTIONS:
1. Analyse the corpus findings SPECIFICALLY through the lens of your angle
2. Identify patterns, implications, and connections that a generalist might miss
3. Challenge any assumptions in the existing findings where evidence warrants
4. Propose specific hypotheses or conclusions supported by the evidence
5. Note what additional evidence would strengthen or weaken your analysis
6. If prior analysis exists for this angle, BUILD ON IT — add new insight, \
   don't repeat what's already been said

OUTPUT: A focused analytical report (500-2000 chars) with specific citations \
to evidence. Be substantive, not vague. Every claim must be grounded in \
the corpus findings."""


def spawn_specialist_thinkers(
    corpus,
    angles: list[str],
    user_query: str,
    iteration: int,
) -> list[int]:
    """Spawn parallel specialist thinkers, one per angle.

    Each specialist runs as an HTTP LLM call via ``corpus._http_complete``.
    Results are persisted as thought rows.  Thread-safe via
    ``corpus._write_lock``.

    Args:
        corpus: The ``CorpusStore`` instance.
        angles: List of angle strings to specialise on.
        user_query: The user's original query for context.
        iteration: Current pipeline iteration number.

    Returns:
        List of thought row IDs created by specialists.
    """
    if not angles:
        return []

    # Limit to configured max
    angles = angles[:_MAX_SPECIALISTS]

    # Prepare prompts for each specialist
    tasks: list[tuple[str, str, list[dict]]] = []
    corpus_summary = corpus.format_for_thinker()

    for angle in angles:
        prior = corpus.get_thoughts_by_angle(angle)
        prompt = _build_specialist_prompt(
            angle, corpus_summary, prior, user_query,
        )
        tasks.append((angle, prompt, prior))

    # Run specialists in parallel via ThreadPoolExecutor
    thought_ids: list[int] = []
    workers = min(_MAX_SPECIALISTS, len(tasks))

    def _run_specialist(
        angle: str, prompt: str,
    ) -> tuple[str, str]:
        """Execute one specialist and return (angle, response)."""
        response = corpus._http_complete(
            prompt, caller=f"swarm_specialist_{angle[:30]}",
            max_tokens=2048,
        )
        return angle, response

    logger.info(
        "Spawning %d specialist thinkers for angles: %s",
        len(tasks), [a for a, _, _ in tasks],
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_specialist, angle, prompt): angle
            for angle, prompt, _ in tasks
        }
        for fut in as_completed(futures):
            angle = futures[fut]
            try:
                _, response = fut.result()
                if response and len(response.strip()) >= _MIN_THOUGHT_CHARS:
                    tid = corpus.admit_thought(
                        reasoning=response,
                        angle=angle,
                        strategy="specialist_analysis",
                        iteration=iteration,
                    )
                    thought_ids.append(tid)
                    logger.info(
                        "Specialist '%s' produced thought #%d (%d chars)",
                        angle, tid, len(response),
                    )
                else:
                    logger.info(
                        "Specialist '%s' produced insufficient output (%d chars) — skipping",
                        angle, len(response.strip()) if response else 0,
                    )
            except Exception:
                logger.warning(
                    "Specialist '%s' failed (non-fatal)",
                    angle, exc_info=True,
                )

    logger.info(
        "Specialist spawning complete: %d/%d produced thoughts",
        len(thought_ids), len(tasks),
    )
    return thought_ids


# ═══════════════════════════════════════════════════════════════════════
# 3. Thought arbitration
# ═══════════════════════════════════════════════════════════════════════

def arbitrate_competing_thoughts(
    corpus,
    angle: str,
    iteration: int,
) -> int | None:
    """Arbitrate between competing specialist thoughts for an angle.

    When multiple thoughts exist for the same angle, this function
    evaluates their evidence quality and produces a verdict thought
    that synthesises the best reasoning.

    Returns the verdict thought ID, or None if arbitration was not needed.
    """
    thoughts = corpus.get_thoughts_by_angle(angle)
    # Only arbitrate when there are multiple competing thoughts
    specialist_thoughts = [
        t for t in thoughts
        if t["strategy"] == "specialist_analysis"
    ]
    if len(specialist_thoughts) < 2:
        return None

    # Already have a verdict for this iteration?
    existing_verdicts = [
        t for t in thoughts
        if t["strategy"] == "arbitration_verdict"
        and t["iteration"] == iteration
    ]
    if existing_verdicts:
        return None

    # Build arbitration prompt
    thought_summaries = []
    for t in specialist_thoughts[-5:]:  # latest 5 specialist thoughts
        thought_summaries.append(
            f"  [thought #{t['id']}, iteration={t['iteration']}]: "
            f"{t['fact'][:800]}"
        )

    prompt = f"""\
You are a research arbitrator. Multiple specialist analysts have produced \
independent analyses for the same research angle. Your job is to evaluate \
their competing conclusions and produce a verdict.

ANGLE: {angle}

SPECIALIST ANALYSES:
{chr(10).join(thought_summaries)}

INSTRUCTIONS:
1. Identify where specialists AGREE — these are high-confidence conclusions
2. Identify where specialists DISAGREE — evaluate which side has stronger evidence
3. For each disagreement, explain WHY one conclusion is stronger
4. Produce a VERDICT that integrates the best reasoning from all specialists
5. Note any remaining uncertainty that needs further evidence

OUTPUT: A concise verdict (300-1000 chars) with clear conclusions and \
confidence levels. Cite specific thought IDs where relevant."""

    try:
        response = corpus._http_complete(
            prompt, caller=f"arbitrate_{angle[:30]}",
            max_tokens=1024,
        )
        if response and len(response.strip()) >= _MIN_THOUGHT_CHARS:
            # Use the most recent specialist thought as parent
            parent_id = specialist_thoughts[-1]["id"]
            tid = corpus.admit_thought(
                reasoning=response,
                parent_thought_id=parent_id,
                angle=angle,
                strategy="arbitration_verdict",
                iteration=iteration,
            )
            logger.info(
                "Arbitration verdict for '%s': thought #%d (%d chars)",
                angle, tid, len(response),
            )
            return tid
    except Exception:
        logger.warning(
            "Arbitration failed for angle '%s' (non-fatal)",
            angle, exc_info=True,
        )
    return None


# ═══════════════════════════════════════════════════════════════════════
# 4. Expansion-depth recursive thought splitting
# ═══════════════════════════════════════════════════════════════════════

def split_broad_thought(
    corpus,
    thought_id: int,
    iteration: int,
) -> list[int]:
    """Split a broad thought into focused child thoughts at deeper depth.

    If a thought contains multiple distinct sub-claims or angles, this
    decomposes it into separate child thoughts, each at
    ``expansion_depth + 1``.

    Respects ``MAX_THOUGHT_DEPTH`` to prevent infinite recursion.

    Returns list of child thought IDs created.
    """
    thought = corpus.get_latest_thought()
    if thought is None:
        return []

    # Find the specific thought by walking thoughts
    thoughts_for_check = corpus.get_thoughts_by_angle(
        corpus.conn.execute(
            "SELECT angle FROM conditions WHERE id = ?",
            [thought_id],
        ).fetchone()[0] if corpus.conn.execute(
            "SELECT angle FROM conditions WHERE id = ?",
            [thought_id],
        ).fetchone() else "",
    )
    target = None
    for t in thoughts_for_check:
        if t["id"] == thought_id:
            target = t
            break
    if target is None:
        return []

    current_depth = target.get("expansion_depth", 0) or 0
    if current_depth >= _MAX_THOUGHT_DEPTH:
        logger.debug(
            "Thought #%d already at max depth %d — not splitting",
            thought_id, _MAX_THOUGHT_DEPTH,
        )
        return []

    # Ask LLM to decompose the thought
    prompt = f"""\
You are a research decomposer. A specialist analyst produced a broad \
analysis that contains multiple distinct sub-claims or angles. Your job \
is to identify and separate these into focused, independent claims.

ORIGINAL ANALYSIS (thought #{thought_id}):
{target['fact'][:3000]}

INSTRUCTIONS:
1. Identify 2-4 distinct sub-claims, hypotheses, or angles in this analysis
2. Each sub-claim should be independently verifiable or analysable
3. Do NOT rephrase the same claim differently — only split genuinely \
   distinct claims
4. If the analysis is already focused on a single claim, respond with \
   exactly: NO_SPLIT_NEEDED

OUTPUT FORMAT:
CLAIM 1: [focused sub-claim with supporting evidence from the original]
CLAIM 2: [focused sub-claim with supporting evidence from the original]
..."""

    try:
        response = corpus._http_complete(
            prompt, caller=f"split_thought_{thought_id}",
            max_tokens=2048,
        )
        if not response or "NO_SPLIT_NEEDED" in response:
            return []

        # Parse claims from response
        claims = re.findall(
            r"CLAIM\s+\d+:\s*(.+?)(?=CLAIM\s+\d+:|$)",
            response,
            re.DOTALL,
        )
        if len(claims) < 2:
            return []

        child_ids: list[int] = []
        for claim in claims[:4]:  # max 4 children
            claim_text = claim.strip()
            if len(claim_text) < 50:
                continue
            cid = corpus.admit_thought(
                reasoning=claim_text,
                parent_thought_id=thought_id,
                angle=target["angle"],
                strategy="thought_split",
                iteration=iteration,
                expansion_depth=current_depth + 1,
            )
            child_ids.append(cid)

        if child_ids:
            logger.info(
                "Split thought #%d into %d children at depth %d: %s",
                thought_id, len(child_ids), current_depth + 1, child_ids,
            )
        return child_ids

    except Exception:
        logger.warning(
            "Thought splitting failed for #%d (non-fatal)",
            thought_id, exc_info=True,
        )
        return []


# ═══════════════════════════════════════════════════════════════════════
# 5. Swarm topology management (SwarmRouter)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AngleState:
    """Track the state of a single research angle in the swarm."""
    name: str
    thought_count: int = 0
    last_thought_chars: int = 0
    iterations_active: int = 0
    converged: bool = False


@dataclass
class SwarmRouter:
    """Manage swarm topology — which angles are active, converged, or new.

    The router tracks angle state across iterations and makes decisions
    about which angles need specialist attention.  It detects convergence
    (when an angle stops producing new insight) and prevents wasted LLM
    calls on exhausted angles.
    """
    angles: dict[str, AngleState] = field(default_factory=dict)
    max_specialists: int = _MAX_SPECIALISTS

    def update_from_corpus(self, corpus) -> None:
        """Refresh angle state from the corpus thought rows."""
        distinct_angles = corpus.get_distinct_thought_angles()
        for angle_name in distinct_angles:
            thoughts = corpus.get_thoughts_by_angle(angle_name)
            specialist_thoughts = [
                t for t in thoughts if t["strategy"] == "specialist_analysis"
            ]
            if angle_name not in self.angles:
                self.angles[angle_name] = AngleState(name=angle_name)
            state = self.angles[angle_name]
            new_count = len(specialist_thoughts)
            if new_count > state.thought_count:
                state.iterations_active += 1
                # Check for convergence: if latest thought is very short
                # compared to prior, the angle may be exhausted
                if specialist_thoughts:
                    state.last_thought_chars = len(
                        specialist_thoughts[0]["fact"]
                    )
                    if (
                        state.iterations_active >= 3
                        and state.last_thought_chars < _MIN_THOUGHT_CHARS
                    ):
                        state.converged = True
                        logger.info(
                            "Angle '%s' converged after %d iterations",
                            angle_name, state.iterations_active,
                        )
            state.thought_count = new_count

    def select_angles(self, requested_angles: list[str]) -> list[str]:
        """Filter requested angles, removing converged ones.

        Returns the angles that should receive specialist attention this
        iteration, respecting the max_specialists limit.
        """
        active: list[str] = []
        for angle in requested_angles:
            if angle in self.angles and self.angles[angle].converged:
                logger.debug("Skipping converged angle: %s", angle)
                continue
            active.append(angle)
            if len(active) >= self.max_specialists:
                break
        return active


# ═══════════════════════════════════════════════════════════════════════
# 6. Top-level orchestrator (called from condition_manager)
# ═══════════════════════════════════════════════════════════════════════

# Module-level router persists across iterations within a pipeline run
_swarm_router: SwarmRouter | None = None


def _get_router() -> SwarmRouter:
    """Get or create the module-level SwarmRouter."""
    global _swarm_router
    if _swarm_router is None:
        _swarm_router = SwarmRouter()
    return _swarm_router


def reset_swarm_router() -> None:
    """Reset the swarm router (called on pipeline cleanup)."""
    global _swarm_router
    _swarm_router = None


def run_swarm_cycle(state: dict, corpus) -> list[int]:
    """Execute one full swarm cycle: spawn → arbitrate → split.

    Called from ``maestro_condition_callback`` after the periodic synthesis.

    Returns list of all new thought IDs created during this cycle.
    """
    if not _SWARM_ENABLED:
        return []

    # Check minimum corpus size
    finding_count = corpus.conn.execute(
        "SELECT COUNT(*) FROM conditions "
        "WHERE row_type = 'finding' AND consider_for_use = TRUE"
    ).fetchone()[0]
    if finding_count < _MIN_FINDINGS_FOR_SWARM:
        logger.debug(
            "Swarm skipped: only %d findings (need %d)",
            finding_count, _MIN_FINDINGS_FOR_SWARM,
        )
        return []

    iteration = state.get("_corpus_iteration", 0)
    user_query = state.get("user_query", "")
    strategy = state.get("research_strategy", "")

    # Extract angles from thinker strategy
    angles = extract_angles(strategy)
    if not angles:
        logger.debug("No specialist angles found in thinker strategy")
        return []

    # Route through topology manager
    router = _get_router()
    router.update_from_corpus(corpus)
    active_angles = router.select_angles(angles)
    if not active_angles:
        logger.info("All requested angles converged — swarm cycle skipped")
        return []

    all_thought_ids: list[int] = []

    # Phase 1: Spawn parallel specialists
    specialist_ids: list[int] = []
    try:
        specialist_ids = spawn_specialist_thinkers(
            corpus, active_angles, user_query, iteration,
        )
        all_thought_ids.extend(specialist_ids)
    except Exception:
        logger.warning("Specialist spawning failed (non-fatal)", exc_info=True)

    # Phase 2: Arbitrate competing thoughts per angle
    for angle in active_angles:
        try:
            verdict_id = arbitrate_competing_thoughts(
                corpus, angle, iteration,
            )
            if verdict_id is not None:
                all_thought_ids.append(verdict_id)
        except Exception:
            logger.warning(
                "Arbitration for '%s' failed (non-fatal)",
                angle, exc_info=True,
            )

    # Phase 3: Split broad thoughts at shallow depth
    for tid in specialist_ids:
        try:
            child_ids = split_broad_thought(corpus, tid, iteration)
            all_thought_ids.extend(child_ids)
        except Exception:
            logger.warning(
                "Thought splitting for #%d failed (non-fatal)",
                tid, exc_info=True,
            )

    logger.info(
        "Swarm cycle complete: %d new thoughts from %d angles",
        len(all_thought_ids), len(active_angles),
    )

    # Update router state after this cycle
    router.update_from_corpus(corpus)

    return all_thought_ids
