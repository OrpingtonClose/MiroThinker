# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Mesh + circular gossip topology for cross-angle bridge workers.

Bridge workers sit above angle coordinators.  Each receives ONE
condensed summary per top-level angle (still tiny context).  They
exchange CollisionStatements — the ONLY payload that travels the mesh.

Communication flow per round:
1. Each bridge worker reads its own angle summary + received collisions
2. Produces new CollisionStatements (structured JSON output)
3. Circular pass: each worker receives the top collisions from the
   previous worker in ring order
4. Targeted questions are routed to the specific peer that can answer

The mesh + circular hybrid maximises serendipity surface area while
keeping per-agent context microscopic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from swarm.topologies.collision import CollisionStatement, parse_collisions_from_response

logger = logging.getLogger(__name__)


# ── Round-specific bridge prompts ─────────────────────────────────────

BRIDGE_ROUND_PROMPTS: dict[int, str] = {
    1: (
        "ROUND 1 — SERENDIPITY HUNTING (Connection Discovery):\n"
        "You are a narrow-domain specialist with only your own angle summary. "
        "Your peers are specialists in adjacent domains.\n"
        "For each key point in a peer's collision or summary, explicitly ask:\n"
        "  'Does this resolve an anomaly, explain an unexplained pattern, or "
        "create a non-obvious causal link in my own data?'\n"
        "Use abductive reasoning: 'What surprising unification or analogy "
        "emerges here?'\n"
        "For each connection you find, assign a serendipity_score (1-5):\n"
        "  1=trivial overlap, 2=expected correlation, 3=interesting, "
        "4=surprising bridge, 5=paradigm-shifting\n"
        "Also output ONE targeted_question you wish you could ask a specific peer."
    ),
    2: (
        "ROUND 2 — CAUSAL DEPTH & SECOND-ORDER EFFECTS:\n"
        "Build on the connections from Round 1. For each promising bridge:\n"
        "  (a) Trace the exact causal chain using specific evidence\n"
        "  (b) Ask 'What does this predict?' and 'What second-order or "
        "compounding effect appears when combined with other peers?'\n"
        "  (c) Hunt for contradictions, hidden analogies, and emergent hypotheses\n"
        "Use explicit chain-of-thought + self-critique before finalizing each insight.\n"
        "Upgrade serendipity_scores if deeper analysis reveals the connection "
        "is more surprising than initially assessed."
    ),
    3: (
        "ROUND 3 — GAPS + FINAL SYNTHESIS:\n"
        "List precise research gaps as questions aimed at specific peers or angles.\n"
        "Produce your final collision statements — the highest-value cross-domain "
        "insights you've found across all rounds.\n"
        "Each collision must carry: exact causal chain, prediction, and "
        "serendipity_score. Preserve all exact numbers and provenance.\n"
        "These are the collision statements that will feed the Serendipity Panel "
        "and Queen — make them count."
    ),
}


def _build_bridge_prompt(
    worker_name: str,
    angle_summary: str,
    received_collisions: list[CollisionStatement],
    round_num: int,
    peer_summaries: dict[str, str] | None = None,
    targeted_answers: list[str] | None = None,
) -> str:
    """Build the prompt for a bridge worker round.

    Context is always tiny: own angle summary + received collision
    statements + optional targeted answers from peers.
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    round_prompt = BRIDGE_ROUND_PROMPTS.get(round_num, "")

    parts = [
        f"You are {worker_name}, a cross-angle bridge specialist in a "
        f"multi-agent research conversation. Today is: {date}\n",
        f"YOUR ANGLE SUMMARY (this is your entire view of your domain):\n"
        f"{angle_summary}\n",
    ]

    # Add peer angle summaries (Round 1 only — after that, collisions carry the info)
    if peer_summaries and round_num <= 1:
        parts.append("PEER ANGLE SUMMARIES (condensed views of other domains):\n")
        for peer_name, summary in peer_summaries.items():
            parts.append(f"--- {peer_name} ---\n{summary[:2000]}\n")
        parts.append("")

    # Add received collisions from prior rounds
    if received_collisions:
        parts.append(
            f"COLLISION STATEMENTS FROM PEERS ({len(received_collisions)} received):\n"
        )
        # Show top collisions by serendipity score, cap at 15
        sorted_collisions = sorted(
            received_collisions,
            key=lambda c: c.serendipity_score,
            reverse=True,
        )[:15]
        for cs in sorted_collisions:
            parts.append(cs.to_prompt_line())
            parts.append("")

    # Add targeted answers if any peers answered our questions
    if targeted_answers:
        parts.append("ANSWERS TO YOUR TARGETED QUESTIONS:\n")
        for answer in targeted_answers:
            parts.append(f"  {answer}\n")
        parts.append("")

    # Round-specific instructions
    parts.append(f"\n{round_prompt}\n")

    # Output format
    parts.append(
        "\nOUTPUT FORMAT: Respond with a JSON array of collision statements. "
        "Each object must have: from_worker, peer_worker, collision, "
        "causal_link, serendipity_score (1-5). Optionally: prediction, "
        "targeted_question.\n"
        'Example: [{"from_worker": "' + worker_name + '", "peer_worker": '
        '"Molecular", "collision": "Iron chelation pattern explains '
        'hematocrit anomaly", "causal_link": "Tren increases ferritin → '
        'iron sequestration → altered erythropoiesis", "serendipity_score": 4, '
        '"prediction": "Elevated RBC under chronic tren + iron supplementation", '
        '"targeted_question": "Molecular worker: any data on AR-ferritin '
        'interaction?"}]\n\n'
        "Respond with ONLY the JSON array:"
    )

    return "\n".join(parts)


class BridgeWorker:
    """A cross-angle bridge worker in the mesh topology.

    Context = one condensed summary per angle (own angle in full,
    peers as summaries).  Never sees raw corpus data.

    Attributes:
        name: Worker identifier (e.g. ``"Bridge-Molecular"``).
        angle: The angle this worker represents.
        angle_summary: Condensed summary from the angle coordinator.
        received_collisions: CollisionStatements received from peers.
        produced_collisions: CollisionStatements this worker has produced.
    """

    def __init__(
        self,
        name: str,
        angle: str,
        angle_summary: str,
    ) -> None:
        self.name = name
        self.angle = angle
        self.angle_summary = angle_summary
        self.received_collisions: list[CollisionStatement] = []
        self.produced_collisions: list[CollisionStatement] = []

    def receive(self, collisions: list[CollisionStatement]) -> None:
        """Receive collision statements from the mesh."""
        self.received_collisions.extend(collisions)

    async def run_round(
        self,
        round_num: int,
        complete_fn,
        peer_summaries: dict[str, str] | None = None,
        targeted_answers: list[str] | None = None,
    ) -> list[CollisionStatement]:
        """Execute one conversation round.

        Args:
            round_num: 1-indexed round number.
            complete_fn: Async LLM completion callable.
            peer_summaries: Condensed summaries from other angles
                (used in Round 1 for initial context).
            targeted_answers: Responses to this worker's targeted
                questions from prior rounds.

        Returns:
            New CollisionStatements produced this round.
        """
        prompt = _build_bridge_prompt(
            worker_name=self.name,
            angle_summary=self.angle_summary,
            received_collisions=self.received_collisions,
            round_num=round_num,
            peer_summaries=peer_summaries,
            targeted_answers=targeted_answers,
        )

        response = await complete_fn(prompt)

        new_collisions = parse_collisions_from_response(
            response, self.name, round_num,
        )
        self.produced_collisions.extend(new_collisions)

        logger.info(
            "bridge=<%s>, round=<%d>, produced=<%d>, total_received=<%d> | "
            "bridge round complete",
            self.name, round_num, len(new_collisions),
            len(self.received_collisions),
        )

        return new_collisions

    @property
    def top_collisions(self) -> list[CollisionStatement]:
        """Return collisions sorted by serendipity score (descending)."""
        return sorted(
            self.produced_collisions,
            key=lambda c: c.serendipity_score,
            reverse=True,
        )


async def run_mesh_rounds(
    bridge_workers: list[BridgeWorker],
    complete_fn,
    num_rounds: int = 3,
    max_concurrency: int = 6,
    peer_summaries: dict[str, str] | None = None,
    on_event=None,
) -> list[CollisionStatement]:
    """Run mesh + circular gossip rounds across bridge workers.

    Each round:
    1. All bridge workers run in parallel (bounded by max_concurrency)
    2. Circular pass: each worker receives top collisions from the
       previous worker in ring order
    3. Targeted questions are collected and routed

    Args:
        bridge_workers: List of BridgeWorker instances.
        complete_fn: Async LLM completion callable.
        num_rounds: Number of conversation rounds.
        max_concurrency: Max parallel LLM calls.
        peer_summaries: Initial angle summaries for Round 1 context.
        on_event: Optional async callback for progress events.

    Returns:
        All collision statements produced across all rounds,
        sorted by serendipity score descending.
    """
    all_collisions: list[CollisionStatement] = []
    sem = asyncio.Semaphore(max_concurrency)

    async def _emit(event: dict) -> None:
        if on_event:
            try:
                await on_event(event)
            except Exception:
                pass

    for round_num in range(1, num_rounds + 1):
        round_collisions: list[CollisionStatement] = []
        round_failures = 0

        # Collect targeted questions from prior round to route as answers
        targeted_answers_by_worker: dict[str, list[str]] = {
            w.name: [] for w in bridge_workers
        }
        if round_num > 1:
            for w in bridge_workers:
                for cs in w.produced_collisions:
                    if cs.targeted_question and cs.round == round_num - 1:
                        # Route to the peer mentioned in the question
                        for target in bridge_workers:
                            if target.name != w.name:
                                # Simple routing: send to all peers
                                # (could be smarter with name matching)
                                targeted_answers_by_worker[target.name].append(
                                    f"[Question from {w.name}]: {cs.targeted_question}"
                                )

        async def _bounded_round(worker: BridgeWorker) -> list[CollisionStatement]:
            nonlocal round_failures
            async with sem:
                try:
                    return await worker.run_round(
                        round_num=round_num,
                        complete_fn=complete_fn,
                        peer_summaries=peer_summaries if round_num == 1 else None,
                        targeted_answers=targeted_answers_by_worker.get(worker.name),
                    )
                except Exception:
                    round_failures += 1
                    logger.warning(
                        "bridge=<%s>, round=<%d> | bridge round failed",
                        worker.name, round_num,
                    )
                    return []

        # Run all bridge workers in parallel
        results = await asyncio.gather(
            *[_bounded_round(w) for w in bridge_workers]
        )

        for worker_collisions in results:
            round_collisions.extend(worker_collisions)

        all_collisions.extend(round_collisions)

        # Circular pass: each worker receives top 3 collisions from
        # the previous worker in ring order
        for i, worker in enumerate(bridge_workers):
            prev_worker = bridge_workers[(i - 1) % len(bridge_workers)]
            top_from_prev = [
                c for c in prev_worker.produced_collisions
                if c.round == round_num
            ]
            top_from_prev.sort(key=lambda c: c.serendipity_score, reverse=True)
            worker.receive(top_from_prev[:3])

        logger.info(
            "mesh_round=<%d>, collisions=<%d>, failures=<%d> | mesh round complete",
            round_num, len(round_collisions), round_failures,
        )

        await _emit({
            "type": "mesh_round",
            "round": round_num,
            "total_rounds": num_rounds,
            "collisions_produced": len(round_collisions),
            "failures": round_failures,
        })

        # Early stop if no new collisions
        if len(round_collisions) == 0 and round_num >= 2:
            logger.info(
                "mesh_round=<%d> | no new collisions, early convergence",
                round_num,
            )
            break

    # Sort all collisions by serendipity score
    all_collisions.sort(key=lambda c: c.serendipity_score, reverse=True)
    return all_collisions
