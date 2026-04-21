# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Angle coordinators — the middle tier of the hierarchy.

Each coordinator owns one top-level angle and manages N leaf workers.
It NEVER sees raw corpus data.  Its inputs are leaf worker summaries
only.  Its output is a single condensed angle summary that feeds the
bridge workers.

    Leaf A.0 ─┐
    Leaf A.1 ──┤── Coordinator A ──► Bridge Worker (receives condensed summary)
    Leaf A.2 ─┘

The coordinator's job:
1. Aggregate: Merge leaf summaries into a coherent angle narrative
2. Deduplicate: Remove overlap between leaf sub-clusters
3. Prioritize: Surface the highest-value findings and connections
4. Compress: Stay within the context budget for bridge workers
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _build_coordinator_prompt(
    date: str,
    angle: str,
    n_leaves: int,
    leaf_summaries_text: str,
    query: str,
    max_chars: int,
) -> str:
    """Build the coordinator aggregation prompt."""
    return (
        f"You are an Angle Coordinator responsible for synthesizing findings "
        f"from {n_leaves} leaf workers who each analysed a small sub-cluster "
        f"of research findings within your angle. Today is: {date}\n\n"
        f"YOUR ANGLE: {angle}\n\n"
        f"LEAF WORKER SUMMARIES:\n"
        f"{leaf_summaries_text}\n\n"
        f"USER QUERY: {query}\n\n"
        f"YOUR JOB:\n"
        f"1. MERGE these leaf summaries into ONE coherent narrative for your angle.\n"
        f"2. REMOVE redundancy — where leaves cover the same finding, keep "
        f"the version with the strongest evidence chain.\n"
        f"3. SURFACE cross-leaf connections — where one leaf's finding explains "
        f"or contradicts another leaf's finding within this angle.\n"
        f"4. PRESERVE exact numbers, source citations, and confidence levels.\n"
        f"5. PRIORITIZE: The highest-value findings and connections should be "
        f"stated first. This summary will be read by cross-angle bridge "
        f"workers who need to quickly grasp what YOUR angle has found.\n"
        f"6. Stay under {max_chars} characters.\n\n"
        f"Produce your consolidated angle summary:"
    )


async def coordinate_angle(
    angle: str,
    leaf_summaries: dict[str, str],
    query: str,
    max_chars: int,
    complete_fn,
) -> str:
    """Aggregate leaf worker summaries into a single angle summary.

    Args:
        angle: The angle name.
        leaf_summaries: Mapping of leaf_cluster_id -> summary text.
        query: The user's research query.
        max_chars: Maximum chars for the output summary.
        complete_fn: Async LLM completion callable.

    Returns:
        Consolidated angle summary string.
    """
    if not leaf_summaries:
        return ""

    # If only one leaf, its summary IS the angle summary
    if len(leaf_summaries) == 1:
        return next(iter(leaf_summaries.values()))[:max_chars]

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    leaf_text = ""
    for cluster_id, summary in leaf_summaries.items():
        leaf_text += f"\n--- Leaf {cluster_id} ---\n{summary[:max_chars]}\n"

    prompt = _build_coordinator_prompt(
        date=date,
        angle=angle,
        n_leaves=len(leaf_summaries),
        leaf_summaries_text=leaf_text,
        query=query,
        max_chars=max_chars,
    )

    try:
        result = await complete_fn(prompt)
        if result and len(result.strip()) > 50:
            return result[:max_chars]
    except Exception:
        logger.warning(
            "angle=<%s> | coordinator aggregation failed, falling back to concatenation",
            angle,
        )

    # Fallback: concatenate leaf summaries
    parts = [f"[{cid}] {s}" for cid, s in leaf_summaries.items()]
    return "\n\n".join(parts)[:max_chars]


async def coordinate_all_angles(
    angle_leaf_summaries: dict[str, dict[str, str]],
    query: str,
    max_chars: int,
    complete_fn,
    max_concurrency: int = 6,
) -> dict[str, str]:
    """Run all angle coordinators in parallel.

    Args:
        angle_leaf_summaries: Mapping of angle_name -> {cluster_id: summary}.
        query: The user's research query.
        max_chars: Maximum chars per angle summary.
        complete_fn: Async LLM completion callable.
        max_concurrency: Max parallel coordinator calls.

    Returns:
        Mapping of angle_name -> consolidated summary.
    """
    sem = asyncio.Semaphore(max_concurrency)
    results: dict[str, str] = {}

    async def _bounded(angle: str, leaf_sums: dict[str, str]) -> None:
        async with sem:
            results[angle] = await coordinate_angle(
                angle=angle,
                leaf_summaries=leaf_sums,
                query=query,
                max_chars=max_chars,
                complete_fn=complete_fn,
            )

    await asyncio.gather(*[
        _bounded(angle, leaf_sums)
        for angle, leaf_sums in angle_leaf_summaries.items()
    ])

    logger.info(
        "angles=<%d>, total_chars=<%d> | all coordinators complete",
        len(results), sum(len(s) for s in results.values()),
    )

    return results
