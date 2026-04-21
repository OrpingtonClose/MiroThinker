# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Hierarchical sub-clustering for large corpus decomposition.

Recursively splits angle findings into leaf clusters until every
cluster is below the size threshold.  This guarantees every leaf
worker operates on a tiny, manageable slice regardless of total
corpus size.

    Corpus (10k findings)
        → Angle A (2k findings)
            → Leaf A.1 (20 findings)
            → Leaf A.2 (18 findings)
            → ...
        → Angle B (1.5k findings)
            → Leaf B.1 (22 findings)
            → ...

Each leaf cluster becomes one leaf worker.  Angle coordinators
see only the leaf summaries, never the raw data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum findings per leaf cluster.  Tuned so that the raw text of
# one leaf fits comfortably in the context window of even a small model
# (~8-12k chars for 15-25 findings of ~400 chars each).
DEFAULT_MAX_LEAF_SIZE = 25
DEFAULT_MIN_LEAF_SIZE = 5


@dataclass
class LeafCluster:
    """A leaf-level cluster of findings assigned to one leaf worker.

    Attributes:
        cluster_id: Unique identifier (e.g. ``"A.1"`` for angle A, sub-cluster 1).
        angle: Parent angle name.
        findings: List of ``(finding_key, finding_text)`` tuples.
        raw_content: Pre-joined text of all findings (for worker prompt).
    """

    cluster_id: str
    angle: str
    findings: list[tuple[str, str]] = field(default_factory=list)
    raw_content: str = ""

    def __post_init__(self) -> None:
        if not self.raw_content and self.findings:
            self.raw_content = "\n".join(text for _, text in self.findings)

    @property
    def size(self) -> int:
        """Number of findings in this cluster."""
        return len(self.findings)

    @property
    def char_count(self) -> int:
        """Total character count of raw content."""
        return len(self.raw_content)


def cluster_findings(
    findings: list[tuple[str, str]],
    angle: str,
    angle_index: int,
    max_leaf_size: int = DEFAULT_MAX_LEAF_SIZE,
    min_leaf_size: int = DEFAULT_MIN_LEAF_SIZE,
) -> list[LeafCluster]:
    """Split an angle's findings into leaf clusters.

    Uses keyword-based grouping first (group findings that share
    significant terms), then falls back to even splitting if grouping
    doesn't produce enough clusters.

    Args:
        findings: List of ``(key, text)`` tuples for this angle.
        angle: Angle name (for cluster_id prefix).
        angle_index: Numeric index of this angle (for cluster_id).
        max_leaf_size: Maximum findings per leaf cluster.
        min_leaf_size: Minimum findings per leaf cluster.

    Returns:
        List of LeafCluster objects, each with at most ``max_leaf_size``
        findings.
    """
    n = len(findings)
    prefix = f"A{angle_index}"

    # If already small enough, return as single cluster
    if n <= max_leaf_size:
        return [LeafCluster(
            cluster_id=f"{prefix}.0",
            angle=angle,
            findings=list(findings),
        )]

    # Try keyword-based grouping
    clusters = _keyword_cluster(findings, max_leaf_size, min_leaf_size)

    if len(clusters) < 2:
        # Keyword clustering didn't produce meaningful groups — even split
        clusters = _even_split(findings, max_leaf_size)

    result = []
    for i, cluster_findings_list in enumerate(clusters):
        result.append(LeafCluster(
            cluster_id=f"{prefix}.{i}",
            angle=angle,
            findings=cluster_findings_list,
        ))

    logger.info(
        "angle=<%s>, findings=<%d>, leaf_clusters=<%d> | hierarchical clustering complete",
        angle, n, len(result),
    )
    return result


def _keyword_cluster(
    findings: list[tuple[str, str]],
    max_size: int,
    min_size: int,
) -> list[list[tuple[str, str]]]:
    """Group findings by shared significant terms.

    Extracts important terms from each finding (nouns, proper nouns,
    technical terms), then groups findings that share the most terms.
    Simple but effective for research corpora where domain terminology
    naturally clusters findings.
    """
    # Extract significant terms per finding
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "and", "but", "or", "nor", "not", "no",
        "so", "if", "than", "that", "this", "these", "those", "it",
        "its", "they", "them", "their", "we", "our", "you", "your",
        "he", "she", "him", "her", "his", "who", "which", "what",
        "when", "where", "how", "all", "each", "every", "both",
        "few", "more", "most", "some", "any", "other", "such",
        "only", "also", "very", "just", "about", "up", "out", "then",
        "there", "here", "over", "under", "again", "further", "once",
    }

    finding_terms: list[set[str]] = []
    for _, text in findings:
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        terms = {w for w in words if w not in stop_words and len(w) >= 4}
        finding_terms.append(terms)

    # Find the most common terms to use as cluster anchors
    term_freq: dict[str, int] = {}
    for terms in finding_terms:
        for t in terms:
            term_freq[t] = term_freq.get(t, 0) + 1

    # Pick anchor terms: frequent enough to form clusters, not so frequent
    # they appear in everything
    n = len(findings)
    anchors = [
        t for t, freq in sorted(term_freq.items(), key=lambda x: -x[1])
        if min_size <= freq <= max_size and freq < n * 0.8
    ][:20]  # max 20 anchors

    if not anchors:
        return []

    # Assign each finding to the best matching anchor
    buckets: dict[str, list[tuple[str, str]]] = {a: [] for a in anchors}
    assigned: set[int] = set()

    for idx, (key, text) in enumerate(findings):
        terms = finding_terms[idx]
        best_anchor = ""
        best_overlap = 0
        for anchor in anchors:
            if anchor in terms:
                overlap = len(terms & {anchor})
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_anchor = anchor
        if best_anchor:
            buckets[best_anchor].append((key, text))
            assigned.add(idx)

    # Collect non-empty buckets
    clusters = [items for items in buckets.values() if items]

    # Add unassigned findings to the smallest cluster
    unassigned = [(key, text) for idx, (key, text) in enumerate(findings)
                  if idx not in assigned]
    if unassigned and clusters:
        smallest = min(clusters, key=len)
        smallest.extend(unassigned)
    elif unassigned:
        clusters.append(unassigned)

    # Split any cluster that's still too large
    final: list[list[tuple[str, str]]] = []
    for cluster in clusters:
        if len(cluster) > max_size:
            final.extend(_even_split(cluster, max_size))
        elif len(cluster) >= 2:  # drop singletons, merge later
            final.append(cluster)

    return final if len(final) >= 2 else []


def _even_split(
    findings: list[tuple[str, str]],
    max_size: int,
) -> list[list[tuple[str, str]]]:
    """Split findings into chunks of at most max_size."""
    return [
        findings[i:i + max_size]
        for i in range(0, len(findings), max_size)
    ]
