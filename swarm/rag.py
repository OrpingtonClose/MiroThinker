# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm-internal RAG — targeted retrieval from the hive's persistent memory.

Between gossip rounds each bee can query the persistent store for findings
from OTHER bees that relate to its current analysis.  This replaces the
"dump all peer summaries" approach with targeted retrieval:

    Bee A's current focus: iron metabolism + erythropoiesis
    → RAG query extracts concepts: ["iron", "hepcidin", "erythropoietin"]
    → Store returns: Bee B's gossip_round_1 entry mentioning "hematocrit
      went from 42 to 49.6 after 8 weeks on tren-e"
    → Injected as "FROM THE HIVE" in Bee A's next gossip prompt

The RAG keeps context windows clean — a bee gets 3-5 relevant findings
from the store, not 30k chars of everything everyone ever wrote.

Keyword-based scoring — no embedding model needed.  With local Gemma 4
on H200, the wall-clock cost of these queries is negligible.
"""

from __future__ import annotations

import logging
import re
from collections import Counter

from swarm.lineage import LineageEntry

logger = logging.getLogger(__name__)

# Common stopwords to filter out during concept extraction
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "each", "few", "more", "most", "other", "some", "such", "no",
    "only", "own", "same", "than", "too", "very", "just", "because",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "you", "your", "he", "she", "his", "her",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "all", "any", "about", "up", "down", "here", "there", "also",
    "data", "findings", "analysis", "evidence", "research", "study",
    "suggests", "indicates", "shows", "found", "however", "therefore",
    "moreover", "furthermore", "additionally", "specifically",
    "particularly", "regarding", "related", "based", "according",
    "while", "although", "though", "whether", "given", "using",
    "including", "noted", "reported", "observed", "associated",
})

# Minimum term length to consider
_MIN_TERM_LEN = 3


def extract_concepts(text: str, top_k: int = 15) -> list[str]:
    """Extract key concepts from a text for RAG querying.

    Uses term frequency to identify the most distinctive words.
    Filters out stopwords and very short tokens.

    Args:
        text: Source text (typically a bee's current summary).
        top_k: Maximum number of concepts to return.

    Returns:
        List of key terms sorted by frequency (most frequent first).
    """
    if not text:
        return []

    # Tokenize: lowercase, split on non-alphanumeric
    tokens = re.findall(r"[a-z][a-z0-9]{2,}", text.lower())

    # Filter stopwords and very short tokens
    filtered = [
        t for t in tokens
        if t not in _STOPWORDS and len(t) >= _MIN_TERM_LEN
    ]

    if not filtered:
        return []

    # Count and return top-k
    counts = Counter(filtered)
    return [term for term, _ in counts.most_common(top_k)]


def score_relevance(concepts: list[str], content: str) -> float:
    """Score how relevant a piece of content is to a set of concepts.

    Simple keyword overlap scoring: count how many concepts appear in the
    content, weighted by whether the match is exact or partial.

    Args:
        concepts: Key terms to search for.
        content: Text to score against.

    Returns:
        Relevance score (0.0 = no match, higher = more relevant).
    """
    if not concepts or not content:
        return 0.0

    content_lower = content.lower()
    score = 0.0

    for concept in concepts:
        # Exact word boundary match gets full weight
        if re.search(rf"\b{re.escape(concept)}\b", content_lower):
            score += 1.0
        # Partial / substring match gets half weight
        elif concept in content_lower:
            score += 0.5

    return score


def query_hive(
    entries: list[LineageEntry],
    concepts: list[str],
    exclude_angle: str,
    top_k: int = 5,
    min_score: float = 1.5,
    max_chars_per_entry: int = 2000,
) -> list[str]:
    """Find the top-K most relevant entries from OTHER bees.

    Searches the accumulated lineage entries for content that matches
    the querying bee's key concepts, excluding entries from the same angle.

    Args:
        entries: All lineage entries emitted so far in this run.
        concepts: Key terms from the querying bee's current summary.
        exclude_angle: The querying bee's own angle (excluded from results).
        top_k: Maximum number of entries to return.
        min_score: Minimum relevance score to include.
        max_chars_per_entry: Truncate long entries to this length.

    Returns:
        List of formatted strings, each describing a relevant finding
        from another bee (includes angle and phase for attribution).
    """
    if not entries or not concepts:
        return []

    scored: list[tuple[float, LineageEntry]] = []

    for entry in entries:
        # Skip same-angle entries (the whole point is cross-angle discovery)
        if entry.angle == exclude_angle:
            continue
        # Skip empty or system entries
        if not entry.content or entry.content.startswith("["):
            continue
        # Skip corpus_analysis entries (metadata, not bee output)
        if entry.phase == "corpus_analysis":
            continue

        relevance = score_relevance(concepts, entry.content)
        if relevance >= min_score:
            scored.append((relevance, entry))

    # Sort by relevance descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[str] = []
    for relevance, entry in scored[:top_k]:
        content = entry.content
        if len(content) > max_chars_per_entry:
            content = content[:max_chars_per_entry] + "..."

        phase_label = entry.phase.replace("_", " ").title()
        angle_label = entry.angle or "cross-angle"
        results.append(
            f"[{angle_label} — {phase_label}] {content}"
        )

    if results:
        logger.debug(
            "hive_query concepts=<%s>, exclude=<%s>, results=<%d> | "
            "hive memory query complete",
            concepts[:5], exclude_angle, len(results),
        )

    return results
