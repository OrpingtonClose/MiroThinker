# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Agent-callable tools for querying and managing persistent knowledge.

These @tool-decorated functions give the agent direct access to the
KnowledgeStore so it can:
- Recall what it learned in previous sessions
- Manually store important insights mid-conversation
- Check knowledge coverage before starting a new research task
"""

from __future__ import annotations

import json
import logging

from strands import tool

from knowledge_store import Entity, Insight, get_knowledge_store

logger = logging.getLogger(__name__)


@tool
def recall_knowledge(
    query: str,
    max_results: int = 10,
    min_confidence: float = 0.0,
    topic: str = "",
) -> str:
    """Search past research knowledge for relevant facts and insights.

    Use this BEFORE starting a new research task to check what you
    already know. This saves time and avoids duplicate work.

    Args:
        query: Search query — describe what you're looking for.
        max_results: Maximum number of results to return.
        min_confidence: Minimum confidence threshold (0.0-1.0).
        topic: Optional topic filter (e.g., "GLP-1", "SEC filings").

    Returns:
        Formatted list of relevant past insights with metadata.
    """
    store = get_knowledge_store()
    insights = store.search_insights(
        query=query,
        limit=max_results,
        min_confidence=min_confidence,
        topic=topic,
    )

    if not insights:
        return f"No prior knowledge found for: {query}"

    lines = [f"Found {len(insights)} relevant prior insight(s):\n"]
    for i, ins in enumerate(insights, 1):
        fact = ins.get("fact", "")
        conf = ins.get("confidence", 0.0)
        src = ins.get("source_url", "")
        topic_tag = ins.get("topic", "")
        created = ins.get("created_at", "")[:10]  # date only
        access = ins.get("access_count", 0)

        line = f"{i}. {fact}"
        meta = []
        if topic_tag:
            meta.append(f"topic: {topic_tag}")
        if src:
            meta.append(f"source: {src}")
        meta.append(f"confidence: {conf:.2f}")
        meta.append(f"stored: {created}")
        if access > 1:
            meta.append(f"accessed {access}x")
        line += f"\n   [{', '.join(meta)}]"
        lines.append(line)

    return "\n".join(lines)


@tool
def store_insight(
    fact: str,
    source_url: str = "",
    topic: str = "",
    confidence: float = 0.7,
) -> str:
    """Store an important finding for future reference.

    Use this when you discover a key fact that would be valuable for
    future research queries. The insight persists across sessions.

    Args:
        fact: The factual claim to store (one clear sentence).
        source_url: URL where this fact was found.
        topic: Topic tag for categorization.
        confidence: How confident you are in this fact (0.0-1.0).

    Returns:
        Confirmation message with the stored insight ID.
    """
    store = get_knowledge_store()

    if store.has_similar_insight(fact):
        return f"Similar insight already exists, skipping: {fact[:80]}..."

    insight = Insight(
        fact=fact,
        source_url=source_url,
        topic=topic,
        confidence=max(0.0, min(1.0, confidence)),
    )
    insight_id = store.store_insight(insight)

    logger.info("id=<%d>, topic=<%s> | insight manually stored", insight_id, topic)
    return f"Stored insight #{insight_id}: {fact[:80]}..."


@tool
def recall_entities(query: str = "", limit: int = 20) -> str:
    """Look up named entities (people, compounds, organizations) from past research.

    Entities are automatically tracked across all conversations. Use
    this to find what you know about a specific entity or to see the
    most frequently researched entities.

    Args:
        query: Entity name to search for. Empty returns top entities by mention count.
        limit: Maximum results to return.

    Returns:
        Formatted list of entities with metadata.
    """
    store = get_knowledge_store()

    if query:
        entities = store.search_entities(query, limit=limit)
    else:
        entities = store.get_top_entities(limit=limit)

    if not entities:
        msg = f"No entities found matching: {query}" if query else "No entities tracked yet"
        return msg

    lines = [f"{'Matching' if query else 'Top'} entities ({len(entities)}):\n"]
    for e in entities:
        name = e.get("name", "")
        etype = e.get("entity_type", "")
        mentions = e.get("mention_count", 0)
        desc = e.get("description", "")

        line = f"- {name}"
        if etype:
            line += f" ({etype})"
        line += f" — {mentions} mention(s)"
        if desc:
            line += f"\n  {desc[:120]}"
        lines.append(line)

    return "\n".join(lines)


@tool
def knowledge_stats() -> str:
    """Get summary statistics about accumulated knowledge.

    Shows total insights, entities, top topics, average confidence,
    and most frequently accessed facts. Use this to understand what
    the knowledge base covers before starting research.

    Returns:
        Formatted knowledge statistics.
    """
    store = get_knowledge_store()
    stats = store.get_stats()

    lines = [
        "=== KNOWLEDGE BASE STATISTICS ===",
        f"Total insights: {stats['total_insights']}",
        f"Total entities: {stats['total_entities']}",
        f"Average confidence: {stats['avg_confidence']:.2f}",
    ]

    if stats["top_topics"]:
        lines.append("\nTop topics:")
        for t in stats["top_topics"]:
            lines.append(f"  - {t['topic']} ({t['count']} insights)")

    if stats["most_accessed"]:
        lines.append("\nMost accessed facts:")
        for m in stats["most_accessed"]:
            lines.append(f"  - [{m['access_count']}x] {m['fact']}")

    return "\n".join(lines)
