# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Lightweight knowledge graph for cross-query entity linking.

Stores entities (findings, sources, topics) and relationships between
them in a JSONL-backed graph.  This allows the agent to:

  - Track which sources corroborate each other
  - Link findings across different sub-queries
  - Identify knowledge gaps (entities with few connections)
  - Build a structured evidence map for synthesis

The graph is intentionally simple — no external DB, no complex query
language.  It's a Python dict + JSONL persistence, exposed as ADK
FunctionTools so the LLM can build the graph during research.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# ── Storage ──────────────────────────────────────────────────────────

_GRAPH_DIR = Path(os.environ.get(
    "FINDINGS_DIR", os.path.join(os.path.expanduser("~"), ".mirothinker")
))
_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# In-memory graph — persisted to JSONL on every mutation
_entities: Dict[str, dict] = {}  # id → {type, name, props, ts}
_edges: List[dict] = []          # [{src, tgt, rel, weight, ts}]

_graph_file: Path = _GRAPH_DIR / "knowledge_graph.jsonl"


def _persist() -> None:
    """Write the full graph to JSONL (entities then edges)."""
    with open(_graph_file, "w") as f:
        for eid, ent in _entities.items():
            f.write(json.dumps({"_t": "entity", "id": eid, **ent}, ensure_ascii=False) + "\n")
        for edge in _edges:
            f.write(json.dumps({"_t": "edge", **edge}, ensure_ascii=False) + "\n")


def load_graph() -> None:
    """Load graph from JSONL file (called on startup / resume)."""
    global _entities, _edges
    _entities = {}
    _edges = []
    if not _graph_file.exists():
        return
    for line in _graph_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_t") == "entity":
            eid = obj.pop("id")
            obj.pop("_t")
            _entities[eid] = obj
        elif obj.get("_t") == "edge":
            obj.pop("_t")
            _edges.append(obj)


def clear_graph() -> None:
    """Clear the in-memory graph and delete the JSONL file."""
    global _entities, _edges
    _entities = {}
    _edges = []
    if _graph_file.exists():
        _graph_file.unlink()


# ── FunctionTool implementations ─────────────────────────────────────


async def add_entity(
    entity_id: str,
    entity_type: str,
    name: str,
    properties: str = "{}",
) -> str:
    """Add or update an entity in the knowledge graph.

    Use this to register findings, sources, topics, or any named concept
    you discover during research.  Entities are the nodes of the graph.

    Args:
        entity_id: Unique identifier (e.g. "model:deepseek-r1", "source:localaimaster.com").
        entity_type: Category — "model", "source", "topic", "vendor", "person", etc.
        name: Human-readable name.
        properties: JSON string of additional properties (e.g. '{"url": "...", "rating": 8}').

    Returns:
        Confirmation message.
    """
    try:
        props = json.loads(properties) if properties else {}
    except json.JSONDecodeError:
        props = {"raw": properties}

    _entities[entity_id] = {
        "type": entity_type,
        "name": name,
        "props": props,
        "ts": time.time(),
    }
    _persist()
    logger.info("KG: added entity %s (%s)", entity_id, entity_type)
    return f"Entity added: {entity_id} ({entity_type}: {name})"


async def add_edge(
    source_id: str,
    target_id: str,
    relationship: str,
    weight: float = 1.0,
) -> str:
    """Add a relationship (edge) between two entities in the knowledge graph.

    Use this to record that two entities are related — e.g. a source
    corroborates a finding, a model is published by a vendor, etc.

    Args:
        source_id: Entity ID of the source node.
        target_id: Entity ID of the target node.
        relationship: Type of relationship (e.g. "corroborates", "published_by",
            "contradicts", "mentions", "alternative_to", "hosted_on").
        weight: Strength of the relationship (0.0-1.0, default 1.0).

    Returns:
        Confirmation message.
    """
    edge = {
        "src": source_id,
        "tgt": target_id,
        "rel": relationship,
        "weight": min(max(weight, 0.0), 1.0),
        "ts": time.time(),
    }
    _edges.append(edge)
    _persist()
    logger.info("KG: added edge %s -[%s]-> %s", source_id, relationship, target_id)
    return f"Edge added: {source_id} -[{relationship}]-> {target_id}"


async def query_graph(
    entity_id: str = "",
    entity_type: str = "",
    relationship: str = "",
) -> str:
    """Query the knowledge graph for entities and their connections.

    Returns matching entities with their edges.  Use this to find
    corroborating sources, identify knowledge gaps, or map the
    evidence structure before writing a synthesis.

    Args:
        entity_id: If set, return this entity and all its connections.
        entity_type: If set, return all entities of this type.
        relationship: If set, return all edges of this type.

    Returns:
        JSON object with matching entities and edges.
    """
    result_entities: Dict[str, dict] = {}
    result_edges: List[dict] = []

    if entity_id:
        if entity_id in _entities:
            result_entities[entity_id] = _entities[entity_id]
        for edge in _edges:
            if edge["src"] == entity_id or edge["tgt"] == entity_id:
                result_edges.append(edge)
                # Include connected entities
                other = edge["tgt"] if edge["src"] == entity_id else edge["src"]
                if other in _entities:
                    result_entities[other] = _entities[other]
    elif entity_type:
        for eid, ent in _entities.items():
            if ent["type"] == entity_type:
                result_entities[eid] = ent
        # Include edges between matched entities
        matched_ids = set(result_entities.keys())
        for edge in _edges:
            if edge["src"] in matched_ids or edge["tgt"] in matched_ids:
                result_edges.append(edge)
    elif relationship:
        for edge in _edges:
            if edge["rel"] == relationship:
                result_edges.append(edge)
                for eid in (edge["src"], edge["tgt"]):
                    if eid in _entities:
                        result_entities[eid] = _entities[eid]
    else:
        # Return full graph summary
        result_entities = dict(_entities)
        result_edges = list(_edges)

    output = {
        "entity_count": len(result_entities),
        "edge_count": len(result_edges),
        "entities": {eid: {**ent, "props": ent.get("props", {})} for eid, ent in result_entities.items()},
        "edges": result_edges,
    }
    return json.dumps(output, ensure_ascii=False)


async def find_gaps() -> str:
    """Identify entities with few or no connections (knowledge gaps).

    Returns entities sorted by connection count (ascending) so the
    agent can prioritise further research on poorly-connected nodes.

    Returns:
        JSON array of {entity_id, name, type, connection_count} sorted
        by connection_count ascending.
    """
    conn_count: Dict[str, int] = {eid: 0 for eid in _entities}
    for edge in _edges:
        if edge["src"] in conn_count:
            conn_count[edge["src"]] += 1
        if edge["tgt"] in conn_count:
            conn_count[edge["tgt"]] += 1

    gaps = []
    for eid, count in sorted(conn_count.items(), key=lambda x: x[1]):
        ent = _entities[eid]
        gaps.append({
            "entity_id": eid,
            "name": ent["name"],
            "type": ent["type"],
            "connection_count": count,
        })

    return json.dumps(gaps, ensure_ascii=False)


# ── Public FunctionTool instances ─────────────────────────────────────

add_entity_tool = FunctionTool(add_entity)
add_edge_tool = FunctionTool(add_edge)
query_graph_tool = FunctionTool(query_graph)
find_gaps_tool = FunctionTool(find_gaps)

KNOWLEDGE_GRAPH_TOOLS = [
    add_entity_tool,
    add_edge_tool,
    query_graph_tool,
    find_gaps_tool,
]
