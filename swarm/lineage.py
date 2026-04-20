# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Lineage tracking for swarm synthesis — every phase emits structured entries.

The LineageStore protocol defines a minimal interface for recording what
happened at each phase of the swarm pipeline.  Entries are immutable
records with parent pointers, forming a DAG from final reports back to
raw corpus sections.

Implementations:

- ``InMemoryLineageStore``: lightweight fallback, keeps entries in a list.
  Useful for standalone swarm runs, testing, inspection after a run, and
  JSON serialization.  Pure in-memory — no external dependencies.
- ``apps.strands-agent.corpus.ConditionStore`` and
  ``apps.adk-agent.models.corpus_store.CorpusStore``: the DuckDB-backed
  research corpora now expose a native ``emit()`` that satisfies this
  protocol.  Passing one as ``SwarmConfig(lineage_store=...)`` unifies
  swarm provenance, research findings, thoughts, and synthesis reports
  into a single queryable store (see their ``get_by_phase()``,
  ``get_by_angle()``, and ``get_lineage_chain()`` helpers).  Prefer this
  for production — ``InMemoryLineageStore`` is intended as the
  dependency-free fallback.

Usage:
    store = InMemoryLineageStore()
    swarm = GossipSwarm(complete=fn, config=SwarmConfig(lineage_store=store))
    result = await swarm.synthesize(corpus=..., query=...)
    for entry in store.entries:
        print(entry.phase, entry.angle, len(entry.content))
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LineageEntry:
    """A single immutable record in the lineage DAG.

    Attributes:
        entry_id: Unique identifier (auto-generated if not provided).
        phase: Pipeline phase that produced this entry.
            One of: corpus_analysis, worker_synthesis, gossip_round_N,
            serendipity, queen_merge, knowledge_report.
        angle: Worker angle name (empty for non-worker phases).
        content: The text produced at this phase.
        parent_ids: IDs of entries this one was derived from.
        metadata: Arbitrary key-value metadata (round number, info gain, etc.).
        timestamp: Unix timestamp of creation.
    """

    entry_id: str
    phase: str
    angle: str = ""
    content: str = ""
    parent_ids: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@runtime_checkable
class LineageStore(Protocol):
    """Protocol for lineage tracking backends.

    Any object implementing ``emit()`` can be used as a lineage store.
    The swarm engine calls ``emit()`` at each phase boundary.
    """

    def emit(self, entry: LineageEntry) -> None:
        """Record a lineage entry.

        Args:
            entry: The lineage entry to store.
        """
        ...


class InMemoryLineageStore:
    """Simple in-memory lineage store — keeps all entries in a list.

    Entries can be inspected after a run, serialized to JSON, or fed
    into an external store (e.g. ADK CorpusStore).

    Thread-safe: emit() appends to a list (GIL-protected for simple appends).
    """

    def __init__(self) -> None:
        self.entries: list[LineageEntry] = []

    def emit(self, entry: LineageEntry) -> None:
        """Record a lineage entry.

        Args:
            entry: The lineage entry to store.
        """
        self.entries.append(entry)

    def get_by_phase(self, phase: str) -> list[LineageEntry]:
        """Filter entries by phase name.

        Args:
            phase: Phase name to filter by (exact match or prefix match
                for gossip rounds, e.g. "gossip_round" matches all rounds).

        Returns:
            List of matching entries.
        """
        return [e for e in self.entries if e.phase == phase or e.phase.startswith(phase)]

    def get_by_angle(self, angle: str) -> list[LineageEntry]:
        """Filter entries by worker angle.

        Args:
            angle: Angle name to filter by.

        Returns:
            List of matching entries.
        """
        return [e for e in self.entries if e.angle == angle]

    def get_lineage_chain(self, entry_id: str) -> list[LineageEntry]:
        """Walk the parent chain from an entry back to its roots.

        Args:
            entry_id: Starting entry ID.

        Returns:
            List of entries from the given entry back to root(s),
            in reverse chronological order.
        """
        index = {e.entry_id: e for e in self.entries}
        chain: list[LineageEntry] = []
        visited: set[str] = set()
        queue = [entry_id]

        while queue:
            eid = queue.pop(0)
            if eid in visited or eid not in index:
                continue
            visited.add(eid)
            entry = index[eid]
            chain.append(entry)
            queue.extend(entry.parent_ids)

        return chain

    def to_dicts(self) -> list[dict]:
        """Serialize all entries to a list of plain dicts.

        Returns:
            List of dicts suitable for JSON serialization.
        """
        return [
            {
                "entry_id": e.entry_id,
                "phase": e.phase,
                "angle": e.angle,
                "content_chars": len(e.content),
                "parent_ids": list(e.parent_ids),
                "metadata": dict(e.metadata),
                "timestamp": e.timestamp,
            }
            for e in self.entries
        ]
