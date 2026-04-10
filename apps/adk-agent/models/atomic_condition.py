# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
AtomicCondition — a single compressed research finding (Atom of Thought).

Each atom is an indivisible unit of research output carrying:
- The factual claim itself
- Source provenance (URL)
- Confidence score (0.0–1.0)
- Verification status lifecycle: "" → "speculative" → "verified" | "fabricated"
- Expansion metadata (parent lineage, strategy that produced it, depth)
- Angle tracking (which research angle produced it)

These atoms flow through the research loop:
  1. Researcher produces raw findings
  2. Condition manager decomposes them into AtomicConditions
  3. Stored in DuckDB corpus (structured, queryable)
  4. Thinker reads the structured corpus and reasons about gaps
  5. Loop repeats until thinker signals EVIDENCE_SUFFICIENT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class AtomicCondition:
    """A single compressed research finding (Atom of Thought)."""

    fact: str
    source_url: str = ""
    confidence: float = 0.5
    verification_status: str = ""  # "", "speculative", "verified", "fabricated"
    angle: str = ""
    parent_id: int | None = None
    strategy: str = ""  # which expansion strategy produced this
    expansion_depth: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Single-table architecture fields
    row_type: str = "finding"  # 'finding' | 'similarity' | 'contradiction' | 'raw' | 'synthesis'
    related_id: int | None = None  # second parent for relationship rows
    consider_for_use: bool = True  # universal exclusion flag

    def __post_init__(self) -> None:
        # Clamp confidence to [0.0, 1.0]
        self.confidence = max(0.0, min(1.0, self.confidence))
