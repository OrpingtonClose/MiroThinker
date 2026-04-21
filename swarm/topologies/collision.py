# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""CollisionStatement — the communication primitive for the mesh topology.

A CollisionStatement is the ONLY payload that travels between bridge
workers.  It captures a single cross-domain insight: where one
specialist's data explains an anomaly in another specialist's domain.

Design constraints:
- Tiny: ~500 chars max per statement (keeps mesh bandwidth microscopic)
- Structured: machine-parseable for routing and scoring
- Self-contained: carries enough provenance to be useful without the
  original raw data

The serendipity_score (1-5) determines propagation priority:
  1 = trivial overlap (same fact restated)
  2 = expected correlation (known relationship)
  3 = interesting connection (non-obvious but plausible)
  4 = surprising bridge (cross-domain, requires expertise to see)
  5 = paradigm-shifting (reframes understanding of both domains)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass
class CollisionStatement:
    """A single cross-domain insight discovered during mesh conversation.

    Attributes:
        from_worker: Name/angle of the worker who produced this statement.
        peer_worker: Name/angle of the peer whose data triggered it.
        collision: The insight itself — what connection was found.
        causal_link: The mechanistic chain connecting the two domains.
        serendipity_score: 1-5 rating of how surprising the connection is.
        prediction: What this connection predicts should also be true.
        targeted_question: A question this worker wants to ask a specific
            peer (routed dynamically by the orchestrator).
        round: Which conversation round produced this statement.
    """

    from_worker: str
    peer_worker: str
    collision: str
    causal_link: str
    serendipity_score: int = 3
    prediction: str | None = None
    targeted_question: str | None = None
    round: int = 0

    def to_json(self) -> str:
        """Serialize to compact JSON."""
        return json.dumps(asdict(self), indent=2)

    def to_prompt_line(self) -> str:
        """Format for injection into a worker prompt."""
        score_label = {1: "trivial", 2: "expected", 3: "interesting",
                       4: "surprising", 5: "paradigm-shifting"}
        label = score_label.get(self.serendipity_score, "unknown")
        line = (
            f"[{self.from_worker} → {self.peer_worker}] "
            f"({label}, score={self.serendipity_score}) "
            f"{self.collision}"
        )
        if self.causal_link:
            line += f"\n  Causal chain: {self.causal_link}"
        if self.prediction:
            line += f"\n  Predicts: {self.prediction}"
        return line

    @classmethod
    def from_dict(cls, data: dict) -> CollisionStatement:
        """Create from a dict (parsed from LLM JSON output)."""
        return cls(
            from_worker=str(data.get("from_worker", "")),
            peer_worker=str(data.get("peer_worker", "")),
            collision=str(data.get("collision", "")),
            causal_link=str(data.get("causal_link", "")),
            serendipity_score=int(data.get("serendipity_score", 3)),
            prediction=data.get("prediction"),
            targeted_question=data.get("targeted_question"),
            round=int(data.get("round", 0)),
        )


def parse_collisions_from_response(
    raw: str,
    from_worker: str,
    round_num: int,
) -> list[CollisionStatement]:
    """Parse LLM response into CollisionStatements.

    Tries JSON first, then falls back to line-by-line extraction.
    Graceful — never raises, returns empty list on total failure.
    """
    # Try JSON array parse
    try:
        # Find the JSON array in the response (may have preamble text)
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, list):
                results = []
                for item in data:
                    if isinstance(item, dict):
                        cs = CollisionStatement.from_dict(item)
                        cs.from_worker = cs.from_worker or from_worker
                        cs.round = round_num
                        if cs.collision:  # skip empty
                            results.append(cs)
                if results:
                    return results
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract collision-like statements from free text
    results: list[CollisionStatement] = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or len(line) < 30:
            continue
        # Look for patterns that indicate a collision statement
        collision_markers = [
            "explains", "illuminates", "bridges", "connects",
            "predicts", "surprising", "anomaly", "mechanism",
            "because", "therefore", "this means",
        ]
        if any(marker in line.lower() for marker in collision_markers):
            results.append(CollisionStatement(
                from_worker=from_worker,
                peer_worker="unknown",
                collision=line[:500],
                causal_link="",
                serendipity_score=3,
                round=round_num,
            ))

    return results[:10]  # cap at 10 per response
