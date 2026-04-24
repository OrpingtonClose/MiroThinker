# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Swarm angle definitions — thin loader.

Angle content (descriptions, enrichment queries, research query) is loaded
from a local JSON file at runtime.  The JSON path is resolved from the
``SWARM_ANGLES_JSON`` environment variable or defaults to
``~/miro_prompts/angles.json``.

The JSON schema::

    {
      "angles": [
        {
          "label": "...",
          "description": "...",
          "enrichment_queries": ["...", ...],
          "key_compounds": ["...", ...],
          "key_interactions": ["...", ...]
        },
        ...
      ],
      "confidence_tiers": "... multiline string ...",
      "swarm_query": "... the full research query ..."
    }
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ANGLES_PATH = Path.home() / "miro_prompts" / "angles.json"


@dataclass
class AngleDefinition:
    """A single swarm angle with enrichment context."""

    label: str
    description: str
    enrichment_queries: list[str] = field(default_factory=list)
    key_compounds: list[str] = field(default_factory=list)
    key_interactions: list[str] = field(default_factory=list)


def _resolve_angles_path() -> Path:
    """Return the path to the angles JSON file."""
    env = os.environ.get("SWARM_ANGLES_JSON")
    if env:
        return Path(env)
    return _DEFAULT_ANGLES_PATH


def _load_angles_data() -> dict:
    """Load and cache the raw JSON data."""
    path = _resolve_angles_path()
    if not path.exists():
        logger.error(
            "path=<%s> | angles JSON not found — set SWARM_ANGLES_JSON or "
            "place angles.json at the default path",
            path,
        )
        return {"angles": [], "confidence_tiers": "", "swarm_query": ""}

    with open(path) as f:
        data = json.load(f)

    logger.info(
        "path=<%s>, angle_count=<%d> | loaded angle definitions",
        path, len(data.get("angles", [])),
    )
    return data


# Load once at import time
_DATA = _load_angles_data()

ALL_ANGLES: list[AngleDefinition] = [
    AngleDefinition(
        label=a["label"],
        description=a["description"],
        enrichment_queries=a.get("enrichment_queries", []),
        key_compounds=a.get("key_compounds", []),
        key_interactions=a.get("key_interactions", []),
    )
    for a in _DATA.get("angles", [])
]

REQUIRED_ANGLE_LABELS: list[str] = [a.label for a in ALL_ANGLES]

ALL_ENRICHMENT_QUERIES: list[str] = []
for _angle in ALL_ANGLES:
    ALL_ENRICHMENT_QUERIES.extend(_angle.enrichment_queries)

CONFIDENCE_TIERS: str = _DATA.get("confidence_tiers", "")


def get_swarm_query() -> str:
    """Return the master swarm query loaded from the angles JSON."""
    return _DATA.get("swarm_query", "")
