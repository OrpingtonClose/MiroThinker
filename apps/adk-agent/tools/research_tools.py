# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Research orchestration tools — store_finding, read_findings.

JSONL-based external accumulator so the LLM can persist findings
*outside* its context window.  Older findings can be trimmed by
Keep-K-Recent without data loss.

Note: For link extraction, use Firecrawl's ``firecrawl_map`` MCP tool
instead of a custom implementation — it does the same thing natively.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# ── Configurable paths ──────────────────────────────────────────────
FINDINGS_DIR = Path(os.environ.get("FINDINGS_DIR", "/tmp/mirothinker"))
FINDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Current session findings file — set by batch orchestrator or defaults
_findings_file: Path = FINDINGS_DIR / "findings.jsonl"


def set_findings_file(name: str) -> Path:
    """Set the active findings file (called by batch orchestrator)."""
    global _findings_file
    _findings_file = FINDINGS_DIR / name
    return _findings_file


def get_findings_file() -> Path:
    """Return the current findings file path."""
    return _findings_file


def clear_findings() -> None:
    """Clear the current findings file (called between batch runs)."""
    if _findings_file.exists():
        _findings_file.unlink()


# ── store_finding ───────────────────────────────────────────────────


async def store_finding(
    name: str,
    url: str,
    category: str,
    summary: str,
    rating: int = 0,
) -> str:
    """Store an evaluated finding to persistent JSONL storage.

    Findings persist outside the LLM context window so Keep-K-Recent
    can trim old tool results without losing accumulated data.

    Args:
        name: Short name / title of the finding.
        url: Source URL.
        category: Category (e.g. "vendor", "forum", "news", "academic").
        summary: One-paragraph evaluation summary.
        rating: Quality rating 1-10 (0 = unrated).

    Returns:
        Confirmation message.
    """
    finding = {
        "name": name,
        "url": url,
        "category": category,
        "summary": summary,
        "rating": rating,
        "ts": time.time(),
    }
    with open(_findings_file, "a") as f:
        f.write(json.dumps(finding, ensure_ascii=False) + "\n")

    logger.info("Stored finding: %s (%s)", name, category)
    return f"Stored: {name} [{category}] (rating={rating})"


# ── read_findings ───────────────────────────────────────────────────


async def read_findings(category: str = "") -> str:
    """Read back all stored findings, optionally filtered by category.

    Args:
        category: If non-empty, only return findings matching this category.

    Returns:
        JSON array of finding objects.
    """
    if not _findings_file.exists():
        return json.dumps([])

    findings = []
    for line in _findings_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if category and obj.get("category", "") != category:
            continue
        findings.append(obj)

    return json.dumps(findings, ensure_ascii=False)


# ── Public FunctionTool instances ───────────────────────────────────

store_finding_tool = FunctionTool(store_finding)
read_findings_tool = FunctionTool(read_findings)

RESEARCH_TOOLS = [store_finding_tool, read_findings_tool]
