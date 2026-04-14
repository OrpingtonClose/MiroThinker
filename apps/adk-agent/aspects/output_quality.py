# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Output quality validators -- detect repetition, stretching, template
leaks, and instruction regurgitation in block outputs.

These validators run as part of the ``after()`` hook on blocks that
produce text output (thinker, synthesiser).  They add warning metrics
to the result but do NOT block execution -- quality issues are
informational, not fatal.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def check_repetition(text: str, threshold: float = 0.3) -> dict[str, Any]:
    """Detect excessive sentence-level repetition.

    Returns metrics dict with repetition_ratio and flagged sentences.
    A ratio > threshold indicates the output is stretching thin content.
    """
    if not text or len(text) < 100:
        return {"repetition_ratio": 0.0, "flagged": False}

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
    if len(sentences) < 3:
        return {"repetition_ratio": 0.0, "flagged": False}

    # Trigram overlap between consecutive sentences
    def _trigrams(s: str) -> set[str]:
        words = s.lower().split()
        return {" ".join(words[i:i+3]) for i in range(len(words) - 2)}

    overlaps = []
    for i in range(1, len(sentences)):
        t1 = _trigrams(sentences[i - 1])
        t2 = _trigrams(sentences[i])
        if t1 and t2:
            overlap = len(t1 & t2) / max(len(t1), len(t2))
            overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    return {
        "repetition_ratio": round(avg_overlap, 3),
        "flagged": avg_overlap > threshold,
    }


def check_stretching(text: str, min_density: float = 0.4) -> dict[str, Any]:
    """Detect content stretching -- low information density.

    Measures the ratio of unique content words to total words.
    Low ratio indicates padding, filler, or repetitive phrasing.
    """
    if not text or len(text) < 100:
        return {"density": 1.0, "flagged": False}

    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    if len(words) < 20:
        return {"density": 1.0, "flagged": False}

    stops = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "her", "was", "one", "our", "out", "has", "its",
        "this", "that", "with", "from", "have", "been", "will",
        "would", "could", "should", "about", "which", "their",
        "there", "these", "those", "into", "also", "more",
    }
    content_words = [w for w in words if w not in stops]
    if not content_words:
        return {"density": 0.0, "flagged": True}

    unique = set(content_words)
    density = len(unique) / len(content_words)
    return {
        "density": round(density, 3),
        "flagged": density < min_density,
    }


def check_template_leaks(text: str) -> dict[str, Any]:
    """Detect template variable leaks in output.

    Catches patterns like {research_findings}, {corpus_for_synthesis},
    or other template variables that should have been interpolated.
    """
    if not text:
        return {"leaks": [], "flagged": False}

    # Match {variable_name} patterns that look like template variables
    pattern = r'\{[a-z_]+\}'
    matches = re.findall(pattern, text)

    # Filter to known template variables
    known_templates = {
        "{research_findings}", "{corpus_for_synthesis}",
        "{research_strategy}", "{user_query}",
        "{expansion_targets}", "{prev_thinker_strategies}",
    }
    leaks = [m for m in matches if m in known_templates]
    return {
        "leaks": leaks,
        "flagged": bool(leaks),
    }


def check_instruction_regurgitation(text: str) -> dict[str, Any]:
    """Detect when the model regurgitates its system instructions.

    Catches patterns where the output contains fragments of the
    prompt/instruction rather than actual analysis.
    """
    if not text or len(text) < 200:
        return {"flagged": False, "markers": []}

    # Instruction-like patterns that shouldn't appear in output
    markers = []
    instruction_patterns = [
        r"you are a (?:research|specialist|expert)",
        r"your (?:job|task|role) is to",
        r"output (?:only valid )?json",
        r"output format:",
        r"strict rules:",
        r"instructions:",
        r"do not (?:rephrase|repeat|include)",
        r"respond with exactly:",
    ]
    for pattern in instruction_patterns:
        if re.search(pattern, text.lower()):
            markers.append(pattern)

    return {
        "flagged": len(markers) >= 2,  # Single match could be coincidence
        "markers": markers[:5],
    }


def validate_output_quality(text: str) -> dict[str, Any]:
    """Run all quality validators on a text output.

    Returns a combined metrics dict with per-validator results.
    """
    results: dict[str, Any] = {
        "repetition": check_repetition(text),
        "stretching": check_stretching(text),
        "template_leaks": check_template_leaks(text),
        "instruction_regurgitation": check_instruction_regurgitation(text),
    }
    results["any_flagged"] = any(
        v.get("flagged", False) for v in results.values()
    )
    return results
