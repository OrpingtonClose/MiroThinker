# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
After-agent callback for the researcher: decomposes findings into atoms.

After the researcher outputs its findings text (stored in
``state["research_findings"]``), this callback:

  1. Parses the text into individual ``AtomicCondition`` objects
  2. Stores them in the shared ``CorpusStore`` (DuckDB)
  3. Overwrites ``state["research_findings"]`` with the structured corpus
     formatted for the thinker's next iteration

This bridges the gap between the researcher's free-text output and the
structured corpus that the thinker reads.  The researcher writes prose;
the condition manager turns it into queryable atoms.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

from google.adk.agents.callback_context import CallbackContext
from google.genai import types as genai_types

from models.atomic_condition import AtomicCondition

if TYPE_CHECKING:
    from models.corpus_store import CorpusStore

logger = logging.getLogger(__name__)

# URL pattern for extracting source URLs from findings text
_URL_RE = re.compile(r"https?://[^\s)\]\"'>]+")

# Confidence keywords → score mapping
_CONFIDENCE_KEYWORDS: list[tuple[list[str], float]] = [
    (["confirmed", "verified", "established", "proven", "definitive"], 0.9),
    (["strong evidence", "multiple sources", "well-documented", "reliable"], 0.8),
    (["likely", "probable", "credible", "reported by"], 0.7),
    (["suggests", "indicates", "appears", "some evidence"], 0.6),
    (["possible", "may", "could", "uncertain", "mixed"], 0.5),
    (["speculative", "unconfirmed", "rumour", "anecdotal", "unclear"], 0.3),
    (["unlikely", "doubtful", "disputed", "contradicted"], 0.2),
]


def _infer_confidence(text: str) -> float:
    """Infer confidence from keyword heuristics in the text."""
    lower = text.lower()
    for keywords, score in _CONFIDENCE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return score
    return 0.5  # default


def _extract_source_url(text: str) -> str:
    """Extract the first URL from a text chunk."""
    match = _URL_RE.search(text)
    return match.group(0).rstrip(".,;:") if match else ""


def _parse_findings_to_conditions(
    findings_text: str,
    iteration: int = 0,
) -> list[AtomicCondition]:
    """Parse free-text findings into AtomicCondition objects.

    Splits on paragraph breaks, numbered lists, or bullet points.
    Each chunk becomes one atom with inferred confidence and source URL.
    """
    if not findings_text or findings_text.strip() == "(no findings yet)":
        return []

    # Split on common delimiters: numbered items, bullets, double newlines
    chunks = re.split(
        r"\n\s*(?:\d+[\.\)]\s+|[-*•]\s+|\n)",
        findings_text,
    )

    conditions: list[AtomicCondition] = []
    for chunk in chunks:
        text = chunk.strip()
        # Skip preamble, headers, and very short chunks
        if not text or len(text) < 20:
            continue
        # Skip lines that look like section headers
        if text.startswith("===") or text.startswith("##") or text.startswith("---"):
            continue
        # Skip lines that are just metadata (e.g. "CORPUS: 5 conditions")
        if text.startswith("CORPUS:") or text.startswith("STATUS SUMMARY:"):
            continue

        source_url = _extract_source_url(text)
        confidence = _infer_confidence(text)

        conditions.append(AtomicCondition(
            fact=text,
            source_url=source_url,
            confidence=confidence,
            verification_status="verified",  # tool-grounded from researcher
            angle=f"iteration_{iteration}",
            expansion_depth=0,
        ))

    return conditions


def researcher_condition_callback(
    callback_context: CallbackContext,
) -> Optional[genai_types.Content]:
    """After-agent callback: decompose researcher findings into atoms.

    Reads ``state["research_findings"]``, parses into AtomicConditions,
    stores in the CorpusStore, and updates state with formatted corpus.
    """
    state = callback_context.state

    findings_text = state.get("research_findings", "")
    if not findings_text:
        return None

    # Get or create the corpus store
    # The store is kept in state as a serialised reference;
    # we use a module-level singleton for the DuckDB connection.
    corpus = _get_corpus(state)

    # Track iteration number
    iteration = state.get("_corpus_iteration", 0)

    # Parse findings into atoms
    new_conditions = _parse_findings_to_conditions(findings_text, iteration)
    if new_conditions:
        ids = corpus.admit_batch(new_conditions)
        logger.info(
            "Condition manager: admitted %d/%d conditions (iteration %d, total %d)",
            len(ids), len(new_conditions), iteration, corpus.count(),
        )

    # Update state with structured corpus for thinker
    state["research_findings"] = corpus.format_for_thinker()
    state["_corpus_iteration"] = iteration + 1

    # Also store synthesiser-formatted version for the final stage
    state["corpus_for_synthesis"] = corpus.format_for_synthesiser()

    return None  # preserve original researcher output


# ---------------------------------------------------------------------------
# Module-level corpus store singleton
# ---------------------------------------------------------------------------
# Each pipeline run creates a fresh CorpusStore.  The store lives for the
# duration of the LoopAgent execution.  We use a dict keyed by a session
# marker so concurrent pipelines don't collide.

_corpus_stores: dict[str, "CorpusStore"] = {}


def _get_corpus(state: dict) -> "CorpusStore":
    """Get or create the CorpusStore for this session."""
    from models.corpus_store import CorpusStore

    # Use session-level key to isolate concurrent runs
    key = state.get("_corpus_key", "default")
    if key not in _corpus_stores:
        _corpus_stores[key] = CorpusStore()
        logger.info("Created new CorpusStore for key=%s", key)
    return _corpus_stores[key]


def reset_corpus(state: dict) -> None:
    """Reset the corpus store for a new pipeline run.

    Closes and removes the previous CorpusStore (if any) to avoid
    leaking DuckDB connections in long-running servers.
    """
    import uuid

    # Close and remove the previous corpus store if it exists
    old_key = state.get("_corpus_key")
    if old_key and old_key in _corpus_stores:
        _corpus_stores[old_key].close()
        del _corpus_stores[old_key]

    key = f"corpus_{uuid.uuid4().hex[:8]}"
    state["_corpus_key"] = key
    # Provide fallback so synthesiser never gets a raw template literal
    state.setdefault("corpus_for_synthesis", "(no findings)")
    logger.info("Reset corpus store with key=%s", key)
