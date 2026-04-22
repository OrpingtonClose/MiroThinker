# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ConditionStore-backed tools for swarm worker agents.

Each tool wraps a ConditionStore query or mutation behind the Strands
``@tool`` decorator.  Workers receive these as their toolset and use
them to explore the corpus, discover peer findings, and store their
own analysis — all without knowing they're in a swarm.

The tools enforce a key invariant: **workers never hold the full corpus
in their context window**.  They pull data on demand via search and
retrieval, process it in their (small) context, and write findings back.
Context window size becomes irrelevant — the store IS the memory.

Usage:
    tools = build_worker_tools(store, worker_angle="insulin_timing")
    agent = Agent(tools=tools, ...)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from typing import TYPE_CHECKING, Any

from strands import tool

if TYPE_CHECKING:
    from corpus import ConditionStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state — each worker agent gets its own closure over these
# ---------------------------------------------------------------------------
# We use a factory function (build_worker_tools) that captures the store
# and worker identity in closures.  This avoids global mutable state and
# lets multiple workers run concurrently with their own tool instances.


def _keyword_score(query_terms: list[str], text: str) -> float:
    """Score text relevance against query terms via keyword overlap."""
    if not query_terms or not text:
        return 0.0
    text_lower = text.lower()
    score = 0.0
    for term in query_terms:
        if re.search(rf"\b{re.escape(term)}\b", text_lower):
            score += 1.0
        elif term in text_lower:
            score += 0.5
    return score


def _extract_terms(text: str, top_k: int = 20) -> list[str]:
    """Extract key terms from text for search scoring."""
    stopwords = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "but", "or", "not", "this", "that", "it", "its",
        "they", "them", "their", "we", "our", "you", "your",
        "what", "which", "who", "where", "when", "why", "how",
        "all", "any", "about", "also", "data", "findings",
    })
    tokens = re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
    filtered = [t for t in tokens if t not in stopwords and len(t) >= 3]
    if not filtered:
        return []
    counts = Counter(filtered)
    return [term for term, _ in counts.most_common(top_k)]


def build_worker_tools(
    store: "ConditionStore",
    worker_angle: str,
    worker_id: str,
    phase: str = "worker",
) -> list[Any]:
    """Build a set of @tool-decorated functions bound to a specific worker.

    Each tool closes over the store, worker identity, and a thread-local
    counter for tracking how many findings this worker has stored.

    Args:
        store: The ConditionStore backing all reads/writes.
        worker_angle: This worker's assigned research angle.
        worker_id: Unique identifier for this worker.
        phase: Current swarm phase (for event logging).

    Returns:
        List of tool-decorated callables ready for a Strands Agent.
    """
    _finding_count = {"n": 0}
    _lock = threading.Lock()

    def _log_tool_call(tool_name: str, args: dict, result_summary: str) -> int:
        """Log a tool invocation as a graph node in the ConditionStore.

        Every tool call becomes a row_type='tool_call' condition, creating
        a complete audit trail of how each worker explored the corpus.
        """
        from datetime import datetime, timezone

        fact = f"[{tool_name}] {json.dumps(args, default=str)[:500]}"
        metadata = json.dumps({
            "tool": tool_name,
            "args": args,
            "worker_id": worker_id,
            "result_summary": result_summary[:300],
        })
        now = datetime.now(timezone.utc).isoformat()

        with store._lock:
            cid = store._next_id
            store._next_id += 1
            store.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_type, source_ref, row_type,
                    consider_for_use, angle, strategy,
                    created_at, phase)
                   VALUES (?, ?, 'tool_call', ?, 'tool_call',
                           FALSE, ?, ?, ?, ?)""",
                [
                    cid, fact, f"{worker_id}/{tool_name}",
                    worker_angle, metadata, now, phase,
                ],
            )

        logger.debug(
            "worker=<%s>, tool=<%s>, event_id=<%d> | tool call logged",
            worker_id, tool_name, cid,
        )
        return cid

    @tool
    def search_corpus(query: str, max_results: int = 15) -> str:
        """Search the research corpus for findings relevant to a query.

        Use this to pull evidence from the corpus that relates to your
        current line of reasoning.  Returns findings with their source
        attribution and confidence scores.

        Args:
            query: Natural language search query describing what you need.
            max_results: Maximum number of findings to return.

        Returns:
            Formatted text block of matching findings with sources.
        """
        query_terms = _extract_terms(query)
        if not query_terms:
            return "(no searchable terms in query)"

        with store._lock:
            rows = store.conn.execute(
                """SELECT id, fact, source_url, source_type, confidence,
                          angle, verification_status
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                   ORDER BY confidence DESC
                   LIMIT 500""",
            ).fetchall()

        if not rows:
            return "(corpus is empty — no findings available)"

        # Score and rank by relevance
        scored = []
        for row in rows:
            cid, fact, src_url, src_type, conf, angle, vstatus = row
            score = _keyword_score(query_terms, fact)
            # Boost findings from same angle slightly
            if angle and angle.lower() in worker_angle.lower():
                score *= 1.2
            if score > 0:
                scored.append((score, cid, fact, src_url, src_type, conf, vstatus))

        scored.sort(key=lambda x: (-x[0], -x[5]))
        top = scored[:max_results]

        if not top:
            _log_tool_call("search_corpus", {"query": query}, "no matches")
            return f"(no findings match query: {query})"

        lines = []
        for score, cid, fact, src_url, src_type, conf, vstatus in top:
            src_tag = f"[{src_type}]" if src_type else ""
            url_tag = f" ({src_url})" if src_url else ""
            conf_tag = f" [conf={conf:.2f}]" if conf != 0.5 else ""
            lines.append(f"[#{cid}]{src_tag}{conf_tag} {fact}{url_tag}")

        _log_tool_call("search_corpus", {"query": query, "max_results": max_results}, f"{len(top)} results")
        return {
            "status": "success",
            "content": [{"text": f"=== {len(top)} findings for: {query} ===\n" + "\n".join(lines)}],
        }

    @tool
    def get_peer_insights(topic: str, max_results: int = 10) -> str:
        """Retrieve insights from other specialists about a specific topic.

        Use this when you encounter something cross-domain — another
        specialist may have already analyzed it from their perspective.
        Returns findings from OTHER angles (not your own).

        Args:
            topic: The topic or concept you want peer insights about.
            max_results: Maximum number of peer insights to return.

        Returns:
            Formatted text block of peer findings with their angles.
        """
        topic_terms = _extract_terms(topic)
        if not topic_terms:
            return "(no searchable terms in topic)"

        # Get findings from other angles (worker synthesis outputs + gossip)
        with store._lock:
            rows = store.conn.execute(
                """SELECT id, fact, source_type, confidence, angle, phase
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND angle != ?
                     AND angle != ''
                     AND row_type IN ('finding', 'thought', 'insight')
                   ORDER BY confidence DESC
                   LIMIT 500""",
                [worker_angle],
            ).fetchall()

        if not rows:
            return "(no peer insights available yet — you may be the first to analyze)"

        scored = []
        for row in rows:
            cid, fact, src_type, conf, angle, row_phase = row
            score = _keyword_score(topic_terms, fact)
            if score > 0:
                scored.append((score, cid, fact, src_type, conf, angle, row_phase))

        scored.sort(key=lambda x: (-x[0], -x[4]))
        top = scored[:max_results]

        if not top:
            _log_tool_call("get_peer_insights", {"topic": topic}, "no matches")
            return f"(no peer insights match topic: {topic})"

        lines = []
        for score, cid, fact, src_type, conf, angle, row_phase in top:
            lines.append(f"[{angle}] [conf={conf:.2f}] {fact}")

        _log_tool_call("get_peer_insights", {"topic": topic, "max_results": max_results}, f"{len(top)} results")
        return {
            "status": "success",
            "content": [{"text": f"=== {len(top)} peer insights on: {topic} ===\n" + "\n".join(lines)}],
        }

    @tool
    def store_finding(
        fact: str,
        confidence: float = 0.7,
        evidence_source: str = "",
        reasoning: str = "",
    ) -> str:
        """Store a research finding you have discovered or synthesized.

        Call this whenever you reach a conclusion supported by evidence.
        Your finding becomes available to other specialists working on
        related topics.  Store specific, evidence-backed claims — not
        summaries or opinions.

        Args:
            fact: The specific finding or claim (be precise and evidence-based).
            confidence: Your confidence in this finding (0.0 to 1.0).
            evidence_source: URL or description of the evidence source.
            reasoning: Brief explanation of your reasoning chain.

        Returns:
            Confirmation with the finding's ID.
        """
        if not fact or not fact.strip():
            return {"status": "error", "content": [{"text": "Finding cannot be empty"}]}

        confidence = max(0.0, min(1.0, confidence))

        metadata = json.dumps({
            "worker_id": worker_id,
            "reasoning": reasoning,
        }) if reasoning else ""

        with store._lock:
            cid = store._next_id
            store._next_id += 1
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            store.conn.execute(
                """INSERT INTO conditions
                   (id, fact, source_url, source_type, row_type,
                    consider_for_use, confidence, angle, strategy,
                    created_at, phase, verification_status)
                   VALUES (?, ?, ?, 'worker_analysis', 'finding',
                           TRUE, ?, ?, ?, ?, ?, 'speculative')""",
                [
                    cid, fact.strip(), evidence_source,
                    confidence, worker_angle, metadata,
                    now, phase,
                ],
            )

        with _lock:
            _finding_count["n"] += 1

        logger.debug(
            "worker=<%s>, finding_id=<%d>, confidence=<%.2f> | finding stored",
            worker_id, cid, confidence,
        )

        _log_tool_call("store_finding", {"fact": fact[:100], "confidence": confidence}, f"stored as #{cid}")
        return {
            "status": "success",
            "content": [{"text": f"Finding #{cid} stored (confidence={confidence:.2f})"}],
        }

    @tool
    def check_contradictions(claim: str) -> str:
        """Check if a claim contradicts existing findings in the corpus.

        Use this when you encounter conflicting evidence or want to
        verify whether your analysis aligns with or challenges what
        others have found.

        Args:
            claim: The specific claim to check for contradictions.

        Returns:
            Any contradicting or supporting findings from the corpus.
        """
        claim_terms = _extract_terms(claim, top_k=10)
        if not claim_terms:
            return "(no searchable terms in claim)"

        with store._lock:
            rows = store.conn.execute(
                """SELECT id, fact, confidence, angle, verification_status
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                   ORDER BY confidence DESC
                   LIMIT 300""",
            ).fetchall()

        if not rows:
            return "(no existing findings to check against)"

        related = []
        for row in rows:
            cid, fact, conf, angle, vstatus = row
            score = _keyword_score(claim_terms, fact)
            if score > 0:
                related.append((score, cid, fact, conf, angle, vstatus))

        related.sort(key=lambda x: -x[0])
        top = related[:10]

        if not top:
            _log_tool_call("check_contradictions", {"claim": claim[:100]}, "no matches")
            return f"(no related findings found for: {claim})"

        lines = []
        for score, cid, fact, conf, angle, vstatus in top:
            status = f" [{vstatus}]" if vstatus else ""
            lines.append(f"[#{cid}] [{angle}] [conf={conf:.2f}]{status} {fact}")

        _log_tool_call("check_contradictions", {"claim": claim[:100]}, f"{len(top)} related")
        return {
            "status": "success",
            "content": [{"text": f"=== {len(top)} related findings ===\n" + "\n".join(lines)
                         + "\n\nCompare these with your claim and reason about "
                         "whether they support, contradict, or extend it."}],
        }

    @tool
    def get_research_gaps() -> str:
        """Identify topics with low coverage or unresolved questions.

        Use this to discover what areas need more investigation.
        Returns angles with few findings, low-confidence claims,
        and unresolved contradictions.

        Returns:
            Summary of research gaps and low-coverage areas.
        """
        lines = []

        # Angle coverage
        with store._lock:
            angle_stats = store.conn.execute(
                """SELECT angle, COUNT(*) as cnt,
                          AVG(confidence) as avg_conf
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                     AND angle != ''
                   GROUP BY angle
                   ORDER BY cnt ASC""",
            ).fetchall()

        if angle_stats:
            lines.append("=== ANGLE COVERAGE ===")
            for angle, cnt, avg_conf in angle_stats:
                status = "LOW" if cnt < 5 else "OK"
                lines.append(f"  [{status}] {angle}: {cnt} findings, avg_conf={avg_conf:.2f}")

        # Low-confidence findings
        with store._lock:
            low_conf = store.conn.execute(
                """SELECT COUNT(*) FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                     AND confidence < 0.4""",
            ).fetchone()

        if low_conf and low_conf[0] > 0:
            lines.append(f"\n{low_conf[0]} findings have confidence < 0.4 (weak evidence)")

        # Speculative findings
        with store._lock:
            speculative = store.conn.execute(
                """SELECT COUNT(*) FROM conditions
                   WHERE consider_for_use = TRUE
                     AND verification_status = 'speculative'""",
            ).fetchone()

        if speculative and speculative[0] > 0:
            lines.append(f"{speculative[0]} findings are still speculative (unverified)")

        if not lines:
            _log_tool_call("get_research_gaps", {}, "empty corpus")
            return "(no gap analysis available — corpus may be empty)"

        _log_tool_call("get_research_gaps", {}, f"{len(lines)} gap lines")
        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}],
        }

    @tool
    def get_corpus_section(offset: int = 0, max_chars: int = 8000) -> str:
        """Read a chunk of your assigned corpus section.

        The corpus is too large to read all at once.  Call this
        repeatedly with increasing offsets to read through your
        section.  Process each chunk before requesting the next.

        Args:
            offset: Character offset to start reading from (0-indexed).
            max_chars: Maximum characters to return in this chunk.

        Returns:
            A chunk of your corpus section with position info.
        """
        # The corpus section is stored in the worker's raw conditions
        with store._lock:
            rows = store.conn.execute(
                """SELECT fact FROM conditions
                   WHERE row_type = 'raw'
                     AND angle = ?
                   ORDER BY id ASC""",
                [worker_angle],
            ).fetchall()

        if not rows:
            # Fall back to finding-level data for this angle
            with store._lock:
                rows = store.conn.execute(
                    """SELECT fact FROM conditions
                       WHERE consider_for_use = TRUE
                         AND row_type = 'finding'
                         AND angle = ?
                       ORDER BY id ASC""",
                    [worker_angle],
                ).fetchall()

        if not rows:
            return "(no corpus data assigned to your angle)"

        full_text = "\n\n".join(row[0] for row in rows)
        total_chars = len(full_text)
        chunk = full_text[offset:offset + max_chars]
        remaining = max(0, total_chars - offset - max_chars)

        if not chunk:
            _log_tool_call("get_corpus_section", {"offset": offset}, "end of section")
            return "(you have read the entire section — no more data)"

        _log_tool_call("get_corpus_section", {"offset": offset, "max_chars": max_chars}, f"{len(chunk)} chars returned")
        return {
            "status": "success",
            "content": [{"text": f"[chars {offset}-{offset + len(chunk)} of {total_chars}, "
                         f"{remaining} remaining]\n\n{chunk}"}],
        }

    return [
        search_corpus,
        get_peer_insights,
        store_finding,
        check_contradictions,
        get_research_gaps,
        get_corpus_section,
    ]
