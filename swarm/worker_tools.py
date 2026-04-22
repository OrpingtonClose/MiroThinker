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

24-hour continuous operation:
    Every tool return is capped by ``max_return_chars`` (default 6000).
    With 600K+ findings in the store after hundreds of runs, uncapped
    tool returns would overflow any model's context window.  The budget
    ensures workers always get a curated slice, never the raw dump.

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


# Hard ceiling on characters returned by any single tool call.
# Prevents context overflow when the store grows to 600K+ findings
# during 24-hour continuous operation.  Workers get a curated slice
# of the highest-relevance findings, not a raw dump.
DEFAULT_MAX_RETURN_CHARS = 6000


def _truncate_to_budget(
    lines: list[str],
    budget: int,
    header: str = "",
) -> str:
    """Join lines up to a character budget, appending a truncation notice.

    Args:
        lines: Lines to join.
        budget: Maximum total characters.
        header: Optional header prepended before counting.

    Returns:
        Joined text within budget.
    """
    if not lines:
        return header or "(no results)"

    result_parts = []
    used = len(header)
    included = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        result_parts.append(line)
        used += len(line) + 1
        included += 1

    text = header + "\n".join(result_parts) if header else "\n".join(result_parts)
    if included < len(lines):
        text += f"\n\n[... {len(lines) - included} more results truncated to fit context budget]"
    return text


def build_worker_tools(
    store: "ConditionStore",
    worker_angle: str,
    worker_id: str,
    phase: str = "worker",
    max_return_chars: int = DEFAULT_MAX_RETURN_CHARS,
    source_model: str = "",
    source_run: str = "",
) -> list[Any]:
    """Build a set of @tool-decorated functions bound to a specific worker.

    Each tool closes over the store, worker identity, and a thread-local
    counter for tracking how many findings this worker has stored.

    Args:
        store: The ConditionStore backing all reads/writes.
        worker_angle: This worker's assigned research angle.
        worker_id: Unique identifier for this worker.
        phase: Current swarm phase (for event logging).
        max_return_chars: Hard ceiling on characters any tool call returns.
        source_model: Model name for provenance tracking (#192).
        source_run: Run identifier for cross-run comparison (#192).

    Returns:
        List of tool-decorated callables ready for a Strands Agent.
    """
    _finding_count = {"n": 0}
    _lock = threading.Lock()
    _budget = max_return_chars

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

        Results are capped to fit your context window.

        Args:
            query: Natural language search query describing what you need.
            max_results: Maximum number of findings to return.

        Returns:
            Formatted text block of matching findings with sources.
        """
        query_terms = _extract_terms(query)
        if not query_terms:
            return "(no searchable terms in query)"

        # Use LIMIT proportional to store size but cap at 2000 for speed
        with store._lock:
            total_count = store.conn.execute(
                "SELECT COUNT(*) FROM conditions WHERE consider_for_use = TRUE AND row_type = 'finding'"
            ).fetchone()[0]
            fetch_limit = min(2000, max(500, total_count))
            rows = store.conn.execute(
                """SELECT id, fact, source_url, source_type, confidence,
                          angle, verification_status
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type = 'finding'
                   ORDER BY confidence DESC
                   LIMIT ?""",
                [fetch_limit],
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
        header = f"=== {len(top)} findings for: {query} (store has {total_count} total) ===\n"
        text = _truncate_to_budget(lines, _budget, header)
        return {
            "status": "success",
            "content": [{"text": text}],
        }

    @tool
    def get_peer_insights(topic: str, max_results: int = 10) -> str:
        """Retrieve insights from other specialists about a specific topic.

        Use this when you encounter something cross-domain — another
        specialist may have already analyzed it from their perspective.
        Returns findings from OTHER angles (not your own).

        Results are capped to fit your context window.

        Args:
            topic: The topic or concept you want peer insights about.
            max_results: Maximum number of peer insights to return.

        Returns:
            Formatted text block of peer findings with their angles.
        """
        topic_terms = _extract_terms(topic)
        if not topic_terms:
            return "(no searchable terms in topic)"

        # Prioritize worker-generated insights over raw corpus paragraphs
        with store._lock:
            rows = store.conn.execute(
                """SELECT id, fact, source_type, confidence, angle, phase
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND angle != ?
                     AND angle != ''
                     AND row_type IN ('finding', 'thought', 'insight')
                   ORDER BY
                     CASE WHEN source_type = 'worker_analysis' THEN 0 ELSE 1 END,
                     confidence DESC
                   LIMIT 1000""",
                [worker_angle],
            ).fetchall()

        if not rows:
            return "(no peer insights available yet — you may be the first to analyze)"

        scored = []
        for row in rows:
            cid, fact, src_type, conf, angle, row_phase = row
            score = _keyword_score(topic_terms, fact)
            # Boost worker-generated findings over raw corpus paragraphs
            if src_type == "worker_analysis":
                score *= 2.0
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
        header = f"=== {len(top)} peer insights on: {topic} ===\n"
        text = _truncate_to_budget(lines, _budget, header)
        return {
            "status": "success",
            "content": [{"text": text}],
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

        cid = store.admit(
            fact=fact.strip(),
            source_url=evidence_source,
            source_type="worker_analysis",
            row_type="finding",
            confidence=confidence,
            angle=worker_angle,
            strategy=metadata,
            verification_status="speculative",
            source_model=source_model,
            source_run=source_run,
            phase=phase,
        )

        if cid is None:
            return {"status": "error", "content": [{"text": "Finding could not be stored"}]}

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
                   LIMIT 1000""",
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
        header = f"=== {len(top)} related findings ===\n"
        footer = ("\n\nCompare these with your claim and reason about "
                  "whether they support, contradict, or extend it.")
        text = _truncate_to_budget(lines, _budget - len(footer), header) + footer
        return {
            "status": "success",
            "content": [{"text": text}],
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
    def get_corpus_section(offset: int = 0, max_chars: int = 6000) -> str:
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
        # Cap max_chars to the tool budget
        max_chars = min(max_chars, _budget)

        # Stream rows instead of loading full corpus into memory.
        # With 150MB corpus, concatenating all rows would OOM.
        with store._lock:
            row_count = store.conn.execute(
                """SELECT COUNT(*) FROM conditions
                   WHERE row_type = 'raw' AND angle = ?""",
                [worker_angle],
            ).fetchone()[0]

        if row_count == 0:
            with store._lock:
                row_count = store.conn.execute(
                    """SELECT COUNT(*) FROM conditions
                       WHERE consider_for_use = TRUE
                         AND row_type = 'finding' AND angle = ?""",
                    [worker_angle],
                ).fetchone()[0]

        if row_count == 0:
            return "(no corpus data assigned to your angle)"

        # Paginated fetch: skip rows until we reach the offset,
        # then collect up to max_chars
        row_type_filter = "row_type = 'raw'" if row_count > 0 else (
            "consider_for_use = TRUE AND row_type = 'finding'"
        )
        # Re-check which type has data
        with store._lock:
            raw_count = store.conn.execute(
                "SELECT COUNT(*) FROM conditions WHERE row_type = 'raw' AND angle = ?",
                [worker_angle],
            ).fetchone()[0]

        if raw_count > 0:
            row_type_filter = "row_type = 'raw'"
        else:
            row_type_filter = "consider_for_use = TRUE AND row_type = 'finding'"

        # Estimate total chars from row count (avoid full scan)
        # Fetch rows in pages, accumulate chars until offset, then collect chunk
        page_size = 100
        chars_seen = 0
        chunk_parts: list[str] = []
        chunk_started = False
        chunk_chars = 0
        done = False
        total_chars_estimate = 0

        for page in range(0, row_count, page_size):
            if done:
                break
            with store._lock:
                rows = store.conn.execute(
                    f"""SELECT fact FROM conditions
                       WHERE {row_type_filter} AND angle = ?
                       ORDER BY id ASC
                       LIMIT ? OFFSET ?""",
                    [worker_angle, page_size, page],
                ).fetchall()

            for (fact_text,) in rows:
                fact_len = len(fact_text) + 2  # +2 for "\n\n" separator
                total_chars_estimate += fact_len

                if not chunk_started:
                    if chars_seen + fact_len > offset:
                        # This row contains the start of our chunk
                        chunk_started = True
                        local_offset = offset - chars_seen
                        snippet = fact_text[local_offset:]
                        chunk_parts.append(snippet)
                        chunk_chars += len(snippet)
                    else:
                        chars_seen += fact_len
                else:
                    if chunk_chars + fact_len > max_chars:
                        done = True
                        break
                    chunk_parts.append(fact_text)
                    chunk_chars += fact_len

        # If we didn't scan all rows, estimate remaining
        if not done and page + page_size < row_count:
            avg_row_chars = total_chars_estimate / max(1, (page + page_size))
            total_chars_estimate = int(avg_row_chars * row_count)

        chunk = "\n\n".join(chunk_parts)
        remaining = max(0, total_chars_estimate - offset - len(chunk))

        if not chunk:
            _log_tool_call("get_corpus_section", {"offset": offset}, "end of section")
            return "(you have read the entire section — no more data)"

        _log_tool_call(
            "get_corpus_section",
            {"offset": offset, "max_chars": max_chars},
            f"{len(chunk)} chars returned",
        )
        return {
            "status": "success",
            "content": [{"text": f"[chars {offset}-{offset + len(chunk)} of ~{total_chars_estimate}, "
                         f"~{remaining} remaining]\n\n{chunk}"}],
        }

    @tool
    def find_connections(angle_a: str, angle_b: str, max_results: int = 8) -> str:
        """Discover cross-domain connections between two research angles.

        Finds findings from angle_a and angle_b that share related terms,
        even if the connection is not obvious.  Use this to discover
        interactions like dietary iron + trenbolone hematocrit effects.

        Args:
            angle_a: First research angle or topic.
            angle_b: Second research angle or topic.
            max_results: Maximum connection pairs to return.

        Returns:
            Pairs of findings from different angles that may interact.
        """
        # Get top findings from each angle
        with store._lock:
            rows_a = store.conn.execute(
                """SELECT id, fact, confidence, angle
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                     AND (angle LIKE ? OR angle LIKE ?)
                   ORDER BY confidence DESC
                   LIMIT 200""",
                [f"%{angle_a}%", f"%{angle_a.lower()}%"],
            ).fetchall()
            rows_b = store.conn.execute(
                """SELECT id, fact, confidence, angle
                   FROM conditions
                   WHERE consider_for_use = TRUE
                     AND row_type IN ('finding', 'thought', 'insight')
                     AND (angle LIKE ? OR angle LIKE ?)
                   ORDER BY confidence DESC
                   LIMIT 200""",
                [f"%{angle_b}%", f"%{angle_b.lower()}%"],
            ).fetchall()

        if not rows_a or not rows_b:
            return f"(insufficient findings for {angle_a} and/or {angle_b})"

        # Extract terms from each set and find overlap
        terms_a_all = set()
        for _, fact, _, _ in rows_a:
            terms_a_all.update(_extract_terms(fact, top_k=10))
        terms_b_all = set()
        for _, fact, _, _ in rows_b:
            terms_b_all.update(_extract_terms(fact, top_k=10))

        # Shared terms suggest a connection exists
        shared_terms = terms_a_all & terms_b_all

        # Score pairs by shared-term overlap
        pairs: list[tuple[float, str]] = []
        for id_a, fact_a, conf_a, ang_a in rows_a[:50]:
            terms_a = set(_extract_terms(fact_a, top_k=10))
            for id_b, fact_b, conf_b, ang_b in rows_b[:50]:
                terms_b = set(_extract_terms(fact_b, top_k=10))
                overlap = terms_a & terms_b
                if overlap:
                    score = len(overlap) * (conf_a + conf_b)
                    bridging = overlap & shared_terms
                    bridge_str = ", ".join(sorted(bridging)[:5]) if bridging else "indirect"
                    pairs.append((
                        score,
                        f"CONNECTION via [{bridge_str}]:\n"
                        f"  A [{ang_a}] conf={conf_a:.2f}: {fact_a[:200]}\n"
                        f"  B [{ang_b}] conf={conf_b:.2f}: {fact_b[:200]}",
                    ))

        pairs.sort(key=lambda x: -x[0])
        top_pairs = pairs[:max_results]

        if not top_pairs:
            _log_tool_call(
                "find_connections",
                {"angle_a": angle_a, "angle_b": angle_b},
                "no connections",
            )
            return (
                f"(no direct term overlap between {angle_a} and {angle_b} findings "
                f"— shared vocabulary: {', '.join(sorted(shared_terms)[:10]) or 'none'})"
            )

        lines = [p[1] for p in top_pairs]
        _log_tool_call(
            "find_connections",
            {"angle_a": angle_a, "angle_b": angle_b},
            f"{len(top_pairs)} connections",
        )
        header = (
            f"=== {len(top_pairs)} connections between {angle_a} and {angle_b} ===\n"
            f"Shared vocabulary: {', '.join(sorted(shared_terms)[:15])}\n\n"
        )
        text = _truncate_to_budget(lines, _budget, header)
        return {
            "status": "success",
            "content": [{"text": text}],
        }

    @tool
    def get_knowledge_briefing() -> str:
        """Get a condensed briefing of all accumulated knowledge.

        Returns the latest rolling summary for each research angle,
        giving you a quick overview of what the system knows so far
        without reading every individual finding.

        Returns:
            Condensed knowledge briefing across all angles.
        """
        with store._lock:
            summaries = store.conn.execute(
                """SELECT angle, summary, finding_count, run_number
                   FROM knowledge_summaries
                   ORDER BY id DESC""",
            ).fetchall()

        if not summaries:
            # Fall back to angle stats if no summaries exist yet
            with store._lock:
                stats = store.conn.execute(
                    """SELECT angle, COUNT(*) as cnt,
                              AVG(confidence) as avg_conf
                       FROM conditions
                       WHERE consider_for_use = TRUE
                         AND row_type IN ('finding', 'thought', 'insight')
                         AND angle != ''
                       GROUP BY angle
                       ORDER BY cnt DESC""",
                ).fetchall()

            if not stats:
                return "(no knowledge accumulated yet)"

            lines = ["=== ANGLE OVERVIEW (no summaries yet) ==="]
            for angle, cnt, avg_conf in stats:
                lines.append(f"  {angle}: {cnt} findings, avg_conf={avg_conf:.2f}")

            _log_tool_call("get_knowledge_briefing", {}, f"{len(stats)} angles")
            return {
                "status": "success",
                "content": [{"text": _truncate_to_budget(lines, _budget)}],
            }

        # Deduplicate: keep only latest summary per angle
        seen_angles: set[str] = set()
        lines = ["=== KNOWLEDGE BRIEFING ==="]
        for angle, summary, finding_count, run_number in summaries:
            if angle in seen_angles:
                continue
            seen_angles.add(angle)
            lines.append(
                f"\n--- {angle} ({finding_count} findings, run #{run_number}) ---\n"
                f"{summary[:800]}"
            )

        _log_tool_call("get_knowledge_briefing", {}, f"{len(seen_angles)} angle summaries")
        text = _truncate_to_budget(lines, _budget)
        return {
            "status": "success",
            "content": [{"text": text}],
        }

    return [
        search_corpus,
        get_peer_insights,
        store_finding,
        check_contradictions,
        get_research_gaps,
        get_corpus_section,
        find_connections,
        get_knowledge_briefing,
    ]
