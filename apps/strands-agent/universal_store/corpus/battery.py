"""Corpus Algorithm Battery — 13-step algorithmic core for the Universal Store.

Ported from the abandoned ADK-agent (apps/adk-agent/models/corpus_store.py).
Every step is a pure function (no side effects) operating on plain dicts/lists.
The battery is consumed by swarm and flock actors via the CorpusAlgorithmBattery class.

All steps are traced via the @traced decorator for observability.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from universal_store.protocols import RowType
from universal_store.trace import traced


# ---------------------------------------------------------------------------
# Helpers (private, untraced)
# ---------------------------------------------------------------------------

def _parse_finding_id(finding: dict) -> int:
    """Extract a stable integer ID from a finding dict."""
    if "id" in finding:
        return int(finding["id"])
    # Fallback: hash the fact text for deterministic pseudo-ID
    return hash(finding.get("fact", "")) & 0x7FFFFFFF


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _signature_terms(text: str) -> set[str]:
    """Extract proper nouns, numbers, and technical terms for overlap heuristics."""
    stops = {
        "the", "this", "that", "these", "there", "their", "a", "an", "in", "it",
        "is", "are", "for", "from", "with", "has", "have", "was", "were", "will",
        "been", "some", "many", "most", "several", "according", "however",
        "although", "research", "studies", "recent", "new", "our", "its", "other",
        "such", "both", "each", "one", "two", "three", "more", "less", "much",
        "very", "while", "when", "where", "what", "which", "who", "how", "why",
        "also", "but", "yet", "nor", "not", "all", "any", "no", "only", "so",
        "too", "than", "they", "we", "he", "she", "you", "may", "can", "would",
        "could", "should", "must", "shall", "might",
    }
    terms: set[str] = set()
    for word in text.split():
        clean = word.strip(".,;:!?\"'()[]")
        if not clean:
            continue
        low = clean.lower()
        if low in stops:
            continue
        if clean[0].isupper() and len(clean) > 1:
            terms.add(low)
        elif any(c.isdigit() for c in clean):
            terms.add(low)
        elif "-" in clean and len(clean) > 5:
            terms.add(low)
    return terms


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-like overlap between two texts (0.0–1.0)."""
    ta = set(_normalize_text(a).split())
    tb = set(_normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Best-effort ISO8601 parser."""
    if not ts:
        return None
    try:
        # Replace Z with +00:00 for fromisoformat
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# CorpusAlgorithmBattery
# ---------------------------------------------------------------------------

class CorpusAlgorithmBattery:
    """13-step algorithmic battery for scoring, clustering, contradiction
    detection, and quality assessment.

    Every method is a pure function: it receives plain Python data structures
    and returns plain Python data structures.  No database side effects.
    """

    # -- 1. score_finding ---------------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="score_finding")
    async def score_finding(finding: dict) -> dict:
        """Compute a composite quality score from six gradient dimensions.

        Dimensions (all 0.0–1.0, read from *finding*):
            confidence, novelty, specificity, relevance, actionability,
            fabrication_risk (inverted — higher risk reduces score).

        Returns a new dict with the original keys plus:
            ``composite_quality`` (float 0.0–1.0)
            ``quality_tier`` ("excellent" | "good" | "fair" | "poor")
        """
        confidence = float(finding.get("confidence", 0.5))
        novelty = float(finding.get("novelty_score", finding.get("novelty", 0.5)))
        specificity = float(finding.get("specificity_score", finding.get("specificity", 0.5)))
        relevance = float(finding.get("relevance_score", finding.get("relevance", 0.5)))
        actionability = float(finding.get("actionability_score", finding.get("actionability", 0.5)))
        fabrication_risk = float(finding.get("fabrication_risk", 0.0))

        # Weights derived from ADK composite_quality formula
        composite = (
            confidence * 0.20
            + novelty * 0.15
            + specificity * 0.20
            + relevance * 0.20
            + actionability * 0.15
            + (1.0 - fabrication_risk) * 0.10
        )
        composite = max(0.0, min(1.0, composite))

        if composite >= 0.75:
            tier = "excellent"
        elif composite >= 0.55:
            tier = "good"
        elif composite >= 0.35:
            tier = "fair"
        else:
            tier = "poor"

        result = dict(finding)
        result["composite_quality"] = round(composite, 4)
        result["quality_tier"] = tier
        return result

    # -- 2. detect_contradictions -------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="detect_contradictions")
    async def detect_contradictions(findings: list[dict]) -> list[tuple[int, int, float]]:
        """Pairwise contradiction detection using lightweight heuristics.

        Two findings are contradiction *candidates* when:
        - They share significant signature-term overlap (same topic)
        - Their confidence scores differ by > 0.3 (one strong, one weak)

        Severity is a weighted product of topic overlap and confidence gap.

        Returns:
            List of (id_a, id_b, severity) tuples, severity in (0.0, 1.0].
        """
        contradictions: list[tuple[int, int, float]] = []
        n = len(findings)
        if n < 2:
            return contradictions

        indexed = [
            {
                "id": _parse_finding_id(f),
                "fact": f.get("fact", ""),
                "confidence": float(f.get("confidence", 0.5)),
                "terms": _signature_terms(f.get("fact", "")),
            }
            for f in findings
        ]

        for i in range(n):
            a = indexed[i]
            for j in range(i + 1, n):
                b = indexed[j]
                if not a["terms"] or not b["terms"]:
                    continue
                shared = len(a["terms"] & b["terms"])
                min_terms = min(len(a["terms"]), len(b["terms"]))
                if min_terms == 0:
                    continue
                overlap = shared / min_terms
                # Need some topical overlap to be contradiction candidates
                if overlap < 0.15:
                    continue
                conf_gap = abs(a["confidence"] - b["confidence"])
                if conf_gap < 0.25:
                    continue
                # Severity: high overlap + high confidence gap = strong contradiction
                severity = overlap * conf_gap
                if severity > 0.15:
                    contradictions.append((a["id"], b["id"], round(severity, 4)))

        # Deduplicate and sort by severity descending
        seen: set[tuple[int, int]] = set()
        deduped: list[tuple[int, int, float]] = []
        for id_a, id_b, sev in sorted(contradictions, key=lambda x: x[2], reverse=True):
            key = (min(id_a, id_b), max(id_a, id_b))
            if key not in seen:
                seen.add(key)
                deduped.append((key[0], key[1], sev))
        return deduped

    # -- 3. union_find_clustering -------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="union_find_clustering")
    async def union_find_clustering(findings: list[dict]) -> dict[int, int]:
        """Semantic similarity clustering via Union-Find.

        Clusters are formed when findings share >= 2 signature terms
        OR have a token Jaccard overlap > 0.5.

        Returns:
            Mapping {finding_id: cluster_id} where cluster_id is the
            root representative of that finding's cluster.
        """
        if not findings:
            return {}

        indexed = [
            {
                "id": _parse_finding_id(f),
                "fact": f.get("fact", ""),
                "terms": _signature_terms(f.get("fact", "")),
            }
            for f in findings
        ]

        parent: dict[int, int] = {item["id"]: item["id"] for item in indexed}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        n = len(indexed)
        for i in range(n):
            a = indexed[i]
            for j in range(i + 1, n):
                b = indexed[j]
                linked = False
                if a["terms"] and b["terms"]:
                    shared = len(a["terms"] & b["terms"])
                    if shared >= 2:
                        linked = True
                if not linked:
                    overlap = _token_overlap(a["fact"], b["fact"])
                    if overlap > 0.5:
                        linked = True
                if linked:
                    union(a["id"], b["id"])

        # Compress paths and assign canonical cluster IDs
        root_to_cluster: dict[int, int] = {}
        result: dict[int, int] = {}
        for item in indexed:
            fid = item["id"]
            root = find(fid)
            if root not in root_to_cluster:
                root_to_cluster[root] = len(root_to_cluster)
            result[fid] = root_to_cluster[root]
        return result

    # -- 4. build_narrative_chains ------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="build_narrative_chains")
    async def build_narrative_chains(findings: list[dict]) -> list[list[int]]:
        """Build temporal/causal chains from parent_id / related_id edges.

        A chain is a directed path of length >= 2.  Roots are nodes with
        outgoing edges but no incoming edges.  Cycles are broken by
        disallowing revisits within a single path.

        Returns:
            List of chains, each chain is a list of finding IDs in order.
        """
        if not findings:
            return []

        # Build adjacency list from parent_id / related_id
        graph: dict[int, list[int]] = {}
        all_targets: set[int] = set()
        id_set: set[int] = set()

        for f in findings:
            fid = _parse_finding_id(f)
            id_set.add(fid)
            parent_id = f.get("parent_id")
            related_id = f.get("related_id")
            if parent_id is not None and int(parent_id) in id_set:
                src = int(parent_id)
                graph.setdefault(src, []).append(fid)
                all_targets.add(fid)
            if related_id is not None and int(related_id) in id_set:
                src = int(related_id)
                graph.setdefault(src, []).append(fid)
                all_targets.add(fid)

        # Also build edges from explicit row_type hints (causal_link, temporal)
        for f in findings:
            fid = _parse_finding_id(f)
            row_type = f.get("row_type", "")
            if row_type in ("causal_link", "temporal", "supporting", RowType.CONNECTION.value):
                parent_id = f.get("parent_id")
                if parent_id is not None and int(parent_id) in id_set:
                    src = int(parent_id)
                    if fid not in graph.get(src, []):
                        graph.setdefault(src, []).append(fid)
                        all_targets.add(fid)

        roots = set(graph.keys()) - all_targets
        if not roots:
            roots = set(graph.keys())

        chains: list[list[int]] = []
        visited_global: set[int] = set()

        for root in roots:
            if root in visited_global:
                continue
            stack: list[tuple[int, list[int]]] = [(root, [root])]
            while stack:
                node, path = stack.pop()
                extended = False
                for tgt in graph.get(node, []):
                    if tgt not in path:
                        stack.append((tgt, path + [tgt]))
                        extended = True
                if not extended and len(path) >= 2:
                    chains.append(path)
                    visited_global.update(path)

        # Deduplicate subset chains (prefer longer)
        chains.sort(key=len, reverse=True)
        covered: set[int] = set()
        final: list[list[int]] = []
        for chain in chains:
            if not covered.issuperset(chain):
                final.append(chain)
                covered.update(chain)
        return final

    # -- 5. compress_redundancy ---------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="compress_redundancy")
    async def compress_redundancy(findings: list[dict]) -> list[int]:
        """Identify redundant findings that should be deprioritized.

        Redundancy is detected via high token overlap (> 0.85) OR high
        duplication_score on the finding dict itself.

        Returns:
            List of finding IDs to deprioritize (keep the higher-confidence
            representative of each redundant group).
        """
        if not findings:
            return []

        indexed = [
            {
                "id": _parse_finding_id(f),
                "fact": f.get("fact", ""),
                "confidence": float(f.get("confidence", 0.5)),
                "dup_score": float(f.get("duplication_score", -1.0)),
            }
            for f in findings
        ]

        n = len(indexed)
        redundant: set[int] = set()
        grouped: set[tuple[int, int]] = set()

        for i in range(n):
            a = indexed[i]
            if a["id"] in redundant:
                continue
            for j in range(i + 1, n):
                b = indexed[j]
                if b["id"] in redundant:
                    continue
                pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
                if pair in grouped:
                    continue
                is_dup = False
                if a["dup_score"] > 0.85 or b["dup_score"] > 0.85:
                    is_dup = True
                elif _token_overlap(a["fact"], b["fact"]) > 0.85:
                    is_dup = True
                if is_dup:
                    grouped.add(pair)
                    # Deprioritize the lower-confidence one
                    if a["confidence"] >= b["confidence"]:
                        redundant.add(b["id"])
                    else:
                        redundant.add(a["id"])

        return sorted(redundant)

    # -- 6. quality_gate ----------------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="quality_gate")
    async def quality_gate(
        findings: list[dict],
        threshold: float = 0.25,
    ) -> list[dict]:
        """Filter findings below a minimum composite quality threshold.

        If a finding does not yet have ``composite_quality``, it is scored
        inline via :meth:`score_finding`.

        Returns:
            Findings with ``composite_quality`` >= *threshold*.
        """
        passed: list[dict] = []
        for f in findings:
            scored = f if "composite_quality" in f else await CorpusAlgorithmBattery.score_finding(f)
            if scored.get("composite_quality", 0.0) >= threshold:
                passed.append(scored)
        return passed

    # -- 7. evaluate_evidentiary_strength -----------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="evaluate_evidentiary_strength")
    async def evaluate_evidentiary_strength(finding: dict) -> float:
        """Assess how well-supported a single finding is.

        Factors:
            - Has source URL (+0.25)
            - Has named source_ref / source_type (+0.15)
            - Confidence > 0.6 (+0.20)
            - Fabrication risk < 0.3 (+0.20)
            - Specificity > 0.5 (+0.20)

        Returns:
            Evidentiary strength in [0.0, 1.0].
        """
        score = 0.0
        if finding.get("source_url"):
            score += 0.25
        if finding.get("source_ref") or finding.get("source_type"):
            score += 0.15
        if float(finding.get("confidence", 0.0)) > 0.6:
            score += 0.20
        if float(finding.get("fabrication_risk", 1.0)) < 0.3:
            score += 0.20
        if float(finding.get("specificity_score", finding.get("specificity", 0.0))) > 0.5:
            score += 0.20
        return round(min(1.0, score), 4)

    # -- 8. detect_methodological_bias --------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="detect_methodological_bias")
    async def detect_methodological_bias(findings: list[dict]) -> list[dict]:
        """Flag findings that cluster by the same method/source_type.

        A method is considered over-represented when it accounts for > 50%
        of all findings.  Every finding from that method gets a
        ``methodological_bias_flag`` added.

        Returns:
            Copy of *findings* with ``methodological_bias_flag`` (bool) and
            ``method_bias_ratio`` (float) keys added.
        """
        if not findings:
            return []

        total = len(findings)
        method_counts: dict[str, int] = {}
        for f in findings:
            method = f.get("source_type", "") or f.get("strategy", "") or "unknown"
            method_counts[method] = method_counts.get(method, 0) + 1

        dominant_method = ""
        dominant_count = 0
        for m, c in method_counts.items():
            if c > dominant_count:
                dominant_count = c
                dominant_method = m

        ratio = dominant_count / total if total else 0.0
        flagged = []
        for f in findings:
            method = f.get("source_type", "") or f.get("strategy", "") or "unknown"
            copy = dict(f)
            copy["methodological_bias_flag"] = (method == dominant_method) and (ratio > 0.5)
            copy["method_bias_ratio"] = round(ratio, 4)
            flagged.append(copy)
        return flagged

    # -- 9. temporal_supersession -------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="temporal_supersession")
    async def temporal_supersession(findings: list[dict]) -> list[tuple[int, int]]:
        """Find newer evidence that overrides older evidence on the same topic.

        Two findings are supersession candidates when:
        - They share >= 2 signature terms (same topic)
        - One has a parseable ``created_at`` that is > 1 day newer
        - The newer finding has higher confidence

        Returns:
            List of (older_id, newer_id) tuples.
        """
        pairs: list[tuple[int, int]] = []
        if not findings:
            return pairs

        indexed = [
            {
                "id": _parse_finding_id(f),
                "fact": f.get("fact", ""),
                "confidence": float(f.get("confidence", 0.5)),
                "terms": _signature_terms(f.get("fact", "")),
                "ts": _parse_iso_timestamp(f.get("created_at")),
            }
            for f in findings
        ]

        n = len(indexed)
        for i in range(n):
            a = indexed[i]
            for j in range(i + 1, n):
                b = indexed[j]
                if not a["terms"] or not b["terms"]:
                    continue
                if len(a["terms"] & b["terms"]) < 2:
                    continue
                if a["ts"] is None or b["ts"] is None:
                    continue
                delta = (b["ts"] - a["ts"]).total_seconds()
                if delta > 86400 and b["confidence"] > a["confidence"]:
                    pairs.append((a["id"], b["id"]))
                elif delta < -86400 and a["confidence"] > b["confidence"]:
                    pairs.append((b["id"], a["id"]))
        return pairs

    # -- 10. cross_angle_validation -----------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="cross_angle_validation")
    async def cross_angle_validation(findings: list[dict]) -> dict[int, float]:
        """Validate findings by checking cross-angle confirmation.

        A finding receives a higher validation score when other findings
        from *different* angles share significant signature-term overlap
        with it.  This measures independent corroboration.

        Returns:
            Mapping {finding_id: validation_score} where score is in [0.0, 1.0].
        """
        if not findings:
            return {}

        indexed = [
            {
                "id": _parse_finding_id(f),
                "angle": f.get("angle", ""),
                "terms": _signature_terms(f.get("fact", "")),
            }
            for f in findings
        ]

        result: dict[int, float] = {}
        n = len(indexed)
        for i in range(n):
            a = indexed[i]
            confirming_angles: set[str] = set()
            for j in range(n):
                if i == j:
                    continue
                b = indexed[j]
                if not a["terms"] or not b["terms"]:
                    continue
                if a["angle"] == b["angle"]:
                    continue
                shared = len(a["terms"] & b["terms"])
                min_terms = min(len(a["terms"]), len(b["terms"]))
                if min_terms and shared / min_terms > 0.3:
                    confirming_angles.add(b["angle"])
            # Score saturates at ~5 confirming angles
            score = min(1.0, len(confirming_angles) * 0.2)
            result[a["id"]] = round(score, 4)
        return result

    # -- 11. information_gain_tracking --------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="information_gain_tracking")
    async def information_gain_tracking(history: list[dict]) -> float:
        """Compute cumulative information gain across a history of events.

        Each item in *history* should have at least:
            ``trigram_pool`` (set of str) or ``text`` (str) — the content
            ``information_gain`` (float) — optional pre-computed gain

        Falls back to set-difference of trigram pools between consecutive items.

        Returns:
            Cumulative information gain in [0.0, 1.0].
        """
        if not history:
            return 0.0
        if len(history) == 1:
            return float(history[0].get("information_gain", 0.5))

        def _trigrams(text: str) -> set[str]:
            words = text.lower().split()
            if len(words) < 3:
                return set(words)
            return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}

        gains: list[float] = []
        prev_pool: set[str] | None = None
        for item in history:
            if "information_gain" in item:
                gains.append(float(item["information_gain"]))
                continue
            text = item.get("text", item.get("fact", ""))
            pool = _trigrams(text)
            if prev_pool is not None:
                new_trigrams = pool - prev_pool
                gain = len(new_trigrams) / len(pool) if pool else 0.0
                gains.append(gain)
            else:
                gains.append(0.5)  # first item baseline
            prev_pool = pool

        # Cumulative: average gain, decayed by count to avoid inflation
        if not gains:
            return 0.0
        avg_gain = sum(gains) / len(gains)
        # Diminishing-returns penalty for very long histories
        decay = 1.0 / math.sqrt(len(gains))
        return round(avg_gain * decay, 4)

    # -- 12. convergence_detection ------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="convergence_detection")
    async def convergence_detection(
        score_history: list[float],
        window_size: int = 3,
        threshold: float = 0.02,
    ) -> tuple[bool, str]:
        """Detect whether the system has converged from a score history.

        Uses a rolling coefficient of variation (CV = std / mean) over the
        last *window_size* scores.  When CV drops below *threshold*, the
        system is considered converged.

        Args:
            score_history: Chronological list of scores (e.g. coverage scores).
            window_size: Number of recent scores to evaluate.
            threshold: CV threshold for convergence.

        Returns:
            (converged: bool, reason: str)
        """
        if len(score_history) < window_size:
            return False, "insufficient_history"

        recent = score_history[-window_size:]
        mean = sum(recent) / len(recent)
        if mean == 0.0:
            return False, "zero_mean"

        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        std = math.sqrt(variance)
        cv = std / mean

        if cv < threshold:
            return True, f"cv={cv:.4f}_below_threshold"

        # Additional check: absolute change in last step is tiny
        if len(score_history) >= 2:
            last_delta = abs(score_history[-1] - score_history[-2])
            if last_delta < threshold * 0.5:
                return True, f"last_delta={last_delta:.4f}_negligible"

        return False, f"cv={cv:.4f}_still_varying"

    # -- 13. synthesis_readiness --------------------------------------------

    @staticmethod
    @traced(actor_id="CorpusAlgorithmBattery", event_type="synthesis_readiness")
    async def synthesis_readiness(findings: list[dict]) -> bool:
        """Determine whether the corpus is ready for diffusion synthesis.

        Criteria (all must pass):
            - At least 3 findings total
            - At least 50% of findings have composite_quality >= 0.4
            - No more than 30% of findings are flagged with contradiction
            - At least 2 distinct angles represented
            - Average confidence >= 0.4

        Returns:
            True if findings are ready for synthesis.
        """
        if not findings:
            return False
        if len(findings) < 3:
            return False

        # Ensure scoring is present
        scored = []
        for f in findings:
            # Skip non-finding rows from readiness calculation
            if f.get("row_type", RowType.FINDING.value) != RowType.FINDING.value:
                continue
            if "composite_quality" in f:
                scored.append(f)
            else:
                scored.append(await CorpusAlgorithmBattery.score_finding(f))

        if len(scored) < 3:
            return False

        quality_ok = sum(
            1 for s in scored if s.get("composite_quality", 0.0) >= 0.4
        ) / len(scored)
        if quality_ok < 0.5:
            return False

        contradiction_ratio = sum(
            1 for s in scored if s.get("contradiction_flag", False)
        ) / len(scored)
        if contradiction_ratio > 0.30:
            return False

        angles = {s.get("angle", "") for s in scored if s.get("angle")}
        if len(angles) < 2:
            return False

        avg_conf = sum(float(s.get("confidence", 0.0)) for s in scored) / len(scored)
        if avg_conf < 0.4:
            return False

        return True
