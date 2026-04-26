"""Semantic Connection Worker — 3-stage pipeline for typed relationship detection.

The ``SemanticConnectionWorker`` receives ``StoreDelta`` events, generates
heuristic candidate pairs, gates them via embedding cosine similarity, and
verifies the survivors with a batched LLM placeholder. Verified connections
are persisted to ``semantic_connections`` and emitted as
``ConnectionDetected`` events.
"""
from __future__ import annotations

import asyncio
import math
import random
import re
from datetime import datetime
from typing import Any

from universal_store.actors.base import Actor
from universal_store.config import UnifiedConfig
from universal_store.protocols import (
    ConnectionDetected,
    ConnectionType,
    Event,
    StoreDelta,
    StoreProtocol,
)
from universal_store.trace import TraceStore, trace_block

# Shared system prompt prefix for vLLM prefix caching.
_SYSTEM_PROMPT_PREFIX = (
    "You are a semantic relationship verifier. Analyze pairs of research findings "
    "and classify their relationship using one of the supported connection types. "
    "Respond with JSON containing 'connection_type', 'confidence', and 'explanation'."
)


class SemanticConnectionWorker(Actor):
    """Actor that runs a 3-stage pipeline to detect verified, typed relationships.

    Parameters
    ----------
    actor_id:
        Unique identifier for this actor.
    store:
        A ``StoreProtocol`` implementation (e.g. DuckDB-backed store).
    config:
        ``UnifiedConfig`` instance; falls back to ``from_env()`` if omitted.
    """

    def __init__(
        self,
        actor_id: str,
        store: StoreProtocol,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig.from_env()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        """Main loop: react to ``StoreDelta`` events and run the pipeline."""
        trace = await TraceStore.get()
        await trace.record(
            actor_id=self.actor_id,
            event_type="semantic_worker_started",
        )

        while not self._shutdown:
            try:
                event = await asyncio.wait_for(self.mailbox.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if event.event_type == "StoreDelta":
                row_types = event.payload.get("row_types", [])
                if "finding" in row_types:
                    async with trace_block(
                        self.actor_id,
                        "semantic_pipeline",
                        "",
                        {"rows_added": event.payload.get("rows_added", 0)},
                    ):
                        await self._process_pipeline(event)
            else:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="semantic_worker_ignored_event",
                    payload={"event_type": event.event_type},
                )

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    async def _process_pipeline(self, event: StoreDelta) -> None:
        """Execute heuristic filter → embedding gate → LLM verify → store & emit."""
        trace = await TraceStore.get()

        # Stage 1 — Heuristic filter
        async with trace_block(self.actor_id, "heuristic_filter_stage", ""):
            candidates = await self.heuristic_filter(
                self.store,
                max_candidates=self.config.semantic.heuristic_max_candidates,
            )
            await trace.record(
                actor_id=self.actor_id,
                event_type="heuristic_filter_complete",
                payload={"candidates": len(candidates)},
            )

        if not candidates:
            await trace.record(
                actor_id=self.actor_id,
                event_type="semantic_pipeline_no_candidates",
            )
            return

        # Stage 2 — Embedding gate (MUST NOT be skipped)
        async with trace_block(self.actor_id, "embedding_gate_stage", ""):
            gated = await self.embedding_gate(
                candidates,
                threshold=self.config.semantic.embedding_threshold,
            )
            await trace.record(
                actor_id=self.actor_id,
                event_type="embedding_gate_complete",
                payload={"gated": len(gated)},
            )

        if not gated:
            await trace.record(
                actor_id=self.actor_id,
                event_type="semantic_pipeline_no_gated_candidates",
            )
            return

        # Stage 3 — LLM verification (placeholder)
        async with trace_block(self.actor_id, "llm_verify_stage", ""):
            verified = await self.llm_verify_batch(
                gated,
                batch_size=self.config.semantic.llm_verification_batch_size,
            )
            await trace.record(
                actor_id=self.actor_id,
                event_type="llm_verify_complete",
                payload={"verified": len(verified)},
            )

        if not verified:
            return

        await self._store_and_emit(verified)

    # ------------------------------------------------------------------
    # Stage 1: Heuristic filter
    # ------------------------------------------------------------------

    async def heuristic_filter(
        self,
        store: StoreProtocol,
        max_candidates: int = 1000,
    ) -> list[tuple[int, int]]:
        """Generate candidate pairs using lightweight heuristics.

        Signals considered per pair:
        - Term overlap (Jaccard similarity > 0.2)
        - Cluster membership (same ``cluster_id``, different ``angle``)
        - Shared ``source_url`` or ``source_type``
        - Temporal proximity (within 24 hours)

        Feedback-loop guard:
        - Pairs whose findings originate from the same ``diffusion_pass`` are
          excluded.

        Returns the top *max_candidates* pairs ordered by a composite score.
        """
        trace = await TraceStore.get()

        sql = """
            SELECT id, fact, angle, cluster_id, source_url, source_type, created_at, diffusion_pass
            FROM conditions
            WHERE consider_for_use = TRUE
              AND row_type = 'finding'
            ORDER BY id DESC
            LIMIT 5000
        """
        rows = await self._store_query(sql)
        if len(rows) < 2:
            return []

        # Pre-compute term sets and build inverted index
        findings: list[dict[str, Any]] = []
        term_index: dict[str, list[int]] = {}
        for row in rows:
            fid = row["id"]
            fact = row.get("fact") or ""
            terms = set(re.findall(r"[a-z0-9]{3,}", fact.lower()))
            findings.append({**row, "_terms": terms})
            for term in terms:
                term_index.setdefault(term, []).append(fid)

        # Bucket by cluster / source / time to keep the search space tractable
        cluster_buckets: dict[int, list[int]] = {}
        source_url_buckets: dict[str, list[int]] = {}
        source_type_buckets: dict[str, list[int]] = {}
        time_buckets: dict[str, list[int]] = {}

        for f in findings:
            fid = f["id"]
            if f.get("cluster_id", -1) != -1:
                cluster_buckets.setdefault(f["cluster_id"], []).append(fid)
            if f.get("source_url"):
                source_url_buckets.setdefault(f["source_url"], []).append(fid)
            if f.get("source_type"):
                source_type_buckets.setdefault(f["source_type"], []).append(fid)
            created = f.get("created_at", "")
            if created:
                time_buckets.setdefault(created[:10], []).append(fid)

        seen: set[tuple[int, int]] = set()
        candidate_pairs: list[tuple[int, int]] = []

        def _add_pair(a: int, b: int) -> None:
            if a == b:
                return
            pair = (a, b) if a < b else (b, a)
            if pair in seen:
                return
            seen.add(pair)
            candidate_pairs.append(pair)

        # Pairs from buckets
        for bucket in (
            list(cluster_buckets.values())
            + list(source_url_buckets.values())
            + list(source_type_buckets.values())
            + list(time_buckets.values())
        ):
            ids = sorted(bucket)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    _add_pair(ids[i], ids[j])

        # Pairs from term overlap (catches semantic similarity not in buckets)
        for f in findings:
            fid = f["id"]
            neighbors: set[int] = set()
            for term in f["_terms"]:
                neighbors.update(term_index.get(term, []))
            for nid in neighbors:
                if nid != fid:
                    _add_pair(fid, nid)

        id_to_finding = {f["id"]: f for f in findings}
        scored: list[tuple[float, int, int]] = []

        for a, b in candidate_pairs:
            fa = id_to_finding.get(a)
            fb = id_to_finding.get(b)
            if not fa or not fb:
                continue

            # Feedback-loop guard: same diffusion pass
            dp_a = fa.get("diffusion_pass")
            dp_b = fb.get("diffusion_pass")
            if dp_a is not None and dp_b is not None and dp_a == dp_b and dp_a not in (0, ""):
                continue

            score = 0.0

            # 1. Jaccard term overlap
            terms_a = fa["_terms"]
            terms_b = fb["_terms"]
            union = terms_a | terms_b
            if union:
                jaccard = len(terms_a & terms_b) / len(union)
                if jaccard > 0.2:
                    score += jaccard * 2.0

            # 2. Cluster membership (same cluster, different angle)
            if (
                fa.get("cluster_id", -1) != -1
                and fa.get("cluster_id") == fb.get("cluster_id")
                and fa.get("angle") != fb.get("angle")
            ):
                score += 1.5

            # 3. Shared source
            if fa.get("source_url") and fa.get("source_url") == fb.get("source_url"):
                score += 1.0
            elif fa.get("source_type") and fa.get("source_type") == fb.get("source_type"):
                score += 0.5

            # 4. Temporal proximity
            ca = fa.get("created_at", "")
            cb = fb.get("created_at", "")
            if ca and cb:
                try:
                    ta = datetime.fromisoformat(ca.replace("Z", "+00:00"))
                    tb = datetime.fromisoformat(cb.replace("Z", "+00:00"))
                    diff_h = abs((ta - tb).total_seconds()) / 3600.0
                    if diff_h < 24.0:
                        score += 1.0 - (diff_h / 24.0)
                except Exception:
                    pass

            if score > 0.0:
                scored.append((score, a, b))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_candidates]

        await trace.record(
            actor_id=self.actor_id,
            event_type="heuristic_filter_scored",
            payload={
                "pairs_considered": len(candidate_pairs),
                "scored_pairs": len(scored),
                "returned": len(top),
            },
        )

        return [(a, b) for _, a, b in top]

    # ------------------------------------------------------------------
    # Stage 2: Embedding gate
    # ------------------------------------------------------------------

    async def embedding_gate(
        self,
        candidates: list[tuple[int, int]],
        threshold: float = 0.72,
    ) -> list[tuple[int, int]]:
        """Filter candidate pairs by cosine similarity of their embeddings.

        Embeddings are read from ``condition_embeddings``. Pairs that share the
        same ``angle`` face an elevated threshold (``threshold + 0.10``).
        """
        trace = await TraceStore.get()
        if not candidates:
            return []

        all_ids = list({i for pair in candidates for i in pair})
        id_csv = ",".join(str(i) for i in all_ids)

        # Load embeddings
        emb_rows = await self._store_query(
            f"SELECT condition_id, embedding FROM condition_embeddings WHERE condition_id IN ({id_csv})"
        )
        embeddings: dict[int, list[float]] = {
            r["condition_id"]: r["embedding"] for r in emb_rows
        }

        # Load angles for same-angle boost
        angle_rows = await self._store_query(
            f"SELECT id, angle FROM conditions WHERE id IN ({id_csv})"
        )
        angles: dict[int, str] = {r["id"]: r.get("angle", "") or "" for r in angle_rows}

        filtered: list[tuple[int, int]] = []
        for a, b in candidates:
            emb_a = embeddings.get(a)
            emb_b = embeddings.get(b)
            if not emb_a or not emb_b:
                continue

            sim = self._cosine_similarity(emb_a, emb_b)
            angle_a = angles.get(a, "")
            angle_b = angles.get(b, "")

            effective_threshold = threshold
            if angle_a and angle_a == angle_b:
                effective_threshold = threshold + self.config.semantic.embedding_same_angle_boost

            if sim >= effective_threshold:
                filtered.append((a, b))

        await trace.record(
            actor_id=self.actor_id,
            event_type="embedding_gate_result",
            payload={
                "input": len(candidates),
                "passed": len(filtered),
                "threshold": threshold,
            },
        )

        return filtered

    # ------------------------------------------------------------------
    # Stage 3: LLM verification (placeholder)
    # ------------------------------------------------------------------

    async def llm_verify_batch(
        self,
        candidates: list[tuple[int, int]],
        batch_size: int = 32,
    ) -> list[dict]:
        """Batched LLM verification placeholder.

        Each returned dict contains:
        ``source_id``, ``target_id``, ``connection_type``, ``confidence``,
        ``explanation``.

        The shared ``_SYSTEM_PROMPT_PREFIX`` is prepended to every batch to
        demonstrate vLLM prefix caching.
        """
        trace = await TraceStore.get()
        if not candidates:
            return []

        all_ids = list({i for pair in candidates for i in pair})
        id_csv = ",".join(str(i) for i in all_ids)
        fact_rows = await self._store_query(
            f"SELECT id, fact FROM conditions WHERE id IN ({id_csv})"
        )
        facts: dict[int, str] = {r["id"]: r.get("fact", "") or "" for r in fact_rows}

        verified: list[dict] = []
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start : batch_start + batch_size]
            batch_idx = batch_start // batch_size

            async with trace_block(
                self.actor_id,
                "llm_verify_batch",
                "",
                {"batch_index": batch_idx, "batch_size": len(batch)},
            ):
                batch_input = [
                    {
                        "source_id": a,
                        "target_id": b,
                        "source_fact": facts.get(a, ""),
                        "target_fact": facts.get(b, ""),
                    }
                    for a, b in batch
                ]

                results = await self._placeholder_llm_verify(batch_input)

                for item, result in zip(batch_input, results):
                    verified.append({
                        "source_id": item["source_id"],
                        "target_id": item["target_id"],
                        "connection_type": result.get("connection_type", ConnectionType.SUPPORTING),
                        "confidence": float(result.get("confidence", 0.0)),
                        "explanation": result.get("explanation", ""),
                    })

                await trace.record(
                    actor_id=self.actor_id,
                    event_type="llm_verify_batch_done",
                    payload={
                        "batch_index": batch_idx,
                        "batch_size": len(batch),
                        "verdicts": len(results),
                    },
                )

        return verified

    async def _placeholder_llm_verify(
        self,
        batch: list[dict[str, Any]],
        system_prompt: str = _SYSTEM_PROMPT_PREFIX,
    ) -> list[dict[str, Any]]:
        """Placeholder LLM verifier — never calls a real API.

        The *system_prompt* argument demonstrates shared-prefix caching for
        vLLM-compatible inference stacks.
        """
        # Build a fake prompt that starts with the shared prefix
        _ = system_prompt + "\n\n" + "\n".join(
            f"Pair {i+1}: [{item['source_id']}] {item['source_fact'][:120]} → "
            f"[{item['target_id']}] {item['target_fact'][:120]}"
            for i, item in enumerate(batch)
        )

        await asyncio.sleep(0.01)
        return [
            {
                "connection_type": random.choice(list(ConnectionType)),
                "confidence": round(random.uniform(0.65, 0.95), 3),
                "explanation": (
                    f"Placeholder verdict for {item['source_id']} → {item['target_id']}: "
                    f"semantic relationship detected via shared context."
                ),
            }
            for item in batch
        ]

    # ------------------------------------------------------------------
    # Persistence & event emission
    # ------------------------------------------------------------------

    async def _store_and_emit(self, verified: list[dict]) -> None:
        """Persist verified connections and emit ``ConnectionDetected`` events."""
        trace = await TraceStore.get()
        min_confidence = self.config.semantic.min_confidence_for_storage
        run_number = await self._get_current_run_number()

        stored_count = 0
        emitted_count = 0

        for verdict in verified:
            confidence = float(verdict.get("confidence", 0.0))
            if confidence < min_confidence:
                continue

            source_id = verdict["source_id"]
            target_id = verdict["target_id"]
            connection_type = str(verdict.get("connection_type", ConnectionType.SUPPORTING))

            row = {
                "source_condition_id": source_id,
                "target_condition_id": target_id,
                "connection_type": connection_type,
                "directionality": "symmetric",
                "detection_stage": "llm_verified",
                "confidence": confidence,
                "evidence_text": str(verdict.get("explanation", ""))[:2000],
                "evaluator_angle": "semantic_worker",
                "run_number": run_number,
                "embedding_similarity": None,
                "llm_verdict_json": str(verdict),
            }

            try:
                await self._store_insert("semantic_connections", row)
                stored_count += 1
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="semantic_connection_stored",
                    payload={
                        "source_id": source_id,
                        "target_id": target_id,
                        "connection_type": connection_type,
                        "confidence": confidence,
                    },
                )
            except Exception as exc:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="semantic_connection_store_failed",
                    error=exc,
                    payload={"source_id": source_id, "target_id": target_id},
                )
                continue

            # Emit event for high-confidence connections
            if confidence > min_confidence:
                evt = ConnectionDetected(
                    source_id=source_id,
                    target_id=target_id,
                    connection_type=connection_type,
                    confidence=confidence,
                )
                await self.send_to_parent(evt)
                emitted_count += 1
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="connection_detected_emitted",
                    payload={
                        "source_id": source_id,
                        "target_id": target_id,
                        "connection_type": connection_type,
                        "confidence": confidence,
                    },
                )

        await trace.record(
            actor_id=self.actor_id,
            event_type="semantic_pipeline_finished",
            payload={"stored": stored_count, "emitted": emitted_count},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two equal-length vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _get_current_run_number(self) -> int:
        """Return the latest ``run_number`` from the ``runs`` table, or ``0``."""
        try:
            rows = await self._store_query(
                "SELECT COALESCE(MAX(run_number), 0) AS run_number FROM runs"
            )
            if rows:
                return int(rows[0].get("run_number", 0))
        except Exception:
            pass
        return 0

    # Normalise sync / async store access so the worker is runnable with either
    async def _store_query(self, sql: str, params: tuple | None = None) -> list[dict]:
        fn = self.store.query
        if asyncio.iscoroutinefunction(fn):
            return await fn(sql, params)
        return await asyncio.to_thread(fn, sql, params)

    async def _store_execute(self, sql: str, params: tuple | None = None) -> list[dict]:
        fn = self.store.execute
        if asyncio.iscoroutinefunction(fn):
            return await fn(sql, params)
        return await asyncio.to_thread(fn, sql, params)

    async def _store_insert(self, table: str, row: dict) -> int:
        fn = self.store.insert
        if asyncio.iscoroutinefunction(fn):
            return await fn(table, row)
        return await asyncio.to_thread(fn, table, row)
