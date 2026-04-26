"""Source ingestion pipeline — selection → download → extraction → chunk + embed.

The ``SourceIngestionActor`` is a ``Supervisor`` that manages four specialist
workers:

* ``SelectionWorker`` – queries Anna's Archive (placeholder), ranks by metadata
  score, and forwards the top-N candidates.
* ``DownloadWorker`` – streams downloads with resume support, computes
  byte-level and text-level fingerprints, and enforces three-strikes blocking.
* ``ExtractionWorker`` – extracts text from PDF/EPUB placeholders, runs a
  quality gate, and OCR fallback logic.
* ``ChunkEmbedWorker`` – performs hybrid chunking (structural → fixed-size →
  semantic), generates placeholder embeddings, and persists rows in the
  ``chunks`` table.

Every stage emits trace records.  No entire book is loaded into memory;
all file operations are chunked or streamed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from universal_store.actors.base import Actor
from universal_store.actors.supervisor import Supervisor
from universal_store.config import UnifiedConfig
from universal_store.protocols import Event, StoreDelta, StoreProtocol
from universal_store.trace import TraceStore, trace_block


# ---------------------------------------------------------------------------
# Placeholders for external systems
# ---------------------------------------------------------------------------

async def _query_annas_archive_api(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Placeholder for Anna's Archive API.

    Returns a list of synthetic book metadata dictionaries.  In production this
    would hit the real API, parse results, and return normalized records.
    """
    await asyncio.sleep(0.01)
    return [
        {
            "source_url": f"https://annas-archive.org/placeholder/{query.replace(' ', '_')}/{i}",
            "title": f"Placeholder Book {i} — {query}",
            "isbn": f"978-3-16-148410-{i}",
            "doi": f"10.1000/xyz{i}",
            "estimated_size_mb": 10 + i * 5,
            "metadata_score": max(0.0, 0.95 - i * 0.08),
            "source_type": "epub" if i % 2 == 0 else "pdf",
        }
        for i in range(limit)
    ]


async def _stream_download_placeholder(
    source_url: str,
    destination: Path,
    expected_bytes: int | None = None,
    resume_byte: int = 0,
) -> dict[str, Any]:
    """Placeholder streaming download with resume support.

    Writes deterministic text-like filler bytes to *destination* so that
    downstream extraction can treat the file as UTF-8 without loading the
    whole object into memory.
    """
    await asyncio.sleep(0.01)
    chunk_size = 8192
    total_written = resume_byte
    target = expected_bytes or (1024 * 1024)  # 1 MiB default
    filler = b"Placeholder book content for testing. "

    with destination.open("ab" if resume_byte else "wb") as fh:
        while total_written < target:
            to_write = min(chunk_size, target - total_written)
            repeats = (to_write // len(filler)) + 1
            data = (filler * repeats)[:to_write]
            fh.write(data)
            total_written += to_write

    return {
        "bytes_downloaded": total_written,
        "resume_from": resume_byte,
        "completed": True,
    }


async def _generate_embedding(text: str) -> list[float]:
    """Placeholder embedding generator.

    Returns a 768-dimensional zero vector.  In production this would call the
    configured embedding service (e.g. sentence-transformers, OpenAI, etc.).
    """
    await asyncio.sleep(0.001)
    return [0.0] * 768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _hash_file_stream(path: Path) -> str:
    """Compute SHA-256 of *path* by reading in 64 KiB chunks."""

    def _run() -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    return await asyncio.to_thread(_run)


async def _compute_text_hashes(path: Path) -> tuple[str, str]:
    """Return ``(text_sha256, text_simhash)`` for *path*.

    Reads the file in 64 KiB chunks, decodes each chunk as UTF-8, and feeds
    the resulting bytes into two separate hashers.  The simhash value is a
    truncated SHA-256 placeholder; a real implementation would use an
    incremental simhash algorithm.
    """

    def _run() -> tuple[str, str]:
        text_hasher = hashlib.sha256()
        simhash_hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                encoded = text.encode("utf-8")
                text_hasher.update(encoded)
                simhash_hasher.update(encoded)
        return text_hasher.hexdigest(), simhash_hasher.hexdigest()[:16]

    return await asyncio.to_thread(_run)


async def _check_three_strikes(store: StoreProtocol, source_url: str) -> bool:
    """Return ``True`` if *source_url* is blocked by the three-strikes rule.

    A source is blocked when its ``source_utility_log`` contains three or more
    entries with ``utility_score < 0.2``, or when
    ``block_future_downloads`` is already set.
    """
    existing = await store.query(
        "SELECT id FROM source_fingerprints WHERE source_url = ? LIMIT 1",
        (source_url,),
    )
    if not existing:
        return False

    fp_id = existing[0]["id"]

    strike_rows = await store.query(
        """
        SELECT COUNT(*) AS cnt
        FROM source_utility_log
        WHERE source_fingerprint_id = ? AND utility_score < 0.2
        """,
        (fp_id,),
    )
    if strike_rows and strike_rows[0]["cnt"] >= 3:
        return True

    flag_rows = await store.query(
        """
        SELECT block_future_downloads
        FROM source_utility_log
        WHERE source_fingerprint_id = ?
        LIMIT 1
        """,
        (fp_id,),
    )
    if flag_rows and flag_rows[0].get("block_future_downloads"):
        return True

    return False


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class SelectionWorker(Actor):
    """Queries Anna's Archive, ranks results by metadata score, and forwards top-N."""

    def __init__(
        self,
        actor_id: str = "selection_worker",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig()

    async def _run(self) -> None:
        trace = await TraceStore.get()
        while not self._shutdown:
            event = await self.mailbox.get()
            if event.event_type == "IngestBooks":
                async with trace_block(
                    self.actor_id,
                    "selection_cycle",
                    payload={"query": event.payload.get("query")},
                ):
                    await self._handle_ingest(event, trace)
            else:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="selection_unknown_event",
                    payload={"event_type": event.event_type},
                )

    async def _handle_ingest(self, event: Event, trace: TraceStore) -> None:
        query = event.payload.get("query", "")
        top_n = event.payload.get("top_n", self.config.external.max_books_per_query)

        await trace.record(
            actor_id=self.actor_id,
            event_type="selection_query_start",
            payload={"query": query, "top_n": top_n},
        )

        results = await _query_annas_archive_api(query, limit=top_n * 2)
        ranked = sorted(
            results,
            key=lambda r: r.get("metadata_score", 0.0),
            reverse=True,
        )[:top_n]

        await trace.record(
            actor_id=self.actor_id,
            event_type="selection_query_end",
            payload={"candidates": len(results), "selected": len(ranked)},
        )

        for item in ranked:
            await self.send_to_parent(
                Event(
                    "SourceSelected",
                    {
                        "source_url": item["source_url"],
                        "title": item.get("title"),
                        "isbn": item.get("isbn", ""),
                        "doi": item.get("doi", ""),
                        "estimated_size_mb": item.get("estimated_size_mb", 0),
                        "source_type": item.get("source_type", "pdf"),
                        "metadata": item,
                        "query": query,
                    },
                )
            )


class DownloadWorker(Actor):
    """Streams source downloads with resume support and multi-layer fingerprinting."""

    def __init__(
        self,
        actor_id: str = "download_worker",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig()

    async def _run(self) -> None:
        trace = await TraceStore.get()
        while not self._shutdown:
            event = await self.mailbox.get()
            if event.event_type == "SourceSelected":
                async with trace_block(
                    self.actor_id,
                    "download_cycle",
                    payload={"url": event.payload.get("source_url")},
                ):
                    await self._handle_download(event, trace)
            else:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="download_unknown_event",
                    payload={"event_type": event.event_type},
                )

    async def _handle_download(self, event: Event, trace: TraceStore) -> None:
        payload = event.payload
        source_url = payload["source_url"]
        isbn = payload.get("isbn", "")
        doi = payload.get("doi", "")
        estimated_size_mb = payload.get("estimated_size_mb", 0)
        source_type = payload.get("source_type", "pdf")

        # --- large-file guard ------------------------------------------------
        if estimated_size_mb > self.config.external.max_total_mb_per_query:
            await trace.record(
                actor_id=self.actor_id,
                event_type="download_rejected_too_large",
                payload={"source_url": source_url, "size_mb": estimated_size_mb},
            )
            return

        # --- three-strikes guard ---------------------------------------------
        if self.store and await _check_three_strikes(self.store, source_url):
            await trace.record(
                actor_id=self.actor_id,
                event_type="download_blocked_three_strikes",
                payload={"source_url": source_url},
            )
            await self.send_to_parent(
                Event(
                    "DownloadBlocked",
                    {"source_url": source_url, "reason": "three_strikes"},
                )
            )
            return

        # --- stream download --------------------------------------------------
        tmp_dir = Path(tempfile.gettempdir())
        dest = tmp_dir / (
            f"us_dl_{hashlib.sha256(source_url.encode()).hexdigest()[:16]}.{source_type}"
        )
        resume_byte = dest.stat().st_size if dest.exists() else 0

        await trace.record(
            actor_id=self.actor_id,
            event_type="download_start",
            payload={
                "source_url": source_url,
                "destination": str(dest),
                "resume_from": resume_byte,
                "expected_bytes": int(estimated_size_mb * 1024 * 1024),
            },
        )

        meta = await _stream_download_placeholder(
            source_url,
            dest,
            expected_bytes=int(estimated_size_mb * 1024 * 1024),
            resume_byte=resume_byte,
        )

        byte_sha = await _hash_file_stream(dest)
        text_sha, text_simhash = await _compute_text_hashes(dest)

        await trace.record(
            actor_id=self.actor_id,
            event_type="download_complete",
            payload={
                "source_url": source_url,
                "byte_sha256": byte_sha,
                "text_sha256": text_sha,
                "text_simhash": text_simhash,
                "bytes_downloaded": meta["bytes_downloaded"],
            },
        )

        # --- persist fingerprints ---------------------------------------------
        fingerprint_id = 0
        if self.store:
            fingerprint_id = await self.store.insert(
                "source_fingerprints",
                {
                    "source_url": source_url,
                    "source_type": source_type,
                    "byte_sha256": byte_sha,
                    "text_sha256": text_sha,
                    "text_simhash": text_simhash,
                    "isbn": isbn,
                    "doi": doi,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "last_accessed_at": datetime.utcnow().isoformat(),
                    "extraction_status": "pending",
                    "download_method": "placeholder_stream",
                    "metadata_json": json.dumps(payload.get("metadata", {})),
                },
            )

            await self.store.insert(
                "source_utility_log",
                {
                    "source_fingerprint_id": fingerprint_id,
                    "query_text": payload.get("query", ""),
                    "times_queried": 0,
                    "utility_score": 0.5,
                    "utility_verdict": "pending",
                    "last_queried_at": datetime.utcnow().isoformat(),
                    "block_future_downloads": False,
                },
            )

        await self.send_to_parent(
            Event(
                "DownloadComplete",
                {
                    "source_url": source_url,
                    "file_path": str(dest),
                    "fingerprint_id": fingerprint_id,
                    "source_type": source_type,
                    "metadata": payload.get("metadata", {}),
                },
            )
        )


class ExtractionWorker(Actor):
    """Extracts text from PDF/EPUB, runs a quality gate, and OCR fallback."""

    def __init__(
        self,
        actor_id: str = "extraction_worker",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig()

    async def _run(self) -> None:
        trace = await TraceStore.get()
        while not self._shutdown:
            event = await self.mailbox.get()
            if event.event_type == "DownloadComplete":
                async with trace_block(
                    self.actor_id,
                    "extraction_cycle",
                    payload={"url": event.payload.get("source_url")},
                ):
                    await self._handle_extraction(event, trace)
            else:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="extraction_unknown_event",
                    payload={"event_type": event.event_type},
                )

    async def _handle_extraction(self, event: Event, trace: TraceStore) -> None:
        payload = event.payload
        file_path = Path(payload["file_path"])
        source_url = payload["source_url"]
        fingerprint_id = payload.get("fingerprint_id", 0)
        source_type = payload.get("source_type", "pdf")

        await trace.record(
            actor_id=self.actor_id,
            event_type="extraction_start",
            payload={
                "source_url": source_url,
                "file_path": str(file_path),
                "source_type": source_type,
            },
        )

        # Stream-read the raw file and write decoded text to a side-car file.
        text_path = file_path.with_suffix(".extracted.txt")

        def _extract() -> tuple[int, int, str, str]:
            text_hasher = hashlib.sha256()
            simhash_hasher = hashlib.sha256()
            char_count = 0
            non_printable = 0
            with file_path.open("rb") as src, text_path.open(
                "w", encoding="utf-8", errors="replace"
            ) as dst:
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    encoded = text.encode("utf-8")
                    text_hasher.update(encoded)
                    simhash_hasher.update(encoded)
                    char_count += len(text)
                    non_printable += sum(
                        1 for c in text if not c.isprintable() and not c.isspace()
                    )
                    dst.write(text)
            return (
                char_count,
                non_printable,
                text_hasher.hexdigest(),
                simhash_hasher.hexdigest()[:16],
            )

        char_count, non_printable, text_sha, text_simhash = await asyncio.to_thread(
            _extract
        )

        # --- quality gate -----------------------------------------------------
        garble_ratio = non_printable / max(char_count, 1)
        ocr_needed = garble_ratio > self.config.sources.ocr_quality_threshold

        await trace.record(
            actor_id=self.actor_id,
            event_type="extraction_quality_gate",
            payload={
                "source_url": source_url,
                "char_count": char_count,
                "garble_ratio": garble_ratio,
                "ocr_needed": ocr_needed,
            },
        )

        if ocr_needed:
            # Placeholder OCR path: in production this would invoke an OCR
            # pipeline and overwrite *text_path* with corrected text.
            await trace.record(
                actor_id=self.actor_id,
                event_type="extraction_ocr_placeholder",
                payload={"source_url": source_url, "reason": "garble_ratio_exceeded"},
            )

        # --- update fingerprints with text hashes -----------------------------
        if self.store and fingerprint_id:
            await self.store.execute(
                """
                UPDATE source_fingerprints
                SET text_sha256 = ?,
                    text_simhash = ?,
                    extraction_status = 'completed',
                    last_accessed_at = ?
                WHERE id = ?
                """,
                (text_sha, text_simhash, datetime.utcnow().isoformat(), fingerprint_id),
            )

        await trace.record(
            actor_id=self.actor_id,
            event_type="extraction_end",
            payload={
                "source_url": source_url,
                "text_path": str(text_path),
                "text_length": char_count,
            },
        )

        await self.send_to_parent(
            Event(
                "ExtractionComplete",
                {
                    "source_url": source_url,
                    "text_path": str(text_path),
                    "fingerprint_id": fingerprint_id,
                    "source_type": source_type,
                    "metadata": payload.get("metadata", {}),
                },
            )
        )


class ChunkEmbedWorker(Actor):
    """Hybrid chunking + embedding generation with configurable limits."""

    def __init__(self,
        actor_id: str = "chunk_embed_worker",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id)
        self.store = store
        self.config = config or UnifiedConfig()

    async def _run(self) -> None:
        trace = await TraceStore.get()
        while not self._shutdown:
            event = await self.mailbox.get()
            if event.event_type == "ExtractionComplete":
                async with trace_block(
                    self.actor_id,
                    "chunk_embed_cycle",
                    payload={"url": event.payload.get("source_url")},
                ):
                    await self._handle_chunk_embed(event, trace)
            else:
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="chunk_embed_unknown_event",
                    payload={"event_type": event.event_type},
                )

    async def _handle_chunk_embed(self, event: Event, trace: TraceStore) -> None:
        payload = event.payload
        text_path = Path(payload["text_path"])
        source_url = payload["source_url"]
        fingerprint_id = payload.get("fingerprint_id", 0)

        max_chunks = self.config.sources.max_chunks_per_source
        chunk_char_limit = max(self.config.sources.chunk_size_tokens * 4, 1)
        overlap_chars = max(self.config.sources.chunk_overlap_tokens * 4, 0)

        await trace.record(
            actor_id=self.actor_id,
            event_type="chunk_embed_start",
            payload={
                "source_url": source_url,
                "max_chunks": max_chunks,
                "chunk_char_limit": chunk_char_limit,
                "overlap_chars": overlap_chars,
            },
        )

        rows_added = 0
        chunk_index = 0

        async def _flush_buffer(buf: str, idx: int) -> int:
            nonlocal rows_added
            stripped = buf.strip()
            if not stripped:
                return idx
            chunk_row = {
                "source_fingerprint_id": fingerprint_id,
                "chunk_index": idx,
                "content": stripped,
                "token_count": len(stripped.split()),
                "char_count": len(stripped),
                "created_at": datetime.utcnow().isoformat(),
            }
            embedding = await _generate_embedding(stripped)
            chunk_row["embedding"] = embedding
            if self.store:
                chunk_id = await self.store.insert("chunks", chunk_row)
                rows_added += 1
                await trace.record(
                    actor_id=self.actor_id,
                    event_type="chunk_embed_inserted",
                    payload={"chunk_id": chunk_id, "chunk_index": idx},
                )
            return idx + 1

        buffer = ""
        with text_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.strip()
                if self._is_header(stripped):
                    if buffer.strip():
                        chunk_index = await _flush_buffer(buffer, chunk_index)
                        if chunk_index >= max_chunks:
                            break
                    buffer = stripped + "\n"
                else:
                    buffer += line
                    if len(buffer) >= chunk_char_limit:
                        chunk_index = await _flush_buffer(buffer, chunk_index)
                        if chunk_index >= max_chunks:
                            break
                        buffer = (
                            buffer[-overlap_chars:]
                            if overlap_chars < len(buffer)
                            else ""
                        )
            else:
                # Loop finished without break — flush remainder.
                if buffer.strip() and chunk_index < max_chunks:
                    chunk_index = await _flush_buffer(buffer, chunk_index)

        # --- fallback: fixed-size sliding window over whole file -------------
        if chunk_index == 0:
            buffer = ""
            with text_path.open("r", encoding="utf-8", errors="replace") as fh:
                while True:
                    data = fh.read(chunk_char_limit)
                    if not data:
                        break
                    buffer = data
                    chunk_index = await _flush_buffer(buffer, chunk_index)
                    if chunk_index >= max_chunks:
                        break
                    buffer = (
                        buffer[-overlap_chars:]
                        if overlap_chars < len(buffer)
                        else ""
                    )

        await trace.record(
            actor_id=self.actor_id,
            event_type="chunk_embed_end",
            payload={"source_url": source_url, "chunks_produced": chunk_index, "rows_added": rows_added},
        )

        await self.send_to_parent(
            StoreDelta(rows_added=rows_added, row_types=["chunk"])
        )

    @staticmethod
    def _is_header(line: str) -> bool:
        """Match Markdown-style or plain-text headers (h1/h2/h3)."""
        return bool(re.match(r"^#{1,3}\s+", line))


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class SourceIngestionActor(Supervisor):
    """Top-level supervisor for the source-ingestion pipeline.

    Spawns four workers and routes events between them:

    * ``IngestBooks``        → ``SelectionWorker``
    * ``SourceSelected``     → ``DownloadWorker``
    * ``DownloadComplete``   → ``ExtractionWorker``
    * ``ExtractionComplete`` → ``ChunkEmbedWorker``
    * ``StoreDelta``         → propagated upward
    """

    def __init__(
        self,
        actor_id: str = "source_ingestion",
        store: StoreProtocol | None = None,
        config: UnifiedConfig | None = None,
    ):
        super().__init__(actor_id, strategy="restart", max_restarts=3, restart_window_s=60.0)
        self.store = store
        self.config = config or UnifiedConfig()
        self._register_children()

    def _register_children(self) -> None:
        self.register_child(
            "selection_worker",
            lambda: SelectionWorker(
                actor_id="selection_worker", store=self.store, config=self.config
            ),
        )
        self.register_child(
            "download_worker",
            lambda: DownloadWorker(
                actor_id="download_worker", store=self.store, config=self.config
            ),
        )
        self.register_child(
            "extraction_worker",
            lambda: ExtractionWorker(
                actor_id="extraction_worker", store=self.store, config=self.config
            ),
        )
        self.register_child(
            "chunk_embed_worker",
            lambda: ChunkEmbedWorker(
                actor_id="chunk_embed_worker", store=self.store, config=self.config
            ),
        )

    def start(self) -> None:
        """Start the supervisor and then spawn all registered workers."""
        super().start()
        asyncio.create_task(self.spawn_all_registered())

    async def _handle_event(self, event: Event) -> None:
        trace = await TraceStore.get()
        await trace.record(
            actor_id=self.actor_id,
            event_type="source_ingestion_routing",
            payload={"event_type": event.event_type, "source": event.source_actor},
        )

        if event.event_type == "IngestBooks":
            child = self._children.get("selection_worker")
            if child:
                await child.send(event)
        elif event.event_type == "SourceSelected":
            child = self._children.get("download_worker")
            if child:
                await child.send(event)
        elif event.event_type == "DownloadComplete":
            child = self._children.get("extraction_worker")
            if child:
                await child.send(event)
        elif event.event_type == "ExtractionComplete":
            child = self._children.get("chunk_embed_worker")
            if child:
                await child.send(event)
        elif event.event_type == "StoreDelta":
            await self.send_to_parent(event)
        else:
            # Default: broadcast to every child (useful for control events).
            await self.broadcast_to_children(event)
