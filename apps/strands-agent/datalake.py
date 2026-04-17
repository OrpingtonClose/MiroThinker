# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
B2 datalake for persistent research artifact storage using RO-Crate format.

Stores downloaded books, papers, videos, and other research artifacts in
Backblaze B2 following the RO-Crate 1.2 specification (https://w3id.org/ro/crate/1.2).
Each artifact is a self-describing Research Object with:

  1. ro-crate-metadata.json  — JSON-LD metadata (title, author, source, hashes)
  2. raw.<ext>               — original file (PDF/EPUB/video/etc.)
  3. extracted.txt           — extracted plain text content

RO-Crate is an established standard for packaging research data with rich,
machine-readable metadata. It is FAIR-compliant (Findable, Accessible,
Interoperable, Reusable) and widely used in digital preservation.

Bucket layout (content-addressed by SHA-256 of raw file):
    mirothinker-datalake/
        crates/
            <sha256>/
                ro-crate-metadata.json
                raw.<ext>
                extracted.txt
        index/
            catalog.jsonl             # append-only catalog for fast lookups

Content addressing means the same file is never uploaded twice, even if
acquired from different sources or in different sessions.

Environment variables:
    B2_KEY_ID              — Backblaze application key ID
    B2_APPLICATION_KEY     — Backblaze application key
    B2_DATALAKE_BUCKET     — bucket name (default: mirothinker-datalake)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

B2_KEY_ID = os.getenv("B2_KEY_ID", "")
B2_APP_KEY = os.getenv("B2_APPLICATION_KEY", "")
B2_BUCKET_NAME = os.getenv("B2_DATALAKE_BUCKET", "mirothinker-datalake")

# ── Singleton B2 client ───────────────────────────────────────────────

_b2_api = None
_b2_bucket = None
_b2_lock = threading.Lock()


def _get_bucket():
    """Lazily initialise and return the B2 bucket handle (thread-safe)."""
    global _b2_api, _b2_bucket

    if _b2_bucket is not None:
        return _b2_bucket

    with _b2_lock:
        if _b2_bucket is not None:
            return _b2_bucket

        key_id = B2_KEY_ID
        app_key = B2_APP_KEY

        if not key_id or not app_key:
            logger.debug("B2_KEY_ID / B2_APPLICATION_KEY not set — datalake disabled")
            return None

        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
            from b2sdk.v2.exception import NonExistentBucket

            info = InMemoryAccountInfo()
            _b2_api = B2Api(info)
            _b2_api.authorize_account("production", key_id, app_key)

            try:
                _b2_bucket = _b2_api.get_bucket_by_name(B2_BUCKET_NAME)
                logger.info("B2 datalake connected: bucket=%s", B2_BUCKET_NAME)
            except NonExistentBucket:
                _b2_bucket = _b2_api.create_bucket(
                    B2_BUCKET_NAME,
                    bucket_type="allPrivate",
                    lifecycle_rules=[],
                )
                logger.info("B2 datalake created new bucket: %s", B2_BUCKET_NAME)
        except ImportError:
            logger.debug("b2sdk not installed — datalake disabled. Install: pip install b2sdk")
            return None
        except Exception as exc:
            logger.error("B2 datalake init failed: %s", exc)
            return None

    return _b2_bucket


def is_configured() -> bool:
    """Check if B2 datalake credentials are available."""
    return bool(B2_KEY_ID and B2_APP_KEY)


# ── Content addressing ────────────────────────────────────────────────


def content_hash(data: bytes) -> str:
    """SHA-256 hash of content bytes."""
    return hashlib.sha256(data).hexdigest()


# ── Low-level B2 operations ──────────────────────────────────────────


def _upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> bool:
    """Upload raw bytes to B2. Returns True on success."""
    bucket = _get_bucket()
    if bucket is None:
        return False

    try:
        from b2sdk.v2 import UploadSourceBytes
        bucket.upload(UploadSourceBytes(data), file_name=key, content_type=content_type)
        logger.debug("B2 uploaded %s (%d bytes)", key, len(data))
        return True
    except Exception as exc:
        logger.error("B2 upload failed %s: %s", key, exc)
        return False


def _download_bytes(key: str) -> Optional[bytes]:
    """Download raw bytes from B2. Returns None if not found."""
    bucket = _get_bucket()
    if bucket is None:
        return None

    try:
        import io
        dl = bucket.download_file_by_name(key)
        buf = io.BytesIO()
        dl.save(buf)
        return buf.getvalue()
    except Exception:
        return None


def _file_exists(key: str) -> bool:
    """Check if a file exists in B2."""
    bucket = _get_bucket()
    if bucket is None:
        return False

    try:
        # List files with exact prefix — if any match, file exists
        file_versions = bucket.list_file_versions(file_name=key, fetch_count=1)
        for fv, _ in file_versions:
            if fv.file_name == key:
                return True
        return False
    except Exception:
        return False


# ── RO-Crate creation ────────────────────────────────────────────────


def _build_rocrate_metadata(
    sha: str,
    raw_filename: str,
    extracted_filename: str,
    metadata: dict,
) -> dict:
    """Build an RO-Crate 1.2 metadata document as a Python dict.

    Uses the RO-Crate JSON-LD structure directly rather than the rocrate
    library, to avoid filesystem I/O overhead (we upload to B2, not disk).
    """
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Determine encoding format from file extension
    ext = metadata.get("file_extension", "")
    encoding_formats = {
        "pdf": "application/pdf",
        "epub": "application/epub+zip",
        "txt": "text/plain",
        "html": "text/html",
        "htm": "text/html",
        "mp4": "video/mp4",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
    }
    encoding_format = encoding_formats.get(ext, "application/octet-stream")

    # Build the @graph
    graph = [
        # Root dataset (the crate itself)
        {
            "@id": "./",
            "@type": "Dataset",
            "name": metadata.get("title", "Unknown artifact"),
            "description": metadata.get("description", f"Research artifact: {metadata.get('title', 'Unknown')}"),
            "datePublished": now_iso,
            "license": metadata.get("license", ""),
            "hasPart": [
                {"@id": raw_filename},
                {"@id": extracted_filename},
            ],
        },
        # RO-Crate metadata descriptor
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.2"},
        },
        # The raw artifact file
        {
            "@id": raw_filename,
            "@type": "File",
            "name": metadata.get("title", "Unknown"),
            "encodingFormat": encoding_format,
            "contentSize": str(metadata.get("raw_size_bytes", 0)),
            "sha256": sha,
            "dateCreated": now_iso,
        },
        # The extracted text file
        {
            "@id": extracted_filename,
            "@type": "File",
            "name": f"Extracted text: {metadata.get('title', 'Unknown')}",
            "encodingFormat": "text/plain",
            "contentSize": str(metadata.get("text_size_chars", 0)),
            "description": "Plain text extracted from the raw artifact",
            "dateCreated": now_iso,
        },
    ]

    # Add author entity if provided
    author = metadata.get("author", "")
    if author:
        author_id = f"#author-{author.lower().replace(' ', '-')[:50]}"
        graph[2]["author"] = {"@id": author_id}
        graph.append({
            "@id": author_id,
            "@type": "Person",
            "name": author,
        })

    # Add source/provenance metadata to the raw file entry
    raw_entry = graph[2]
    if metadata.get("source"):
        raw_entry["isBasedOn"] = metadata["source"]
    if metadata.get("source_url"):
        raw_entry["url"] = metadata["source_url"]
    if metadata.get("isbn"):
        raw_entry["isbn"] = metadata["isbn"]
    if metadata.get("doi"):
        raw_entry["identifier"] = f"https://doi.org/{metadata['doi']}"
    if metadata.get("year"):
        raw_entry["datePublished"] = str(metadata["year"])
    if metadata.get("language"):
        raw_entry["inLanguage"] = metadata["language"]
    if metadata.get("pages"):
        raw_entry["numberOfPages"] = str(metadata["pages"])
    if metadata.get("publisher"):
        raw_entry["publisher"] = metadata["publisher"]

    # Add category as additionalType
    category = metadata.get("category", "")
    if category:
        type_map = {
            "books": "Book",
            "papers": "ScholarlyArticle",
            "videos": "VideoObject",
            "web": "WebPage",
        }
        schema_type = type_map.get(category, "CreativeWork")
        raw_entry["@type"] = ["File", schema_type]

    return {
        "@context": "https://w3id.org/ro/crate/1.2/context",
        "@graph": graph,
    }


# ── Catalog index ─────────────────────────────────────────────────────


def _append_to_catalog(sha: str, metadata: dict) -> bool:
    """Append an entry to the JSONL catalog index for fast lookups."""
    entry = {
        "content_hash": sha,
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "category": metadata.get("category", ""),
        "source": metadata.get("source", ""),
        "file_extension": metadata.get("file_extension", ""),
        "raw_size_bytes": metadata.get("raw_size_bytes", 0),
        "text_size_chars": metadata.get("text_size_chars", 0),
        "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if metadata.get("doi"):
        entry["doi"] = metadata["doi"]
    if metadata.get("isbn"):
        entry["isbn"] = metadata["isbn"]
    if metadata.get("source_url"):
        entry["source_url"] = metadata["source_url"]

    line = json.dumps(entry, ensure_ascii=False) + "\n"
    return _upload_bytes(
        line.encode("utf-8"),
        f"index/catalog_{sha[:16]}.jsonl",
        content_type="application/x-ndjson",
    )


# ── High-level API ────────────────────────────────────────────────────


def store_artifact(
    raw_content: bytes,
    extracted_text: str,
    category: str,
    metadata: dict,
    file_extension: str = "",
) -> Optional[str]:
    """Store a complete research artifact as an RO-Crate in the datalake.

    Content-addressed: if the same raw content was already stored, this
    is a no-op (deduplication by SHA-256).

    Args:
        raw_content: Original file bytes (PDF, EPUB, MP4, HTML, etc.)
        extracted_text: Plain text extracted from the artifact.
        category: Artifact type — "books", "papers", "videos", "web".
        metadata: Dict with title, author, source, url, doi, isbn, etc.
        file_extension: File extension for the raw file (e.g., "pdf", "epub").

    Returns:
        The SHA-256 content hash (acts as the artifact ID), or None on failure.
    """
    sha = content_hash(raw_content)
    ext = file_extension or metadata.get("file_extension", "bin")

    # Crate paths in B2
    crate_prefix = f"crates/{sha}"
    raw_key = f"{crate_prefix}/raw.{ext}"
    text_key = f"{crate_prefix}/extracted.txt"
    meta_key = f"{crate_prefix}/ro-crate-metadata.json"

    # Check if already stored (dedup)
    if _file_exists(meta_key):
        logger.info("Artifact already in datalake: %s (%s)", sha[:16], metadata.get("title", "?")[:50])
        return sha

    # Enrich metadata for RO-Crate
    metadata = {
        **metadata,
        "content_hash": sha,
        "category": category,
        "file_extension": ext,
        "raw_size_bytes": len(raw_content),
        "text_size_chars": len(extracted_text),
    }

    # Build RO-Crate metadata document
    rocrate_meta = _build_rocrate_metadata(
        sha=sha,
        raw_filename=f"raw.{ext}",
        extracted_filename="extracted.txt",
        metadata=metadata,
    )
    meta_json = json.dumps(rocrate_meta, ensure_ascii=False, indent=2).encode("utf-8")

    # Upload all three parts of the crate
    ok_raw = _upload_bytes(raw_content, raw_key)
    ok_text = _upload_bytes(
        extracted_text.encode("utf-8"), text_key,
        content_type="text/plain; charset=utf-8",
    )
    ok_meta = _upload_bytes(meta_json, meta_key, content_type="application/ld+json")

    if ok_raw and ok_text and ok_meta:
        # Only index after all uploads succeed to avoid phantom catalog entries
        _append_to_catalog(sha, metadata)
        logger.info(
            "Datalake stored RO-Crate: %s/%s (%s, %d bytes raw, %d chars text)",
            category, sha[:16], metadata.get("title", "?")[:50],
            len(raw_content), len(extracted_text),
        )
        return sha

    logger.warning(
        "Datalake partial upload: raw=%s text=%s meta=%s for %s",
        ok_raw, ok_text, ok_meta, sha[:16],
    )
    return sha if ok_raw else None


def lookup_artifact(
    sha: str = "",
    raw_content: bytes = b"",
    category: str = "",
) -> Optional[dict]:
    """Look up an artifact by content hash or raw content.

    Returns metadata dict with 'text' key containing extracted text, or None.
    """
    if not sha and raw_content:
        sha = content_hash(raw_content)
    if not sha:
        return None

    meta_key = f"crates/{sha}/ro-crate-metadata.json"
    meta_bytes = _download_bytes(meta_key)
    if meta_bytes is None:
        return None

    try:
        rocrate = json.loads(meta_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    # Extract metadata from the RO-Crate @graph
    result = {"content_hash": sha, "_rocrate": rocrate}
    for entity in rocrate.get("@graph", []):
        eid = entity.get("@id", "")
        if eid.startswith("raw."):
            result["title"] = entity.get("name", "")
            result["encoding_format"] = entity.get("encodingFormat", "")
            result["content_size"] = entity.get("contentSize", "")
            result["sha256"] = entity.get("sha256", "")
            if entity.get("url"):
                result["source_url"] = entity["url"]
            if entity.get("identifier"):
                result["doi"] = entity["identifier"]
            if entity.get("isbn"):
                result["isbn"] = entity["isbn"]
            # Resolve author
            author_ref = entity.get("author", {})
            if isinstance(author_ref, dict) and "@id" in author_ref:
                for e in rocrate.get("@graph", []):
                    if e.get("@id") == author_ref["@id"]:
                        result["author"] = e.get("name", "")
                        break

    # Fetch extracted text
    text_key = f"crates/{sha}/extracted.txt"
    text_bytes = _download_bytes(text_key)
    if text_bytes:
        result["text"] = text_bytes.decode("utf-8", errors="replace")

    return result


def get_artifact_text(sha: str) -> Optional[str]:
    """Get just the extracted text for an artifact by hash."""
    text_bytes = _download_bytes(f"crates/{sha}/extracted.txt")
    if text_bytes:
        return text_bytes.decode("utf-8", errors="replace")
    return None


def get_artifact_raw(sha: str, file_extension: str = "bin") -> Optional[bytes]:
    """Get the raw file bytes for an artifact by hash."""
    return _download_bytes(f"crates/{sha}/raw.{file_extension}")


def list_artifacts(limit: int = 100) -> list[dict]:
    """List artifacts in the datalake by reading catalog index entries.

    Returns list of metadata dicts (lightweight — doesn't download full text).
    """
    bucket = _get_bucket()
    if bucket is None:
        return []

    entries = []
    try:
        for file_version, _ in bucket.ls_folder("index/", fetch_count=limit):
            if not file_version.file_name.endswith(".jsonl"):
                continue
            data = _download_bytes(file_version.file_name)
            if data:
                for line in data.decode("utf-8", errors="replace").strip().split("\n"):
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if len(entries) >= limit:
                break
    except Exception as exc:
        logger.debug("Failed to list datalake artifacts: %s", exc)

    return entries[:limit]
