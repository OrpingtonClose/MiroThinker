# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Persistent cache & storage layer for the Strands research agent.

Architecture:
- SQLite database for metadata index (URL, content hash, timestamps, TTL, etc.)
- Content-addressed blob storage (SHA-256 hash → file on disk)
- Deduplication: same content from different URLs stored once
- TTL support: stale entries can be refreshed on next access

Location: ~/.mirothinker/cache/
  ├── cache.db          (SQLite metadata index)
  └── blobs/            (content-addressed files: ab/abcdef1234...ext)

This module provides both a Python API (for direct use in tool code) and
@tool-decorated functions for the agent to call directly.

Resolves: https://github.com/OrpingtonClose/MiroThinker/issues/93
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from strands import tool

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

CACHE_DIR = Path(
    os.environ.get("MIROTHINKER_CACHE_DIR", os.path.expanduser("~/.mirothinker/cache"))
)
BLOB_DIR = CACHE_DIR / "blobs"
DB_PATH = CACHE_DIR / "cache.db"

# Default TTL: 7 days (seconds).  Override per-entry or via env.
DEFAULT_TTL = int(os.environ.get("CACHE_DEFAULT_TTL", str(7 * 24 * 3600)))

# ── Initialisation ────────────────────────────────────────────────────


def _ensure_dirs() -> None:
    """Create cache directories if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BLOB_DIR.mkdir(parents=True, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    _ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS cache_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            content_type TEXT NOT NULL DEFAULT 'text/html',
            source_type TEXT NOT NULL DEFAULT 'web',
            title       TEXT DEFAULT '',
            summary     TEXT DEFAULT '',
            blob_path   TEXT NOT NULL,
            blob_size   INTEGER NOT NULL DEFAULT 0,
            ttl         INTEGER NOT NULL DEFAULT 604800,
            created_at  REAL NOT NULL,
            accessed_at REAL NOT NULL,
            expires_at  REAL NOT NULL,
            metadata    TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_cache_url ON cache_entries(url);
        CREATE INDEX IF NOT EXISTS idx_cache_hash ON cache_entries(content_hash);
        CREATE INDEX IF NOT EXISTS idx_cache_source ON cache_entries(source_type);
        CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at);

        CREATE TABLE IF NOT EXISTS cache_tags (
            entry_id INTEGER NOT NULL,
            tag      TEXT NOT NULL,
            FOREIGN KEY (entry_id) REFERENCES cache_entries(id) ON DELETE CASCADE,
            PRIMARY KEY (entry_id, tag)
        );

        CREATE INDEX IF NOT EXISTS idx_tags_tag ON cache_tags(tag);
    """)
    conn.commit()


# Initialise on import
_db_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    """Get or create the singleton database connection."""
    global _db_conn
    if _db_conn is None:
        _db_conn = _get_db()
        _init_schema(_db_conn)
    return _db_conn


# ── Content-addressed blob storage ────────────────────────────────────


def content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(data).hexdigest()


def _blob_path(sha256: str, ext: str = "") -> Path:
    """Return the blob path for a given hash: blobs/ab/abcdef...ext"""
    prefix = sha256[:2]
    subdir = BLOB_DIR / prefix
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"{sha256}{ext}"


def store_blob(data: bytes, ext: str = "") -> tuple[str, Path]:
    """Store content-addressed blob. Returns (hash, path).

    If blob already exists (same hash), skips writing (deduplication).
    """
    sha = content_hash(data)
    path = _blob_path(sha, ext)
    if not path.exists():
        path.write_bytes(data)
        logger.debug("Stored blob %s (%d bytes)", sha[:12], len(data))
    else:
        logger.debug("Blob %s already exists (deduplicated)", sha[:12])
    return sha, path


def read_blob(sha256: str, ext: str = "") -> bytes | None:
    """Read blob by hash. Returns None if not found."""
    path = _blob_path(sha256, ext)
    if path.exists():
        return path.read_bytes()
    return None


# ── Core cache operations (Python API) ────────────────────────────────


def cache_put(
    url: str,
    content: bytes | str,
    content_type: str = "text/html",
    source_type: str = "web",
    title: str = "",
    summary: str = "",
    tags: list[str] | None = None,
    ttl: int | None = None,
    metadata: dict | None = None,
) -> dict:
    """Store content in the cache.

    Args:
        url: Source URL (primary key for lookups).
        content: Raw content (bytes or str).
        content_type: MIME type (text/html, application/pdf, video/mp4, etc.).
        source_type: Category (web, youtube, paper, book, forum, etc.).
        title: Human-readable title.
        summary: Brief summary or extracted text snippet.
        tags: Optional tags for categorisation.
        ttl: Time-to-live in seconds (default: 7 days).
        metadata: Additional metadata dict.

    Returns:
        Dict with entry details including content_hash and id.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    if ttl is None:
        ttl = DEFAULT_TTL

    sha, blob_path = store_blob(content)
    now = time.time()

    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO cache_entries
            (url, content_hash, content_type, source_type, title, summary,
             blob_path, blob_size, ttl, created_at, accessed_at, expires_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            url,
            sha,
            content_type,
            source_type,
            title,
            summary,
            str(blob_path.relative_to(CACHE_DIR)),
            len(content),
            ttl,
            now,
            now,
            now + ttl,
            json.dumps(metadata or {}),
        ),
    )
    entry_id = cursor.lastrowid

    if tags:
        db.executemany(
            "INSERT OR IGNORE INTO cache_tags (entry_id, tag) VALUES (?, ?)",
            [(entry_id, tag) for tag in tags],
        )

    db.commit()

    logger.info(
        "Cached: %s [%s] %s (%d bytes, hash=%s)",
        title or url,
        source_type,
        content_type,
        len(content),
        sha[:12],
    )

    return {
        "id": entry_id,
        "url": url,
        "content_hash": sha,
        "content_type": content_type,
        "source_type": source_type,
        "blob_size": len(content),
        "expires_at": now + ttl,
    }


def cache_get(
    url: str | None = None,
    content_hash_val: str | None = None,
    include_expired: bool = False,
) -> dict | None:
    """Look up a cache entry by URL or content hash.

    Returns the most recent matching entry, or None if not found.
    Updates accessed_at timestamp on hit.
    """
    db = get_db()
    now = time.time()

    if url:
        if include_expired:
            row = db.execute(
                "SELECT * FROM cache_entries WHERE url = ? ORDER BY created_at DESC LIMIT 1",
                (url,),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT * FROM cache_entries WHERE url = ? AND expires_at > ? ORDER BY created_at DESC LIMIT 1",
                (url, now),
            ).fetchone()
    elif content_hash_val:
        row = db.execute(
            "SELECT * FROM cache_entries WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
            (content_hash_val,),
        ).fetchone()
    else:
        return None

    if row is None:
        return None

    # Update accessed_at
    db.execute(
        "UPDATE cache_entries SET accessed_at = ? WHERE id = ?",
        (now, row["id"]),
    )
    db.commit()

    # Get tags
    tags = [
        r["tag"]
        for r in db.execute(
            "SELECT tag FROM cache_tags WHERE entry_id = ?", (row["id"],)
        ).fetchall()
    ]

    # Read blob content
    blob_rel_path = row["blob_path"]
    blob_full_path = CACHE_DIR / blob_rel_path
    content = blob_full_path.read_bytes() if blob_full_path.exists() else None

    return {
        "id": row["id"],
        "url": row["url"],
        "content_hash": row["content_hash"],
        "content_type": row["content_type"],
        "source_type": row["source_type"],
        "title": row["title"],
        "summary": row["summary"],
        "blob_size": row["blob_size"],
        "ttl": row["ttl"],
        "created_at": row["created_at"],
        "accessed_at": row["accessed_at"],
        "expires_at": row["expires_at"],
        "metadata": json.loads(row["metadata"] or "{}"),
        "tags": tags,
        "content": content,
        "expired": row["expires_at"] < now,
    }


def cache_search(
    source_type: str | None = None,
    tag: str | None = None,
    query: str | None = None,
    limit: int = 20,
    include_expired: bool = False,
) -> list[dict]:
    """Search cache entries by source type, tag, or text query.

    Args:
        source_type: Filter by source type (web, youtube, paper, etc.).
        tag: Filter by tag.
        query: Full-text search in title and summary.
        limit: Max results (default 20).
        include_expired: Include expired entries.

    Returns:
        List of cache entry dicts (without blob content).
    """
    db = get_db()
    now = time.time()

    conditions = []
    params: list = []

    if not include_expired:
        conditions.append("e.expires_at > ?")
        params.append(now)

    if source_type:
        conditions.append("e.source_type = ?")
        params.append(source_type)

    if query:
        conditions.append("(e.title LIKE ? OR e.summary LIKE ? OR e.url LIKE ?)")
        pattern = f"%{query}%"
        params.extend([pattern, pattern, pattern])

    where = " AND ".join(conditions) if conditions else "1=1"

    if tag:
        sql = f"""
            SELECT e.* FROM cache_entries e
            JOIN cache_tags t ON e.id = t.entry_id
            WHERE {where} AND t.tag = ?
            ORDER BY e.created_at DESC LIMIT ?
        """
        params.extend([tag, limit])
    else:
        sql = f"""
            SELECT e.* FROM cache_entries e
            WHERE {where}
            ORDER BY e.created_at DESC LIMIT ?
        """
        params.append(limit)

    rows = db.execute(sql, params).fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "url": row["url"],
            "content_hash": row["content_hash"],
            "content_type": row["content_type"],
            "source_type": row["source_type"],
            "title": row["title"],
            "summary": row["summary"],
            "blob_size": row["blob_size"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
            "expired": row["expires_at"] < now,
        })

    return results


def cache_stats() -> dict:
    """Return cache statistics."""
    db = get_db()
    now = time.time()

    total = db.execute("SELECT COUNT(*) as c FROM cache_entries").fetchone()["c"]
    active = db.execute(
        "SELECT COUNT(*) as c FROM cache_entries WHERE expires_at > ?", (now,)
    ).fetchone()["c"]
    expired = total - active

    total_size = db.execute(
        "SELECT COALESCE(SUM(blob_size), 0) as s FROM cache_entries"
    ).fetchone()["s"]

    by_source = {}
    for row in db.execute(
        "SELECT source_type, COUNT(*) as c, SUM(blob_size) as s FROM cache_entries GROUP BY source_type"
    ).fetchall():
        by_source[row["source_type"]] = {
            "count": row["c"],
            "total_bytes": row["s"],
        }

    return {
        "total_entries": total,
        "active_entries": active,
        "expired_entries": expired,
        "total_bytes": total_size,
        "total_mb": round(total_size / (1024 * 1024), 2),
        "by_source_type": by_source,
        "cache_dir": str(CACHE_DIR),
        "db_path": str(DB_PATH),
    }


def cache_evict_expired() -> int:
    """Remove expired entries and their blobs. Returns count of evicted entries."""
    db = get_db()
    now = time.time()

    # Get expired entries
    expired = db.execute(
        "SELECT id, content_hash, blob_path FROM cache_entries WHERE expires_at <= ?",
        (now,),
    ).fetchall()

    if not expired:
        return 0

    evicted = 0
    for row in expired:
        # Check if other entries reference the same blob
        other_refs = db.execute(
            "SELECT COUNT(*) as c FROM cache_entries WHERE content_hash = ? AND id != ? AND expires_at > ?",
            (row["content_hash"], row["id"], now),
        ).fetchone()["c"]

        # Delete the entry
        db.execute("DELETE FROM cache_entries WHERE id = ?", (row["id"],))
        evicted += 1

        # Only delete blob if no other active entries reference it
        if other_refs == 0:
            blob_full = CACHE_DIR / row["blob_path"]
            if blob_full.exists():
                blob_full.unlink()
                logger.debug("Deleted orphan blob: %s", row["blob_path"])

    db.commit()
    logger.info("Evicted %d expired cache entries", evicted)
    return evicted


# ═══════════════════════════════════════════════════════════════════════
# Agent-facing @tool functions
# ═══════════════════════════════════════════════════════════════════════


@tool
def cache_store_content(
    url: str,
    content: str,
    content_type: str = "text/html",
    source_type: str = "web",
    title: str = "",
    summary: str = "",
    tags: str = "[]",
    ttl_seconds: int = 0,
) -> str:
    """Store web content, documents, or any text in the persistent cache.

    Use this to save downloaded content so it doesn't need to be re-fetched
    in future sessions.  Content is deduplicated by SHA-256 hash — storing
    the same content twice costs nothing.

    Args:
        url: The source URL this content came from.
        content: The text content to store.
        content_type: MIME type (text/html, text/plain, application/pdf, etc.).
        source_type: Category: web, youtube, paper, book, forum, video_transcript.
        title: Human-readable title for the content.
        summary: Brief summary (first 500 chars of extracted text, or your own summary).
        tags: JSON array of tags for categorisation (e.g. '["banana", "heritage", "poland"]').
        ttl_seconds: Time-to-live in seconds. 0 = use default (7 days).

    Returns:
        Confirmation with entry ID and content hash.
    """
    try:
        tag_list = json.loads(tags) if tags else []
    except (json.JSONDecodeError, TypeError):
        tag_list = []

    result = cache_put(
        url=url,
        content=content,
        content_type=content_type,
        source_type=source_type,
        title=title,
        summary=summary[:500],
        tags=tag_list,
        ttl=ttl_seconds if ttl_seconds > 0 else None,
    )

    return (
        f"Cached: {title or url}\n"
        f"  ID: {result['id']}\n"
        f"  Hash: {result['content_hash'][:16]}...\n"
        f"  Size: {result['blob_size']:,} bytes\n"
        f"  Type: {result['source_type']}/{result['content_type']}\n"
        f"  Expires: {int(result['expires_at'] - time.time())}s from now"
    )


@tool
def cache_lookup(
    url: str = "",
    content_hash_value: str = "",
) -> str:
    """Look up content in the persistent cache by URL or content hash.

    Use this BEFORE fetching a URL to check if you already have the content
    from a previous session.  Returns the cached content if found.

    Args:
        url: The URL to look up.
        content_hash_value: Alternatively, look up by SHA-256 content hash.

    Returns:
        Cached content if found, or "CACHE_MISS" if not in cache.
    """
    entry = cache_get(url=url or None, content_hash_val=content_hash_value or None)

    if entry is None:
        return "CACHE_MISS"

    content_str = ""
    if entry.get("content"):
        try:
            content_str = entry["content"].decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            content_str = f"[Binary content, {entry['blob_size']} bytes]"

    expired_note = " (EXPIRED — consider refreshing)" if entry.get("expired") else ""

    header = (
        f"CACHE_HIT{expired_note}\n"
        f"  URL: {entry['url']}\n"
        f"  Title: {entry['title']}\n"
        f"  Type: {entry['source_type']}/{entry['content_type']}\n"
        f"  Hash: {entry['content_hash'][:16]}...\n"
        f"  Size: {entry['blob_size']:,} bytes\n"
        f"  Cached: {int(time.time() - entry['created_at'])}s ago\n"
        f"  Tags: {entry.get('tags', [])}\n"
        f"---\n"
    )

    # Truncate very large content for the agent's context window
    if len(content_str) > 50000:
        content_str = content_str[:50000] + f"\n\n[...truncated, {entry['blob_size']:,} bytes total]"

    return header + content_str


@tool
def cache_search_entries(
    source_type: str = "",
    tag: str = "",
    query: str = "",
    limit: int = 20,
) -> str:
    """Search the persistent cache for previously stored content.

    Use this to find content from previous research sessions — papers,
    web pages, transcripts, forum posts, etc. Returns metadata (not content)
    for matching entries.

    Args:
        source_type: Filter by type: web, youtube, paper, book, forum, video_transcript.
        tag: Filter by tag.
        query: Search in titles, summaries, and URLs.
        limit: Maximum results to return (default 20).

    Returns:
        JSON list of matching cache entries with metadata.
    """
    results = cache_search(
        source_type=source_type or None,
        tag=tag or None,
        query=query or None,
        limit=limit,
    )

    if not results:
        return "No cache entries found matching your criteria."

    formatted = [f"Found {len(results)} cached entries:\n"]
    for r in results:
        age = int(time.time() - r["created_at"])
        age_str = f"{age // 3600}h" if age > 3600 else f"{age // 60}m"
        expired = " [EXPIRED]" if r.get("expired") else ""
        formatted.append(
            f"  [{r['id']}] {r['title'] or r['url'][:80]}\n"
            f"       {r['source_type']}/{r['content_type']} | "
            f"{r['blob_size']:,} bytes | {age_str} ago{expired}"
        )

    return "\n".join(formatted)


@tool
def cache_statistics() -> str:
    """Show persistent cache statistics — total entries, size, breakdown by type.

    Returns:
        Human-readable cache statistics.
    """
    stats = cache_stats()

    lines = [
        "Cache Statistics:",
        f"  Total entries: {stats['total_entries']}",
        f"  Active: {stats['active_entries']}",
        f"  Expired: {stats['expired_entries']}",
        f"  Total size: {stats['total_mb']} MB ({stats['total_bytes']:,} bytes)",
        f"  Location: {stats['cache_dir']}",
        "",
        "By source type:",
    ]

    for stype, info in stats.get("by_source_type", {}).items():
        size_mb = round(info["total_bytes"] / (1024 * 1024), 2)
        lines.append(f"  {stype}: {info['count']} entries, {size_mb} MB")

    return "\n".join(lines)


@tool
def cache_evict() -> str:
    """Evict expired entries from the cache to free disk space.

    Removes entries past their TTL and cleans up orphan blobs
    (blobs not referenced by any active entry).

    Returns:
        Number of entries evicted.
    """
    count = cache_evict_expired()
    return f"Evicted {count} expired cache entries."


# ── Tool registry ─────────────────────────────────────────────────────

CACHE_TOOLS = [
    cache_store_content,
    cache_lookup,
    cache_search_entries,
    cache_statistics,
    cache_evict,
]
