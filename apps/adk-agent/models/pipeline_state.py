# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""PipelineStateRegistry -- typed, TTL-aware corpus continuity store.

Replaces the ad-hoc module-level variables ``_last_corpus_db_path``,
``_corpus_continuity_map``, and ``_prev_corpus_db_path`` with a single
typed abstraction that:

- Indexes by query fingerprint (first 200 chars, lowered, stripped)
- Supports TTL-based eviction (default 2 hours)
- Is thread-safe via a ``threading.Lock``
- Provides explicit ``register()`` / ``lookup()`` / ``evict()`` API

This ensures corpus continuity across AG-UI HTTP requests without
relying on fragile module-level mutable state.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default TTL: 2 hours.  Corpora older than this are evicted on next
# ``lookup()`` or ``evict()`` call.  Configurable via environment.
_DEFAULT_TTL_SECONDS = 7200


def _fingerprint(query: str) -> str:
    """Stable fingerprint for a user query (lowercase, trimmed, 200 chars)."""
    return query[:200].strip().lower()


@dataclass
class _CorpusEntry:
    """A single registered corpus with metadata."""
    db_path: str
    query_fingerprint: str
    registered_at: float = field(default_factory=time.time)
    iteration_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class PipelineStateRegistry:
    """Thread-safe registry mapping query fingerprints to corpus DB paths.

    Usage::

        registry = PipelineStateRegistry.instance()
        registry.register("what are health benefits of turmeric", "/path/to/corpus.duckdb")
        path = registry.lookup("what are health benefits of turmeric")
    """

    _instance: Optional["PipelineStateRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._entries: dict[str, _CorpusEntry] = {}  # fingerprint → entry
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        # Backwards-compatible fallback: last registered path (any query)
        self._last_path: str = ""

    @classmethod
    def instance(cls) -> "PipelineStateRegistry":
        """Get the module-level singleton (lazy, thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def register(
        self,
        query: str,
        db_path: str,
        iteration_count: int = 0,
    ) -> None:
        """Register a corpus DB path for a query fingerprint."""
        fp = _fingerprint(query)
        with self._lock:
            self._entries[fp] = _CorpusEntry(
                db_path=db_path,
                query_fingerprint=fp,
                iteration_count=iteration_count,
            )
            self._last_path = db_path
            logger.info(
                "PipelineStateRegistry: registered corpus for "
                "fp='%.40s...' → %s (iter=%d)",
                fp, db_path, iteration_count,
            )

    def lookup(self, query: str) -> Optional[str]:
        """Look up the corpus DB path for a query.

        Returns ``None`` if no entry exists or if the entry has expired.
        Automatically evicts expired entries on access.
        """
        fp = _fingerprint(query)
        now = time.time()
        with self._lock:
            entry = self._entries.get(fp)
            if entry is None:
                return None
            if now - entry.registered_at > self._ttl:
                del self._entries[fp]
                logger.debug(
                    "PipelineStateRegistry: evicted expired entry for fp='%.40s...'",
                    fp,
                )
                return None
            entry.last_accessed = now
            return entry.db_path

    def lookup_fallback(self) -> Optional[str]:
        """Return the last registered path (any query) as a fallback.

        This preserves the old ``_last_corpus_db_path`` behavior for
        edge cases where the query fingerprint doesn't match.
        """
        with self._lock:
            return self._last_path or None

    def evict_expired(self) -> int:
        """Remove all entries older than TTL.  Returns count evicted."""
        now = time.time()
        evicted = 0
        with self._lock:
            expired = [
                fp for fp, entry in self._entries.items()
                if now - entry.registered_at > self._ttl
            ]
            for fp in expired:
                del self._entries[fp]
                evicted += 1
        if evicted:
            logger.info("PipelineStateRegistry: evicted %d expired entries", evicted)
        return evicted

    def remove(self, query: str) -> bool:
        """Explicitly remove an entry.  Returns True if found."""
        fp = _fingerprint(query)
        with self._lock:
            if fp in self._entries:
                del self._entries[fp]
                return True
        return False

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def snapshot(self) -> dict[str, str]:
        """Return a copy of all fingerprint → db_path mappings."""
        with self._lock:
            return {fp: e.db_path for fp, e in self._entries.items()}
