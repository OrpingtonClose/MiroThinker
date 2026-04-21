# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Persistent discovery store for exhaustive discovery mode.

Stores ALL discovered items (name + URL + metadata) in a JSONL-backed
store so nothing is ever trimmed from the LLM's awareness.  Instead of
dumping raw JSON into the prompt, the LLM receives a compact digest
(counts by category) and can drill down via the ``check_discovered``
and ``list_discovered`` tools.

Architecture:
  1. Items land in the store immediately on discovery (zero trimming)
  2. An automatic digest builder groups items by domain/category
  3. The LLM gets the digest + tool access instead of a raw JSON dump
  4. Zero data loss — the store has everything, the prompt has a summary
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# ── Storage ──────────────────────────────────────────────────────────

_STORE_DIR = Path(os.environ.get(
    "FINDINGS_DIR", os.path.join(os.path.expanduser("~"), ".mirothinker")
))
_STORE_DIR.mkdir(parents=True, exist_ok=True)

_store_file: Path = _STORE_DIR / "discovery_items.jsonl"

# In-memory store — persisted to JSONL on every mutation
_items: list[dict] = []
_seen_urls: set[str] = set()


def _persist() -> None:
    """Write all items to JSONL."""
    with open(_store_file, "w") as f:
        for item in _items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_store() -> None:
    """Load items from JSONL file (called on startup / resume)."""
    global _items, _seen_urls
    _items = []
    _seen_urls = set()
    if not _store_file.exists():
        return
    for line in _store_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        _items.append(obj)
        url = obj.get("url", "")
        if url:
            _seen_urls.add(url)
    logger.info("discovery_store | loaded %d items from %s", len(_items), _store_file)


def clear_store() -> None:
    """Clear the in-memory store and delete the JSONL file."""
    global _items, _seen_urls
    _items = []
    _seen_urls = set()
    if _store_file.exists():
        _store_file.unlink()


def add_items(new_items: list[dict]) -> int:
    """Add new items to the store, deduplicating by URL.

    Args:
        new_items: List of dicts with at least 'name' and 'url' keys.

    Returns:
        Number of genuinely new items added.
    """
    added = 0
    for item in new_items:
        url = item.get("url", "")
        if url and url in _seen_urls:
            continue
        if url:
            _seen_urls.add(url)
        item["_discovered_at"] = time.time()
        _items.append(item)
        added += 1
    if added > 0:
        _persist()
    return added


def get_all_items() -> list[dict]:
    """Return the full list of discovered items (for final output)."""
    return list(_items)


def get_seen_urls() -> set[str]:
    """Return the set of already-discovered URLs."""
    return set(_seen_urls)


def total_count() -> int:
    """Return total number of discovered items."""
    return len(_items)


# ── Digest builder (automatic summarisation) ─────────────────────────


def _extract_domain(url: str) -> str:
    """Extract a readable domain category from a URL."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Strip www prefix
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return "unknown"


def _categorise_domain(domain: str) -> str:
    """Map a domain to a broad category for the digest."""
    academic = {"pubmed", "ncbi", "arxiv", "scholar.google", "semanticscholar",
                "doi.org", "sciencedirect", "springer", "wiley", "nature.com",
                "bmj.com", "thelancet", "pnas.org", "cell.com", "plos"}
    forums = {"reddit.com", "forums.", "forum.", "quora.com", "stackexchange",
              "bodybuilding.com", "t-nation.com", "muscletalk"}
    news = {"bbc", "cnn", "reuters", "nytimes", "guardian", "washingtonpost",
            "apnews", "bloomberg", "vice", "wired"}
    vendor = {"amazon", "ebay", "alibaba", "shopify", "etsy"}
    video = {"youtube.com", "youtu.be", "vimeo", "twitch", "rumble"}
    wiki = {"wikipedia.org", "wikimedia", "wiktionary"}
    gov = {".gov", ".mil", ".edu"}

    lower = domain.lower()
    for kw in academic:
        if kw in lower:
            return "academic"
    for kw in forums:
        if kw in lower:
            return "forum"
    for kw in news:
        if kw in lower:
            return "news"
    for kw in vendor:
        if kw in lower:
            return "vendor"
    for kw in video:
        if kw in lower:
            return "video"
    for kw in wiki:
        if kw in lower:
            return "wiki"
    for kw in gov:
        if kw in lower:
            return "government"
    return "web"


def build_digest() -> str:
    """Build a compact digest of all discovered items.

    Groups items by category and domain, producing a summary like:
      "147 items discovered: 42 academic (pubmed: 18, arxiv: 12, ...),
       28 forum (reddit: 15, ...), 35 news (...), 42 web (...)"

    This goes into the prompt instead of the raw JSON list.

    Returns:
        Human-readable digest string.
    """
    if not _items:
        return "No items discovered yet."

    # Group by category and domain
    category_counts: Counter[str] = Counter()
    domain_by_category: dict[str, Counter[str]] = {}

    for item in _items:
        url = item.get("url", "")
        domain = _extract_domain(url) if url else "no-url"
        category = _categorise_domain(domain)
        category_counts[category] += 1
        if category not in domain_by_category:
            domain_by_category[category] = Counter()
        domain_by_category[category][domain] += 1

    # Build readable digest
    total = len(_items)
    parts = []
    for cat, count in category_counts.most_common():
        top_domains = domain_by_category[cat].most_common(3)
        domain_str = ", ".join(f"{d}: {c}" for d, c in top_domains)
        if len(domain_by_category[cat]) > 3:
            domain_str += f", +{len(domain_by_category[cat]) - 3} more"
        parts.append(f"{count} {cat} ({domain_str})")

    return f"{total} items discovered: {'; '.join(parts)}"


# ── FunctionTool implementations (LLM-callable) ─────────────────────


async def check_discovered(url: str = "", name: str = "") -> str:
    """Check if an item has already been discovered.

    Use this before adding a new item to avoid duplicates. You can
    search by URL (exact match) or name (substring match).

    Args:
        url: URL to check for exact match.
        name: Name substring to search for (case-insensitive).

    Returns:
        JSON with match status and any matching items.
    """
    matches = []

    if url:
        for item in _items:
            if item.get("url", "") == url:
                matches.append(item)

    if name:
        name_lower = name.lower()
        for item in _items:
            item_name = item.get("name", "").lower()
            if name_lower in item_name:
                matches.append(item)

    # Deduplicate
    seen = set()
    unique_matches = []
    for m in matches:
        key = m.get("url", "") or m.get("name", "")
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    return json.dumps({
        "found": len(unique_matches) > 0,
        "match_count": len(unique_matches),
        "matches": unique_matches,
    }, ensure_ascii=False)


async def list_discovered(
    category: str = "",
    domain: str = "",
    limit: int = 50,
) -> str:
    """List discovered items, optionally filtered by category or domain.

    Args:
        category: Filter by category (academic, forum, news, web, video, etc).
        domain: Filter by domain substring (e.g. "reddit", "pubmed").
        limit: Maximum items to return (default 50). Use 0 for all.

    Returns:
        JSON with filtered items and total count.
    """
    filtered = []
    for item in _items:
        url = item.get("url", "")
        item_domain = _extract_domain(url) if url else ""
        item_category = _categorise_domain(item_domain)

        if category and item_category != category.lower():
            continue
        if domain and domain.lower() not in item_domain.lower():
            continue
        filtered.append({
            "name": item.get("name", ""),
            "url": url,
            "category": item_category,
            "domain": item_domain,
        })

    total = len(filtered)
    if limit > 0:
        filtered = filtered[:limit]

    return json.dumps({
        "total_matching": total,
        "returned": len(filtered),
        "items": filtered,
    }, ensure_ascii=False)


# ── Public FunctionTool instances ─────────────────────────────────────

check_discovered_tool = FunctionTool(check_discovered)
list_discovered_tool = FunctionTool(list_discovered)

DISCOVERY_TOOLS = [
    check_discovered_tool,
    list_discovered_tool,
]
