# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Book acquisition pipeline for the Strands research agent.

Multi-source book search, download, text extraction, and persistent caching.
Designed for documentary research, deep knowledge base construction, and
long-horizon agent tasks that need access to full book/textbook content.

Sources (ordered by reliability):
  1. Open Library / Internet Archive — free, legal, public API
  2. Library Genesis — HTML scraping + MD5 download resolution
  3. Anna's Archive — HTML scraping (API requires donation key)
  4. Sci-Hub — DOI → PDF for paywalled papers
  5. Project Gutenberg — free, legal, classic works
  6. Google Books — metadata enrichment, ISBN lookup (free API)
  7. ISBNdb — deep book metadata, 108M+ titles (paid)
  8. HathiTrust — 17M+ digitized books, public domain OCR text (free)
  9. Internet Archive (full API) — advanced search, bulk download (free)
  10. Unpaywall — legal open-access PDF finder for DOIs (free)
  11. CrossRef — citation graph traversal, DOI metadata (free)

Format support:
  - PDF → pdfplumber (pure Python)
  - EPUB → ebooklib + BeautifulSoup (pure Python)
  - Plain text / HTML → direct

All downloaded content is stored in the persistent cache (cache.py) with
source_type="book" for cross-session reuse. Books are indexed by title,
author, ISBN, and content hash for deduplication.

Resolves: https://github.com/OrpingtonClose/MiroThinker/issues/92
Depends: https://github.com/OrpingtonClose/MiroThinker/issues/93 (cache)
"""

from __future__ import annotations

import html as html_module
import io
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote, quote_plus, urljoin

from strands import tool

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

# LibGen mirrors — ordered by reliability (updated 2025)
LIBGEN_MIRRORS = [
    os.environ.get("LIBGEN_MIRROR", "https://libgen.li"),
    "https://libgen.is",
    "https://libgen.rs",
]

# Anna's Archive domains — .org was suspended Jan 2026
ANNAS_DOMAINS = [
    os.environ.get("ANNAS_DOMAIN", "https://annas-archive.se"),
    "https://annas-archive.li",
    "https://annas-archive.org",
]

# Sci-Hub domains
SCIHUB_DOMAINS = [
    os.environ.get("SCIHUB_DOMAIN", "https://sci-hub.se"),
    "https://sci-hub.st",
    "https://sci-hub.ru",
]

# ── Proxy configuration ───────────────────────────────────────────────
# Bright Data and Oxylabs provide residential proxies that bypass cloud IP
# blocks on LibGen, Anna's Archive, Sci-Hub, etc.

BRIGHT_DATA_API_KEY = os.environ.get("BRIGHT_DATA_API_KEY", "")
BRIGHT_DATA_CUSTOMER_ID = os.environ.get("BRIGHTDATA_CUSTOMER_ID", "hl_dc044bf4")
BRIGHT_DATA_ZONE = os.environ.get("BRIGHTDATA_ZONE", "mcp_unlocker")

OXYLABS_USERNAME = os.environ.get("OXYLABS_USERNAME", "")
OXYLABS_PASSWORD = os.environ.get("OXYLABS_PASSWORD", "")

# ── New service API keys ──────────────────────────────────────────────

# Unpaywall — free, just needs an email address (no key)
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")

# Google Books API — free, optional API key for higher quotas
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "")

# ISBNdb — paid service, 108M+ titles, 19 data points per book
ISBNDB_API_KEY = os.environ.get("ISBNDB_API_KEY", "")

# Springer Nature — free key, 280K+ open access articles
SPRINGER_API_KEY = os.environ.get("SPRINGER_API_KEY", "")

# HathiTrust — free, 17M+ digitized books
# No API key needed for basic catalog/metadata access

# Internet Archive — free, S3-like API for bulk ops
IA_ACCESS_KEY = os.environ.get("IA_ACCESS_KEY", "")
IA_SECRET_KEY = os.environ.get("IA_SECRET_KEY", "")


# ── Text extraction helpers ───────────────────────────────────────────


def _extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 100) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages[:max_pages]):
                text = page.extract_text()
                if text:
                    pages.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(pages) if pages else "[No text extracted from PDF]"
        except ImportError:
            return "[ERROR] No PDF library. Install: pip install pdfplumber"

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(pages) if pages else "[No text extracted from PDF]"


def _extract_text_from_epub(epub_bytes: bytes) -> str:
    """Extract text from EPUB bytes using ebooklib + BeautifulSoup."""
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        return "[ERROR] ebooklib not installed. Install: pip install EbookLib"

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return "[ERROR] beautifulsoup4 not installed. Install: pip install beautifulsoup4"

    # Write to temp file (ebooklib needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(epub_bytes)
        tmp_path = f.name

    try:
        book = epub.read_epub(tmp_path, options={"ignore_ncx": True})

        # Extract metadata
        meta_lines = []
        title = book.get_metadata("DC", "title")
        if title:
            meta_lines.append(f"Title: {title[0][0]}")
        creator = book.get_metadata("DC", "creator")
        if creator:
            meta_lines.append(f"Author: {creator[0][0]}")
        language = book.get_metadata("DC", "language")
        if language:
            meta_lines.append(f"Language: {language[0][0]}")

        # Extract table of contents
        toc = book.toc
        toc_lines = []
        if toc:
            toc_lines.append("\n--- TABLE OF CONTENTS ---")
            for item in toc:
                if isinstance(item, tuple):
                    # Nested section
                    section, children = item
                    toc_lines.append(f"  {section.title}")
                    for child in children:
                        toc_lines.append(f"    - {child.title}")
                else:
                    toc_lines.append(f"  {item.title}")

        # Extract text from all document items
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")

                # Try to find chapter title
                chapter_title = ""
                for tag in ["h1", "h2", "h3"]:
                    heading = soup.find(tag)
                    if heading:
                        chapter_title = heading.get_text(strip=True)
                        break

                text = soup.get_text(separator="\n", strip=True)
                if text and len(text.strip()) > 50:  # Skip near-empty pages
                    header = f"\n--- {chapter_title} ---" if chapter_title else "\n--- Section ---"
                    chapters.append(f"{header}\n{text}")

        if not chapters:
            return "[No text extracted from EPUB]"

        parts = []
        if meta_lines:
            parts.append("\n".join(meta_lines))
        if toc_lines:
            parts.append("\n".join(toc_lines))
        parts.append("\n".join(chapters))

        return "\n\n".join(parts)

    except Exception as exc:
        return f"[ERROR] EPUB extraction failed: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_text(content: bytes, filename: str = "", content_type: str = "") -> str:
    """Auto-detect format and extract text."""
    lower_name = filename.lower()
    lower_ct = content_type.lower()

    if lower_name.endswith(".epub") or "epub" in lower_ct:
        return _extract_text_from_epub(content)
    elif lower_name.endswith(".pdf") or "pdf" in lower_ct:
        return _extract_text_from_pdf(content)
    elif lower_name.endswith((".htm", ".html")) or "html" in lower_ct:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            return content.decode("utf-8", errors="replace")
    elif lower_name.endswith((".txt", ".md", ".rst")) or "text/" in lower_ct:
        return content.decode("utf-8", errors="replace")
    else:
        # Try PDF first (most common for academic content)
        if content[:5] == b"%PDF-":
            try:
                return _extract_text_from_pdf(content)
            except Exception:
                pass
        # Try EPUB (ZIP-based)
        if content[:2] == b"PK":
            try:
                return _extract_text_from_epub(content)
            except Exception:
                pass
        # Fallback to UTF-8
        return content.decode("utf-8", errors="replace")


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_module.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ── HTTP helpers ──────────────────────────────────────────────────────

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _get_http_client(timeout: int = 30, use_proxy: bool = False):
    """Get httpx client with browser-like headers.

    When use_proxy=True, routes through Bright Data or Oxylabs residential
    proxy to bypass cloud IP blocks on shadow libraries.
    """
    import httpx

    proxy_url = None
    if use_proxy:
        proxy_url = _get_proxy_url()

    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=15 if use_proxy else 10),
        follow_redirects=True,
        proxy=proxy_url,
        verify=True,  # keep SSL verification even with proxies
        headers={"User-Agent": _BROWSER_UA},
    )


def _get_proxy_url() -> Optional[str]:
    """Build a proxy URL from Bright Data or Oxylabs credentials.

    Tries Bright Data first (Web Unlocker), then Oxylabs as fallback.
    Returns None if neither is configured.
    """
    if BRIGHT_DATA_API_KEY:
        return (
            f"https://brd-customer-{BRIGHT_DATA_CUSTOMER_ID}"
            f"-zone-{BRIGHT_DATA_ZONE}"
            f":{BRIGHT_DATA_API_KEY}@brd.superproxy.io:33335"
        )
    if OXYLABS_USERNAME and OXYLABS_PASSWORD:
        return (
            f"https://{OXYLABS_USERNAME}:{OXYLABS_PASSWORD}"
            f"@unblock.oxylabs.io:60000"
        )
    return None


def _has_proxy() -> bool:
    """Check if any proxy is configured."""
    return bool(BRIGHT_DATA_API_KEY or (OXYLABS_USERNAME and OXYLABS_PASSWORD))


def _try_mirrors(mirrors: list[str], path_fn, **kwargs) -> Optional[object]:
    """Try a request across multiple mirror domains.

    First attempts direct connection on all mirrors, then retries with
    proxy if available (shadow libraries often block cloud IPs).
    """
    import httpx

    # Pass 1: direct connection
    for mirror in mirrors:
        try:
            url = path_fn(mirror)
            with _get_http_client() as client:
                resp = client.get(url, **kwargs)
                if resp.status_code == 200:
                    return resp
        except Exception as exc:
            logger.debug("Mirror %s failed (direct): %s", mirror, exc)
            continue

    # Pass 2: retry with proxy if available
    if _has_proxy():
        logger.info("Direct mirrors failed — retrying with proxy")
        for mirror in mirrors:
            try:
                url = path_fn(mirror)
                with _get_http_client(timeout=45, use_proxy=True) as client:
                    resp = client.get(url, **kwargs)
                    if resp.status_code == 200:
                        logger.info("Proxy succeeded for mirror %s", mirror)
                        return resp
            except Exception as exc:
                logger.debug("Mirror %s failed (proxy): %s", mirror, exc)
                continue

    return None


# ── Cache integration ─────────────────────────────────────────────────


def _cache_store_book(url: str, content: bytes, text: str, title: str,
                      author: str = "", metadata: dict = None,
                      tags: list[str] = None) -> bool:
    """Store a book in the persistent cache. Returns True on success."""
    try:
        from cache import cache_put
        tag_list = list(tags) if tags else []
        if "book" not in tag_list:
            tag_list.append("book")

        # Store extracted text (main entry — what the agent reads)
        # Books are static content — use 10-year TTL to avoid silent expiry
        book_ttl = 10 * 365 * 24 * 3600
        cache_put(
            url=url,
            content=text,
            content_type="text/plain",
            source_type="book",
            title=title,
            summary=text[:500],
            tags=tag_list,
            metadata=metadata or {},
            ttl=book_ttl,
        )

        # Store raw binary (for re-extraction or format conversion)
        if content and len(content) > 0:
            cache_put(
                url=f"raw-book://{url}",
                content=content,
                content_type="application/octet-stream",
                source_type="book_raw",
                title=f"[RAW] {title}",
                summary=f"Raw binary, {len(content):,} bytes",
                tags=tag_list,
                ttl=book_ttl,
            )

        logger.info("Cached book: %s (%d chars text, %d bytes raw)",
                     title, len(text), len(content))
        return True
    except ImportError:
        logger.debug("Cache not available — book not persisted")
        return False


def _cache_lookup_book(url: str) -> Optional[str]:
    """Look up a book in cache by URL. Returns text if found."""
    try:
        from cache import cache_get
        cached = cache_get(url=url)
        if cached and cached.get("content"):
            try:
                return cached["content"].decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                if isinstance(cached["content"], str):
                    return cached["content"]
    except ImportError:
        pass
    return None


def _doi_cache_alias(cache_key: str, result: str, title: str) -> None:
    """Cache a download result under the doi:// key for future DOI lookups."""
    try:
        from cache import cache_put
        cache_put(
            url=cache_key,
            content=result,
            content_type="text/plain",
            source_type="book",
            title=title or "Unknown",
            summary=result[:500],
            tags=["book", "doi"],
            ttl=10 * 365 * 24 * 3600,
        )
    except ImportError:
        pass


def _datalake_store(content: bytes, text: str, title: str, author: str,
                    filename: str, md5: str, doi: str, source: str,
                    cache_key: str) -> None:
    """Upload artifact to B2 datalake as RO-Crate. Non-blocking — failures are logged."""
    try:
        from datalake import store_artifact, is_configured
        if not is_configured():
            return

        # Determine file extension from filename
        ext = ""
        if filename and "." in filename:
            ext = filename.rsplit(".", 1)[-1].lower()
        if not ext:
            if content[:5] == b"%PDF-":
                ext = "pdf"
            elif content[:2] == b"PK":
                ext = "epub"
            else:
                ext = "txt"

        store_artifact(
            raw_content=content,
            extracted_text=text,
            category="books",
            metadata={
                "title": title or filename,
                "author": author,
                "source": source,
                "source_url": cache_key if cache_key.startswith("http") else "",
                "md5": md5,
                "doi": doi,
                "filename": filename,
            },
            file_extension=ext,
        )
    except ImportError:
        logger.debug("Datalake module not available — skipping B2 upload")
    except Exception as exc:
        logger.warning("Datalake upload failed (non-fatal): %s", exc)


def _datalake_lookup(content_hash: str) -> Optional[str]:
    """Look up extracted text from B2 datalake by content hash."""
    try:
        from datalake import get_artifact_text, is_configured
        if not is_configured():
            return None
        return get_artifact_text(content_hash)
    except ImportError:
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 1: OPEN LIBRARY / INTERNET ARCHIVE
# ═══════════════════════════════════════════════════════════════════════


def _search_open_library(query: str, max_results: int = 10) -> list[dict]:
    """Search Open Library's public API."""
    import httpx

    results = []
    try:
        with _get_http_client() as client:
            resp = client.get(
                "https://openlibrary.org/search.json",
                params={
                    "q": query,
                    "limit": max_results,
                    "fields": "key,title,author_name,first_publish_year,isbn,language,number_of_pages_median,edition_count,ia,availability",
                },
            )
            if resp.status_code != 200:
                return results

            data = resp.json()
            for doc in data.get("docs", [])[:max_results]:
                ia_ids = doc.get("ia", [])
                has_fulltext = bool(ia_ids)

                results.append({
                    "title": doc.get("title", "Unknown"),
                    "authors": doc.get("author_name", []),
                    "year": doc.get("first_publish_year", ""),
                    "isbn": (doc.get("isbn", []) or [""])[0],
                    "pages": doc.get("number_of_pages_median", ""),
                    "editions": doc.get("edition_count", 0),
                    "languages": doc.get("language", []),
                    "has_fulltext": has_fulltext,
                    "ia_id": ia_ids[0] if ia_ids else "",
                    "open_library_key": doc.get("key", ""),
                    "url": f"https://openlibrary.org{doc.get('key', '')}",
                    "source": "Open Library",
                })
    except Exception as exc:
        logger.debug("Open Library search failed: %s", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 2: LIBRARY GENESIS
# ═══════════════════════════════════════════════════════════════════════


def _search_libgen(query: str, max_results: int = 10) -> list[dict]:
    """Search Library Genesis via HTML scraping.

    Tries direct connection first, then retries with proxy if available.
    """
    import httpx

    results = []

    for use_proxy in ([False, True] if _has_proxy() else [False]):
        for mirror in LIBGEN_MIRRORS:
            try:
                search_url = (
                    f"{mirror}/search.php?"
                    f"req={quote_plus(query)}&lg_topic=libgen&open=0"
                    f"&view=simple&res={max_results}&phrase=1&column=def"
                )

                with _get_http_client(timeout=45 if use_proxy else 30,
                                      use_proxy=use_proxy) as client:
                    resp = client.get(search_url)

                if resp.status_code != 200:
                    continue

                html = resp.text
                results = _parse_libgen_results(html, mirror, max_results)
                if results:
                    if use_proxy:
                        logger.info("LibGen search succeeded via proxy (%s)", mirror)
                    return results

            except Exception as exc:
                logger.debug("LibGen mirror %s failed (%s): %s",
                             mirror, "proxy" if use_proxy else "direct", exc)
                continue

        if results:
            break

    return results


def _parse_libgen_results(html: str, mirror: str, max_results: int) -> list[dict]:
    """Parse LibGen search results from HTML table."""
    results = []

    # LibGen results are in a table with class "c" or in table rows
    # Each row has: ID, Author(s), Title, Publisher, Year, Pages, Language, Size, Extension, Mirror links
    # We look for rows that contain MD5 links

    # Find all MD5 hashes in download links
    md5_pattern = re.compile(r'md5=([a-fA-F0-9]{32})', re.IGNORECASE)

    # Try to extract table rows
    # Pattern: <tr ...> cells </tr> where one cell has an <a> with book title
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL | re.IGNORECASE)

    rows = row_pattern.findall(html)

    for row in rows:
        cells = cell_pattern.findall(row)
        if len(cells) < 8:
            continue

        # Extract MD5 from this row
        md5_match = md5_pattern.search(row)
        if not md5_match:
            continue

        md5 = md5_match.group(1).lower()

        # Extract fields from cells (typical LibGen table order):
        # ID, Author, Title (with link), Publisher, Year, Pages, Language, Size, Extension
        try:
            author = _strip_html(cells[1])[:200]
            title_html = cells[2]
            # Extract title from <a> tag
            title_match = re.search(r'<a[^>]*>(.*?)</a>', title_html, re.DOTALL)
            title = _strip_html(title_match.group(1)) if title_match else _strip_html(title_html)
            title = title[:300]

            publisher = _strip_html(cells[3])[:100]
            year = _strip_html(cells[4])[:10]
            pages = _strip_html(cells[5])[:10]
            language = _strip_html(cells[6])[:20]
            size = _strip_html(cells[7])[:20]
            extension = _strip_html(cells[8])[:10] if len(cells) > 8 else ""
        except (IndexError, AttributeError):
            continue

        if not title or len(title) < 3:
            continue

        results.append({
            "title": title,
            "authors": [a.strip() for a in author.split(";") if a.strip()],
            "year": year,
            "publisher": publisher,
            "pages": pages,
            "language": language,
            "size": size,
            "format": extension.lower(),
            "md5": md5,
            "url": f"{mirror}/get.php?md5={md5}",
            "download_page": f"{mirror}/get.php?md5={md5}",
            "source": "LibGen",
        })

        if len(results) >= max_results:
            break

    return results


def _download_libgen_book(md5: str) -> Optional[tuple[bytes, str]]:
    """Download a book from LibGen by MD5 hash. Returns (content, filename) or None.

    Tries direct connection first, then retries with proxy if available.
    """
    import httpx

    for use_proxy in ([False, True] if _has_proxy() else [False]):
        for mirror in LIBGEN_MIRRORS:
            try:
                get_url = f"{mirror}/get.php?md5={md5}"

                with _get_http_client(timeout=45 if use_proxy else 30,
                                      use_proxy=use_proxy) as client:
                    resp = client.get(get_url)

                if resp.status_code != 200:
                    continue

                # The get.php page usually contains a download link
                html = resp.text

                # Pattern 1: Direct download link (e.g., <a href="...">GET</a>)
                download_links = re.findall(
                    r'<a\s+href="([^"]+)"[^>]*>\s*(?:GET|Download|Click)\s*</a>',
                    html, re.IGNORECASE
                )

                # Pattern 2: Any link to a file with a book extension
                if not download_links:
                    download_links = re.findall(
                        r'href="([^"]*\.(?:pdf|epub|djvu|mobi|azw3)[^"]*)"',
                        html, re.IGNORECASE
                    )

                # Pattern 3: Cloudflare/library.lol links
                if not download_links:
                    download_links = re.findall(
                        r'href="(https?://[^"]*(?:cloudflare|library\.lol|download)[^"]*)"',
                        html, re.IGNORECASE
                    )

                for link in download_links:
                    if link.startswith("/"):
                        link = f"{mirror}{link}"
                    elif not link.startswith("http"):
                        link = urljoin(get_url, link)

                    try:
                        with _get_http_client(timeout=60 if use_proxy else 30,
                                              use_proxy=use_proxy) as dl_client:
                            dl_resp = dl_client.get(link)
                        if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                            cd = dl_resp.headers.get("content-disposition", "")
                            fn_match = re.search(r'filename="?([^";\n]+)', cd)
                            filename = fn_match.group(1).strip() if fn_match else f"book_{md5[:8]}.pdf"
                            if use_proxy:
                                logger.info("LibGen download succeeded via proxy (%s)", mirror)
                            return dl_resp.content, filename
                    except Exception as exc:
                        logger.debug("Download link %s failed: %s", link[:80], exc)
                        continue

            except Exception as exc:
                logger.debug("LibGen mirror %s failed for MD5 %s (%s): %s",
                             mirror, md5, "proxy" if use_proxy else "direct", exc)
                continue

    return None


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 3: ANNA'S ARCHIVE
# ═══════════════════════════════════════════════════════════════════════


def _search_annas_archive(query: str, max_results: int = 10,
                          content_type: str = "") -> list[dict]:
    """Search Anna's Archive via HTML scraping.

    Tries direct connection first, then retries with proxy if available.
    """
    import httpx

    results = []
    content_filter = f"&content={content_type}" if content_type else ""

    for use_proxy in ([False, True] if _has_proxy() else [False]):
        for domain in ANNAS_DOMAINS:
            try:
                search_url = f"{domain}/search?q={quote_plus(query)}{content_filter}"

                with _get_http_client(timeout=45 if use_proxy else 30,
                                      use_proxy=use_proxy) as client:
                    resp = client.get(search_url)

                if resp.status_code != 200:
                    continue

                html = resp.text
                # Anna's Archive uses /md5/ links for results
                md5_links = re.findall(r'/md5/([a-f0-9]{32})', html)
                seen = set()

                for md5 in md5_links:
                    if md5 in seen:
                        continue
                    seen.add(md5)

                    pattern = re.compile(
                        rf'href="/md5/{md5}"[^>]*>(.*?)</a>',
                        re.DOTALL | re.IGNORECASE
                    )
                    match = pattern.search(html)
                    title = _strip_html(match.group(1))[:300] if match else f"[Book — MD5: {md5[:12]}...]"

                    results.append({
                        "title": title,
                        "md5": md5,
                        "url": f"{domain}/md5/{md5}",
                        "source": "Anna's Archive",
                    })

                    if len(results) >= max_results:
                        break

                if results:
                    if use_proxy:
                        logger.info("Anna's Archive search succeeded via proxy (%s)", domain)
                    return results

            except Exception as exc:
                logger.debug("Anna's Archive %s failed (%s): %s",
                             domain, "proxy" if use_proxy else "direct", exc)
                continue

        if results:
            break

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 4: SCI-HUB (DOI → PDF)
# ═══════════════════════════════════════════════════════════════════════


def _download_from_scihub(doi: str) -> Optional[tuple[bytes, str]]:
    """Download a paper from Sci-Hub by DOI. Returns (content, filename) or None.

    Tries direct connection first, then retries with proxy if available.
    """
    import httpx

    for use_proxy in ([False, True] if _has_proxy() else [False]):
        for domain in SCIHUB_DOMAINS:
            try:
                url = f"{domain}/{doi}"

                with _get_http_client(timeout=45 if use_proxy else 30,
                                      use_proxy=use_proxy) as client:
                    resp = client.get(url)

                if resp.status_code != 200:
                    continue

                html = resp.text

                # Find the PDF embed/iframe
                pdf_match = re.search(
                    r'(?:src|href)="((?:https?:)?//[^"]*\.pdf[^"]*)"',
                    html, re.IGNORECASE
                )
                if not pdf_match:
                    pdf_match = re.search(
                        r'<iframe[^>]*src="([^"]+)"',
                        html, re.IGNORECASE
                    )

                if not pdf_match:
                    continue

                pdf_url = pdf_match.group(1)
                if pdf_url.startswith("//"):
                    pdf_url = "https:" + pdf_url

                with _get_http_client(timeout=60 if use_proxy else 30,
                                      use_proxy=use_proxy) as dl_client:
                    pdf_resp = dl_client.get(pdf_url)

                if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                    safe_doi = re.sub(r'[/\\:*?"<>|]', '_', doi)
                    if use_proxy:
                        logger.info("Sci-Hub download succeeded via proxy (%s)", domain)
                    return pdf_resp.content, f"{safe_doi}.pdf"

            except Exception as exc:
                logger.debug("Sci-Hub %s failed for DOI %s (%s): %s",
                             domain, doi, "proxy" if use_proxy else "direct", exc)
                continue

    return None


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 5: PROJECT GUTENBERG
# ═══════════════════════════════════════════════════════════════════════


def _search_gutenberg(query: str, max_results: int = 10) -> list[dict]:
    """Search Project Gutenberg via the Gutendex API."""
    import httpx

    results = []
    try:
        with _get_http_client() as client:
            resp = client.get(
                "https://gutendex.com/books/",
                params={"search": query, "page": 1},
            )

        if resp.status_code != 200:
            return results

        data = resp.json()
        for book in data.get("results", [])[:max_results]:
            # Find best download format
            formats = book.get("formats", {})
            download_url = ""
            file_format = ""
            for fmt_key in ["text/plain; charset=utf-8", "text/plain",
                            "application/epub+zip", "application/pdf",
                            "text/html; charset=utf-8", "text/html"]:
                if fmt_key in formats:
                    download_url = formats[fmt_key]
                    file_format = fmt_key.split("/")[-1].split(";")[0]
                    break

            authors = [
                a.get("name", "") for a in book.get("authors", [])
            ]

            results.append({
                "title": book.get("title", "Unknown"),
                "authors": authors,
                "year": "",  # Gutenberg doesn't expose year well
                "language": ", ".join(book.get("languages", [])),
                "download_count": book.get("download_count", 0),
                "format": file_format,
                "url": download_url,
                "gutenberg_id": book.get("id", ""),
                "source": "Project Gutenberg",
            })
    except Exception as exc:
        logger.debug("Gutenberg search failed: %s", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 6: GOOGLE BOOKS (metadata enrichment)
# ═══════════════════════════════════════════════════════════════════════


def _search_google_books(query: str, max_results: int = 10) -> list[dict]:
    """Search Google Books API for metadata enrichment."""
    import httpx

    results = []
    try:
        params = {
            "q": query,
            "maxResults": min(max_results, 40),
            "printType": "books",
        }
        if GOOGLE_BOOKS_API_KEY:
            params["key"] = GOOGLE_BOOKS_API_KEY

        with _get_http_client() as client:
            resp = client.get(
                "https://www.googleapis.com/books/v1/volumes",
                params=params,
            )

        if resp.status_code != 200:
            return results

        for item in resp.json().get("items", [])[:max_results]:
            info = item.get("volumeInfo", {})
            access = item.get("accessInfo", {})

            isbn_13 = ""
            isbn_10 = ""
            for ident in info.get("industryIdentifiers", []):
                if ident.get("type") == "ISBN_13":
                    isbn_13 = ident.get("identifier", "")
                elif ident.get("type") == "ISBN_10":
                    isbn_10 = ident.get("identifier", "")

            results.append({
                "title": info.get("title", "Unknown"),
                "authors": info.get("authors", []),
                "year": (info.get("publishedDate", "") or "")[:4],
                "isbn": isbn_13 or isbn_10,
                "pages": info.get("pageCount", ""),
                "language": info.get("language", ""),
                "publisher": info.get("publisher", ""),
                "categories": info.get("categories", []),
                "description": (info.get("description", "") or "")[:300],
                "preview_link": info.get("previewLink", ""),
                "has_fulltext": access.get("viewability", "") in ("ALL_PAGES", "PARTIAL"),
                "url": info.get("infoLink", ""),
                "source": "Google Books",
            })
    except Exception as exc:
        logger.debug("Google Books search failed: %s", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 7: ISBNdb (deep book metadata — 108M+ titles)
# ═══════════════════════════════════════════════════════════════════════


def _search_isbndb(query: str, max_results: int = 10) -> list[dict]:
    """Search ISBNdb for detailed book metadata."""
    import httpx

    if not ISBNDB_API_KEY:
        return []

    results = []
    try:
        with _get_http_client() as client:
            resp = client.get(
                f"https://api2.isbndb.com/books/{quote_plus(query)}",
                headers={"Authorization": ISBNDB_API_KEY},
                params={"pageSize": min(max_results, 20)},
            )

        if resp.status_code != 200:
            return results

        for book in resp.json().get("books", [])[:max_results]:
            results.append({
                "title": book.get("title", "Unknown"),
                "authors": book.get("authors", []),
                "year": str(book.get("date_published", ""))[:4],
                "isbn": book.get("isbn13", book.get("isbn", "")),
                "pages": book.get("pages", ""),
                "publisher": book.get("publisher", ""),
                "language": book.get("language", ""),
                "binding": book.get("binding", ""),
                "dimensions": book.get("dimensions", ""),
                "msrp": book.get("msrp", ""),
                "subjects": book.get("subjects", []),
                "synopsis": (book.get("synopsis", "") or "")[:300],
                "url": f"https://isbndb.com/book/{book.get('isbn13', '')}",
                "source": "ISBNdb",
            })
    except Exception as exc:
        logger.debug("ISBNdb search failed: %s", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 8: HATHITRUST (17M+ digitized books, public domain full text)
# ═══════════════════════════════════════════════════════════════════════


def _search_hathitrust(query: str, max_results: int = 10) -> list[dict]:
    """Search HathiTrust catalog for digitized books."""
    import httpx

    results = []
    try:
        with _get_http_client() as client:
            resp = client.get(
                "https://catalog.hathitrust.org/api/volumes/brief/json/title/{}.json".format(
                    quote(query, safe='')
                ),
            )

        if resp.status_code != 200:
            # Try full-text search via Solr
            with _get_http_client() as client:
                resp = client.get(
                    "https://babel.hathitrust.org/cgi/ls",
                    params={
                        "a": "srchls",
                        "q1": query,
                        "lmt": max_results,
                        "output": "json",
                    },
                )
            if resp.status_code != 200:
                return results

        data = resp.json()

        # Handle brief/volumes response — items is a LIST, records is a DICT
        records_dict = data.get("records", {})
        items_list = data.get("items", [])
        if isinstance(items_list, list) and items_list:
            for item in items_list[:max_results]:
                htid = item.get("htid", "")
                from_record = item.get("fromRecord", "")
                rec = records_dict.get(from_record, {}) if from_record else {}
                has_fulltext = item.get("usRightsString", "") == "Full view"

                authors_raw = rec.get("authors", {})
                author_list = list(authors_raw.keys()) if isinstance(authors_raw, dict) else []

                results.append({
                    "title": " ".join(rec.get("titles", ["Unknown"])),
                    "authors": author_list,
                    "year": rec.get("publishDates", [""])[0] if rec.get("publishDates") else "",
                    "isbn": rec.get("isbns", [""])[0] if rec.get("isbns") else "",
                    "has_fulltext": has_fulltext,
                    "ht_id": htid,
                    "rights": item.get("usRightsString", ""),
                    "url": f"https://babel.hathitrust.org/cgi/pt?id={htid}",
                    "source": "HathiTrust",
                })
        # Handle search response
        elif isinstance(data, list):
            for item in data[:max_results]:
                results.append({
                    "title": item.get("title", "Unknown"),
                    "authors": [item.get("author", "")],
                    "year": item.get("date", ""),
                    "has_fulltext": item.get("rights", "") == "Full view",
                    "ht_id": item.get("htid", ""),
                    "url": f"https://babel.hathitrust.org/cgi/pt?id={item.get('htid', '')}",
                    "source": "HathiTrust",
                })
    except Exception as exc:
        logger.debug("HathiTrust search failed: %s", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 9: INTERNET ARCHIVE (full API — beyond Open Library)
# ═══════════════════════════════════════════════════════════════════════


def _search_internet_archive(query: str, max_results: int = 10) -> list[dict]:
    """Search Internet Archive's advanced search API.

    Goes beyond Open Library to find audiobooks, magazines, historical
    texts, and other formats in the Archive's 40M+ items.
    """
    import httpx

    results = []
    try:
        with _get_http_client() as client:
            resp = client.get(
                "https://archive.org/advancedsearch.php",
                params={
                    "q": f"{query} AND mediatype:texts",
                    "fl[]": "identifier,title,creator,date,description,downloads,format,language,subject",
                    "sort[]": "downloads desc",
                    "rows": max_results,
                    "page": 1,
                    "output": "json",
                },
            )

        if resp.status_code != 200:
            return results

        for doc in resp.json().get("response", {}).get("docs", [])[:max_results]:
            ia_id = doc.get("identifier", "")

            results.append({
                "title": doc.get("title", "Unknown"),
                "authors": [doc["creator"]] if isinstance(doc.get("creator"), str)
                           else (doc.get("creator", []) or []),
                "year": str(doc.get("date", ""))[:4],
                "language": doc.get("language", ""),
                "downloads": doc.get("downloads", 0),
                "subjects": doc.get("subject", []) if isinstance(doc.get("subject"), list) else [],
                "description": (doc.get("description") or "")[:300],
                "ia_id": ia_id,
                "has_fulltext": True,
                "url": f"https://archive.org/details/{ia_id}",
                "download_url": f"https://archive.org/download/{ia_id}",
                "source": "Internet Archive",
            })
    except Exception as exc:
        logger.debug("Internet Archive search failed: %s", exc)

    return results


# ═══════════════════════════════════════════════════════════════════════
# AGENT TOOLS — exposed to the Strands agent
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_books(
    query: str,
    sources: str = "all",
    max_results: int = 10,
) -> str:
    """Search for books, textbooks, and documents across multiple sources.

    Searches up to 9 sources simultaneously: Open Library, Library Genesis,
    Anna's Archive, Project Gutenberg, Google Books, ISBNdb, HathiTrust,
    and Internet Archive. Returns unified results with metadata and
    download availability.

    Use this to find books for documentary research, knowledge base
    construction, or any task requiring deep reading.

    Args:
        query: Search query — title, author, ISBN, topic, etc.
        sources: Which sources to search: "all", "openlibrary", "libgen",
                 "annas", "gutenberg", "googlebooks", "isbndb",
                 "hathitrust", "internetarchive".
                 Comma-separated for multiple (default "all").
        max_results: Maximum results per source (default 10).

    Returns:
        Formatted list of books with metadata and availability.
    """
    source_list = [s.strip().lower() for s in sources.split(",")]
    search_all = "all" in source_list
    all_results = []

    # Search each source
    if search_all or "openlibrary" in source_list:
        try:
            ol_results = _search_open_library(query, max_results)
            all_results.extend(ol_results)
        except Exception as exc:
            logger.debug("Open Library search error: %s", exc)

    if search_all or "libgen" in source_list:
        try:
            lg_results = _search_libgen(query, max_results)
            all_results.extend(lg_results)
        except Exception as exc:
            logger.debug("LibGen search error: %s", exc)

    if search_all or "annas" in source_list:
        try:
            aa_results = _search_annas_archive(query, max_results)
            all_results.extend(aa_results)
        except Exception as exc:
            logger.debug("Anna's Archive search error: %s", exc)

    if search_all or "gutenberg" in source_list:
        try:
            pg_results = _search_gutenberg(query, max_results)
            all_results.extend(pg_results)
        except Exception as exc:
            logger.debug("Gutenberg search error: %s", exc)

    if search_all or "googlebooks" in source_list:
        try:
            gb_results = _search_google_books(query, max_results)
            all_results.extend(gb_results)
        except Exception as exc:
            logger.debug("Google Books search error: %s", exc)

    if search_all or "isbndb" in source_list:
        try:
            isbn_results = _search_isbndb(query, max_results)
            all_results.extend(isbn_results)
        except Exception as exc:
            logger.debug("ISBNdb search error: %s", exc)

    if search_all or "hathitrust" in source_list:
        try:
            ht_results = _search_hathitrust(query, max_results)
            all_results.extend(ht_results)
        except Exception as exc:
            logger.debug("HathiTrust search error: %s", exc)

    if search_all or "internetarchive" in source_list:
        try:
            ia_results = _search_internet_archive(query, max_results)
            all_results.extend(ia_results)
        except Exception as exc:
            logger.debug("Internet Archive search error: %s", exc)

    if not all_results:
        return f"No books found for: {query}\nTry different search terms or check that the sources are accessible."

    # Format results
    formatted = [f"**Book search: {query}** ({len(all_results)} results)\n"]

    for i, r in enumerate(all_results, 1):
        authors_raw = r.get("authors", [])
        if isinstance(authors_raw, dict):
            authors_raw = list(authors_raw.keys())
        authors = ", ".join(str(a) for a in list(authors_raw)[:3]) or "Unknown author"
        year = f" ({r.get('year', '')})" if r.get("year") else ""
        source = r.get("source", "Unknown")
        fmt = f" [{r.get('format', '').upper()}]" if r.get("format") else ""
        size = f" {r.get('size', '')}" if r.get("size") else ""
        pages = f" {r.get('pages', '')}pp" if r.get("pages") else ""
        isbn = f"\n     ISBN: {r.get('isbn', '')}" if r.get("isbn") else ""
        md5 = f"\n     MD5: {r.get('md5', '')}" if r.get("md5") else ""
        dl_info = ""
        if r.get("has_fulltext"):
            src = r.get("source", "")
            if src == "HathiTrust":
                dl_info = "\n     [FULL TEXT AVAILABLE via HathiTrust]"
            elif src == "Internet Archive":
                dl_info = "\n     [FULL TEXT AVAILABLE via Internet Archive]"
            else:
                dl_info = "\n     [FULL TEXT AVAILABLE]"
        elif r.get("md5"):
            dl_info = "\n     [DOWNLOADABLE via LibGen/Anna's Archive]"
        elif r.get("url") and r.get("source") == "Project Gutenberg":
            dl_info = "\n     [FREE DOWNLOAD — public domain]"

        formatted.append(
            f"  {i}. **{r.get('title', 'Unknown')}**\n"
            f"     {authors}{year} | {source}{fmt}{size}{pages}{isbn}{md5}\n"
            f"     URL: {r.get('url', 'N/A')}{dl_info}"
        )

    return "\n\n".join(formatted)


@tool
def download_book(
    url: str = "",
    md5: str = "",
    doi: str = "",
    title: str = "",
    author: str = "",
    source: str = "auto",
) -> str:
    """Download a book/paper from a given URL, MD5 hash, or DOI.

    Downloads the file, extracts text (PDF, EPUB, or plain text),
    and caches it persistently for future sessions. Always check the
    cache before downloading.

    Supports multiple acquisition methods:
    - Direct URL (any HTTP link to a PDF/EPUB)
    - LibGen MD5 hash (resolves download through mirrors)
    - DOI (resolves through Sci-Hub for paywalled papers)
    - Anna's Archive MD5 link

    Args:
        url: Direct download URL (PDF, EPUB, or book page).
        md5: LibGen/Anna's Archive MD5 hash for the book.
        doi: DOI for the paper (uses Sci-Hub as fallback).
        title: Book/paper title (for cache metadata).
        author: Author name(s) (for cache metadata).
        source: Source hint: "auto", "libgen", "scihub", "direct".

    Returns:
        Extracted text from the book, or an error message.
    """
    import httpx

    if not url and not md5 and not doi:
        return "[ERROR] Provide at least one of: url, md5, or doi"

    # Build a cache key
    cache_key = url or (f"md5://{md5}" if md5 else "") or (f"doi://{doi}" if doi else "")

    # Check cache first
    cached = _cache_lookup_book(cache_key)
    if cached:
        return f"[FROM CACHE]\nTitle: {title or 'Unknown'}\n---\n{cached}"

    content = None
    filename = ""
    http_content_type = ""

    # Strategy 1: Direct URL download
    if url and source in ("auto", "direct"):
        try:
            with _get_http_client() as client:
                resp = client.get(url)
            if resp.status_code == 200 and len(resp.content) > 1000:
                content = resp.content
                # Determine filename
                cd = resp.headers.get("content-disposition", "")
                fn_match = re.search(r'filename="?([^";\n]+)', cd)
                filename = fn_match.group(1).strip() if fn_match else url.split("/")[-1].split("?")[0]
                http_content_type = resp.headers.get("content-type", "")
        except Exception as exc:
            logger.debug("Direct download failed: %s", exc)

    # Strategy 2: LibGen MD5
    if content is None and md5 and source in ("auto", "libgen"):
        result = _download_libgen_book(md5)
        if result:
            content, filename = result

    # Strategy 3: Sci-Hub DOI
    if content is None and doi and source in ("auto", "scihub"):
        result = _download_from_scihub(doi)
        if result:
            content, filename = result

    # Strategy 4: If URL looks like an Anna's Archive page, try to find download
    if content is None and url and "/md5/" in url:
        aa_md5 = re.search(r'/md5/([a-f0-9]{32})', url)
        if aa_md5:
            # Try LibGen with this MD5
            result = _download_libgen_book(aa_md5.group(1))
            if result:
                content, filename = result

    if content is None:
        hints = []
        if not md5 and not doi:
            hints.append("Try providing an MD5 hash (from search_books) or DOI")
        if doi:
            hints.append(f"Sci-Hub may be down. Try accessing manually: {SCIHUB_DOMAINS[0]}/{doi}")
        if md5:
            hints.append(f"LibGen mirrors may be down. Try: {LIBGEN_MIRRORS[0]}/get.php?md5={md5}")
        return (
            f"[DOWNLOAD FAILED] Could not acquire the book.\n"
            + "\n".join(hints)
        )

    # Extract text — filename handles extension-based detection,
    # http_content_type carries the MIME type from direct URL downloads
    text = _extract_text(content, filename=filename, content_type=http_content_type)

    if not text or text.startswith("[ERROR]") or text.startswith("[No text"):
        return f"[EXTRACTION FAILED] Downloaded {len(content):,} bytes but could not extract text.\nFilename: {filename}"

    # Cache locally
    _cache_store_book(
        url=cache_key,
        content=content,
        text=text,
        title=title or filename,
        author=author,
        metadata={
            "md5": md5,
            "doi": doi,
            "filename": filename,
            "size_bytes": len(content),
            "source": source,
        },
        tags=["book", source] if source != "auto" else ["book"],
    )

    # Upload to B2 datalake (RO-Crate format) for cross-session persistence
    _datalake_store(content, text, title, author, filename, md5, doi, source, cache_key)

    # Truncate for context window
    text_len = len(text)
    if text_len > 120000:
        text = (
            text[:120000]
            + f"\n\n[...TRUNCATED — full text is {text_len:,} chars. "
            f"Use read_book_section to read specific chapters.]"
        )

    return (
        f"**{title or filename or 'Downloaded book'}**\n"
        f"Author: {author or 'Unknown'}\n"
        f"File: {filename} ({len(content):,} bytes)\n"
        f"Text: {text_len:,} characters extracted\n"
        f"---\n\n{text}"
    )


@tool
def download_book_by_doi(
    doi: str,
    title: str = "",
    author: str = "",
) -> str:
    """Download a paper/book by DOI. Tries Sci-Hub and open access sources.

    Given a DOI (e.g. "10.1017/S0140525X07000891"), attempts to acquire
    the full text through:
    1. CrossRef metadata → open access PDF link
    2. Sci-Hub → direct PDF
    3. Cache lookup for previously downloaded version

    Args:
        doi: The DOI (e.g. "10.1017/S0140525X07000891").
        title: Paper title (optional, for metadata).
        author: Author(s) (optional, for metadata).

    Returns:
        Extracted full text, or error with fallback suggestions.
    """
    # Clean DOI
    doi = doi.strip()
    doi = re.sub(r"^https?://doi\.org/", "", doi)

    cache_key = f"doi://{doi}"
    cached = _cache_lookup_book(cache_key)
    if cached:
        return f"[FROM CACHE]\nDOI: {doi}\nTitle: {title or 'Unknown'}\n---\n{cached}"

    import httpx

    # Strategy 1: Unpaywall — find legal open access version FIRST
    # This is free, fast, and legal — check before anything else
    if UNPAYWALL_EMAIL:
        try:
            resp = httpx.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": UNPAYWALL_EMAIL},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if not title:
                    title = data.get("title", "")
                if data.get("is_oa"):
                    best_oa = data.get("best_oa_location", {}) or {}
                    pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url", "")
                    if pdf_url:
                        logger.info("Unpaywall found OA version: %s (%s)",
                                    doi, best_oa.get("host_type", ""))
                        result = download_book(
                            url=pdf_url, doi=doi, title=title, author=author, source="direct"
                        )
                        if "[DOWNLOAD FAILED]" not in result and "[ERROR]" not in result:
                            _doi_cache_alias(cache_key, result, title)
                            return result
        except Exception as exc:
            logger.debug("Unpaywall lookup failed for %s: %s", doi, exc)

    # Strategy 2: CrossRef for open access PDF link
    try:
        resp = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "MiroThinker/1.0 (mailto:{})".format(
                UNPAYWALL_EMAIL or "research@mirothinker.ai"
            )},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            if not title:
                title = " ".join(data.get("title", ["Unknown"]))
            if not author:
                authors = data.get("author", [])
                author = ", ".join(
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in authors[:3]
                )

            # Look for open-access PDF link
            for link in data.get("link", []):
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL", "")
                    if pdf_url:
                        result = download_book(
                            url=pdf_url, doi=doi, title=title, author=author, source="direct"
                        )
                        if "[DOWNLOAD FAILED]" not in result and "[ERROR]" not in result:
                            _doi_cache_alias(cache_key, result, title)
                            return result
    except Exception:
        pass

    # Strategy 3: Sci-Hub (last resort for paywalled papers)
    return download_book(doi=doi, title=title, author=author, source="scihub")


@tool
def book_library(
    search: str = "",
    limit: int = 50,
) -> str:
    """List all books in the persistent cache, optionally filtered by search.

    Shows all previously downloaded and cached books with their metadata.
    Use this to check what's already available before downloading again.

    Args:
        search: Optional search string to filter by title/tags.
        limit: Maximum number of results (default 50).

    Returns:
        Formatted list of cached books.
    """
    try:
        from cache import get_db
    except ImportError:
        return "[ERROR] Cache not available."

    conn = get_db()

    if search:
        rows = conn.execute(
            """
            SELECT id, url, title, summary, blob_size, created_at, metadata
            FROM cache_entries
            WHERE source_type = 'book'
              AND (title LIKE ? OR summary LIKE ? OR url LIKE ?)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (f"%{search}%", f"%{search}%", f"%{search}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, url, title, summary, blob_size, created_at, metadata
            FROM cache_entries
            WHERE source_type = 'book'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        if search:
            return f"No cached books matching: {search}"
        return "No books in cache yet. Use search_books + download_book to acquire some."

    formatted = [f"**Book Library** ({len(rows)} books)\n"]
    for row in rows:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        size = row["blob_size"]
        size_str = f"{size:,}" if size < 1_000_000 else f"{size / 1_000_000:.1f}M"
        created = time.strftime("%Y-%m-%d", time.localtime(row["created_at"]))

        formatted.append(
            f"  - **{row['title']}**\n"
            f"    {size_str} chars | Cached: {created}\n"
            f"    URL: {row['url']}"
        )

    return "\n\n".join(formatted)


@tool
def read_book_section(
    url: str = "",
    md5: str = "",
    doi: str = "",
    search_text: str = "",
    section_number: int = 0,
    max_chars: int = 15000,
) -> str:
    """Read a specific section of a cached book.

    Use this to read chunks of a large book without loading the entire
    text into context. You can navigate by:
    - Section number (0-based, sections split by headings/page breaks)
    - Text search (finds the section containing the search string)

    Args:
        url: Cache URL of the book.
        md5: MD5 hash (if book was from LibGen/Anna's Archive).
        doi: DOI (if book was a paper).
        search_text: Search for a section containing this text.
        section_number: Which section to read (0-based).
        max_chars: Maximum characters to return (default 15000).

    Returns:
        The requested section of the book, with navigation hints.
    """
    # Build cache key
    cache_key = url or (f"md5://{md5}" if md5 else "") or (f"doi://{doi}" if doi else "")
    if not cache_key:
        return "[ERROR] Provide url, md5, or doi to identify the book"

    text = _cache_lookup_book(cache_key)
    if not text:
        return f"[NOT FOUND] No cached book for: {cache_key}\nUse download_book first."

    # Split into sections (by page breaks, chapter headings, or --- markers)
    sections = re.split(
        r'\n(?=--- (?:Page \d+|Section|Chapter|[A-Z][^-\n]{3,}) ---)',
        text
    )
    if len(sections) <= 1:
        # Try splitting by double newlines if no clear sections
        sections = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    total_sections = len(sections)

    if search_text:
        # Find section containing the search text
        for i, section in enumerate(sections):
            if search_text.lower() in section.lower():
                section_number = i
                break
        else:
            return (
                f"[NOT FOUND] '{search_text}' not found in this book.\n"
                f"Book has {total_sections} sections."
            )

    if section_number >= total_sections:
        return (
            f"[OUT OF RANGE] Section {section_number} doesn't exist.\n"
            f"Book has {total_sections} sections (0-{total_sections - 1})."
        )

    section = sections[section_number][:max_chars]

    return (
        f"**Section {section_number + 1}/{total_sections}**\n"
        f"(Chars: {len(section):,} / {len(text):,} total)\n"
        f"---\n\n{section}\n\n"
        f"---\n"
        f"[Navigate: section_number={max(0, section_number - 1)} (prev) | "
        f"section_number={min(total_sections - 1, section_number + 1)} (next)]"
    )


@tool
def search_book_content(
    query: str,
    max_results: int = 10,
    context_chars: int = 500,
) -> str:
    """Full-text search across ALL cached books.

    Searches the text content of every book in the cache for the
    given query. Returns matching passages with surrounding context.

    Useful for finding specific facts, quotes, or passages across
    your entire book library.

    Args:
        query: Text to search for (case-insensitive).
        max_results: Maximum matching passages to return (default 10).
        context_chars: Characters of context around each match (default 500).

    Returns:
        Matching passages with book titles and locations.
    """
    try:
        from cache import get_db, read_blob
    except ImportError:
        return "[ERROR] Cache not available."

    conn = get_db()

    # Get all book entries
    rows = conn.execute(
        """
        SELECT id, url, title, content_hash, blob_path
        FROM cache_entries
        WHERE source_type = 'book'
        ORDER BY created_at DESC
        """,
    ).fetchall()

    if not rows:
        return "No books in cache. Use search_books + download_book first."

    query_lower = query.lower()
    matches = []

    for row in rows:
        try:
            blob_data = read_blob(row["content_hash"])
            if not blob_data:
                continue
            text = blob_data.decode("utf-8", errors="replace")
        except Exception:
            continue

        text_lower = text.lower()
        start = 0
        while start < len(text_lower) and len(matches) < max_results:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break

            # Extract context
            ctx_start = max(0, pos - context_chars // 2)
            ctx_end = min(len(text), pos + len(query) + context_chars // 2)
            context = text[ctx_start:ctx_end]

            # Highlight the match
            match_start = pos - ctx_start
            match_end = match_start + len(query)
            highlighted = (
                context[:match_start]
                + f"**{context[match_start:match_end]}**"
                + context[match_end:]
            )

            matches.append({
                "title": row["title"],
                "url": row["url"],
                "position": pos,
                "context": highlighted.strip(),
            })

            start = pos + len(query)

    if not matches:
        return f"No matches for '{query}' across {len(rows)} cached books."

    formatted = [f"**Full-text search: '{query}'** ({len(matches)} matches in {len(rows)} books)\n"]
    for i, m in enumerate(matches, 1):
        formatted.append(
            f"  {i}. **{m['title']}** (position {m['position']:,})\n"
            f"     ...{m['context']}...\n"
            f"     URL: {m['url']}"
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# NEW SERVICE TOOLS — Unpaywall, Google Books, HathiTrust, IA, CrossRef
# ═══════════════════════════════════════════════════════════════════════


@tool
def unpaywall_lookup(doi: str) -> str:
    """Look up open access availability for a paper via Unpaywall.

    Given a DOI, checks Unpaywall's database of 50,000+ repositories to find
    legal free versions of paywalled papers. Returns direct PDF links when
    available, along with OA status (gold/green/hybrid/bronze).

    FREE — no API key needed, just an email address (set UNPAYWALL_EMAIL).
    Rate limit: 100,000 calls/day.

    Args:
        doi: The DOI to look up (e.g. "10.1038/s41586-023-06647-8").

    Returns:
        Open access information including PDF links and OA status.
    """
    import httpx

    doi = doi.strip()
    doi = re.sub(r"^https?://doi\.org/", "", doi)

    email = UNPAYWALL_EMAIL
    if not email:
        return (
            "[TOOL_ERROR] UNPAYWALL_EMAIL not set. "
            "Add your email to .env — Unpaywall is free, no API key needed."
        )

    try:
        resp = httpx.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=15,
        )
        if resp.status_code == 404:
            return f"DOI not found in Unpaywall: {doi}"
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Unpaywall lookup failed: {exc}"

    data = resp.json()
    is_oa = data.get("is_oa", False)
    title = data.get("title", "Unknown")
    journal = data.get("journal_name", "")
    year = data.get("year", "")
    oa_status = data.get("oa_status", "closed")

    lines = [
        f"**{title}**",
        f"DOI: {doi}",
        f"Journal: {journal} ({year})",
        f"Open Access: {'YES' if is_oa else 'NO'} ({oa_status})",
    ]

    if is_oa:
        best = data.get("best_oa_location", {}) or {}
        pdf_url = best.get("url_for_pdf", "")
        landing_url = best.get("url", "")
        host_type = best.get("host_type", "")
        license_val = best.get("license", "")

        lines.append(f"Host: {host_type}")
        if license_val:
            lines.append(f"License: {license_val}")
        if pdf_url:
            lines.append(f"PDF: {pdf_url}")
        if landing_url and landing_url != pdf_url:
            lines.append(f"Landing page: {landing_url}")

        # List all OA locations
        all_locs = data.get("oa_locations", [])
        if len(all_locs) > 1:
            lines.append(f"\nAll OA locations ({len(all_locs)}):")
            for loc in all_locs:
                loc_url = loc.get("url_for_pdf") or loc.get("url", "")
                loc_host = loc.get("host_type", "")
                loc_license = loc.get("license", "")
                lines.append(f"  - {loc_host}: {loc_url} ({loc_license})")
    else:
        lines.append("\nNo open access version found.")
        lines.append("Consider using download_book_by_doi to try Sci-Hub as a fallback.")

    return "\n".join(lines)


@tool
def google_books_metadata(
    query: str = "",
    isbn: str = "",
    max_results: int = 5,
) -> str:
    """Look up book metadata from Google Books API.

    Enriches book information with descriptions, categories, page counts,
    cover images, and preview links. Search by title/author or ISBN.

    FREE — no API key required (but GOOGLE_BOOKS_API_KEY gets higher quotas).

    Args:
        query: Search query (title, author, etc.).
        isbn: ISBN-10 or ISBN-13 for precise lookup.
        max_results: Maximum results (default 5).

    Returns:
        Detailed book metadata from Google Books.
    """
    import httpx

    if not query and not isbn:
        return "[ERROR] Provide a query or ISBN"

    search_q = f"isbn:{isbn}" if isbn else query

    params = {
        "q": search_q,
        "maxResults": min(max_results, 40),
        "printType": "books",
    }
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY

    try:
        resp = httpx.get(
            "https://www.googleapis.com/books/v1/volumes",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Google Books API failed: {exc}"

    items = resp.json().get("items", [])
    if not items:
        return f"No results on Google Books for: {search_q}"

    formatted = [f"**Google Books: {search_q}** ({len(items)} results)\n"]

    for item in items[:max_results]:
        info = item.get("volumeInfo", {})
        access = item.get("accessInfo", {})

        isbn_13 = ""
        isbn_10 = ""
        for ident in info.get("industryIdentifiers", []):
            if ident.get("type") == "ISBN_13":
                isbn_13 = ident.get("identifier", "")
            elif ident.get("type") == "ISBN_10":
                isbn_10 = ident.get("identifier", "")

        authors = ", ".join(info.get("authors", ["Unknown"]))
        categories = ", ".join(info.get("categories", []))
        description = (info.get("description", "") or "")[:400]

        entry = [
            f"  **{info.get('title', 'Unknown')}**",
            f"  Authors: {authors}",
            f"  Published: {info.get('publishedDate', '?')} by {info.get('publisher', '?')}",
        ]
        if isbn_13:
            entry.append(f"  ISBN-13: {isbn_13}")
        if isbn_10:
            entry.append(f"  ISBN-10: {isbn_10}")
        if info.get("pageCount"):
            entry.append(f"  Pages: {info['pageCount']}")
        if categories:
            entry.append(f"  Categories: {categories}")
        if info.get("averageRating"):
            entry.append(f"  Rating: {info['averageRating']}/5 ({info.get('ratingsCount', 0)} ratings)")
        if description:
            entry.append(f"  Description: {description}")
        if info.get("previewLink"):
            entry.append(f"  Preview: {info['previewLink']}")
        entry.append(f"  Viewability: {access.get('viewability', 'unknown')}")

        formatted.append("\n".join(entry))

    return "\n\n".join(formatted)


@tool
def crossref_citation_graph(
    doi: str,
    depth: int = 1,
    max_refs: int = 20,
) -> str:
    """Explore the citation graph around a paper via CrossRef.

    Given a DOI, returns the paper's references (what it cites) and
    citing works (who cites it). Useful for building knowledge graphs
    and finding related work for documentary research.

    FREE — include UNPAYWALL_EMAIL for polite pool (50 req/sec vs 1 req/sec).

    Args:
        doi: The DOI to explore.
        depth: How many levels deep to traverse (1 = immediate refs, default 1).
        max_refs: Maximum references/citations to return per level (default 20).

    Returns:
        Citation graph with paper metadata.
    """
    import httpx

    doi = doi.strip()
    doi = re.sub(r"^https?://doi\.org/", "", doi)

    polite_email = UNPAYWALL_EMAIL or "research@mirothinker.ai"

    try:
        resp = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": f"MiroThinker/1.0 (mailto:{polite_email})"},
            timeout=30,
        )
        if resp.status_code != 200:
            return f"DOI not found in CrossRef: {doi}"

        data = resp.json().get("message", {})
    except Exception as exc:
        return f"[TOOL_ERROR] CrossRef lookup failed: {exc}"

    title = " ".join(data.get("title", ["Unknown"]))
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in data.get("author", [])[:5]
    ]
    pub_info = data.get("published-print") or data.get("published-online") or {}
    year_parts = pub_info.get("date-parts", [[""]])
    year = str(year_parts[0][0]) if year_parts and year_parts[0] else ""
    journal = " ".join(data.get("container-title", [""]))
    cited_by = data.get("is-referenced-by-count", 0)

    lines = [
        f"**{title}**",
        f"Authors: {', '.join(authors)}",
        f"Journal: {journal} ({year})",
        f"DOI: {doi}",
        f"Cited by: {cited_by} works",
        f"Reference count: {data.get('reference-count', 0)}",
    ]

    # References (what this paper cites)
    refs = data.get("reference", [])[:max_refs]
    if refs:
        lines.append(f"\n**References** ({len(refs)} of {data.get('reference-count', 0)}):")
        for i, ref in enumerate(refs, 1):
            ref_title = ref.get("article-title", ref.get("unstructured", ""))[:120]
            ref_doi = ref.get("DOI", "")
            ref_year = ref.get("year", "")
            ref_author = ref.get("author", "")
            doi_str = f" DOI:{ref_doi}" if ref_doi else ""
            lines.append(f"  {i}. {ref_author} ({ref_year}) {ref_title}{doi_str}")

    # Citing works (who cites this paper) — requires a separate query
    if cited_by > 0 and depth >= 1:
        try:
            citing_resp = httpx.get(
                "https://api.crossref.org/works",
                params={
                    "query": title[:200],
                    "filter": f"references:{doi}",
                    "rows": max_refs,
                    "sort": "is-referenced-by-count",
                    "order": "desc",
                },
                headers={"User-Agent": f"MiroThinker/1.0 (mailto:{polite_email})"},
                timeout=30,
            )
            if citing_resp.status_code == 200:
                citing_items = citing_resp.json().get("message", {}).get("items", [])
                if citing_items:
                    lines.append(f"\n**Cited by** ({len(citing_items)} of {cited_by}):")
                    for i, item in enumerate(citing_items[:max_refs], 1):
                        c_title = " ".join(item.get("title", ["?"]))[:120]
                        c_doi = item.get("DOI", "")
                        c_pub = item.get("published-print") or item.get("published-online") or {}
                        c_year_parts = c_pub.get("date-parts", [[""]])
                        c_year = str(c_year_parts[0][0]) if c_year_parts and c_year_parts[0] else ""
                        c_cited = item.get("is-referenced-by-count", 0)
                        lines.append(f"  {i}. ({c_year}) {c_title} [cited by {c_cited}] DOI:{c_doi}")
        except Exception:
            pass

    return "\n".join(lines)


@tool
def hathitrust_read(
    ht_id: str = "",
    query: str = "",
    page: int = 1,
) -> str:
    """Read content from HathiTrust's digitized book collection.

    HathiTrust has 17M+ digitized volumes. Public domain content is fully
    readable. Search the full-text index or read specific pages.

    FREE — no API key needed.

    Args:
        ht_id: HathiTrust volume ID (e.g. "mdp.39015027794331").
        query: Search query for full-text search across all volumes.
        page: Page number to read (default 1).

    Returns:
        Page text or search results from HathiTrust.
    """
    import httpx

    if ht_id:
        # Read a specific page from a known volume
        try:
            # HathiTrust Data API — get OCR text for a page
            resp = httpx.get(
                f"https://babel.hathitrust.org/cgi/imgsrv/ocr?id={ht_id}&seq={page}",
                headers={"User-Agent": _BROWSER_UA},
                timeout=30,
                follow_redirects=True,
            )
            if resp.status_code == 200 and resp.text.strip():
                text = resp.text
                return (
                    f"**HathiTrust Volume: {ht_id}** — Page {page}\n"
                    f"---\n{text}\n---\n"
                    f"[Next page: hathitrust_read(ht_id=\"{ht_id}\", page={page + 1})]"
                )
            else:
                return (
                    f"Could not read page {page} of {ht_id}. "
                    f"The volume may not be public domain or the page may not exist.\n"
                    f"View online: https://babel.hathitrust.org/cgi/pt?id={ht_id}"
                )
        except Exception as exc:
            return f"[TOOL_ERROR] HathiTrust read failed: {exc}"

    elif query:
        # Full-text search
        try:
            resp = httpx.get(
                "https://babel.hathitrust.org/cgi/ls",
                params={
                    "a": "srchls",
                    "q1": query,
                    "lmt": 10,
                },
                headers={"User-Agent": _BROWSER_UA},
                timeout=30,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return f"HathiTrust search failed (HTTP {resp.status_code})"

            # Parse search results from HTML
            html = resp.text
            # Extract result items
            results = []
            # Simple extraction of result links and titles
            import re as re_mod
            items = re_mod.findall(
                r'id=([^&"]+).*?class="result-title"[^>]*>(.*?)</a>',
                html, re_mod.DOTALL
            )
            if items:
                results = [{"ht_id": hid, "title": re_mod.sub(r'<[^>]+>', '', title_html).strip() or f"[Volume {hid}]"} for hid, title_html in items[:10]]
            else:
                # Fallback: just find HT IDs and titles
                items = re_mod.findall(
                    r'id=([a-z]+\.\d+)',
                    html
                )
                if items:
                    results = [{"ht_id": hid, "title": f"[Volume {hid}]"} for hid in items[:10]]

            if not results and not items:
                return f"No HathiTrust results for: {query}"

            lines = [f"**HathiTrust search: {query}** ({len(results) or len(items)} results)\n"]
            for item in (results or items)[:10]:
                if isinstance(item, dict):
                    lines.append(
                        f"  - {item['title']}\n"
                        f"    HT ID: {item['ht_id']}\n"
                        f"    URL: https://babel.hathitrust.org/cgi/pt?id={item['ht_id']}"
                    )
                else:
                    lines.append(
                        f"  - Volume: {item}\n"
                        f"    URL: https://babel.hathitrust.org/cgi/pt?id={item}"
                    )
            return "\n\n".join(lines)

        except Exception as exc:
            return f"[TOOL_ERROR] HathiTrust search failed: {exc}"

    return "[ERROR] Provide either ht_id (to read a volume) or query (to search)"


@tool
def internet_archive_download(
    ia_id: str,
    file_format: str = "auto",
    title: str = "",
    author: str = "",
) -> str:
    """Download a book/document from Internet Archive by identifier.

    Given an Internet Archive item ID, downloads the best available format
    (PDF, EPUB, DjVu, plain text) and extracts the text content.

    FREE — no API key needed for public items. Set IA_ACCESS_KEY and
    IA_SECRET_KEY for access to restricted items.

    Args:
        ia_id: Internet Archive item identifier (e.g. "thebookoflife00up" or from search results).
        file_format: Preferred format: "auto", "pdf", "epub", "txt", "djvu".
        title: Book title (optional, for metadata).
        author: Author (optional, for metadata).

    Returns:
        Extracted text from the downloaded book.
    """
    import httpx

    cache_key = f"ia://{ia_id}"
    cached = _cache_lookup_book(cache_key)
    if cached:
        return f"[FROM CACHE]\nIA ID: {ia_id}\nTitle: {title or 'Unknown'}\n---\n{cached}"

    # Get item metadata to find downloadable files
    try:
        resp = httpx.get(
            f"https://archive.org/metadata/{ia_id}",
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
        metadata = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Failed to get IA metadata for {ia_id}: {exc}"

    if not title:
        title = metadata.get("metadata", {}).get("title", ia_id)
    if not author:
        creator = metadata.get("metadata", {}).get("creator", "")
        author = creator if isinstance(creator, str) else ", ".join(creator) if isinstance(creator, list) else ""

    # Find best downloadable file
    files = metadata.get("files", [])
    format_priority = {
        "auto": ["Text", "DjVuTXT", "Plain text", "PDF", "EPUB", "DjVu"],
        "pdf": ["PDF"],
        "epub": ["EPUB"],
        "txt": ["Text", "DjVuTXT", "Plain text"],
        "djvu": ["DjVu"],
    }
    preferred = format_priority.get(file_format.lower(), format_priority["auto"])

    download_file = None
    for fmt in preferred:
        for f in files:
            f_format = f.get("format", "")
            f_name = f.get("name", "")
            if fmt.lower() in f_format.lower() or f_name.endswith(f".{fmt.lower()}"):
                download_file = f
                break
        if download_file:
            break

    # Fallback: try any text-like file
    if not download_file:
        for f in files:
            name = f.get("name", "").lower()
            if name.endswith((".txt", ".pdf", ".epub", ".htm", ".html")):
                download_file = f
                break

    if not download_file:
        return (
            f"No downloadable text found for IA item: {ia_id}\n"
            f"Available formats: {', '.join(set(f.get('format', '') for f in files))}\n"
            f"View online: https://archive.org/details/{ia_id}"
        )

    file_name = download_file["name"]
    download_url = f"https://archive.org/download/{ia_id}/{quote(file_name)}"

    # Download the file
    try:
        headers = {"User-Agent": _BROWSER_UA}
        if IA_ACCESS_KEY and IA_SECRET_KEY:
            headers["Authorization"] = f"LOW {IA_ACCESS_KEY}:{IA_SECRET_KEY}"

        with _get_http_client(timeout=120) as client:
            resp = client.get(download_url, headers=headers)
            resp.raise_for_status()
            content = resp.content
    except Exception as exc:
        return f"[TOOL_ERROR] Download failed for {download_url}: {exc}"

    # Extract text
    text = _extract_text(content, filename=file_name)

    if not text or text.startswith("[ERROR]") or text.startswith("[No text"):
        return f"[EXTRACTION FAILED] Downloaded {len(content):,} bytes from IA but could not extract text."

    # Cache and store
    _cache_store_book(
        url=cache_key,
        content=content,
        text=text,
        title=title,
        author=author,
        metadata={
            "ia_id": ia_id,
            "filename": file_name,
            "size_bytes": len(content),
            "source": "Internet Archive",
        },
        tags=["book", "internet_archive"],
    )

    _datalake_store(content, text, title, author, file_name, "", "", "Internet Archive", cache_key)

    # Truncate for context
    text_len = len(text)
    if text_len > 120000:
        text = (
            text[:120000]
            + f"\n\n[...TRUNCATED — full text is {text_len:,} chars. "
            f"Use read_book_section to read specific chapters.]"
        )

    return (
        f"**{title}**\n"
        f"Author: {author or 'Unknown'}\n"
        f"Source: Internet Archive ({ia_id})\n"
        f"File: {file_name} ({len(content):,} bytes)\n"
        f"Text: {text_len:,} characters extracted\n"
        f"---\n\n{text}"
    )


@tool
def extraction_status() -> str:
    """Show which document extraction backends are available.

    The pipeline can use multiple backends for PDF text extraction:
    - Docling (best for complex layouts, tables, math)
    - Mathpix (best for STEM content, LaTeX equations)
    - pdfplumber (baseline, always available)

    Returns:
        Status of each extraction backend.
    """
    try:
        from extraction import get_extraction_status
        status = get_extraction_status()
    except ImportError:
        status = {
            "docling": False,
            "mathpix": False,
            "pdfplumber": True,
            "active_backend": "pdfplumber",
        }

    lines = ["**PDF Extraction Backends**\n"]
    for backend, available in status.items():
        if backend == "active_backend":
            lines.append(f"  Active backend: {available}")
        else:
            icon = "AVAILABLE" if available else "not configured"
            lines.append(f"  - {backend}: {icon}")

    lines.append("\nSet EXTRACTION_BACKEND in .env to force a specific backend.")
    lines.append("Install Docling: pip install docling")
    lines.append("Configure Mathpix: set MATHPIX_APP_ID and MATHPIX_APP_KEY")

    return "\n".join(lines)


# ── Tool registry ─────────────────────────────────────────────────────

BOOK_TOOLS = [
    search_books,
    download_book,
    download_book_by_doi,
    book_library,
    read_book_section,
    search_book_content,
    unpaywall_lookup,
    google_books_metadata,
    crossref_citation_graph,
    hathitrust_read,
    internet_archive_download,
    extraction_status,
]
