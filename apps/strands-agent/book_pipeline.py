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
    elif lower_name.endswith((".txt", ".md", ".rst")) or "text/" in lower_ct:
        return content.decode("utf-8", errors="replace")
    elif lower_name.endswith((".htm", ".html")) or "html" in lower_ct:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
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


def _get_http_client(timeout: int = 30):
    """Get httpx client with browser-like headers."""
    import httpx
    return httpx.Client(
        timeout=httpx.Timeout(timeout, connect=10),
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    )


def _try_mirrors(mirrors: list[str], path_fn, **kwargs) -> Optional[object]:
    """Try a request across multiple mirror domains."""
    import httpx

    for mirror in mirrors:
        try:
            url = path_fn(mirror)
            with _get_http_client() as client:
                resp = client.get(url, **kwargs)
                if resp.status_code == 200:
                    return resp
        except Exception as exc:
            logger.debug("Mirror %s failed: %s", mirror, exc)
            continue
    return None


# ── Cache integration ─────────────────────────────────────────────────


def _cache_store_book(url: str, content: bytes, text: str, title: str,
                      author: str = "", metadata: dict = None,
                      tags: list[str] = None) -> bool:
    """Store a book in the persistent cache. Returns True on success."""
    try:
        from cache import cache_put
        tag_list = tags or []
        tag_list.append("book")

        # Store extracted text (main entry — what the agent reads)
        cache_put(
            url=url,
            content=text,
            content_type="text/plain",
            source_type="book",
            title=title,
            summary=text[:500],
            tags=tag_list,
            metadata=metadata or {},
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
    """Search Library Genesis via HTML scraping."""
    import httpx

    results = []

    for mirror in LIBGEN_MIRRORS:
        try:
            search_url = (
                f"{mirror}/search.php?"
                f"req={quote_plus(query)}&lg_topic=libgen&open=0"
                f"&view=simple&res={max_results}&phrase=1&column=def"
            )

            with _get_http_client() as client:
                resp = client.get(search_url)

            if resp.status_code != 200:
                continue

            html = resp.text
            results = _parse_libgen_results(html, mirror, max_results)
            if results:
                return results

        except Exception as exc:
            logger.debug("LibGen mirror %s failed: %s", mirror, exc)
            continue

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
    """Download a book from LibGen by MD5 hash. Returns (content, filename) or None."""
    import httpx

    for mirror in LIBGEN_MIRRORS:
        try:
            get_url = f"{mirror}/get.php?md5={md5}"

            with _get_http_client() as client:
                resp = client.get(get_url)

            if resp.status_code != 200:
                continue

            # The get.php page usually contains a download link
            # Look for direct download link in the HTML
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
                    with _get_http_client() as dl_client:
                        dl_resp = dl_client.get(link)
                    if dl_resp.status_code == 200 and len(dl_resp.content) > 1000:
                        # Determine filename
                        cd = dl_resp.headers.get("content-disposition", "")
                        fn_match = re.search(r'filename="?([^";\n]+)', cd)
                        filename = fn_match.group(1).strip() if fn_match else f"book_{md5[:8]}.pdf"
                        return dl_resp.content, filename
                except Exception as exc:
                    logger.debug("Download link %s failed: %s", link[:80], exc)
                    continue

        except Exception as exc:
            logger.debug("LibGen mirror %s failed for MD5 %s: %s", mirror, md5, exc)
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 3: ANNA'S ARCHIVE
# ═══════════════════════════════════════════════════════════════════════


def _search_annas_archive(query: str, max_results: int = 10,
                          content_type: str = "") -> list[dict]:
    """Search Anna's Archive via HTML scraping."""
    import httpx

    results = []
    content_filter = f"&content={content_type}" if content_type else ""

    for domain in ANNAS_DOMAINS:
        try:
            search_url = f"{domain}/search?q={quote_plus(query)}{content_filter}"

            with _get_http_client() as client:
                resp = client.get(search_url)

            if resp.status_code != 200:
                continue

            html = resp.text
            # Anna's Archive uses /md5/ links for results
            md5_links = re.findall(r'/md5/([a-f0-9]{32})', html)
            seen = set()

            # Try to extract titles from nearby text
            for md5 in md5_links:
                if md5 in seen:
                    continue
                seen.add(md5)

                # Find the block containing this MD5 and extract info
                # Look for text near the MD5 link
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
                return results

        except Exception as exc:
            logger.debug("Anna's Archive %s failed: %s", domain, exc)
            continue

    return results


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 4: SCI-HUB (DOI → PDF)
# ═══════════════════════════════════════════════════════════════════════


def _download_from_scihub(doi: str) -> Optional[tuple[bytes, str]]:
    """Download a paper from Sci-Hub by DOI. Returns (content, filename) or None."""
    import httpx

    for domain in SCIHUB_DOMAINS:
        try:
            url = f"{domain}/{doi}"

            with _get_http_client() as client:
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
                # Try iframe src
                pdf_match = re.search(
                    r'<iframe[^>]*src="([^"]+)"',
                    html, re.IGNORECASE
                )

            if not pdf_match:
                continue

            pdf_url = pdf_match.group(1)
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url

            with _get_http_client() as dl_client:
                pdf_resp = dl_client.get(pdf_url)

            if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                safe_doi = re.sub(r'[/\\:*?"<>|]', '_', doi)
                return pdf_resp.content, f"{safe_doi}.pdf"

        except Exception as exc:
            logger.debug("Sci-Hub %s failed for DOI %s: %s", domain, doi, exc)
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
# AGENT TOOLS — exposed to the Strands agent
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_books(
    query: str,
    sources: str = "all",
    max_results: int = 10,
) -> str:
    """Search for books, textbooks, and documents across multiple sources.

    Searches Open Library, Library Genesis, Anna's Archive, and Project
    Gutenberg simultaneously. Returns unified results with metadata and
    download availability.

    Use this to find books for documentary research, knowledge base
    construction, or any task requiring deep reading.

    Args:
        query: Search query — title, author, ISBN, topic, etc.
        sources: Which sources to search: "all", "openlibrary", "libgen",
                 "annas", "gutenberg". Comma-separated for multiple.
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

    if not all_results:
        return f"No books found for: {query}\nTry different search terms or check that the sources are accessible."

    # Format results
    formatted = [f"**Book search: {query}** ({len(all_results)} results)\n"]

    for i, r in enumerate(all_results, 1):
        authors = ", ".join(r.get("authors", [])[:3]) or "Unknown author"
        year = f" ({r.get('year', '')})" if r.get("year") else ""
        source = r.get("source", "Unknown")
        fmt = f" [{r.get('format', '').upper()}]" if r.get("format") else ""
        size = f" {r.get('size', '')}" if r.get("size") else ""
        pages = f" {r.get('pages', '')}pp" if r.get("pages") else ""
        isbn = f"\n     ISBN: {r.get('isbn', '')}" if r.get("isbn") else ""
        md5 = f"\n     MD5: {r.get('md5', '')}" if r.get("md5") else ""
        dl_info = ""
        if r.get("has_fulltext"):
            dl_info = "\n     [FULL TEXT AVAILABLE via Internet Archive]"
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
    cache_key = url or f"md5://{md5}" or f"doi://{doi}"

    # Check cache first
    cached = _cache_lookup_book(cache_key)
    if cached:
        return f"[FROM CACHE]\nTitle: {title or 'Unknown'}\n---\n{cached}"

    content = None
    filename = ""

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
                ct = resp.headers.get("content-type", "")
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

    # Extract text
    ct = ""
    if filename:
        ct = filename  # _extract_text uses filename for format detection
    text = _extract_text(content, filename=filename, content_type=ct)

    if not text or text.startswith("[ERROR]") or text.startswith("[No text"):
        return f"[EXTRACTION FAILED] Downloaded {len(content):,} bytes but could not extract text.\nFilename: {filename}"

    # Cache it
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

    # Truncate for context window
    if len(text) > 120000:
        text = (
            text[:120000]
            + f"\n\n[...TRUNCATED — full text is {len(text):,} chars. "
            f"Use read_book_section to read specific chapters.]"
        )

    return (
        f"**{title or filename or 'Downloaded book'}**\n"
        f"Author: {author or 'Unknown'}\n"
        f"File: {filename} ({len(content):,} bytes)\n"
        f"Text: {len(text):,} characters extracted\n"
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

    # Try CrossRef for open access PDF first
    import httpx

    try:
        resp = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "MiroThinker/1.0"},
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
                            return result
    except Exception:
        pass

    # Try Sci-Hub
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


# ── Tool registry ─────────────────────────────────────────────────────

BOOK_TOOLS = [
    search_books,
    download_book,
    download_book_by_doi,
    book_library,
    read_book_section,
    search_book_content,
]
