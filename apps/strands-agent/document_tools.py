# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Scientific document & book acquisition tools for the Strands research agent.

Tier strategy (most legal → most comprehensive):
  1. Open Access — arXiv, PubMed Central, Semantic Scholar (MCP servers exist)
  2. Institutional — CrossRef, DOI resolution, publisher APIs
  3. Grey — Anna's Archive search, LibGen, Sci-Hub (legal grey area)
  4. Paid — Publisher paywalls (user provides credentials)

All downloaded documents are stored in the persistent cache (cache.py) for
reuse across sessions.

PDF text extraction uses pdfplumber (pure Python, no external deps).

Resolves: https://github.com/OrpingtonClose/MiroThinker/issues/92
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

from strands import tool

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────


def _extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 50) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        # Fallback: try PyPDF2
        try:
            from PyPDF2 import PdfReader
            import io

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages[:max_pages]):
                text = page.extract_text()
                if text:
                    pages.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(pages) if pages else "[No text extracted from PDF]"
        except ImportError:
            return (
                "[TOOL_ERROR] No PDF extraction library available. "
                "Install with: pip install pdfplumber  or  pip install PyPDF2"
            )

    import io

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(pages) if pages else "[No text extracted from PDF]"


def _resolve_doi(doi: str) -> dict | None:
    """Resolve a DOI to metadata via CrossRef API."""
    import httpx

    try:
        resp = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            return {
                "title": " ".join(data.get("title", ["Unknown"])),
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in data.get("author", [])
                ],
                "year": str(data.get("published-print", data.get("published-online", {}))
                            .get("date-parts", [[""]])[0][0]),
                "journal": " ".join(data.get("container-title", [""])),
                "doi": doi,
                "url": data.get("URL", f"https://doi.org/{doi}"),
                "abstract": data.get("abstract", ""),
                "type": data.get("type", ""),
            }
    except Exception as exc:
        logger.debug("DOI resolution failed for %s: %s", doi, exc)
    return None


# ═══════════════════════════════════════════════════════════════════════
# OPEN ACCESS — arXiv, PubMed, OpenAlex
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_open_access(
    query: str,
    source: str = "all",
    max_results: int = 10,
) -> str:
    """Search open-access scientific literature.

    Searches multiple open-access sources for papers, preprints, and articles.
    Returns structured metadata with download links where available.

    Note: arXiv and Semantic Scholar are also available as MCP servers with
    richer capabilities. Use this tool for quick multi-source searches.

    Args:
        query: Search query (title, keywords, author names, etc.).
        source: Which source to search: "all", "openalex", "crossref", "pubmed".
        max_results: Maximum results per source (default 10).

    Returns:
        Formatted list of papers with metadata and download links.
    """
    import httpx

    results = []

    # OpenAlex — free, comprehensive, 250M+ works
    if source in ("all", "openalex"):
        try:
            resp = httpx.get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "per_page": max_results,
                    "sort": "relevance_score:desc",
                    "select": "id,title,authorships,publication_year,primary_location,open_access,cited_by_count,doi",
                },
                headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
                timeout=30,
            )
            if resp.status_code == 200:
                for work in resp.json().get("results", []):
                    authors = [
                        a.get("author", {}).get("display_name", "")
                        for a in work.get("authorships", [])[:5]
                    ]
                    oa = work.get("open_access", {})
                    loc = work.get("primary_location", {}) or {}
                    source_info = loc.get("source", {}) or {}

                    results.append({
                        "title": work.get("title", "Unknown"),
                        "authors": authors,
                        "year": work.get("publication_year", ""),
                        "journal": source_info.get("display_name", ""),
                        "citations": work.get("cited_by_count", 0),
                        "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
                        "open_access": oa.get("is_oa", False),
                        "pdf_url": oa.get("oa_url", ""),
                        "source": "OpenAlex",
                    })
        except Exception as exc:
            logger.debug("OpenAlex search failed: %s", exc)

    # CrossRef — 150M+ works, best for DOI-based lookup
    if source in ("all", "crossref"):
        try:
            resp = httpx.get(
                "https://api.crossref.org/works",
                params={
                    "query": query,
                    "rows": max_results,
                    "sort": "relevance",
                    "select": "DOI,title,author,published-print,container-title,is-referenced-by-count,link",
                },
                headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
                timeout=30,
            )
            if resp.status_code == 200:
                for item in resp.json().get("message", {}).get("items", []):
                    authors = [
                        f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in item.get("author", [])[:5]
                    ]
                    year_parts = item.get("published-print", {}).get("date-parts", [[""]])
                    year = str(year_parts[0][0]) if year_parts and year_parts[0] else ""

                    # Check for open-access link
                    pdf_url = ""
                    for link in item.get("link", []):
                        if link.get("content-type") == "application/pdf":
                            pdf_url = link.get("URL", "")
                            break

                    results.append({
                        "title": " ".join(item.get("title", ["Unknown"])),
                        "authors": authors,
                        "year": year,
                        "journal": " ".join(item.get("container-title", [""])),
                        "citations": item.get("is-referenced-by-count", 0),
                        "doi": item.get("DOI", ""),
                        "open_access": bool(pdf_url),
                        "pdf_url": pdf_url,
                        "source": "CrossRef",
                    })
        except Exception as exc:
            logger.debug("CrossRef search failed: %s", exc)

    if not results:
        return f"No papers found for: {query}"

    # Deduplicate by DOI
    seen_dois = set()
    deduped = []
    for r in results:
        doi = r.get("doi", "")
        if doi and doi in seen_dois:
            continue
        if doi:
            seen_dois.add(doi)
        deduped.append(r)

    formatted = [f"**Scientific papers for: {query}** ({len(deduped)} results)\n"]
    for i, r in enumerate(deduped, 1):
        authors_str = ", ".join(r["authors"][:3])
        if len(r["authors"]) > 3:
            authors_str += f" +{len(r['authors']) - 3} more"
        oa_tag = " [OPEN ACCESS]" if r.get("open_access") else ""
        pdf_link = f"\n       PDF: {r['pdf_url']}" if r.get("pdf_url") else ""
        doi_link = f"\n       DOI: https://doi.org/{r['doi']}" if r.get("doi") else ""

        formatted.append(
            f"  {i}. {r['title']}\n"
            f"     {authors_str} ({r['year']}) — {r['journal']}{oa_tag}\n"
            f"     Cited by: {r['citations']}{doi_link}{pdf_link}\n"
            f"     Source: {r['source']}"
        )

    return "\n\n".join(formatted)


@tool
def download_paper(
    url: str,
    title: str = "",
    doi: str = "",
    tags: str = "[]",
) -> str:
    """Download a scientific paper/document and extract its text.

    Downloads the PDF (or HTML) from the given URL, extracts text,
    and stores both the raw PDF and extracted text in the persistent cache.

    Works with:
    - Direct PDF links (arXiv, PMC, publisher open-access)
    - DOI URLs (resolves via CrossRef)
    - Any URL that serves a PDF

    Args:
        url: Direct URL to the paper (PDF link preferred).
        title: Paper title (for cache metadata).
        doi: DOI if known (for metadata enrichment).
        tags: JSON array of tags (e.g. '["banana", "genetics", "heritage"]').

    Returns:
        Extracted text from the paper (first 50 pages).
    """
    import httpx

    # Check cache first
    try:
        from cache import cache_get

        cached = cache_get(url=url)
        if cached and cached.get("content"):
            try:
                text = cached["content"].decode("utf-8")
                return f"[FROM CACHE]\nTitle: {cached.get('title', title)}\n---\n{text}"
            except (UnicodeDecodeError, AttributeError):
                pass
    except ImportError:
        pass

    # Resolve DOI for metadata
    metadata = {}
    if doi:
        resolved = _resolve_doi(doi)
        if resolved:
            metadata = resolved
            if not title:
                title = resolved.get("title", "")
            if not url or url.startswith("https://doi.org/"):
                # Try to find PDF URL from metadata
                pdf_url = resolved.get("pdf_url", "")
                if pdf_url:
                    url = pdf_url

    # Download the document
    try:
        resp = httpx.get(
            url,
            headers={
                "User-Agent": "MiroThinker/1.0 (research agent)",
                "Accept": "application/pdf, text/html, */*",
            },
            timeout=60,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Failed to download: {exc}"

    content_type = resp.headers.get("content-type", "").lower()
    raw_content = resp.content

    # Extract text based on content type
    if "pdf" in content_type or url.endswith(".pdf"):
        extracted_text = _extract_text_from_pdf(raw_content)
        doc_type = "application/pdf"
    elif "html" in content_type:
        # For HTML, store as-is (the agent can use jina_read_url for clean extraction)
        extracted_text = raw_content.decode("utf-8", errors="replace")
        doc_type = "text/html"
    else:
        extracted_text = raw_content.decode("utf-8", errors="replace")
        doc_type = content_type or "application/octet-stream"

    # Parse tags
    try:
        tag_list = json.loads(tags) if tags else []
    except (json.JSONDecodeError, TypeError):
        tag_list = []
    tag_list.append("paper")

    # Store in cache (extracted text, not raw PDF — saves context window)
    try:
        from cache import cache_put

        cache_put(
            url=url,
            content=extracted_text,
            content_type="text/plain",
            source_type="paper",
            title=title,
            summary=extracted_text[:500],
            tags=tag_list,
            metadata=metadata,
        )

        # Also cache the raw PDF for future re-extraction
        if "pdf" in doc_type:
            cache_put(
                url=f"raw-pdf://{url}",
                content=raw_content,
                content_type="application/pdf",
                source_type="paper",
                title=f"[RAW PDF] {title}",
                summary="",
                tags=tag_list,
            )
    except ImportError:
        pass

    # Truncate for agent context
    if len(extracted_text) > 80000:
        extracted_text = (
            extracted_text[:80000]
            + f"\n\n[...truncated, {len(raw_content):,} bytes total]"
        )

    return (
        f"**{title or 'Downloaded document'}**\n"
        f"URL: {url}\n"
        f"Type: {doc_type}\n"
        f"Size: {len(raw_content):,} bytes\n"
        f"{'DOI: ' + doi if doi else ''}\n"
        f"---\n\n"
        f"{extracted_text}"
    )


# ═══════════════════════════════════════════════════════════════════════
# ANNA'S ARCHIVE — Search interface for shadow libraries
# ═══════════════════════════════════════════════════════════════════════


@tool
def annas_archive_search(
    query: str,
    content_type: str = "book_any",
    max_results: int = 10,
) -> str:
    """Search Anna's Archive for books, papers, and documents.

    Anna's Archive is a search engine for shadow libraries (Library Genesis,
    Sci-Hub, Z-Library, etc.). It indexes 40M+ books and 100M+ papers.

    IMPORTANT: This tool only SEARCHES — it returns metadata and download
    page links. The user decides whether to download. Some content may be
    under copyright; always check legal status in your jurisdiction.

    Args:
        query: Search query (title, author, ISBN, DOI, etc.).
        content_type: Filter by type:
            "book_any" — all books
            "book_fiction" — fiction books
            "book_nonfiction" — non-fiction books
            "journal_article" — journal articles/papers
            "magazine" — magazines
            "standards_document" — technical standards
        max_results: Maximum results to return (default 10).

    Returns:
        Formatted list of results with metadata and links.
    """
    import httpx

    # Anna's Archive search URL
    base_url = "https://annas-archive.org/search"

    # Map content type to Anna's Archive filter
    content_map = {
        "book_any": "",
        "book_fiction": "&content=book_fiction",
        "book_nonfiction": "&content=book_nonfiction",
        "journal_article": "&content=journal_article",
        "magazine": "&content=magazine",
        "standards_document": "&content=standards_document",
    }
    content_filter = content_map.get(content_type, "")

    search_url = f"{base_url}?q={query}{content_filter}"

    try:
        resp = httpx.get(
            search_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=30,
            follow_redirects=True,
        )

        if resp.status_code != 200:
            return (
                f"[TOOL_ERROR] Anna's Archive returned HTTP {resp.status_code}. "
                f"The site may be temporarily unavailable. "
                f"Try the search URL directly: {search_url}"
            )

        # Parse results from HTML (Anna's Archive doesn't have a public API)
        html = resp.text
        results = _parse_annas_archive_html(html, max_results)

        if not results:
            return (
                f"No results found on Anna's Archive for: {query}\n"
                f"Search URL: {search_url}\n"
                f"Try alternative queries or check the site directly."
            )

        formatted = [
            f"**Anna's Archive results for: {query}** ({len(results)} found)\n"
            f"Search URL: {search_url}\n"
        ]
        for i, r in enumerate(results, 1):
            formatted.append(
                f"  {i}. {r['title']}\n"
                f"     {r.get('author', 'Unknown author')} | {r.get('year', '')} | "
                f"{r.get('language', '')} | {r.get('format', '')}\n"
                f"     {r.get('size', '')} | {r.get('source', '')}\n"
                f"     Link: {r.get('url', '')}"
            )

        return "\n\n".join(formatted)

    except Exception as exc:
        return (
            f"[TOOL_ERROR] Anna's Archive search failed: {exc}\n"
            f"Try the search URL directly: {search_url}"
        )


def _parse_annas_archive_html(html: str, max_results: int) -> list[dict]:
    """Parse Anna's Archive search results from HTML.

    This is brittle (HTML scraping), but Anna's Archive has no public API.
    Falls back gracefully if the HTML structure changes.
    """
    results = []

    # Look for result links — Anna's Archive uses /md5/ links
    # Pattern: <a href="/md5/HASH" ...>
    md5_pattern = re.compile(
        r'<a[^>]*href="(/md5/[a-f0-9]{32})"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )

    # Alternative pattern for newer layout
    result_blocks = re.findall(
        r'href="(/md5/[a-f0-9]{32})".*?</a>',
        html,
        re.DOTALL | re.IGNORECASE,
    )

    if not result_blocks and not md5_pattern.findall(html):
        # Try a broader pattern
        all_links = re.findall(r'/md5/([a-f0-9]{32})', html)
        seen = set()
        for md5 in all_links[:max_results]:
            if md5 not in seen:
                seen.add(md5)
                results.append({
                    "title": f"[Result — see page for details]",
                    "url": f"https://annas-archive.org/md5/{md5}",
                    "md5": md5,
                })
        return results

    matches = md5_pattern.findall(html)
    seen = set()
    for href, title_html in matches[:max_results * 2]:
        md5 = href.split("/")[-1]
        if md5 in seen:
            continue
        seen.add(md5)

        # Clean title from HTML
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if not title or len(title) < 3:
            continue

        results.append({
            "title": title[:200],
            "url": f"https://annas-archive.org{href}",
            "md5": md5,
        })

        if len(results) >= max_results:
            break

    return results


@tool
def resolve_doi_metadata(doi: str) -> str:
    """Resolve a DOI to full paper metadata via CrossRef.

    Given a DOI (e.g. "10.1038/s41586-023-06647-8"), returns:
    - Title, authors, journal, year
    - Abstract (if available)
    - Citation count
    - Publisher URL

    Args:
        doi: The DOI to resolve (e.g. "10.1038/s41586-023-06647-8").

    Returns:
        Formatted paper metadata.
    """
    # Clean DOI
    doi = doi.strip()
    doi = re.sub(r"^https?://doi\.org/", "", doi)

    resolved = _resolve_doi(doi)
    if not resolved:
        return f"Could not resolve DOI: {doi}"

    authors_str = ", ".join(resolved.get("authors", [])[:5])
    if len(resolved.get("authors", [])) > 5:
        authors_str += f" +{len(resolved['authors']) - 5} more"

    abstract = resolved.get("abstract", "")
    # Clean HTML from abstract
    if abstract:
        abstract = re.sub(r"<[^>]+>", "", abstract).strip()

    return (
        f"**{resolved.get('title', 'Unknown')}**\n"
        f"Authors: {authors_str}\n"
        f"Year: {resolved.get('year', 'Unknown')}\n"
        f"Journal: {resolved.get('journal', 'Unknown')}\n"
        f"DOI: {doi}\n"
        f"URL: {resolved.get('url', '')}\n"
        f"Type: {resolved.get('type', 'Unknown')}\n"
        f"\nAbstract:\n{abstract or '(not available)'}"
    )


@tool
def extract_pdf_text(
    url: str,
    max_pages: int = 50,
) -> str:
    """Download a PDF and extract its text content.

    Works with any publicly accessible PDF URL. Extracted text is cached
    for future sessions.

    Args:
        url: Direct URL to a PDF file.
        max_pages: Maximum pages to extract (default 50).

    Returns:
        Extracted text from the PDF.
    """
    import httpx

    # Check cache
    try:
        from cache import cache_get

        cached = cache_get(url=url)
        if cached and cached.get("content"):
            try:
                return f"[FROM CACHE]\n{cached['content'].decode('utf-8')}"
            except (UnicodeDecodeError, AttributeError):
                pass
    except ImportError:
        pass

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
            timeout=60,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Failed to download PDF: {exc}"

    extracted = _extract_text_from_pdf(resp.content, max_pages)

    # Cache the extracted text
    try:
        from cache import cache_put

        cache_put(
            url=url,
            content=extracted,
            content_type="text/plain",
            source_type="paper",
            title=url.split("/")[-1],
            summary=extracted[:500],
            tags=["pdf", "extracted"],
        )
    except ImportError:
        pass

    if len(extracted) > 80000:
        extracted = extracted[:80000] + f"\n\n[...truncated]"

    return extracted


# ── Tool registry ─────────────────────────────────────────────────────

DOCUMENT_TOOLS = [
    search_open_access,
    download_paper,
    annas_archive_search,
    resolve_doi_metadata,
    extract_pdf_text,
]
