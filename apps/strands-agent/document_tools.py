# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Scientific document & book acquisition tools for the Strands research agent.

Tier strategy (most legal → most comprehensive):
  1. Open Access — arXiv, PubMed Central, Semantic Scholar (MCP servers exist)
  2. Institutional — CrossRef, DOI resolution, publisher APIs
  3. Aggregators — CORE (40M+ full-text), Unpaywall (legal OA finder)
  4. Grey — Anna's Archive search, LibGen, Sci-Hub (legal grey area)
  5. Paid — Publisher paywalls (user provides credentials)

Additional sources (new):
  - Semantic Scholar — 200M+ papers, AI embeddings, recommendations
  - CORE — 300M+ metadata, 40M+ open-access full texts
  - Springer Nature — 7M+ docs, 280K+ open access articles
  - Zenodo — CERN's open data/research repository

All downloaded documents are stored in the persistent cache (cache.py) for
reuse across sessions.

PDF text extraction uses pdfplumber (pure Python, no external deps).
Enhanced extraction available via Docling or Mathpix (see extraction.py).

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
from async_http import async_get, async_post

logger = logging.getLogger(__name__)

# ── API keys ──────────────────────────────────────────────────────────

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
CORE_API_KEY = os.getenv("CORE_API_KEY", "")
SPRINGER_API_KEY = os.getenv("SPRINGER_API_KEY", "")
ZENODO_ACCESS_TOKEN = os.getenv("ZENODO_ACCESS_TOKEN", "")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "")


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


async def _resolve_doi(doi: str) -> dict | None:
    """Resolve a DOI to metadata via CrossRef API."""

    try:
        resp = await async_get(
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
async def search_open_access(
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

    results = []

    # OpenAlex — free, comprehensive, 250M+ works
    if source in ("all", "openalex"):
        try:
            resp = await async_get(
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
            resp = await async_get(
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
async def download_paper(
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
        resolved = await _resolve_doi(doi)
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
        resp = await async_get(
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
async def annas_archive_search(
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
        resp = await async_get(
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
async def resolve_doi_metadata(doi: str) -> str:
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

    resolved = await _resolve_doi(doi)
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
async def extract_pdf_text(
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
        resp = await async_get(
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


# ═══════════════════════════════════════════════════════════════════════
# SEMANTIC SCHOLAR — 200M+ papers, AI embeddings, recommendations
# ═══════════════════════════════════════════════════════════════════════


@tool
async def semantic_scholar_search(
    query: str,
    max_results: int = 10,
    year_range: str = "",
    fields_of_study: str = "",
) -> str:
    """Search Semantic Scholar for papers with AI-powered relevance ranking.

    Semantic Scholar indexes 200M+ papers with SPECTER2 embeddings for
    semantic search (not just keyword matching). Returns papers with
    abstracts, citation counts, and open access links.

    FREE — 100 requests per 5 minutes without API key.
    Set SEMANTIC_SCHOLAR_API_KEY for higher rate limits.

    Args:
        query: Search query (natural language works well).
        max_results: Maximum results (default 10, max 100).
        year_range: Filter by year range, e.g. "2020-2024" or "2020-".
        fields_of_study: Filter by field, e.g. "Neuroscience", "Computer Science".

    Returns:
        Formatted list of papers with metadata and links.
    """

    headers = {"User-Agent": "MiroThinker/1.0 (research agent)"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,abstract,citationCount,openAccessPdf,externalIds,publicationTypes,journal,url",
    }
    if year_range:
        params["year"] = year_range
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    try:
        resp = await async_get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        if resp.status_code == 429:
            return (
                "[RATE LIMITED] Semantic Scholar rate limit hit. "
                "Set SEMANTIC_SCHOLAR_API_KEY for higher limits, "
                "or wait a few minutes."
            )
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Semantic Scholar search failed: {exc}"

    data = resp.json()
    papers = data.get("data", [])

    if not papers:
        return f"No Semantic Scholar results for: {query}"

    formatted = [f"**Semantic Scholar: {query}** ({len(papers)} results)\n"]

    for i, paper in enumerate(papers, 1):
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])[:4]
        )
        if len(paper.get("authors") or []) > 4:
            authors += f" +{len(paper['authors']) - 4} more"

        year = paper.get("year", "")
        citations = paper.get("citationCount", 0)
        journal_info = paper.get("journal", {}) or {}
        journal = journal_info.get("name", "")

        oa_pdf = paper.get("openAccessPdf", {}) or {}
        pdf_url = oa_pdf.get("url", "")
        oa_tag = " [OPEN ACCESS]" if pdf_url else ""

        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")

        abstract = (paper.get("abstract", "") or "")[:250]
        if abstract:
            abstract = f"\n     Abstract: {abstract}..."

        doi_line = f"\n     DOI: {doi}" if doi else ""
        arxiv_line = f"\n     arXiv: {arxiv_id}" if arxiv_id else ""
        pdf_line = f"\n     PDF: {pdf_url}" if pdf_url else ""

        formatted.append(
            f"  {i}. **{paper.get('title', 'Unknown')}**\n"
            f"     {authors} ({year}) — {journal}{oa_tag}\n"
            f"     Cited by: {citations}{doi_line}{arxiv_line}{pdf_line}{abstract}"
        )

    return "\n\n".join(formatted)


@tool
async def semantic_scholar_recommend(
    paper_id: str,
    max_results: int = 10,
) -> str:
    """Get paper recommendations from Semantic Scholar.

    Given a paper (by Semantic Scholar ID, DOI, or arXiv ID), returns
    similar papers based on SPECTER2 embeddings. Great for discovering
    related work that keyword searches would miss.

    Args:
        paper_id: Paper identifier — Semantic Scholar ID, DOI (prefix with "DOI:"),
                  or arXiv ID (prefix with "ARXIV:").
        max_results: Maximum recommendations (default 10).

    Returns:
        Recommended papers with similarity scores.
    """

    headers = {"User-Agent": "MiroThinker/1.0 (research agent)"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        resp = await async_get(
            f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}",
            headers=headers,
            params={
                "limit": min(max_results, 100),
                "fields": "title,authors,year,citationCount,openAccessPdf,externalIds,abstract",
            },
            timeout=30,
        )
        if resp.status_code == 404:
            return f"Paper not found: {paper_id}. Try DOI:10.xxx/yyy or ARXIV:2301.xxxxx format."
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Semantic Scholar recommendations failed: {exc}"

    papers = resp.json().get("recommendedPapers", [])
    if not papers:
        return f"No recommendations for paper: {paper_id}"

    formatted = [f"**Recommendations for: {paper_id}** ({len(papers)} papers)\n"]

    for i, paper in enumerate(papers, 1):
        authors = ", ".join(
            a.get("name", "") for a in (paper.get("authors") or [])[:3]
        )
        year = paper.get("year", "")
        citations = paper.get("citationCount", 0)
        ext_ids = paper.get("externalIds", {}) or {}
        doi = ext_ids.get("DOI", "")
        oa_pdf = paper.get("openAccessPdf", {}) or {}
        pdf_url = oa_pdf.get("url", "")
        oa_tag = " [OA]" if pdf_url else ""

        formatted.append(
            f"  {i}. {paper.get('title', '?')} ({year}){oa_tag}\n"
            f"     {authors} | Cited by: {citations}\n"
            f"     {'DOI: ' + doi if doi else ''}"
            f"{'  PDF: ' + pdf_url if pdf_url else ''}"
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# CORE — 300M+ metadata, 40M+ open-access full texts
# ═══════════════════════════════════════════════════════════════════════


@tool
async def search_core(
    query: str,
    max_results: int = 10,
    year_from: int = 0,
    year_to: int = 0,
    full_text: bool = False,
) -> str:
    """Search CORE for open-access papers and full texts.

    CORE aggregates 300M+ metadata records and 40M+ full-text papers from
    institutional repositories worldwide. Often finds open-access versions
    that other sources miss.

    FREE — register for API key at core.ac.uk (set CORE_API_KEY).

    Args:
        query: Search query.
        max_results: Maximum results (default 10, max 100).
        year_from: Filter papers from this year (0 = no filter).
        year_to: Filter papers up to this year (0 = no filter).
        full_text: If True, only return papers with downloadable full text.

    Returns:
        Formatted list of papers with metadata and download links.
    """

    if not CORE_API_KEY:
        return (
            "[TOOL_ERROR] CORE_API_KEY not set. "
            "Register for a free key at https://core.ac.uk/services/api"
        )

    headers = {"Authorization": f"Bearer {CORE_API_KEY}"}

    body = {
        "q": query,
        "limit": min(max_results, 100),
    }
    if year_from or year_to:
        year_filter = ""
        if year_from:
            year_filter += f"yearPublished>={year_from}"
        if year_to:
            if year_filter:
                year_filter += " AND "
            year_filter += f"yearPublished<={year_to}"
        body["q"] = f"({query}) AND ({year_filter})"

    if full_text:
        body["q"] = f"({body['q']}) AND _exists_:fullText"

    try:
        resp = await async_post(
            "https://api.core.ac.uk/v3/search/works",
            headers=headers,
            json=body,
            timeout=30,
        )
        if resp.status_code == 401:
            return "[TOOL_ERROR] CORE API key invalid. Get a new one at https://core.ac.uk/services/api"
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] CORE search failed: {exc}"

    data = resp.json()
    results = data.get("results", [])

    if not results:
        return f"No CORE results for: {query}"

    formatted = [f"**CORE: {query}** ({len(results)} results, {data.get('totalHits', '?')} total)\n"]

    for i, paper in enumerate(results, 1):
        authors = ", ".join(
            (a.get("name", "") if isinstance(a, dict) else str(a))
            for a in (paper.get("authors") or [])[:4]
        )
        year = paper.get("yearPublished", "")
        title = paper.get("title", "Unknown")
        doi = paper.get("doi", "")
        download_url = paper.get("downloadUrl", "")
        has_fulltext = bool(paper.get("fullText") or download_url)

        oa_tag = " [FULL TEXT]" if has_fulltext else ""
        doi_line = f"\n     DOI: {doi}" if doi else ""
        dl_line = f"\n     Download: {download_url}" if download_url else ""
        abstract = (paper.get("abstract", "") or "")[:200]
        abstract_line = f"\n     Abstract: {abstract}..." if abstract else ""

        formatted.append(
            f"  {i}. **{title}** ({year}){oa_tag}\n"
            f"     {authors}{doi_line}{dl_line}{abstract_line}"
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# SPRINGER NATURE — 7M+ docs, 280K+ open access
# ═══════════════════════════════════════════════════════════════════════


@tool
async def search_springer(
    query: str,
    max_results: int = 10,
    content_type: str = "all",
    open_access: bool = False,
) -> str:
    """Search Springer Nature for academic articles and book chapters.

    Springer Nature publishes 7M+ documents including 280K+ open access
    articles. Covers major journals like Nature, Scientific American,
    BioMed Central, etc.

    FREE — register for API key at dev.springernature.com (set SPRINGER_API_KEY).

    Args:
        query: Search query.
        max_results: Maximum results (default 10, max 50).
        content_type: Filter by type: "all", "Journal", "Book".
        open_access: If True, only return open access articles.

    Returns:
        Formatted list of articles/chapters with metadata.
    """

    if not SPRINGER_API_KEY:
        return (
            "[TOOL_ERROR] SPRINGER_API_KEY not set. "
            "Register for a free key at https://dev.springernature.com"
        )

    # Use open access endpoint if requested, otherwise metadata
    if open_access:
        api_url = "https://api.springernature.com/openaccess/json"
    else:
        api_url = "https://api.springernature.com/meta/v2/json"

    params = {
        "q": query,
        "s": 1,
        "p": min(max_results, 50),
        "api_key": SPRINGER_API_KEY,
    }
    if content_type != "all":
        params["q"] += f' type:{content_type}'

    try:
        resp = await async_get(api_url, params=params, timeout=30)
        if resp.status_code == 403:
            return "[TOOL_ERROR] Springer API key invalid or quota exceeded."
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Springer search failed: {exc}"

    data = resp.json()
    records = data.get("records", [])

    if not records:
        return f"No Springer results for: {query}"

    total = data.get("result", [{}])[0].get("total", "?") if data.get("result") else "?"

    formatted = [f"**Springer Nature: {query}** ({len(records)} results, {total} total)\n"]

    for i, rec in enumerate(records, 1):
        title = rec.get("title", "Unknown")
        creators = rec.get("creators", [])
        authors = ", ".join(
            c.get("creator", "") for c in creators[:4]
        )
        pub_name = rec.get("publicationName", "")
        pub_date = rec.get("publicationDate", "")
        doi = rec.get("doi", "")
        is_oa = rec.get("openaccess", "false") == "true"
        abstract_text = rec.get("abstract", "")

        urls = rec.get("url", [])
        pdf_url = ""
        for u in urls:
            if u.get("format") == "pdf":
                pdf_url = u.get("value", "")
                break

        oa_tag = " [OPEN ACCESS]" if is_oa else ""
        doi_line = f"\n     DOI: {doi}" if doi else ""
        pdf_line = f"\n     PDF: {pdf_url}" if pdf_url else ""
        abstract_line = f"\n     Abstract: {abstract_text[:200]}..." if abstract_text else ""

        formatted.append(
            f"  {i}. **{title}**{oa_tag}\n"
            f"     {authors}\n"
            f"     {pub_name} ({pub_date}){doi_line}{pdf_line}{abstract_line}"
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# ZENODO — CERN's open research data repository
# ═══════════════════════════════════════════════════════════════════════


@tool
async def search_zenodo(
    query: str,
    max_results: int = 10,
    resource_type: str = "",
    sort: str = "bestmatch",
) -> str:
    """Search Zenodo for research papers, datasets, and software.

    Zenodo is CERN's open data repository. Papers often deposit supplementary
    data, code, and full datasets here. Also hosts many preprints.

    FREE — optionally set ZENODO_ACCESS_TOKEN for higher limits.

    Args:
        query: Search query.
        max_results: Maximum results (default 10).
        resource_type: Filter by type: "", "publication", "dataset", "software",
                      "poster", "presentation", "image", "video".
        sort: Sort order: "bestmatch", "mostrecent", "-mostrecent".

    Returns:
        Formatted list of Zenodo records with download links.
    """

    params = {
        "q": query,
        "size": min(max_results, 100),
        "sort": sort,
    }
    if resource_type:
        params["type"] = resource_type

    headers = {}
    if ZENODO_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {ZENODO_ACCESS_TOKEN}"

    try:
        resp = await async_get(
            "https://zenodo.org/api/records",
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        return f"[TOOL_ERROR] Zenodo search failed: {exc}"

    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])

    if not hits:
        return f"No Zenodo results for: {query}"

    total = data.get("hits", {}).get("total", "?")

    formatted = [f"**Zenodo: {query}** ({len(hits)} results, {total} total)\n"]

    for i, rec in enumerate(hits, 1):
        metadata = rec.get("metadata", {})
        title = metadata.get("title", "Unknown")
        creators = metadata.get("creators", [])
        authors = ", ".join(c.get("name", "") for c in creators[:4])
        pub_date = metadata.get("publication_date", "")
        res_type = metadata.get("resource_type", {}).get("title", "")
        doi = metadata.get("doi", "")
        access_right = metadata.get("access_right", "")
        description = (metadata.get("description", "") or "")[:200]
        # Strip HTML tags from description
        description = re.sub(r"<[^>]+>", "", description).strip()

        # Get file download links
        files = rec.get("files", [])
        file_info = ""
        if files:
            total_size = sum(f.get("size", 0) for f in files)
            size_str = f"{total_size:,}" if total_size < 1_000_000 else f"{total_size / 1_000_000:.1f}MB"
            file_info = f"\n     Files: {len(files)} ({size_str})"
            # Show first downloadable file
            first_file = files[0]
            file_info += f"\n     Download: {first_file.get('links', {}).get('self', '')}"

        doi_line = f"\n     DOI: {doi}" if doi else ""

        formatted.append(
            f"  {i}. **{title}** [{res_type}]\n"
            f"     {authors} ({pub_date})\n"
            f"     Access: {access_right}{doi_line}{file_info}\n"
            f"     {description}..." if description else
            f"  {i}. **{title}** [{res_type}]\n"
            f"     {authors} ({pub_date})\n"
            f"     Access: {access_right}{doi_line}{file_info}"
        )

    return "\n\n".join(formatted)


# ── Tool registry ─────────────────────────────────────────────────────

DOCUMENT_TOOLS = [
    search_open_access,
    download_paper,
    annas_archive_search,
    resolve_doi_metadata,
    extract_pdf_text,
    semantic_scholar_search,
    semantic_scholar_recommend,
    search_core,
    search_springer,
    search_zenodo,
]
