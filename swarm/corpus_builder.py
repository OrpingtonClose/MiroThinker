# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Corpus builder — acquire external data when no files are attached.

When a user sends a prompt with no corpus files, this module builds an
initial corpus by:

1. Comprehending the query (LLM: entities, domains, sub-questions)
2. Generating search queries from sub-questions
3. Fanning out across available search APIs in parallel
4. Extracting full content from top URLs (Jina Reader / Firecrawl)
5. Hitting academic APIs when academic signals detected (arXiv, Semantic Scholar)
6. Assembling everything into a corpus string for the swarm engine

The module is self-contained — pure httpx calls, no ADK or Strands
framework dependencies.  Search primitives are adapted from
``apps/adk-agent/tools/search_executor.py`` and
``apps/adk-agent/tools/scout.py`` but decoupled from their pipeline state.

Usage:
    corpus = await build_corpus(query, complete_fn, config)
    result = await engine.synthesize(corpus=corpus, query=query)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API keys (read once at module load)
# ---------------------------------------------------------------------------
_BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
_EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
_KAGI_API_KEY = os.environ.get("KAGI_API_KEY", "")
_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
_PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
_JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
_MOJEEK_API_KEY = os.environ.get("MOJEEK_API_KEY", "")
_FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY", "")
_SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

_MAX_CONCURRENT = int(os.environ.get("CORPUS_BUILDER_CONCURRENCY", "6"))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CorpusBuilderConfig:
    """Configuration for the corpus builder.

    Attributes:
        max_search_queries: Maximum search queries to generate from
            sub-questions.
        max_content_extractions: Maximum URLs to extract full content from.
        max_academic_queries: Maximum academic search queries.
        fan_out_width: Number of APIs to query per search query.
        search_timeout_s: Timeout for individual search API calls.
        extraction_timeout_s: Timeout for content extraction calls.
    """

    max_search_queries: int = 12
    max_content_extractions: int = 8
    max_academic_queries: int = 4
    fan_out_width: int = 3
    search_timeout_s: float = 20.0
    extraction_timeout_s: float = 30.0


# ---------------------------------------------------------------------------
# Query comprehension (adapted from scout.py)
# ---------------------------------------------------------------------------

_COMPREHENSION_PROMPT = """\
You are a research analyst. Deeply understand what this query is about \
— not just the surface words, but the full knowledge territory.

Research query: {query}

Analyze this query and output ONLY valid JSON:
{
  "entities": ["every entity, person, substance, concept mentioned or implied"],
  "domains": ["every knowledge domain this touches — be expansive"],
  "sub_questions": ["6-10 concrete sub-questions that research needs to answer"],
  "academic_signals": true/false,
  "search_queries": [
    "8-12 specific search queries that would find authoritative content",
    "include forum queries (site:reddit.com, site:forums.t-nation.com)",
    "include academic queries (pharmacokinetics, dose-response)",
    "include practitioner queries (protocol, dosage, timing)"
  ]
}"""


@dataclass
class QueryComprehension:
    """Semantic understanding of a research query."""

    entities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    academic_signals: bool = False
    search_queries: list[str] = field(default_factory=list)


async def comprehend_query(
    query: str,
    complete: Callable[[str], Awaitable[str]],
) -> QueryComprehension:
    """Decompose a research query into entities, domains, and search queries.

    Args:
        query: The user's research query.
        complete: Async LLM completion function.

    Returns:
        QueryComprehension with entities, domains, sub-questions, and
        generated search queries.
    """
    prompt = _COMPREHENSION_PROMPT.replace("{query}", query)

    try:
        content = await complete(prompt)
    except Exception as exc:
        logger.warning("error=<%s> | query comprehension LLM call failed", exc)
        return _fallback_comprehension(query)

    if not content:
        return _fallback_comprehension(query)

    # Strip markdown fences
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    data = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        pass

    if data is None:
        # LLMs often wrap valid JSON in preamble text; extract the
        # first '{' ... last '}' span and try again.
        brace_start = content.find("{")
        brace_end = content.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                data = json.loads(content[brace_start:brace_end + 1])
                logger.debug("query comprehension JSON extracted from preamble")
            except json.JSONDecodeError:
                pass

    if data is None:
        logger.warning("query comprehension returned invalid JSON, using fallback")
        return _fallback_comprehension(query)

    return QueryComprehension(
        entities=data.get("entities", []),
        domains=data.get("domains", []),
        sub_questions=data.get("sub_questions", []),
        academic_signals=bool(data.get("academic_signals", False)),
        search_queries=data.get("search_queries", []),
    )


def _fallback_comprehension(query: str) -> QueryComprehension:
    """Heuristic fallback when LLM comprehension fails.

    Args:
        query: The user's research query.

    Returns:
        QueryComprehension built from simple word extraction.
    """
    words = [w for w in re.split(r"\W+", query.lower()) if len(w) > 3]
    return QueryComprehension(
        entities=words[:10],
        sub_questions=[query],
        search_queries=[query, f"{query} research", f"{query} protocol"],
    )


# ---------------------------------------------------------------------------
# Search API implementations (self-contained httpx calls)
# ---------------------------------------------------------------------------

async def _search_brave(query: str, num_results: int = 8) -> list[dict[str, str]]:
    """Search Brave and return structured results."""
    if not _BRAVE_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": num_results},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": _BRAVE_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("web", {}).get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("description", ""),
                    "source": "brave",
                }
                for r in results[:num_results]
            ]
    except Exception as exc:
        logger.warning("error=<%s> | brave search failed", exc)
        return []


async def _search_exa(query: str, num_results: int = 8) -> list[dict[str, str]]:
    """Search Exa and return structured results with content."""
    if not _EXA_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                json={
                    "query": query,
                    "numResults": num_results,
                    "type": "auto",
                    "contents": {
                        "text": {"maxCharacters": 5000},
                        "highlights": {"query": query},
                    },
                },
                headers={
                    "x-api-key": _EXA_API_KEY,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("text", "")[:2000],
                    "full_text": r.get("text", ""),
                    "source": "exa",
                }
                for r in results[:num_results]
            ]
    except Exception as exc:
        logger.warning("error=<%s> | exa search failed", exc)
        return []


async def _search_tavily(query: str, num_results: int = 8) -> list[dict[str, str]]:
    """Search Tavily and return structured results."""
    if not _TAVILY_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": _TAVILY_API_KEY,
                    "query": query,
                    "max_results": num_results,
                    "include_raw_content": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                    "full_text": r.get("raw_content", ""),
                    "source": "tavily",
                }
                for r in results[:num_results]
            ]
    except Exception as exc:
        logger.warning("error=<%s> | tavily search failed", exc)
        return []


async def _search_kagi(query: str) -> list[dict[str, str]]:
    """Search Kagi FastGPT and return structured results."""
    if not _KAGI_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://kagi.com/api/v0/fastgpt",
                params={"query": query},
                headers={"Authorization": f"Bot {_KAGI_API_KEY}"},
            )
            resp.raise_for_status()
            data = resp.json()
            output = data.get("data", {}).get("output", "")
            refs = data.get("data", {}).get("references", [])
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                    "source": "kagi",
                }
                for r in refs
            ]
            if output:
                results.insert(0, {
                    "title": "Kagi FastGPT synthesis",
                    "url": "",
                    "snippet": output,
                    "source": "kagi_synthesis",
                })
            return results
    except Exception as exc:
        logger.warning("error=<%s> | kagi search failed", exc)
        return []


async def _search_perplexity(query: str) -> list[dict[str, str]]:
    """Search Perplexity and return structured results."""
    if not _PERPLEXITY_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": query}],
                },
                headers={
                    "Authorization": f"Bearer {_PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            citations = data.get("citations", [])
            results = []
            if content:
                results.append({
                    "title": "Perplexity synthesis",
                    "url": "",
                    "snippet": content,
                    "source": "perplexity_synthesis",
                })
            for url in citations:
                results.append({
                    "title": "",
                    "url": url if isinstance(url, str) else "",
                    "snippet": "",
                    "source": "perplexity_citation",
                })
            return results
    except Exception as exc:
        logger.warning("error=<%s> | perplexity search failed", exc)
        return []


async def _search_mojeek(query: str, num_results: int = 8) -> list[dict[str, str]]:
    """Search Mojeek and return structured results."""
    if not _MOJEEK_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.mojeek.com/search",
                params={
                    "q": query,
                    "fmt": "json",
                    "t": num_results,
                    "api_key": _MOJEEK_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("response", {}).get("results", [])
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("desc", ""),
                    "source": "mojeek",
                }
                for r in results[:num_results]
            ]
    except Exception as exc:
        logger.warning("error=<%s> | mojeek search failed", exc)
        return []


async def _search_duckduckgo(query: str, num_results: int = 8) -> list[dict[str, str]]:
    """Search DuckDuckGo HTML (no API key required).

    Uses the DuckDuckGo HTML endpoint which is free and does not require
    an API key.  Results are parsed from the HTML response.

    Args:
        query: Search query string.
        num_results: Maximum results to return.

    Returns:
        List of search result dicts.
    """
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            )
            resp.raise_for_status()
            text = resp.text

            results: list[dict[str, str]] = []
            # Parse result blocks from the HTML
            blocks = re.findall(
                r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
                text,
                re.DOTALL,
            )
            for href, title_html, snippet_html in blocks[:num_results]:
                title = re.sub(r"<[^>]+>", "", title_html).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
                # DDG wraps URLs in a redirect — extract the actual URL
                url = href
                uddg_match = re.search(r"uddg=([^&]+)", href)
                if uddg_match:
                    from urllib.parse import unquote
                    url = unquote(uddg_match.group(1))
                if title or snippet:
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "duckduckgo",
                    })
            return results
    except Exception as exc:
        logger.warning("error=<%s> | duckduckgo search failed", exc)
        return []


async def _search_arxiv(query: str, num_results: int = 5) -> list[dict[str, str]]:
    """Search arXiv for preprints."""
    sanitised = query.encode("ascii", errors="replace").decode("ascii")
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": f"all:{sanitised}",
                    "start": 0,
                    "max_results": num_results,
                    "sortBy": "relevance",
                },
            )
            resp.raise_for_status()
            text = resp.text
            entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
            results = []
            for entry in entries[:num_results]:
                title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                title = re.sub(r"\s+", " ", title_m.group(1).strip()) if title_m else ""
                summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                summary = re.sub(r"\s+", " ", summary_m.group(1).strip()) if summary_m else ""
                id_m = re.search(r"<id>(.*?)</id>", entry)
                url = id_m.group(1).strip() if id_m else ""
                authors = re.findall(r"<name>(.*?)</name>", entry)
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": summary,
                    "authors": ", ".join(authors),
                    "source": "arxiv",
                })
            return results
    except Exception as exc:
        logger.warning("error=<%s> | arxiv search failed", exc)
        return []


async def _search_semantic_scholar(
    query: str, num_results: int = 5,
) -> list[dict[str, str]]:
    """Search Semantic Scholar for academic papers."""
    sanitised = query.encode("ascii", errors="replace").decode("ascii")
    headers: dict[str, str] = {}
    if _SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = _SEMANTIC_SCHOLAR_API_KEY
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": sanitised,
                    "limit": num_results,
                    "fields": "title,abstract,url,year,citationCount,authors",
                },
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", [])
            return [
                {
                    "title": p.get("title", ""),
                    "url": p.get("url", ""),
                    "snippet": p.get("abstract", "") or "",
                    "year": str(p.get("year", "")),
                    "citations": str(p.get("citationCount", "")),
                    "source": "semantic_scholar",
                }
                for p in papers[:num_results]
            ]
    except Exception as exc:
        logger.warning("error=<%s> | semantic scholar search failed", exc)
        return []


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

async def _extract_jina(url: str) -> str:
    """Extract full markdown content from a URL via Jina Reader."""
    if not _JINA_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "Authorization": f"Bearer {_JINA_API_KEY}",
                    "Accept": "application/json",
                    "X-Retain-Images": "none",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("data", {}).get("content", "")
            title = data.get("data", {}).get("title", "")
            if not content:
                return ""
            return f"[Source: {title} — {url}]\n\n{content}"
    except Exception as exc:
        logger.warning("url=<%s>, error=<%s> | jina extraction failed", url, exc)
        return ""


async def _extract_firecrawl(url: str) -> str:
    """Extract content from a URL via Firecrawl."""
    if not _FIRECRAWL_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                json={"url": url, "formats": ["markdown"]},
                headers={
                    "Authorization": f"Bearer {_FIRECRAWL_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("data", {}).get("markdown", "")
            title = data.get("data", {}).get("metadata", {}).get("title", "")
            if not content:
                return ""
            return f"[Source: {title} — {url}]\n\n{content}"
    except Exception as exc:
        logger.warning("url=<%s>, error=<%s> | firecrawl extraction failed", url, exc)
        return ""


async def extract_url_content(url: str) -> str:
    """Extract full content from a URL using best available extractor.

    Tries Jina first, falls back to Firecrawl.

    Args:
        url: The URL to extract content from.

    Returns:
        Extracted content string, or empty string on failure.
    """
    content = await _extract_jina(url)
    if content:
        return content
    return await _extract_firecrawl(url)


# ---------------------------------------------------------------------------
# URL deduplication and selection
# ---------------------------------------------------------------------------

_SKIP_DOMAINS = {
    "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "youtube.com", "linkedin.com", "pinterest.com",
    "amazon.com",
}


def _select_diverse_urls(
    results: list[dict[str, str]], max_count: int,
) -> list[str]:
    """Select URLs for content extraction, prioritising diversity.

    Args:
        results: Search results with 'url' keys.
        max_count: Maximum number of URLs to return.

    Returns:
        Deduplicated, diverse list of URLs.
    """
    seen_domains: set[str] = set()
    seen_urls: set[str] = set()
    selected: list[str] = []

    # Priority pass: .edu, .gov, .org first
    priority_suffixes = (".edu", ".gov", ".org", ".ac.uk")
    for r in results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        try:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
        except Exception:
            continue
        if domain in _SKIP_DOMAINS:
            continue
        if any(domain.endswith(s) for s in priority_suffixes):
            selected.append(url)
            seen_urls.add(url)
            seen_domains.add(domain)
            if len(selected) >= max_count:
                return selected

    # General pass: one URL per domain
    for r in results:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        try:
            domain = urlparse(url).netloc.lower().removeprefix("www.")
        except Exception:
            continue
        if domain in _SKIP_DOMAINS or domain in seen_domains:
            continue
        selected.append(url)
        seen_urls.add(url)
        seen_domains.add(domain)
        if len(selected) >= max_count:
            break

    return selected


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def _assemble_corpus(
    search_results: list[dict[str, str]],
    extracted_articles: list[str],
    academic_results: list[dict[str, str]],
    query: str,
) -> str:
    """Assemble all gathered material into a single corpus string.

    Args:
        search_results: All search results with snippets.
        extracted_articles: Full article texts from content extraction.
        academic_results: Academic paper results.
        query: The original user query.

    Returns:
        Assembled corpus string.
    """
    sections: list[str] = []

    sections.append(
        f"{'=' * 60}\n"
        f"RESEARCH CORPUS — Automatically acquired for query:\n"
        f"{query}\n"
        f"{'=' * 60}\n"
    )

    # Full articles (richest content)
    if extracted_articles:
        sections.append(
            f"\n{'─' * 40}\n"
            f"FULL ARTICLES ({len(extracted_articles)} extracted)\n"
            f"{'─' * 40}\n"
        )
        for article in extracted_articles:
            sections.append(article)
            sections.append("")

    # Exa/Tavily full text results (inline content from search)
    inline_texts = [
        r for r in search_results
        if r.get("full_text") and len(r.get("full_text", "")) > 500
    ]
    if inline_texts:
        sections.append(
            f"\n{'─' * 40}\n"
            f"SEARCH RESULTS WITH CONTENT ({len(inline_texts)} articles)\n"
            f"{'─' * 40}\n"
        )
        for r in inline_texts:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            source = r.get("source", "")
            sections.append(f"[Source: {title} — {url} (via {source})]")
            sections.append(r["full_text"])
            sections.append("")

    # Academic papers
    if academic_results:
        sections.append(
            f"\n{'─' * 40}\n"
            f"ACADEMIC PAPERS ({len(academic_results)} found)\n"
            f"{'─' * 40}\n"
        )
        for r in academic_results:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            year = r.get("year", "")
            citations = r.get("citations", "")
            authors = r.get("authors", "")
            meta = f" ({year})" if year else ""
            if citations:
                meta += f" [{citations} citations]"
            if authors:
                meta += f" by {authors}"
            sections.append(f"- {title}{meta} [{url}]")
            if snippet:
                sections.append(f"  Abstract: {snippet}")
            sections.append("")

    # Search snippets (least rich but broadest coverage)
    snippet_results = [
        r for r in search_results
        if r.get("snippet") and not r.get("full_text")
    ]
    if snippet_results:
        sections.append(
            f"\n{'─' * 40}\n"
            f"SEARCH SNIPPETS ({len(snippet_results)} results)\n"
            f"{'─' * 40}\n"
        )
        for r in snippet_results:
            title = r.get("title", "")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            source = r.get("source", "")
            sections.append(f"- [{title}]({url}) (via {source})")
            sections.append(f"  {snippet}")
            sections.append("")

    corpus = "\n".join(sections)

    logger.info(
        "corpus_chars=<%d>, articles=<%d>, inline=<%d>, academic=<%d>, snippets=<%d> | "
        "corpus assembly complete",
        len(corpus), len(extracted_articles), len(inline_texts),
        len(academic_results), len(snippet_results),
    )

    return corpus


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def build_corpus(
    query: str,
    complete: Callable[[str], Awaitable[str]],
    config: CorpusBuilderConfig | None = None,
) -> str:
    """Build a research corpus from external sources for a given query.

    This is the pre-swarm corpus acquisition phase. When the user sends
    a prompt with no attached files, this function goes and finds
    relevant material from the web, academic databases, and synthesis
    engines.

    Args:
        query: The user's research query.
        complete: Async LLM completion function for query comprehension.
        config: Optional configuration overrides.

    Returns:
        Assembled corpus string ready for the swarm engine.
    """
    config = config or CorpusBuilderConfig()
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

    # ── Phase 1: Comprehend the query ────────────────────────────
    logger.info("query=<%s> | starting corpus acquisition", query[:100])
    comprehension = await comprehend_query(query, complete)

    search_queries = comprehension.search_queries[:config.max_search_queries]
    if not search_queries:
        search_queries = [query, f"{query} research", f"{query} protocol"]

    logger.info(
        "entities=<%d>, domains=<%d>, queries=<%d>, academic=<%s> | "
        "query comprehension complete",
        len(comprehension.entities), len(comprehension.domains),
        len(search_queries), comprehension.academic_signals,
    )

    # ── Phase 2: Multi-API fan-out search ────────────────────────
    available_fns = _get_available_search_fns()
    all_results: list[dict[str, str]] = []

    async def _run_search(q: str, fn: Any) -> list[dict[str, str]]:
        async with semaphore:
            try:
                return await fn(q)
            except Exception as exc:
                logger.warning(
                    "query=<%s>, fn=<%s>, error=<%s> | search call failed",
                    q[:50], fn.__name__, exc,
                )
                return []

    search_tasks = []
    for i, q in enumerate(search_queries):
        # Pick a subset of APIs per query for diversity
        fns = available_fns[i % len(available_fns):] + available_fns[:i % len(available_fns)]
        for fn in fns[:config.fan_out_width]:
            search_tasks.append(_run_search(q, fn))

    if search_tasks:
        results_lists = await asyncio.gather(*search_tasks)
        for results in results_lists:
            all_results.extend(results)

    logger.info(
        "total_search_results=<%d> | search phase complete",
        len(all_results),
    )

    # ── Phase 3: Content extraction from top URLs ────────────────
    urls = _select_diverse_urls(all_results, config.max_content_extractions)
    extracted_articles: list[str] = []

    async def _extract(url: str) -> str:
        async with semaphore:
            return await extract_url_content(url)

    if urls:
        extraction_tasks = [_extract(u) for u in urls]
        extraction_results = await asyncio.gather(*extraction_tasks)
        extracted_articles = [a for a in extraction_results if a]

    logger.info(
        "urls_attempted=<%d>, articles_extracted=<%d> | content extraction complete",
        len(urls), len(extracted_articles),
    )

    # ── Phase 4: Academic search (conditional) ───────────────────
    academic_results: list[dict[str, str]] = []

    if comprehension.academic_signals:
        academic_queries = search_queries[:config.max_academic_queries]
        academic_tasks = []
        for q in academic_queries:
            academic_tasks.append(_run_search(q, _search_arxiv))
            academic_tasks.append(_run_search(q, _search_semantic_scholar))

        if academic_tasks:
            academic_lists = await asyncio.gather(*academic_tasks)
            for results in academic_lists:
                academic_results.extend(results)

        logger.info(
            "academic_results=<%d> | academic search complete",
            len(academic_results),
        )

    # ── Phase 5: Assemble corpus ─────────────────────────────────
    corpus = _assemble_corpus(
        search_results=all_results,
        extracted_articles=extracted_articles,
        academic_results=academic_results,
        query=query,
    )

    return corpus


def _get_available_search_fns() -> list[Any]:
    """Return all search functions whose API keys are configured."""
    fns: list[Any] = []
    if _EXA_API_KEY:
        fns.append(_search_exa)
    if _BRAVE_API_KEY:
        fns.append(_search_brave)
    if _TAVILY_API_KEY:
        fns.append(_search_tavily)
    if _PERPLEXITY_API_KEY:
        fns.append(_search_perplexity)
    if _MOJEEK_API_KEY:
        fns.append(_search_mojeek)
    if _KAGI_API_KEY:
        fns.append(_search_kagi)
    # DuckDuckGo is always available (no API key needed) — ensures at
    # least one search function exists even with zero configured keys
    if len(fns) < 2:
        fns.append(_search_duckduckgo)
    return fns
