# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Deep academic knowledge tools for the Strands research agent.

Provides access to massive academic databases that most research tools
overlook despite being completely free and requiring no API keys.

Sources:
  1. OpenAlex — 240M+ works, free, no key, replaces Microsoft Academic Graph
  2. PubMed/NCBI — 36M+ biomedical citations, E-utilities API
  3. Wikidata — 100M+ structured entities, SPARQL queries
  4. Google Scholar — proxy-based scraping (via Bright Data or SerpAPI)

All free APIs unless noted.
"""

from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote, quote_plus

from strands import tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# OpenAlex — 240M+ works, completely free, no key
# ═══════════════════════════════════════════════════════════════════════


@tool
def openalex_search(
    query: str,
    max_results: int = 15,
    filter_oa: bool = False,
    sort_by: str = "relevance_score",
) -> str:
    """Search OpenAlex for academic works. 240M+ works, free, no API key.

    OpenAlex replaced Microsoft Academic Graph and is the most comprehensive
    free academic database. Includes papers, books, datasets, and theses.
    Supports filtering by open access, citations, publication year, etc.

    Args:
        query: Search query (title, keywords, concepts).
        max_results: Maximum results (default 15, max 50).
        filter_oa: If True, only return open access works.
        sort_by: Sort order: "relevance_score", "cited_by_count",
                 "publication_date". Default: "relevance_score".

    Returns:
        Formatted list of works with full metadata and download links.
    """
    import httpx

    params: dict = {
        "search": query,
        "per_page": min(max_results, 50),
        "sort": f"{sort_by}:desc",
        "select": "id,title,authorships,publication_year,primary_location,"
                  "open_access,cited_by_count,doi,type,concepts,is_retracted,"
                  "abstract_inverted_index,referenced_works_count",
    }
    if filter_oa:
        params["filter"] = "open_access.is_oa:true"

    try:
        resp = httpx.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": "MiroThinker/1.0 (mailto:support@miromind.ai)"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] OpenAlex search failed: {exc}"

    works = data.get("results", [])
    total = data.get("meta", {}).get("count", 0)

    if not works:
        return f"No OpenAlex results for: {query}"

    formatted = [f"**OpenAlex: {query}** ({len(works)} of {total:,} total)\n"]
    for i, work in enumerate(works[:max_results], 1):
        title = work.get("title", "Unknown")
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])[:5]
        ]
        authors_str = ", ".join(a for a in authors if a)
        if len(work.get("authorships", [])) > 5:
            authors_str += f" +{len(work['authorships']) - 5} more"
        year = work.get("publication_year", "")
        citations = work.get("cited_by_count", 0)
        refs_count = work.get("referenced_works_count", 0)
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        work_type = work.get("type", "")
        is_retracted = work.get("is_retracted", False)

        oa = work.get("open_access", {})
        is_oa = oa.get("is_oa", False)
        pdf_url = oa.get("oa_url", "")

        loc = work.get("primary_location", {}) or {}
        source = (loc.get("source") or {}).get("display_name", "")

        # Reconstruct abstract from inverted index
        abstract = ""
        abstract_idx = work.get("abstract_inverted_index")
        if abstract_idx and isinstance(abstract_idx, dict):
            word_positions = []
            for word, positions in abstract_idx.items():
                for pos in positions:
                    word_positions.append((pos, word))
            word_positions.sort()
            abstract = " ".join(w for _, w in word_positions)[:300]

        # Top concepts
        concepts = work.get("concepts", [])
        concept_names = [c.get("display_name", "") for c in concepts[:3]]

        retracted_flag = " ⚠ RETRACTED" if is_retracted else ""
        oa_flag = " [OPEN ACCESS]" if is_oa else ""

        formatted.append(
            f"  {i}. **{title}**{retracted_flag}{oa_flag}\n"
            f"     {authors_str} ({year})\n"
            f"     {source} | Type: {work_type}\n"
            f"     Citations: {citations} | References: {refs_count}\n"
            + (f"     DOI: {doi}\n" if doi else "")
            + (f"     PDF: {pdf_url}\n" if pdf_url else "")
            + (f"     Concepts: {', '.join(concept_names)}\n" if concept_names else "")
            + (f"     {abstract}..." if abstract else "")
        )

    return "\n\n".join(formatted)


@tool
def openalex_get_work(work_id: str) -> str:
    """Get full details for an OpenAlex work by ID or DOI.

    Args:
        work_id: OpenAlex work ID (e.g. "W2741809807") or DOI
                 (e.g. "10.1038/s41586-019-1099-1").

    Returns:
        Full metadata including abstract, references, and related works.
    """
    import httpx

    # Accept DOI or OpenAlex ID
    if work_id.startswith("10."):
        url = f"https://api.openalex.org/works/doi:{work_id}"
    elif work_id.startswith("https://doi.org/"):
        url = f"https://api.openalex.org/works/doi:{work_id[16:]}"
    else:
        url = f"https://api.openalex.org/works/{work_id}"

    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "MiroThinker/1.0 (mailto:support@miromind.ai)"},
            timeout=30,
        )
        resp.raise_for_status()
        work = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] OpenAlex work lookup failed: {exc}"

    title = work.get("title", "Unknown")
    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
    ]
    year = work.get("publication_year", "")
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    citations = work.get("cited_by_count", 0)
    is_retracted = work.get("is_retracted", False)

    # Abstract
    abstract = ""
    abstract_idx = work.get("abstract_inverted_index")
    if abstract_idx and isinstance(abstract_idx, dict):
        word_positions = []
        for word, positions in abstract_idx.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        abstract = " ".join(w for _, w in word_positions)

    oa = work.get("open_access", {})
    pdf_url = oa.get("oa_url", "")

    # Referenced works (first 10)
    refs = work.get("referenced_works", [])[:10]

    output = [
        f"**{title}**",
        f"Authors: {', '.join(authors)}",
        f"Year: {year} | Citations: {citations}",
    ]
    if is_retracted:
        output.append("⚠ THIS PAPER HAS BEEN RETRACTED")
    if doi:
        output.append(f"DOI: {doi}")
    if pdf_url:
        output.append(f"PDF: {pdf_url}")
    if abstract:
        output.append(f"\n**Abstract:**\n{abstract}")
    if refs:
        output.append(f"\n**References ({len(work.get('referenced_works', []))} total, showing {len(refs)}):**")
        for ref in refs:
            output.append(f"  - {ref}")

    return "\n".join(output)


@tool
def openalex_citation_network(
    work_id: str,
    direction: str = "citations",
    max_results: int = 10,
) -> str:
    """Get citation network for an OpenAlex work (who cites it / what it cites).

    Args:
        work_id: OpenAlex work ID (e.g. "W2741809807") or DOI.
        direction: "citations" (papers citing this work) or
                   "references" (papers this work cites).
        max_results: Maximum results (default 10).

    Returns:
        List of citing/referenced works with metadata.
    """
    import httpx

    # Resolve ID
    if work_id.startswith("10.") or work_id.startswith("https://doi.org/"):
        doi = work_id.replace("https://doi.org/", "")
        try:
            resp = httpx.get(
                f"https://api.openalex.org/works/doi:{doi}",
                headers={"User-Agent": "MiroThinker/1.0 (mailto:support@miromind.ai)"},
                timeout=15,
            )
            resp.raise_for_status()
            oa_id = resp.json().get("id", "").split("/")[-1]
        except Exception:
            return f"[TOOL_ERROR] Could not resolve DOI: {work_id}"
    else:
        oa_id = work_id

    if direction == "citations":
        filter_param = f"cites:{oa_id}"
    else:
        filter_param = f"cited_by:{oa_id}"

    try:
        resp = httpx.get(
            "https://api.openalex.org/works",
            params={
                "filter": filter_param,
                "per_page": min(max_results, 50),
                "sort": "cited_by_count:desc",
                "select": "id,title,authorships,publication_year,cited_by_count,doi,open_access",
            },
            headers={"User-Agent": "MiroThinker/1.0 (mailto:support@miromind.ai)"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] OpenAlex citation network failed: {exc}"

    works = data.get("results", [])
    total = data.get("meta", {}).get("count", 0)
    label = "Papers citing" if direction == "citations" else "References from"

    if not works:
        return f"No {direction} found for: {work_id}"

    formatted = [f"**{label} {work_id}** ({len(works)} of {total:,} total)\n"]
    for i, work in enumerate(works[:max_results], 1):
        title = work.get("title", "Unknown")
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])[:3]
        ]
        year = work.get("publication_year", "")
        citations = work.get("cited_by_count", 0)
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        oa = work.get("open_access", {})
        pdf_url = oa.get("oa_url", "")

        formatted.append(
            f"  {i}. **{title}**\n"
            f"     {', '.join(a for a in authors if a)} ({year}) | Citations: {citations}\n"
            + (f"     DOI: {doi}\n" if doi else "")
            + (f"     PDF: {pdf_url}" if pdf_url else "")
        )

    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# PubMed / NCBI E-utilities — 36M+ biomedical citations
# ═══════════════════════════════════════════════════════════════════════


@tool
def search_pubmed(
    query: str,
    max_results: int = 10,
    sort: str = "relevance",
) -> str:
    """Search PubMed for biomedical literature. Free, no key needed.

    36M+ biomedical citations including MEDLINE, life science journals,
    and online books. Use NCBI E-utilities API.

    Args:
        query: Search query (MeSH terms, keywords, author names).
               Supports PubMed search syntax: [MeSH], [Author], [Journal], etc.
        max_results: Maximum results (default 10, max 100).
        sort: Sort order: "relevance", "pub_date", "most_recent".

    Returns:
        Formatted list of PubMed articles with abstracts.
    """
    import httpx

    email = os.environ.get("NCBI_EMAIL", os.environ.get("UNPAYWALL_EMAIL", "research@miromind.ai"))

    # Step 1: Search for PMIDs
    try:
        search_resp = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": min(max_results, 100),
                "sort": sort,
                "retmode": "json",
                "email": email,
                "tool": "MiroThinker",
            },
            timeout=30,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] PubMed search failed: {exc}"

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    total_count = search_data.get("esearchresult", {}).get("count", "0")

    if not id_list:
        return f"No PubMed results for: {query}"

    # Step 2: Fetch details for each PMID
    try:
        fetch_resp = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
                "email": email,
                "tool": "MiroThinker",
            },
            timeout=30,
        )
        fetch_resp.raise_for_status()
        fetch_data = fetch_resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] PubMed fetch failed: {exc}"

    results = fetch_data.get("result", {})
    uid_order = results.get("uids", id_list)

    formatted = [f"**PubMed: {query}** ({len(uid_order)} of {total_count} total)\n"]
    for i, uid in enumerate(uid_order[:max_results], 1):
        article = results.get(uid, {})
        if not article or not isinstance(article, dict):
            continue

        title = article.get("title", "Unknown")
        authors = article.get("authors", [])
        author_str = ", ".join(a.get("name", "") for a in authors[:5])
        if len(authors) > 5:
            author_str += f" +{len(authors) - 5} more"
        pub_date = article.get("pubdate", "")
        source = article.get("source", "")
        doi_list = article.get("articleids", [])
        doi = ""
        for aid in doi_list:
            if aid.get("idtype") == "doi":
                doi = aid.get("value", "")
                break
        pmid = uid

        formatted.append(
            f"  {i}. **{title}**\n"
            f"     {author_str}\n"
            f"     {source} ({pub_date})\n"
            f"     PMID: {pmid}\n"
            + (f"     DOI: {doi}\n" if doi else "")
            + f"     URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        )

    return "\n\n".join(formatted)


@tool
def pubmed_get_abstract(pmid: str) -> str:
    """Get full abstract for a PubMed article by PMID.

    Args:
        pmid: PubMed ID (e.g. "12345678").

    Returns:
        Full article metadata with abstract text.
    """
    import httpx

    email = os.environ.get("NCBI_EMAIL", os.environ.get("UNPAYWALL_EMAIL", "research@miromind.ai"))

    try:
        resp = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": pmid,
                "rettype": "abstract",
                "retmode": "text",
                "email": email,
                "tool": "MiroThinker",
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text[:5000] if resp.text else f"No abstract found for PMID: {pmid}"
    except Exception as exc:
        return f"[TOOL_ERROR] PubMed abstract fetch failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════
# Wikidata — 100M+ structured entities, SPARQL
# ═══════════════════════════════════════════════════════════════════════


@tool
def wikidata_search(query: str, max_results: int = 10) -> str:
    """Search Wikidata for structured entities. Free, no key needed.

    100M+ entities with structured properties and relationships. Useful
    for verifying facts, finding connections between entities, and
    discovering structured data about people, organizations, events,
    chemicals, diseases, etc.

    Args:
        query: Search query (entity name, concept).
        max_results: Maximum results (default 10).

    Returns:
        List of matching Wikidata entities with descriptions.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": min(max_results, 50),
                "format": "json",
            },
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Wikidata search failed: {exc}"

    entities = data.get("search", [])
    if not entities:
        return f"No Wikidata entities found for: {query}"

    formatted = [f"**Wikidata entities for: {query}** ({len(entities)} results)\n"]
    for i, entity in enumerate(entities[:max_results], 1):
        qid = entity.get("id", "")
        label = entity.get("label", "")
        description = entity.get("description", "")
        url = entity.get("concepturi", f"https://www.wikidata.org/wiki/{qid}")

        formatted.append(
            f"  {i}. **{label}** ({qid})\n"
            f"     {description}\n"
            f"     URL: {url}"
        )

    return "\n\n".join(formatted)


@tool
def wikidata_sparql(query: str) -> str:
    """Execute a SPARQL query on Wikidata. Free, no key needed.

    Wikidata's SPARQL endpoint allows complex structured queries across
    100M+ entities. Use this for finding relationships, aggregations,
    and structured data that text search can't provide.

    Args:
        query: SPARQL query string. Use the Wikidata Query Service syntax.
               Example: "SELECT ?item ?itemLabel WHERE { ?item wdt:P31 wd:Q11344. SERVICE wikibase:label { bd:serviceParam wikibase:language 'en'. } } LIMIT 10"
               (finds instances of chemical elements)

    Returns:
        Query results in formatted table.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://query.wikidata.org/sparql",
            params={"query": query, "format": "json"},
            timeout=60,
            headers={
                "User-Agent": "MiroThinker/1.0 (research agent)",
                "Accept": "application/sparql-results+json",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Wikidata SPARQL query failed: {exc}"

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return "No results from SPARQL query."

    # Get column names
    columns = list(bindings[0].keys()) if bindings else []

    formatted = [f"**Wikidata SPARQL results** ({len(bindings)} rows)\n"]

    # Header
    formatted.append("| " + " | ".join(columns) + " |")
    formatted.append("| " + " | ".join(["---"] * len(columns)) + " |")

    # Rows (cap at 50)
    for row in bindings[:50]:
        values = []
        for col in columns:
            cell = row.get(col, {})
            val = cell.get("value", "")
            # Shorten Wikidata URIs
            if val.startswith("http://www.wikidata.org/entity/"):
                val = val.split("/")[-1]
            values.append(val[:60])
        formatted.append("| " + " | ".join(values) + " |")

    if len(bindings) > 50:
        formatted.append(f"\n... and {len(bindings) - 50} more rows")

    return "\n".join(formatted)


@tool
def wikidata_get_entity(qid: str) -> str:
    """Get detailed properties for a Wikidata entity by QID.

    Args:
        qid: Wikidata entity ID (e.g. "Q42" for Douglas Adams).

    Returns:
        Entity properties including labels, descriptions, aliases, and claims.
    """
    import httpx

    try:
        resp = httpx.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": qid,
                "languages": "en",
                "format": "json",
                "props": "labels|descriptions|aliases|claims|sitelinks",
            },
            timeout=30,
            headers={"User-Agent": "MiroThinker/1.0 (research agent)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[TOOL_ERROR] Wikidata entity fetch failed: {exc}"

    entities = data.get("entities", {})
    entity = entities.get(qid, {})
    if not entity or "missing" in entity:
        return f"Entity {qid} not found on Wikidata."

    label = entity.get("labels", {}).get("en", {}).get("value", qid)
    description = entity.get("descriptions", {}).get("en", {}).get("value", "")
    aliases = [a.get("value", "") for a in entity.get("aliases", {}).get("en", [])]

    output = [
        f"**{label}** ({qid})",
        f"Description: {description}",
    ]
    if aliases:
        output.append(f"Aliases: {', '.join(aliases[:10])}")

    # Extract key claims (properties)
    claims = entity.get("claims", {})
    if claims:
        output.append(f"\n**Properties** ({len(claims)} total, showing top 20):")
        for prop_id, claim_list in list(claims.items())[:20]:
            for claim in claim_list[:1]:  # Just first value per property
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                val_type = datavalue.get("type", "")

                if val_type == "wikibase-entityid":
                    val = datavalue.get("value", {}).get("id", "")
                elif val_type == "string":
                    val = datavalue.get("value", "")
                elif val_type == "time":
                    val = datavalue.get("value", {}).get("time", "")
                elif val_type == "quantity":
                    val = datavalue.get("value", {}).get("amount", "")
                elif val_type == "monolingualtext":
                    val = datavalue.get("value", {}).get("text", "")
                else:
                    val = str(datavalue.get("value", ""))[:80]

                output.append(f"  {prop_id}: {val}")

    # Sitelinks
    sitelinks = entity.get("sitelinks", {})
    if "enwiki" in sitelinks:
        wiki_title = sitelinks["enwiki"].get("title", "")
        output.append(f"\nWikipedia: https://en.wikipedia.org/wiki/{quote(wiki_title)}")

    return "\n".join(output)


@tool
def search_google_scholar(
    query: str,
    max_results: int = 10,
) -> str:
    """Search Google Scholar for academic papers. Needs proxy (Bright Data).

    Google Scholar has the widest coverage of any academic search engine but
    actively blocks automated access. Uses Bright Data proxy when available,
    falls back to SerpAPI.

    Args:
        query: Search query.
        max_results: Maximum results (default 10).

    Returns:
        Formatted list of Google Scholar results.
    """
    import httpx

    # Try SerpAPI first (most reliable)
    serpapi_key = os.environ.get("SERPAPI_KEY", "")
    if serpapi_key:
        try:
            resp = httpx.get(
                "https://serpapi.com/search.json",
                params={
                    "engine": "google_scholar",
                    "q": query,
                    "num": min(max_results, 20),
                    "api_key": serpapi_key,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic_results", [])

            if results:
                formatted = [f"**Google Scholar: {query}** ({len(results)} results)\n"]
                for i, r in enumerate(results[:max_results], 1):
                    title = r.get("title", "Unknown")
                    snippet = r.get("snippet", "")[:200]
                    link = r.get("link", "")
                    pub_info = r.get("publication_info", {}).get("summary", "")
                    citations = r.get("inline_links", {}).get("cited_by", {}).get("total", 0)

                    formatted.append(
                        f"  {i}. **{title}**\n"
                        f"     {pub_info}\n"
                        f"     Citations: {citations}\n"
                        f"     URL: {link}\n"
                        + (f"     {snippet}..." if snippet else "")
                    )
                return "\n\n".join(formatted)
        except Exception as exc:
            logger.debug("SerpAPI Google Scholar failed: %s", exc)

    # Fallback: direct scraping with proxy
    bright_data_key = os.environ.get("BRIGHT_DATA_API_KEY", "")
    customer_id = os.environ.get("BRIGHTDATA_CUSTOMER_ID", "hl_dc044bf4")
    zone = os.environ.get("BRIGHTDATA_ZONE", "mcp_unlocker")

    if bright_data_key:
        proxy_url = f"http://{customer_id}-zone-{zone}:{bright_data_key}@brd.superproxy.io:33335"
        try:
            resp = httpx.get(
                "https://scholar.google.com/scholar",
                params={"q": query, "num": max_results},
                proxy=proxy_url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0"},
            )
            resp.raise_for_status()
            # Basic parsing of Google Scholar HTML
            import re
            titles = re.findall(r'<h3[^>]*class="gs_rt"[^>]*>(.*?)</h3>', resp.text, re.DOTALL)
            snippets = re.findall(r'<div class="gs_rs">(.*?)</div>', resp.text, re.DOTALL)

            if titles:
                formatted = [f"**Google Scholar: {query}** ({len(titles)} results)\n"]
                for i, title_html in enumerate(titles[:max_results], 1):
                    # Clean HTML
                    title = re.sub(r"<[^>]+>", "", title_html).strip()
                    snippet = re.sub(r"<[^>]+>", "", snippets[i - 1]).strip()[:200] if i <= len(snippets) else ""
                    formatted.append(
                        f"  {i}. **{title}**\n"
                        + (f"     {snippet}..." if snippet else "")
                    )
                return "\n\n".join(formatted)
        except Exception as exc:
            logger.debug("Google Scholar proxy scraping failed: %s", exc)

    return (
        f"Google Scholar search requires SERPAPI_KEY or BRIGHT_DATA_API_KEY.\n"
        f"Direct URL: https://scholar.google.com/scholar?q={quote_plus(query)}"
    )


# ── Tool registry ─────────────────────────────────────────────────────

KNOWLEDGE_TOOLS = [
    openalex_search,
    openalex_get_work,
    openalex_citation_network,
    search_pubmed,
    pubmed_get_abstract,
    wikidata_search,
    wikidata_sparql,
    wikidata_get_entity,
    search_google_scholar,
]
