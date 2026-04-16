# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Tool wiring for the Strands Venice agent.

Combines:
- MCP servers (Brave, Firecrawl, Exa, Kagi, Semantic Scholar, arXiv,
  Wikipedia, DuckDB, TranscriptAPI, Bright Data) for rich tool ecosystems
- Native @tool-decorated functions for direct API providers and research
  management (DuckDuckGo, Mojeek, Jina Reader, Google/Serper, Perplexity,
  Grok, Tavily, Exa multi-search, findings store, knowledge graph)

Tools are organised into tiers:
  Tier 1 — Uncensored search: DuckDuckGo, Brave, Exa, Mojeek
  Tier 2 — Content extraction: Jina Reader, Firecrawl, Kagi
  Tier 3 — Censored fallback: Google/Serper
  Deep Research — Perplexity, Grok, Tavily, Exa multi-search
  Research Mgmt — store/read findings, knowledge graph

Reference: apps/adk-agent/tools/mcp_tools.py, research_tools.py,
  deep_research_tools.py, knowledge_graph.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from strands import tool
from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)


def _full_env(**overrides):
    """Return a copy of the current environment with *overrides* applied.

    MCP server subprocesses inherit PATH, HOME, etc. so that ``npx`` and
    other tools resolve correctly.
    """
    env = dict(os.environ)
    env.update(overrides)
    return env


# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Uncensored native tools (no MCP server needed)
# ═══════════════════════════════════════════════════════════════════════


@tool
def duckduckgo_search(query: str, max_results: int = 10) -> str:
    """Search the web using DuckDuckGo. Free, no API key, no tracking, uncensored results.

    Use this as your go-to first search — it's always available and has no content filtering.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results with titles, URLs, and snippets.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return f"No DuckDuckGo results for: {query}"

    formatted = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        body = r.get("body", "")
        formatted.append(f"{i}. [{title}]({url})\n   {body}")
    return "\n\n".join(formatted)


@tool
def mojeek_search(query: str, max_results: int = 10) -> str:
    """Search using Mojeek's independent crawler. Not a Google/Bing proxy — unique results.

    Mojeek has its own crawler and index, so it surfaces content that other
    engines miss entirely. Requires MOJEEK_API_KEY.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted search results with titles, URLs, and descriptions.
    """
    import httpx

    api_key = os.environ.get("MOJEEK_API_KEY", "")
    if not api_key:
        return "Mojeek API key not configured. Set MOJEEK_API_KEY in .env."

    resp = httpx.get(
        "https://api.mojeek.com/search",
        params={"q": query, "fmt": "json", "t": max_results, "api_key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("response", {}).get("results", [])
    if not results:
        return f"No Mojeek results for: {query}"

    formatted = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        desc = r.get("desc", "")
        formatted.append(f"{i}. [{title}]({url})\n   {desc}")
    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — Content extraction native tools
# ═══════════════════════════════════════════════════════════════════════


@tool
def jina_read_url(url: str) -> str:
    """Extract clean text/markdown from any URL using Jina Reader.

    Converts web pages into clean, readable markdown. Fast and reliable
    for most pages. Requires JINA_API_KEY.

    Args:
        url: The URL to extract content from.

    Returns:
        Clean markdown text extracted from the URL (truncated to 15000 chars).
    """
    import httpx

    headers = {}
    api_key = os.environ.get("JINA_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.get(
        f"https://r.jina.ai/{url}",
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text[:15000]


# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — Censored fallback native tools
# ═══════════════════════════════════════════════════════════════════════


@tool
def google_search(query: str, max_results: int = 10) -> str:
    """Search Google via Serper API. Powerful but censored — use as fallback.

    Only use this when uncensored sources (DuckDuckGo, Brave, Exa, Mojeek)
    don't have what you need. Requires SERPER_API_KEY.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 10).

    Returns:
        Formatted Google search results with titles, URLs, and snippets.
    """
    import httpx

    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return "Serper API key not configured. Set SERPER_API_KEY in .env."

    resp = httpx.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    organic = data.get("organic", [])
    if not organic:
        return f"No Google results for: {query}"

    formatted = []
    for i, r in enumerate(organic[:max_results], 1):
        title = r.get("title", "")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        formatted.append(f"{i}. [{title}]({link})\n   {snippet}")
    return "\n\n".join(formatted)


# ═══════════════════════════════════════════════════════════════════════
# DEEP RESEARCH native tools
# ═══════════════════════════════════════════════════════════════════════


@tool
def perplexity_deep_research(query: str, model: str = "sonar-deep-research") -> str:
    """Run a deep research query via Perplexity's sonar-deep-research model.

    Perplexity performs autonomous multi-step web research and returns a
    comprehensive answer with citations. Best for broad, complex topics
    that need synthesis across many sources. High latency (2-4 min).

    Args:
        query: The research question to investigate deeply.
        model: Perplexity model — "sonar-deep-research" (thorough, slow)
            or "sonar" (faster, less thorough).

    Returns:
        Research results with citations, or error message.
    """
    import httpx

    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        return "[TOOL_ERROR] Perplexity deep research unavailable: PERPLEXITY_API_KEY not set."

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research data retrieval agent. Return "
                    "factual findings with full citations (URLs, source "
                    "names, dates). No disclaimers or ethical commentary."
                ),
            },
            {"role": "user", "content": query},
        ],
    }

    try:
        resp = httpx.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )
        if resp.status_code != 200:
            return f"[TOOL_ERROR] Perplexity returned HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        citations = data.get("citations", [])
        citation_text = ""
        if citations:
            citation_lines = []
            for i, url in enumerate(citations[:20], 1):
                if isinstance(url, str):
                    citation_lines.append(f"  [{i}] {url}")
                elif isinstance(url, dict):
                    citation_lines.append(
                        f"  [{i}] {url.get('url', url.get('title', ''))}"
                    )
            if citation_lines:
                citation_text = "\n\n**Sources cited:**\n" + "\n".join(citation_lines)

        result = content + citation_text
        return result if result.strip() else f"Perplexity returned empty results for: {query}"

    except Exception as exc:
        return f"[TOOL_ERROR] Perplexity deep research error: {exc}"


@tool
def grok_deep_research(query: str, search_type: str = "both") -> str:
    """Search the web and/or X/Twitter via Grok's Responses API.

    Grok performs autonomous web + X searches (5-15 per call) and returns
    cited results. Best for current events, social media sentiment, and
    topics with strong X/Twitter discourse. Requires XAI_API_KEY.

    Args:
        query: The research question to investigate.
        search_type: "web", "x" (X/Twitter only), or "both".

    Returns:
        Formatted search results with citations, or error message.
    """
    import httpx

    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        return "[TOOL_ERROR] Grok deep research unavailable: XAI_API_KEY not set."

    api_base = os.environ.get("GROK_RESPONSES_API_BASE", "https://api.x.ai")
    grok_model = os.environ.get("GROK_SEARCH_MODEL", "grok-3")

    tools_list = []
    if search_type in ("web", "both"):
        tools_list.append({"type": "web_search"})
    if search_type in ("x", "both"):
        tools_list.append({"type": "x_search"})
    if not tools_list:
        tools_list = [{"type": "web_search"}, {"type": "x_search"}]

    payload = {
        "model": grok_model,
        "input": query,
        "instructions": (
            "You are a research data retrieval agent. Search for factual "
            "information and return findings with full citations (URLs, "
            "source names, dates). No disclaimers or ethical commentary. "
            "Return raw findings in structured format."
        ),
        "tools": tools_list,
    }

    try:
        resp = httpx.post(
            f"{api_base}/v1/responses",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=300.0,
        )
        if resp.status_code != 200:
            return f"[TOOL_ERROR] Grok returned HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
        output_items = data.get("output", [])
        if not output_items:
            return f"Grok returned no output for: {query}"

        search_count = 0
        search_types_used = []
        citations = []

        for item in output_items:
            item_type = item.get("type", "")
            if item_type in ("web_search_call", "x_search_call"):
                search_count += 1
                st = "web" if item_type == "web_search_call" else "X/Twitter"
                search_types_used.append(st)
            if item_type == "web_search_result":
                for result in item.get("results", []):
                    citations.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                    })

        assistant_text = ""
        for item in reversed(output_items):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for block in item.get("content", []):
                    text = block.get("text", "") or block.get("output_text", "")
                    if text:
                        assistant_text = text
                        break
                if assistant_text:
                    break

        if not assistant_text:
            return f"Grok produced no text output for: {query}"

        search_summary = ", ".join(set(search_types_used)) or "unknown"
        header = (
            f"**Grok Deep Search: {query}**\n"
            f"({search_count} searches via {search_summary})\n\n"
        )
        citation_text = ""
        if citations:
            citation_lines = [
                f"  [{i}] {c['title']} — {c['url']}"
                for i, c in enumerate(citations[:20], 1)
            ]
            citation_text = "\n\n**Sources cited:**\n" + "\n".join(citation_lines)

        return header + assistant_text + citation_text

    except Exception as exc:
        return f"[TOOL_ERROR] Grok deep research error: {exc}"


@tool
def tavily_deep_research(query: str, search_depth: str = "advanced") -> str:
    """Run an advanced search via Tavily's search API.

    Tavily provides AI-optimised search results with extracted content.
    "advanced" triggers deeper crawling and extraction. Best for factual
    questions that need precise, structured data. Requires TAVILY_API_KEY.

    Args:
        query: The research question to investigate.
        search_depth: "basic" for quick results, "advanced" for deeper extraction.

    Returns:
        Formatted search results with content extracts, or error message.
    """
    import httpx

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "[TOOL_ERROR] Tavily deep research unavailable: TAVILY_API_KEY not set."

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
        "max_results": 10,
    }

    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=120.0,
        )
        if resp.status_code != 200:
            return f"[TOOL_ERROR] Tavily returned HTTP {resp.status_code}: {resp.text[:300]}"

        data = resp.json()
        answer = data.get("answer", "")
        results = data.get("results", [])

        if not answer and not results:
            return f"Tavily returned no results for: {query}"

        parts = [f"**Tavily Deep Search: {query}**\n"]
        if answer:
            parts.append(f"Summary: {answer}\n")

        if results:
            parts.append("Results:")
            for i, r in enumerate(results[:10], 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                content = r.get("content", "")
                score = r.get("score", 0)
                parts.append(f"\n  [{i}] {title}")
                parts.append(f"      URL: {url}")
                if score:
                    parts.append(f"      Relevance: {score:.2f}")
                if content:
                    parts.append(f"      {content}")

        return "\n".join(parts)

    except Exception as exc:
        return f"[TOOL_ERROR] Tavily deep research error: {exc}"


@tool
def exa_multi_search(queries: str, num_results_per_query: int = 5) -> str:
    """Run multiple Exa searches in parallel and return unified results.

    Use this when you need to compare multiple topics simultaneously
    (e.g. "compare these 6 companies") or gather data on several entities
    at once. All queries run in parallel for speed.

    Args:
        queries: JSON array of search query strings (max 10).
            Example: '["query one", "query two", "query three"]'
        num_results_per_query: Number of results per query (default 5, max 8).

    Returns:
        JSON object with per-query results and a unified source list.
    """
    import httpx

    api_key = os.environ.get("EXA_API_KEY", "")
    if not api_key:
        return json.dumps({"error": "EXA_API_KEY not set"})

    try:
        query_list = json.loads(queries)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": f"Invalid JSON queries: {queries[:200]}"})

    if not isinstance(query_list, list):
        query_list = [str(query_list)]
    query_list = query_list[:10]
    num_results_per_query = min(num_results_per_query, 8)

    def _search_one(q: str) -> dict:
        try:
            resp = httpx.post(
                "https://api.exa.ai/search",
                json={
                    "query": q,
                    "numResults": num_results_per_query,
                    "type": "auto",
                    "contents": {
                        "text": {"maxCharacters": 5000},
                        "highlights": {"query": q},
                    },
                },
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return {"query": q, "count": len(results), "results": results}
        except Exception as exc:
            return {"query": q, "count": 0, "results": [], "error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(len(query_list), 5)) as pool:
        futures = {pool.submit(_search_one, q): q for q in query_list}
        raw_results = []
        for future in as_completed(futures):
            raw_results.append(future.result())

    # Sort to match original query order
    order = {q: i for i, q in enumerate(query_list)}
    raw_results.sort(key=lambda b: order.get(b["query"], 999))

    all_sources = []
    total_results = 0
    for batch in raw_results:
        total_results += batch["count"]
        for r in batch.get("results", []):
            all_sources.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "query": batch["query"],
            })

    output = {
        "queries_executed": len(query_list),
        "total_results": total_results,
        "per_query": [
            {
                "query": b["query"],
                "count": b["count"],
                "error": b.get("error"),
                "top_results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": " ".join(r.get("highlights", []))[:200]
                        or r.get("text", "")[:200],
                    }
                    for r in b.get("results", [])[:5]
                ],
            }
            for b in raw_results
        ],
        "all_sources": all_sources[:30],
    }
    return json.dumps(output, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# RESEARCH MANAGEMENT native tools (findings store)
# ═══════════════════════════════════════════════════════════════════════

_FINDINGS_DIR = Path(os.environ.get(
    "FINDINGS_DIR", os.path.join(os.path.expanduser("~"), ".mirothinker")
))
_FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
_findings_file: Path = _FINDINGS_DIR / "findings.jsonl"


@tool
def store_finding(
    name: str,
    url: str,
    category: str,
    summary: str,
    rating: int = 0,
) -> str:
    """Store an evaluated finding to persistent JSONL storage.

    Findings persist outside the LLM context window so older findings
    can be trimmed without losing accumulated research data.

    Args:
        name: Short name / title of the finding.
        url: Source URL.
        category: Category (e.g. "vendor", "forum", "news", "academic").
        summary: One-paragraph evaluation summary.
        rating: Quality rating 1-10 (0 = unrated).

    Returns:
        Confirmation message.
    """
    finding = {
        "name": name,
        "url": url,
        "category": category,
        "summary": summary,
        "rating": rating,
        "ts": time.time(),
    }
    with open(_findings_file, "a") as f:
        f.write(json.dumps(finding, ensure_ascii=False) + "\n")

    logger.info("Stored finding: %s (%s)", name, category)
    return f"Stored: {name} [{category}] (rating={rating})"


@tool
def read_findings(category: str = "") -> str:
    """Read back all stored findings, optionally filtered by category.

    Args:
        category: If non-empty, only return findings matching this category.

    Returns:
        JSON array of finding objects.
    """
    if not _findings_file.exists():
        return json.dumps([])

    findings = []
    for line in _findings_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if category and obj.get("category", "") != category:
            continue
        findings.append(obj)

    return json.dumps(findings, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH native tools
# ═══════════════════════════════════════════════════════════════════════

_GRAPH_DIR = _FINDINGS_DIR
_graph_file: Path = _GRAPH_DIR / "knowledge_graph.jsonl"

# In-memory graph — persisted to JSONL on every mutation
_kg_entities: dict[str, dict] = {}
_kg_edges: list[dict] = []


def _kg_persist() -> None:
    """Write the full knowledge graph to JSONL."""
    with open(_graph_file, "w") as f:
        for eid, ent in _kg_entities.items():
            f.write(json.dumps({"_t": "entity", "id": eid, **ent}, ensure_ascii=False) + "\n")
        for edge in _kg_edges:
            f.write(json.dumps({"_t": "edge", **edge}, ensure_ascii=False) + "\n")


def load_knowledge_graph() -> None:
    """Load graph from JSONL file (call on startup / resume)."""
    global _kg_entities, _kg_edges
    _kg_entities = {}
    _kg_edges = []
    if not _graph_file.exists():
        return
    for line in _graph_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_t") == "entity":
            eid = obj.pop("id")
            obj.pop("_t")
            _kg_entities[eid] = obj
        elif obj.get("_t") == "edge":
            obj.pop("_t")
            _kg_edges.append(obj)


@tool
def add_entity(
    entity_id: str,
    entity_type: str,
    name: str,
    properties: str = "{}",
) -> str:
    """Add or update an entity in the knowledge graph.

    Use this to register findings, sources, topics, or any named concept
    you discover during research. Entities are the nodes of the graph.

    Args:
        entity_id: Unique identifier (e.g. "model:deepseek-r1", "source:example.com").
        entity_type: Category — "model", "source", "topic", "vendor", "person", etc.
        name: Human-readable name.
        properties: JSON string of additional properties (e.g. '{"url": "...", "rating": 8}').

    Returns:
        Confirmation message.
    """
    try:
        props = json.loads(properties) if properties else {}
    except json.JSONDecodeError:
        props = {"raw": properties}

    _kg_entities[entity_id] = {
        "type": entity_type,
        "name": name,
        "props": props,
        "ts": time.time(),
    }
    _kg_persist()
    return f"Entity added: {entity_id} ({entity_type}: {name})"


@tool
def add_edge(
    source_id: str,
    target_id: str,
    relationship: str,
    weight: float = 1.0,
) -> str:
    """Add a relationship (edge) between two entities in the knowledge graph.

    Use this to record that two entities are related — e.g. a source
    corroborates a finding, a model is published by a vendor, etc.

    Args:
        source_id: Entity ID of the source node.
        target_id: Entity ID of the target node.
        relationship: Type of relationship (e.g. "corroborates", "published_by",
            "contradicts", "mentions", "alternative_to").
        weight: Strength of the relationship (0.0-1.0, default 1.0).

    Returns:
        Confirmation message.
    """
    edge = {
        "src": source_id,
        "tgt": target_id,
        "rel": relationship,
        "weight": min(max(weight, 0.0), 1.0),
        "ts": time.time(),
    }
    _kg_edges.append(edge)
    _kg_persist()
    return f"Edge added: {source_id} -[{relationship}]-> {target_id}"


@tool
def query_graph(
    entity_id: str = "",
    entity_type: str = "",
    relationship: str = "",
) -> str:
    """Query the knowledge graph for entities and their connections.

    Returns matching entities with their edges. Use this to find
    corroborating sources, identify knowledge gaps, or map the
    evidence structure before writing a synthesis.

    Args:
        entity_id: If set, return this entity and all its connections.
        entity_type: If set, return all entities of this type.
        relationship: If set, return all edges of this type.

    Returns:
        JSON object with matching entities and edges.
    """
    result_entities: dict[str, dict] = {}
    result_edges: list[dict] = []

    if entity_id:
        if entity_id in _kg_entities:
            result_entities[entity_id] = _kg_entities[entity_id]
        for edge in _kg_edges:
            if edge["src"] == entity_id or edge["tgt"] == entity_id:
                result_edges.append(edge)
                other = edge["tgt"] if edge["src"] == entity_id else edge["src"]
                if other in _kg_entities:
                    result_entities[other] = _kg_entities[other]
    elif entity_type:
        for eid, ent in _kg_entities.items():
            if ent["type"] == entity_type:
                result_entities[eid] = ent
        matched_ids = set(result_entities.keys())
        for edge in _kg_edges:
            if edge["src"] in matched_ids or edge["tgt"] in matched_ids:
                result_edges.append(edge)
    elif relationship:
        for edge in _kg_edges:
            if edge["rel"] == relationship:
                result_edges.append(edge)
                for eid in (edge["src"], edge["tgt"]):
                    if eid in _kg_entities:
                        result_entities[eid] = _kg_entities[eid]
    else:
        result_entities = dict(_kg_entities)
        result_edges = list(_kg_edges)

    output = {
        "entity_count": len(result_entities),
        "edge_count": len(result_edges),
        "entities": {
            eid: {**ent, "props": ent.get("props", {})}
            for eid, ent in result_entities.items()
        },
        "edges": result_edges,
    }
    return json.dumps(output, ensure_ascii=False)


@tool
def find_gaps() -> str:
    """Identify entities with few or no connections (knowledge gaps).

    Returns entities sorted by connection count (ascending) so you can
    prioritise further research on poorly-connected nodes.

    Returns:
        JSON array of {entity_id, name, type, connection_count} sorted
        by connection_count ascending.
    """
    conn_count: dict[str, int] = {eid: 0 for eid in _kg_entities}
    for edge in _kg_edges:
        if edge["src"] in conn_count:
            conn_count[edge["src"]] += 1
        if edge["tgt"] in conn_count:
            conn_count[edge["tgt"]] += 1

    gaps = []
    for eid, count in sorted(conn_count.items(), key=lambda x: x[1]):
        ent = _kg_entities[eid]
        gaps.append({
            "entity_id": eid,
            "name": ent["name"],
            "type": ent["type"],
            "connection_count": count,
        })

    return json.dumps(gaps, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════
# MCP server tools (Brave, Firecrawl, Exa, Kagi + new servers)
# ═══════════════════════════════════════════════════════════════════════

# Increase startup_timeout for slow npx/uvx downloads on staging VMs.
_MCP_STARTUP_TIMEOUT = int(os.environ.get("MCP_STARTUP_TIMEOUT", "120"))

# ── Brave Search MCP ─────────────────────────────────────────────────
# npm: @brave/brave-search-mcp-server  (MIT, brave/brave-search-mcp-server)
# Tools: brave_web_search, brave_local_search, brave_image_search,
#   brave_video_search, brave_news_search, brave_summarizer
brave_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "@brave/brave-search-mcp-server"],
            env=_full_env(BRAVE_API_KEY=os.environ.get("BRAVE_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Firecrawl MCP ────────────────────────────────────────────────────
# npm: firecrawl-mcp  (MIT, firecrawl/firecrawl-mcp-server)
# Tools: firecrawl_scrape, firecrawl_crawl, firecrawl_map,
#   firecrawl_search, firecrawl_extract
firecrawl_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "firecrawl-mcp"],
            env=_full_env(FIRECRAWL_API_KEY=os.environ.get("FIRECRAWL_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Exa MCP ──────────────────────────────────────────────────────────
# npm: exa-mcp-server  (MIT, exa-labs/exa-mcp-server)
# Tools: web_search_exa, web_search_advanced_exa, crawling_exa,
#   get_code_context_exa
# Requires: npm install -g exa-mcp-server
exa_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="node",
            args=[
                "-e",
                # Bootstrap Smithery entry-point with config that enables
                # ALL non-deprecated Exa tools.  Smithery reads config from
                # process.argv.slice(2) as key=value pairs.
                "process.argv[2]='enabledTools=web_search_exa,web_search_advanced_exa,crawling_exa,get_code_context_exa';"
                "const r=require('child_process').execSync('npm root -g',{encoding:'utf8'}).trim();"
                "require(r+'/exa-mcp-server/.smithery/stdio/index.cjs');",
            ],
            env=_full_env(EXA_API_KEY=os.environ.get("EXA_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Kagi MCP ─────────────────────────────────────────────────────────
# uvx: kagimcp  (MIT, kagisearch/kagimcp)
# Tools: kagi_search, kagi_summarize, kagi_fastgpt, kagi_enrich_web,
#   kagi_enrich_news
kagi_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="uvx",
            args=["kagimcp"],
            env=_full_env(KAGI_API_KEY=os.environ.get("KAGI_API_KEY", "")),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── TranscriptAPI MCP (remote Streamable HTTP) ──────────────────────
# https://transcriptapi.com  (YouTube transcripts, search, channels)
# Tools: get_youtube_transcript, search_youtube,
#   get_channel_latest_videos, search_channel_videos,
#   list_channel_videos, list_playlist_videos
transcriptapi_mcp = MCPClient(
    lambda: _streamablehttp_transport(
        "https://transcriptapi.com/mcp",
        {"Authorization": f"Bearer {os.environ.get('TRANSCRIPTAPI_KEY', '')}"},
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Semantic Scholar MCP ─────────────────────────────────────────────
# npm: @xbghc/semanticscholar-mcp  (MIT, xbghc/semanticscholar-mcp)
# 200M+ academic papers — paper search, citation graphs, author profiles.
# Tools: search_papers, get_paper, get_paper_citations,
#   get_paper_references, batch_get_papers, search_authors, get_author,
#   get_author_papers, get_recommendations
# API key optional (higher rate limits with key).
semantic_scholar_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "@xbghc/semanticscholar-mcp"],
            env=_full_env(
                SEMANTIC_SCHOLAR_API_KEY=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
            ),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
    prefix="ss",
)

# ── arXiv MCP ────────────────────────────────────────────────────────
# npm: arxiv-mcp-server  (madi/arxiv-mcp-server)
# Free, no API key needed.  Searches arXiv preprints.
# Tools: search_papers, get_paper, search_by_category
arxiv_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "arxiv-mcp-server"],
            env=_full_env(),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
    prefix="arxiv",
)

# ── Wikipedia MCP ────────────────────────────────────────────────────
# npm: wikipedia-mcp  (MIT, timjuenemann/wikipedia-mcp)
# Free, no API key needed.
# Tools: search (Wikipedia search), read (full article content)
wikipedia_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "wikipedia-mcp"],
            env=_full_env(),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
    prefix="wiki",
)

# ── DuckDB MCP ───────────────────────────────────────────────────────
# npm: @seed-ship/duckdb-mcp-native  (MIT, theseedship/duckdb_mcp_node)
# 32+ tools: SQL queries, schema inspection, CSV/Parquet loading,
#   graph algorithms (PageRank, community detection), process mining.
# Tools: query, describe_table, list_tables, load_csv, load_parquet, etc.
duckdb_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "@seed-ship/duckdb-mcp-native"],
            env=_full_env(
                MCP_SECURITY_MODE="development",
                DUCKDB_MEMORY="2GB",
                DUCKDB_THREADS="4",
            ),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
)

# ── Bright Data MCP ──────────────────────────────────────────────────
# npm: @brightdata/mcp  (MIT, brightdata/brightdata-mcp)  2.3K+ stars
# Anti-block web scraping — bypasses CAPTCHAs, geo-restrictions, rate
# limits.  5,000 free requests/month.
# Tools: search_engine, scrape_as_markdown, scrape_as_html, session_stats
brightdata_mcp = MCPClient(
    lambda: stdio_client(
        StdioServerParameters(
            command="npx",
            args=["-y", "@brightdata/mcp"],
            env=_full_env(
                API_TOKEN=os.environ.get("BRIGHT_DATA_API_KEY", ""),
                GROUPS="search,scraping",
            ),
        )
    ),
    startup_timeout=_MCP_STARTUP_TIMEOUT,
    prefix="bd",
)


def _streamablehttp_transport(url: str, headers: dict):
    """Create a StreamableHTTP transport for remote MCP servers."""
    from mcp.client.streamable_http import streamablehttp_client

    return streamablehttp_client(url=url, headers=headers)


# ── Registry mapping ─────────────────────────────────────────────────

# MCP servers gated on API keys — only loaded when key is configured
_MCP_REGISTRY = {
    "BRAVE_API_KEY": brave_mcp,
    "FIRECRAWL_API_KEY": firecrawl_mcp,
    "EXA_API_KEY": exa_mcp,
    "KAGI_API_KEY": kagi_mcp,
    "TRANSCRIPTAPI_KEY": transcriptapi_mcp,
    "BRIGHT_DATA_API_KEY": brightdata_mcp,
}

# MCP servers that are free / don't require API keys — always loaded
_FREE_MCP_SERVERS = [
    semantic_scholar_mcp,
    arxiv_mcp,
    wikipedia_mcp,
    duckdb_mcp,
]


def get_all_mcp_clients():
    """Return list of MCP clients whose API keys are configured, plus free servers."""
    clients = []
    for env_var, client in _MCP_REGISTRY.items():
        if os.environ.get(env_var):
            clients.append(client)
    clients.extend(_FREE_MCP_SERVERS)
    return clients


# ── Native tool tier lists ───────────────────────────────────────────

# Tier 1 uncensored tools — duckduckgo is always available (no key needed),
# mojeek requires MOJEEK_API_KEY so it is gated in get_native_tools().
NATIVE_TOOLS_TIER1 = [duckduckgo_search, mojeek_search]

# Tier 2 content extraction tools — always included
NATIVE_TOOLS_TIER2 = [jina_read_url]

# Tier 3 censored fallback — only if API key is set
NATIVE_TOOLS_TIER3 = [google_search]

# Deep research tools — gated on API keys
NATIVE_TOOLS_DEEP_RESEARCH = [
    perplexity_deep_research,
    grok_deep_research,
    tavily_deep_research,
    exa_multi_search,
]

# Research management tools — always available (no API key needed)
NATIVE_TOOLS_RESEARCH_MGMT = [
    store_finding,
    read_findings,
    add_entity,
    add_edge,
    query_graph,
    find_gaps,
]


def get_native_tools():
    """Return native @tool functions, ordered uncensored-first.

    Tools that require an API key are only included when their key is
    configured, so the LLM won't waste a tool call on a guaranteed error.
    """
    tools = [duckduckgo_search]  # always available (no key needed)
    if os.environ.get("MOJEEK_API_KEY"):
        tools.append(mojeek_search)
    tools.extend(NATIVE_TOOLS_TIER2)
    if os.environ.get("SERPER_API_KEY"):
        tools.extend(NATIVE_TOOLS_TIER3)

    # Deep research tools — gated on API keys
    if os.environ.get("PERPLEXITY_API_KEY"):
        tools.append(perplexity_deep_research)
    if os.environ.get("XAI_API_KEY"):
        tools.append(grok_deep_research)
    if os.environ.get("TAVILY_API_KEY"):
        tools.append(tavily_deep_research)
    if os.environ.get("EXA_API_KEY"):
        tools.append(exa_multi_search)

    # Research management — always available
    tools.extend(NATIVE_TOOLS_RESEARCH_MGMT)
    return tools
