# MiroThinker Research API Reference — Full Capabilities & Integration Strategy

*Compiled from official documentation, April 2026*

---

## Table of Contents
1. [Exa](#1-exa) — Neural/semantic web search with content extraction
2. [Firecrawl](#2-firecrawl) — Web scraping, crawling, structured extraction
3. [Kagi](#3-kagi) — Premium search + universal summarizer + enrichment
4. [Jina](#4-jina) — Reader (URL→markdown), search (SERP), embeddings, reranking
5. [Perplexity](#5-perplexity) — AI-powered search with citations (Sonar), Agent API, Search API
6. [Tavily](#6-tavily) — Search-optimized-for-agents with answer generation
7. [Mojeek](#7-mojeek) — Independent crawler-based search (no Google/Bing dependency)
8. [Marginalia](#8-marginalia) — Small/indie web search engine (non-commercial focus)
9. [Apify](#9-apify) — Actor marketplace for web scraping (Google Scholar, PubMed, etc.)
10. [Consensus](#10-consensus) — 200M+ peer-reviewed papers, study type filtering, MCP protocol
11. [Scite.ai](#11-sciteai) — Smart Citations, citation graph analysis, full-text scientific literature
12. [Integration Strategy for MiroThinker](#12-integration-strategy)

---

## 1. Exa

**What it is:** Embeddings-based web search engine purpose-built for AI. Uses neural search to find semantically relevant pages, not just keyword matches.

**Base URL:** `https://api.exa.ai`  
**Auth:** `x-api-key: YOUR_KEY` header  
**Env var:** `EXA_API_KEY`

### Endpoints

| Endpoint | Method | What it does |
|----------|--------|-------------|
| `/search` | POST | Core search — neural, fast, auto, instant, deep variants |
| `/contents` | POST | Extract contents from URLs (text, highlights, summary) |

### Search Types (the `type` parameter)

| Type | Description | Latency | Best for |
|------|-------------|---------|----------|
| `auto` | Intelligently combines neural + keyword | Medium | Default, general queries |
| `neural` | Pure embeddings-based semantic search | Medium | Conceptual/research queries ("how does lutein protect the retina") |
| `fast` | Streamlined search models | Low | Quick lookups |
| `instant` | Lowest latency, real-time optimized | Very low | Live data, autocomplete |
| `deep-lite` | Lightweight synthesized output | High | Summary answers with sources |
| `deep` | Light deep search with synthesis | High | Research questions needing synthesis |
| `deep-reasoning` | Base deep search with reasoning | Very high | Complex multi-step reasoning |

### Content Extraction Options

```json
{
  "contents": {
    "text": true,                          // Full page text (markdown)
    "highlights": { "maxCharacters": 4000 }, // Key passages with highlighting
    "summary": { "query": "Main findings" }, // LLM-generated summary focused on query
    "subpages": 1,                          // Follow links to subpages
    "subpageTarget": "sources",             // Target type for subpages
    "extras": { "links": 1, "imageLinks": 1 } // Extract links and images
  }
}
```

### Category Filter

```
category: "research paper" | "news" | "company" | "personal site" | "financial report" | "people"
```

**"research paper" is gold for MiroThinker** — restricts results to academic/scientific content.

### Key Filters

| Filter | Example | Purpose |
|--------|---------|---------|
| `includeDomains` | `["pubmed.ncbi.nlm.nih.gov", "scholar.google.com"]` | Restrict to specific sites |
| `excludeDomains` | `["pinterest.com"]` | Block low-quality domains |
| `startPublishedDate` | `"2020-01-01"` | Only recent research |
| `endPublishedDate` | `"2026-01-01"` | Date ceiling |
| `includeText` | `["randomized controlled trial"]` | Must contain these terms |
| `excludeText` | `["advertisement"]` | Must not contain these terms |
| `numResults` | 10 | Max results (default 10) |

### Deep Search with Output Schema

```json
{
  "query": "What is the effect of lutein on macular degeneration?",
  "type": "deep",
  "systemPrompt": "Prefer peer-reviewed sources. Include effect sizes and sample sizes.",
  "outputSchema": {
    "type": "object",
    "properties": {
      "findings": { "type": "array", "items": { "type": "string" } },
      "key_studies": { "type": "array", "items": { "type": "string" } },
      "consensus": { "type": "string" }
    }
  },
  "contents": { "text": true }
}
```

### Optimal MiroThinker Usage
- **Primary search tool** for the researcher agent
- Use `category: "research paper"` for academic queries
- Use `highlights` for extracting key passages without full-page text
- Use `deep` or `deep-reasoning` for complex clinical questions
- Chain with `summary` to get pre-synthesized answers
- Use date filters to prioritize recent research (last 5 years)

### Pricing
- Search: ~$0.001–0.004 per query depending on type
- Deep search: more expensive but includes synthesis
- Contents extraction: additional per URL

---

## 2. Firecrawl

**What it is:** Web scraping API that converts any URL to clean markdown, structured data, or screenshots. Handles JS rendering, proxies, rate limits automatically.

**Base URL:** `https://api.firecrawl.dev/v2`  
**Auth:** `Authorization: Bearer YOUR_KEY`  
**Env var:** `FIRECRAWL_API_KEY`

### Endpoints

| Endpoint | Method | Credits | What it does |
|----------|--------|---------|-------------|
| `/scrape` | POST | 1/page | Convert single URL → markdown/HTML/JSON/screenshot |
| `/crawl` | POST | 1/page | Recursively crawl entire site, returns all pages |
| `/extract` | POST | ~15 tokens/credit | LLM-powered structured data extraction from URLs |
| `/map` | POST | 1/call | Get all URLs from a website (sitemap discovery) |
| `/agent` | POST | varies | AI agent that finds and extracts data without URLs |

### Scrape Formats

| Format | Description |
|--------|-------------|
| `markdown` | Clean markdown — **ideal for LLM input** |
| `html` | Cleaned HTML |
| `rawHtml` | Unmodified original HTML |
| `summary` | AI-generated summary |
| `links` | All links on the page |
| `images` | All image URLs |
| `json` | Structured extraction (requires schema or prompt) |
| `screenshot` | Page screenshot (full page option) |
| `branding` | Brand identity extraction (colors, fonts, etc.) |
| `audio` | Extract MP3 from video URLs (YouTube) |

### Structured Extraction (JSON mode)

```python
from firecrawl import Firecrawl
from pydantic import BaseModel

class StudyData(BaseModel):
    title: str
    authors: list[str]
    sample_size: int
    effect_size: str
    conclusion: str

result = app.scrape('https://pubmed.ncbi.nlm.nih.gov/12345678/', 
    formats=[{"type": "json", "schema": StudyData.model_json_schema()}])
```

### Crawl with Filters

```python
docs = firecrawl.crawl(
    url="https://examine.com/supplements/lutein/",
    limit=50,                    # Max pages
    scrape_options={
        'formats': ['markdown'],
        'only_main_content': True  # Strip nav/footer/sidebar
    },
    include_paths=["*/research/*"],  # Only research pages
    exclude_paths=["*/shop/*"]       # Skip e-commerce pages
)
```

### Site Map Discovery

```python
# Get ALL URLs from a site — only 1 credit regardless of count
urls = firecrawl.map("https://examine.com", search="lutein zeaxanthin")
# Returns ordered list, most relevant first
```

### Agent Mode (NEW)

```python
result = firecrawl.agent(
    prompt="Find the AREDS2 trial results for lutein supplementation",
    # No URLs needed — the agent finds them
)
```

### Optimal MiroThinker Usage
- **Primary content extraction tool** — when researcher has a URL, use Firecrawl to get clean markdown
- Use `only_main_content: True` to strip boilerplate
- Use JSON extraction mode for structured data from clinical trial pages
- Use `map` to discover all pages on a research site before selective scraping
- Use `crawl` for comprehensive extraction from review sites (Examine, WebMD)
- **Cost-effective**: 1 credit per page for markdown extraction

### Pricing
- Scrape: 1 credit/page (JSON mode: +4 credits)
- Crawl: 1 credit/page crawled
- Map: 1 credit/call (regardless of URLs returned)
- Extract: credits based on tokens processed

---

## 3. Kagi

**What it is:** Premium search engine with no ads, no tracking. Includes Universal Summarizer that can summarize any URL, PDF, audio, or video.

**Base URL:** `https://kagi.com/api/v0`  
**Auth:** `Authorization: Bot YOUR_TOKEN`  
**Env var:** `KAGI_API_KEY`

### Endpoints

| Endpoint | Method | Price | What it does |
|----------|--------|-------|-------------|
| `/search` | GET | $0.025/query | Full web search (closed beta) |
| `/summarize` | GET/POST | $0.03/1K tokens or $1/summary (Muriel) | Summarize any content |
| `/enrich/web` | GET | $0.002/query | Non-commercial web content (Teclis index) |
| `/enrich/news` | GET | $0.002/query | News from non-mainstream sources (TinyGem index) |

### Search API

```bash
curl -H "Authorization: Bot $KAGI_API_KEY" \
  "https://kagi.com/api/v0/search?q=lutein+zeaxanthin+AREDS2"
```

Returns: URL, title, snippet, thumbnail, published date. Up to 20 results.

### Universal Summarizer — The Killer Feature

Can summarize **anything**:
- Web pages, articles, forum threads
- PDF documents
- PowerPoint, Word docs
- Audio files (mp3/wav)
- YouTube videos
- Scanned PDFs and images (OCR)
- **Unlimited token length**

```bash
# Summarize a research paper PDF
curl -H "Authorization: Bot $TOKEN" \
  "https://kagi.com/api/v0/summarize?url=https://pubmed.ncbi.nlm.nih.gov/12345678/&engine=muriel&summary_type=takeaway"
```

### Summarizer Engines

| Engine | Style | Cost | Best for |
|--------|-------|------|----------|
| `cecil` | Friendly, descriptive, fast | $0.03/1K tokens | Quick summaries |
| `agnes` | Formal, technical, analytical | $0.03/1K tokens | Research papers |
| `muriel` | Best-in-class enterprise grade | $1/summary flat | Critical analysis, long documents |

### Summary Types
- `summary` — Paragraph(s) of prose
- `takeaway` — Bulleted key points

### Enrichment APIs — Unique to Kagi

These find **non-commercial, indie web** content that Google/Bing miss:

```bash
# Teclis index — small web, personal sites, niche blogs
curl -H "Authorization: Bot $TOKEN" "https://kagi.com/api/v0/enrich/web?q=lutein+macular+pigment"

# TinyGem index — interesting news from non-mainstream sources
curl -H "Authorization: Bot $TOKEN" "https://kagi.com/api/v0/enrich/news?q=carotenoid+research"
```

### Optimal MiroThinker Usage
- **Summarizer** for long research papers — feed PDF URL, get structured takeaways
- **Enrichment APIs** for finding niche/indie content that Exa and Tavily miss
- Use `agnes` engine for formal technical summaries
- Use `muriel` for critical papers where quality matters most ($1/paper is worth it)
- Combine with Firecrawl: Kagi finds the URL → Firecrawl extracts full content
- Enrichment is **uniquely valuable** — surfaces content from personal research blogs, niche forums

### Pricing
- Search: $0.025/query (closed beta)
- Summarizer: $0.03/1K tokens (consumer) or $1/summary (Muriel)
- Enrichment: $0.002/query (only billed when results returned)

---

## 4. Jina

**What it is:** Suite of AI tools — Reader (URL→markdown), Search (web SERP), Embeddings, Reranking. Reader is the standout feature.

**Base URL (Reader):** `https://r.jina.ai/`  
**Base URL (Search):** `https://s.jina.ai/`  
**Base URL (Embeddings):** `https://api.jina.ai/v1/embeddings`  
**Auth:** `Authorization: Bearer YOUR_KEY`  
**Env var:** `JINA_API_KEY`

### Reader API — `r.jina.ai`

Simply prepend `https://r.jina.ai/` to any URL:

```bash
curl "https://r.jina.ai/https://pubmed.ncbi.nlm.nih.gov/12345678/" \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Accept: application/json"
```

Returns: clean markdown with metadata (title, description, content, images with captions).

### Reader Advanced Options

| Header/Param | Description |
|-------------|-------------|
| `X-Engine: browser` | Use headless browser for JS-heavy pages |
| `X-With-Links-Summary: true` | Append "Buttons & Links" section |
| `X-With-Images-Summary: true` | Append "Images" section with captions |
| `X-Target-Selector: article` | Only extract content matching CSS selector |
| `X-Wait-For-Selector: .content` | Wait for dynamic content to load |
| `X-Remove-Selector: nav,footer` | Remove elements before extraction |
| `X-No-Cache: true` | Bypass cache for fresh content |
| `X-With-Iframe: true` | Include iframe content |
| `X-Timeout: 30` | Custom timeout in seconds |
| `X-Proxy-Url: your_proxy` | Use custom proxy |

### Search API — `s.jina.ai`

Web search returning SERP results as LLM-friendly text:

```bash
curl "https://s.jina.ai/?q=lutein+zeaxanthin+clinical+trials" \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Accept: application/json"
```

Returns top 5 results with URL, title, and full content (already extracted).

### Embeddings API

```python
import requests
response = requests.post("https://api.jina.ai/v1/embeddings", 
    headers={"Authorization": "Bearer $JINA_API_KEY"},
    json={
        "model": "jina-embeddings-v5-text-small",  # 677M params, 1024-dim
        "input": ["lutein protects the macula from blue light damage"],
        "task": "retrieval.passage"
    })
```

Models:
- `jina-embeddings-v5-text-small` — 677M params, 32K context, 1024-dim (best quality)
- `jina-embeddings-v5-text-nano` — 239M params, 8K context, 768-dim (low latency)
- `jina-embeddings-v4` — 3.8B params, multimodal (text + images + PDFs)
- `jina-clip-v2` — 885M params, cross-modal text-image retrieval

### Reranker API

```python
response = requests.post("https://api.jina.ai/v1/rerank",
    headers={"Authorization": "Bearer $JINA_API_KEY"},
    json={
        "model": "jina-reranker-v3",
        "query": "lutein macular degeneration prevention",
        "documents": ["doc1...", "doc2...", "doc3..."]
    })
```

### Optimal MiroThinker Usage
- **Reader** as alternative to Firecrawl — simpler API, one-line URL conversion
- **Search** for quick web lookups when Exa returns no results
- **Embeddings** for corpus deduplication — embed findings and cluster similar ones
- **Reranker** for post-processing — rerank Exa/Tavily results by relevance to the thinker's specific question
- Reader's image captioning is unique — useful for extracting data from figures/charts in papers

### Pricing
- Reader: token-based (~$0.02/1M tokens)
- Search: token-based
- Embeddings: model-dependent
- Reranker: per query

---

## 5. Perplexity

**What it is:** AI-powered search that generates answers with citations. Three APIs: Sonar (search+answer), Agent (multi-provider), Search (raw results).

**Base URL:** `https://api.perplexity.ai`  
**Auth:** `Authorization: Bearer YOUR_KEY`  
**Env var:** `PERPLEXITY_API_KEY`

### Three APIs

| API | Endpoint | Best for |
|-----|----------|----------|
| **Sonar** | `/chat/completions` | Direct search+answer with citations (simplest) |
| **Agent** | `/v1/agent` | Complex workflows, tool use, structured output, multi-provider models |
| **Search** | `/search` | Raw ranked web results with advanced filtering |

### Sonar API — Search + Answer

```python
import requests
response = requests.post("https://api.perplexity.ai/chat/completions",
    headers={"Authorization": "Bearer $PERPLEXITY_API_KEY"},
    json={
        "model": "sonar",           # or sonar-pro, sonar-deep-research
        "messages": [{"role": "user", "content": "What does AREDS2 say about lutein vs beta-carotene for AMD?"}],
        "search_recency_filter": "year",  # day, week, month, year
        "return_citations": True,
        "return_related_questions": True
    })
# Response includes: answer text + citations array with URLs
```

### Available Models

| Model | Description | Context | Best for |
|-------|-------------|---------|----------|
| `sonar` | Fast search-powered answers | 128K | Quick factual queries |
| `sonar-pro` | Higher quality, more sources | 200K | Research questions |
| `sonar-deep-research` | Multi-step research with reasoning | 128K | Complex investigation |
| `sonar-reasoning` | Chain-of-thought with search | 128K | Analytical questions |
| `sonar-reasoning-pro` | Extended reasoning + search | 128K | Deep analysis |

### Agent API — Multi-Provider with Tools

```python
response = requests.post("https://api.perplexity.ai/v1/agent",
    headers={"Authorization": "Bearer $PERPLEXITY_API_KEY"},
    json={
        "model": "claude-3.5-sonnet",  # Any supported third-party model
        "messages": [{"role": "user", "content": "..."}],
        "tools": [{"type": "web_search"}],  # Enable web search tool
        "reasoning": {"effort": "high"},
        "output": {"format": "json_schema", "schema": {...}}
    })
```

### Search API — Raw Results

```python
response = requests.post("https://api.perplexity.ai/search",
    headers={"Authorization": "Bearer $PERPLEXITY_API_KEY"},
    json={
        "query": "lutein zeaxanthin clinical trials 2024",
        "search_recency_filter": "year",
        "return_citations": True
    })
```

### Optimal MiroThinker Usage
- **`sonar-deep-research`** for the deep research tool — multi-step investigation with automatic source finding
- **`sonar-pro`** for detailed single-question answers with citations
- Use `return_citations: True` always — gives URLs that can then be scraped via Firecrawl/Jina
- Use `search_recency_filter` to focus on recent research
- **Unique value**: Perplexity synthesizes across multiple sources in a single call — perfect for the "deep research" budget-gated tool in Phase 5
- Agent API allows using Claude/GPT models with Perplexity's search backend

### Pricing
- Sonar: ~$1/1000 queries
- Sonar Pro: ~$5/1000 queries  
- Deep Research: ~$5/query
- Search API: varies

---

## 6. Tavily

**What it is:** Search engine specifically designed for AI agents. Returns clean, relevant results with optional LLM-generated answers.

**Base URL:** `https://api.tavily.com`  
**Auth:** API key in request body  
**Env var:** `TAVILY_API_KEY`

### Endpoints

| Endpoint | Method | Credits | What it does |
|----------|--------|---------|-------------|
| `/search` | POST | 1-2 | Web search optimized for agents |
| `/extract` | POST | varies | Extract content from URLs |
| `/crawl` | POST | varies | Crawl websites |
| `/map` | POST | varies | Discover URLs on a site |

### Search Depths

| Depth | Latency | Credits | Returns |
|-------|---------|---------|---------|
| `ultra-fast` | Lowest | 1 | NLP summary per URL |
| `fast` | Low | 1 | Multiple semantic chunks per URL |
| `basic` | Medium | 1 | NLP summary per URL (balanced) |
| `advanced` | Highest | 2 | Multiple semantic chunks, highest relevance |

### Full Search Example

```python
import requests
response = requests.post("https://api.tavily.com/search", json={
    "api_key": TAVILY_API_KEY,
    "query": "lutein zeaxanthin macular pigment optical density clinical trial",
    "search_depth": "advanced",
    "topic": "general",           # or "news", "finance"
    "time_range": "year",         # day, week, month, year
    "max_results": 10,
    "include_answer": "advanced",  # LLM-generated answer (basic or advanced)
    "include_raw_content": "markdown",  # Full page content in markdown
    "include_domains": ["pubmed.ncbi.nlm.nih.gov", "examine.com"],
    "exclude_domains": ["pinterest.com", "amazon.com"],
    "chunks_per_source": 3        # For advanced depth: max semantic chunks per source
})
```

### Response Structure

```json
{
  "query": "...",
  "answer": "Based on clinical trials, lutein supplementation at 10mg/day...",  
  "results": [
    {
      "title": "AREDS2 Results...",
      "url": "https://...",
      "content": "Relevant excerpt...",  
      "raw_content": "Full markdown...",  // if include_raw_content=true
      "score": 0.95,                     // relevance score
      "published_date": "2024-06-15"
    }
  ],
  "images": [...]  // if include_images=true
}
```

### Extract Endpoint

```python
response = requests.post("https://api.tavily.com/extract", json={
    "api_key": TAVILY_API_KEY,
    "urls": ["https://pubmed.ncbi.nlm.nih.gov/12345678/"]
})
# Returns clean extracted content from each URL
```

### Optimal MiroThinker Usage
- **Best default search for the researcher** — designed specifically for AI agent workflows
- Use `search_depth: "advanced"` with `chunks_per_source: 3` for research queries
- Use `include_answer: "advanced"` to get a pre-synthesized answer alongside raw results
- Use `include_raw_content: "markdown"` to avoid a separate Firecrawl call
- Use `include_domains` to restrict to academic sources
- Use `time_range: "year"` for recent research
- **Unique advantage**: Returns relevance `score` per result — use for quality gating in the corpus

### Pricing
- Free tier: 1,000 searches/month
- Basic/advanced: 1-2 credits per search

---

## 7. Mojeek

**What it is:** Independent search engine with its own crawler — does NOT rely on Google or Bing indexes. Unique results you won't find elsewhere.

**Base URL:** `https://api.mojeek.com`  
**Auth:** `api_key` query parameter  
**Env var:** `MOJEEK_API_KEY`

### Search Endpoint

```bash
curl "https://api.mojeek.com/search?q=lutein+zeaxanthin&api_key=$MOJEEK_API_KEY&fmt=json&t=20&lb=EN&lbb=100"
```

### Key Parameters

| Param | Description | Values |
|-------|-------------|--------|
| `q` | Search query | URL-encoded string |
| `t` | Results per page | 1-20 (default 10) |
| `fmt` | Response format | `json` or `xml` |
| `lb` / `lbb` | Language boost | ISO 639-1 code + boost 1-100 |
| `rb` / `rbb` | Region boost | ISO 3166-1 code + boost 1-100 |
| `since` | Date filter | `day`, `month`, `year`, or `YYYYMMDD` |
| `before` | Date ceiling | Same formats |
| `site` | Restrict to domain | e.g., `pubmed.ncbi.nlm.nih.gov` |
| `fi` | Include domains | Comma-separated (max 25) |
| `fe` | Exclude domains | Comma-separated (max 25) |
| `safe` | Safe search | `0` or `1` |
| `datewr` | Sort by date | `100` for date-sorted |
| `date` | Include last-modified date | `0` or `1` |

### Response Structure

```json
{
  "response": {
    "status": "OK",
    "head": {
      "query": "lutein zeaxanthin",
      "results": 112094,
      "timer": 0.61
    },
    "results": [
      {
        "url": "https://...",
        "title": "...",
        "desc": "...",
        "score": 2.84,
        "image": {"url": "..."}
      }
    ]
  }
}
```

### Why Mojeek Matters for Research

Mojeek crawls the web independently. This means:
1. **Unique results** — pages that Google deprioritizes or Bing misses entirely
2. **No filter bubble** — no personalization, no commercial bias
3. **Privacy-first** — no tracking, no profiling
4. Results may include **obscure research blogs, institutional pages, foreign-language sources** that mainstream engines suppress

### Optimal MiroThinker Usage
- **Diversity tool** — use when Exa/Tavily return similar results, to find alternative sources
- Excellent for finding **niche academic pages** that mainstream search overlooks
- Use `site` parameter to restrict to specific institutional domains
- Use `since` for date filtering
- Combine with Firecrawl/Jina Reader for content extraction (Mojeek only returns snippets)
- **Low concurrency recommended** — rate limits are stricter than Exa/Tavily

### Pricing
- Contact-based for API access

---

## 8. Marginalia

**What it is:** Independent DIY search engine focused on **non-commercial content** and the "small web." Finds personal sites, independent blogs, niche resources that commercial search engines bury.

**Base URL:** `https://api2.marginalia-search.com`  
**Auth:** `API-Key: YOUR_KEY` header  
**Env var:** `MARGINALIA_API_KEY`

### Search Endpoint

```bash
curl -H "API-Key: $MARGINALIA_API_KEY" \
  "https://api2.marginalia-search.com/search?query=lutein+zeaxanthin+macular+pigment&count=20"
```

### Parameters

| Param | Value | Description |
|-------|-------|-------------|
| `query` | string | Search query |
| `count` | 1-100 | Number of results |
| `timeout` | 50-250 | Query timeout in ms |
| `dc` | 1-100 | Max results per domain |
| `page` | int | Pagination (1-indexed) |
| `nsfw` | 0/1 | Content filter |
| `filter` | string | Custom filter name |

### Custom Filters (Unique Feature)

You can create persistent custom filters via the API:

```xml
<filter>
    <domains-include>
        *.edu
        *.ac.uk
    </domains-include>
    <domains-exclude>
        *.pinterest.com
    </domains-exclude>
    <temporal-bias>RECENT</temporal-bias>
    <terms-require>
        clinical trial
    </terms-require>
</filter>
```

### Response

```json
{
  "license": "UNRESTRICTED",
  "query": "lutein zeaxanthin",
  "results": [
    {
      "url": "https://...",
      "title": "...",
      "description": "...",
      "quality": 4.54,    // Marginalia quality score
      "format": "html",
      "resultsFromDomain": 1
    }
  ]
}
```

### Why Marginalia Matters for Research

1. **Surfaces the small web** — independent researchers, personal lab pages, niche health blogs
2. **Quality score** — ranks by content quality, not SEO or backlinks
3. **Anti-commercial bias** — explicitly deprioritizes commercial content
4. **Historical content** — finds older pages that Google has deprioritized
5. **Unique index** — completely independent crawl, no overlap with Google/Bing

### Optimal MiroThinker Usage
- **"Hidden gems" tool** — use specifically to find content that other engines miss
- Perfect for finding **independent researcher blogs**, university course pages, niche health forums
- Use custom filters to create a persistent "academic" filter (`*.edu`, `*.ac.uk`, etc.)
- Lower priority than Exa/Tavily for mainstream research, but **irreplaceable for diversity**
- Quality score can be used as an additional signal for corpus filtering
- **Low concurrency** — rate limited, use sparingly

### Pricing
- Free non-commercial key (email request)
- Paid keys available for commercial use

---

## 9. Apify

**What it is:** Cloud platform and marketplace for web scraping/automation "Actors." Thousands of pre-built scrapers for specific sites — Google Scholar, PubMed, Wikipedia, etc.

**Base URL:** `https://api.apify.com/v2`  
**Auth:** `Authorization: Bearer YOUR_TOKEN` or `?token=YOUR_TOKEN`  
**Env var:** `APIFY_API_KEY`

### How It Works

1. Find an Actor (pre-built scraper) in the Apify Store
2. Run it via API with input parameters
3. Poll for completion
4. Fetch results from the Actor's dataset

### Research-Relevant Actors

| Actor | What it scrapes | Pricing |
|-------|----------------|---------|
| `george.the.developer/google-scholar-scraper` | Google Scholar — papers, citations, author profiles, PDF links, h-index | $4/1000 papers |
| `labrat011/pubmed-scraper` | PubMed — 35M+ medical citations, abstracts, MeSH terms | $0.80/1000 results |
| `easyapi/pubmed-search-scraper` | PubMed articles with comprehensive metadata | $19.99/month + usage |
| Wikipedia scrapers | Wikipedia articles and structured data | Various |
| General web scrapers | Any website with anti-bot protection | Various |

### Running an Actor

```python
from apify_client import ApifyClient
client = ApifyClient(APIFY_API_KEY)

# Run Google Scholar scraper
run_input = {
    "queries": ["lutein zeaxanthin macular degeneration"],
    "maxResults": 50,
    "proxyConfiguration": {"useApifyProxy": True}
}

run = client.actor("george.the.developer/google-scholar-scraper").call(run_input=run_input)

# Fetch results
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(f"Title: {item['title']}")
    print(f"Authors: {item['authors']}")
    print(f"Citations: {item['citationCount']}")
    print(f"PDF: {item.get('pdfLink', 'N/A')}")
```

### PubMed Scraper

```python
run_input = {
    "searchTerms": ["lutein AND zeaxanthin AND macular degeneration"],
    "maxResults": 100
}

run = client.actor("labrat011/pubmed-scraper").call(run_input=run_input)

for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(f"PMID: {item['pmid']}")
    print(f"Title: {item['title']}")
    print(f"Abstract: {item['abstract']}")
    print(f"MeSH: {item['meshTerms']}")
```

### MCP Server

Apify also offers an MCP server for direct integration with LLM agents — can be added as a tool server.

### Optimal MiroThinker Usage
- **Academic paper discovery** — Google Scholar scraper finds papers with citation counts, PDF links, h-index
- **PubMed mining** — Extract medical literature with structured metadata (MeSH terms, abstracts)
- Use for **batch extraction** — when the thinker identifies 50+ papers to analyze
- **Complementary to Exa** — Exa finds relevant pages, Apify extracts structured data from specific sources
- **Anti-bot handling** — Apify's proxies and browser automation handle sites that block direct API access
- Run asynchronously — start an Actor, continue with other research, fetch results later

### Pricing
- Free tier: $5/month platform credits
- Actors: pay-per-event (Google Scholar $4/1000, PubMed $0.80/1000)
- Platform: compute time-based

---

## 10. Consensus

**What it is:** AI-powered academic search engine indexing 200M+ peer-reviewed papers. The only API that provides **study type classification** (RCT, meta-analysis, systematic review, etc.) and **key takeaways** per paper. Consensus focuses on answering research questions with evidence from scientific literature.

**MCP URL:** `https://mcp.consensus.app/mcp` (note: `/mcp` path is required)  
**Auth:** OAuth 2.1 with PKCE (browser-based login via Clerk)  
**Protocol:** MCP (Model Context Protocol) over Streamable HTTP / JSON-RPC  
**Env vars:** `CONSENSUS_REFRESH_TOKEN`, `CONSENSUS_CLIENT_ID`

### Authentication — OAuth 2.1 Flow

Consensus uses OAuth 2.1 with PKCE — same pattern as scite.ai. The flow:

1. **Discovery:** `GET https://consensus.app/.well-known/oauth-authorization-server`
   Returns authorization, token, and registration endpoints.
2. **Dynamic Client Registration (DCR):**
   ```bash
   curl -X POST "https://consensus.app/oauth/register/" \
     -H "Content-Type: application/json" \
     -d '{"client_name": "MiroThinker", "redirect_uris": ["http://localhost:8765/oauth/callback"], "grant_types": ["authorization_code", "refresh_token"], "token_endpoint_auth_method": "none", "scope": "mcp"}'
   ```
   **Note:** Trailing slash on `/oauth/register/` is required.
   Returns: `client_id` (no secret — public client with PKCE)
3. **Authorization:** Open browser to:
   ```
   https://consensus.app/oauth/authorize/?response_type=code&client_id=CLIENT_ID&code_challenge=CHALLENGE&code_challenge_method=S256&redirect_uri=http://localhost:8765/oauth/callback
   ```
   User logs in via Clerk (Google SSO or email). On success, redirects with `?code=AUTH_CODE`.
4. **Token Exchange:**
   ```bash
   curl -X POST "https://consensus.app/oauth/token" \
     -d "grant_type=authorization_code&code=AUTH_CODE&redirect_uri=http://localhost:8765/oauth/callback&client_id=CLIENT_ID&code_verifier=VERIFIER"
   ```
   Returns: `access_token` (short-lived), `refresh_token` (long-lived)
5. **Token Refresh:** Use refresh_token to get new access_token without re-login

**Known Issue (April 2026):** Google SSO via Clerk may loop back to login screen in automated environments. Manual browser login is recommended for initial token acquisition.

### MCP Protocol

Consensus uses the Model Context Protocol (JSON-RPC over HTTP):

```python
import requests

# Initialize MCP session
response = requests.post("https://mcp.consensus.app/mcp",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
    json={"jsonrpc": "2.0", "method": "initialize",
          "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                     "clientInfo": {"name": "MiroThinker", "version": "1.0.0"}}, "id": 1})

# List available tools
response = requests.post("https://mcp.consensus.app/mcp",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
    json={"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 2})

# Call search tool
response = requests.post("https://mcp.consensus.app/mcp",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
    json={"jsonrpc": "2.0", "method": "tools/call",
          "params": {"name": "search",
                     "arguments": {"query": "omega-3 fatty acids diabetes prevention"}}, "id": 3})
```

### The `search` Tool — Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural language research question (best results with specific, well-formed questions) |
| `year_min` | int | Filter papers published from this year onwards |
| `year_max` | int | Filter papers published up to this year |
| `study_types` | array[string] | Filter by study design: `rct`, `meta-analysis`, `systematic-review`, `case-report`, `observational`, `cohort`, `cross-sectional`, `literature-review`, `animal-study`, `in-vitro` |
| `sjr_max` | int | Journal quality quartile (1 = top 25% journals by SCImago Journal Rank) |
| `human` | bool | Restrict to human-subjects studies only |
| `sample_size_min` | int | Minimum sample size (filters out underpowered studies) |

### Response Structure

```json
{
  "papers": [
    {
      "title": "Omega-3 Fatty Acid Supplementation and Risk of Type 2 Diabetes",
      "authors": ["Smith J", "Jones A"],
      "abstract": "Full abstract text...",
      "journal": "The Lancet Diabetes & Endocrinology",
      "year": 2023,
      "citation_count": 142,
      "url": "https://consensus.app/papers/...",
      "study_type": "meta-analysis",
      "takeaway": "Meta-analysis of 12 RCTs (n=45,000) found omega-3 supplementation associated with 15% reduced risk of T2D (RR 0.85, 95% CI 0.78-0.93)"
    }
  ]
}
```

**Note:** `study_type` and `takeaway` fields require a Pro plan. Enterprise plan adds `doi` field. Free plan returns basic paper metadata with up to 3 papers per search.

### Access Tiers

| Tier | Papers/Search | Searches/Month | Study Type | Takeaway | DOI |
|------|--------------|----------------|------------|----------|-----|
| No account | 3 | Unlimited | No | No | No |
| Free | 10 | 30 | No | No | No |
| Pro ($8.99/mo) | 20 | 1,000 | Yes | Yes | No |
| Enterprise | 20+ | Custom | Yes | Yes | Yes |

### Why Consensus is Valuable for MiroThinker

1. **Study type filtering** — The killer feature. No other API lets you filter by RCT, meta-analysis, systematic review, etc. Essential for evidence-based research where study design determines evidence quality.
2. **Sample size filtering** — Filter out underpowered studies (e.g., `sample_size_min: 100`).
3. **Journal quality filtering** — SCImago Journal Rank quartile filtering ensures high-quality sources.
4. **AI-generated takeaways** — Pre-synthesized key findings per paper, saving LLM tokens.
5. **Human-subjects filter** — Critical for health/nutrition research (excludes animal studies when needed).
6. **200M+ papers** — Comprehensive coverage across all scientific disciplines.

### Optimal MiroThinker Usage
- **Evidence hierarchy queries** — Search with `study_types: ["meta-analysis", "systematic-review"]` first, then broaden to RCTs
- **Clinical research** — Use `human: true` + `sample_size_min: 50` for robust human evidence
- **Quality filtering** — Use `sjr_max: 1` or `sjr_max: 2` to restrict to top-tier journals
- **Combine with scite.ai** — Consensus finds the papers by study type, scite.ai shows how they cite each other
- **Combine with Exa/Tavily** — Consensus for structured metadata, Exa/Tavily for full-text content extraction
- **Date-bounded searches** — Use `year_min`/`year_max` to focus on recent evidence

### Token Management
- Access tokens are short-lived
- Use refresh_token to get new access_token without re-login:
  ```bash
  curl -X POST "https://consensus.app/oauth/token" \
    -d "grant_type=refresh_token&refresh_token=REFRESH_TOKEN&client_id=CLIENT_ID"
  ```
- Store refresh_token securely — it's long-lived
- MiroThinker should auto-refresh before each session

### Pricing
- Free tier: 30 searches/month, 10 papers per search (limited metadata)
- Pro: $8.99/month — 1,000 searches/month, study types, takeaways
- Enterprise: Custom pricing, DOIs included, higher limits

---

## 11. Scite.ai

**What it is:** Smart Citations platform that indexes 1.6B+ citations from scientific literature. Shows whether papers *support*, *contrast*, or merely *mention* each other — the only API that provides citation sentiment analysis. Full-text search across peer-reviewed papers with excerpts.

**Base URL:** `https://api.scite.ai/mcp`  
**Auth:** OAuth 2.1 with PKCE (Dynamic Client Registration, no static API key)  
**Protocol:** MCP (Model Context Protocol) over Streamable HTTP / JSON-RPC  
**Env vars:** `SCITE_ACCESS_TOKEN`, `SCITE_REFRESH_TOKEN`, `SCITE_CLIENT_ID`

### Authentication — OAuth 2.1 Flow

Scite.ai uses OAuth 2.1 with PKCE — no static API key. The flow:

1. **Discovery:** `GET https://api.scite.ai/.well-known/oauth-authorization-server`
2. **Dynamic Client Registration (DCR):**
   ```bash
   curl -X POST "https://api.scite.ai/mcp/oauth/register" \
     -H "Content-Type: application/json" \
     -d '{"client_name": "MiroThinker", "redirect_uris": ["http://localhost:8765/oauth/callback"], "grant_types": ["authorization_code", "refresh_token"], "token_endpoint_auth_method": "none", "scope": "mcp"}'
   ```
   Returns: `client_id` (no secret — public client with PKCE)
3. **Authorization:** Open browser to `https://api.scite.ai/mcp/oauth/authorize?response_type=code&client_id=...&code_challenge=...&code_challenge_method=S256`
4. **Token Exchange:**
   ```bash
   curl -X POST "https://api.scite.ai/mcp/oauth/token" \
     -d "grant_type=authorization_code&code=AUTH_CODE&redirect_uri=...&client_id=...&code_verifier=..."
   ```
   Returns: `access_token` (12h TTL), `refresh_token` (long-lived)
5. **Token Refresh:** Use refresh_token to get new access_token without re-login

### MCP Protocol

Scite.ai uses the Model Context Protocol (JSON-RPC over HTTP):

```python
import requests

# Initialize session
response = requests.post("https://api.scite.ai/mcp",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
    json={"jsonrpc": "2.0", "method": "initialize",
          "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                     "clientInfo": {"name": "MiroThinker", "version": "1.0.0"}}, "id": 1})

# Call search_literature
response = requests.post("https://api.scite.ai/mcp",
    headers={"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"},
    json={"jsonrpc": "2.0", "method": "tools/call",
          "params": {"name": "search_literature",
                     "arguments": {"term": "lutein zeaxanthin macular pigment", "limit": 10}}, "id": 2})
```

### The `search_literature` Tool — Full Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `term` | string | Cross-field search query. Supports Boolean (AND, OR, NOT), phrase ("exact"), proximity ("term1 term2"~5) |
| `dois` | array[string] | Filter to specific DOIs (preferred over titles for exact match) |
| `titles` | array[string] | Filter to papers matching these titles |
| `limit` | int | Max results (default 10, max 1000) |
| `offset` | int | Pagination offset |
| `title` | string | Filter by title text |
| `abstract` | string | Filter by abstract text |
| `author` | string | Filter by author name |
| `journal` | string | Filter by journal name |
| `publisher` | string | Filter by publisher |
| `year` | int | Filter by publication year |
| `date_from` | string | Papers from this date (YYYY-MM-DD) |
| `date_to` | string | Papers up to this date |
| `paper_type` | string | Article, Review, Clinical Trial, Meta-Analysis, Case Report |
| `affiliation` | string | Author institution |
| `topic` | string | Research topic/subject area |
| `has_tally` | bool | Papers with Smart Citations |
| `has_retraction` | bool | Retracted papers only |
| `has_concern` | bool | Papers with editorial concerns |
| `supporting_from/to` | int | Min/max supporting citations |
| `contrasting_from/to` | int | Min/max contrasting citations |
| `mentioning_from/to` | int | Min/max mentioning citations |
| `citing_publications_from/to` | int | Min/max total citing publications |

### Response Structure

```json
{
  "hits": [{
    "doi": "10.3390/nu8070426",
    "title": "Lutein, Zeaxanthin and Meso-zeaxanthin Supplementation...",
    "authors": [{"authorName": "Jane Smith"}],
    "abstract": "Full abstract...",
    "year": 2016,
    "journal": "Nutrients",
    "tally": {
      "total": 86,
      "supporting": 5,
      "contrasting": 0,
      "mentioning": 81
    },
    "fulltextExcerpts": ["Relevant passage from the paper (~500 chars)..."],
    "access": {"url": "https://doi.org/...", "accessType": "open", "contentType": "pdf"},
    "citations": [{
      "snippet": "Actual quoted text from a citing paper",
      "type": "supporting",
      "section": "Results",
      "sourceDoi": "10.xxxx/citing-paper",
      "targetDoi": "10.3390/nu8070426"
    }],
    "retraction_notices": [],
    "isOa": true,
    "oaStatus": "gold"
  }]
}
```

### Why Scite.ai is Transformative for MiroThinker

1. **Smart Citations** — The killer feature. Shows *how* papers cite each other:
   - **Supporting**: "Our results confirm the findings of Smith et al."
   - **Contrasting**: "In contrast to Smith et al., we found no significant effect"
   - **Mentioning**: "Smith et al. previously studied this topic"
2. **Citation sentiment analysis** — No other API provides this
3. **Full-text excerpts** — Up to 5 passages (~500 chars each) per paper, matching query
4. **Retraction/correction tracking** — Essential for research integrity
5. **1.6B+ citations indexed** — Comprehensive coverage
6. **Section-level citation context** — Know if a citation appears in Methods vs Results vs Discussion

### Optimal MiroThinker Usage
- **Citation graph traversal** — Find paper A, see who supports/contradicts it, follow the chain
- **Claim verification** — Search a claim, check if supporting > contrasting citations
- **Retraction screening** — Filter `has_retraction: true` before citing any paper
- **Deep reading workflow**: Search broadly → get DOIs → query with targeted `term` to read section by section
- **Combine with Exa/Tavily**: They find URLs, scite.ai validates the science behind them
- Use `supporting_from: 10` to find well-supported papers (consensus indicators)
- Use `contrasting_from: 5` to find controversial papers (active debate)

### Token Management
- Access tokens expire in 12 hours
- Use refresh_token to get new access_token without re-login:
  ```bash
  curl -X POST "https://api.scite.ai/mcp/oauth/token" \
    -d "grant_type=refresh_token&refresh_token=REFRESH_TOKEN&client_id=CLIENT_ID"
  ```
- Store refresh_token securely — it's long-lived
- MiroThinker should auto-refresh before each session

### Pricing
- Requires scite.ai premium subscription (~$10-20/month)
- MCP access included with premium plan
- No per-query charges once subscribed

---

## 12. Integration Strategy for MiroThinker

### Tiered Research Architecture

```
TIER 1: Primary Discovery (every query)
├── Exa (neural search, category: "research paper")
├── Tavily (agent-optimized search, advanced depth)
└── Firecrawl (content extraction from discovered URLs)

TIER 2: Deep Investigation (budget-gated, thinker-directed)
├── Perplexity sonar-deep-research (synthesized multi-step research)
├── Kagi Summarizer (long paper summarization, muriel engine)
└── Apify Google Scholar/PubMed (structured academic data)

TIER 3: Diversity & Hidden Gems (gap-filling)
├── Mojeek (independent index, unique results)
├── Marginalia (small web, non-commercial content)
├── Kagi Enrichment (Teclis/TinyGem indie indexes)
└── Jina Search (SERP fallback)

TIER 4: Content Processing (post-discovery)
├── Jina Reader (URL → clean markdown, simpler than Firecrawl)
├── Jina Embeddings (corpus deduplication, similarity clustering)
├── Jina Reranker (post-hoc relevance reranking)
└── Firecrawl JSON extraction (structured data from pages)
```

### Optimal Query Flow

```
1. THINKER generates research question
   │
2. RESEARCHER executes Tier 1 search:
   ├── Exa search (neural, category: research paper, highlights)
   ├── Tavily search (advanced, include_answer, include_raw_content)
   │   (both run in parallel)
   │
3. Results ingested into corpus → Flock battery scores/filters
   │
4. THINKER reviews, identifies gaps → requests Tier 2:
   ├── Perplexity deep research for synthesis questions
   ├── Apify Google Scholar for citation-rich paper discovery
   ├── Kagi Summarizer for long PDF papers
   │
5. If diversity needed → Tier 3:
   ├── Mojeek for alternative perspectives
   ├── Marginalia for indie/small web sources
   │
6. Content processing (Tier 4):
   ├── Firecrawl/Jina Reader for full text extraction
   ├── Jina Reranker to sort by relevance
   └── Jina Embeddings for dedup before final ingestion
```

### Concurrency & Rate Limits

| API | Recommended Concurrency | Rate Limit Notes |
|-----|------------------------|------------------|
| Exa | 3-5 parallel | Generous limits |
| Tavily | 3-5 parallel | 1000/month free tier |
| Firecrawl | 2-3 parallel | Credit-based |
| Perplexity | 1 (budget-gated) | Expensive per query |
| Kagi | 1-2 parallel | $0.025-$1 per call |
| Jina | 2-3 parallel | Token-based |
| Mojeek | 1 | Stricter rate limits |
| Marginalia | 1 | Rate limited |
| Apify | 1 (async) | Actor-dependent |

### Cost Optimization

| Strategy | Implementation |
|----------|---------------|
| **Cache aggressively** | Store Exa/Tavily results in corpus; don't re-search same query |
| **Use highlights over full text** | Exa highlights (4000 chars) vs full page extraction |
| **Tavily include_raw_content** | Gets full markdown in one call (no separate Firecrawl needed) |
| **Kagi enrichment before search** | $0.002 vs $0.025 — check enrichment first for niche topics |
| **Perplexity only for synthesis** | Don't use for simple factual lookups — use Exa/Tavily instead |
| **Batch Apify runs** | Run once for 50+ papers vs. individual API calls |
| **Jina Reader for simple pages** | Cheaper than Firecrawl for straightforward URL→markdown |

### Citation Graph Analysis — Now Available via Scite.ai

Scite.ai provides **citation graph traversal** with sentiment — the only API that shows whether citations are supporting, contradicting, or mentioning. This is now fully integrated via OAuth 2.1 (see Section 11). Use it to:
- Verify claims by checking support/contrast ratios
- Follow citation chains to find foundational papers
- Screen for retractions before citing any paper
- Identify active scientific debates (high contrasting citation count)

### Recommended `.env` Configuration

```env
# Tier 1: Primary Discovery
EXA_API_KEY=...
TAVILY_API_KEY=...
FIRECRAWL_API_KEY=...

# Tier 2: Deep Investigation (budget-gated)
PERPLEXITY_API_KEY=...
KAGI_API_KEY=...
APIFY_API_KEY=...

# Tier 3: Diversity
MOJEEK_API_KEY=...
MARGINALIA_API_KEY=...
JINA_API_KEY=...

# Tier 4: Content Processing (uses Jina + Firecrawl from above)

# Budget Controls
DEEP_RESEARCH_SESSION_BUDGET=10.00
DEEP_RESEARCH_MONTHLY_BUDGET=200.00

# Consensus (OAuth 2.1 — same pattern as Scite.ai)
CONSENSUS_REFRESH_TOKEN=...  # Long-lived refresh token from OAuth flow
CONSENSUS_CLIENT_ID=...      # OAuth client ID from DCR

# Scite.ai (OAuth 2.1 — no static key, uses refresh token)
SCITE_REFRESH_TOKEN=...  # Long-lived refresh token from OAuth flow
SCITE_CLIENT_ID=...       # OAuth client ID from DCR
```

---

## Appendix: Quick Comparison Matrix

| Feature | Exa | Tavily | Firecrawl | Perplexity | Kagi | Jina | Mojeek | Marginalia | Apify | Consensus | Scite.ai |
|---------|-----|--------|-----------|------------|------|------|--------|------------|-------|-----------|----------|
| Web search | ++ | ++ | - | ++ | + | + | + | + | - | + | + |
| Academic filter | ++ | + | - | + | - | - | - | - | ++ | ++ | ++ |
| Content extraction | + | + | ++ | - | - | ++ | - | - | ++ | - | + |
| Summarization | + | + | - | ++ | ++ | - | - | - | - | + | - |
| Structured extraction | + | - | ++ | + | - | - | - | - | ++ | + | + |
| Study type filtering | - | - | - | - | - | - | - | - | - | ++ | + |
| Citation tracking | - | - | - | + | - | - | - | - | - | - | ++ |
| Smart Citations | - | - | - | - | - | - | - | - | - | - | ++ |
| Retraction screening | - | - | - | - | - | - | - | - | - | - | ++ |
| AI takeaways | - | - | - | + | - | - | - | - | - | ++ | - |
| Indie/small web | - | - | - | - | ++ | - | ++ | ++ | - | - | - |
| Independent index | - | - | - | - | + | - | ++ | ++ | - | + | + |
| PDF handling | - | - | + | - | ++ | + | - | - | - | - | - |
| Anti-bot/proxy | - | - | ++ | - | - | + | - | - | ++ | - | - |
| Embeddings | - | - | - | + | - | ++ | - | - | - | - | - |
| Reranking | - | - | - | - | - | ++ | - | - | - | - | - |
| Async/batch | - | - | + | - | - | - | - | - | ++ | - | - |
| Cost per query | $$ | $ | $ | $$$ | $$ | $ | $ | $ | $$ | $ | flat |

**Legend:** `++` = best-in-class, `+` = capable, `-` = not available, `$` = cheap, `$$` = moderate, `$$$` = expensive, `flat` = subscription-based
