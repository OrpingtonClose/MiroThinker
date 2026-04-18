# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
MCP server configurations for external service integrations.

Documents and provides helper functions for MCP servers that extend the
research pipeline's capabilities. These are meant to be used alongside
the Strands agent's native MCP support.

Supported MCP servers:
  1. paper-search-mcp  — Academic paper search (arXiv, PubMed, Semantic Scholar, CORE, Unpaywall, etc.)
  2. youtube-mcp-server — YouTube transcription & metadata extraction
  3. backblaze-mcp      — Backblaze B2 storage operations
  4. brightdata-mcp     — Bright Data web scraping & proxy
  5. docling-mcp        — Document AI for PDF/DOCX extraction

Each function returns a configuration dict suitable for MCP client setup.
"""

from __future__ import annotations

import os
from typing import Optional


def paper_search_mcp_config(
    unpaywall_email: str = "",
    core_api_key: str = "",
    semantic_scholar_key: str = "",
    zenodo_token: str = "",
    ieee_api_key: str = "",
) -> dict:
    """Configuration for openags/paper-search-mcp.

    Unified academic paper search across:
    - arXiv (free, no key)
    - PubMed / bioRxiv / medRxiv (free, no key)
    - Semantic Scholar (free, key optional for higher limits)
    - CORE (free key required — register at core.ac.uk)
    - Unpaywall (free, just needs an email)
    - Zenodo (free token from zenodo.org/account/settings/applications/)
    - Google Scholar (needs proxy URL)
    - IEEE Xplore (key from developer.ieee.org)
    - ACM Digital Library (key needed)

    Install: pip install paper-search-mcp
    Or:      npx @smithery/cli install @openags/paper-search-mcp --client claude

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "paper-search-mcp",
        "description": "Academic paper search across arXiv, PubMed, Semantic Scholar, CORE, Unpaywall, Zenodo, IEEE, ACM",
        "install": "pip install paper-search-mcp",
        "repo": "https://github.com/openags/paper-search-mcp",
        "command": "python",
        "args": ["-m", "paper_search_mcp.server"],
        "env": {
            "PAPER_SEARCH_MCP_UNPAYWALL_EMAIL": unpaywall_email or os.getenv("UNPAYWALL_EMAIL", ""),
            "PAPER_SEARCH_MCP_CORE_API_KEY": core_api_key or os.getenv("CORE_API_KEY", ""),
            "PAPER_SEARCH_MCP_SEMANTIC_SCHOLAR_API_KEY": semantic_scholar_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
            "PAPER_SEARCH_MCP_ZENODO_ACCESS_TOKEN": zenodo_token or os.getenv("ZENODO_ACCESS_TOKEN", ""),
            "PAPER_SEARCH_MCP_IEEE_API_KEY": ieee_api_key or os.getenv("IEEE_API_KEY", ""),
        },
        "tools": [
            "search_arxiv", "download_arxiv",
            "search_pubmed", "search_biorxiv",
            "search_semantic_scholar",
            "search_core",
            "search_zenodo",
            "search_google_scholar",
            "search_ieee",
        ],
    }


def youtube_mcp_config() -> dict:
    """Configuration for mourad-ghafiri/youtube-mcp-server.

    YouTube video transcription and metadata extraction with:
    - yt-dlp for robust video/audio download
    - Whisper for multilingual transcription (99 languages)
    - VAD (Voice Activity Detection) for precise segmentation
    - Intelligent caching to avoid redundant processing

    Install: git clone https://github.com/mourad-ghafiri/youtube-mcp-server && uv sync

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "youtube-mcp-server",
        "description": "YouTube transcription & metadata with Whisper, VAD, 99 languages",
        "install": "git clone https://github.com/mourad-ghafiri/youtube-mcp-server && cd youtube-mcp-server && uv sync",
        "repo": "https://github.com/mourad-ghafiri/youtube-mcp-server",
        "command": "uv",
        "args": ["run", "python", "-m", "youtube_mcp_server"],
        "env": {},
        "tools": [
            "get_video_metadata",
            "transcribe_video",
        ],
    }


def backblaze_b2_mcp_config(
    key_id: str = "",
    app_key: str = "",
) -> dict:
    """Configuration for braveram/backblaze-mcp.

    Backblaze B2 storage operations:
    - Bucket management (create, delete, list, update)
    - File operations (upload, list, hide, delete)
    - Large file support (multipart uploads >100MB)
    - Application key management

    Install: npx backblaze-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "backblaze-b2",
        "description": "Backblaze B2 cloud storage — buckets, files, large uploads",
        "install": "npm install -g backblaze-mcp",
        "repo": "https://github.com/BraveRam/backblaze-mcp",
        "command": "npx",
        "args": ["backblaze-mcp"],
        "env": {
            "B2_APPLICATION_KEY_ID": key_id or os.getenv("B2_KEY_ID", ""),
            "B2_APPLICATION_KEY": app_key or os.getenv("B2_APPLICATION_KEY", ""),
        },
        "tools": [
            "create_bucket", "delete_bucket", "list_buckets", "update_bucket",
            "upload_file", "list_file_names", "list_file_versions",
            "hide_file", "delete_file_version",
            "start_large_file", "get_upload_part_url", "finish_large_file",
            "create_key", "delete_key", "list_keys",
            "get_download_authorization",
        ],
    }


def brightdata_mcp_config(
    api_token: str = "",
) -> dict:
    """Configuration for luminati-io/brightdata-mcp (Bright Data).

    Web scraping and proxy services:
    - SERP API (search engine results)
    - Web Unlocker (bypass anti-bot)
    - Web Scraper API (structured data extraction)
    - Dataset API (pre-built datasets)

    Install: npx @anthropic/create-mcp@latest

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "brightdata-mcp",
        "description": "Bright Data web scraping — SERP, Web Unlocker, structured extraction",
        "install": "npx @anthropic/create-mcp@latest",
        "repo": "https://github.com/luminati-io/brightdata-mcp",
        "command": "npx",
        "args": ["@anthropic/create-mcp@latest"],
        "env": {
            "API_TOKEN": api_token or os.getenv("BRIGHT_DATA_API_KEY", ""),
        },
        "tools": [
            "search_engine", "scrape_as_markdown", "scrape_as_html",
            "web_data_amazon_product", "web_data_linkedin_profile",
            "session_management",
        ],
    }


def docling_mcp_config() -> dict:
    """Configuration for docling-project/docling-mcp.

    Document AI for PDF/DOCX extraction:
    - Complex table extraction
    - Multi-column layout detection
    - Scanned PDF OCR (with Granite VLM)
    - Math formula identification
    - Batch document processing

    Install: pip install docling-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "docling-mcp",
        "description": "Document AI — tables, multi-column, OCR, math formulas",
        "install": "pip install docling-mcp",
        "repo": "https://github.com/docling-project/docling-mcp",
        "command": "python",
        "args": ["-m", "docling_mcp"],
        "env": {},
        "tools": [
            "convert_document",
            "convert_document_with_images",
            "extract_tables",
            "convert_batch",
        ],
    }


def onionclaw_mcp_config() -> dict:
    """Configuration for ASpirin01/onionclaw-mcp (Dark Web OSINT).

    18 dark web search engines + Tor access + Robin OSINT pipeline +
    4 LLM analysis modes. Enables research into content removed from
    clearnet, whistleblower documents, and censored information.

    Install: pip install onionclaw-mcp
    Requires: Tor running locally (apt install tor && systemctl start tor)

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "onionclaw-mcp",
        "description": "Dark web OSINT — 18 search engines, Tor access, Robin pipeline, LLM analysis",
        "install": "pip install onionclaw-mcp",
        "repo": "https://github.com/ASpirin01/onionclaw-mcp",
        "command": "python",
        "args": ["-m", "onionclaw_mcp"],
        "env": {},
        "tools": [
            "search_onion", "fetch_onion_page", "robin_pipeline",
            "analyze_content", "search_ahmia", "search_torch",
        ],
    }


def onion_search_mcp_config() -> dict:
    """Configuration for Onion Search MCP (anonymous Tor browsing).

    Anonymous browsing with universal browser fingerprint. Searches .onion
    sites and routes traffic through Tor for censorship circumvention.

    Install: pip install onion-search-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "onion-search-mcp",
        "description": "Anonymous Tor browsing — universal fingerprint, .onion search",
        "install": "pip install onion-search-mcp",
        "repo": "https://github.com/nicholasgriffintn/onion-search-mcp",
        "command": "python",
        "args": ["-m", "onion_search_mcp"],
        "env": {},
        "tools": [
            "search_onion_sites", "browse_onion", "check_tor_status",
        ],
    }


def osint_mcp_config() -> dict:
    """Configuration for badchars/osint-mcp (37 OSINT tools).

    Shodan, VirusTotal, Censys, SecurityTrails, DNS, WHOIS, certificate
    transparency, BGP, Wayback Machine, GeoIP. Comprehensive OSINT toolkit.

    Install: pip install osint-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "osint-mcp",
        "description": "OSINT toolkit — Shodan, VirusTotal, Censys, DNS, WHOIS, cert transparency",
        "install": "pip install osint-mcp",
        "repo": "https://github.com/badchars/osint-mcp",
        "command": "python",
        "args": ["-m", "osint_mcp"],
        "env": {
            "SHODAN_API_KEY": os.getenv("SHODAN_API_KEY", ""),
            "VIRUSTOTAL_API_KEY": os.getenv("VIRUSTOTAL_API_KEY", ""),
            "CENSYS_API_ID": os.getenv("CENSYS_API_ID", ""),
            "CENSYS_API_SECRET": os.getenv("CENSYS_API_SECRET", ""),
            "SECURITYTRAILS_API_KEY": os.getenv("SECURITYTRAILS_API_KEY", ""),
        },
        "tools": [
            "shodan_search", "virustotal_scan", "censys_search",
            "dns_lookup", "whois_lookup", "cert_transparency",
            "bgp_lookup", "geoip_lookup",
        ],
    }


def osint_intelligence_mcp_config() -> dict:
    """Configuration for OSINT Intelligence Platform (Telegram archive intelligence).

    65 tools for Telegram channel/group intelligence, message analysis,
    and social media OSINT. Useful for monitoring censored communications.

    Install: pip install osint-intelligence-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "osint-intelligence-mcp",
        "description": "Telegram OSINT — 65 tools for channel/group intelligence and analysis",
        "install": "pip install osint-intelligence-mcp",
        "repo": "https://github.com/shaga/osint-intelligence-mcp",
        "command": "python",
        "args": ["-m", "osint_intelligence_mcp"],
        "env": {
            "TELEGRAM_API_ID": os.getenv("TELEGRAM_API_ID", ""),
            "TELEGRAM_API_HASH": os.getenv("TELEGRAM_API_HASH", ""),
        },
        "tools": [
            "search_telegram_channels", "get_channel_messages",
            "analyze_channel", "search_messages",
        ],
    }


def osint_tools_mcp_config() -> dict:
    """Configuration for OSINT Tools MCP (Sherlock, SpiderFoot, Holehe).

    Username OSINT (Sherlock), reconnaissance (SpiderFoot), and email
    intelligence (Holehe). 183+ stars on GitHub.

    Install: pip install osint-tools-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "osint-tools-mcp",
        "description": "OSINT tools — Sherlock username search, SpiderFoot recon, Holehe email",
        "install": "pip install osint-tools-mcp",
        "repo": "https://github.com/pab47/osint-tools-mcp",
        "command": "python",
        "args": ["-m", "osint_tools_mcp"],
        "env": {},
        "tools": [
            "sherlock_search", "spiderfoot_scan", "holehe_check",
        ],
    }


def scholar_mcp_config() -> dict:
    """Configuration for Scholar MCP (fused multi-source academic search).

    Fused search across 6+ academic sources with ~97% literature coverage.
    Includes optional Sci-Hub integration for full-text access.

    Install: pip install scholar-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "scholar-mcp",
        "description": "Fused academic search — 6+ sources, ~97% coverage, optional Sci-Hub",
        "install": "pip install scholar-mcp",
        "repo": "https://github.com/andybrandt/mcp-scholar",
        "command": "python",
        "args": ["-m", "scholar_mcp"],
        "env": {},
        "tools": [
            "search_papers", "get_paper_details", "get_citations",
            "get_references", "download_paper",
        ],
    }


def pubmed_search_mcp_config() -> dict:
    """Configuration for PubMed Search MCP (42 tools).

    Comprehensive biomedical search across PubMed, Europe PMC, CORE,
    OpenAlex, NCBI databases, and open-access figures. 42 tools total.

    Install: pip install pubmed-search-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "pubmed-search-mcp",
        "description": "PubMed + Europe PMC + CORE + OpenAlex + NCBI + OA figures (42 tools)",
        "install": "pip install pubmed-search-mcp",
        "repo": "https://github.com/pleomax0730/pubmed-search-mcp",
        "command": "python",
        "args": ["-m", "pubmed_search_mcp"],
        "env": {
            "NCBI_API_KEY": os.getenv("NCBI_API_KEY", ""),
        },
        "tools": [
            "search_pubmed", "get_article", "search_europe_pmc",
            "search_core", "search_openalex", "get_oa_figures",
        ],
    }


def wikidata_mcp_config() -> dict:
    """Configuration for official Wikidata MCP (Wikimedia Foundation).

    100M+ structured entities, SPARQL queries, entity lookup.
    Official MCP server from Wikimedia.

    Install: pip install wikidata-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "wikidata-mcp",
        "description": "Wikidata — 100M+ structured entities, SPARQL, official Wikimedia MCP",
        "install": "pip install wikidata-mcp",
        "repo": "https://github.com/wikimedia/wikidata-mcp",
        "command": "python",
        "args": ["-m", "wikidata_mcp"],
        "env": {},
        "tools": [
            "search_entities", "get_entity", "sparql_query",
            "get_claims", "get_sitelinks",
        ],
    }


def wayback_mcp_config() -> dict:
    """Configuration for Wayback Machine MCP (Internet Archive).

    Recover censored/deleted web content. Search and retrieve archived
    web pages from the Internet Archive's Wayback Machine.

    Install: pip install wayback-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "wayback-mcp",
        "description": "Wayback Machine — recover deleted/censored web content",
        "install": "pip install wayback-mcp",
        "repo": "https://github.com/cmorley/wayback-mcp",
        "command": "python",
        "args": ["-m", "wayback_mcp"],
        "env": {},
        "tools": [
            "search_wayback", "get_snapshot", "get_calendar",
            "save_page", "get_closest_snapshot",
        ],
    }


def clinicaltrials_mcp_config() -> dict:
    """Configuration for ClinicalTrials.gov MCP.

    Access the graveyard of suppressed drug trials. Pharma companies bury
    negative results here by never publishing them in journals.

    Install: pip install clinicaltrials-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "clinicaltrials-mcp",
        "description": "ClinicalTrials.gov — suppressed drug trials, terminated studies, buried results",
        "install": "pip install clinicaltrials-mcp",
        "repo": "https://github.com/natetyler/clinicaltrials-mcp",
        "command": "python",
        "args": ["-m", "clinicaltrials_mcp"],
        "env": {},
        "tools": [
            "search_trials", "get_trial", "search_by_condition",
            "search_by_intervention", "get_trial_results",
        ],
    }


def openfda_mcp_config() -> dict:
    """Configuration for OpenFDA MCP (adverse events, recalls).

    FAERS adverse events, drug recalls, device incidents. Contains
    systematically under-reported pharmaceutical adverse reactions.

    Install: pip install openfda-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "openfda-mcp",
        "description": "OpenFDA — FAERS adverse events, drug recalls, device incidents",
        "install": "pip install openfda-mcp",
        "repo": "https://github.com/openai/openfda-mcp",
        "command": "python",
        "args": ["-m", "openfda_mcp"],
        "env": {
            "OPENFDA_API_KEY": os.getenv("OPENFDA_API_KEY", ""),
        },
        "tools": [
            "search_adverse_events", "search_recalls",
            "search_device_events", "drug_labels",
        ],
    }


def sec_edgar_mcp_config() -> dict:
    """Configuration for SEC EDGAR MCP (18 tools).

    Corporate filings, insider trading, fraud in footnotes. Access to
    10-K, 10-Q, 8-K, 13F, and insider trading (Form 4) filings.

    Install: pip install sec-edgar-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "sec-edgar-mcp",
        "description": "SEC EDGAR — 18 tools for corporate filings, insider trading, fraud detection",
        "install": "pip install sec-edgar-mcp",
        "repo": "https://github.com/badbayesian/sec-edgar-mcp",
        "command": "python",
        "args": ["-m", "sec_edgar_mcp"],
        "env": {},
        "tools": [
            "search_filings", "get_filing", "company_filings",
            "insider_trading", "institutional_holdings",
        ],
    }


def court_records_mcp_config() -> dict:
    """Configuration for Court Records MCP (CourtListener/RECAP).

    Free PACER access via CourtListener. Unsealed pharmaceutical lawsuits,
    environmental violations, patent disputes.

    Install: pip install court-records-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "court-records-mcp",
        "description": "CourtListener/RECAP — free PACER, unsealed lawsuits, court opinions",
        "install": "pip install court-records-mcp",
        "repo": "https://github.com/freelawproject/courtlistener-mcp",
        "command": "python",
        "args": ["-m", "court_records_mcp"],
        "env": {
            "COURTLISTENER_API_TOKEN": os.getenv("COURTLISTENER_API_TOKEN", ""),
        },
        "tools": [
            "search_opinions", "search_dockets", "get_opinion",
            "search_recap", "get_document",
        ],
    }


def google_scholar_mcp_config() -> dict:
    """Configuration for Google Scholar MCP.

    Google Scholar scraping with proxy support. Requires Bright Data
    or similar proxy to avoid blocks.

    Install: pip install google-scholar-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "google-scholar-mcp",
        "description": "Google Scholar — widest academic coverage, needs proxy",
        "install": "pip install google-scholar-mcp",
        "repo": "https://github.com/nicholasgriffintn/google-scholar-mcp",
        "command": "python",
        "args": ["-m", "google_scholar_mcp"],
        "env": {
            "PROXY_URL": os.getenv("GOOGLE_SCHOLAR_PROXY_URL", ""),
            "SERPAPI_KEY": os.getenv("SERPAPI_KEY", ""),
        },
        "tools": [
            "search_scholar", "get_paper", "get_citations",
            "get_author_profile",
        ],
    }


def ipfs_mcp_config() -> dict:
    """Configuration for IPFS MCP (decentralized content).

    Access DMCA-removed content persisted on IPFS. Censorship-resistant
    storage for papers, datasets, and documents.

    Install: pip install ipfs-mcp

    Returns:
        MCP server configuration dict.
    """
    return {
        "name": "ipfs-mcp",
        "description": "IPFS — decentralized storage, censorship-resistant content access",
        "install": "pip install ipfs-mcp",
        "repo": "https://github.com/vkirilichev/ipfs-mcp",
        "command": "python",
        "args": ["-m", "ipfs_mcp"],
        "env": {},
        "tools": [
            "ipfs_cat", "ipfs_get", "ipfs_ls", "ipfs_pin",
        ],
    }


def get_all_mcp_configs() -> list[dict]:
    """Return configurations for all supported MCP servers."""
    return [
        # Original servers
        paper_search_mcp_config(),
        youtube_mcp_config(),
        backblaze_b2_mcp_config(),
        brightdata_mcp_config(),
        docling_mcp_config(),
        # Dark web & OSINT
        onionclaw_mcp_config(),
        onion_search_mcp_config(),
        osint_mcp_config(),
        osint_intelligence_mcp_config(),
        osint_tools_mcp_config(),
        # Academic mega-sources
        scholar_mcp_config(),
        pubmed_search_mcp_config(),
        wikidata_mcp_config(),
        google_scholar_mcp_config(),
        # Censorship-resistant infrastructure
        wayback_mcp_config(),
        ipfs_mcp_config(),
        # Government & legal
        clinicaltrials_mcp_config(),
        openfda_mcp_config(),
        sec_edgar_mcp_config(),
        court_records_mcp_config(),
    ]


def get_mcp_status() -> str:
    """Return a human-readable status of MCP server configurations.

    Shows which MCP servers have their required credentials configured
    and which need setup.
    """
    lines = ["**MCP Server Status**\n"]

    configs = get_all_mcp_configs()
    for cfg in configs:
        name = cfg["name"]
        env = cfg.get("env", {})

        # Check if required env vars are set
        has_creds = True
        missing = []
        for key, val in env.items():
            if not val:
                has_creds = False
                missing.append(key)

        if not env:
            status = "READY (no credentials needed)"
        elif has_creds:
            status = "CONFIGURED"
        else:
            status = f"NEEDS SETUP — missing: {', '.join(missing)}"

        lines.append(
            f"  - **{name}**: {status}\n"
            f"    {cfg['description']}\n"
            f"    Install: `{cfg['install']}`"
        )

    return "\n\n".join(lines)
