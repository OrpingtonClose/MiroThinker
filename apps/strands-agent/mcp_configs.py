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


def get_all_mcp_configs() -> list[dict]:
    """Return configurations for all supported MCP servers."""
    return [
        paper_search_mcp_config(),
        youtube_mcp_config(),
        backblaze_b2_mcp_config(),
        brightdata_mcp_config(),
        docling_mcp_config(),
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
