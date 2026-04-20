"""API key provider registry for MiroThinker integration tests.

Follows the Strands SDK ProviderInfo pattern — each provider has:
- An environment variable holding the API key
- A pytest mark that skips tests when the key is missing
- A minimal health-check callable that validates the key works

Usage in tests:
    pytestmark = providers.venice.mark

    def test_venice_responds(venice_client):
        ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pytest import mark


@dataclass
class ProviderInfo:
    """Provider-based info for services that require an API key."""

    id: str
    env_var: str
    base_url: str = ""
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mark(self):
        """Pytest mark that skips when the env var is missing or empty."""
        return mark.skipif(
            not os.environ.get(self.env_var),
            reason=f"{self.env_var} environment variable missing",
        )

    @property
    def api_key(self) -> str:
        return os.environ.get(self.env_var, "")


# ── Search providers ──────────────────────────────────────────────────

venice = ProviderInfo(
    id="venice",
    env_var="VENICE_API_KEY",
    base_url="https://api.venice.ai/api/v1",
    description="Venice AI — uncensored LLM inference",
)

brave = ProviderInfo(
    id="brave",
    env_var="BRAVE_API_KEY",
    base_url="https://api.search.brave.com/res/v1",
    description="Brave Search — independent web search",
)

exa = ProviderInfo(
    id="exa",
    env_var="EXA_API_KEY",
    base_url="https://api.exa.ai",
    description="Exa — neural search engine",
)

firecrawl = ProviderInfo(
    id="firecrawl",
    env_var="FIRECRAWL_API_KEY",
    base_url="https://api.firecrawl.dev/v1",
    description="Firecrawl — web scraping and extraction",
)

jina = ProviderInfo(
    id="jina",
    env_var="JINA_API_KEY",
    base_url="https://r.jina.ai",
    description="Jina Reader — URL text extraction",
)

kagi = ProviderInfo(
    id="kagi",
    env_var="KAGI_API_KEY",
    base_url="https://kagi.com/api/v0",
    description="Kagi — premium search",
)

mojeek = ProviderInfo(
    id="mojeek",
    env_var="MOJEEK_API_KEY",
    base_url="https://api.mojeek.com/search",
    description="Mojeek — independent crawler, not a Google proxy",
)

perplexity = ProviderInfo(
    id="perplexity",
    env_var="PERPLEXITY_API_KEY",
    base_url="https://api.perplexity.ai",
    description="Perplexity — AI-powered deep research",
)

tavily = ProviderInfo(
    id="tavily",
    env_var="TAVILY_API_KEY",
    base_url="https://api.tavily.com",
    description="Tavily — research-optimized search",
)

xai = ProviderInfo(
    id="xai",
    env_var="XAI_API_KEY",
    base_url="https://api.x.ai/v1",
    description="xAI Grok — uncensored reasoning",
)

# ── Data providers ────────────────────────────────────────────────────

transcript_api = ProviderInfo(
    id="transcript_api",
    env_var="TRANSCRIPTAPI_KEY",
    base_url="https://www.transcriptapi.com/api",
    description="TranscriptAPI — YouTube transcript extraction",
)

bright_data = ProviderInfo(
    id="bright_data",
    env_var="BRIGHT_DATA_API_KEY",
    base_url="https://api.brightdata.com",
    description="Bright Data — web scraping proxy network",
)

# ── Storage providers ─────────────────────────────────────────────────

b2 = ProviderInfo(
    id="b2",
    env_var="B2_APPLICATION_KEY",
    base_url="https://api.backblazeb2.com/b2api/v2",
    description="Backblaze B2 — object storage data lake",
    extra={"key_id_var": "B2_KEY_ID"},
)


# ── All providers ─────────────────────────────────────────────────────

all_providers = [
    venice, brave, exa, firecrawl, jina, kagi, mojeek,
    perplexity, tavily, xai, transcript_api, bright_data, b2,
]

search_providers = [
    brave, exa, firecrawl, jina, kagi, mojeek,
    perplexity, tavily, xai,
]

llm_providers = [venice, perplexity, xai]
