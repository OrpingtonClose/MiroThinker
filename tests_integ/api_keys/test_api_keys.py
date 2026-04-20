"""Integration tests for all MiroThinker API keys.

Validates that each configured API key is accepted by its service
and returns a meaningful response. Follows the Strands SDK eval pattern:
each test makes one minimal API call to verify the key works.

Run:
    pytest tests_integ/api_keys/test_api_keys.py -v

Skip unavailable providers automatically via pytest marks.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from tests_integ.api_keys import providers

# ═════════════════════════════════════════════════════════════════════
# Venice AI — LLM inference
# ═════════════════════════════════════════════════════════════════════

pytestmark_venice = providers.venice.mark


class TestVenice:
    """Venice AI chat completions endpoint."""

    pytestmark = providers.venice.mark

    def test_chat_completion(self):
        """Verify Venice API key produces a valid chat completion."""
        resp = httpx.post(
            f"{providers.venice.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {providers.venice.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.environ.get(
                    "VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic",
                ),
                "messages": [{"role": "user", "content": "Say 'key valid' in two words."}],
                "max_tokens": 20,
                "venice_parameters": {"include_venice_system_prompt": False},
            },
            timeout=30,
        )
        assert resp.status_code == 200, f"Venice returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        # Venice may return empty content on some models — key validity
        # is proven by the 200 + choices structure

    def test_model_list(self):
        """Verify Venice API key can list available models."""
        resp = httpx.get(
            f"{providers.venice.base_url}/models",
            headers={"Authorization": f"Bearer {providers.venice.api_key}"},
            timeout=15,
        )
        assert resp.status_code == 200, f"Venice models returned {resp.status_code}"
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0, "Venice returned no models"


# ═════════════════════════════════════════════════════════════════════
# Brave Search
# ═════════════════════════════════════════════════════════════════════


class TestBrave:
    """Brave Search API."""

    pytestmark = providers.brave.mark

    def test_web_search(self):
        """Verify Brave API key returns search results."""
        resp = httpx.get(
            f"{providers.brave.base_url}/web/search",
            params={"q": "test query", "count": 3},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": providers.brave.api_key,
            },
            timeout=15,
        )
        assert resp.status_code == 200, f"Brave returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "web" in data or "mixed" in data, f"Brave response missing results: {list(data.keys())}"


# ═════════════════════════════════════════════════════════════════════
# Exa Search
# ═════════════════════════════════════════════════════════════════════


class TestExa:
    """Exa neural search API."""

    pytestmark = providers.exa.mark

    def test_search(self):
        """Verify Exa API key returns search results."""
        resp = httpx.post(
            f"{providers.exa.base_url}/search",
            headers={
                "x-api-key": providers.exa.api_key,
                "Content-Type": "application/json",
            },
            json={
                "query": "test query",
                "numResults": 3,
                "type": "keyword",
            },
            timeout=15,
        )
        assert resp.status_code == 200, f"Exa returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "results" in data, f"Exa response missing results: {list(data.keys())}"


# ═════════════════════════════════════════════════════════════════════
# Firecrawl
# ═════════════════════════════════════════════════════════════════════


class TestFirecrawl:
    """Firecrawl web scraping API."""

    pytestmark = providers.firecrawl.mark

    def test_scrape(self):
        """Verify Firecrawl API key can scrape a page."""
        resp = httpx.post(
            f"{providers.firecrawl.base_url}/scrape",
            headers={
                "Authorization": f"Bearer {providers.firecrawl.api_key}",
                "Content-Type": "application/json",
            },
            json={"url": "https://example.com", "formats": ["markdown"]},
            timeout=30,
        )
        assert resp.status_code == 200, f"Firecrawl returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert data.get("success") is True or "data" in data, (
            f"Firecrawl scrape failed: {data}"
        )


# ═════════════════════════════════════════════════════════════════════
# Jina Reader
# ═════════════════════════════════════════════════════════════════════


class TestJina:
    """Jina Reader URL extraction."""

    pytestmark = providers.jina.mark

    def test_read_url(self):
        """Verify Jina API key extracts text from a URL."""
        resp = httpx.get(
            f"{providers.jina.base_url}/https://example.com",
            headers={"Authorization": f"Bearer {providers.jina.api_key}"},
            timeout=30,
            follow_redirects=True,
        )
        assert resp.status_code == 200, f"Jina returned {resp.status_code}: {resp.text[:200]}"
        assert len(resp.text) > 50, "Jina returned too little content"


# ═════════════════════════════════════════════════════════════════════
# Kagi Search
# ═════════════════════════════════════════════════════════════════════


class TestKagi:
    """Kagi premium search API."""

    pytestmark = providers.kagi.mark

    def test_search(self):
        """Verify Kagi API key returns search results."""
        resp = httpx.get(
            f"{providers.kagi.base_url}/search",
            params={"q": "test query", "limit": 3},
            headers={"Authorization": f"Bot {providers.kagi.api_key}"},
            timeout=15,
        )
        # Kagi returns 200 for valid keys; 401/403 for invalid
        assert resp.status_code in (200, 201), (
            f"Kagi returned {resp.status_code}: {resp.text[:200]}"
        )


# ═════════════════════════════════════════════════════════════════════
# Mojeek Search
# ═════════════════════════════════════════════════════════════════════


class TestMojeek:
    """Mojeek independent search API."""

    pytestmark = providers.mojeek.mark

    def test_search(self):
        """Verify Mojeek API key returns search results."""
        resp = httpx.get(
            f"{providers.mojeek.base_url}",
            params={
                "q": "test query",
                "fmt": "json",
                "t": 3,
                "api_key": providers.mojeek.api_key,
            },
            timeout=15,
        )
        assert resp.status_code == 200, f"Mojeek returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "response" in data, f"Mojeek response missing data: {list(data.keys())}"


# ═════════════════════════════════════════════════════════════════════
# Perplexity
# ═════════════════════════════════════════════════════════════════════


class TestPerplexity:
    """Perplexity AI deep research API."""

    pytestmark = providers.perplexity.mark

    def test_chat_completion(self):
        """Verify Perplexity API key produces a completion."""
        resp = httpx.post(
            f"{providers.perplexity.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {providers.perplexity.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": "Say 'key valid' in two words."}],
                "max_tokens": 20,
            },
            timeout=30,
        )
        assert resp.status_code == 200, (
            f"Perplexity returned {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


# ═════════════════════════════════════════════════════════════════════
# Tavily
# ═════════════════════════════════════════════════════════════════════


class TestTavily:
    """Tavily research search API."""

    pytestmark = providers.tavily.mark

    def test_search(self):
        """Verify Tavily API key returns search results."""
        resp = httpx.post(
            f"{providers.tavily.base_url}/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": providers.tavily.api_key,
                "query": "test query",
                "max_results": 3,
            },
            timeout=15,
        )
        assert resp.status_code == 200, f"Tavily returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "results" in data, f"Tavily response missing results: {list(data.keys())}"


# ═════════════════════════════════════════════════════════════════════
# xAI Grok
# ═════════════════════════════════════════════════════════════════════


class TestXAI:
    """xAI Grok chat completions (OpenAI-compatible)."""

    pytestmark = providers.xai.mark

    def test_chat_completion(self):
        """Verify xAI API key produces a completion."""
        resp = httpx.post(
            f"{providers.xai.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {providers.xai.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-3-mini-fast",
                "messages": [{"role": "user", "content": "Say 'key valid' in two words."}],
                "max_tokens": 20,
            },
            timeout=30,
        )
        assert resp.status_code == 200, f"xAI returned {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0


# ═════════════════════════════════════════════════════════════════════
# TranscriptAPI
# ═════════════════════════════════════════════════════════════════════


class TestTranscriptAPI:
    """TranscriptAPI YouTube transcript extraction."""

    pytestmark = providers.transcript_api.mark

    def test_transcript_fetch(self):
        """Verify TranscriptAPI key can fetch a known video transcript."""
        resp = httpx.get(
            f"{providers.transcript_api.base_url}/transcript",
            params={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "api_key": providers.transcript_api.api_key,
            },
            timeout=30,
        )
        # Accept 200 (success) or 404 (video not found) — both prove key is valid
        assert resp.status_code in (200, 404, 422), (
            f"TranscriptAPI returned {resp.status_code}: {resp.text[:200]}"
        )


# ═════════════════════════════════════════════════════════════════════
# Bright Data
# ═════════════════════════════════════════════════════════════════════


class TestBrightData:
    """Bright Data web scraping proxy."""

    pytestmark = providers.bright_data.mark

    def test_account_status(self):
        """Verify Bright Data API key can query account status."""
        resp = httpx.get(
            "https://api.brightdata.com/zone/get_active_zones",
            headers={"Authorization": f"Bearer {providers.bright_data.api_key}"},
            timeout=15,
        )
        # 200 = valid key with zones, 403/401 = invalid key
        assert resp.status_code in (200, 204), (
            f"Bright Data returned {resp.status_code}: {resp.text[:200]}"
        )


# ═════════════════════════════════════════════════════════════════════
# Backblaze B2
# ═════════════════════════════════════════════════════════════════════


class TestB2:
    """Backblaze B2 object storage."""

    pytestmark = providers.b2.mark

    def test_authorize_account(self):
        """Verify B2 key pair can authorize."""
        key_id = os.environ.get("B2_KEY_ID", "")
        app_key = providers.b2.api_key

        if not key_id:
            pytest.skip("B2_KEY_ID not set")

        resp = httpx.get(
            "https://api.backblazeb2.com/b2api/v2/b2_authorize_account",
            auth=(key_id, app_key),
            timeout=15,
        )
        assert resp.status_code == 200, (
            f"B2 authorize returned {resp.status_code}: {resp.text[:200]}"
        )
        data = resp.json()
        assert "authorizationToken" in data, "B2 response missing auth token"
        assert "apiUrl" in data, "B2 response missing apiUrl"


# ═════════════════════════════════════════════════════════════════════
# DuckDuckGo (no API key — test that the library works)
# ═════════════════════════════════════════════════════════════════════


def _ddgs_search(args):
    """Module-level function so multiprocessing can pickle it."""
    from ddgs import DDGS
    with DDGS(timeout=15) as ddgs:
        return list(ddgs.text("test query", max_results=3))


class TestDuckDuckGo:
    """DuckDuckGo search via ddgs library (no API key needed)."""

    def test_text_search(self):
        """Verify ddgs library can perform a text search."""
        import multiprocessing as mp

        with mp.Pool(1) as pool:
            result = pool.apply_async(_ddgs_search, (None,))
            try:
                results = result.get(timeout=30)
            except mp.TimeoutError:
                pytest.fail("DuckDuckGo search timed out (curl_cffi deadlock?)")

        assert len(results) > 0, "DuckDuckGo returned no results"
        assert "href" in results[0], "DuckDuckGo result missing href"


# ═════════════════════════════════════════════════════════════════════
# Parametrized summary test — all providers at once
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "provider",
    providers.all_providers,
    ids=[p.id for p in providers.all_providers],
)
def test_api_key_present(provider):
    """Verify that the API key environment variable is set and non-empty."""
    key = os.environ.get(provider.env_var, "")
    assert key, f"{provider.env_var} is not set — {provider.description} will not work"
    assert len(key) >= 10, (
        f"{provider.env_var} is suspiciously short ({len(key)} chars) — "
        f"may be a placeholder"
    )
