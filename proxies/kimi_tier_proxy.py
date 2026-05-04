#!/usr/bin/env python3
"""Kimi Tier Race Proxy — Kimi K2.6 orchestrated tier racing with Letta memory.

Replaces tier_chooser_proxy.py with a Kimi K2.6 orchestrator that:
  - Calls specialist models as tools (the "race")
  - Uses Letta as a memory layer (core blocks + recall + per-conversation SQLite)
  - Has external data services (Brave, Tavily, Wikipedia) for fresh information
  - Handles remote API errors with structured categories and progress streaming
  - Maintains the same external OpenAI-compatible API surface

Port: 9901 (configurable via KIMI_TIER_PROXY_PORT)
"""

import asyncio
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from shared import (
    INGEST_DB_PATH,
    RequestTracker,
    create_app,
    env_int,
    http_client,
    is_utility_request,
    make_sse_chunk,
    register_ingest_routes,
    register_standard_routes,
    setup_logging,
    stream_passthrough,
)
import knowledge_client
from search_providers import _search_searxng
from media_enrichment import (
    enrich_with_media_structured,
    fetch_transcript_for_video,
    search_youtube_videos,
    _extract_youtube_id as media_extract_yt_id,
    MEDIA_ENRICHMENT_MAX_VIDEOS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.getenv("KIMI_TIER_PROXY_LOG_DIR", "/opt/kimi_tier_logs")
log = setup_logging("kimi-tier-proxy", LOG_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LISTEN_PORT = env_int("KIMI_TIER_PROXY_PORT", 9901, minimum=1)
INGEST_DB = os.getenv("INGEST_DB", INGEST_DB_PATH)

MAX_CONCURRENT_MODELS = env_int("KIMI_TIER_MAX_CONCURRENT", 10, minimum=1)
MODEL_TIMEOUT = int(os.getenv("KIMI_TIER_MODEL_TIMEOUT", "90"))

# Kimi orchestrator
KIMI_MODEL = os.getenv("KIMI_TIER_ORCHESTRATOR_MODEL", "moonshotai/kimi-k2.6")
KIMI_API_KEY = os.environ.get("MOONSHOT_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_TIER_BASE_URL", "https://api.moonshot.ai/v1")

# External data services
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Letta memory
MEMORY_DIR = os.getenv("KIMI_TIER_MEMORY_DIR", "/opt/kimi_tier_memory")
MEMORY_DB_DIR = os.path.join(MEMORY_DIR, "conversations")

# Media enrichment
IMAGE_ENRICHMENT_ENABLED = os.getenv("KIMI_TIER_IMAGE_ENRICHMENT", "true").lower() in ("1", "true", "yes")

# Deployment environment
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "staging").lower()

# ---------------------------------------------------------------------------
# Provider Registry — route models to their native APIs
# ---------------------------------------------------------------------------
PROVIDER_REGISTRY: dict[str, dict[str, str]] = {
    "openai":       {"base_url": "https://api.openai.com/v1",                                    "key_env": "OPENAI_API_KEY"},
    "anthropic":    {"base_url": "https://api.anthropic.com/v1",                                   "key_env": "ANTHROPIC_API_KEY"},
    "google":       {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",       "key_env": "GEMINI_API_KEY"},
    "x-ai":         {"base_url": "https://api.x.ai/v1",                                          "key_env": "XAI_API_KEY"},
    "deepseek":     {"base_url": "https://api.deepseek.com",                                     "key_env": "DEEPSEEK_API_KEY"},
    "mistralai":    {"base_url": "https://api.mistral.ai/v1",                                    "key_env": "MISTRAL_NATIVE_API_KEY"},
    "qwen":         {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",       "key_env": "DASHSCOPE_API_KEY"},
    "moonshotai":   {"base_url": "https://api.moonshot.ai/v1",                                   "key_env": "MOONSHOT_API_KEY"},
    "minimax":      {"base_url": "https://api.minimax.chat/v1",                                  "key_env": "MINIMAX_API_KEY"},
    "stepfun":      {"base_url": "https://api.stepfun.com/v1",                                   "key_env": "STEPFUN_API_KEY"},
    "z-ai":         {"base_url": "https://open.bigmodel.cn/api/paas/v4",                         "key_env": "ZHIPU_API_KEY"},
    "nvidia":       {"base_url": "https://integrate.api.nvidia.com/v1",                          "key_env": "NVIDIA_API_KEY"},
    "venice":       {"base_url": "https://api.venice.ai/api/v1",                                 "key_env": "VENICE_API_KEY"},
    "nous-research": {"base_url": "https://api.nousresearch.com/v1",                             "key_env": "NOUS_API_KEY"},
}

NATIVE_MODEL_MAP: dict[str, str] = {
    # xAI
    "grok-4.3":             "grok-4-3",
    # Mistral
    "mistral-large-3":       "mistral-large-latest",
    "mistral-medium-3.5":   "mistral-medium-3-5",
    # Moonshot Kimi
    "kimi-k2.6":            "kimi-k2.6",
    # Zhipu GLM
    "glm-5.1":              "glm-5.1",
    "glm-5-turbo":          "glm-5-turbo",
    # Google Gemini
    "gemini-3.1-flash":    "gemini-3-flash-preview",
    "gemini-3.1-pro":      "gemini-3.1-pro-preview",
    # OpenAI
    "gpt-5.5":              "gpt-5.5",
    "gpt-5.4-mini":        "gpt-5.4-mini",
    "gpt-oss-120b":         "gpt-oss-120b",
    # Anthropic
    "claude-haiku-4.6":    "claude-haiku-4-6-20251001",
    "claude-sonnet-4.6":   "claude-sonnet-4-6",
    "claude-opus-4.7":     "claude-opus-4-7",
    # Qwen/DashScope
    "qwen3.6-flash":       "qwen3.6-flash",
    "qwen3.6-max-preview":  "qwen3.6-max-preview",
    # DeepSeek
    "deepseek-v4-flash":    "deepseek-v4-flash",
    "deepseek-v4-pro":      "deepseek-v4-pro",
    "deepseek-v4-pro-max":  "deepseek-v4-pro",
    # MiniMax
    "minimax-m2.7":         "minimax-m2.7",
    # Xiaomi
    "mimo-v2-pro":          "mimo-v2-pro",
    "mimo-v2-flash":        "mimo-v2-flash",
    # Tencent
    "hy3-preview":          "hy3-preview",
    # NVIDIA
    "nemotron-3-super":     "nemotron-3-super",
    # StepFun
    "step-3.5-flash":       "step-3.5-flash",
    # Ling
    "ling-2.6-1t":          "ling-2.6-1t",
    # Venice
    "venice-uncensored-1.2": "venice-uncensored-1.2",
    # Nous
    "hermes-4-70b":         "hermes-4-70b",
}


def resolve_provider(model: str) -> tuple[str, str, str]:
    """Resolve a model ID to (base_url, api_key, native_model_name)."""
    parts = model.split("/", 1)
    if len(parts) == 2:
        prefix, model_name = parts
        entry = PROVIDER_REGISTRY.get(prefix)
        if entry:
            key = os.environ.get(entry["key_env"], "")
            if key:
                native_name = NATIVE_MODEL_MAP.get(model_name, model_name)
                return entry["base_url"], key, native_name
    return OPENROUTER_BASE, OPENROUTER_KEY, model


log.info(
    f"Config: env={DEPLOYMENT_ENV}, port={LISTEN_PORT}, "
    f"max_concurrent={MAX_CONCURRENT_MODELS}, timeout={MODEL_TIMEOUT}s, "
    f"kimi_model={KIMI_MODEL}"
)

# ---------------------------------------------------------------------------
# Model Roster — May 2026
# ---------------------------------------------------------------------------
TIER_MODELS = {
    "quick": [
        "anthropic/claude-haiku-4.6",
        "google/gemini-3.1-flash",
        "openai/gpt-5.4-mini",
        "x-ai/grok-4.3",
        "deepseek/deepseek-v4-flash",
        "qwen/qwen3.6-flash",
        "inclusionai/ling-2.6-1t",
        "z-ai/glm-5-turbo",
        "nvidia/nemotron-3-super",
        "stepfun/step-3.5-flash",
        "openai/gpt-oss-120b",
        "xiaomi/mimo-v2-flash",
    ],
    "medium": [
        "anthropic/claude-sonnet-4.6",
        "google/gemini-3.1-pro",
        "openai/gpt-5.4",
        "x-ai/grok-4.3",
        "deepseek/deepseek-v4-pro",
        "qwen/qwen3.6-max-preview",
        "mistralai/mistral-large-3",
        "z-ai/glm-5.1",
        "minimax/minimax-m2.7",
        "inclusionai/ling-2.6-1t",
        "xiaomi/mimo-v2-pro",
        "tencent/hy3-preview",
    ],
    "full-throttle": [
        "anthropic/claude-opus-4.7",
        "google/gemini-3.1-pro",
        "openai/gpt-5.5",
        "x-ai/grok-4.3",
        "deepseek/deepseek-v4-pro-max",
        "qwen/qwen3.6-max-preview",
        "moonshotai/kimi-k2.6",
        "mistralai/mistral-medium-3.5",
        "z-ai/glm-5.1",
        "minimax/minimax-m2.7",
        "inclusionai/ling-2.6-1t",
        "xiaomi/mimo-v2-pro",
        "tencent/hy3-preview",
        "venice/venice-uncensored-1.2",
        "nous-research/hermes-4-70b",
    ],
}

ALL_TIER_MODELS: list[str] = []
for models in TIER_MODELS.values():
    ALL_TIER_MODELS.extend(models)

_background_tasks: set[asyncio.Task] = set()

# ============================================================================
# Remote API Error Management
# ============================================================================

class RemoteCallError(BaseModel):
    category: str = Field(description="Error category: rate_limited, auth_failed, timeout, server_error, content_filtered, context_overflow, model_not_found, quota_exceeded, network_error, stream_interrupted")
    message: str = Field(description="Human-readable error message")
    retry_after: Optional[int] = Field(default=None, description="Seconds to wait before retry (rate limits)")
    retryable: bool = Field(default=False, description="Whether Kimi should retry this call")
    permanent: bool = Field(default=False, description="Whether this provider is dead for the session")


class RemoteCallResult(BaseModel):
    provider: str
    model: str
    status: str = Field(description="success or error")
    content: Optional[str] = None
    error: Optional[RemoteCallError] = None
    elapsed_ms: int = 0
    tokens_used: Optional[dict] = None


# Refusal detection (reused from tier_chooser_proxy)
REFUSAL_PATTERNS = [
    re.compile(r"I (?:cannot|can't|won't|am not able to|am unable to|must decline to|have to decline)", re.I),
    re.compile(r"I'm (?:sorry|afraid|not able to)", re.I),
    re.compile(r"(?:against|violates?) (?:my|the) (?:guidelines|policies|terms|ethics|rules)", re.I),
    re.compile(r"I (?:don't|do not) (?:feel comfortable|think I should)", re.I),
    re.compile(r"I'm not (?:comfortable|going to|willing to|able to)", re.I),
]

def is_refusal(content: str) -> bool:
    for pattern in REFUSAL_PATTERNS:
        if pattern.search(content):
            return True
    return False


# Provider-specific body tweaks
_MAX_COMPLETION_TOKENS_PREFIXES = {"openai"}
_NO_CUSTOM_TEMPERATURE_MODELS = {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini"}
_FORCE_STREAM_MODELS = {"glm-5", "qwq-plus", "qwen3-max-thinking"}


def _build_body(
    native_model: str,
    messages: list[dict],
    provider_prefix: str,
    *,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict:
    body: dict = {
        "model": native_model,
        "messages": messages,
        "stream": stream,
    }
    bare_model = native_model.split("/")[-1]
    if bare_model not in _NO_CUSTOM_TEMPERATURE_MODELS:
        body["temperature"] = temperature
    if provider_prefix in _MAX_COMPLETION_TOKENS_PREFIXES:
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
    return body


def _extract_content(message: dict) -> str:
    content = message.get("content", "") or ""
    if content.strip():
        return content
    return message.get("reasoning_content", "") or ""


async def call_model_structured(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
) -> RemoteCallResult:
    """Call a single model with structured error reporting."""
    start = time.monotonic()
    base_url, api_key, native_model = resolve_provider(model)
    is_openrouter = (base_url == OPENROUTER_BASE)
    provider_prefix = "" if is_openrouter else (model.split("/", 1)[0] if "/" in model else "")
    provider_label = "OpenRouter" if is_openrouter else base_url.split("//")[1].split("/")[0]

    # Force-stream for models that need it
    bare = model.split("/")[-1] if "/" in model else model
    if bare in _FORCE_STREAM_MODELS:
        chunks: list[str] = []
        try:
            async for chunk in stream_model(model, messages, temperature=temperature, max_tokens=max_tokens, req_id=req_id, content_only=True):
                chunks.append(chunk)
            text = "".join(chunks)
            elapsed = int((time.monotonic() - start) * 1000)
            if not text:
                return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="server_error", message="Stream returned empty content", retryable=False, permanent=False), elapsed_ms=elapsed)
            if is_refusal(text):
                return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="content_filtered", message="Model refused to answer", retryable=False, permanent=False), elapsed_ms=elapsed)
            return RemoteCallResult(provider=provider_label, model=model, status="success", content=text, elapsed_ms=elapsed)
        except asyncio.TimeoutError:
            elapsed = int((time.monotonic() - start) * 1000)
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="timeout", message=f"Timed out after {MODEL_TIMEOUT}s", retryable=False, permanent=False), elapsed_ms=elapsed)
        except Exception as e:
            elapsed = int((time.monotonic() - start) * 1000)
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="network_error", message=str(e), retryable=True, permanent=False), elapsed_ms=elapsed)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://deep-search.uk"
        headers["X-Title"] = "Kimi Tier Race Proxy"

    body = _build_body(native_model, messages, provider_prefix, temperature=temperature, max_tokens=max_tokens, stream=False)
    client = http_client()

    try:
        resp = await asyncio.wait_for(
            client.post(f"{base_url}/chat/completions", json=body, headers=headers),
            timeout=MODEL_TIMEOUT,
        )
        elapsed = int((time.monotonic() - start) * 1000)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", "30"))
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="rate_limited", message=f"Rate limited, retry after {retry_after}s", retry_after=retry_after, retryable=True, permanent=False), elapsed_ms=elapsed)

        if resp.status_code in (401, 403):
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="auth_failed", message=f"Authentication failed (HTTP {resp.status_code})", retryable=False, permanent=True), elapsed_ms=elapsed)

        if resp.status_code == 404:
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="model_not_found", message=f"Model not found (HTTP 404)", retryable=False, permanent=True), elapsed_ms=elapsed)

        if resp.status_code in (402,) and "quota" in resp.text.lower():
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="quota_exceeded", message="Quota exceeded", retryable=False, permanent=True), elapsed_ms=elapsed)

        if resp.status_code == 400 and "context" in resp.text.lower():
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="context_overflow", message="Context window exceeded", retryable=False, permanent=False), elapsed_ms=elapsed)

        if resp.status_code >= 500:
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="server_error", message=f"Server error (HTTP {resp.status_code})", retryable=True, permanent=False), elapsed_ms=elapsed)

        if resp.status_code != 200:
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="server_error", message=f"Unexpected HTTP {resp.status_code}: {resp.text[:200]}", retryable=False, permanent=False), elapsed_ms=elapsed)

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="server_error", message="Empty choices array", retryable=False, permanent=False), elapsed_ms=elapsed)

        msg = choices[0].get("message", {})
        text = _extract_content(msg)
        if not text:
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="server_error", message="Empty content in response", retryable=False, permanent=False), elapsed_ms=elapsed)

        if is_refusal(text):
            return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="content_filtered", message="Model refused to answer", retryable=False, permanent=False), elapsed_ms=elapsed)

        usage = data.get("usage", {})
        tokens = {"prompt": usage.get("prompt_tokens", 0), "completion": usage.get("completion_tokens", 0)} if usage else None

        return RemoteCallResult(provider=provider_label, model=model, status="success", content=text, elapsed_ms=elapsed, tokens_used=tokens)

    except asyncio.TimeoutError:
        elapsed = int((time.monotonic() - start) * 1000)
        return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="timeout", message=f"Timed out after {MODEL_TIMEOUT}s", retryable=False, permanent=False), elapsed_ms=elapsed)
    except ConnectionError as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="network_error", message=str(e), retryable=True, permanent=False), elapsed_ms=elapsed)
    except Exception as e:
        elapsed = int((time.monotonic() - start) * 1000)
        return RemoteCallResult(provider=provider_label, model=model, status="error", error=RemoteCallError(category="network_error", message=f"{type(e).__name__}: {e}", retryable=True, permanent=False), elapsed_ms=elapsed)


async def call_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    error_out: list | None = None,
) -> str:
    """Backward-compatible wrapper: returns text or empty string."""
    result = await call_model_structured(model, messages, temperature=temperature, max_tokens=max_tokens, req_id=req_id)
    if result.status == "success":
        return result.content or ""
    if error_out is not None and result.error:
        error_out.append(f"[{result.error.category.upper()}] {result.provider}: {result.error.message}")
    return ""


async def stream_model(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    req_id: str = "",
    content_only: bool = False,
) -> AsyncGenerator[str, None]:
    """Stream a single model's response."""
    base_url, api_key, native_model = resolve_provider(model)
    is_openrouter = (base_url == OPENROUTER_BASE)
    provider_prefix = "" if is_openrouter else (model.split("/", 1)[0] if "/" in model else "")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        headers["HTTP-Referer"] = "https://deep-search.uk"
        headers["X-Title"] = "Kimi Tier Race Proxy"

    body = _build_body(native_model, messages, provider_prefix, temperature=temperature, max_tokens=max_tokens, stream=True)
    client = http_client()
    try:
        async with client.stream("POST", f"{base_url}/chat/completions", json=body, headers=headers, timeout=MODEL_TIMEOUT) as resp:
            if resp.status_code != 200:
                error_text = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                log.warning(f"[{req_id}] stream {model} error {resp.status_code}: {error_text}")
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if content_only:
                        text_chunk = delta.get("content", "")
                    else:
                        text_chunk = delta.get("content", "") or delta.get("reasoning_content", "")
                    if text_chunk:
                        yield text_chunk
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.error(f"[{req_id}] stream {model} exception: {e}")
        raise


# ============================================================================
# Letta Memory Layer — per-conversation SQLite
# ============================================================================

def _conversation_db_path(conversation_id: str) -> Path:
    """Get the SQLite path for a conversation."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", conversation_id)
    return Path(MEMORY_DB_DIR) / f"{safe_id}.db"


def _init_conversation_db(db_path: Path) -> None:
    """Initialize the conversation SQLite schema."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS race_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            tier TEXT NOT NULL,
            score INTEGER DEFAULT 0,
            latency_ms INTEGER DEFAULT 0,
            winner INTEGER DEFAULT 0,
            error_category TEXT,
            content_preview TEXT
        );
        CREATE TABLE IF NOT EXISTS errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT,
            retryable INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS model_stats (
            model TEXT PRIMARY KEY,
            win_count INTEGER DEFAULT 0,
            total_calls INTEGER DEFAULT 0,
            avg_latency_ms INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            last_success TEXT
        );
        CREATE TABLE IF NOT EXISTS user_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            turn INTEGER,
            original_response TEXT,
            correction TEXT,
            reason TEXT
        );
        CREATE TABLE IF NOT EXISTS core_memory (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS recall_fts USING fts5(
            content, tokens='porter unicode61'
        );
    """)
    conn.commit()
    conn.close()


def get_or_create_conversation_db(conversation_id: str) -> Path:
    """Get the DB path, creating the schema if needed."""
    db_path = _conversation_db_path(conversation_id)
    if not db_path.exists():
        Path(MEMORY_DB_DIR).mkdir(parents=True, exist_ok=True)
        _init_conversation_db(db_path)
    return db_path


def store_race_result(conversation_id: str, model: str, tier: str, result: RemoteCallResult, winner: bool = False) -> None:
    """Store a race result in the conversation DB (fire-and-forget)."""
    try:
        db_path = get_or_create_conversation_db(conversation_id)
        conn = sqlite3.connect(str(db_path))
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO race_results (timestamp, model, tier, score, latency_ms, winner, error_category, content_preview) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, model, tier, 0 if result.status == "error" else 1, result.elapsed_ms, 1 if winner else 0, result.error.category if result.error else None, (result.content or "")[:500])
        )
        # Update model stats
        conn.execute(
            "INSERT INTO model_stats (model, total_calls, avg_latency_ms, error_count, last_success) VALUES (?, 1, ?, ?, ?) "
            "ON CONFLICT(model) DO UPDATE SET total_calls = total_calls + 1, avg_latency_ms = (avg_latency_ms * (total_calls - 1) + ?) / total_calls, "
            "error_count = error_count + ?, last_success = CASE WHEN ? = 'success' THEN ? ELSE last_success END",
            (model, result.elapsed_ms, 0 if result.status == "success" else 1, now if result.status == "success" else None,
             result.elapsed_ms, 0 if result.status == "success" else 1, result.status, now if result.status == "success" else None)
        )
        if winner and result.status == "success":
            conn.execute("UPDATE model_stats SET win_count = win_count + 1 WHERE model = ?", (model,))
        if result.error:
            conn.execute(
                "INSERT INTO errors (timestamp, provider, model, category, message, retryable, permanent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now, result.provider, model, result.error.category, result.error.message, int(result.error.retryable), int(result.error.permanent))
            )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to store race result for {conversation_id}: {e}")


def get_core_memory(conversation_id: str) -> dict[str, str]:
    """Read all core memory blocks for a conversation."""
    try:
        db_path = get_or_create_conversation_db(conversation_id)
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT key, value FROM core_memory").fetchall()
        conn.close()
        return dict(rows)
    except Exception:
        return {}


def update_core_memory(conversation_id: str, key: str, value: str) -> None:
    """Update a core memory block."""
    try:
        db_path = get_or_create_conversation_db(conversation_id)
        conn = sqlite3.connect(str(db_path))
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO core_memory (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?",
            (key, value, now, value, now)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to update core memory for {conversation_id}: {e}")


def recall_search(conversation_id: str, query: str, n: int = 5) -> list[str]:
    """Search across conversation history using FTS5."""
    try:
        db_path = get_or_create_conversation_db(conversation_id)
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT content FROM recall_fts WHERE content MATCH ? ORDER BY rank LIMIT ?",
            (query, n)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def store_recall(conversation_id: str, content: str) -> None:
    """Store content in the recall FTS index."""
    try:
        db_path = get_or_create_conversation_db(conversation_id)
        conn = sqlite3.connect(str(db_path))
        conn.execute("INSERT INTO recall_fts (content) VALUES (?)", (content,))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to store recall for {conversation_id}: {e}")


def get_model_status_block(conversation_id: str) -> str:
    """Build a MODEL_STATUS core memory block from recent errors."""
    try:
        db_path = get_or_create_conversation_db(conversation_id)
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT model, COUNT(*) as cnt, GROUP_CONCAT(DISTINCT category) as cats "
            "FROM errors WHERE timestamp > datetime('now', '-24 hours') GROUP BY model HAVING cnt > 0"
        ).fetchall()
        conn.close()
        if not rows:
            return "All models healthy"
        lines = []
        for model, cnt, cats in rows:
            lines.append(f"- {model}: {cnt} error(s) in last 24h ({cats})")
        return "\n".join(lines)
    except Exception:
        return "Status unavailable"


# ============================================================================
# External Data Services — Brave, Tavily, Wikipedia
# ============================================================================

async def brave_search(query: str, count: int = 5) -> list[dict]:
    """Search the web via Brave Search API."""
    if not BRAVE_API_KEY:
        return [{"error": "Brave API key not configured"}]
    client = http_client()
    try:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": count},
            headers={"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"},
        )
        if resp.status_code != 200:
            return [{"error": f"Brave returned HTTP {resp.status_code}"}]
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", [])[:count]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            })
        return results
    except Exception as e:
        return [{"error": f"Brave search failed: {e}"}]


async def tavily_extract(url: str) -> str:
    """Extract clean text from a URL via Tavily."""
    if not TAVILY_API_KEY:
        return "[Tavily API key not configured]"
    client = http_client()
    try:
        resp = await client.post(
            "https://api.tavily.com/extract",
            json={"urls": [url], "extract_depth": "basic"},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {TAVILY_API_KEY}"},
        )
        if resp.status_code != 200:
            return f"[Tavily extract failed: HTTP {resp.status_code}]"
        data = resp.json()
        results = data.get("results", [])
        if results:
            return results[0].get("raw_content", "")[:10000]
        return "[No content extracted]"
    except Exception as e:
        return f"[Tavily extract failed: {e}]"


async def wikipedia_lookup(query: str) -> str:
    """Look up a topic on Wikipedia."""
    client = http_client()
    try:
        # Search for the article
        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
        )
        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return "[No Wikipedia article found]"
        page_id = search_results[0]["pageid"]
        # Get the summary
        summary_resp = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{search_results[0]['title'].replace(' ', '_')}"
        )
        if summary_resp.status_code == 200:
            summary_data = summary_resp.json()
            return summary_data.get("extract", "[No extract available]")
        return "[Wikipedia lookup failed]"
    except Exception as e:
        return f"[Wikipedia lookup failed: {e}]"


# ============================================================================
# Kimi K2.6 Orchestrator — direct API tool-call loop
# ============================================================================

_KIMI_SYSTEM_PROMPT = """You are Kimi K2.6, an orchestrator that coordinates multiple specialist AI models to produce the best possible answer.

You have access to these tools:
1. **call_model** — Call a specialist model with a prompt and get its response
2. **web_search** — Search the web for current information (Brave Search)
3. **extract_url** — Extract clean text from a URL (Tavily)
4. **wikipedia_lookup** — Look up factual information on Wikipedia
5. **recall_memory** — Search past conversation history for relevant context
6. **update_core_memory** — Update persistent memory that persists across turns

Your orchestration strategy:
- For fresh/time-sensitive information, call data services FIRST, then inject the results as context when calling specialist models
- Call at least 3 specialist models for non-trivial queries; call all models for full-throttle queries
- After receiving model responses, synthesize the best answer by combining unique insights from each model
- If a model returns an error, note it and proceed with other models
- If a model's response is a refusal, weight other models' responses higher
- Store important findings in core memory for future turns

When synthesizing:
- Extract and combine ALL unique facts, details, examples, code snippets, URLs, and actionable information
- If models disagree, present both perspectives with reasoning
- Preserve specific numbers, dates, names, URLs, code blocks, and technical details
- Do NOT add hedging, disclaimers, or meta-commentary about the synthesis process
- Do NOT mention that multiple models were consulted
- The final answer must read as a single authoritative response
- Prioritize depth, specificity, and completeness over brevity
- If one model provides a unique insight that others missed, ALWAYS include it"""


def _build_kimi_tools(tier: str) -> list[dict]:
    """Build tool definitions for Kimi's tool-calling."""
    models = TIER_MODELS.get(tier, [])
    model_names = [m.split("/")[-1] for m in models]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "call_model",
                "description": f"Call a specialist AI model. Available models for this tier: {', '.join(model_names)}. The model will respond to the prompt you provide. You can inject context from web_search or extract_url results into the prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "enum": model_names,
                            "description": "The model to call",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to send to the model. Include any fresh data from web_search/extract_url as context.",
                        },
                    },
                    "required": ["model_name", "prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information. Use when you need fresh data, recent events, or facts you're uncertain about.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "count": {"type": "integer", "default": 5, "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_url",
                "description": "Extract clean text content from a URL. Use when you need to read a specific page found via web_search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to extract content from"},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "wikipedia_lookup",
                "description": "Look up a topic on Wikipedia. Use for factual, encyclopedic information. Fast and free.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Topic to look up"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall_memory",
                "description": "Search past conversation history for relevant context. Returns matching conversation snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "n": {"type": "integer", "default": 5, "description": "Number of results"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_core_memory",
                "description": "Update a persistent memory block that will be visible in future turns. Use for noting model performance, user preferences, or important facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Memory block key (e.g. 'user_preferences', 'model_status')"},
                        "value": {"type": "string", "description": "Memory block content"},
                    },
                    "required": ["key", "value"],
                },
            },
        },
    ]
    return tools


def _resolve_model_from_name(model_name: str, tier: str) -> str:
    """Resolve a short model name back to the full model ID."""
    for full_id in TIER_MODELS.get(tier, []):
        if full_id.split("/")[-1] == model_name:
            return full_id
    return model_name


async def execute_kimi_tool(
    tool_name: str,
    tool_args: dict,
    tier: str,
    messages: list[dict],
    conversation_id: str,
    req_id: str,
) -> str:
    """Execute a tool call from Kimi and return the result."""
    if tool_name == "call_model":
        model_name = tool_args.get("model_name", "")
        prompt = tool_args.get("prompt", "")
        full_model = _resolve_model_from_name(model_name, tier)

        # Build messages for the specialist model: system prompt + user prompt
        model_messages = [
            {"role": "system", "content": "You are a specialist AI model. Provide the most comprehensive, accurate, and detailed response possible. Include specific facts, numbers, code examples, and actionable information."},
            {"role": "user", "content": prompt},
        ]

        result = await call_model_structured(full_model, model_messages, temperature=0.7, max_tokens=4096, req_id=req_id)

        # Store result in conversation DB
        store_race_result(conversation_id, full_model, tier, result)

        if result.status == "success":
            # Store in recall for future searches
            store_recall(conversation_id, f"[{model_name}] {result.content[:2000]}")
            return result.content or ""
        elif result.error:
            emoji = {
                "rate_limited": "⚠️", "auth_failed": "🔴", "timeout": "⏱️",
                "server_error": "🔴", "content_filtered": "🚫", "context_overflow": "📏",
                "model_not_found": "🔴", "quota_exceeded": "💰", "network_error": "🌐",
                "stream_interrupted": "⚠️",
            }.get(result.error.category, "❌")
            return f"{emoji} [{model_name}] {result.error.category}: {result.error.message}"
        return f"[{model_name}] Unknown error"

    elif tool_name == "web_search":
        results = await brave_search(tool_args.get("query", ""), tool_args.get("count", 5))
        if results and "error" not in results[0]:
            formatted = []
            for r in results:
                formatted.append(f"- **{r.get('title', '')}** ({r.get('url', '')})\n  {r.get('description', '')}")
            return "\n\n".join(formatted)
        return f"Web search error: {results[0].get('error', 'Unknown')}" if results else "No results"

    elif tool_name == "extract_url":
        return await tavily_extract(tool_args.get("url", ""))

    elif tool_name == "wikipedia_lookup":
        return await wikipedia_lookup(tool_args.get("query", ""))

    elif tool_name == "recall_memory":
        results = recall_search(conversation_id, tool_args.get("query", ""), tool_args.get("n", 5))
        if results:
            return "\n---\n".join(results)
        return "No relevant memories found"

    elif tool_name == "update_core_memory":
        update_core_memory(conversation_id, tool_args.get("key", ""), tool_args.get("value", ""))
        return "Core memory updated"

    return f"Unknown tool: {tool_name}"


async def run_kimi_orchestration(
    tier: str,
    messages: list[dict],
    user_query: str,
    req_id: str,
    conversation_id: str = "",
) -> AsyncGenerator[str, None]:
    """Run the Kimi K2.6 orchestration loop with tool calling."""
    request_id = f"chatcmpl-kimi-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = f"kimi-tier-race-{tier}"

    def _chunk(content: str, finish_reason=None, reasoning_content: str = None):
        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data)}\n\n"

    if not conversation_id:
        conversation_id = f"conv-{uuid.uuid4().hex[:12]}"

    # Initialize conversation DB
    get_or_create_conversation_db(conversation_id)

    # Build core memory context for Kimi's system prompt
    core_mem = get_core_memory(conversation_id)
    model_status = get_model_status_block(conversation_id)
    memory_context = ""
    if core_mem:
        memory_context += "\n\n[CURRENT MEMORY]\n"
        for k, v in core_mem.items():
            memory_context += f"[{k}]\n{v}\n\n"
    memory_context += f"\n[MODEL_STATUS]\n{model_status}"

    system_prompt = _KIMI_SYSTEM_PROMPT + memory_context

    # Build Kimi's messages
    kimi_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    tools = _build_kimi_tools(tier)
    client = http_client()

    # Fire media enrichment concurrently (not for quick tier)
    is_quick_tier = (tier == "quick")
    media_task = None
    if not is_quick_tier:
        media_task = asyncio.create_task(enrich_with_media_structured(user_query, req_id))

    # Kimi orchestration loop — max 15 tool-call rounds
    MAX_ROUNDS = 15
    for round_num in range(MAX_ROUNDS):
        # Call Kimi
        headers = {
            "Authorization": f"Bearer {KIMI_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "kimi-k2.6",
            "messages": kimi_messages,
            "tools": tools,
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 8192,
        }

        # Collect Kimi's response — both text and tool calls
        collected_text = ""
        tool_calls = []
        current_tool_call = None
        finish_reason = None

        try:
            async with client.stream(
                "POST",
                f"{KIMI_BASE_URL}/chat/completions",
                json=body,
                headers=headers,
                timeout=120,
            ) as resp:
                if resp.status_code != 200:
                    error_text = (await resp.aread()).decode("utf-8", errors="replace")[:500]
                    log.error(f"[{req_id}] Kimi API error {resp.status_code}: {error_text}")
                    yield _chunk(f"Orchestrator error: HTTP {resp.status_code}", finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        # Collect text content
                        text = delta.get("content", "")
                        if text:
                            collected_text += text

                        # Collect tool calls
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx >= len(tool_calls):
                                    tool_calls.append({"id": tc.get("id", ""), "function": {"name": "", "arguments": ""}})
                                if tc.get("id"):
                                    tool_calls[idx]["id"] = tc["id"]
                                if "function" in tc:
                                    if tc["function"].get("name"):
                                        tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                                    if tc["function"].get("arguments"):
                                        tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            log.error(f"[{req_id}] Kimi stream error: {e}")
            yield _chunk(f"Orchestrator stream error: {e}", finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        # Stream Kimi's reasoning as reasoning_content
        if collected_text:
            yield _chunk("", reasoning_content=f"[Kimi round {round_num + 1}]\n{collected_text[:500]}\n")

        # Add Kimi's assistant message to conversation
        assistant_msg = {"role": "assistant", "content": collected_text or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        if not assistant_msg["content"] and not tool_calls:
            # Kimi returned nothing — we're done
            break
        kimi_messages.append(assistant_msg)

        # If no tool calls, Kimi is done — stream the final text
        if not tool_calls or finish_reason == "stop":
            # Store in recall
            store_recall(conversation_id, collected_text[:3000])

            # Collect media
            media_items = []
            if media_task is not None:
                try:
                    media_items = await asyncio.wait_for(media_task, timeout=8.0)
                except Exception:
                    media_items = []

            # For quick tier, just return the synthesis
            if is_quick_tier or not media_items:
                yield _chunk(collected_text, finish_reason="stop")
            else:
                # Inject media into the synthesis (reuse existing media injection)
                from tier_chooser_proxy import _inject_media_into_text
                final_text = await _inject_media_into_text(collected_text, media_items, req_id)
                yield _chunk(final_text if final_text else collected_text, finish_reason="stop")

            yield "data: [DONE]\n\n"
            return

        # Execute tool calls
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                func_args = {}

            log.info(f"[{req_id}] Kimi tool call: {func_name}({json.dumps(func_args)[:200]})")
            yield _chunk("", reasoning_content=f"  → {func_name}({json.dumps(func_args)[:100]}...)\n")

            result = await execute_kimi_tool(func_name, func_args, tier, messages, conversation_id, req_id)

            # Add tool result to conversation
            kimi_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result[:8000],  # Truncate to avoid context overflow
            })

            yield _chunk("", reasoning_content=f"  ← {func_name}: {result[:100]}...\n")

    # If we hit max rounds, get a final synthesis from Kimi
    log.warning(f"[{req_id}] Kimi hit max {MAX_ROUNDS} rounds — forcing final synthesis")
    final_body = {
        "model": "kimi-k2.6",
        "messages": kimi_messages + [{"role": "user", "content": "Please provide your final synthesized answer now. Do not make any more tool calls."}],
        "temperature": 0.3,
        "max_tokens": 8192,
        "stream": False,
    }
    try:
        resp = await asyncio.wait_for(
            client.post(f"{KIMI_BASE_URL}/chat/completions", json=final_body, headers=headers),
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text:
                yield _chunk(text, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return
    except Exception as e:
        log.error(f"[{req_id}] Final synthesis failed: {e}")

    yield _chunk("Orchestrator reached maximum rounds without completing synthesis.", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================================
# Fallback: Parallel race (when Kimi is unavailable)
# ============================================================================

async def run_parallel_race(
    tier: str,
    messages: list[dict],
    user_query: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Fallback parallel race — same as tier_chooser_proxy logic."""
    request_id = f"chatcmpl-race-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model_id = f"tier-race-{tier}"

    def _chunk(content: str, finish_reason=None, reasoning_content: str = None):
        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(data)}\n\n"

    models = TIER_MODELS.get(tier, [])
    if not models:
        yield _chunk(f"Unknown tier: {tier}", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    is_quick_tier = (tier == "quick")
    media_task = None
    if not is_quick_tier:
        media_task = asyncio.create_task(enrich_with_media_structured(user_query, req_id))

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_MODELS)

    async def query_model(model_name: str) -> RemoteCallResult:
        async with semaphore:
            return await call_model_structured(
                model_name, messages,
                temperature=0.7, max_tokens=4096, req_id=req_id,
            )

    tasks = [asyncio.create_task(query_model(m)) for m in models]
    results: list[RemoteCallResult] = []
    completed = 0

    yield _chunk("", reasoning_content=f"PARALLEL RACE [{tier.upper()}] — racing {len(models)} models...\n")

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        short_model = result.model.split("/")[-1]
        if result.status == "error" and result.error:
            status = f"{result.error.category.upper()}"
        elif result.status == "success" and is_refusal(result.content or ""):
            status = "REFUSAL"
        else:
            status = "OK"
        log.info(f"[{req_id}] [{completed}/{len(models)}] {short_model}: {status}")
        yield _chunk("", reasoning_content=f"  [{completed}/{len(models)}] {short_model}: {status}\n")

    valid = [r for r in results if r.status == "success" and r.content and not is_refusal(r.content)]
    if not valid:
        error_details = []
        for r in results:
            short = r.model.split("/")[-1]
            if r.error:
                error_details.append(f"  • {short}: {r.error.category} — {r.error.message}")
            elif is_refusal(r.content or ""):
                error_details.append(f"  • {short}: refused to answer")
        msg = f"All {len(models)} models in the {tier} tier failed\n\nDetails:\n" + "\n".join(error_details)
        msg += "\n\nTry rephrasing your question."
        if media_task is not None:
            media_task.cancel()
        yield _chunk(msg, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Quick tier: return the longest valid response
    if is_quick_tier:
        best = max(valid, key=lambda r: len(r.content or ""))
        yield _chunk(best.content, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Synthesis for medium/full-throttle
    # Use Kimi as the synthesizer if available, otherwise use the first valid model
    synthesis_model = KIMI_MODEL if KIMI_API_KEY else "google/gemini-3.1-flash"
    response_parts = []
    for r in valid:
        short = r.model.split("/")[-1]
        response_parts.append(f"--- {short} ---\n{r.content}\n")
    responses_text = "\n".join(response_parts)

    synthesis_prompt = (
        "You are a synthesis engine. You have received multiple responses from different AI models "
        "answering the same question. Produce a single comprehensive answer that captures the maximum "
        "wealth of information from ALL responses.\n\n"
        "Rules:\n"
        "- Extract and combine ALL unique facts, details, examples, code snippets, URLs, and actionable information\n"
        "- If models disagree, present both perspectives with reasoning\n"
        "- Preserve specific numbers, dates, names, URLs, code blocks, and technical details\n"
        "- Do NOT add hedging, disclaimers, or meta-commentary\n"
        "- Do NOT mention that multiple models were consulted\n"
        "- The final answer must read as a single authoritative response\n"
        "- Prioritize depth, specificity, and completeness over brevity"
    )

    synthesis_messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": f"User question: {user_query}\n\nModel responses:\n{responses_text}\n\nProduce the most comprehensive, information-rich answer possible."},
    ]

    yield _chunk("", reasoning_content=f"\n{len(valid)} model(s) responded. Synthesizing...\n")

    synthesised = await call_model(synthesis_model, synthesis_messages, temperature=0.3, max_tokens=8192, req_id=req_id)

    if not synthesised:
        fallback = max(valid, key=lambda r: len(r.content or ""))
        yield _chunk(fallback.content, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Collect media
    media_items = []
    if media_task is not None:
        try:
            media_items = await asyncio.wait_for(media_task, timeout=8.0)
        except Exception:
            media_items = []

    if media_items:
        from tier_chooser_proxy import _inject_media_into_text
        final_text = await _inject_media_into_text(synthesised, media_items, req_id)
        yield _chunk(final_text if final_text else synthesised, finish_reason="stop")
    else:
        yield _chunk(synthesised, finish_reason="stop")

    yield "data: [DONE]\n\n"


# ============================================================================
# Single-model streaming access
# ============================================================================

async def _stream_single_model(
    model: str,
    messages: list[dict],
    req_id: str,
) -> AsyncGenerator[str, None]:
    request_id = f"chatcmpl-tier-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    try:
        async for content_chunk in stream_model(model, messages, temperature=0.7, max_tokens=4096, req_id=req_id):
            yield make_sse_chunk(content_chunk, request_id=request_id, created=created, model_id=f"tier-{model}")
    except Exception as e:
        log.error(f"[{req_id}] {model} direct-stream error: {e}")
    yield make_sse_chunk("", request_id=request_id, created=created, model_id=f"tier-{model}", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================================
# /v1/models endpoint
# ============================================================================

_PRODUCTION_MODELS = [
    {
        "id": "kimi-tier-race-full-throttle",
        "object": "model",
        "created": 1700000000,
        "owned_by": "kimi-tier-proxy",
        "name": "Kimi Smart Combined",
    },
    {
        "id": "kimi-tier-race-medium",
        "object": "model",
        "created": 1700000000,
        "owned_by": "kimi-tier-proxy",
        "name": "Kimi Balanced",
    },
]

def build_model_list() -> list[dict]:
    models = list(_PRODUCTION_MODELS)
    if DEPLOYMENT_ENV != "production":
        for tier_name in TIER_MODELS:
            if tier_name == "full-throttle":
                continue  # Already in production models
            models.append({
                "id": f"kimi-tier-race-{tier_name}",
                "object": "model",
                "created": 1700000000,
                "owned_by": "kimi-tier-proxy",
                "name": f"Kimi Tier Race: {tier_name.replace('-', ' ').title()}",
            })
        for tier_name, tier_models in TIER_MODELS.items():
            for m in tier_models:
                models.append({
                    "id": f"kimi-tier-{m}",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "kimi-tier-proxy",
                    "name": f"[{tier_name.replace('-', ' ').title()}] {m}",
                })
    return models


# ============================================================================
# FastAPI app
# ============================================================================

app = create_app("Kimi Tier Race Proxy")
tracker = RequestTracker()

register_standard_routes(
    app,
    service_name="kimi-tier-proxy",
    log_dir=LOG_DIR,
    tracker=tracker,
    health_extras={"port": LISTEN_PORT, "env": DEPLOYMENT_ENV, "tiers": list(TIER_MODELS.keys()), "kimi_model": KIMI_MODEL},
)
register_ingest_routes(app, INGEST_DB, log)


@app.get("/v1/models")
async def list_models():
    return JSONResponse({"object": "list", "data": build_model_list()})


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    req_id = f"req-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {"message": f"Invalid request body: {e}", "type": "invalid_request"}})

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse(status_code=400, content={"error": {"message": "messages array is required", "type": "invalid_request"}})

    requested_model = body.get("model", "kimi-tier-race-quick")
    utility = is_utility_request(messages)

    log.info(f"[{req_id}] New request: model={requested_model}, messages={len(messages)}, utility={utility}")
    tracker.start(req_id, model=requested_model, messages=len(messages))

    # Extract user query
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            break

    if not user_query:
        tracker.finish(req_id)
        return JSONResponse(status_code=400, content={"error": {"message": "No user message found", "type": "invalid_request"}})

    # Extract conversation ID for memory (from custom header or generate)
    conversation_id = request.headers.get("X-Conversation-ID", f"conv-{uuid.uuid4().hex[:12]}")

    # Utility requests — passthrough to a cheap fast model
    if utility:
        client_wants_stream = body.get("stream", False)
        base_url, api_key, native_model = resolve_provider("google/gemini-3.1-flash")
        if not client_wants_stream:
            from shared import utility_passthrough_json
            result = await utility_passthrough_json(body, req_id=req_id, upstream_base=base_url, upstream_key=api_key, upstream_model=native_model, log=log)
            tracker.finish(req_id)
            return result
        generator = stream_passthrough(messages, body, req_id=req_id, upstream_base=base_url, upstream_key=api_key, upstream_model=native_model, model_id=requested_model, tracker=tracker, log=log)
    elif requested_model.startswith("kimi-tier-race-"):
        tier = requested_model.replace("kimi-tier-race-", "")
        if tier not in TIER_MODELS:
            tracker.finish(req_id)
            return JSONResponse(status_code=400, content={"error": {"message": f"Unknown tier: {tier}. Valid: {list(TIER_MODELS.keys())}", "type": "invalid_request"}})
        # Use Kimi orchestration if API key available, otherwise fallback to parallel race
        if KIMI_API_KEY:
            generator = run_kimi_orchestration(tier, messages, user_query, req_id, conversation_id=conversation_id)
        else:
            log.warning(f"[{req_id}] No Kimi API key — falling back to parallel race")
            generator = run_parallel_race(tier, messages, user_query, req_id)
    elif requested_model.startswith("kimi-tier-"):
        actual_model = requested_model[10:]  # strip "kimi-tier-"
        if actual_model not in ALL_TIER_MODELS:
            tracker.finish(req_id)
            return JSONResponse(status_code=400, content={"error": {"message": f"Unknown model: {actual_model}", "type": "invalid_request"}})
        generator = _stream_single_model(actual_model, messages, req_id)
    else:
        # Default to quick race
        if KIMI_API_KEY:
            generator = run_kimi_orchestration("quick", messages, user_query, req_id, conversation_id=conversation_id)
        else:
            generator = run_parallel_race("quick", messages, user_query, req_id)

    async def tracked_generator():
        try:
            async for chunk in generator:
                yield chunk
        finally:
            tracker.finish(req_id)

    return StreamingResponse(
        tracked_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting Kimi Tier Race Proxy on port {LISTEN_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
