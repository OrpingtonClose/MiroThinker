# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Backend risk classification for model endpoints.

Every model backend (local vLLM, OpenRouter free tier, paid API, etc.) carries
operational risk.  A free tier can be revoked mid-run.  A paid API can be
rate-limited.  A local GPU can OOM.  This module classifies backends by risk
and wraps the ``complete`` callable with protective behaviour:

    - Adaptive rate limiting (token bucket per risk tier)
    - Response caching (avoid re-firing identical prompts to fragile backends)
    - Retry with exponential backoff
    - Graceful degradation to a fallback backend
    - Telemetry (failure counts, throttle events, cache hits)

Risk tiers:

    SELF_HOSTED   — local vLLM / Ollama.  No external dependency.
                    Lowest risk.  No rate limiting.  No caching.
    PAID_API      — commercial API with quota.  Medium risk.
                    Moderate rate limiting.  Optional caching.
    FREE_TIER     — free API tier (OpenRouter :free, etc.).  High risk.
                    Aggressive rate limiting.  Mandatory caching.
                    Proactive result preservation.
    EXPERIMENTAL  — untested or beta endpoints.  Highest risk.
                    Strict rate limiting.  Mandatory caching.
                    Very conservative retry policy.

Usage:

    from swarm.backend import BackendConfig, BackendRiskTier, wrap_complete

    config = BackendConfig.free_tier(
        name="ling-2.6-1t-openrouter",
        model="inclusionai/ling-2.6-1t:free",
        api_base="https://openrouter.ai/api/v1",
    )

    safe_complete = wrap_complete(raw_complete_fn, config)
    # safe_complete is a drop-in replacement for any CompleteFn
    # — it throttles, caches, retries, and falls back automatically.

Vestigial dynamic allocation:
    The ``BackendConfig`` carries enough metadata (model name, API base,
    risk tier, rate limits) to support future dynamic model allocation
    across self-hosted and remote backends.  The ``BackendRegistry``
    stub is provided for that future use.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Type alias — same as swarm.config.CompleteFn
CompleteFn = Callable[[str], Awaitable[str]]


class BackendRiskTier(Enum):
    """Risk classification for model backends.

    Determines rate limiting, caching, and retry behaviour.
    """

    SELF_HOSTED = "self_hosted"
    PAID_API = "paid_api"
    FREE_TIER = "free_tier"
    EXPERIMENTAL = "experimental"


@dataclass
class BackendMetrics:
    """Runtime telemetry for a backend.

    Attributes:
        total_calls: Total calls attempted.
        cache_hits: Calls served from cache.
        retries: Total retry attempts across all calls.
        failures: Calls that failed after all retries.
        throttle_events: Times the rate limiter delayed a call.
        total_tokens_in: Approximate input tokens (if tracked).
        total_tokens_out: Approximate output tokens (if tracked).
    """

    total_calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    throttle_events: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of calls served from cache."""
        if self.total_calls == 0:
            return 0.0
        return self.cache_hits / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Fraction of calls that failed after retries."""
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls


@dataclass
class BackendConfig:
    """Configuration for a model backend.

    Attributes:
        name: Human-readable backend identifier for logging.
        model: Model identifier (e.g. "inclusionai/ling-2.6-1t:free").
        api_base: API endpoint URL.
        risk_tier: Risk classification driving protective behaviour.
        max_requests_per_minute: Rate limit ceiling.  0 = unlimited.
        min_request_interval_s: Minimum seconds between requests.
            Enforced even if RPM budget remains.  Prevents burst patterns
            that trigger provider-side abuse detection.
        max_retries: Retry attempts on transient failures (429, 500, timeout).
        retry_base_delay_s: Base delay for exponential backoff.
        enable_cache: Cache responses by prompt hash.
        cache_max_entries: Maximum cached responses (LRU eviction).
        fallback: Optional fallback backend for graceful degradation.
            When the primary fails after all retries, the request is
            forwarded to this backend's ``complete`` function.
        api_key_env: Environment variable name holding the API key.
        batch_concurrency: Maximum concurrent requests within a batch.
            Lower values reduce burst pressure on fragile backends.
    """

    name: str = "default"
    model: str = "default"
    api_base: str = "http://localhost:8000/v1"
    risk_tier: BackendRiskTier = BackendRiskTier.SELF_HOSTED
    max_requests_per_minute: int = 0
    min_request_interval_s: float = 0.0
    max_retries: int = 2
    retry_base_delay_s: float = 1.0
    enable_cache: bool = False
    cache_max_entries: int = 10000
    fallback: BackendConfig | None = None
    api_key_env: str = ""
    batch_concurrency: int = 20

    @classmethod
    def self_hosted(
        cls,
        name: str = "local-vllm",
        model: str = "default",
        api_base: str = "http://localhost:8000/v1",
    ) -> BackendConfig:
        """Create a self-hosted backend config (local vLLM / Ollama).

        No rate limiting, no caching, minimal retries.
        """
        return cls(
            name=name,
            model=model,
            api_base=api_base,
            risk_tier=BackendRiskTier.SELF_HOSTED,
            max_requests_per_minute=0,
            min_request_interval_s=0.0,
            max_retries=1,
            retry_base_delay_s=0.5,
            enable_cache=False,
            batch_concurrency=50,
        )

    @classmethod
    def paid_api(
        cls,
        name: str,
        model: str,
        api_base: str,
        *,
        rpm: int = 600,
        api_key_env: str = "",
        fallback: BackendConfig | None = None,
    ) -> BackendConfig:
        """Create a paid API backend config.

        Paid APIs are first-class external services — call them
        frequently and without shame.  The only protection is basic
        retry/timeout handling and a generous RPM ceiling to avoid
        accidental self-DoS.  No artificial throttling, no caching
        (responses are unique), high concurrency.
        """
        return cls(
            name=name,
            model=model,
            api_base=api_base,
            risk_tier=BackendRiskTier.PAID_API,
            max_requests_per_minute=rpm,
            min_request_interval_s=0.0,
            max_retries=3,
            retry_base_delay_s=1.0,
            enable_cache=False,
            api_key_env=api_key_env,
            batch_concurrency=40,
            fallback=fallback,
        )

    @classmethod
    def free_tier(
        cls,
        name: str,
        model: str,
        api_base: str,
        *,
        rpm: int = 20,
        api_key_env: str = "",
        fallback: BackendConfig | None = None,
    ) -> BackendConfig:
        """Create a free-tier backend config (OpenRouter :free, etc.).

        Aggressive rate limiting, mandatory caching, conservative retries.
        The system treats this backend as a gift that can be revoked —
        every successful response is cached, rate limits are strict,
        and a fallback is strongly recommended.
        """
        return cls(
            name=name,
            model=model,
            api_base=api_base,
            risk_tier=BackendRiskTier.FREE_TIER,
            max_requests_per_minute=rpm,
            min_request_interval_s=1.0,
            max_retries=2,
            retry_base_delay_s=5.0,
            enable_cache=True,
            cache_max_entries=50000,
            api_key_env=api_key_env,
            batch_concurrency=5,
            fallback=fallback,
        )

    @classmethod
    def experimental(
        cls,
        name: str,
        model: str,
        api_base: str,
        *,
        rpm: int = 10,
        api_key_env: str = "",
        fallback: BackendConfig | None = None,
    ) -> BackendConfig:
        """Create an experimental backend config.

        Strictest rate limiting, mandatory caching, very conservative.
        """
        return cls(
            name=name,
            model=model,
            api_base=api_base,
            risk_tier=BackendRiskTier.EXPERIMENTAL,
            max_requests_per_minute=rpm,
            min_request_interval_s=2.0,
            max_retries=1,
            retry_base_delay_s=10.0,
            enable_cache=True,
            cache_max_entries=50000,
            api_key_env=api_key_env,
            batch_concurrency=3,
            fallback=fallback,
        )


# ---------------------------------------------------------------------------
# Rate limiter — token bucket with minimum interval enforcement
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Token bucket rate limiter with minimum inter-request interval.

    Args:
        rpm: Maximum requests per minute.  0 = unlimited.
        min_interval_s: Minimum seconds between consecutive requests.
    """

    def __init__(self, rpm: int, min_interval_s: float) -> None:
        self._rpm = rpm
        self._min_interval_s = min_interval_s
        self._tokens = float(rpm) if rpm > 0 else float("inf")
        self._max_tokens = float(rpm) if rpm > 0 else float("inf")
        self._last_refill = time.monotonic()
        self._last_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Wait until a request is allowed.

        Returns:
            Seconds spent waiting (0.0 if no wait was needed).
        """
        waited = 0.0
        async with self._lock:
            now = time.monotonic()

            # Refill tokens based on elapsed time
            if self._rpm > 0:
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * (self._rpm / 60.0),
                )
                self._last_refill = now

            # Wait for token availability
            if self._rpm > 0 and self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / (self._rpm / 60.0)
                waited += wait_time
                await asyncio.sleep(wait_time)
                self._tokens = 1.0
                self._last_refill = time.monotonic()

            # Enforce minimum interval between requests
            if self._min_interval_s > 0:
                since_last = time.monotonic() - self._last_request
                if since_last < self._min_interval_s:
                    gap = self._min_interval_s - since_last
                    waited += gap
                    await asyncio.sleep(gap)

            # Consume one token
            if self._rpm > 0:
                self._tokens -= 1.0
            self._last_request = time.monotonic()

        return waited


# ---------------------------------------------------------------------------
# Response cache — hash-based LRU for identical prompts
# ---------------------------------------------------------------------------


class _ResponseCache:
    """Simple LRU cache keyed by prompt hash.

    Args:
        max_entries: Maximum cached responses before LRU eviction.
    """

    def __init__(self, max_entries: int = 10000) -> None:
        self._max_entries = max_entries
        # Ordered dict would be cleaner but dict preserves insertion order
        # in Python 3.7+ and we just trim from the front
        self._cache: dict[str, str] = {}

    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Stable hash for cache key — first 32 hex chars of SHA-256."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]

    def get(self, prompt: str) -> str | None:
        """Look up a cached response.  Returns None on miss."""
        key = self._hash_prompt(prompt)
        if key in self._cache:
            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return None

    def put(self, prompt: str, response: str) -> None:
        """Cache a response.  Evicts oldest entry if at capacity."""
        key = self._hash_prompt(prompt)
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = response
        # Evict oldest entries if over capacity
        while len(self._cache) > self._max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

    @property
    def size(self) -> int:
        """Number of cached responses."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# Risk-aware complete wrapper
# ---------------------------------------------------------------------------


class RiskAwareBackend:
    """Wraps a raw ``CompleteFn`` with risk-aware protective behaviour.

    The wrapper is itself a ``CompleteFn`` — drop-in compatible with
    FlockQueryManager, MCPSwarmEngine, GossipSwarm, etc.

    Args:
        complete: The raw completion callable to protect.
        config: Backend configuration including risk tier and limits.
        fallback_complete: Optional fallback callable.  Used when the
            primary backend fails after all retries.
    """

    def __init__(
        self,
        complete: CompleteFn,
        config: BackendConfig,
        fallback_complete: CompleteFn | None = None,
    ) -> None:
        self._complete = complete
        self.config = config
        self._fallback = fallback_complete
        self.metrics = BackendMetrics()

        # Initialize rate limiter
        self._limiter = _RateLimiter(
            rpm=config.max_requests_per_minute,
            min_interval_s=config.min_request_interval_s,
        )

        # Initialize cache if enabled
        self._cache: _ResponseCache | None = None
        if config.enable_cache:
            self._cache = _ResponseCache(max_entries=config.cache_max_entries)

        # Concurrency semaphore for batch control
        self._semaphore = asyncio.Semaphore(config.batch_concurrency)

        # Track consecutive failures for circuit breaking
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

        logger.info(
            "backend=<%s>, tier=<%s>, rpm=<%d>, cache=<%s>, concurrency=<%d> | "
            "risk-aware backend initialized",
            config.name, config.risk_tier.value,
            config.max_requests_per_minute,
            config.enable_cache, config.batch_concurrency,
        )

    async def __call__(self, prompt: str) -> str:
        """Execute a completion with risk-aware protection.

        Flow:
            1. Check cache → return if hit
            2. Check circuit breaker → fail fast if open
            3. Acquire rate limit token (may wait)
            4. Acquire concurrency semaphore
            5. Call the underlying complete function
            6. On failure: retry with exponential backoff
            7. On exhausted retries: try fallback
            8. Cache successful response

        Args:
            prompt: The prompt to complete.

        Returns:
            The completion response string.
        """
        self.metrics.total_calls += 1

        # 1. Check cache
        if self._cache is not None:
            cached = self._cache.get(prompt)
            if cached is not None:
                self.metrics.cache_hits += 1
                return cached

        # 2. Circuit breaker — if too many consecutive failures, fail fast
        if self._circuit_open_until > time.monotonic():
            logger.warning(
                "backend=<%s> | circuit breaker open, trying fallback",
                self.config.name,
            )
            return await self._try_fallback(prompt)

        # 3. Rate limiting
        waited = await self._limiter.acquire()
        if waited > 0:
            self.metrics.throttle_events += 1
            logger.debug(
                "backend=<%s>, waited_s=<%.2f> | rate limited",
                self.config.name, waited,
            )

        # 4 + 5. Execute with concurrency control and retries
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            async with self._semaphore:
                try:
                    response = await self._complete(prompt)

                    # Treat empty responses as failures — many
                    # completion functions swallow exceptions and
                    # return "" instead of raising.  Without this
                    # check, retries/circuit-breaker/fallback never
                    # activate for silent failures (e.g. HTTP 429
                    # caught inside _openrouter_complete).
                    if not response:
                        msg = "backend returned empty response"
                        raise RuntimeError(msg)

                    # Success — reset circuit breaker and cache
                    self._consecutive_failures = 0
                    if self._cache is not None:
                        self._cache.put(prompt, response)

                    return response

                except Exception as exc:
                    last_error = exc
                    self.metrics.retries += 1
                    self._consecutive_failures += 1

                    # Check if we should open the circuit breaker
                    # 10 consecutive failures → open for 60s
                    if self._consecutive_failures >= 10:
                        cooldown = 60.0
                        self._circuit_open_until = time.monotonic() + cooldown
                        logger.warning(
                            "backend=<%s>, consecutive_failures=<%d>, "
                            "cooldown_s=<%.0f> | circuit breaker opened",
                            self.config.name, self._consecutive_failures,
                            cooldown,
                        )

                    if attempt < self.config.max_retries:
                        delay = self.config.retry_base_delay_s * (2 ** (attempt - 1))
                        logger.warning(
                            "backend=<%s>, attempt=<%d/%d>, error=<%s>, "
                            "retry_delay_s=<%.1f> | retrying",
                            self.config.name, attempt,
                            self.config.max_retries, exc, delay,
                        )
                        await asyncio.sleep(delay)

        # All retries exhausted
        self.metrics.failures += 1
        logger.warning(
            "backend=<%s>, retries_exhausted=<%d>, last_error=<%s> | "
            "all retries failed",
            self.config.name, self.config.max_retries, last_error,
        )

        return await self._try_fallback(prompt)

    async def _try_fallback(self, prompt: str) -> str:
        """Attempt the fallback backend.  Returns empty string if none."""
        if self._fallback is not None:
            logger.info(
                "backend=<%s> | degrading to fallback",
                self.config.name,
            )
            try:
                return await self._fallback(prompt)
            except Exception as exc:
                logger.warning(
                    "backend=<%s>, fallback_error=<%s> | fallback also failed",
                    self.config.name, exc,
                )
        return ""

    def summary(self) -> dict:
        """Return a summary dict of backend metrics for logging/events."""
        return {
            "backend": self.config.name,
            "risk_tier": self.config.risk_tier.value,
            "model": self.config.model,
            "total_calls": self.metrics.total_calls,
            "cache_hits": self.metrics.cache_hits,
            "cache_hit_rate": round(self.metrics.cache_hit_rate, 3),
            "retries": self.metrics.retries,
            "failures": self.metrics.failures,
            "failure_rate": round(self.metrics.failure_rate, 3),
            "throttle_events": self.metrics.throttle_events,
            "cache_size": self._cache.size if self._cache else 0,
        }


def wrap_complete(
    complete: CompleteFn,
    config: BackendConfig,
    fallback_complete: CompleteFn | None = None,
) -> RiskAwareBackend:
    """Wrap a raw completion callable with risk-aware protection.

    This is the primary entry point.  The returned object is callable
    with the same signature as the input ``complete`` function.

    Args:
        complete: Raw completion callable.
        config: Backend configuration.
        fallback_complete: Optional fallback callable.

    Returns:
        A ``RiskAwareBackend`` instance (callable as CompleteFn).
    """
    return RiskAwareBackend(complete, config, fallback_complete)


# ---------------------------------------------------------------------------
# Backend registry — vestigial scaffold for dynamic model allocation
# ---------------------------------------------------------------------------


class BackendRegistry:
    """Registry of available model backends.

    Vestigial — provides the data structure for future dynamic allocation
    where the system selects backends based on query requirements, cost,
    and current health.  For now, it's a simple name→config lookup.

    Usage:
        registry = BackendRegistry()
        registry.register(BackendConfig.free_tier(
            name="ling-2.6-1t",
            model="inclusionai/ling-2.6-1t:free",
            api_base="https://openrouter.ai/api/v1",
        ))
        config = registry.get("ling-2.6-1t")
    """

    def __init__(self) -> None:
        self._backends: dict[str, BackendConfig] = {}

    def register(self, config: BackendConfig) -> None:
        """Register a backend configuration.

        Args:
            config: The backend config to register.
        """
        self._backends[config.name] = config
        logger.info(
            "backend=<%s>, tier=<%s>, model=<%s> | backend registered",
            config.name, config.risk_tier.value, config.model,
        )

    def get(self, name: str) -> BackendConfig | None:
        """Look up a backend by name.

        Args:
            name: Backend identifier.

        Returns:
            The BackendConfig, or None if not found.
        """
        return self._backends.get(name)

    def list_backends(self) -> list[BackendConfig]:
        """Return all registered backends.

        Returns:
            List of all registered BackendConfig objects.
        """
        return list(self._backends.values())

    def by_tier(self, tier: BackendRiskTier) -> list[BackendConfig]:
        """Return backends matching a risk tier.

        Args:
            tier: The risk tier to filter by.

        Returns:
            List of matching BackendConfig objects.
        """
        return [b for b in self._backends.values() if b.risk_tier == tier]
