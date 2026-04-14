# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Typed error taxonomy for the pipeline.

Four categories replace 15+ identical ``except Exception: logger.warning("(non-fatal)")``
handlers.  Each category carries distinct semantics that the
``ErrorEscalationAspect`` and ``QualityManifest`` consume:

- **PipelineCritical**: Pipeline cannot continue.  Abort immediately.
  Example: DuckDB connection lost, Flock extension missing.

- **PipelineDegraded**: Pipeline can continue but quality is reduced.
  Must be tracked in the quality manifest so the report is honest.
  Example: One search API returned 401, scoring partially failed.

- **PipelineTransient**: Temporary failure that may resolve on retry.
  Logged at WARNING, counted, but not surfaced in the manifest unless
  it becomes chronic (>3 occurrences of the same transient error).
  Example: Network timeout on a single API call, rate limit hit.

- **PipelineIgnorable**: Cosmetic / diagnostic failure that has zero
  impact on research quality.  Logged at DEBUG only.
  Example: Dashboard event emission failed, health check logging error.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all typed pipeline errors."""

    category: str = "unknown"

    def __init__(self, message: str, *, source: str = "", cause: Exception | None = None) -> None:
        self.source = source
        self.cause = cause
        super().__init__(message)


class PipelineCritical(PipelineError):
    """Pipeline cannot continue — abort immediately.

    Examples:
    - DuckDB connection lost
    - Flock extension missing or version mismatch
    - Corpus file corrupted
    """

    category = "critical"


class PipelineDegraded(PipelineError):
    """Pipeline continues but quality is reduced.

    Must be tracked in the quality manifest.

    Examples:
    - Search API returned 401 (one source lost)
    - Scoring partially failed (some findings unscored)
    - Thought admission failed (reasoning chain incomplete)
    """

    category = "degraded"


class PipelineTransient(PipelineError):
    """Temporary failure that may self-resolve.

    Counted; becomes degraded if chronic (>3 same-source occurrences).

    Examples:
    - Network timeout on a single API call
    - Rate limit hit (will retry next iteration)
    - Temporary DuckDB lock contention
    """

    category = "transient"


class PipelineIgnorable(PipelineError):
    """Cosmetic failure with zero quality impact.

    Logged at DEBUG only; not tracked in manifest.

    Examples:
    - Dashboard event emission failed
    - Health check logging error
    - Swarm router reset failed
    """

    category = "ignorable"


def classify_error(
    error: Exception,
    *,
    source: str = "",
    context: str = "",
) -> PipelineError:
    """Classify a raw exception into a typed pipeline error.

    Uses heuristics based on the exception type and message to determine
    the appropriate category.  This is the central classification point
    that replaces ad-hoc ``(non-fatal)`` annotations.
    """
    msg = str(error)
    err_type = type(error).__name__

    # Critical: DuckDB connection errors
    if "duckdb" in err_type.lower() or "duckdb" in msg.lower():
        if any(kw in msg.lower() for kw in ["connection", "corrupt", "permission"]):
            return PipelineCritical(
                f"DuckDB error in {source}: {msg}",
                source=source, cause=error,
            )

    # Critical: Flock extension errors
    if "flock" in msg.lower() and any(
        kw in msg.lower() for kw in ["not found", "missing", "version"]
    ):
        return PipelineCritical(
            f"Flock extension error in {source}: {msg}",
            source=source, cause=error,
        )

    # Transient: network/timeout errors
    if any(kw in err_type.lower() for kw in ["timeout", "connection"]):
        return PipelineTransient(
            f"Network error in {source}: {msg}",
            source=source, cause=error,
        )
    if any(kw in msg.lower() for kw in ["timeout", "timed out", "rate limit", "429", "503"]):
        return PipelineTransient(
            f"Transient error in {source}: {msg}",
            source=source, cause=error,
        )

    # Ignorable: dashboard/logging errors
    if source in ("dashboard", "health_check", "emit_event", "swarm_router_reset"):
        return PipelineIgnorable(
            f"Ignorable error in {source}: {msg}",
            source=source, cause=error,
        )

    # Degraded: API auth errors
    if any(kw in msg.lower() for kw in ["401", "403", "unauthorized", "forbidden"]):
        return PipelineDegraded(
            f"Auth error in {source}: {msg}",
            source=source, cause=error,
        )

    # Default: degraded (conservative — track it)
    return PipelineDegraded(
        f"Unclassified error in {source}: {msg}",
        source=source, cause=error,
    )
