#!/usr/bin/env python3
# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""OpenTelemetry tracing for the H200 swarm pipeline.

Provides comprehensive per-phase, per-worker, per-query tracing with
automatic export to OTLP JSON files and optional Backblaze B2 upload
for post-run analysis.

Trace hierarchy:
    pipeline (root span)
    ├── phase.corpus_analysis
    ├── phase.worker_map
    │   ├── worker.<angle_label>
    │   │   └── llm.complete (per LLM call)
    │   └── worker.<angle_label>
    ├── phase.gossip
    │   ├── gossip.round.1
    │   │   └── worker.<angle_label>
    │   └── gossip.round.2
    ├── phase.serendipity
    ├── phase.queen_merge
    ├── phase.flock
    │   ├── flock.query.<query_type>.<idx>
    │   └── flock.query.<query_type>.<idx>
    └── phase.b2_upload

Usage:
    from tracing import init_tracing, traced_complete_fn, shutdown_tracing
    from tracing import upload_traces_to_b2

    # At pipeline start
    tracer, trace_dir = init_tracing(run_id="run_20260416_120000")

    # Wrap completion functions
    traced_worker = traced_complete_fn(raw_worker_fn, tracer, "worker.bee")
    traced_flock = traced_complete_fn(raw_flock_fn, tracer, "flock.eval")

    # ... run pipeline ...

    # At pipeline end
    shutdown_tracing()
    upload_traces_to_b2(trace_dir, run_id="run_20260416_120000")
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# ── Trace file location ──────────────────────────────────────────────

_DEFAULT_TRACE_DIR = "/workspace/traces"

# ── Global tracer reference ──────────────────────────────────────────

_tracer = None
_provider = None
_trace_dir: Path | None = None


def init_tracing(
    run_id: str,
    trace_dir: str = _DEFAULT_TRACE_DIR,
    service_name: str = "mirothinker-swarm",
) -> "tuple[Any, Path]":
    """Initialise OpenTelemetry tracing with a JSON file exporter.

    Creates a TracerProvider that writes spans to a JSONL file in
    ``trace_dir``.  Each span is one JSON line for easy post-processing.

    Also configures OTLP HTTP export if OTEL_EXPORTER_OTLP_ENDPOINT is
    set (e.g. for Phoenix or Jaeger).

    Args:
        run_id: Unique identifier for this pipeline run.
        trace_dir: Directory to write trace files.
        service_name: OTel service name attribute.

    Returns:
        Tuple of (tracer, trace_dir_path).
    """
    global _tracer, _provider, _trace_dir

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )
    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed — tracing disabled. "
            "Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )
        # Return a no-op tracer
        return _NoOpTracer(), Path(trace_dir)

    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    _trace_dir = trace_path

    trace_file = trace_path / f"{run_id}_spans.jsonl"

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "run.id": run_id,
    })

    _provider = TracerProvider(resource=resource)

    # File exporter — writes spans as JSONL for B2 upload
    file_exporter = _JsonlFileExporter(str(trace_file))
    _provider.add_span_processor(SimpleSpanProcessor(file_exporter))

    # Optional OTLP HTTP exporter (for Phoenix, Jaeger, etc.)
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            _provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
            logger.info(
                "otel_endpoint=<%s> | OTLP HTTP exporter enabled",
                otlp_endpoint,
            )
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-http not installed — "
                "OTLP export disabled"
            )

    trace.set_tracer_provider(_provider)
    _tracer = _provider.get_tracer(service_name)

    logger.info(
        "run_id=<%s>, trace_file=<%s> | OpenTelemetry tracing initialised",
        run_id, trace_file,
    )

    return _tracer, trace_path


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        logger.info("tracing shutdown complete — all spans flushed")


def get_tracer() -> Any:
    """Return the global tracer (or a no-op if tracing is not initialised)."""
    return _tracer or _NoOpTracer()


# ── JSONL file exporter ──────────────────────────────────────────────


class _JsonlFileExporter:
    """Exports spans as JSON lines to a local file.

    Each span is serialised as a single JSON line containing:
    - trace_id, span_id, parent_span_id
    - name, kind, status
    - start_time, end_time, duration_ms
    - attributes (flat dict)
    - events (list of {name, timestamp, attributes})
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._fh = open(file_path, "a")  # noqa: SIM115

    def export(self, spans: "Any") -> "Any":
        """Export a batch of spans to the JSONL file."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        for span in spans:
            record = self._span_to_dict(span)
            self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Close the file handle."""
        self._fh.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: ARG002
        """Flush pending writes."""
        self._fh.flush()
        return True

    @staticmethod
    def _span_to_dict(span: "Any") -> dict[str, Any]:
        """Convert an OTel ReadableSpan to a plain dict."""
        ctx = span.get_span_context()
        parent = span.parent

        start_ns = span.start_time or 0
        end_ns = span.end_time or 0
        duration_ms = (end_ns - start_ns) / 1_000_000

        events = []
        for event in span.events:
            events.append({
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": dict(event.attributes) if event.attributes else {},
            })

        status_code = "UNSET"
        status_desc = ""
        if span.status is not None:
            status_code = span.status.status_code.name
            status_desc = span.status.description or ""

        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
            "parent_span_id": format(parent.span_id, "016x") if parent else None,
            "name": span.name,
            "kind": span.kind.name if span.kind else "INTERNAL",
            "status": status_code,
            "status_description": status_desc,
            "start_time": start_ns,
            "end_time": end_ns,
            "duration_ms": round(duration_ms, 2),
            "attributes": dict(span.attributes) if span.attributes else {},
            "events": events,
            "resource": dict(span.resource.attributes) if span.resource else {},
        }


# ── Traced completion wrapper ────────────────────────────────────────


def traced_complete_fn(
    raw_fn: Callable[[str], Awaitable[str]],
    span_name_prefix: str,
    *,
    model_name: str = "",
    backend: str = "",
) -> Callable[[str], Awaitable[str]]:
    """Wrap a CompleteFn with OpenTelemetry span instrumentation.

    Each call creates a child span under the current context with:
    - prompt length (chars and estimated tokens)
    - response length
    - model name and backend
    - latency

    Args:
        raw_fn: The original async completion function.
        span_name_prefix: Prefix for the span name (e.g. "worker.bee").
        model_name: Model identifier for span attributes.
        backend: Backend identifier (e.g. "deepseek-api", "vllm-local").

    Returns:
        A traced async completion function with the same signature.
    """

    async def _traced(prompt: str) -> str:
        tracer = get_tracer()
        with tracer.start_as_current_span(
            f"{span_name_prefix}.complete",
            attributes={
                "llm.model": model_name,
                "llm.backend": backend,
                "llm.prompt_chars": len(prompt),
                "llm.prompt_tokens_est": len(prompt) // 4,
            },
        ) as span:
            t0 = time.monotonic()
            try:
                result = await raw_fn(prompt)
                elapsed_ms = (time.monotonic() - t0) * 1000
                span.set_attribute("llm.response_chars", len(result))
                span.set_attribute("llm.response_tokens_est", len(result) // 4)
                span.set_attribute("llm.latency_ms", round(elapsed_ms, 1))
                span.set_attribute("llm.success", bool(result))
                if not result:
                    span.add_event("empty_response")
                return result
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                span.set_attribute("llm.latency_ms", round(elapsed_ms, 1))
                span.set_attribute("llm.success", False)
                span.set_attribute("llm.error", str(exc))
                span.add_event("llm_error", {"error.message": str(exc)})
                raise

    return _traced


def trace_phase(
    phase_name: str,
    attributes: dict[str, Any] | None = None,
) -> Any:
    """Create a span for a pipeline phase.

    Usage:
        with trace_phase("worker_map", {"worker_count": 12}) as span:
            # ... do phase work ...
            span.set_attribute("phase.result_count", 12)

    Args:
        phase_name: Name of the phase (e.g. "corpus_analysis", "gossip").
        attributes: Optional initial span attributes.

    Returns:
        A context manager that yields the span.
    """
    tracer = get_tracer()
    return tracer.start_as_current_span(
        f"phase.{phase_name}",
        attributes=attributes or {},
    )


def trace_event(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Add an event to the current span.

    Args:
        name: Event name.
        attributes: Event attributes.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes=attributes or {})
    except Exception:
        pass


# ── B2 trace upload ──────────────────────────────────────────────────


def upload_traces_to_b2(
    trace_dir: Path | str,
    run_id: str,
    bucket_name: str = "mirothinker-traces",
) -> list[str]:
    """Upload all trace files from a run to Backblaze B2.

    Uploads every file in trace_dir matching the run_id pattern,
    plus any metrics JSON files.

    Args:
        trace_dir: Directory containing trace files.
        run_id: Run identifier (used as B2 key prefix).
        bucket_name: B2 bucket name.

    Returns:
        List of B2 download URLs for uploaded files.
    """
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        logger.warning(
            "trace_dir=<%s> | trace directory does not exist — skipping B2 upload",
            trace_path,
        )
        return []

    b2_key_id = os.environ.get("B2_APPLICATION_KEY_ID", "") or os.environ.get("B2_KEY_ID", "")
    b2_app_key = os.environ.get("B2_APPLICATION_KEY", "")

    if not b2_key_id or not b2_app_key:
        logger.warning(
            "B2_APPLICATION_KEY_ID / B2_APPLICATION_KEY not set — "
            "trace upload skipped. Set these env vars to enable B2 export"
        )
        return []

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo, UploadSourceBytes
        from b2sdk.v2.exception import NonExistentBucket
    except ImportError:
        logger.warning(
            "b2sdk not installed — trace upload skipped. "
            "Install: pip install b2sdk"
        )
        return []

    # Initialise B2
    info = InMemoryAccountInfo()
    api = B2Api(info)
    api.authorize_account("production", b2_key_id, b2_app_key)

    try:
        bucket = api.get_bucket_by_name(bucket_name)
    except NonExistentBucket:
        bucket = api.create_bucket(
            bucket_name,
            bucket_type="allPrivate",
            lifecycle_rules=[],
        )
        logger.info("created new B2 bucket: %s", bucket_name)

    # Collect files to upload
    files_to_upload = []
    for f in sorted(trace_path.iterdir()):
        if f.is_file() and (
            f.suffix in (".jsonl", ".json", ".log", ".md")
            or run_id in f.name
        ):
            files_to_upload.append(f)

    if not files_to_upload:
        logger.info(
            "trace_dir=<%s>, run_id=<%s> | no trace files found to upload",
            trace_path, run_id,
        )
        return []

    urls: list[str] = []
    for f in files_to_upload:
        key = f"traces/{run_id}/{f.name}"
        content_type = _guess_content_type(f.suffix)

        data = f.read_bytes()
        source = UploadSourceBytes(data)
        file_version = bucket.upload(
            source,
            file_name=key,
            content_type=content_type,
        )
        download_url = api.get_download_url_for_fileid(file_version.id_)
        urls.append(download_url)
        logger.info(
            "file=<%s>, size=<%d>, key=<%s> | uploaded to B2",
            f.name, len(data), key,
        )

    logger.info(
        "run_id=<%s>, files_uploaded=<%d> | B2 trace upload complete",
        run_id, len(urls),
    )
    return urls


def upload_output_dir_to_b2(
    output_dir: Path | str,
    run_id: str,
    bucket_name: str = "mirothinker-traces",
) -> list[str]:
    """Upload all output files (reports, metrics, worker summaries) to B2.

    Uploads everything in the output directory — reports, metrics JSON,
    worker summaries, serendipity insights, pipeline logs.

    Args:
        output_dir: Directory containing pipeline output files.
        run_id: Run identifier (used as B2 key prefix).
        bucket_name: B2 bucket name.

    Returns:
        List of B2 download URLs for uploaded files.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning(
            "output_dir=<%s> | output directory does not exist — skipping B2 upload",
            output_path,
        )
        return []

    b2_key_id = os.environ.get("B2_APPLICATION_KEY_ID", "") or os.environ.get("B2_KEY_ID", "")
    b2_app_key = os.environ.get("B2_APPLICATION_KEY", "")

    if not b2_key_id or not b2_app_key:
        logger.warning(
            "B2 credentials not set — output upload skipped"
        )
        return []

    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo, UploadSourceBytes
        from b2sdk.v2.exception import NonExistentBucket
    except ImportError:
        logger.warning("b2sdk not installed — output upload skipped")
        return []

    info = InMemoryAccountInfo()
    api = B2Api(info)
    api.authorize_account("production", b2_key_id, b2_app_key)

    try:
        bucket = api.get_bucket_by_name(bucket_name)
    except NonExistentBucket:
        bucket = api.create_bucket(
            bucket_name,
            bucket_type="allPrivate",
            lifecycle_rules=[],
        )

    urls: list[str] = []
    for f in sorted(output_path.iterdir()):
        if not f.is_file():
            continue
        if f.suffix not in (".md", ".json", ".log", ".txt", ".jsonl", ".duckdb"):
            continue

        key = f"outputs/{run_id}/{f.name}"
        content_type = _guess_content_type(f.suffix)

        data = f.read_bytes()
        source = UploadSourceBytes(data)
        file_version = bucket.upload(
            source,
            file_name=key,
            content_type=content_type,
        )
        download_url = api.get_download_url_for_fileid(file_version.id_)
        urls.append(download_url)
        logger.info(
            "file=<%s>, size=<%d>, key=<%s> | uploaded output to B2",
            f.name, len(data), key,
        )

    logger.info(
        "run_id=<%s>, files_uploaded=<%d> | B2 output upload complete",
        run_id, len(urls),
    )
    return urls


def _guess_content_type(suffix: str) -> str:
    """Map file suffix to MIME type."""
    return {
        ".jsonl": "application/jsonlines",
        ".json": "application/json",
        ".md": "text/markdown",
        ".log": "text/plain",
        ".txt": "text/plain",
        ".duckdb": "application/octet-stream",
    }.get(suffix, "application/octet-stream")


# ── No-op tracer (when OTel is not installed) ────────────────────────


class _NoOpSpan:
    """No-op span that silently ignores all operations."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer returned when OpenTelemetry is not available."""

    def start_as_current_span(
        self, name: str, **kwargs: Any,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()
