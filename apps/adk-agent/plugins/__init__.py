# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""ADK plugin stack and OpenTelemetry observability setup.

Provides two public helpers consumed by ``main.py``:

* ``setup_otel()``   — configures dual OTel span exporters (SQLite archive +
  optional Phoenix dashboard) so ADK's built-in tracing streams rich metrics
  to both destinations with zero custom instrumentation.

* ``build_plugins()`` — returns the ordered list of ADK ``BasePlugin``
  instances that replace ~660 lines of custom callback code:

  1. **ContextFilterPlugin** → replaces Keep-K-Recent, Adaptive K, dynamic
     compression.  Works because MiroThinker's important data flows through
     session *state* (blackboard), not raw conversation context.
  2. **ReflectAndRetryToolPlugin** → replaces Bad Result Detection (Alg 4)
     and Arg Fix (Alg 8).  Self-healing: catches tool failures, shows the
     LLM structured reflection guidance, retries up to N times.
  3. **GlobalInstructionPlugin** → replaces Dedup Guard (Alg 2).  Injects
     cross-cutting instructions into every agent's system prompt.
  4. **LoggingPlugin** → ADK execution visibility at every lifecycle point
     (LLM requests/responses, tool calls, agent transitions, token counts).
  5. **DebugLoggingPlugin** → complete YAML trace dump for local debugging
     (conditional on ``ADK_DEBUG=1``).

Plugin execution order matters — PluginManager runs in list order and the
first non-None return short-circuits later plugins + agent callbacks:

  ContextFilter  → trims context BEFORE other plugins see it
  ReflectAndRetry → catches tool errors before they propagate
  GlobalInstruction → injects instructions into (already-trimmed) context
  Logging        → observes the final state (read-only)
  DebugLogging   → observes and dumps (read-only)
"""

from __future__ import annotations

import logging
import os
from typing import List

from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

# ── OTel span archive directory ────────────────────────────────────
_FINDINGS_DIR = os.environ.get("FINDINGS_DIR", "/tmp/mirothinker")
_SPANS_DB = os.path.join(_FINDINGS_DIR, "adk_spans.db")

# ── Phoenix configuration ──────────────────────────────────────────
_PHOENIX_ENABLED = os.environ.get("PHOENIX_ENABLED", "").strip() == "1"
_PHOENIX_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
)


def setup_otel() -> None:
    """Configure dual OTel span exporters for ADK's built-in tracing.

    Exporter 1 — **SqliteSpanExporter** (always on):
      Writes every OTel span to a local SQLite file that can be queried
      with DuckDB, backed up to B2, or fed into any dashboard tool.

    Exporter 2 — **Phoenix OTLP** (when ``PHOENIX_ENABLED=1``):
      Streams spans to a running Phoenix server for real-time dashboard,
      trace waterfall views, latency histograms, and evaluation framework.

    Both coexist in the same ``TracerProvider`` — same spans, two
    destinations, zero custom instrumentation.
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    from google.adk.telemetry.setup import OTelHooks, maybe_set_otel_providers
    from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter

    os.makedirs(_FINDINGS_DIR, exist_ok=True)

    processors = []

    # 1. SQLite archive — always on
    sqlite_exporter = SqliteSpanExporter(db_path=_SPANS_DB)
    processors.append(SimpleSpanProcessor(sqlite_exporter))
    logger.info("OTel: SQLite span archive → %s", _SPANS_DB)

    # 2. Phoenix OTLP — opt-in via PHOENIX_ENABLED=1
    if _PHOENIX_ENABLED:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            phoenix_exporter = OTLPSpanExporter(endpoint=_PHOENIX_ENDPOINT)
            processors.append(BatchSpanProcessor(phoenix_exporter))
            logger.info("OTel: Phoenix OTLP → %s", _PHOENIX_ENDPOINT)
        except ImportError:
            logger.warning(
                "OTel: Phoenix requested but opentelemetry-exporter-otlp "
                "not installed — skipping"
            )
    else:
        logger.info("OTel: Phoenix disabled (set PHOENIX_ENABLED=1 to enable)")

    hooks = OTelHooks(span_processors=processors)
    maybe_set_otel_providers([hooks])
    logger.info("OTel: %d span processor(s) registered", len(processors))


# ── Global instruction for all agents ──────────────────────────────
# Replaces the custom Dedup Guard (Algorithm 2) callback.  The LLM
# enforces dedup via its own reasoning — no session-state machinery.
_GLOBAL_INSTRUCTION = """\
SEARCH QUALITY RULES (apply to every agent):
1. NEVER repeat a search query you have already executed in this session.
   Before making any search call, review your conversation history and
   confirm the query is novel.  If a previous search returned no results,
   rephrase with different keywords — do not retry the same query.
2. Always cite sources with full URLs.
3. Prefer diverse source types (news, academic, vendor, forums).
4. If a tool call fails, try a DIFFERENT tool or query — do not retry
   the same call with the same arguments.
"""


def build_plugins() -> List[BasePlugin]:
    """Return the ordered list of ADK plugins for every Runner / App.

    The list is built fresh on each call so environment variables are
    re-read (useful for tests that toggle ``ADK_DEBUG``).
    """
    from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
    from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
    from google.adk.plugins.logging_plugin import LoggingPlugin
    from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin

    plugins: List[BasePlugin] = [
        # 1. Context management — replaces Keep-K-Recent + dynamic compression
        ContextFilterPlugin(
            num_invocations_to_keep=int(
                os.environ.get("CONTEXT_INVOCATIONS_TO_KEEP", "2")
            ),
        ),
        # 2. Self-healing tools — replaces Bad Result Detection + Arg Fix
        ReflectAndRetryToolPlugin(
            max_retries=int(os.environ.get("TOOL_MAX_RETRIES", "2")),
            throw_exception_if_retry_exceeded=False,
        ),
        # 3. Cross-cutting instructions — replaces Dedup Guard
        GlobalInstructionPlugin(global_instruction=_GLOBAL_INSTRUCTION),
        # 4. ADK execution visibility (console)
        LoggingPlugin(),
    ]

    # 5. Debug YAML traces — opt-in via ADK_DEBUG=1
    if os.environ.get("ADK_DEBUG", "").strip() == "1":
        from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin

        debug_path = os.path.join(_FINDINGS_DIR, "adk_debug.yaml")
        plugins.append(DebugLoggingPlugin(output_path=debug_path))
        logger.info("DebugLoggingPlugin enabled → %s", debug_path)

    logger.info(
        "ADK plugins: %s",
        ", ".join(p.name for p in plugins),
    )
    return plugins
