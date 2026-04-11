# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""AG-UI FastAPI server for MiroThinker.

Wraps the MiroThinker pipeline_agent with the AG-UI protocol middleware,
enabling rich frontends (CopilotKit, custom React, etc.) to interact with
the agent via Server-Sent Events streaming.

The pipeline architecture separates reasoning from tool execution:

    SequentialAgent("mirothinker_pipeline")
    └── LoopAgent("research_loop", max_iterations=3)
    │     ├── thinker           → uncensored reasoning, no web tools
    │     └── researcher        → tool-capable, calls executor
    └── synthesiser  → final uncensored report writing, no tools

Logging and observability is sent to **three frontends simultaneously**:

1. **Phoenix (Arize)** — OTel traces with LLM calls, tool invocations,
   latency, and token counts.  Requires ``PHOENIX_ENABLED=1``.
2. **AG-UI dashboard** — real-time SSE stream + HTML reports with
   pipeline KPIs, tool breakdowns, and event timelines.
3. **Next.js / Gradio** — streaming reasoning + tool-call events
   rendered as cards in the chat UI (handled by AG-UI protocol).
Usage:
    PHOENIX_ENABLED=1 uvicorn server:app --host 0.0.0.0 --port 8000 --reload

The AG-UI endpoint is mounted at POST /  (root).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from google.adk.apps import App

from agents.pipeline import pipeline_agent
from dashboard import set_active_collector, unregister_collector
from dashboard.collector import PipelineCollector
from dashboard.sse import mount_dashboard_routes
from plugins import setup_otel, build_plugins

# ── Logging configuration ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── OTel + Phoenix tracing ────────────────────────────────────────
# This is the critical missing piece: setup_otel() was only called in
# main.py (CLI), so AG-UI server runs never produced OTel spans.
# Now both SQLite archive and Phoenix OTLP exporters are active for
# every request that flows through the server.
setup_otel()
logger.info("OTel tracing initialised for AG-UI server")


# ── Request logging middleware ────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with timing, status, and size metadata.

    This feeds structured data to the console log (and any log
    aggregator) so operators can see exactly what the server is doing.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()

        # Log request arrival
        body_bytes = 0
        if request.method in ("POST", "PUT", "PATCH"):
            # Read the body length without consuming it
            body = await request.body()
            body_bytes = len(body)

        logger.info(
            "REQ  %s %s %s body=%dB request_id=%s client=%s",
            request.method,
            request.url.path,
            request.url.query or "",
            body_bytes,
            request_id,
            request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "ERR  %s %s 500 %.0fms request_id=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                request_id,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        content_length = response.headers.get("content-length", "?")

        logger.info(
            "RESP %s %s %d %.0fms size=%sB request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            content_length,
            request_id,
        )
        return response


# ── AG-UI run lifecycle middleware ────────────────────────────────

class AGUIRunCollectorMiddleware(BaseHTTPMiddleware):
    """Create a PipelineCollector for each AG-UI POST / run.

    This bridges the gap between the AG-UI endpoint (which streams SSE
    events to frontends) and the dashboard collector (which accumulates
    structured KPIs for the SSE dashboard and HTML reports).

    For every POST to ``/``, we:
    1. Create a fresh ``PipelineCollector`` with a unique session ID.
    2. Register it as the active collector (so ``/dashboard/stream``
       immediately starts broadcasting its snapshots).
    3. Let the AG-UI endpoint handle the request normally.
    4. **Wrap the SSE response body iterator** so that finalization
       happens only after the full stream has been consumed — not when
       ``call_next()`` returns (which is just response headers for
       ``StreamingResponse``).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method != "POST" or request.url.path != "/":
            return await call_next(request)

        session_id = uuid.uuid4().hex[:16]

        # Try to extract query from the request body
        query = "(unknown)"
        try:
            body = await request.body()
            data = json.loads(body)
            messages = data.get("messages", [])
            if messages:
                last_user = [m for m in messages if m.get("role") == "user"]
                if last_user:
                    query = last_user[-1].get("content", "(empty)")[:200]
        except Exception:
            pass

        collector = PipelineCollector(query=query, session_id=session_id)
        set_active_collector(collector)

        logger.info(
            "AG-UI RUN START session=%s query=%r",
            session_id,
            query[:80],
        )
        collector.phase_start("ag_ui_request", "pipeline_agent")

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            collector.phase_end("ag_ui_request", "error")
            collector.finalize(result_text="")
            unregister_collector(session_id)
            set_active_collector(None)
            logger.error(
                "AG-UI RUN ERROR session=%s elapsed=%.1fs",
                session_id,
                elapsed,
            )
            raise

        # For StreamingResponse (SSE), call_next returns as soon as the
        # response *headers* are ready — the body streams afterwards.
        # We must defer finalization until the body iterator is fully
        # consumed, otherwise the collector will be empty.
        original_iterator = response.body_iterator

        async def _finalizing_iterator():
            """Yield all chunks from the original body, then finalize.

            Uses try/except/else/finally so that cleanup happens even on
            ``GeneratorExit`` (client disconnect).  ``GeneratorExit``
            inherits from ``BaseException``, so a bare ``except Exception``
            would miss it — the ``finally`` block is the safety net.
            """
            try:
                async for chunk in original_iterator:
                    yield chunk
            except Exception:
                collector.phase_end("ag_ui_request", "error")
                collector.finalize(result_text="")
                elapsed = time.perf_counter() - start
                logger.error(
                    "AG-UI RUN STREAM ERROR session=%s elapsed=%.1fs",
                    session_id,
                    elapsed,
                )
                raise
            else:
                elapsed = time.perf_counter() - start
                collector.phase_end("ag_ui_request", "ok")
                dashboard_data = collector.finalize(result_text="")

                kpi = dashboard_data.get("kpi", {})
                logger.info(
                    "AG-UI RUN END session=%s elapsed=%.1fs "
                    "tool_calls=%d llm_calls=%d adk_events=%d "
                    "prompt_tokens=%d completion_tokens=%d",
                    session_id,
                    elapsed,
                    kpi.get("tool_calls", 0),
                    kpi.get("llm_calls", 0),
                    kpi.get("adk_events", 0),
                    kpi.get("prompt_tokens_est", 0),
                    kpi.get("completion_tokens_est", 0),
                )
            finally:
                # Safety net for GeneratorExit (client disconnect) and
                # any other BaseException that bypasses except/else.
                if not collector._finalized:
                    collector.phase_end("ag_ui_request", "disconnected")
                    collector.finalize(result_text="")
                    elapsed = time.perf_counter() - start
                    logger.warning(
                        "AG-UI RUN DISCONNECTED session=%s elapsed=%.1fs",
                        session_id,
                        elapsed,
                    )
                unregister_collector(session_id)
                set_active_collector(None)

        response.body_iterator = _finalizing_iterator()
        return response


# ── Application lifecycle ─────────────────────────────────────────

def _validate_env() -> None:
    """Check API keys and critical env vars at startup.

    Logs warnings for missing keys so operators know immediately which
    tool providers will be unavailable — rather than discovering it
    mid-pipeline when a tool call fails.
    """
    checks = {
        "OPENAI_API_KEY": "LLM provider (Venice/OpenAI)",
        "BRAVE_API_KEY": "Brave Search tools",
        "EXA_API_KEY": "Exa search/crawl tools",
        "FIRECRAWL_API_KEY": "Firecrawl scrape tools",
    }
    optional = {
        "TRANSCRIPTAPI_KEY": "TranscriptAPI (video transcription)",
        "KAGI_API_KEY": "Kagi search",
    }
    all_ok = True
    for key, description in checks.items():
        val = os.environ.get(key, "")
        if not val:
            logger.warning("  MISSING  %-22s → %s will be unavailable", key, description)
            all_ok = False
        else:
            logger.info("  OK       %-22s → %s", key, description)
    for key, description in optional.items():
        val = os.environ.get(key, "")
        if val:
            logger.info("  OK       %-22s → %s", key, description)
        else:
            logger.info("  SKIP     %-22s → %s (optional)", key, description)
    if all_ok:
        logger.info("  All required API keys present")
    else:
        logger.warning(
            "  Some API keys are missing — affected tools will fail at runtime"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log startup and shutdown with environment summary."""
    phoenix = "enabled" if os.environ.get("PHOENIX_ENABLED") == "1" else "disabled"
    model = os.environ.get("ADK_MODEL", os.environ.get("MODEL_NAME", "unknown"))
    base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("BASE_URL", "unknown"))

    logger.info("=" * 60)
    logger.info("MiroThinker AG-UI server starting")
    logger.info("  Model       : %s", model)
    logger.info("  LLM base URL: %s", base_url)
    logger.info("  Phoenix     : %s", phoenix)
    logger.info("  ADK debug   : %s", os.environ.get("ADK_DEBUG", "0"))
    logger.info("  CORS        : allow_origins=['*']")
    logger.info("  DuckDB+Flock: required (duckdb <1.5.0)")
    logger.info("-" * 60)
    _validate_env()
    logger.info("=" * 60)
    yield
    logger.info("MiroThinker AG-UI server shutting down")


# ── FastAPI app ─────────────────────────────────────────────────────

app = FastAPI(
    title="MiroThinker AG-UI",
    description="AG-UI protocol endpoint for MiroThinker deep-research agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware is evaluated in reverse registration order (last registered
# runs first).  We want: RequestLogging → AGUIRunCollector → handler.
app.add_middleware(AGUIRunCollectorMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# CORS — allow all origins so any frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── AG-UI agent wrapper ────────────────────────────────────────────
# Use App + from_app() to get ADK plugin support (logging, tracing,
# context filtering, reflect-and-retry) in the server path too.

plugins = build_plugins()
logger.info("ADK plugins loaded: %s", ", ".join(p.name for p in plugins))

adk_app = App(
    name="mirothinker_adk",
    root_agent=pipeline_agent,
    plugins=plugins,
)

adk_agent = ADKAgent.from_app(
    adk_app,
    user_id="server_user",
    execution_timeout_seconds=86400,  # 24h — AG-UI is a transport layer, not an execution controller.  The pipeline controls its own lifecycle via LoopAgent max_iterations and the EVIDENCE_SUFFICIENT sentinel.
)

# Mount the AG-UI SSE endpoint at root
add_adk_fastapi_endpoint(app, adk_agent, path="/")

# ── Dashboard endpoints ───────────────────────────────────────────
mount_dashboard_routes(app)

logger.info("MiroThinker AG-UI server ready at http://0.0.0.0:8000")
