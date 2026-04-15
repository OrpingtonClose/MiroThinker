# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Venice GLM-4.7 uncensored research agent — Strands Agents SDK.

Features used:
- OpenAI-compatible model provider (Venice AI)
- MCP tool integration (Brave, Firecrawl, Exa, Kagi)
- Streaming responses (PrintingCallbackHandler)
- Conversation memory (SlidingWindowConversationManager)
- Agent loop with automatic tool dispatch
- Multi-agent orchestration (planner + researcher via agent-as-tool)
- Guardrails (callback-based pre/post processing with budget limits)
- OpenTelemetry observability (OTEL_EXPORTER_OTLP_ENDPOINT)
"""

from __future__ import annotations

import logging
import os
import sys
import time

from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import (
    CompositeCallbackHandler,
    PrintingCallbackHandler,
)
from strands.agent.conversation_manager import SlidingWindowConversationManager

from config import build_model
from prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, SYSTEM_PROMPT
from tools import get_all_mcp_clients

logger = logging.getLogger(__name__)

# ── Guardrails: budget tracking callback ─────────────────────────────
# Tracks actual tool invocations (not user queries) via Strands' callback
# system.  The callback fires for every streaming event; we only increment
# when ``current_tool_use`` is present with a tool ``name`` and we haven't
# already counted that specific invocation (keyed by ``toolUseId``).
# Similar to the budget gating in apps/adk-agent/.env.example lines 43-44.

_session_start = time.time()
_tool_call_count = 0
_seen_tool_use_ids: set[str] = set()
_MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "200"))
_SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))


def reset_budget() -> None:
    """Reset per-request budget counters.

    Call this before each HTTP request so that budget globals don't
    accumulate across requests in a long-running server process.
    """
    global _session_start, _tool_call_count, _seen_tool_use_ids
    _session_start = time.time()
    _tool_call_count = 0
    _seen_tool_use_ids = set()


def budget_callback(**kwargs) -> None:
    """Callback-handler guardrail that counts actual tool invocations.

    Strands fires this callback for every streaming event.  We detect new
    tool calls by checking for the ``current_tool_use`` kwarg with a tool
    ``name`` and a unique ``toolUseId``.  Each unique ID is counted once.
    """
    global _tool_call_count

    tool_use = kwargs.get("current_tool_use")
    if not tool_use or not tool_use.get("name"):
        return

    tool_use_id = tool_use.get("toolUseId", "")
    if tool_use_id in _seen_tool_use_ids:
        return
    _seen_tool_use_ids.add(tool_use_id)

    _tool_call_count += 1

    elapsed = time.time() - _session_start
    if elapsed > _SESSION_TIMEOUT:
        logger.warning(
            "Session timeout reached (%.0fs > %ds). Consider wrapping up.",
            elapsed,
            _SESSION_TIMEOUT,
        )

    if _tool_call_count > _MAX_TOOL_CALLS:
        logger.warning(
            "Tool call budget exceeded (%d > %d). Consider wrapping up.",
            _tool_call_count,
            _MAX_TOOL_CALLS,
        )

    if _tool_call_count % 10 == 0:
        logger.info(
            "Budget: %d tool calls, %.0fs elapsed",
            _tool_call_count,
            elapsed,
        )


def _build_callback_handler():
    """Build a composite callback handler: printing + budget guardrail."""
    return CompositeCallbackHandler(PrintingCallbackHandler(), budget_callback)


# ── OpenTelemetry setup ──────────────────────────────────────────────
# Strands has built-in OTEL support.  Set OTEL_EXPORTER_OTLP_ENDPOINT
# in .env to stream traces to Phoenix or any OTEL backend.
# (matches the pattern at apps/adk-agent/.env.example lines 96-100)


def _setup_otel() -> None:
    """Configure OpenTelemetry tracing if OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OTEL tracing enabled → %s", endpoint)
    except ImportError:
        logger.warning(
            "OTEL packages not installed. Run: pip install opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-http"
        )


# ── Agent factories ──────────────────────────────────────────────────


def _enter_mcp_clients(mcp_clients):
    """Enter MCP client contexts and collect tools, with rollback on failure.

    If the Nth client's ``__enter__()`` or ``list_tools_sync()`` raises,
    all previously-entered clients are cleaned up so their subprocesses
    (npx, node, uvx) don't leak.
    """
    entered: list = []
    tool_list: list = []
    try:
        for client in mcp_clients:
            client.__enter__()
            entered.append(client)
            tool_list.extend(client.list_tools_sync())
    except Exception:
        for c in entered:
            try:
                c.__exit__(None, None, None)
            except Exception:
                pass
        raise
    return tool_list


def create_single_agent(tool_list=None, mcp_clients=None):
    """Create a single-agent setup with all tools directly available.

    Use this for simple interactive sessions where one agent handles
    both search and synthesis.

    Args:
        tool_list: Pre-built list of MCP tools.  When *None* the
            function enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.
    """
    model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tool_list,
        conversation_manager=conversation_manager,
        callback_handler=_build_callback_handler(),
    )
    return agent, mcp_clients or []


def create_multi_agent(tool_list=None, mcp_clients=None):
    """Create a planner + researcher multi-agent setup.

    The researcher agent has direct access to all MCP tools and handles
    web search/scraping.  The planner agent delegates to the researcher
    via the agent-as-tool pattern and handles strategic decomposition
    and synthesis.

    Args:
        tool_list: Pre-built list of MCP tools.  When *None* the
            function enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.

    Returns:
        Tuple of (planner_agent, researcher_agent, mcp_clients).
    """
    model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

    # Researcher: tool-capable agent that does the actual searching
    researcher = Agent(
        model=model,
        system_prompt=RESEARCHER_PROMPT,
        tools=tool_list,
        conversation_manager=SlidingWindowConversationManager(
            window_size=15,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(),
    )

    # Planner: strategic agent that delegates to the researcher
    planner = Agent(
        model=model,
        system_prompt=PLANNER_PROMPT,
        tools=[
            researcher.as_tool(
                name="researcher",
                description=(
                    "Deep web research agent with access to Brave Search, "
                    "Firecrawl, Exa, and Kagi. Delegate any web search, "
                    "scrape, or data retrieval task to this tool."
                ),
            ),
        ],
        conversation_manager=conversation_manager,
        callback_handler=_build_callback_handler(),
    )
    return planner, researcher, mcp_clients or []


def _cleanup_mcp(mcp_clients):
    """Gracefully close all MCP client connections."""
    for client in mcp_clients:
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass


def main():
    """Interactive REPL for the Venice GLM-4.7 agent."""
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    _setup_otel()

    # Choose mode based on --multi flag
    multi_agent = "--multi" in sys.argv

    if multi_agent:
        print("Venice GLM-4.7 Uncensored Research Agent (Strands — Multi-Agent)")
        agent, _researcher, mcp_clients = create_multi_agent()
    else:
        print("Venice GLM-4.7 Uncensored Research Agent (Strands)")
        agent, mcp_clients = create_single_agent()

    tool_count = len(agent.tool_registry.get_all_tools_config())
    print(f"Tools loaded: {tool_count}")
    print("Type 'quit' to exit.\n")

    try:
        while True:
            try:
                query = input("You: ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            response = agent(query)
            print(f"\nAgent: {response}\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _cleanup_mcp(mcp_clients)
        print("MCP connections closed.")


if __name__ == "__main__":
    main()
