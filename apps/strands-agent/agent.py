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
import queue
import sys
import threading
import time

from pathlib import Path

from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import (
    CompositeCallbackHandler,
    PrintingCallbackHandler,
)
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.vended_plugins.skills import AgentSkills

from config import build_model
from prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, SYSTEM_PROMPT
from tools import get_all_mcp_clients, get_native_tools

logger = logging.getLogger(__name__)

# ── Skills plugin (progressive disclosure) ────────────────────────────
# Skills are loaded lazily: only metadata (name + description) is injected
# into the system prompt at startup.  Full instructions are loaded on-demand
# when the agent calls the `skills` tool.  This keeps the context lean while
# preserving access to specialised techniques.

_SKILLS_DIR = Path(__file__).parent / "skills"


def _build_skills_plugin() -> AgentSkills | None:
    """Build the AgentSkills plugin from the skills/ directory.

    Returns None if no skills directory exists or no skills are found,
    allowing the agent to run without skills.
    """
    if not _SKILLS_DIR.is_dir():
        logger.info("No skills directory found at %s — running without skills", _SKILLS_DIR)
        return None

    try:
        plugin = AgentSkills(skills=[str(_SKILLS_DIR)])
        skills = plugin.get_available_skills()
        if skills:
            logger.info(
                "AgentSkills loaded: %s",
                ", ".join(s.name for s in skills),
            )
            return plugin
        logger.info("Skills directory exists but no valid skills found")
        return None
    except Exception:
        logger.exception("Failed to load AgentSkills plugin")
        return None

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


class StreamCapture:
    """Thread-safe callback that captures streaming tokens to a queue.

    Activate before a request to start capturing; deactivate after.
    When no queue is active, tokens are silently dropped.
    ``PrintingCallbackHandler`` is included separately in the composite
    handler so REPL users still see real-time stdout output.
    """

    def __init__(self):
        self._queue: queue.Queue | None = None
        self._lock = threading.Lock()
        self.tool_events: list[dict] = []
        self._seen_tool_ids: set[str] = set()
        self.all_text: list[str] = []
        self.response_text: list[str] = []
        self.reasoning_text: list[str] = []

    def activate(self) -> queue.Queue:
        """Start capturing. Returns queue the caller reads from."""
        with self._lock:
            q: queue.Queue = queue.Queue()
            self._queue = q
            self.tool_events.clear()
            self._seen_tool_ids.clear()
            self.all_text.clear()
            self.response_text.clear()
            self.reasoning_text.clear()
            return q

    def deactivate(self):
        """Stop capturing and send sentinel so readers know we're done."""
        with self._lock:
            if self._queue is not None:
                self._queue.put(None)
            self._queue = None

    def __call__(self, **kwargs):
        # Only accumulate data when a consumer is actively capturing
        # (i.e. activate() has been called).  This prevents unbounded
        # memory growth from /query endpoints that never activate.
        with self._lock:
            active = self._queue is not None
        if not active:
            return

        # Capture streaming text tokens (both regular data and reasoning text)
        data = kwargs.get("data", "")
        reasoning = kwargs.get("reasoningText", "")

        # Track reasoning and response text separately.
        # all_text = everything (for logging); response_text = data only (for answer fallback)
        if reasoning and isinstance(reasoning, str):
            self.all_text.append(reasoning)
            self.reasoning_text.append(reasoning)
            with self._lock:
                if self._queue is not None:
                    self._queue.put(("thinking", reasoning))

        if data and isinstance(data, str):
            self.all_text.append(data)
            self.response_text.append(data)
            with self._lock:
                if self._queue is not None:
                    self._queue.put(("text", data))

        # Capture tool invocations (deduplicated by toolUseId)
        # Tools can come via either 'current_tool_use' or 'event.contentBlockStart'
        tool_use = kwargs.get("current_tool_use")
        if not tool_use or not tool_use.get("name"):
            tool_use = (
                kwargs.get("event", {})
                .get("contentBlockStart", {})
                .get("start", {})
                .get("toolUse")
            )
        if tool_use and tool_use.get("name"):
            tid = tool_use.get("toolUseId", "")
            if tid and tid not in self._seen_tool_ids:
                self._seen_tool_ids.add(tid)
                event = {
                    "tool": tool_use["name"],
                    "input": str(tool_use.get("input", {}))[:500],
                    "time": time.time(),
                }
                self.tool_events.append(event)
                with self._lock:
                    if self._queue is not None:
                        self._queue.put(("tool", event))


# Global stream-capture instance shared by all agents
stream_capture = StreamCapture()


def _build_callback_handler():
    """Build a composite callback handler: printing + streaming capture + budget guardrail."""
    return CompositeCallbackHandler(PrintingCallbackHandler(), stream_capture, budget_callback)


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


def _build_tool_list(mcp_tools):
    """Assemble the full tool list with uncensored-first ordering.

    Tool ordering matters — LLMs naturally prefer tools listed earlier.
    Order: Tier 1 uncensored natives → MCP tools (Brave, Exa, Semantic
    Scholar, arXiv, Wikipedia, etc.) → Tier 2 extraction → Deep research
    → Research management → Tier 3 censored fallback.

    Args:
        mcp_tools: Tools collected from MCP clients via list_tools_sync().

    Returns:
        Combined tool list ordered for uncensored-first preference.
    """
    native = get_native_tools()
    from tools import (
        NATIVE_TOOLS_TIER1,
        NATIVE_TOOLS_TIER3,
        NATIVE_TOOLS_DEEP_RESEARCH,
        NATIVE_TOOLS_RESEARCH_MGMT,
    )

    def _tool_name(t):
        return t.tool_name if hasattr(t, "tool_name") else t.__name__

    tier1_names = {_tool_name(t) for t in NATIVE_TOOLS_TIER1}
    tier3_names = {_tool_name(t) for t in NATIVE_TOOLS_TIER3}
    deep_names = {_tool_name(t) for t in NATIVE_TOOLS_DEEP_RESEARCH}
    mgmt_names = {_tool_name(t) for t in NATIVE_TOOLS_RESEARCH_MGMT}
    special_names = tier1_names | tier3_names | deep_names | mgmt_names

    native_first = [t for t in native if _tool_name(t) in tier1_names]
    native_mid = [t for t in native if _tool_name(t) not in special_names]  # Tier 2 only
    native_deep = [t for t in native if _tool_name(t) in deep_names]
    native_mgmt = [t for t in native if _tool_name(t) in mgmt_names]
    native_last = [t for t in native if _tool_name(t) in tier3_names]

    return [
        *native_first,   # Tier 1: duckduckgo_search, mojeek_search
        *mcp_tools,      # MCP: Brave, Exa, Semantic Scholar, arXiv, Wikipedia, etc.
        *native_mid,     # Tier 2: jina_read_url
        *native_deep,    # Deep research: perplexity, grok, tavily, exa_multi
        *native_mgmt,    # Research mgmt: findings store, knowledge graph
        *native_last,    # Tier 3: google_search (censored fallback — always last)
    ]


def create_single_agent(tool_list=None, mcp_clients=None):
    """Create a single-agent setup with all tools directly available.

    Use this for simple interactive sessions where one agent handles
    both search and synthesis.  Tools are ordered uncensored-first.
    The AgentSkills plugin is attached when skills are available,
    enabling progressive disclosure of specialised techniques.

    Args:
        tool_list: Pre-built list of tools.  When *None* the function
            enters its own MCP clients and builds the list (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.
    """
    model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        mcp_tools = _enter_mcp_clients(mcp_clients)
        tool_list = _build_tool_list(mcp_tools)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

    plugins = []
    skills_plugin = _build_skills_plugin()
    if skills_plugin is not None:
        plugins.append(skills_plugin)

    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tool_list,
        conversation_manager=conversation_manager,
        callback_handler=_build_callback_handler(),
        plugins=plugins or None,
    )
    return agent, mcp_clients or []


def create_multi_agent(tool_list=None, mcp_clients=None):
    """Create a planner + researcher multi-agent setup.

    The researcher agent has direct access to all tools (ordered
    uncensored-first) and handles web search/scraping.  The planner
    agent delegates to the researcher via the agent-as-tool pattern
    and handles strategic decomposition and synthesis.

    Args:
        tool_list: Pre-built list of tools.  When *None* the function
            enters its own MCP clients and builds the list (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.

    Returns:
        Tuple of (planner_agent, researcher_agent, mcp_clients).
    """
    model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        mcp_tools = _enter_mcp_clients(mcp_clients)
        tool_list = _build_tool_list(mcp_tools)

    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True,
    )

    plugins = []
    skills_plugin = _build_skills_plugin()
    if skills_plugin is not None:
        plugins.append(skills_plugin)

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
        plugins=plugins or None,
    )

    # Planner: strategic agent that delegates to the researcher
    planner = Agent(
        model=model,
        system_prompt=PLANNER_PROMPT,
        tools=[
            researcher.as_tool(
                name="researcher",
                description=(
                    "Deep web research agent with uncensored-first tool "
                    "priority: DuckDuckGo, Brave, Exa, Mojeek for search; "
                    "Jina Reader, Firecrawl, Kagi for extraction; "
                    "Semantic Scholar, arXiv for academic papers; "
                    "Wikipedia, DuckDB for reference data; "
                    "Perplexity, Grok, Tavily for deep research; "
                    "store_finding/read_findings + knowledge graph for "
                    "persisting research; Google/Serper as censored fallback. "
                    "Delegate any web search, scrape, or data retrieval task "
                    "to this tool."
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
