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
import threading
import time

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import (
    CompositeCallbackHandler,
    PrintingCallbackHandler,
)
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.vended_plugins.skills import AgentSkills

from config import build_model, build_model_with_selection
from prompts import RESEARCHER_PROMPT, SYSTEM_PROMPT
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
#
# Each research task in the AsyncTaskPool owns its own ResearcherBudget
# instance so parallel researchers don't clobber each other's counters.
# The single-agent /query path uses a module-level ``_default_budget``
# that is reset per request (see ``reset_budget``).

_MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "200"))
_SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))


@dataclass
class ResearcherBudget:
    """Per-instance budget + cancellation state for a researcher agent.

    The ``callback`` bound method plugs into Strands' composite callback
    handler. It is safe to use multiple instances concurrently from
    different threads — each instance's counters and ``seen_tool_use_ids``
    set are independent.
    """

    session_start: float = field(default_factory=time.time)
    tool_call_count: int = 0
    seen_tool_use_ids: set[str] = field(default_factory=set)
    max_tool_calls: int = field(default_factory=lambda: _MAX_TOOL_CALLS)
    session_timeout: int = field(default_factory=lambda: _SESSION_TIMEOUT)
    cancel_flag: threading.Event | None = None

    def reset(self) -> None:
        """Reset counters so a long-lived holder can reuse this budget."""
        self.session_start = time.time()
        self.tool_call_count = 0
        self.seen_tool_use_ids = set()

    def callback(self, **kwargs) -> None:
        """Callback-handler guardrail that counts actual tool invocations.

        Strands fires this callback for every streaming event. We detect
        new tool calls by checking for ``current_tool_use`` with a tool
        ``name`` and a unique ``toolUseId`` — each unique ID counted once.

        Also checks ``cancel_flag`` on every event (not just tool calls)
        so the agent loop can be aborted promptly during async job
        cancellation.
        """
        if self.cancel_flag is not None and self.cancel_flag.is_set():
            from jobs import JobCancelledError

            logger.info("cancel flag set, aborting agent loop")
            raise JobCancelledError("Job cancelled by user")

        tool_use = kwargs.get("current_tool_use")
        if not tool_use or not tool_use.get("name"):
            return

        tool_use_id = tool_use.get("toolUseId", "")
        if tool_use_id in self.seen_tool_use_ids:
            return
        self.seen_tool_use_ids.add(tool_use_id)

        self.tool_call_count += 1

        elapsed = time.time() - self.session_start
        if elapsed > self.session_timeout:
            logger.warning(
                "Session timeout reached (%.0fs > %ds). Consider wrapping up.",
                elapsed,
                self.session_timeout,
            )

        if self.tool_call_count > self.max_tool_calls:
            logger.warning(
                "Tool call budget exceeded (%d > %d). Consider wrapping up.",
                self.tool_call_count,
                self.max_tool_calls,
            )

        if self.tool_call_count % 10 == 0:
            logger.info(
                "Budget: %d tool calls, %.0fs elapsed",
                self.tool_call_count,
                elapsed,
            )


# Default budget used by the single-agent /query path. Research tasks
# get their own ``ResearcherBudget`` via ``create_researcher_instance``.
_default_budget = ResearcherBudget()


def set_cancel_flag(flag: threading.Event | None) -> None:
    """Set or clear the cancel flag on the default (single-agent) budget.

    Task-pool researcher instances carry their own ``cancel_flag`` via
    their own ``ResearcherBudget`` and do not touch this function.
    """
    _default_budget.cancel_flag = flag


def reset_budget() -> None:
    """Reset counters on the default (single-agent) budget.

    Call this before each HTTP request on the single-agent path so
    budgets don't accumulate across requests in a long-running server.
    """
    _default_budget.reset()


def budget_callback(**kwargs) -> None:
    """Module-level wrapper so the default budget plugs into callback handlers."""
    _default_budget.callback(**kwargs)


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


def _build_callback_handler(budget: ResearcherBudget | None = None):
    """Build a composite callback handler: printing + streaming capture + budget guardrail.

    When ``budget`` is None the module-level ``_default_budget`` is used
    (single-agent path). Research-task instances pass their own budget
    so parallel researchers don't share counters.
    """
    budget_cb = budget.callback if budget is not None else budget_callback
    return CompositeCallbackHandler(PrintingCallbackHandler(), stream_capture, budget_cb)


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
    """Enter MCP client contexts and collect tools, skipping failed clients.

    Each client is entered independently.  If a client's ``__enter__()`` or
    ``list_tools_sync()`` raises, that client is cleaned up and skipped —
    the remaining clients still load normally.  This prevents a single
    flaky MCP server (e.g. TranscriptAPI) from killing ALL MCP tools.

    Failed clients are removed from *mcp_clients* in-place so that the
    caller's shutdown cleanup (``_cleanup_mcp``) only calls ``__exit__``
    on clients that are in a properly-entered state.
    """
    entered: list = []
    tool_list: list = []
    for client in mcp_clients:
        enter_succeeded = False
        try:
            client.__enter__()
            enter_succeeded = True
            tool_list.extend(client.list_tools_sync())
            entered.append(client)
        except Exception:
            logger.warning(
                "MCP client failed to initialise, skipping: %s",
                getattr(client, "_transport_factory", client),
                exc_info=True,
            )
            # Only call __exit__ if __enter__ succeeded (context manager protocol)
            if enter_succeeded:
                try:
                    client.__exit__(None, None, None)
                except Exception:
                    pass
    # Remove failed clients so _cleanup_mcp at shutdown won't double-exit them
    mcp_clients[:] = entered
    return tool_list


def _build_tool_list(mcp_tools, censored_mcp_tools=None):
    """Assemble the full tool list with uncensored-first ordering.

    Tool ordering matters — LLMs naturally prefer tools listed earlier.
    Order: Tier 1 uncensored natives → uncensored MCP tools (Brave,
    Semantic Scholar, arXiv, Wikipedia, etc.) → Tier 2 extraction →
    Deep research → Research management → Tier 3 censorship-sensitive
    (Google, Exa MCP, exa_multi_search).

    Args:
        mcp_tools: Tools collected from uncensored MCP clients.
        censored_mcp_tools: Tools from censorship-sensitive MCP clients
            (e.g. Exa). Placed last alongside Tier 3 natives.

    Returns:
        Combined tool list ordered for uncensored-first preference.
    """
    if censored_mcp_tools is None:
        censored_mcp_tools = []

    native = get_native_tools()
    from tools import (
        NATIVE_TOOLS_CENSORED,
        NATIVE_TOOLS_TIER1,
        NATIVE_TOOLS_DEEP_RESEARCH,
        NATIVE_TOOLS_RESEARCH_MGMT,
    )

    def _tool_name(t):
        return t.tool_name if hasattr(t, "tool_name") else t.__name__

    tier1_names = {_tool_name(t) for t in NATIVE_TOOLS_TIER1}
    censored_names = {_tool_name(t) for t in NATIVE_TOOLS_CENSORED}
    deep_names = {_tool_name(t) for t in NATIVE_TOOLS_DEEP_RESEARCH}
    mgmt_names = {_tool_name(t) for t in NATIVE_TOOLS_RESEARCH_MGMT}
    special_names = tier1_names | censored_names | deep_names | mgmt_names

    native_first = [t for t in native if _tool_name(t) in tier1_names]
    native_mid = [t for t in native if _tool_name(t) not in special_names]  # Tier 2 only
    native_deep = [t for t in native if _tool_name(t) in deep_names]
    native_mgmt = [t for t in native if _tool_name(t) in mgmt_names]
    native_censored = [t for t in native if _tool_name(t) in censored_names]

    # Deduplicate: MCP tools take precedence over native tools with the same
    # name (e.g. TranscriptAPI MCP's search_youtube vs native REST fallback).
    # Native fallbacks are only kept when the MCP version is absent.
    all_mcp = mcp_tools + censored_mcp_tools
    mcp_names = {_tool_name(t) for t in all_mcp}

    def _dedup(tools: list) -> list:
        return [t for t in tools if _tool_name(t) not in mcp_names]

    return [
        *_dedup(native_first),    # Tier 1: duckduckgo_search, mojeek_search
        *mcp_tools,               # Uncensored MCP: Brave, Semantic Scholar, arXiv, etc.
        *_dedup(native_mid),      # Tier 2: jina_read_url, YouTube tools (deduped)
        *_dedup(native_deep),     # Deep research: perplexity, grok, tavily
        *_dedup(native_mgmt),     # Research mgmt: findings store, knowledge graph
        *_dedup(native_censored), # Censorship-sensitive native: google_search, exa_multi
        *censored_mcp_tools,      # Censorship-sensitive MCP: Exa (rejects health/PED)
    ]


def create_single_agent(tool_list=None, mcp_clients=None, user_query=None):
    """Create a single-agent setup with all tools directly available.

    Use this for simple interactive sessions where one agent handles
    both search and synthesis.  Tools are ordered uncensored-first.
    The AgentSkills plugin is attached when skills are available,
    enabling progressive disclosure of specialised techniques.

    When *user_query* is provided and MODEL_SELECTION=runtime, the model
    is selected via runtime censorship probing.  Otherwise falls back
    to the static VENICE_MODEL.

    Args:
        tool_list: Pre-built list of tools.  When *None* the function
            enters its own MCP clients and builds the list (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.
        user_query: Optional query for runtime model selection.
    """
    if user_query:
        model, selection = build_model_with_selection(user_query)
        if selection:
            logger.info("Runtime model selected: %s", selection.label)
    else:
        model = build_model()
    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients, censored_clients = get_all_mcp_clients()
        mcp_tools = _enter_mcp_clients(mcp_clients)
        censored_tools = _enter_mcp_clients(censored_clients)
        tool_list = _build_tool_list(mcp_tools, censored_tools)
        # Merge so caller's _cleanup_mcp closes all entered clients
        mcp_clients.extend(censored_clients)

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


def create_researcher_instance(
    tools: list,
    budget: ResearcherBudget | None = None,
    user_query: str | None = None,
):
    """Create a fresh researcher Agent with its own budget + conversation history.

    Used by ``AsyncTaskPool.launch_research`` to spawn an isolated
    researcher per task. MCP clients are NOT created here — they are
    shared connection pools entered once at startup; the caller passes
    their already-bound tool list.

    Args:
        tools: Shared tool list (MCP + native), built once at startup.
        budget: Per-task budget tracker. A fresh one is created when None.
        user_query: Optional query for runtime model selection.

    Returns:
        A Strands ``Agent`` with RESEARCHER_PROMPT, an independent
        conversation history, and a composite callback handler bound
        to the provided ``budget``.
    """
    if budget is None:
        budget = ResearcherBudget()

    if user_query:
        model, selection = build_model_with_selection(user_query)
        if selection:
            logger.info("Runtime model selected: %s", selection.label)
    else:
        model = build_model()

    agent = Agent(
        model=model,
        system_prompt=RESEARCHER_PROMPT,
        tools=list(tools),
        conversation_manager=SlidingWindowConversationManager(
            window_size=15,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(budget),
    )
    return agent


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

    print("Miro Research Agent (Strands — single-agent mode)")
    print("For multi-agent research, use POST /query/multi via the API.")
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
