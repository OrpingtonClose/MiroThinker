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
- Adaptive loop breaking (temperature escalation on repeated queries)
- OpenTelemetry observability (OTEL_EXPORTER_OTLP_ENDPOINT)
"""

from __future__ import annotations

import copy
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any

from dotenv import load_dotenv
from strands import Agent
from strands.handlers.callback_handler import (
    CompositeCallbackHandler,
    PrintingCallbackHandler,
)
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.types._events import ToolResultEvent
from strands.types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

from config import build_model
from prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, SYSTEM_PROMPT
from tools import get_all_mcp_clients

logger = logging.getLogger(__name__)

# ── Observability: tool call counter (soft, no exceptions) ───────────
# Tracks actual tool invocations for logging/metrics only.
# Loop prevention is handled by AdaptiveResearcherTool (temperature
# escalation + query deduplication), not by hard budget exceptions.

_session_start = time.time()
_tool_call_count = 0
_seen_tool_use_ids: set[str] = set()


# Keep for backward compat — main.py imports this but it is never raised.
class BudgetExceededError(Exception):
    """Legacy — kept for import compatibility.  Never raised."""


def reset_budget() -> None:
    """Reset per-request budget counters.

    Call this before each HTTP request so that budget globals don't
    accumulate across requests in a long-running server process.
    """
    global _session_start, _tool_call_count, _seen_tool_use_ids
    _session_start = time.time()
    _tool_call_count = 0
    _seen_tool_use_ids = set()
    # Also reset the adaptive researcher memory if available
    if _adaptive_researcher is not None:
        _adaptive_researcher.reset()


def budget_callback(**kwargs) -> None:
    """Callback that counts tool invocations (observability only, no exceptions).

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
    if _tool_call_count % 10 == 0:
        logger.info(
            "Budget: %d tool calls, %.0fs elapsed",
            _tool_call_count,
            elapsed,
        )


# ── Adaptive loop breaking ───────────────────────────────────────────
# Instead of hard budget exceptions, detect when the planner sends
# repeated/similar queries to the researcher and progressively
# escalate the researcher model's temperature to force divergent
# thinking.  On 3+ near-identical queries, return cached results
# with a firm instruction to synthesize.

_BASE_TEMPERATURE = float(os.environ.get("BASE_TEMPERATURE", "0.7"))
_TEMP_STEP = float(os.environ.get("TEMP_ESCALATION_STEP", "0.2"))
_MAX_TEMPERATURE = float(os.environ.get("MAX_TEMPERATURE", "1.5"))
_SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.6"))
_MAX_SIMILAR_CALLS = int(os.environ.get("MAX_SIMILAR_CALLS", "3"))

# Global reference set in create_multi_agent, read by reset_budget
_adaptive_researcher: AdaptiveResearcherTool | None = None


def _normalize_query(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class AdaptiveResearcherTool(AgentTool):
    """Wraps the researcher agent-as-tool with adaptive loop prevention.

    Instead of hard budget limits that raise exceptions, this tool:

    1. **Tracks** every query delegated to the researcher.
    2. **Detects** when the planner sends a similar query (Jaccard > threshold).
    3. **Escalates temperature** on the researcher's model for each repeat,
       forcing more creative/divergent search strategies.
    4. **Returns cached results** after ``MAX_SIMILAR_CALLS`` near-identical
       queries, with a firm instruction to stop searching and synthesize.

    This approach lets complex queries run freely (no arbitrary tool-call
    ceiling) while surgically stopping actual loops.
    """

    def __init__(
        self,
        inner_tool: AgentTool,
        researcher_model: Any,
        *,
        base_temperature: float = _BASE_TEMPERATURE,
        temp_step: float = _TEMP_STEP,
        max_temperature: float = _MAX_TEMPERATURE,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        max_similar_calls: int = _MAX_SIMILAR_CALLS,
    ) -> None:
        super().__init__()
        self._inner = inner_tool
        self._researcher_model = researcher_model
        self._base_temp = base_temperature
        self._temp_step = temp_step
        self._max_temp = max_temperature
        self._sim_threshold = similarity_threshold
        self._max_similar = max_similar_calls

        # Per-request state (cleared by reset())
        self._query_history: list[tuple[str, str]] = []  # (normalized, result)

    # ── AgentTool interface ───────────────────────────────────────

    @property
    def tool_name(self) -> str:
        return self._inner.tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        return self._inner.tool_spec

    @property
    def tool_type(self) -> str:
        return self._inner.tool_type

    # ── Similarity helpers ───────────────────────────────────────

    def _count_similar(self, normalized: str) -> int:
        """Count how many past queries are similar to *normalized*."""
        return sum(
            1
            for pq, _ in self._query_history
            if _jaccard_similarity(normalized, pq) >= self._sim_threshold
        )

    def _best_cached_result(self, normalized: str) -> str | None:
        """Return the result of the most similar past query, or None."""
        best_sim = 0.0
        best_result = None
        for pq, result in self._query_history:
            sim = _jaccard_similarity(normalized, pq)
            if sim > best_sim:
                best_sim = sim
                best_result = result
        return best_result if best_sim >= self._sim_threshold else None

    # ── Temperature control ──────────────────────────────────────

    def _set_temperature(self, temp: float) -> dict:
        """Set researcher model temperature.  Returns old params for restore."""
        old_params = copy.deepcopy(dict(self._researcher_model.config.get("params", {})))
        new_params = copy.deepcopy(old_params)
        new_params["temperature"] = min(temp, self._max_temp)
        self._researcher_model.config["params"] = new_params
        logger.info("Researcher temperature set to %.2f", new_params["temperature"])
        return old_params

    def _restore_params(self, old_params: dict) -> None:
        """Restore researcher model params after a call."""
        self._researcher_model.config["params"] = old_params

    # ── Main dispatch ────────────────────────────────────────────

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> ToolGenerator:
        """Intercept researcher calls with adaptive temperature escalation."""
        # Extract the query string the planner is sending
        tool_input = tool_use["input"]
        if isinstance(tool_input, dict):
            query = tool_input.get("input", "")
        elif isinstance(tool_input, str):
            query = tool_input
        else:
            query = str(tool_input)

        normalized = _normalize_query(query)
        similar_count = self._count_similar(normalized)

        # ── Case 1: too many similar queries → return cached result ──
        if similar_count >= self._max_similar:
            cached = self._best_cached_result(normalized) or "(no prior results)"
            logger.warning(
                "Researcher called %d times with similar query — returning "
                "cached result.  Query: %.100s",
                similar_count + 1,
                query,
            )
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use["toolUseId"],
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"[DUPLICATE QUERY — returning cached result from "
                                f"attempt {similar_count}]\n\n{cached}\n\n---\n"
                                f"You have already searched for this same topic "
                                f"{similar_count} times. STOP delegating and "
                                f"SYNTHESIZE your answer from the data above NOW."
                            )
                        }
                    ],
                }
            )
            return

        # ── Case 2: similar query detected → escalate temperature ────
        old_params = None
        if similar_count > 0:
            new_temp = self._base_temp + (similar_count * self._temp_step)
            old_params = self._set_temperature(new_temp)
            logger.info(
                "Similar query #%d detected (sim>%.2f). Temperature: %.2f → %.2f. "
                "Query: %.100s",
                similar_count + 1,
                self._sim_threshold,
                self._base_temp,
                new_temp,
                query,
            )

            # Append a diversity hint to the query
            diversity_hint = (
                "\n\n[SYSTEM NOTE: Your previous search on this topic returned "
                "limited or duplicate results.  Try COMPLETELY DIFFERENT search "
                "terms, alternative tools (e.g. Exa instead of Brave, or "
                "Firecrawl for deep scraping), different languages, or "
                "unconventional angles.  Do NOT repeat the same queries.]"
            )
            modified_input = (
                {"input": query + diversity_hint}
                if isinstance(tool_input, dict)
                else query + diversity_hint
            )
            tool_use = dict(tool_use)
            tool_use["input"] = modified_input

        # ── Delegate to the real researcher ──────────────────────
        result_text = ""
        try:
            async for event in self._inner.stream(tool_use, invocation_state, **kwargs):
                # Capture result text for caching
                if isinstance(event, ToolResultEvent):
                    tr = event.tool_result
                    for block in tr.get("content", []):
                        if isinstance(block, dict) and "text" in block:
                            result_text += block["text"]
                yield event
        finally:
            if old_params is not None:
                self._restore_params(old_params)

        # Store in history for future similarity checks
        self._query_history.append((normalized, result_text[:2000]))
        logger.info(
            "Researcher call #%d completed. %d chars returned. Query: %.80s",
            len(self._query_history),
            len(result_text),
            query,
        )

    # ── Lifecycle ────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear query history.  Called at the start of each request."""
        self._query_history.clear()


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
    """Create a planner + researcher multi-agent setup with adaptive loop prevention.

    The researcher agent has direct access to all MCP tools and handles
    web search/scraping.  The planner agent delegates to the researcher
    via an **AdaptiveResearcherTool** wrapper that:

    - Tracks query similarity across delegations
    - Escalates the researcher's sampling temperature on repeated queries
      so the model is forced to explore different search strategies
    - Returns cached results after ``MAX_SIMILAR_CALLS`` near-identical
      queries, with a firm instruction to synthesize

    This replaces the previous hard budget-exception approach with a
    behaviour-aware mechanism that lets complex queries run freely while
    surgically stopping actual loops.

    Args:
        tool_list: Pre-built list of MCP tools.  When *None* the
            function enters its own MCP clients (REPL use-case).
        mcp_clients: MCP clients that were entered to produce
            *tool_list*.  Returned as-is for the caller to manage.

    Returns:
        Tuple of (planner_agent, researcher_agent, mcp_clients).
    """
    global _adaptive_researcher

    # Separate model instances so temperature changes on the researcher
    # don't affect the planner's reasoning.
    planner_model = build_model()
    researcher_model = build_model(temperature=_BASE_TEMPERATURE)

    owns_clients = tool_list is None
    if owns_clients:
        mcp_clients = get_all_mcp_clients()
        tool_list = _enter_mcp_clients(mcp_clients)

    # Researcher: tool-capable agent that does the actual searching
    researcher = Agent(
        model=researcher_model,
        system_prompt=RESEARCHER_PROMPT,
        tools=tool_list,
        conversation_manager=SlidingWindowConversationManager(
            window_size=15,
            should_truncate_results=True,
        ),
        callback_handler=_build_callback_handler(),
    )

    # Wrap the researcher tool with adaptive loop prevention
    raw_tool = researcher.as_tool(
        name="researcher",
        description=(
            "Deep web research agent with access to Brave Search, "
            "Firecrawl, Exa, and Kagi. Delegate any web search, "
            "scrape, or data retrieval task to this tool."
        ),
    )
    adaptive_tool = AdaptiveResearcherTool(raw_tool, researcher_model)
    _adaptive_researcher = adaptive_tool

    # Planner: strategic agent that delegates to the researcher
    planner = Agent(
        model=planner_model,
        system_prompt=PLANNER_PROMPT,
        tools=[adaptive_tool],
        conversation_manager=SlidingWindowConversationManager(
            window_size=20,
            should_truncate_results=True,
        ),
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

            reset_budget()
            response = agent(query)
            print(f"\nAgent: {response}\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        _cleanup_mcp(mcp_clients)
        print("MCP connections closed.")


if __name__ == "__main__":
    main()
