# Async Orchestrator Migration Manifest

This document is the **source of truth** for migrating `apps/strands-agent/`
from a blocking orchestrator to an orchestrator-centric architecture with
parallel, branching, and merging async heterogeneous jobs.

If design decisions change during implementation, update this manifest
**first**, then the code.

---

## Section 1 — Architecture Overview

Three-layer architecture:

```
┌────────────────────────────────────────────────────────────────┐
│  deepagents orchestrator  (planning, TodoList, Skills, Summ.)  │
│  └── launches heterogeneous background tasks ──────┐           │
└────────────────────────────────┬───────────────────┼───────────┘
                                 │                   │
                     ┌───────────▼───────────┐  ┌────▼─────────┐
                     │  AsyncTaskPool        │  │ Corpus tools │
                     │  (ThreadPoolExecutor) │  │ (read-only,  │
                     │                       │  │  instant)    │
                     └───┬───────────┬───────┘  └──────┬───────┘
                         │           │                 │
            ┌────────────▼───┐   ┌───▼──────────┐      │
            │ Strands        │   │ GossipSwarm  │      │
            │ researcher(s)  │   │ synthesis    │      │
            │ (tool exec)    │   │              │      │
            └───┬────────────┘   └──────┬───────┘      │
                │                       │              │
                │   ┌───────────────────▼──────────────▼────┐
                └──▶│  ConditionStore (DuckDB, per-job)     │
                    │  single shared state                  │
                    └───────────────────────────────────────┘
```

Key rules:
- **deepagents = planning layer.** Middleware stack (TodoListMiddleware,
  SummarizationMiddleware, SkillsMiddleware) stays. The orchestrator
  plans, launches tasks, monitors progress, reads corpus state.
- **Strands = tool execution layer.** The Strands researcher agent is what
  actually calls MCP servers, scrapers, transcript APIs, etc. The
  orchestrator never calls tools directly — it delegates via tasks.
- **GossipSwarm = analysis layer.** Unchanged. Consumes corpus text,
  emits synthesis reports, stored back in corpus as `row_type='synthesis'`.
- **All long-running work happens via an AsyncTaskPool.** The orchestrator
  never blocks on a tool call. It launches, checks, and awaits.
- **The ConditionStore is the single shared state (per-job, DuckDB).**
  All task results auto-ingest. Gossip reads from it. Reports build from it.
- **The orchestrator is swappable** via a `ResearchOrchestrator` protocol.
  Current backend is LangChain/deepagents; future backends (pure Strands,
  custom, etc.) slot in by implementing the protocol.

---

## Section 2 — Current State (What Exists)

Every Python file in `apps/strands-agent/` and its role:

| File | Role |
|------|------|
| `main.py` | FastAPI server: job lifecycle, SSE streaming, OpenAI-compatible endpoints, lifespan wiring. Holds `_research_lock`, `_researcher_agent` singleton, `_job_cancel_event` contextvar. |
| `orchestrator.py` | deepagents orchestrator factory — `create_orchestrator()` returns `CompiledStateGraph` via `create_deep_agent()`. Holds `ORCHESTRATOR_PROMPT` and `build_venice_model()`. |
| `agent.py` | Strands Agent factories (`create_single_agent`, `create_researcher_agent`), module-level budget tracking (`_tool_call_count`, `_seen_tool_use_ids`, `_cancel_flag`), `StreamCapture`, MCP client entry/cleanup, OTEL setup. |
| `corpus.py` | `ConditionStore` (DuckDB, 30+ column schema with gradient scoring). Single unified store (post PR #154 — LineageStore merged in). |
| `corpus_tools.py` | Tools for orchestrator: `query_corpus`, `assess_coverage`, `get_gap_analysis`, `trigger_gossip` (blocking), `build_report`. Per-task contextvar `_current_store`. |
| `swarm_bridge.py` | Bridge to GossipSwarm engine. Venice API `CompleteFn` wrappers (`worker_complete`, `queen_complete`, `serendipity_complete`). Exposes `gossip_synthesize()` async entrypoint. |
| `jobs.py` | `JobState` dataclass, `JobStore` registry, SSE event queue, `JobCancelledError`, cancellation signalling. |
| `tools.py` | MCP client configs (Brave, Firecrawl, Exa, Kagi, Semantic Scholar, arXiv, Wikipedia, TranscriptAPI, etc.), native tool definitions, tier lists. |
| `youtube_tools.py` | YouTube intelligence tools (transcript download, bulk transcribe, channel harvest with Apify/Bright Data/yt-dlp cascade, comments). Houses `youtube_harvest_channel` (10+ minute blocking tool). |
| `config.py` | Model builder (Venice API via OpenAI-compatible, `build_model`, `build_model_with_selection`). |
| `prompts.py` | System prompts for researcher and single agent (`SYSTEM_PROMPT`, `RESEARCHER_PROMPT`). |
| `mcp_configs.py` | MCP server configurations for all 25+ servers (stdio/SSE/HTTP transports). |
| `cache.py` | Persistent cache for transcripts, scraped pages, LLM responses. |
| `datalake.py` | Persistent data lake for large artefacts (harvested channels, book content, etc.). |
| `forum_tools.py` | Forum scraping (MesoRx, EliteFitness, international forums). |
| `document_tools.py` | PDF / EPUB / document extraction and analysis. |
| `extraction.py` | Structured data extraction helpers. |
| `government_tools.py` | Government data tools (FOIA, regulations, etc.). |
| `integrity_tools.py` | Scientific integrity / retraction / contradiction tools. |
| `knowledge_tools.py` | Knowledge graph management tools. |
| `osint_tools.py` | OSINT tools (Tor/dark web, people search, etc.). |
| `preprint_tools.py` | arXiv, bioRxiv, medRxiv search + ingestion. |
| `atomizer.py` | Atomic condition extraction from raw text. |
| `model_selector.py` | Runtime censorship probing + model selection. |
| `book_pipeline.py` | Full-book ingestion + chapterisation pipeline. |

---

## Section 3 — Problems with Current Architecture

1. **`run_research` is blocking.** Holds `_research_lock`, one research
   task at a time. `main.py:135`.
2. **`trigger_gossip` is blocking.** Runs full 6-worker × 3-round swarm
   pipeline synchronously inside a `ThreadPoolExecutor(max_workers=1)`.
   `corpus_tools.py:208–234`.
3. **YouTube harvest blocks the orchestrator.** `youtube_harvest_channel`
   can take 10+ minutes (Apify polling). If the orchestrator calls it
   directly via the researcher, nothing else can run.
4. **No parallel task execution.** The orchestrator calls tools sequentially
   through LangGraph's tool-node.
5. **`_research_lock = threading.Lock()` is global.** Prevents concurrent
   researcher invocations even if we wanted them.
6. **The researcher agent is a module-level singleton.** `_researcher_agent`
   is created once in `lifespan()` and re-used. Can't run multiple
   instances in parallel without corrupting each other's `messages` buffer.
7. **Budget tracking uses module-level globals.** `_tool_call_count`,
   `_seen_tool_use_ids`, `_session_start` in `agent.py` — not per-task
   safe, leaks between concurrent researcher calls.
8. **Cancel bridge is global.** The `_job_cancel_event` contextvar is
   per-job but `_cancel_flag` (the threading.Event the budget_callback
   reads) is a module-level global in `agent.py`.

---

## Section 4 — New Architecture: AsyncTaskPool

New module `apps/strands-agent/task_pool.py`:

```python
# apps/strands-agent/task_pool.py
"""Async task pool for parallel heterogeneous background work.

The orchestrator launches tasks via launch_*() tools. Each task runs
in its own thread (Strands agents are sync) with its own agent instance.
Results are auto-ingested into the shared ConditionStore.

Task types:
- research: Strands researcher agent doing web/forum/academic search
- harvest:  YouTube channel bulk download (Apify/yt-dlp cascade)
- gossip:   GossipSwarm synthesis on current corpus
- ingest:   Knowledge engine ETL pipeline (future)
"""

@dataclass
class TaskState:
    task_id: str
    task_type: str              # research | harvest | gossip | ingest
    description: str
    status: str = "pending"     # pending | running | complete | failed | cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    result_summary: str = ""
    error: str | None = None
    progress: str = ""
    ingested_count: int = 0     # conditions ingested into ConditionStore


class AsyncTaskPool:
    """Manages concurrent background tasks for the orchestrator.

    One pool per /query/multi job. Each pool owns:
    - A ThreadPoolExecutor (max_concurrent workers)
    - A task registry (task_id -> TaskState)
    - A reference to the job's ConditionStore (for auto-ingest)
    - A reference to the job's event queue (for task_* event forwarding)
    - A list of shared MCP tools (passed to per-task researcher agents)
    - A cancel event (bridged from the job's asyncio cancel_event)
    """

    def __init__(
        self,
        store: ConditionStore,
        tools: list,
        job_cancel_event: threading.Event | None = None,
        event_emit: Callable[[dict], None] | None = None,
        max_concurrent: int = 4,
    ): ...

    def launch_research(self, task_desc: str) -> str:
        """Launch a research task. Returns task_id immediately.

        Creates a FRESH Strands researcher agent (via
        create_researcher_instance) with its own ResearcherBudget and
        cancel flag — NOT the module-level singleton.
        """

    def launch_harvest(
        self, channel: str, max_videos: int = 0, language: str = "en",
    ) -> str:
        """Launch a YouTube harvest task. Returns task_id immediately."""

    def launch_gossip(self, iteration: int = 0) -> str:
        """Launch a gossip synthesis task. Returns task_id immediately."""

    def check_tasks(self) -> list[dict]:
        """Return status snapshot of all tasks."""

    def await_tasks(
        self, task_ids: list[str], timeout: float = 600,
    ) -> list[dict]:
        """Block until specified tasks complete (or timeout). Returns results."""

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""

    def shutdown(self) -> None:
        """Cancel all in-flight work and shut down the executor."""
```

### Auto-ingest rules per task type

- **research**: writer thread captures the agent's final response and
  calls `store.ingest_raw(text, source_type="researcher", source_ref=task_desc[:200])`.
  `TaskState.ingested_count = len(ids)`.
- **harvest**: after `youtube_harvest_channel` returns, the task pool
  walks the cache for the just-harvested channel and calls
  `store.ingest_raw(transcript, source_type="youtube", source_ref=video_url)`
  per video. `TaskState.ingested_count` is the total number of ingested
  conditions.
- **gossip**: calls `gossip_synthesize(corpus, query)`, writes the result
  via `store.admit_synthesis(report, iteration, metrics)`.
  `TaskState.ingested_count = 1`.

### Cancellation

- Each task has its own `threading.Event` for fine-grained cancellation
  (stored inside the closure capturing the task's agent/budget).
- The pool subscribes to the job-level cancel event: when the job is
  cancelled, `AsyncTaskPool.shutdown()` sets each task's cancel event
  and the executor is shut down with `cancel_futures=True`.
- Per-task `cancel_task(task_id)` sets only that task's cancel event.

### Event forwarding

When `event_emit` is provided, the pool emits:
- `{"type": "task_launched", "task_id": ..., "task_type": ..., "description": ...}`
- `{"type": "task_progress", "task_id": ..., "progress": ...}`  (optional, sparse)
- `{"type": "task_completed", "task_id": ..., "ingested_count": ..., "summary": ...}`
- `{"type": "task_failed", "task_id": ..., "error": ...}`
- `{"type": "task_cancelled", "task_id": ...}`

These are forwarded into the job's SSE queue in `main.py`.

---

## Section 5 — New Orchestrator Tools

New module `apps/strands-agent/task_tools.py`:

```python
# apps/strands-agent/task_tools.py
"""Async task tools for the orchestrator.

These replace the blocking run_research and trigger_gossip tools
with non-blocking launch_*() / check_tasks() / await_tasks() pattern.
All tools resolve the per-job AsyncTaskPool via a contextvar (same
pattern as corpus_tools._current_store).
"""

def launch_research(task: str) -> str:
    """Launch a background research task. Returns immediately with task ID.

    The researcher agent will execute the task in a background thread,
    automatically ingesting findings into the corpus when complete.

    Use check_tasks() to monitor progress, or await_tasks() to wait
    for completion.

    Args:
        task: Detailed description of what to research.

    Returns:
        Task ID string for tracking.
    """

def launch_harvest(channel: str, max_videos: int = 0, language: str = "en") -> str:
    """Launch a YouTube channel harvest in the background.

    Downloads transcripts and comments from the channel using
    Apify/Bright Data/yt-dlp cascade. This is a LONG-RUNNING task
    (potentially 10+ minutes for large channels).

    Results are cached and auto-ingested into the corpus.
    """

def launch_gossip(iteration: int = 0) -> str:
    """Launch gossip swarm synthesis in the background.

    Exports the current corpus, runs the 6-worker gossip swarm
    with 3 rounds, and stores the synthesis back in the corpus.
    """

def check_tasks() -> str:
    """Check status of all background tasks.

    Returns a structured summary of all running, completed, and
    failed tasks with their progress and result summaries.

    Use this to monitor parallel work and decide what to do next.
    """

def await_tasks(task_ids: str, timeout: float = 600) -> str:
    """Wait for specific background tasks to complete.

    Blocks until all specified tasks finish (or timeout).
    Returns results and ingestion summaries for each task.

    Args:
        task_ids: JSON array of task ID strings to wait for.
        timeout: Maximum seconds to wait (default 600).
    """
```

Tools are **plain Python callables** so that `create_deep_agent()` can
bind them via its automatic `@tool` inference (same pattern used today
for `corpus_tools`).

---

## Section 6 — Per-Task Researcher Agent Instances

The module-level singleton pattern must go. Concretely:

### Changes in `agent.py`

1. **Delete module globals:**
   - `_session_start`
   - `_tool_call_count`
   - `_seen_tool_use_ids`
   - `_cancel_flag`
   - `set_cancel_flag()`
   - `reset_budget()` (replaced by constructing a fresh `ResearcherBudget`)
2. **Add `ResearcherBudget` dataclass:**

   ```python
   @dataclass
   class ResearcherBudget:
       session_start: float = field(default_factory=time.time)
       tool_call_count: int = 0
       seen_tool_use_ids: set[str] = field(default_factory=set)
       max_tool_calls: int = int(os.environ.get("MAX_TOOL_CALLS", "200"))
       session_timeout: int = int(os.environ.get("SESSION_TIMEOUT", "3600"))
       cancel_flag: threading.Event | None = None

       def callback(self, **kwargs) -> None:
           """budget_callback as a bound method — per-instance state."""
   ```

3. **Add `create_researcher_instance(tools, cancel_event=None)` factory.**
   Returns a fresh `Agent` with its own conversation history, its own
   `ResearcherBudget`, its own composite callback handler. MCP clients
   are **shared** (thread-safe connection pools entered once at startup).
4. **`create_single_agent` keeps its existing shape** (still used by
   `/query` — single-turn single-thread path).
5. **`reset_budget()` is removed.** `/query` and `/v1/chat/completions`
   single-agent path create a new budget per request by wrapping the
   single agent's callback handler. The single agent can still be reused
   across requests; what changes is that the budget is now request-scoped
   via a closure.

### Changes in `main.py`

- Remove `_research_lock = threading.Lock()`.
- Remove `_job_cancel_event` contextvar (replaced by passing the job's
  threading cancel event into the `AsyncTaskPool`).
- Remove the `run_research`, `_invoke_researcher` functions (replaced
  by the pool's `launch_research`).
- The module-level `_researcher_agent` singleton becomes unused and is
  removed. We keep only `_single_agent` (for `/query`) and the MCP
  client list + tool list.

### Budget-per-request for `/query`

The single-agent path still needs a budget. Approach:
- `create_single_agent()` accepts an optional `budget: ResearcherBudget`.
  When None, the agent uses a fresh one. For the long-lived `_single_agent`
  used by `/query`, we swap its budget by reassigning the relevant
  callback handler component before each request (mirroring today's
  `reset_budget()` pattern, but instance-scoped).

---

## Section 7 — Updated Orchestrator Factory

`apps/strands-agent/orchestrator.py`:

- `create_orchestrator()` signature changes:
  ```python
  def create_orchestrator(
      task_pool: AsyncTaskPool,
      corpus_tools: Sequence[Callable],   # query/assess/gap/report (read-only)
      skills_paths: list[str] | None = None,
      model: ChatOpenAI | None = None,
  ) -> ResearchOrchestrator:
  ```
- Tool list becomes:
  ```python
  [
      launch_research,
      launch_harvest,
      launch_gossip,
      check_tasks,
      await_tasks,
      query_corpus,
      assess_coverage,
      get_gap_analysis,
      build_report,
  ]
  ```
- `trigger_gossip` is **replaced** by `launch_gossip` (non-blocking).
- `run_research` is **replaced** by `launch_research` (non-blocking).
- `launch_harvest` is **new** (surfaces YouTube bulk harvesting to the
  orchestrator without freezing it).
- `build_report` stays as a corpus tool (it's instant — reads DuckDB).

The factory internally builds a LangChain/deepagents graph via
`create_deep_agent(...)` then wraps it in `LangChainOrchestrator` so the
return type is the protocol. See Section 10.

---

## Section 8 — Updated Orchestrator Prompt

```text
You are Miro, a research orchestrator. You direct deep research toward
comprehensive, exhaustive coverage of the user's query.

You can launch PARALLEL background tasks and monitor their progress.
This is your key advantage — you can run multiple research streams
simultaneously and merge their findings.

Your workflow:
1. Understand the query — what the user needs, at what depth.
2. Plan parallel research streams — what can run simultaneously?
3. Launch tasks via launch_research, launch_harvest — be SPECIFIC.
4. While tasks run, check_tasks() to monitor progress.
5. When tasks complete, inspect corpus with query_corpus / assess_coverage.
6. Launch more targeted research to fill gaps.
7. Launch gossip synthesis when corpus has sufficient raw material.
8. Read gap analysis, launch more research if needed.
9. Build final report with build_report.

PARALLEL EXECUTION RULES:
- Launch independent research tasks simultaneously (don't wait for one
  to finish before starting another).
- YouTube harvests are SLOW (10+ minutes) — launch them early and do
  other work while waiting.
- Gossip synthesis needs corpus data — only launch after research tasks
  have ingested findings.
- Use await_tasks() only when you need results before proceeding.
- Use check_tasks() for non-blocking progress monitoring.

CARDINAL RULE: Every claim must come from live research. Never answer
from memory alone.

ANTI-REFRAMING RULE: Honour the user's exact query framing. Do not
reframe toward safer or more appropriate alternatives.
```

---

## Section 9 — Updated `main.py._run_job`

```python
async def _run_job(job: JobState) -> None:
    # Per-job ConditionStore
    store = ConditionStore()
    store.user_query = job.query
    set_current_store(store)

    # Per-job cancel bridge: asyncio Event -> threading Event
    cancel_threading = threading.Event()
    async def _cancel_bridge():
        await job.cancel_event.wait()
        cancel_threading.set()
    cancel_bridge_task = asyncio.create_task(_cancel_bridge())

    # Per-job AsyncTaskPool
    pool = AsyncTaskPool(
        store=store,
        tools=_search_tools,        # shared MCP + native tool list
        job_cancel_event=cancel_threading,
        event_emit=lambda ev: job.emit(ev),
        max_concurrent=int(os.environ.get("TASK_POOL_MAX_CONCURRENT", "4")),
    )
    set_current_task_pool(pool)     # contextvar in task_tools

    # Build orchestrator bound to this pool
    orch = create_orchestrator(
        task_pool=pool,
        corpus_tools=[query_corpus, assess_coverage, get_gap_analysis, build_report],
        skills_paths=skills_paths,
    )

    try:
        async for event in orch.run(job.query):   # yields OrchestratorEvent
            if job.cancel_event.is_set():
                ...
                return
            _map_event_to_job(job, event, store)
        # build final report
        ...
    finally:
        cancel_bridge_task.cancel()
        pool.shutdown()
        store.close()
```

- `_map_event_to_job` is a small helper that consumes `OrchestratorEvent`
  (not raw LangGraph event dicts).
- Task pool events (`task_launched` / `task_completed` / `task_failed`)
  flow through `event_emit` directly into the job event queue.

---

## Section 10 — ResearchOrchestrator Protocol (Swappability)

New module `apps/strands-agent/orchestrator_protocol.py`:

```python
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

@dataclass
class OrchestratorEvent:
    type: str           # tool_start | tool_end | task_launched | task_completed
                        # | stream | error | final
    name: str = ""
    data: dict = field(default_factory=dict)


@runtime_checkable
class ResearchOrchestrator(Protocol):
    async def run(self, query: str) -> AsyncIterator[OrchestratorEvent]:
        """Run the orchestrator on a query. Yields events."""
        ...
```

New module `apps/strands-agent/orchestrator_langchain.py`:
- `LangChainOrchestrator` — wraps a `CompiledStateGraph` and
  `astream_events()`.
- Maps LangGraph events (`on_tool_start`, `on_tool_end`,
  `on_chat_model_stream`) to `OrchestratorEvent`.
- **This is the only file outside `orchestrator.py` that imports
  LangChain / LangGraph.** No other file in `apps/strands-agent/` may
  import from `langchain_core`, `langchain_openai`, or `langgraph`
  after this migration.

`orchestrator.py` becomes a thin factory that:
1. Builds a `ChatOpenAI` Venice model.
2. Builds the tool list (see Section 7).
3. Calls `create_deep_agent(...)`.
4. Wraps the result in `LangChainOrchestrator(...)` and returns it
   typed as `ResearchOrchestrator`.

Future: `ORCHESTRATOR_BACKEND=langchain|strands|...` env var selects
the implementation. For now `langchain` is the only option.

---

## Section 11 — Migration Path (Ordered Steps)

Each phase leaves the system in a runnable state.

### Phase 1 — Foundation (no behaviour change)
1. Create `MANIFEST.md` (this document).
2. Create `orchestrator_protocol.py` with `ResearchOrchestrator`
   protocol and `OrchestratorEvent` dataclass.
3. Create `task_pool.py` with `TaskState` and a skeletal
   `AsyncTaskPool` (constructors, registry, `check_tasks`, `shutdown`
   — launch methods raise `NotImplementedError`).
4. Create `task_tools.py` with tool stubs that raise
   `RuntimeError("not yet wired")` if the contextvar is unset.

### Phase 2 — Task pool implementation
5. Implement `AsyncTaskPool.launch_research()` — fresh researcher per
   task via `create_researcher_instance`, ThreadPoolExecutor, auto-ingest.
6. Implement `AsyncTaskPool.launch_harvest()` — background
   `youtube_harvest_channel` call, walk cache, auto-ingest transcripts.
7. Implement `AsyncTaskPool.launch_gossip()` — background
   `gossip_synthesize`, `store.admit_synthesis`.
8. Implement `check_tasks()` and `await_tasks()`.

### Phase 3 — Per-task agent instances
9. Refactor `agent.py`: extract `ResearcherBudget` class, add
   `create_researcher_instance` factory, remove module-level budget globals.
10. Update `create_single_agent` to accept an optional budget.
11. MCP clients remain shared (entered once at startup, passed to all
    instances).
12. Remove `_research_lock` and `_job_cancel_event` from `main.py`.

### Phase 4 — Orchestrator migration
13. Create `orchestrator_langchain.py` — `LangChainOrchestrator`
    wrapping `CompiledStateGraph`, mapping LangGraph events to
    `OrchestratorEvent`.
14. Update `orchestrator.py`: `create_orchestrator()` accepts
    `task_pool`, returns `ResearchOrchestrator`, uses new tool list.
15. Update `ORCHESTRATOR_PROMPT`.
16. Update `_run_job()` in `main.py` to consume `OrchestratorEvent`
    instead of raw LangGraph event dicts.

### Phase 5 — Cleanup
17. Remove old `run_research()` and `_invoke_researcher()` from `main.py`.
18. Remove old `trigger_gossip()` from `corpus_tools.py`.
19. Keep `query_corpus`, `assess_coverage`, `get_gap_analysis`,
    `build_report`, `set_current_store` in `corpus_tools.py`.
20. `/query` single-agent endpoint — unchanged (uses the Strands single
    agent directly, not the orchestrator).

---

## Section 12 — Explicitly Deferred

- **HiveSwarm layered knowledge model** (L0–L3, multi-channel gossip,
  pointer network) integration into GossipSwarm — later.
- **Phase-shifting** (augmented gossip rounds with tool access) — later.
- **Thoughts table / research swarm reading deliberation state** — later.
- **Further ConditionStore refactoring** — already unified with LineageStore
  (PR #154); any additional schema changes deferred.
- **ADK agent (`apps/adk-agent/`) deprecation** — later.
- **Strands-native ResearchOrchestrator backend** — stub only, not
  implemented in this PR.

---

## Section 13 — File Inventory

### New files
- `apps/strands-agent/MANIFEST.md` — this document.
- `apps/strands-agent/task_pool.py` — `AsyncTaskPool`, `TaskState`.
- `apps/strands-agent/task_tools.py` — `launch_research`,
  `launch_harvest`, `launch_gossip`, `check_tasks`, `await_tasks`,
  `set_current_task_pool` contextvar.
- `apps/strands-agent/orchestrator_protocol.py` — `ResearchOrchestrator`
  protocol, `OrchestratorEvent`.
- `apps/strands-agent/orchestrator_langchain.py` — LangChain/deepagents
  implementation of `ResearchOrchestrator`.

### Modified files
- `apps/strands-agent/orchestrator.py` — new tool list, accept
  `task_pool`, return `ResearchOrchestrator`, updated prompt.
- `apps/strands-agent/main.py` — `_run_job` creates `AsyncTaskPool`,
  consumes `OrchestratorEvent`, removes `_research_lock`, removes
  `run_research`, removes `_researcher_agent` singleton.
- `apps/strands-agent/agent.py` — extract `ResearcherBudget`, add
  `create_researcher_instance`, keep `create_single_agent` shape,
  remove module-level budget globals.
- `apps/strands-agent/corpus_tools.py` — remove `trigger_gossip`
  (moved to `task_tools`), keep all read-only tools.
- `apps/strands-agent/pyproject.toml` — add the new module names to
  the `py-modules` list so `pip install -e` picks them up.

### Unchanged files
- `apps/strands-agent/swarm_bridge.py` — still provides
  `gossip_synthesize`, called by task pool instead of `corpus_tools`.
- `apps/strands-agent/corpus.py` — `ConditionStore` unchanged.
- `apps/strands-agent/jobs.py` — unchanged (pool emits events via the
  pre-existing `JobState.emit` pattern).
- `apps/strands-agent/tools.py` — MCP configs unchanged.
- `apps/strands-agent/youtube_tools.py` — unchanged (task pool calls
  `youtube_harvest_channel` internally).
- `apps/strands-agent/config.py`, `prompts.py`, `cache.py`,
  `datalake.py` — unchanged.
- `swarm/` directory — GossipSwarm engine unchanged.
- All other tool files (`forum_tools.py`, `osint_tools.py`, …) —
  unchanged.

---

## Section 14 — Testing Strategy

### Unit tests (fast, no network)
- `AsyncTaskPool` with mock researcher / mock harvest / mock gossip
  callables — verify parallel execution, auto-ingest call counts, task
  cancellation, `await_tasks` blocking semantics, `check_tasks` snapshot.
- `ResearchOrchestrator` protocol compliance: `LangChainOrchestrator`
  is a `runtime_checkable` instance of `ResearchOrchestrator`.
- `ResearcherBudget.callback` — raises `JobCancelledError` when
  `cancel_flag.is_set()`, increments `tool_call_count` on new
  `toolUseId`, deduplicates via `seen_tool_use_ids`.

### Integration tests (may hit network, gated)
- Launch 3 research tasks in parallel, verify all findings appear in
  `ConditionStore`.
- Launch a harvest task + research task simultaneously, verify both
  complete and both ingest into the corpus.
- Launch gossip after research, verify synthesis row stored.

### End-to-end
- Full orchestrator run through `/query/multi` — SSE stream contains
  `task_launched` / `task_completed` events, final report is non-empty,
  corpus_stats are populated.

---

## Change Log

- **v0.1** — Initial manifest authored during async-orchestrator
  migration. Captures Phases 1–5 plan, all design decisions, and the
  swappable `ResearchOrchestrator` protocol.
