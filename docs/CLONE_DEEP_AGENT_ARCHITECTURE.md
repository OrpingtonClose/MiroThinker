# Clone Deep Agent Architecture

## Status: Design Document

Architecture for wrapping tool-armed clones in a Deep Agent long-planning async shell, with orchestrator-driven retirement rules.

Supersedes the flat query-fanout pattern in `swarm/research_organizer.py`.

---

## 1. Problem Statement

### Current State (Flat Fanout)

`research_organizer.py` implements clone research as a single-pass pipeline:

```
Extract doubts → Generate queries → Fan-out search APIs → Extract content → Synthesize → Store
```

This has three fundamental weaknesses:

1. **No planning** — the clone generates 3-5 search queries from the doubt and fires them all. No strategy, no prioritization, no "what do I search for AFTER I see the first results?"

2. **No iteration** — if the first round of searches returns irrelevant results, the clone has no recourse. It synthesizes whatever it got and moves on. A human researcher would pivot: "those results were about general insulin, not the Milos protocol specifically — let me search for 'Milos Sarcev pre-workout slin dose' instead."

3. **No retirement** — every clone runs to completion regardless of whether its doubt has already been resolved by another clone, whether it's finding anything useful, or whether the swarm has moved past the issue.

### Target State (Deep Agent Shell)

Each clone is a full **Deep Agent** — a LangGraph-based planning agent with:
- The worker's accumulated conversation context as initialization
- Search and extraction tools (Brave, Exa, Tavily, Kagi, Firecrawl, Jina, arXiv, Semantic Scholar)
- A planning loop that evaluates what it found, decides what to search next, and iterates until the doubt is resolved or budget is exhausted
- Async execution — doesn't block the swarm's next wave
- Orchestrator-driven retirement signals that can kill the clone mid-research

---

## 2. Architecture Overview

```
                    ┌──────────────────────────────────────────────┐
                    │           ORCHESTRATOR (mcp_engine.py)        │
                    │                                              │
                    │  Wave N completes → worker transcripts       │
                    │           │                                   │
                    │           ▼                                   │
                    │  ┌────────────────────┐                      │
                    │  │ RESEARCH ORGANIZER  │                      │
                    │  │                    │                      │
                    │  │ 1. Read transcripts │                      │
                    │  │ 2. Extract doubts   │                      │
                    │  │ 3. Prioritize       │                      │
                    │  │ 4. Check retirement │◄──── Retirement     │
                    │  │    pre-conditions   │      Rules Engine    │
                    │  │ 5. Spawn clones     │                      │
                    │  └────────┬───────────┘                      │
                    │           │                                   │
                    │     spawn N async clones                     │
                    │           │                                   │
                    └───────────┼───────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
   ┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
   │  CLONE AGENT 1   │ │ CLONE AGENT 2│ │ CLONE AGENT N│
   │  (Deep Agent)    │ │ (Deep Agent) │ │ (Deep Agent) │
   │                  │ │              │ │              │
   │  Context:        │ │              │ │              │
   │  - Worker conv   │ │              │ │              │
   │  - Specific doubt│ │              │ │              │
   │                  │ │              │ │              │
   │  Planning loop:  │ │              │ │              │
   │  think → search  │ │              │ │              │
   │  → evaluate      │ │              │ │              │
   │  → pivot or done │ │              │ │              │
   │                  │ │              │ │              │
   │  Tools:          │ │              │ │              │
   │  - brave_search  │ │              │ │              │
   │  - exa_search    │ │              │ │              │
   │  - tavily_search │ │              │ │              │
   │  - extract_url   │ │              │ │              │
   │  - arxiv_search  │ │              │ │              │
   │  - sem_scholar   │ │              │ │              │
   │  - store_finding │ │              │ │              │
   │  - check_retire  │ │              │ │              │
   └────────┬─────────┘ └──────┬───────┘ └──────┬───────┘
            │                  │                 │
            │    async findings flow             │
            ▼                  ▼                 ▼
   ┌─────────────────────────────────────────────────────┐
   │              CONDITION STORE (DuckDB)                │
   │                                                     │
   │  source_type = 'clone_research'                     │
   │  → appears as §8 FRESH EVIDENCE in wave N+1         │
   └─────────────────────────────────────────────────────┘
```

---

## 3. Clone Deep Agent Specification

### 3.1 Creation

Each clone is instantiated as a Deep Agent via `create_deep_agent()` from the Deep Agents SDK, or (for local-only mode without LangGraph Platform) as a self-contained asyncio planning loop that mirrors the same behavior.

```python
# Conceptual — actual implementation may use create_deep_agent()
# or a lighter-weight local planning loop depending on infra

clone_agent = CloneDeepAgent(
    # Identity
    name=f"clone_{worker_id}_{doubt_hash[:8]}",
    angle=worker_angle,
    doubt=research_need,

    # Context from parent worker
    worker_context=worker_transcript,
    worker_system_prompt=worker_system_prompt,
    knowledge_summary=rolling_summary_for_angle,

    # Tools
    tools=[
        brave_search,
        exa_search,
        tavily_search,
        kagi_search,
        extract_url,        # Firecrawl / Jina
        arxiv_search,
        semantic_scholar,
        store_finding,      # Write directly to ConditionStore
        check_retirement,   # Query orchestrator retirement signal
    ],

    # Budget
    max_tool_calls=budget.max_tool_calls,       # default: 15
    max_search_api_calls=budget.max_searches,   # default: 8
    max_extraction_calls=budget.max_extracts,   # default: 3
    timeout_s=budget.timeout_s,                 # default: 180

    # Planning
    model=config.clone_model,    # Tier C model (tool-calling + uncensored)
    temperature=0.4,             # Moderate creativity for search pivots

    # Retirement
    retirement_checker=retirement_checker,
)
```

### 3.2 System Prompt

The clone's system prompt carries the worker's full reasoning context plus a research mandate:

```
You are an expert researcher in {angle}. You have been reasoning about
this domain and identified a specific gap in your understanding:

DOUBT: {doubt.doubt}
DATA NEEDED: {doubt.data_needed}

Your accumulated knowledge so far:
{knowledge_summary}

Your previous reasoning (excerpt):
{worker_transcript_tail}

YOUR MISSION: Resolve this doubt using targeted research. You have
search tools. Use them strategically:

1. PLAN first — what sources would have this data? Academic papers?
   Forum posts? Clinical data? Think about WHERE this information
   would live before searching.

2. SEARCH with expert precision — you know the domain terminology.
   Use specific terms, not generic queries. If you know the mechanism
   involves GLUT4 translocation, search for that, not "insulin muscle."

3. EVALUATE each result — does it actually address your doubt, or is
   it tangential? If tangential, PIVOT your search strategy.

4. ITERATE — if the first search doesn't resolve the doubt, refine
   your query. Follow citation chains. Try different sources.

5. STORE findings immediately when you find something relevant.
   Don't wait until the end. Use store_finding for each specific
   factual claim with its source.

6. STOP when either:
   a. The doubt is resolved (you found the specific data needed)
   b. You've exhausted your search budget
   c. You conclude the data doesn't exist in publicly available sources
   d. The retirement checker signals you to stop

Report what you found AND what you didn't find. Negative results
("this data doesn't appear to exist in published literature") are
valuable — they prevent the worker from continuing to expect data
that isn't available.
```

### 3.3 Planning Loop

Unlike the flat fanout, the Deep Agent clone runs a **plan-act-evaluate** loop:

```
┌──────────────────────────────────────────────┐
│                 CLONE LOOP                    │
│                                              │
│  ┌──────┐     ┌──────┐     ┌──────────┐     │
│  │ PLAN │────►│ ACT  │────►│ EVALUATE │     │
│  │      │     │      │     │          │     │
│  │What  │     │Search│     │Did this  │     │
│  │source│     │or    │     │resolve   │     │
│  │would │     │extract│    │the doubt?│     │
│  │have  │     │      │     │          │     │
│  │this? │     │      │     │          │     │
│  └──────┘     └──────┘     └────┬─────┘     │
│      ▲                          │            │
│      │                          │            │
│      │    NO: pivot strategy    │            │
│      ◄──────────────────────────┤            │
│                                 │            │
│                          YES: store & exit   │
│                                 │            │
│                                 ▼            │
│                          ┌──────────┐        │
│                          │  STORE   │        │
│                          │ findings │        │
│                          │ + report │        │
│                          └──────────┘        │
│                                              │
│  BUDGET GUARD: checked before each ACT step  │
│  RETIREMENT: checked between iterations      │
└──────────────────────────────────────────────┘
```

Each iteration:
1. **Plan**: The LLM decides what to search and why (informed by what it already found)
2. **Act**: Execute 1-2 tool calls (search, extraction)
3. **Evaluate**: Did the results advance toward resolving the doubt? Should it pivot?
4. **Decide**: Continue (with refined strategy), store intermediate finding, or terminate

This typically runs 3-6 iterations before the doubt is resolved or the budget is exhausted.

---

## 4. Retirement Rules

The orchestrator controls clone lifecycle through a **Retirement Rules Engine**. Rules are checked at two points:

- **Pre-spawn**: Before creating a clone, check if the doubt even needs investigating
- **Mid-flight**: Between each planning iteration, the clone calls `check_retirement` to see if the orchestrator wants it to stop

### 4.1 Pre-Spawn Retirement (Orchestrator decides NOT to spawn)

These rules prevent wasting resources on clones that shouldn't exist:

| Rule | Condition | Rationale |
|------|-----------|-----------|
| **DOUBT_ALREADY_RESOLVED** | ConditionStore contains `≥2` findings with `confidence ≥ 0.8` matching this doubt's angle + keywords, stored since last wave | Another clone or a previous wave already found the answer |
| **ANGLE_SATURATED** | `store.count(angle=X, source_type='clone_research') > threshold` for this run | This angle has received enough clone attention; marginal returns declining |
| **GLOBAL_BUDGET_EXHAUSTED** | Total clone API calls this run `≥ max_clone_api_calls_per_run` | Hard ceiling on external API spend |
| **WAVE_TOO_EARLY** | `wave < min_clone_wave` (default: 2) | Workers haven't accumulated enough context for clones to carry meaningful expertise |
| **DUPLICATE_DOUBT** | Semantic similarity `> 0.85` with a doubt already being investigated by an active clone | Two workers expressed the same gap — one clone is enough |

### 4.2 Mid-Flight Retirement (Orchestrator signals clone to stop)

These rules are checked via the `check_retirement` tool that each clone calls between planning iterations:

| Rule | Signal | Action |
|------|--------|--------|
| **TIMEOUT** | `elapsed_time > clone_timeout_s` | Kill immediately, store partial findings |
| **DOUBT_RESOLVED_EXTERNALLY** | Another clone stored findings resolving this doubt (checked via ConditionStore query) | Graceful stop — store what you have, mark doubt as resolved |
| **DIMINISHING_RETURNS** | Last `N` search calls returned 0 new findings (default N=3) | Stop searching, synthesize what you have |
| **BUDGET_EXHAUSTED** | Clone's tool calls hit `max_tool_calls` or search calls hit `max_search_api_calls` | Forced stop, synthesize remaining |
| **WAVE_ADVANCING** | The orchestrator has started wave N+1 (swarm moved on) | Graceful stop — store findings, they'll appear in wave N+2's data packages |
| **CONVERGENCE_ACHIEVED** | Global convergence detected — the swarm has enough information | Kill all active clones, store partial findings |

### 4.3 Retirement Signal Protocol

```python
@dataclass
class RetirementSignal:
    """Signal from orchestrator to a running clone."""
    should_retire: bool
    reason: str                    # e.g., "doubt_resolved_externally"
    urgency: str                   # "immediate" | "graceful"
    # immediate = stop now, store what you have
    # graceful  = finish current search, synthesize, then stop
```

The clone queries the retirement checker between iterations:

```python
# Inside clone planning loop
signal = await retirement_checker.check(
    clone_id=self.clone_id,
    doubt=self.doubt,
    angle=self.angle,
    tool_calls_used=self.tool_call_count,
    searches_used=self.search_count,
    findings_stored=self.findings_count,
    elapsed_s=time.monotonic() - self.start_time,
)

if signal.should_retire:
    if signal.urgency == "immediate":
        await self._store_partial_findings()
        return
    else:  # graceful
        await self._synthesize_and_store()
        return
```

### 4.4 Retirement Checker Implementation

```python
class RetirementChecker:
    """Evaluates retirement conditions for active clones.

    The orchestrator creates one RetirementChecker per run and passes
    it to all clones.  The checker reads from the ConditionStore and
    from shared orchestrator state (active clones, wave status, budgets).
    """

    def __init__(
        self,
        store: ConditionStore,
        config: RetirementConfig,
    ) -> None:
        self.store = store
        self.config = config
        self._global_api_calls = 0
        self._active_clones: dict[str, CloneStatus] = {}
        self._wave_advancing = False
        self._convergence_achieved = False

    async def check(self, **clone_state) -> RetirementSignal:
        # 1. Immediate kills
        if self._convergence_achieved:
            return RetirementSignal(True, "convergence_achieved", "immediate")

        if clone_state["elapsed_s"] > self.config.clone_timeout_s:
            return RetirementSignal(True, "timeout", "immediate")

        if self._global_api_calls >= self.config.max_clone_api_calls_per_run:
            return RetirementSignal(True, "global_budget_exhausted", "immediate")

        # 2. Graceful stops
        if self._wave_advancing:
            return RetirementSignal(True, "wave_advancing", "graceful")

        if clone_state["tool_calls_used"] >= self.config.max_tool_calls_per_clone:
            return RetirementSignal(True, "clone_budget_exhausted", "graceful")

        # 3. Doubt resolved externally?
        if await self._doubt_resolved(clone_state["doubt"], clone_state["angle"]):
            return RetirementSignal(True, "doubt_resolved_externally", "graceful")

        # 4. Diminishing returns (tracked by clone itself)
        if clone_state.get("consecutive_empty_searches", 0) >= self.config.max_empty_searches:
            return RetirementSignal(True, "diminishing_returns", "graceful")

        return RetirementSignal(False, "", "")
```

---

## 5. Budget Management

### 5.1 Three-Tier Budget Hierarchy

```
RUN BUDGET (per synthesize() call)
├── max_clone_api_calls_per_run: 100        # Total search API calls across all clones
├── max_concurrent_clones: 4                # Parallel clone agents
├── max_clone_waves: 3                      # Clone research triggers at most every N waves
│
├── WAVE BUDGET (per wave's research organizer invocation)
│   ├── max_clones_per_wave: 4              # Max clones spawned per wave
│   ├── max_doubts_per_worker: 3            # Doubts extracted per transcript
│   │
│   └── CLONE BUDGET (per individual clone agent)
│       ├── max_tool_calls: 15              # Total tool invocations
│       ├── max_search_api_calls: 8         # Subset: search-specific calls
│       ├── max_extraction_calls: 3         # Subset: URL extraction calls
│       ├── timeout_s: 180                  # Wall-clock timeout
│       └── max_planning_iterations: 6      # Plan-act-evaluate loops
```

### 5.2 Budget Accounting

The orchestrator tracks budget consumption across all clones:

```python
@dataclass
class CloneBudgetTracker:
    """Tracks aggregate clone resource consumption."""
    total_api_calls: int = 0
    total_extractions: int = 0
    total_findings_stored: int = 0
    total_clones_spawned: int = 0
    total_clones_retired: int = 0
    retirement_reasons: dict[str, int] = field(default_factory=dict)
    # e.g., {"diminishing_returns": 2, "doubt_resolved_externally": 1}
```

Budget metrics are stored in the ConditionStore as `clone_budget_metric` rows for observability.

---

## 6. Async Execution Model

### 6.1 Non-Blocking Clone Execution

Clones run as `asyncio.Task` objects — they don't block the wave loop. The orchestrator spawns them and moves on:

```python
# In mcp_engine.py synthesize() loop:

# Spawn clones (non-blocking)
clone_tasks = await research_organizer.spawn_clone_agents(
    store=self.store,
    worker_results=successful_results,
    wave=wave,
    run_id=run_id,
    retirement_checker=retirement_checker,
)

# Continue to next wave immediately
# Clones run in background, storing findings as they go
# Wave N+1 workers see clone findings in §8 FRESH EVIDENCE
```

### 6.2 Two Execution Modes

#### Mode A: Local Async (default — no external infrastructure)

Clones run as asyncio tasks in the same process. Uses the existing `research_organizer.py` search functions but wraps them in a planning loop.

```python
async def _run_clone_planning_loop(
    clone: CloneDeepAgent,
    retirement_checker: RetirementChecker,
) -> CloneResearchResult:
    """Local planning loop for a single clone."""
    for iteration in range(clone.max_planning_iterations):
        # Check retirement
        signal = await retirement_checker.check(...)
        if signal.should_retire:
            break

        # Plan: LLM decides next action
        plan = await clone.plan(iteration)

        # Act: Execute tool calls
        results = await clone.execute_tools(plan.tool_calls)

        # Evaluate: Did we make progress?
        evaluation = await clone.evaluate(results)

        # Store any findings immediately
        for finding in evaluation.new_findings:
            clone.store_finding(finding)

        if evaluation.doubt_resolved:
            break
        # Otherwise: loop continues with refined strategy

    return clone.get_result()
```

#### Mode B: Deep Agent (LangGraph Platform — when available)

Clones run as full Deep Agents on LangGraph Platform, using the `AsyncSubAgent` pattern from the Deep Agents SDK. This is the target architecture for H200 deployment where you want clones running on separate GPU instances.

```python
from deepagents import create_deep_agent, AsyncSubAgent

# Define clone as an AsyncSubAgent
clone_subagent = AsyncSubAgent(
    name=f"clone_{angle}_{doubt_hash}",
    description=f"Expert researcher resolving: {doubt.doubt}",
    graph_id="clone_researcher",
    url=langgraph_platform_url,
)

# The orchestrator (itself a Deep Agent) can spawn these via task tool
# and monitor them via check_async_task / cancel_async_task
```

The choice between Mode A and Mode B is a deployment config, not a code change. The planning loop logic is identical — only the execution substrate differs.

---

## 7. Integration Points

### 7.1 mcp_engine.py Changes

```python
# Current (synchronous, blocking):
clone_results = await run_research_organizer(
    store=self.store,
    worker_results=successful_results,
    wave=wave,
    ...
)

# New (async, non-blocking):
active_clone_tasks = await spawn_clone_deep_agents(
    store=self.store,
    worker_results=successful_results,
    wave=wave,
    retirement_checker=retirement_checker,
    budget_tracker=budget_tracker,
    ...
)
# Engine continues to next wave
# Clone findings land in store asynchronously
```

### 7.2 research_organizer.py Changes

The existing module gets a new top-level entry point:

```python
async def spawn_clone_deep_agents(
    store: ConditionStore,
    worker_results: list[dict[str, Any]],
    wave: int,
    run_id: str,
    complete: Callable[[str], Awaitable[str]],
    retirement_checker: RetirementChecker,
    budget_tracker: CloneBudgetTracker,
    config: ResearchOrganizerConfig | None = None,
) -> list[asyncio.Task]:
    """Spawn clone Deep Agents for gap resolution (non-blocking).

    Returns a list of asyncio.Task objects. The caller does NOT
    await these — they run in the background and store findings
    directly to the ConditionStore.
    """
```

The existing `run_clone_research()` function is replaced by `_run_clone_planning_loop()` which implements the plan-act-evaluate cycle.

### 7.3 ConditionStore Schema

New columns/rows for clone lifecycle tracking:

```sql
-- Clone lifecycle events (stored as metric rows)
INSERT INTO conditions (row_type, fact, angle, source_type, ...)
VALUES (
    'clone_lifecycle',
    'clone_insulin_timing_abc12345 retired: diminishing_returns after 12 tool calls, 3 findings',
    'insulin_timing',
    'clone_lifecycle',
    ...
);
```

### 7.4 Data Package §8 (no change needed)

The existing `build_fresh_evidence_section()` in `research_organizer.py` already reads `source_type='clone_research'` findings. The Deep Agent clones use the same `store_finding()` path, so §8 FRESH EVIDENCE works unchanged.

---

## 8. Orchestrator Retirement Decision Flow

This is the complete decision tree the orchestrator evaluates before and during clone execution:

```
WAVE N COMPLETES
│
├─ Extract doubts from all worker transcripts
│
├─ For each doubt:
│   │
│   ├─ DOUBT_ALREADY_RESOLVED?
│   │   (ConditionStore has ≥2 high-confidence findings for this)
│   │   → YES: Skip. Don't spawn clone.
│   │
│   ├─ DUPLICATE_DOUBT?
│   │   (Another active clone is already investigating this)
│   │   → YES: Skip. Let the existing clone handle it.
│   │
│   ├─ ANGLE_SATURATED?
│   │   (This angle already got N clone runs this session)
│   │   → YES: Deprioritize. Only spawn if budget is abundant.
│   │
│   ├─ GLOBAL_BUDGET_EXHAUSTED?
│   │   (Total API calls across all clones this run ≥ ceiling)
│   │   → YES: Stop spawning. Wait for next run.
│   │
│   └─ WAVE_TOO_EARLY?
│       (wave < 2, workers haven't built enough context)
│       → YES: Skip. Clones with shallow context waste resources.
│
├─ Prioritize remaining doubts by expected information gain
│
├─ Spawn top N clones (up to max_clones_per_wave)
│
├─ WHILE clones are running:
│   │
│   │  (Each clone calls check_retirement between iterations)
│   │
│   ├─ TIMEOUT? → Immediate kill
│   ├─ CONVERGENCE_ACHIEVED? → Kill all clones
│   ├─ WAVE_ADVANCING? → Graceful stop (finish current search)
│   ├─ DOUBT_RESOLVED_EXTERNALLY? → Graceful stop
│   ├─ DIMINISHING_RETURNS? → Graceful stop
│   └─ CLONE_BUDGET_EXHAUSTED? → Graceful stop
│
├─ Collect partial/full results from all clones
│
├─ Update budget tracker
│
└─ Log retirement reasons to ConditionStore for observability
```

---

## 9. Configuration

```python
@dataclass
class CloneDeepAgentConfig:
    """Configuration for the clone Deep Agent system."""

    # --- Execution mode ---
    execution_mode: str = "local_async"   # "local_async" | "langgraph_platform"
    langgraph_url: str | None = None      # Required for langgraph_platform mode

    # --- Run budget ---
    max_clone_api_calls_per_run: int = 100
    max_concurrent_clones: int = 4

    # --- Wave budget ---
    max_clones_per_wave: int = 4
    max_doubts_per_worker: int = 3
    trigger_every_n_waves: int = 2
    trigger_uncertainty_threshold: int = 3
    min_clone_wave: int = 2

    # --- Clone budget ---
    max_tool_calls_per_clone: int = 15
    max_search_api_calls_per_clone: int = 8
    max_extraction_calls_per_clone: int = 3
    max_planning_iterations: int = 6
    clone_timeout_s: float = 180.0
    max_empty_searches: int = 3

    # --- Retirement ---
    angle_saturation_threshold: int = 10   # Max clone runs per angle per session
    doubt_similarity_threshold: float = 0.85
    min_confidence_for_resolved: float = 0.8
    min_findings_for_resolved: int = 2

    # --- Model ---
    clone_model: str = ""     # Tier C model; falls back to config.model
    clone_temperature: float = 0.4
```

---

## 10. Observability

Every clone lifecycle event is a row in ConditionStore:

| Event | `row_type` | `source_type` | Key fields |
|-------|-----------|---------------|------------|
| Clone spawned | `metric` | `clone_spawned` | doubt, angle, parent_worker_id |
| Finding stored | `finding` | `clone_research` | fact, confidence, source_ref |
| Retirement signal | `metric` | `clone_retired` | reason, tool_calls_used, findings_stored, elapsed_s |
| Budget snapshot | `metric` | `clone_budget` | total_api_calls, total_clones, total_findings |

The orchestrator emits a summary metric at the end of each research organizer invocation:

```python
store.emit_metric(
    "research_organizer_summary",
    {
        "wave": wave,
        "clones_spawned": budget_tracker.total_clones_spawned,
        "clones_retired": budget_tracker.total_clones_retired,
        "retirement_reasons": budget_tracker.retirement_reasons,
        "total_api_calls": budget_tracker.total_api_calls,
        "total_findings": budget_tracker.total_findings_stored,
    },
    source_run=run_id,
    iteration=wave,
)
```

---

## 11. Relationship to Architecture Doc

This design implements and extends several sections from `SWARM_WAVE_ARCHITECTURE.md`:

| Architecture Doc Section | This Design |
|---|---|
| "Unleashed Clones: Tool-Armed Expert Researchers" (§1525-1667) | Full implementation with Deep Agent shell |
| "When to Unleash" (§1608-1614) | Retirement Rules Engine §4.1 (pre-spawn) |
| "Budget the clone. Max 5 tool calls per unleash." (§1667) | Three-tier budget hierarchy §5.1 |
| "Context Isolation" (§1653-1657) | Invariant preserved — clone is a separate agent, only findings flow back |
| Invariant 13: "Unleashed clones resolve doubts, not explore randomly" | Planning loop enforces doubt-focused research §3.3 |
| Stage 6 trigger: "high hedging_density AND many unanswered researchQuestions" | Pre-spawn retirement rules §4.1 |

### What this adds beyond the architecture doc:

1. **Deep Agent planning shell** — the architecture doc describes clones as tool-armed agents but doesn't specify the planning/iteration mechanism. This design wraps them in a plan-act-evaluate loop.

2. **Mid-flight retirement** — the architecture doc only describes when to unleash. This design adds when and how to retire mid-execution.

3. **Async execution** — the architecture doc implies clones run between waves (blocking). This design makes them non-blocking background tasks.

4. **Dual execution mode** — local asyncio for development/testing, LangGraph Platform for H200 deployment.

5. **Budget accounting** — the architecture doc mentions "max 5 tool calls per unleash." This design provides a three-tier budget hierarchy with global, per-wave, and per-clone limits plus runtime tracking.

---

## 12. Implementation Plan

### Phase 1: Local Planning Loop (Mode A)

1. Refactor `research_organizer.py` — replace `run_clone_research()` with `_run_clone_planning_loop()`
2. Implement `RetirementChecker` with all 6 mid-flight rules
3. Implement `CloneBudgetTracker` with three-tier budgets
4. Add pre-spawn retirement checks to `run_research_organizer()`
5. Make clone spawning non-blocking (`asyncio.create_task`)
6. Add lifecycle event logging to ConditionStore
7. Update `mcp_engine.py` to pass retirement checker and budget tracker

### Phase 2: Deep Agent Shell (Mode B)

1. Define clone as a `SubAgent` spec with tool list and system prompt
2. Wire `create_deep_agent()` for clone instantiation
3. Implement `AsyncSubAgent` wrapper for LangGraph Platform deployment
4. Add `check_retirement` as a native tool in the Deep Agent's toolset
5. Test with LangGraph Platform (requires deployment infrastructure)

### Phase 3: Observability + Tuning

1. Dashboard for clone lifecycle (spawns, retirements, findings per clone)
2. A/B test: flat fanout vs planning loop (same doubts, measure finding quality)
3. Tune budget parameters based on E2E runs
4. Add semantic dedup between clone findings and existing store content
