# Swarm V5 Architecture Proposal: Conversational Store, Library Station, Dynamic Topology

## Status: Proposal — reconciles and supersedes aspects of V4 docs

This document proposes architectural changes that address the systemic
failure identified across 12 E2E tests: **the store is a dumb bucket
that every consumer queries without scoping, producing redundant data
packages, stale contamination, and context waste.**

The proposal introduces five interconnected changes:

1. **RAG-mediated store access** — agents converse with the store, never query it
2. **Library station pattern** — clones delegate tool use to a specialist retriever
3. **Incremental data packages** — wave N delivers only what changed since wave N-1
4. **Dynamic model topology** — model swapping for escalation, Architect role for cross-domain synthesis
5. **Revised 3-tier model mapping** — collapse Tier C into Tier A (clones = bees + librarian card)

### Relationship to Existing Documents

| Document | Status Under This Proposal |
|---|---|
| `SWARM_WAVE_ARCHITECTURE.md` | **Partially superseded** — data package structure (§1-§7) replaced by incremental + RAG delivery. Wave lifecycle unchanged. Flock/session proxy patterns retained. |
| `CLONE_DEEP_AGENT_ARCHITECTURE.md` | **Superseded by §2** — clone no longer uses tools directly. Planning loop retained but acts through library station, not raw search APIs. Retirement rules retained verbatim. |
| `MODEL_ROLE_ASSIGNMENT.md` | **Updated by §5** — 4 tiers → 3 tiers. Role requirements refined. Candidate table unchanged. |
| `SWARM_CONVERSATION_ARCHITECTURE.md` | **Retained and extended** — deliberate misassignment, worldview prompts, persistent store RAG all remain. Per-agent RAG formalizes the "FROM THE HIVE" pattern. |
| `STORE_ARCHITECTURE.md` | **Extended** — audit trail DAG unchanged. Cloned context as Flock backend generalized into per-agent RAG. The 42 columns become RAG facets. |
| `FLOCK_SERENDIPITY_ARCHITECTURE.md` | **Retained** — Flock SQL templates unchanged. Clone-as-Flock-evaluator is a natural extension of the cloned-context-as-Flock-backend pattern already documented. |

---

## 1. RAG-Mediated Store Access

### The Problem

Every consumer (data package builder, research organizer, report
generator, angle detector, fresh evidence, gap analysis) writes ad-hoc
SQL against `conditions` with no contract about what data it should see.
26+ queries across 5 files have zero `source_run`, `wave`, or `phase`
filtering. The result:

- **Stale contamination** — 4,243 findings from prior runs collapse
  angle detection to 1 angle
- **Redundant data packages** — wave 3's data package contains everything
  from waves 1+2 again (bee context is 60-70% redundant)
- **No incremental delivery** — the data package builder re-queries the
  entire store every wave
- **External callers bypass the store API** — 14 of 26 unscoped queries
  reach directly into `store.conn.execute()`

### The Fix: Agents Converse With the Store

Instead of building data packages by querying the store and stuffing
results into prompts, each agent interacts with the store through a
**RAG interface**. The agent asks questions in natural language and gets
relevant findings back. The agent never knows the store exists — it just
has conversations.

```
CURRENT (V4):
  Data package builder → SQL query → dump 500 findings into prompt
  
PROPOSED (V5):
  Bee asks: "What do we know about insulin timing relative to carb intake?"
  → RAG retrieves the 12 most relevant findings, filtered by run/wave/angle
  → Bee reasons over focused, relevant context
```

The store's 42 columns — the gradient flags, expansion system,
clustering, contradiction tracking — are not dead weight to strip.
They are the **filtering vocabulary** that makes RAG scoping powerful.
Each column becomes a facet:

| Column | RAG Facet Use |
|---|---|
| `source_run` | Scope to current run (eliminates stale contamination) |
| `phase` | Scope to relevant pipeline stages |
| `wave` (derived from `iteration`) | Scope to new-since-last-wave (enables incremental delivery) |
| `angle` | Scope to this bee's domain + deliberate off-angle injection |
| `row_type` | Finding vs thought vs synthesis vs raw |
| `consider_for_use` | Active vs retired |
| `confidence` | Quality threshold |
| `verification_status` | Speculative vs verified |
| `source_type` | Researcher vs clone vs corpus_builder |
| `novelty_score` | Prioritize genuinely new information |
| `contradiction_flag` | Surface disputed findings for challenges |
| `expansion_fulfilled` | Skip already-addressed gaps |

A **query profile** defines which facets each consumer uses:

```python
# Data package §3: CORPUS EVIDENCE for wave 3 insulin bee
profile = QueryProfile(
    source_run=current_run,
    wave_range=(2, 3),          # Only new since wave 1 (already digested)
    angle="insulin_timing",
    angle_off_ratio=0.25,       # 25% deliberate misassignment
    row_type=["finding"],
    consider_for_use=True,
    min_confidence=0.3,
    exclude_contradiction=False, # Include contradicted findings for challenge
)

# §8 FRESH EVIDENCE for wave 3
profile = QueryProfile(
    source_run=current_run,
    source_type="clone_research",
    wave_range=(2, 3),          # Only clone findings from last wave
    angle="insulin_timing",
    consider_for_use=True,
)
```

### Per-Agent RAG

Each agent gets its own RAG instance tuned to its role:

- The insulin timing bee's RAG is weighted toward pharmacokinetics,
  timing, dosing — scoped to its angle with 25% off-angle injection
- The hepatotoxicity bee's RAG surfaces liver function, adverse events,
  toxicology
- The clone's RAG inherits the parent bee's profile plus adds the
  librarian's findings
- The serendipity worker's RAG deliberately maximizes cross-angle
  retrieval

As the bee reasons and asks more refined questions across waves, its
RAG profile sharpens. Wave 1: broad retrieval. Wave 3: laser-focused
on the specific gaps the bee has identified. The "chiseling" is
automatic — the query profile narrows as the bee's understanding
deepens.

### Hybrid: Briefing + On-Demand RAG

Pure RAG is demand-driven — it only returns what the agent asks for.
The agent might not ask about things it doesn't know it doesn't know.
The data package pattern forces exposure to unexpected findings.

The hybrid approach:

1. **Compact briefing** (pushed, not pulled) — new contradictions,
   fresh clone findings, changed verification statuses since last wave.
   Small, focused, curated by the orchestrator. This is the
   "incremental data package" (§3 below).
2. **On-demand RAG** (pulled by the agent) — the bee can ask follow-up
   questions during reasoning. "What do other angles say about iron
   metabolism?" → RAG returns cross-angle findings the bee didn't know
   to look for.

The briefing ensures the bee sees critical changes. The RAG ensures
the bee can explore when its reasoning leads somewhere unexpected.

### Implementation Path

The Flock-with-cloned-context pattern from `STORE_ARCHITECTURE.md`
already implements per-agent RAG at a low level: Flock SQL queries
routed through the session proxy produce domain-expert results. This
proposal generalizes that pattern:

- The session proxy handles context prepending (unchanged)
- Query profiles replace ad-hoc SQL with parameterized, scoped queries
- The data package builder uses profiles instead of raw SQL
- External callers (`data_package.py`, `mcp_engine.py`,
  `research_organizer.py`, `worker_tools.py`) must go through the
  profile-based API — no more direct `store.conn.execute()`

The store query contract problem dissolves. Not by fixing WHERE
clauses one by one, but by putting profiles + RAG between agents
and the store. Agents never see SQL.

---

## 2. Library Station Pattern

### The Problem With Tool-Armed Clones

`CLONE_DEEP_AGENT_ARCHITECTURE.md` gives clones 8 search tools
(brave_search, exa_search, tavily_search, etc.) and a plan-act-evaluate
loop. But the bee workers are **tool-free reasoners by design**. When
a bee gets cloned, it inherits the bee's reasoning context but is
suddenly expected to be a skilled tool-caller. This contradicts the
core architecture: the clone carries the bee's expertise about WHAT
needs answering, but it doesn't suddenly become good at HOW to search.

### The Library Station

The clone doesn't use tools directly. It visits a **library station** —
a shared service that handles all retrieval mechanics.

```
Clone (scholar):
  "I need primary sources on rapid-acting insulin analog dose-response
   curves in the 4-8 IU range, specifically pre-workout timing relative
   to carbohydrate ingestion"

Library station (librarian):
  → Translates to MeSH terms for PubMed
  → Queries PubMed, Semantic Scholar, ClinicalTrials.gov in parallel
  → Follows citation chain on the top hit
  → Returns 7 structured findings with source URLs

Clone (evaluates with expertise):
  "This PubMed paper uses subcutaneous injection but the bodybuilding
   context is intramuscular. I need IM-specific data."
  → Makes another reference request to the library station
```

The clone's planning loop from `CLONE_DEEP_AGENT_ARCHITECTURE.md` is
retained but reframed: the clone **plans research questions**, not
search queries. The plan-act-evaluate loop becomes
plan-ask-evaluate-refine.

### The Clone Has One Tool

```python
clone_agent = CloneAgent(
    # ... context from parent worker (unchanged) ...
    
    tools=[
        ask_librarian,      # Natural language research request → structured findings
        store_finding,      # Write directly to ConditionStore
        check_retirement,   # Query orchestrator retirement signal
    ],
    
    # Budget (unchanged from CLONE_DEEP_AGENT_ARCHITECTURE.md)
    max_planning_iterations=budget.max_iterations,  # default: 6
    timeout_s=budget.timeout_s,                     # default: 180
)
```

The clone doesn't need to know PubMed syntax, forum URL patterns,
or API rate limits. It describes what it needs. The librarian
handles the mechanics.

### Clone System Prompt (Revised)

Replaces §3.2 of `CLONE_DEEP_AGENT_ARCHITECTURE.md`:

```
You are an expert researcher in {angle}. You have been reasoning about
this domain and identified a specific gap in your understanding:

DOUBT: {doubt.doubt}
DATA NEEDED: {doubt.data_needed}

Your accumulated knowledge so far:
{knowledge_summary}

Your previous reasoning (excerpt):
{worker_transcript_tail}

YOUR MISSION: Resolve this doubt by asking your research assistant for
information. You have access to a librarian who can search academic
databases, community forums, regulatory filings, and the general web.

HOW TO USE THE LIBRARIAN:
- Describe WHAT you need to know and WHY — the librarian handles WHERE
  and HOW to search
- Be specific: "I need hepatotoxicity data for HGH at 4 IU daily,
  specifically ALT/AST elevation data from human studies" is better
  than "search for HGH liver effects"
- You MUST use the librarian for EVERY factual uncertainty. Do NOT
  reason from memory. Do NOT speculate when you can verify.
- Ask about multiple aspects in parallel if your doubt spans domains
- Evaluate what comes back: Does it actually answer your doubt? If
  partially, refine your question. If the data contradicts your
  expectation, investigate further.

An unresearched doubt is a failure. Your job is to turn doubts into
verified findings backed by primary sources.
```

### The Librarian Agent

The librarian is a **fast, tool-calling specialist** — the only agent
in the entire swarm that needs excellent tool-calling capability.

```
Librarian characteristics:
- Query reformulation: translates natural language → MeSH terms,
  Boolean queries, forum-specific keyword patterns
- Source awareness: knows which database has what
  (PubMed for pharmacokinetics, forums for anecdotal dosing,
   ClinicalTrials.gov for study designs, FDA FAERS for adverse events)
- Citation chain following: when a result references a key study,
  the librarian fetches that study too
- Diminishing returns detection: 3 searches returned redundant
  results → return what you have
- Parallel dispatch: PubMed AND forums simultaneously for
  cross-domain requests
- Error resilience: API down? Skip it, try next source.
  Circuit breaker logic lives here.
- NO topic judgment: retrieves faithfully regardless of content.
  That's the clone's responsibility.
```

The librarian is a shared service — all clones visit the same station.
The station doesn't understand insulin or tren or GH. It knows how
to find things. The intelligence about WHAT to find lives in the clone.
The skill of HOW to find it lives in the station.

### Retirement Rules

Unchanged from `CLONE_DEEP_AGENT_ARCHITECTURE.md` §4. The 5 pre-spawn
and 6 mid-flight retirement rules apply identically. The budget
hierarchy (run → wave → clone) is unchanged.

---

## 3. Incremental Data Packages

### The Problem

The data package builder (`data_package.py`) rebuilds all 7 sections
from scratch every wave. §3 CORPUS EVIDENCE re-queries the entire
store for the angle's findings. Wave 3's package contains everything
from waves 1+2 again.

The bee's conversation grows:

```
[wave 1 package: 80K] + [wave 1 reasoning: 30K]
+ [wave 2 package: 120K] + [wave 2 reasoning: 30K]  ← wave 1 data AGAIN
+ [wave 3 package: 160K] + [wave 3 reasoning: 30K]  ← waves 1+2 AGAIN
= 450K total, 60-70% redundant
```

This is the same root problem: unfiltered store queries. No `wave`
filter, no `created_at > last_wave_timestamp` filter. The data
package doesn't know what the bee already saw.

### The Fix: Delta Packages

Once the store queries use proper profiles with wave scoping (§1),
incremental packages fall out naturally:

```
Wave 1: FULL package (first exposure)
  §1 Knowledge state: EMPTY
  §2 Corpus material: Full assigned section
  §3 From the hive: EMPTY
  §4 Cross-domain: EMPTY
  §5 Challenges: EMPTY
  §6 Research gaps: Implicit
  §7 Previous output: EMPTY

Wave 2: DELTA package
  §1 Knowledge state: Rolling summary of wave 1 (compact)
  §2 Corpus material: NEW findings since wave 1 only (delta query)
  §3 From the hive: Cross-angle findings from wave 1 (RAG-scored)
  §4 Cross-domain: Validated connections from wave 1 clones
  §5 Challenges: Contradictions identified in wave 1
  §6 Research gaps: Gaps from clone analysis
  §7 Previous output: Bee's own wave 1 output
  §8 Fresh evidence: Clone findings from wave 1 research

Wave 3: DELTA package
  §2 Corpus material: NEW findings since wave 2 only
  §3 From the hive: NEW cross-angle findings from wave 2
  §4 Cross-domain: NEW validated connections
  §5 Challenges: NEW contradictions
  §8 Fresh evidence: Clone findings from wave 2 research
```

The bee's context drops from 450K to ~180K. Same information,
65% less context. More room for the bee's own reasoning. And
the bee isn't re-processing evidence it already synthesized.

### Context Management in the Bee

The bee's context management is a **prerequisite** for the clone
pattern working at scale. If the bee manages context well, the
clone inherits a well-organized mind. If it doesn't, the clone
inherits a mess.

Effective context management:

1. **Rolling compaction** — after processing wave 2's data, wave 1's
   raw evidence is gone from context. The bee's own conclusions
   survive. The data is compacted, not the reasoning.
2. **Working memory** — the bee maintains a structured epistemic state:
   what it believes, what it's uncertain about, what contradicts.
   This is what hooks can capture — not just the transcript, but
   the bee's uncertainty map.
3. **Relevance pruning** — when context approaches limits, the oldest
   raw evidence is pruned first. The bee's reasoning chains are
   preserved longest (highest value per token).
4. **Explicit doubt tagging** — when the bee encounters something it
   can't resolve, it marks it: `"UNRESOLVED: I don't know if the
   15-min or 30-min insulin timing window is correct for IM injection."`
   This makes the Research Organizer's job trivial (grep for
   UNRESOLVED) and gives the clone a clear mission.

The bee's context hygiene determines the clone's effectiveness.
A well-managed bee → clone arrives with clear research agenda.
A poorly-managed bee → clone can't articulate what it doesn't know.

This means context management isn't just performance optimization —
it's a **functional requirement** for the clone pattern. The bee
must be designed with the expectation that its context will be
inherited.

---

## 4. Dynamic Model Topology

### Current State: Flat Topology

All bees use the same model. All clones use the same model. No
escalation, no cross-domain synthesis role, no dynamic adaptation.

### Proposed: Three Topology Innovations

#### 4.1 Model Swapping for Escalation

A bee normally runs on Tier A (fast, 1M context — e.g.,
Kimi-Linear-48B-A3B). But when the Research Organizer detects a
critical situation — two high-confidence findings that directly
contradict each other, a clone that returned unexpected complexity —
it can escalate:

```
Research Organizer decision:
  "Worker Alpha has an unresolved contradiction between insulin
   sensitivity enhancement (conf 0.87) and insulin resistance
   induction (conf 0.82). Escalate to deep reasoning model."

→ Worker Alpha switches to Tier B (Ring-2.5-1T) for wave 4
→ After wave 4, Alpha drops back to Tier A
```

The expensive model is used surgically for the hard problems.
Not burned on routine synthesis.

Escalation triggers:
- Unresolved high-confidence contradictions
- Clone research revealed unexpected domain complexity
- The bee's novelty rate dropped below threshold (stuck, needs
  deeper reasoning to break through)
- Cross-domain findings that require reasoning across multiple
  specialist domains simultaneously

#### 4.2 The Architect Role

A new agent role that doesn't exist in V4: the **Architect**.

```
Ring-2.5-1T "Architect" (256K, deepest reasoning)
  ├── talks to Bee Alpha (1M context, insulin angle)
  ├── talks to Bee Beta (1M context, hepatotoxicity angle)
  ├── talks to Bee Gamma (1M context, HGH mechanisms angle)
  └── talks to Bee Delta (1M context, tren interactions angle)
```

The Architect doesn't have 1M context — it can't hold everything.
But it has the deepest reasoning capability. It asks the bees
targeted questions:

```
Architect → Bee Alpha:
  "Given your insulin expertise, is there a mechanism by which
   tren-induced insulin resistance could be overcome with timing
   manipulation?"

Bee Alpha answers from its deep context.
The Architect synthesizes across all bees' answers.
```

This is the "big reasoner / deep context" split: reasoning depth
and context depth live in different agents. The Architect reasons
across domains but relies on specialists for domain depth. The
specialists have massive context but defer to the Architect for
cross-domain synthesis.

The Architect replaces the current report generator's role as
cross-domain synthesizer. The report generator becomes a **writer**
that takes the Architect's synthesis and produces prose. Two
different skills, two different models.

#### 4.3 Clone as Flock Query Evaluator

From `STORE_ARCHITECTURE.md`, cloned worker contexts already drive
Flock SQL queries. This proposal extends that pattern: the clone
doesn't just provide context for Flock — it **directs** Flock
operations using its accumulated expertise.

Flock's `llm_filter` uses a generic LLM to judge relevance. With
the clone's context prepended, that judgment becomes expert:

```sql
-- Generic Flock: "relevant to HGH hepatotoxicity" → keyword matching
-- Clone-scored Flock: the clone KNOWS that IGF-1 liver metabolism
-- is relevant because HGH's hepatotoxicity is mediated through IGF-1
SELECT * FROM conditions
WHERE flock_filter(fact, 'relevant to HGH hepatotoxicity at 4 IU') = TRUE
-- Session proxy prepends clone's full conversation → expert judgment
```

The clone becomes the **relevance judge** for store operations.
The store IS the library, Flock IS the librarian's hands, and the
clone IS the scholar directing retrieval.

This partially collapses the library station into the store itself
for **within-store** operations. The external library station (§2)
handles **outside-store** retrieval (web search, PubMed, forums).
The clone directs both.

---

## 5. Revised Model-to-Role Mapping

### The Collapse: 4 Tiers → 3 Tiers

The V4 doc (`MODEL_ROLE_ASSIGNMENT.md`) defines:

- Tier A — Deep Context Reasoning (Workers, Serendipity)
- Tier B — Deep Analytical Synthesis (Organizer, Angles, Report)
- Tier C — Expert Tool Use (Clones)
- Tier D — Fast Structured Classification (Extractors, Compactors)

**Tier C collapses into Tier A.** A clone carries the bee's FULL
conversation history — if the bee ran 3 waves with 200K data
packages each, the clone context can be 500K-1M. The clone needs
the same context capacity as the bee. And under the library station
pattern, the clone has one simple tool (`ask_librarian`), not 8
search APIs. It doesn't need a tool-calling specialist model.

The clone is a bee with a librarian card. Same model, same context,
different prompt and one tool attachment.

### Revised 3-Tier Mapping

| Tier | Roles | Context | Key Requirement | Candidate |
|---|---|---|---|---|
| **A — Deep Reasoners** | Bee Workers, Clone Scholars, Serendipity Worker | **1M** | Maximum context + abliterated + deep reasoning | Kimi-Linear-48B-A3B abl. |
| **B — Strategic Thinkers** | Research Organizer, Angle Detector, Report Generator, Architect | **256K** | Deep reasoning + structured output + abliterated | Ring-2.5-1T abl. (when available) |
| **D — Fast Specialists** | **Librarian**, Finding Extractor, Compactor, Cataloguer, Summarizer | **8-64K** | Speed + tool-calling (librarian) or structured output (others) | Ling-2.6-Flash |

### Full Role Requirements (Revised)

| # | Role | Context | Reasoning | Uncensored | Tool-Calling | Speed | Structured Output |
|---|---|---|---|---|---|---|---|
| 1 | Query Comprehension | 8K | Moderate | Required | None | Low | High |
| 2 | Corpus Search Strategy | 8K | Moderate | Required | None | Low | High |
| 3 | Angle Detector | 128K | High | Required | None | Low | Critical |
| 4 | Bee Workers | **1M** | Maximum | **Mandatory** | None | Low | Not needed |
| 5 | Finding Extractor | 8-16K | Low-moderate | Moderate | None | **Critical** | Critical |
| 6 | Research Organizer | 256-512K | High | Mandatory | None | Medium | High |
| 7 | Clone Scholars | **1M** | High | **Mandatory** | Minimal (1 tool) | Medium | Moderate |
| 8 | Rolling Summarizer | 32-64K | Moderate | Required | None | High | Low |
| 9 | Compactor | 64-128K | Moderate | Low | None | High | Critical |
| 10 | Serendipity Worker | **1M** | Maximum | **Mandatory** | None | Low | Moderate |
| 11 | Report Generator | 256K+ | Maximum | **Mandatory** | None | Low | Moderate |
| 12 | Cataloguer | 4-8K | Low | Low | None | **Critical** | Critical |
| **13** | **Librarian** (NEW) | **32-64K** | Moderate | Required | **Excellent** | **High** | Critical |
| **14** | **Architect** (NEW) | **256K** | **Maximum** | **Mandatory** | None | Low | Moderate |

The Librarian (role 13) is the only agent that needs excellent
tool-calling. Every other role either doesn't use tools or uses
one simple tool. This makes the Librarian the odd one out —
fast like Tier D, but tool-specialized in a way that might justify
a different model if Ling-2.6-Flash's tool-calling isn't strong enough.

The Architect (role 14) is the deepest reasoner in the system —
the model that synthesizes across all specialist domains. It sees
256K of strategic context (not raw findings) and produces the
cross-domain synthesis that the Report Generator then writes up.

---

## 6. Reconciliation: What Changes, What Stays

### Unchanged

- **Tool-free bee workers** — core design principle retained
- **Deliberate misassignment** (20-30% off-angle data) — retained
  from `SWARM_CONVERSATION_ARCHITECTURE.md`
- **Worldview prompts** — bee identity as interpretive lens retained
- **Audit trail DAG** — every row has `parent_id` lineage, unchanged
- **Flock SQL templates** — the 13-step battery from
  `FLOCK_SERENDIPITY_ARCHITECTURE.md` unchanged
- **Session proxy + vLLM prefix caching** — unchanged
- **Retirement rules** — 11 rules from
  `CLONE_DEEP_AGENT_ARCHITECTURE.md` retained verbatim
- **Clone budget hierarchy** (run → wave → clone) — unchanged

### Changed

| V4 | V5 | Why |
|---|---|---|
| Data packages rebuilt from scratch every wave | Incremental delta packages | 60-70% context reduction; eliminates redundant re-processing |
| Clones have 8 search tools | Clones have 1 tool (ask_librarian) | Clones are reasoners, not tool-callers; separation of concerns |
| 4 model tiers (A/B/C/D) | 3 model tiers (A/B/D) | Clones need same context as bees; Tier C collapses into Tier A |
| Ad-hoc SQL queries throughout pipeline | Profile-based RAG through store API | Eliminates unscoped queries; store columns become facets |
| Flat topology (all bees equal) | Architect + escalation | Big reasoner for cross-domain synthesis; surgical deep-reasoning for hard problems |
| No explicit librarian role | Library station as shared service | Tool mastery concentrated in one specialist; all clones benefit |
| 12 LLM roles | 14 LLM roles (+Librarian, +Architect) | New roles from topology changes |
| Report generator does cross-domain synthesis AND writing | Architect synthesizes, Report Generator writes | Different skills, different models |

### New Concepts

| Concept | Definition |
|---|---|
| **Query Profile** | Parameterized scoping specification that defines which store rows a consumer is allowed to see. Replaces ad-hoc SQL. |
| **Library Station** | Shared tool-calling agent that handles all external retrieval (PubMed, forums, web search). Clones describe WHAT they need; the station handles HOW. |
| **Architect** | Deepest-reasoning agent that doesn't hold full context but synthesizes across all specialist bees via targeted questions. |
| **Delta Package** | Incremental data package delivering only what changed since the bee's last wave. Requires wave-scoped store queries. |
| **Model Escalation** | Dynamic model swapping where specific bees get a more powerful model for one wave when the Research Organizer detects critical reasoning challenges. |
| **Context Hygiene** | The bee's responsibility to maintain a well-organized epistemic state (beliefs, uncertainties, explicit doubt tags) so its clone inherits a clear research agenda. |

---

## 7. Implementation Phases

### Phase 1: Store Query Contracts (P0 from Problem Manifest)

**Prerequisite for everything else.**

1. Add `current_run` parameter to `ConditionStore.__init__()`
2. Build `QueryProfile` class with all facet parameters
3. Implement profile-based query methods on `ConditionStore`
4. Migrate all 26+ unscoped queries (including 14 external callers)
   to use profiles
5. Add the 4 indices from the Problem Manifest (these accelerate
   the profile-based queries)

**Verification:** Run E2E with reused store file. Confirm wave 3
sees only current-run findings, not 4,243 stale entries.

### Phase 2: Incremental Data Packages

1. Add wave tracking to `QueryProfile` (wave high-water mark per
   bee per angle)
2. Modify `data_package.py` to use delta queries for §2 and §3
3. Add §1 Knowledge State as rolling compacted summary (not raw
   evidence rehash)
4. Add §8 Fresh Evidence with proper wave scoping

**Verification:** Measure bee context size across waves. Should
plateau (not grow linearly).

### Phase 3: Library Station

1. Build librarian agent with source router
   (medical, community, regulatory, general_web)
2. Implement `ask_librarian` tool for clones
3. Revise clone system prompt (§2 of this doc)
4. Wire PubMed, forum search, YouTube transcripts, ClinicalTrials.gov
   into librarian's toolset
5. Add circuit breaker logic inside librarian (not per-clone)

**Verification:** Clone spawned for "HGH hepatotoxicity" doubt
returns PubMed-sourced findings, not just general web results.

### Phase 4: Dynamic Topology

1. Implement model escalation in Research Organizer
   (detect critical contradictions → flag bee for escalation)
2. Build Architect agent (cross-domain synthesizer)
3. Implement Architect ↔ bee dialogue protocol
4. Refactor report generation: Architect synthesizes →
   Report Generator writes

**Verification:** Architect produces cross-domain synthesis that
references specific findings from multiple bees' domains.

### Phase 5: Per-Agent RAG

1. Implement on-demand RAG (bee can ask questions during reasoning,
   beyond the pushed briefing)
2. Per-agent RAG profile sharpening (profiles narrow as bee's
   understanding deepens across waves)
3. Hybrid briefing + RAG delivery model

**Verification:** Bee in wave 3 asks a cross-domain question →
gets targeted findings it didn't know to look for.

---

## 8. Risk Assessment

| Risk | Mitigation |
|---|---|
| RAG retrieval misses critical findings the full dump would catch | Hybrid briefing + RAG. The briefing pushes critical changes; RAG handles exploration. |
| Library station becomes a bottleneck (all clones waiting) | Multiple librarian instances. Stateless design — no session affinity needed. |
| Model escalation adds complexity to the orchestrator | Start with manual escalation (Research Organizer decides). Automate triggers later. |
| Architect role requires a model that isn't yet abliterated (Ring-2.5-1T) | Fallback to Qwen2.5-72B abliterated. The role exists independent of the specific model. |
| Incremental packages lose context the bee needs | §7 PREVIOUS OUTPUT carries the bee's own prior reasoning. §1 KNOWLEDGE STATE carries the rolling summary. Raw evidence from prior waves is accessible via on-demand RAG if the bee needs it. |
| Per-agent RAG profiles may over-narrow (filter bubble) | Deliberate misassignment (25% off-angle) is enforced at the profile level, not optional. |
