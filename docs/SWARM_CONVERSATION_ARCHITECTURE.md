# Swarm Conversation Architecture

## The Problem

The swarm has access to a large corpus of findings from diverse sources —
PubMed abstracts, bodybuilding forum posts, YouTube transcripts, textbook
excerpts. Hidden in this corpus are two things:

1. **Needles** — individually important facts (Milos Sarcev's exact insulin
   timing, a specific iron metabolism paper, a practitioner's bloodwork).
2. **Threads** — connections between facts that are buried in completely
   unrelated-looking locations. The iron data is in a hepatic hepcidin
   paper. The hematocrit data is in a bodybuilding forum post. Nobody
   looking at either location would think to connect them.

The threads are the hard part. A single reader with the full corpus finds
needles but misses threads because it can't hold 50k chars in working
memory and cross-reference everything against everything.

That's why we have multiple bees. But the current architecture fails at
thread-discovery because of how it distributes data.

## Why the Current Architecture Misses Threads

Today, findings are distributed by angle relevance: the molecular bee gets
molecular data, the practitioner bee gets practitioner data. Each bee
deeply processes its own pile. During gossip rounds they exchange
*summaries*.

The practitioner bee summarizes "hematocrit goes up on tren" — obvious,
triggers no connection. The molecular bee summarizes "hepcidin regulation
affects iron sequestration" — sounds irrelevant to the practitioner bee.
The thread stays hidden because each bee only ever deeply processes data
from its own *obvious* location.

Summaries kill threads. When the practitioner compresses "my hematocrit
went from 42 to 49.6 after 8 weeks on 400mg tren-e, doc said donate
blood" into "hematocrit goes up on tren," the specific numbers that would
trigger the molecular bee's recognition are lost. 49.6 maps to hemoglobin
~16.5 maps to erythropoietin response maps to iron mobilization. The
summary stripped the thread-end.

## Core Mechanism: Deliberate Misassignment

The primary mechanism for thread-discovery is **putting the right wrong
data in front of the right specialist**.

Slice assignment should NOT be pure angle-matching. Instead:

- **70–80% on-angle**: the molecular bee gets mostly molecular data, so it
  has deep domain knowledge to resonate against.
- **20–30% off-angle**: deliberately inject raw findings from distant
  angles. This is the hay from the wrong barn.

The off-angle portion is where threads get discovered. The on-angle
portion is what gives the bee enough domain depth to *recognize* the
thread when it encounters it.

### How Thread-Discovery Works

A thread is found at the moment a bee encounters a fact that **resonates
with something it already knows from its own domain**. The molecular bee
reads "my hematocrit went from 42 to 49.6 after 8 weeks on 400mg tren-e"
and its molecular worldview activates:

> 49.6 → hemoglobin ~16.5 → erythropoietin upregulation → but tren
> doesn't directly stimulate EPO → so what's driving it? → iron
> mobilization from hepatic stores → hepcidin suppression → I have a
> paper on hepcidin regulation right here in my slice.

Thread found. Because the molecular bee saw raw data from a non-obvious
location while holding domain-specific knowledge that made the connection
visible.

### Angle Distance for Pairing

Off-angle findings should come from **maximally distant** angles. The
molecular bee benefits most from practitioner data (real-world
observations it can explain mechanistically). It benefits least from the
biochemistry bee's data (too similar, low surprise potential).

Distance heuristic: angles that share fewer corpus findings are more
distant. In the LLM distribution call, track which findings could
plausibly belong to multiple angles — those are the candidates for
cross-pollination. Findings that only one angle would claim are the
most valuable to give to a *different* angle.

## Bee Identity: Core Interest as Worldview

Each bee's system prompt doesn't just say "you cover molecular topics."
It defines a **worldview** — a lens through which EVERYTHING is
interpreted:

```
You are a molecular biologist. Everything you encounter, you interpret
through cellular mechanisms, receptor binding, gene expression, protein
folding, metabolic pathways. When you read a practitioner's observation
about bloodwork, you don't see "hematocrit went up." You see a signal
about erythropoiesis, iron homeostasis, oxygen-carrying capacity at the
cellular level. Your job is not to summarize what you read — it is to
EXPLAIN what you read through your domain.
```

This worldview is what converts raw foreign data into thread-discovery.
Without it, the molecular bee reads practitioner data and just passes it
through. With it, the molecular bee *reinterprets* practitioner data and
produces connections the practitioner couldn't see.

## Persistent Store: Bee Memory Outside Context

### The Problem with Ephemeral Conversation

Today, everything a bee produces is ephemeral. Worker Phase 1 outputs,
gossip round outputs, serendipity insights — they exist only in the
SwarmResult object returned at the end. Between rounds, a bee's previous
output is compressed into `max_summary_chars` and the reasoning chain
that produced it is lost.

Worse: when bee B finds the iron-hematocrit connection in round 2, bee C
(safety specialist) can only learn about it if bee B's round 2 summary
mentions it prominently enough to survive compression. If the connection
was one of many findings, it may be trimmed. The thread dies.

### Solution: All Swarm Doings Documented

Every phase of every bee's work is persisted to the ConditionStore via
the existing `LineageStore` protocol:

| Phase | What's stored | row_type |
|-------|--------------|----------|
| Phase 1: Worker synthesis | Each bee's initial analysis of its slice | `thought` |
| Gossip round N | Each bee's round output (connections, predictions, evidence chains) | `thought` |
| Serendipity bridge | Cross-angle convergences and contradictions | `thought` |
| Queen merge | Final stitched document | `synthesis` |

The ConditionStore already implements `emit(LineageEntry)` with all the
right column mappings. It just needs to be wired in (pass
`lineage_store=store` to SwarmConfig).

### Swarm-Internal RAG

Bees can query the persistent store between rounds without stuffing
everything into their context window. This is the mechanism that lets
threads compound across rounds:

1. Bee B finds iron-hematocrit connection in round 2 → written to store
2. In round 3, bee C (safety specialist) receives a round prompt that
   includes: "Query the store for findings related to your current
   analysis."
3. Bee C's RAG query: concepts from its own analysis → store returns
   bee B's iron-hematocrit finding
4. Bee C connects: iron overload + hepatotoxicity = compounding risk

The RAG keeps context windows clean — a bee gets 3–5 relevant findings
from the store, not 30k chars of everything everyone ever wrote.

**Implementation**: Between gossip rounds, the engine:
1. Extracts key concepts from each bee's current summary
2. Queries the store for entries matching those concepts (semantic or
   keyword search against the `fact` column)
3. Injects the top-K most relevant entries (from OTHER bees) as a
   "FROM THE HIVE" section in the gossip prompt
4. The bee sees targeted, relevant findings from the persistent store
   without its context being overwhelmed

With local Gemma 4 on H200, the RAG queries cost nothing — zero marginal
token cost, just wall-clock time for the store query.

## Conversation Protocol

### Round Structure

```
Round 1: DEEP ANALYSIS
  Each bee analyzes its slice (70-80% on-angle + 20-30% off-angle)
  through its core-interest worldview. Mandatory output:
  - Findings with implications
  - Cross-domain implications (what does this mean for other domains?)
  - Thread-ends: "I noticed X which might connect to Y but I need
    data about Z to confirm"
  → All output persisted to store

Round 2+: CROSS-POLLINATION + RAG
  Each bee receives:
  - Its own previous summary
  - Its original raw slice
  - Peer summaries from selected diverse peers (DAR selection)
  - "FROM THE HIVE": RAG results from the store — findings from
    other bees that are conceptually relevant to this bee's current
    analysis
  
  The bee:
  1. Re-reads its raw data with new context from peers + hive
  2. Traces connections DEEP (evidence chain + prediction + falsification)
  3. Emits targeted questions: "I need to know about [concept]"
  → All output persisted to store
  → Targeted questions become RAG queries for next round

Convergence:
  - Thread discovery rate drops (store write rate for new connections)
  - Workers agree (Jaccard similarity above threshold)
  - AND no new external data arrived (corpus delta is empty)
  - Hard ceiling: max_gossip_rounds
```

### Context Swap (Future Enhancement)

In select rounds, two maximally-distant bees swap their **raw findings**
(not summaries, not reasoning). Each interprets the other's raw data
through its own worldview. This is the most expensive operation but also
the highest-probability thread-discovery mechanism.

Selection criteria for swap pairs:
- Maximum angle distance (molecular ↔ practitioner, not molecular ↔
  biochemistry)
- At least one bee has emitted "thread-end" questions that the other's
  data might answer
- Not every round — perhaps round 2 or round 3, after bees have
  established their own deep domain knowledge

With local Gemma 4 on H200, the cost is wall-clock time only. Even
swapping all pairs (10 for 5 bees) adds ~2-3 minutes.

## Data Flow

```
                 ConditionStore (DuckDB)
                 ┌─────────────────────────────────────┐
                 │  findings (from research producers)  │
                 │  + bee outputs (from swarm phases)   │
                 │  + thread discoveries                │
                 │  + research gaps                     │
                 └──────┬──────────────┬───────────────┘
                        │              │
              ┌─────────┘              └──────────┐
              │ initial slice                     │ RAG queries
              │ (70-80% on-angle                  │ between rounds
              │  20-30% off-angle)                │
              ▼                                   ▼
    ┌──────────────┐  peer summaries  ┌──────────────┐
    │   Bee A      │◄────────────────►│   Bee B      │
    │   Molecular  │  (conversation)  │  Practitioner│
    │   worldview  │                  │   worldview  │
    └──────┬───────┘                  └──────┬───────┘
           │                                 │
           │ writes findings,                │ writes findings,
           │ connections,                    │ connections,
           │ thread-ends                     │ thread-ends
           │                                 │
           └──────────┬──────────────────────┘
                      ▼
              ConditionStore
              (all doings documented)
                      │
                      ▼
            ┌──────────────────┐
            │ Serendipity      │  reads ALL bee outputs
            │ Bridge           │  from store
            │ (2-pass)         │
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │ Queen            │  reads bee outputs +
            │ (editor)         │  serendipity from store
            └──────────────────┘
```

## Implementation Sequence

### Phase 1: Wire Up Persistence (immediate)
- Pass `lineage_store=ConditionStore` in `swarm_bridge.py`
- All bee outputs now persisted — zero code changes in engine

### Phase 2: Deliberate Misassignment (next)
- Modify `distribute_findings_to_angles()` in `angles.py`
- After LLM assigns findings to angles, take 20-30% of each angle's
  findings and redistribute to the most distant angle
- Distance = inverse of co-assignment frequency

### Phase 3: Swarm-Internal RAG (next)
- Between gossip rounds in `engine.py`, query the store for entries
  relevant to each bee's current concepts
- Inject as "FROM THE HIVE" section in gossip prompt
- Requires: concept extraction from bee summary + keyword/semantic
  search against store

### Phase 4: Core Interest Worldview Prompts (next)
- Rework `_build_synth_prompt` and `_build_gossip_prompt` in `worker.py`
- System prompt establishes worldview identity, not just topic coverage
- Off-angle data in the slice triggers worldview-based reinterpretation

### Phase 5: Context Swap Rounds (future)
- Select 2-3 maximally-distant pairs per designated swap round
- Swap raw findings (not summaries)
- Each bee produces a "foreign data through my lens" analysis
- Persisted to store for other bees to discover via RAG

## Key Constraints

- **No worker hosts the whole corpus.** Information asymmetry is the
  operating principle. Off-angle injection is 20-30%, not 100%.
- **Raw data, not summaries, for cross-pollination.** Summaries kill
  threads by compressing away the specific details that trigger
  connections.
- **Local Gemma 4 on H200.** Token cost is zero. The only cost is
  wall-clock time. This makes RAG queries, extra rounds, and context
  swaps cheap.
- **Flat architecture.** No hierarchy. All bees are peers. The queen
  is an editor, not a supervisor.
- **Persistent store is both audit trail AND working memory.** Bees
  read from it (RAG) and write to it (lineage). The store is the
  hive's shared brain.
