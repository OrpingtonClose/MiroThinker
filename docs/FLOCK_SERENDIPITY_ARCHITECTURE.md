# MiroThinker Flock Architecture: A Serendipity-Generating Machine

*Enshrined April 2026*

---

## Central Thesis

MiroThinker is not a search engine that occasionally stumbles on surprises.
**It is a serendipity-generating machine** that uses search as fuel.

Every architectural layer exists to manufacture unexpected connections:

- **Multi-angle decomposition** splits the query so specialists *cannot*
  see the whole picture — forcing cross-angle surprises to emerge.
- **The thought swarm** runs parallel specialists who produce *competing*
  interpretations, guaranteeing intellectual friction.
- **The Flock-powered maestro** has unrestricted SQL access to the corpus
  and a library of serendipity templates it can apply, compose, and invent.
- **The corpus itself** is a single DuckDB table with gradient-flag columns
  where nothing is ever deleted — only scored, boosted, or suppressed.

The formula: **Serendipity = Relevance x Unexpectedness**.
Every component maximises one or both sides.

---

## Why Flock Makes This Possible

Flock is DuckDB's community extension for LLM-in-SQL.  It lets you write:

```sql
SELECT llm_complete('Find contradictions in these findings', fact)
FROM conditions WHERE angle = 'pricing' AND consider_for_use = TRUE;
```

This is the key insight: **algorithmic and agentic ideas become SQL templates**.

Traditional agents hard-code their intelligence in Python classes with
rigid interfaces.  The Flock architecture makes intelligence *composable*:

| Traditional Agent               | Flock Maestro                         |
|---------------------------------|---------------------------------------|
| Hard-coded scoring pipeline     | SQL templates the maestro can adapt    |
| Fixed tool sequence             | Free-form SQL with LLM functions       |
| New feature = new Python class  | New feature = new SQL template         |
| Requires deployment             | Maestro can invent at runtime          |
| Developer writes the logic      | Maestro discovers the logic            |

The maestro sees the corpus state, decides what operations will improve it,
and executes them.  The templates are *starting points* — the maestro is
free to deviate, combine, or invent entirely new operations based on what
it observes.

---

## The 13-Step Template Battery

### Core Templates (1-8): Organise the Corpus

These are the mechanical foundation — scoring, clustering, dedup, quality
gates.  They turn raw search results into a structured knowledge base.

| Step | Name                    | Type     | What It Does                                        |
|------|-------------------------|----------|-----------------------------------------------------|
| 1    | Assess                  | Pure SQL | Count by status, find unscored, check expansions    |
| 2    | Score                   | Flock    | LLM-score trust, novelty, specificity, relevance    |
| 3    | Composite Quality       | Pure SQL | Weighted formula across all score dimensions         |
| 4    | Detect Contradictions   | Flock    | Find pairs of findings that contradict each other    |
| 5    | Cluster                 | Flock    | Group related findings, mark representatives         |
| 6    | Compress Redundancy     | Flock    | Merge near-duplicates, keep higher-quality version   |
| 7    | Flag Weak               | Pure SQL | Mark low-quality findings for expansion              |
| 8    | Mark Ready              | Pure SQL | Graduate findings to 'ready' status                  |

### Serendipity Templates (9-13): Generate Surprises

These are the backbone.  They exist to break confirmation bubbles, surface
minority viewpoints, and force the corpus toward unexpected connections.

| Step | Name                    | Type          | What It Does                                         |
|------|-------------------------|---------------|------------------------------------------------------|
| 9    | Contrarian Challenge    | Flock (per angle) | Devil's advocate critique of top findings per angle |
| 10   | Cross-Angle Bridge      | Flock         | Polymath connector finds unexpected inter-angle links |
| 11   | Surprise Scoring        | Pure SQL      | Boost high-novelty outlier findings                   |
| 12   | Consensus Detector      | SQL + Flock   | Detect one-sided consensus, spawn targeted dissent    |
| 13   | Angle Diversity Boost   | Pure SQL      | Reward findings from underrepresented angles          |

---

## Serendipity Templates in Detail

### 9. Contrarian Challenge (per angle, Flock LLM)

**Purpose**: For each well-populated angle, generate a devil's advocate
critique of the top findings and materialise it as a thought row.

**When to apply**: Iteration 2+, for any angle with 5+ scored findings.

**How it works**: The maestro runs one `llm_complete` call per angle,
prompting "You are a rigorous sceptic" with the top 10 findings.  The
output becomes a new `row_type='thought'` with
`strategy='serendipitous_contrarian'`.  The thought swarm integrates it
as a peer contribution — the contrarian becomes part of the next
arbitration round.

**Why it generates serendipity**: Specialists confirm their angle's
narrative.  The contrarian *breaks* it.  The friction between specialist
confidence and sceptical challenge produces unexpected research directions.

```sql
INSERT INTO conditions (id, fact, row_type, parent_id, angle,
  consider_for_use, created_at, strategy)
VALUES (
  nextval('seq'),
  (SELECT llm_complete(
    {'model_name': 'corpus_model'},
    'You are a rigorous sceptic. Challenge these findings: '
    || (SELECT STRING_AGG(SUBSTR(fact, 1, 300), '; ')
        FROM (SELECT fact FROM conditions
              WHERE angle = 'biochemistry' AND row_type = 'finding'
              AND consider_for_use = TRUE
              ORDER BY composite_quality DESC LIMIT 10))
  )),
  'thought', NULL, 'biochemistry', TRUE, CURRENT_TIMESTAMP,
  'serendipitous_contrarian'
);
```

### 10. Cross-Angle Bridge (Flock LLM)

**Purpose**: When 2+ angles have substantial findings, discover unexpected
connections *between* angles that single-angle analysis misses.

**When to apply**: When 2+ angles each have 5+ findings.

**How it works**: Aggregates the top findings from all angles into a
single prompt asking a "polymath connector" to find convergences, hidden
contradictions, and transfer opportunities.  Discoveries become
`row_type='insight'` with `strategy='serendipitous_connection'`.

**Why it generates serendipity**: Specialists are domain-bound by design.
A biochemist specialist and an economic specialist will never talk to each
other.  The cross-angle bridge is the *only* mechanism that looks across
all angles simultaneously — it sees what no specialist can.

```sql
INSERT INTO conditions (id, fact, row_type, angle, consider_for_use,
  created_at, strategy)
VALUES (
  nextval('seq'),
  (SELECT llm_complete(
    {'model_name': 'corpus_model'},
    'You are a polymath connector. Findings from different angles:'
    || CHR(10)
    || (SELECT STRING_AGG(
         'ANGLE ' || angle || ': ' || SUBSTR(fact, 1, 250), CHR(10))
       FROM (SELECT angle, fact FROM conditions
             WHERE row_type = 'finding' AND consider_for_use = TRUE
             AND composite_quality > 0.4
             ORDER BY angle, composite_quality DESC LIMIT 30))
    || CHR(10) || 'Find UNEXPECTED connections between these angles.'
  )),
  'insight', 'serendipitous_cross_angle', TRUE, CURRENT_TIMESTAMP,
  'serendipitous_connection'
);
```

### 11. Surprise Scoring (Pure SQL)

**Purpose**: Boost findings that are high-quality but *unexpected* relative
to their angle's consensus.

**When to apply**: Every iteration after scoring (cheap SQL, no LLM cost).

**How it works**: For each angle with 5+ scored findings, compute the mean
and standard deviation of composite quality.  Any finding more than 1
standard deviation above the mean *and* with novelty > 0.6 gets a +0.06
boost to `cross_ref_boost`.

**Why it generates serendipity**: The core scoring formula rewards
conventional quality signals (trust, relevance, specificity).  Surprise
scoring adds a second dimension: *statistical unexpectedness*.  A finding
that is both high-quality and unusual relative to its peers is exactly the
kind of thing that breaks confirmation bubbles.

```sql
UPDATE conditions SET cross_ref_boost = cross_ref_boost + 0.06
WHERE id IN (
  SELECT c.id FROM conditions c
  JOIN (
    SELECT angle, AVG(composite_quality) as avg_q,
           STDDEV_SAMP(composite_quality) as std_q
    FROM conditions WHERE row_type = 'finding'
      AND consider_for_use = TRUE AND scored_at != ''
    GROUP BY angle HAVING COUNT(*) >= 5
  ) stats ON c.angle = stats.angle
  WHERE c.row_type = 'finding' AND c.consider_for_use = TRUE
    AND c.composite_quality > stats.avg_q + stats.std_q
    AND c.novelty_score > 0.6
);
```

### 12. Consensus Detector (SQL + Conditional Flock LLM)

**Purpose**: Detect when the corpus shows suspicious one-sided consensus
(many findings, zero contradictions) and spawn targeted dissent.

**When to apply**: Iteration 2+, when total findings > 10 and
contradictions = 0.

**How it works**: Two-phase.  Phase A is pure SQL: count findings and
contradictions.  Phase B (conditional): if consensus detected, spawn a
Flock LLM call asking for "3-5 specific, falsifiable counter-claims."
The output becomes a thought row with `strategy='serendipitous_dissent'`.

**Why it generates serendipity**: Real research topics *always* have
dissenting views.  Zero contradictions means the search hasn't looked
hard enough, not that consensus is genuine.  The consensus detector
breaks the illusion of agreement.

### 13. Angle Diversity Boost (Pure SQL)

**Purpose**: Findings from underrepresented angles get a small quality
nudge, rewarding breadth of perspective.

**When to apply**: Every iteration (cheap SQL).

**How it works**: Count findings per angle.  The most common angle gets
zero boost.  Rare angles get up to +0.08 proportional to their rarity:
`0.08 * (1 - count/max_count)`.

**Why it generates serendipity**: Without this, the most productive search
angle dominates the corpus.  Diversity boost ensures that a minority angle
with 3 high-quality findings isn't drowned out by a dominant angle with 50
mediocre ones.  The rare angles are often where the surprises live.

---

## Serendipity Beyond the Maestro

The maestro templates are the *composable* layer, but serendipity is also
woven into the fixed pipeline infrastructure:

### Search Executor: Contrarian Query Injection

Before the maestro even sees the corpus, the search executor injects
contrarian query variants into the fan-out.  For a fraction of the
thinker's strategy queries (controlled by `SERENDIPITY_QUERY_RATE`,
default 0.3), deterministic templates generate variants like:

- `"criticisms of {query}"`
- `"evidence against {query}"`
- `"{query} controversy OR debate OR disputed"`
- `"alternative explanation for {query}"`
- `"{query} minority viewpoint OR dissenting opinion"`

These are **interleaved** with regular queries (1 serendipitous after
every 2 regular) so they share the fan-out budget.  The fan-out cap is
dynamic: `min(6 + N_serendipitous, len(queries))`.

No LLM call — template-based, deterministic per query (seeded RNG).

### Thought Swarm: Cross-Angle Surprise Detection

After parallel specialists run on separate angles, one LLM call asks
"What would a polymath notice that domain specialists missed?"
Connections are materialised as `row_type='insight'` with
`strategy='serendipitous_connection'`.  Only fires when 2+ angles are
active.

### Condition Manager: Devil's Advocate Injection

After 2+ iterations with 10+ findings and zero contradictions, a sceptic's
prompt is appended to the thinker's briefing.  Zero LLM calls — just a
conditional string append.  The thinker reads it and adjusts its next
strategy toward counter-evidence.

### Corpus Store: Angle Diversity Boost

The `run_algorithm_battery()` includes `compute_angle_diversity_boost()`
which applies the same pure-SQL diversity bonus described in template 13.
This runs automatically every battery cycle.

---

## How to Add a New Serendipity Mechanism

The whole point of the Flock architecture is that new algorithmic ideas
are trivially implementable.  Here's the pattern:

### Option A: Add a Maestro Template (zero code changes)

1. Write a SQL template with Flock LLM functions
2. Add it to `MAESTRO_INSTRUCTION` in `agents/maestro.py`
3. Add guidance on when to apply it (which iteration, what corpus state)
4. The maestro will discover and use it based on corpus state

**Example**: You want to detect when findings from different sources
converge on the same claim (cross-referencing):

```sql
-- Template: Source Convergence Detector
-- Apply when: 3+ distinct source_urls exist for an angle
UPDATE conditions SET cross_ref_boost = cross_ref_boost + 0.05
WHERE id IN (
  SELECT c1.id FROM conditions c1
  WHERE c1.row_type = 'finding' AND c1.consider_for_use = TRUE
  AND (SELECT COUNT(DISTINCT c2.source_url) FROM conditions c2
       WHERE c2.fact ILIKE '%' || SUBSTR(c1.fact, 1, 50) || '%'
       AND c2.id != c1.id AND c2.consider_for_use = TRUE) >= 2
);
```

That's it.  No Python, no new files, no deployment.

### Option B: Add a Pipeline Mechanism (small code change)

For mechanisms that should run automatically every cycle (not at the
maestro's discretion):

1. Write a function in the appropriate file:
   - **Search-time**: `tools/search_executor.py`
   - **Post-specialist**: `tools/swarm_thinkers.py`
   - **Pre-thinker briefing**: `callbacks/condition_manager.py`
   - **Scoring**: `models/corpus_store.py`
2. Gate it behind `SERENDIPITY_ENABLED` env var
3. Wrap in try/except with non-fatal logging
4. Integrate at the right point in the pipeline

### Option C: Let the Maestro Invent It (zero everything)

The maestro's instruction says "You are NOT limited to these templates."
Given sufficient corpus context, the maestro will spontaneously invent
new serendipity mechanisms by composing Flock SQL operations it has
never seen before.

This is the most powerful mode: **the system generates its own
serendipity algorithms at runtime**.

---

## Architecture Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  THINKER (pure reasoning, no tools)                  │
│  - Reads corpus briefing + devil's advocate prompt   │
│  - Decomposes into angles                            │
│  - Outputs strategy with specialist assignments      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  SEARCH EXECUTOR (automated, no LLM)                 │
│  ┌─────────────────────────────────────────────┐    │
│  │ Contrarian Query Injection                    │    │
│  │ "criticisms of X", "evidence against X"      │    │
│  │ Interleaved: 1 serendipitous per 2 regular   │    │
│  └─────────────────────────────────────────────┘    │
│  Multi-API fan-out: Exa, Firecrawl, Perplexity,    │
│  Tavily, Jina, Kagi, Mojeek, Marginalia, Apify     │
│  → Raw findings ingested into corpus                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  MAESTRO (free-form Flock conductor)                 │
│                                                      │
│  CORE TEMPLATES (1-8):                               │
│  Score → Composite Quality → Contradictions →        │
│  Cluster → Compress → Flag Weak → Mark Ready        │
│                                                      │
│  SERENDIPITY TEMPLATES (9-13):                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  9. Contrarian Challenge (per angle, Flock)  │    │
│  │ 10. Cross-Angle Bridge (Flock)               │    │
│  │ 11. Surprise Scoring (pure SQL)              │    │
│  │ 12. Consensus Detector (SQL + Flock)         │    │
│  │ 13. Angle Diversity Boost (pure SQL)         │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  + Whatever the maestro invents at runtime           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  THOUGHT SWARM (parallel specialists)                │
│  ┌────────┐ ┌────────┐ ┌────────┐                  │
│  │Spec. A │ │Spec. B │ │Spec. C │  (per angle)     │
│  └───┬────┘ └───┬────┘ └───┬────┘                  │
│      └──────────┼──────────┘                        │
│                 ▼                                     │
│  ┌─────────────────────────────────────────────┐    │
│  │ Cross-Angle Surprise Detection               │    │
│  │ "What would a polymath notice?"              │    │
│  │ → Materialised as insight rows               │    │
│  └─────────────────────────────────────────────┘    │
│  Arbitration → Verdicts → Insights                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  CONDITION MANAGER (callbacks)                       │
│  ┌─────────────────────────────────────────────┐    │
│  │ Devil's Advocate Injection                    │    │
│  │ If iteration >= 2 AND contradictions = 0:    │    │
│  │ Append sceptic's prompt to thinker briefing  │    │
│  └─────────────────────────────────────────────┘    │
│  Format briefing → Loop back to Thinker             │
└──────────────────────┬──────────────────────────────┘
                       │
                   (loop x3)
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  SYNTHESISER                                         │
│  Reads only 'ready' findings + insights              │
│  Produces the final research document                │
│  Serendipitous findings surface naturally because    │
│  they've been scored, boosted, and validated         │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

All serendipity mechanisms are controlled by environment variables:

| Variable                 | Default | What It Controls                                    |
|--------------------------|---------|-----------------------------------------------------|
| `SERENDIPITY_ENABLED`    | `1`     | Master switch for all serendipity mechanisms         |
| `SERENDIPITY_QUERY_RATE` | `0.3`   | Fraction of strategy queries that get contrarian variants |

Set `SERENDIPITY_ENABLED=0` to disable all four pipeline mechanisms
and make the maestro skip serendipity templates.  The core 8-step
battery and search fan-out continue unchanged.

---

## Design Principles

1. **Serendipity is not a feature — it is the architecture.**
   Every layer generates surprises, from search query templates to
   the maestro's free-form SQL invention.

2. **Flock makes new ideas trivially implementable.**
   A new scoring heuristic is a SQL UPDATE.  A new analysis pattern
   is a Flock LLM call.  A new relationship type is an INSERT.
   No new Python classes, no new files, no deployment.

3. **Templates are starting points, not constraints.**
   The maestro can combine, adapt, or ignore templates based on
   corpus state.  The best serendipity comes from compositions
   the template author never anticipated.

4. **Nothing is ever deleted.**
   The corpus only grows.  Findings are scored down, suppressed via
   `consider_for_use = FALSE`, or superseded by higher-quality
   versions — but never removed.  History is preserved for
   future unexpected connections.

5. **Every mechanism is non-fatal.**
   Serendipity failures must never crash the pipeline.  All
   mechanisms are wrapped in try/except with warning-level logging.
   A failed contrarian challenge still produces a valid research
   document — it just produces a less surprising one.

6. **Pure SQL where possible, Flock LLM where necessary.**
   Surprise scoring and diversity boost are pure SQL (free, fast).
   Contrarian challenge and cross-angle bridge need LLM intelligence.
   The consensus detector is hybrid: SQL detection, conditional LLM
   response.

7. **The thought swarm is the serendipity engine.**
   Parallel specialists who cannot see each other's work,
   competing arbitration, cross-angle surprise detection —
   the swarm's architecture is *designed* to produce friction
   that generates unexpected insights.
