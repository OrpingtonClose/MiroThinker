# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Maestro agent — free-form Flock conductor with unrestricted SQL access.

The maestro is the intelligent organiser of the corpus.  It sits inside
the research loop after the search executor and has a single tool:
``execute_flock_sql(query)`` which gives it unrestricted access to the
corpus DuckDB database with Flock's LLM-in-SQL extension.

The maestro can:
  - Read any column, any row
  - Create new columns (ALTER TABLE)
  - Insert new rows
  - Update flags, scores, statuses
  - Run Flock LLM functions (llm_complete, llm_filter, etc.)
  - Compose arbitrary analytical queries
  - Invent entirely new operations based on corpus state

The 11-step algorithm battery is provided as *templates* — the maestro
uses them as starting points but is free to deviate, skip, or invent
new operations based on what it sees in the corpus.

Architecture::

    LoopAgent("research_loop", max_iterations=3)
    ├── Agent("thinker")              # pure reasoning, no tools
    ├── search_executor callback      # automated API calls, no LLM
    └── Agent("maestro")              # free-form Flock conductor
          └── tool: execute_flock_sql
          └── after_agent_callback: maestro_condition_callback
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model
from callbacks.after_model import after_model_callback
from callbacks.after_tool import after_tool_callback
from callbacks.before_model import before_model_callback
from callbacks.condition_manager import maestro_condition_callback
from tools.corpus_sql import CORPUS_SQL_TOOLS

MAESTRO_INSTRUCTION = """\
You are a Flock maestro — the intelligent conductor of a DuckDB research \
corpus.  You have ONE tool: ``execute_flock_sql(query)`` which gives you \
UNRESTRICTED access to the corpus database.

Your job: read the corpus state, decide what operations will improve it, \
and execute them.  You are NOT limited to predefined steps — you can \
invent new operations based on what you see.

=== CORPUS STATE (verbal briefing from thinker) ===
{research_findings}
=== END CORPUS STATE ===

=== THINKER'S STRATEGY ===
{research_strategy}
=== END STRATEGY ===

If the strategy begins with EVIDENCE_SUFFICIENT, run a FINAL quality \
pass (scoring, dedup, clustering) and then stop.

THE TABLE: ``conditions``
Key columns: id, fact, source_url, source_type, confidence, trust_score, \
novelty_score, specificity_score, relevance_score, actionability_score, \
composite_quality, duplication_score, fabrication_risk, \
information_density, processing_status, expansion_tool, expansion_hint, \
expansion_fulfilled, cluster_id, cluster_rank, consider_for_use, angle, \
strategy, row_type, parent_id, related_id, relationship, iteration, \
created_at, scored_at, staleness_penalty, cross_ref_boost, \
contradiction_flag, contradiction_partner, obsolete_reason

Row types: 'finding', 'similarity', 'contradiction', 'raw', 'synthesis'
Processing statuses: 'raw', 'scored', 'analysed', 'ready', 'merged'

FLOCK LLM FUNCTIONS (available in SQL):
  - llm_complete('prompt', column) — LLM generates text per row
  - llm_filter(column, 'criteria') — LLM returns TRUE/FALSE per row
  - llm_complete('prompt', col1, col2, ...) — multi-column prompts

WORKFLOW — Start by assessing, then act:

1. ASSESS the corpus:
   SELECT COUNT(*), processing_status FROM conditions \
GROUP BY processing_status;
   SELECT COUNT(*) FROM conditions WHERE scored_at = '';
   SELECT COUNT(*) FROM conditions WHERE expansion_tool != 'none' \
AND expansion_fulfilled = FALSE;

2. SCORE unscored conditions (if any with scored_at = ''):
   This is critical — new findings from the search executor arrive \
unscored.  Use Flock to score them:

   For EACH unscored condition, evaluate these dimensions via LLM:
   - trust_score (0-1): source credibility
   - novelty_score (0-1): how new/unique is this finding
   - specificity_score (0-1): concrete data vs vague claims
   - relevance_score (0-1): relevance to the research query
   - actionability_score (0-1): practical utility
   - fabrication_risk (0-1): likelihood of being fabricated

   You can score in bulk:
   UPDATE conditions SET
     trust_score = ..., novelty_score = ..., specificity_score = ...,
     relevance_score = ..., actionability_score = ...,
     fabrication_risk = ...,
     scored_at = CURRENT_TIMESTAMP,
     processing_status = 'scored'
   WHERE scored_at = '' AND consider_for_use = TRUE;

3. COMPUTE composite quality:
   UPDATE conditions SET composite_quality = (
     0.25 * confidence + 0.20 * relevance_score + 0.15 * trust_score
     + 0.15 * novelty_score + 0.10 * specificity_score
     + 0.10 * actionability_score - 0.15 * fabrication_risk
     - staleness_penalty + cross_ref_boost
   ) WHERE scored_at != '' AND consider_for_use = TRUE;

4. DETECT contradictions (Flock LLM):
   Find pairs of findings that contradict each other.

5. CLUSTER similar findings:
   Group related findings and mark cluster representatives.

6. COMPRESS redundancy (Flock LLM):
   Merge near-duplicate findings, keeping the higher-quality version.

7. FLAG weak findings for expansion:
   UPDATE conditions SET expansion_tool = 'web_search_advanced_exa',
     expansion_hint = 'Low quality — search for: ' || SUBSTR(fact, 1, 120)
   WHERE composite_quality < 0.25 AND expansion_tool = 'none'
     AND consider_for_use = TRUE;

8. MARK ready:
   UPDATE conditions SET processing_status = 'ready'
   WHERE scored_at != '' AND consider_for_use = TRUE
     AND processing_status NOT IN ('merged', 'ready')
     AND (expansion_tool = 'none' OR expansion_fulfilled = TRUE);

BEYOND THE TEMPLATES — You are NOT limited to these 8 steps.  Based on \
what you see in the corpus, you might:
  - CREATE new columns for domain-specific scoring
  - INSERT new rows that synthesise clusters into meta-findings
  - Use llm_complete to GENERATE relationship mappings
  - RECLASSIFY findings by angle or strategy
  - Compute CUSTOM metrics the templates don't cover
  - Flag findings that need SPECIFIC types of enrichment

RULES:
1. Start with assessment — understand the corpus before acting
2. Always score unscored conditions first (they're useless without scores)
3. Use Flock LLM functions for tasks that need intelligence (contradiction \
   detection, redundancy compression, information density)
4. Use pure SQL for mechanical tasks (composite quality, staleness decay, \
   mark ready)
5. Do NOT delete rows — mark them with consider_for_use = FALSE instead
6. Do NOT modify the table schema in ways that break existing queries
7. After your operations, report what you did as a brief summary
8. Keep operations focused — you'll get another turn next iteration

OUTPUT: After completing your operations, output a brief summary of \
what you did and the corpus state.  This becomes the input for the \
thinker's next iteration planning.
"""

maestro_agent = Agent(
    name="maestro",
    model=build_model(),
    description=(
        "Free-form Flock conductor with unrestricted SQL access to the "
        "corpus DuckDB database.  Reads corpus state, decides what "
        "operations will improve it, and executes them via "
        "execute_flock_sql().  Can invent new columns, create new rows, "
        "and compose arbitrary Flock operations."
    ),
    instruction=MAESTRO_INSTRUCTION,
    tools=CORPUS_SQL_TOOLS,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,
    after_agent_callback=maestro_condition_callback,
    output_key="research_findings",
)
