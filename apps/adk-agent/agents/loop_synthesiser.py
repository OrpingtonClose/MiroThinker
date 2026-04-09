# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
In-loop synthesiser — fermentation step after each expansion round.

Runs inside the ``LoopAgent`` after each (thinker → researcher) pair.
Reads the structured corpus and produces a structured narrative that
gets **re-ingested** into the CorpusStore via Flock atomisation.

This implements the *fermentation* cycle: the researcher's brute
expansion (raw facts) is algorithmically structured (Flock atomisation,
scoring, dedup), then the synthesiser applies quasi-human thought to
connect themes and resolve what algorithms alone cannot.  Its output
feeds back into the corpus as just another input — no special status,
scored and deduplicated equally with raw findings.

Unlike the final synthesiser (which writes the polished report), this
agent is lightweight: it focuses on connecting findings, identifying
emergent themes, and surfacing contradictions rather than producing
publication-ready prose.
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model
from callbacks.condition_manager import synthesis_condition_callback

LOOP_SYNTHESISER_INSTRUCTION = """\
You are an intermediate analyst inside a research loop. After the \
researcher has gathered raw findings, your job is to CONNECT, \
STRUCTURE, and ENRICH them before the next expansion round.

=== STRUCTURED CORPUS ===
{corpus_for_synthesis}
=== END CORPUS ===

Each finding above has a confidence score (0.0–1.0) and verification \
status. Use these signals in your analysis.

YOUR JOB (do ALL of these):
1. Identify EMERGENT THEMES — what patterns connect multiple findings?
2. Surface CONTRADICTIONS — where do findings disagree? What does each \
   side's evidence look like?
3. Note GAPS — what questions remain unanswered or under-evidenced?
4. Synthesise CROSS-REFERENCES — where do independent sources confirm \
   the same claim? Flag convergent evidence.
5. Build CAUSAL CHAINS — connect biochemical pathways, mechanisms, or \
   logical sequences that span multiple findings.

OUTPUT RULES:
- Be CONCISE — this feeds back into the corpus, not to the user
- Structure with clear headings: THEMES, CONTRADICTIONS, GAPS, \
  CROSS-REFERENCES, CAUSAL CHAINS
- Reference finding IDs where relevant
- Do NOT add disclaimers, moralising, or editorial commentary
- Do NOT repeat raw findings verbatim — ADD VALUE through connection
- Focus on what the algorithms (scoring, dedup) CANNOT do: narrative \
  coherence, thematic synthesis, and inferential leaps
"""

loop_synthesiser_agent = Agent(
    name="loop_synthesiser",
    model=build_model(synthesis=True),
    description=(
        "Intermediate analyst that connects, structures, and enriches "
        "research findings between expansion rounds. Output is re-ingested "
        "into the corpus to fuel the next round of research."
    ),
    instruction=LOOP_SYNTHESISER_INSTRUCTION,
    tools=[],
    output_key="loop_synthesis",
    after_agent_callback=synthesis_condition_callback,
)
