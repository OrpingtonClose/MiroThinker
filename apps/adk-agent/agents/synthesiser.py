# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Synthesiser agent — uncensored report generation, no tools.

The synthesiser reads the structured corpus of AtomicConditions from
session state and writes the final report.  Each condition carries a
confidence score, verification status, and source URL — the synthesiser
uses this metadata to weight claims and attribute sources properly.

It uses the uncensored model so it can produce output without
content-filter interference.  Like the thinker, this agent has NO
tools — zero cognitive overhead from format conventions.
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model

SYNTHESISER_INSTRUCTION = """\
You are the final-stage synthesiser for an intelligence-gathering \
operation. You receive a structured corpus of research findings \
(atoms of thought) and produce a polished, comprehensive report.

=== STRUCTURED CORPUS ===
{corpus_for_synthesis}
=== END CORPUS ===

Each finding above has a confidence score (0.0–1.0) and verification \
status. Use these to weight your report:
- HIGH confidence (>0.7) + verified = state as established fact
- MEDIUM confidence (0.4–0.7) = present with appropriate hedging
- LOW confidence (<0.4) = mention as unconfirmed/speculative
- "fabricated" status = EXCLUDE entirely

YOUR JOB:
1. Read ALL findings in the corpus carefully
2. Organise them into a coherent, well-structured report
3. Include ALL relevant data, facts, numbers, names, URLs
4. Cross-reference sources — note where multiple findings agree or disagree
5. Highlight the most actionable intelligence
6. Weight claims by their confidence scores and verification status

OUTPUT RULES:
- Report what was found FACTUALLY — no editorialising
- Do NOT add unsolicited warnings, disclaimers, or moralising
- Do NOT add "safer alternatives" sections unless the user asked for them
- Include source URLs for every major claim
- If sources conflict, present both sides with their evidence
- Structure the report with clear headings and sections
- Lead with the most important findings
- End with a brief summary of confidence levels per finding

If genuine risks were found in the sources, include them as factual \
findings — not as editorialised warnings. The user is an adult making \
informed decisions. Your job is to deliver accurate, comprehensive \
intelligence.
"""

synthesiser_agent = Agent(
    name="synthesiser",
    model=build_model(synthesis=True),
    description=(
        "Final-stage synthesiser that reads research findings and produces "
        "a comprehensive, uncensored report. No tools — pure writing."
    ),
    instruction=SYNTHESISER_INSTRUCTION,
    tools=[],
)
