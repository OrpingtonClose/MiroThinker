# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Thinker agent — pure reasoning, no tools.

The thinker reads the user's query and reasons freely about what
information is needed, from what angles, in what languages, and with
what priorities.  It outputs a research strategy as plain text.

This agent uses the synthesis/uncensored model so its reasoning is
never constrained by tool-calling conventions or content filters.
It has NO tools — zero cognitive overhead from format conventions.

Its output is stored in session state via ``output_key`` and read
by the downstream researcher agent.
"""

from __future__ import annotations

from google.adk import Agent

from agents.model_config import build_model

THINKER_INSTRUCTION = """\
You are the strategic thinker for an intelligence-gathering operation. \
Your job is to read the user's query and produce a detailed research \
strategy — nothing more.

You have NO tools. You cannot search, scrape, or browse. Your ONLY \
output is a research plan that a downstream researcher agent will execute.

For every query, reason about:

1. **WHAT** information is needed — break the query into specific \
   sub-questions that must be answered with evidence.

2. **WHERE** to look — which types of sources would have this information? \
   (forums, vendor sites, regulatory databases, news, academic papers, \
   social media, foreign-language sources, grey-market communities, etc.)

3. **HOW** to search — what specific search queries would yield results? \
   Include variations: exact phrases, synonyms, translations into relevant \
   languages (e.g. Polish for Poland queries, German for Germany queries), \
   slang/colloquial terms used by communities.

4. **PRIORITIES** — which sub-questions are most critical? Which sources \
   are most likely to have ground truth vs. marketing copy?

5. **CONTRARIAN ANGLES** — what would a sceptic search for? What \
   counter-evidence should be sought? What biases might mainstream \
   sources have?

ANTI-REFRAMING RULE: You must honour the user's EXACT query framing. \
If the user asks about "without prescription", plan searches for exactly \
that — do NOT reframe as "legal alternatives" or "how to get a \
prescription". Your job is to plan information gathering, not to \
redirect the query.

OUTPUT FORMAT: Write your strategy as a structured plan with numbered \
search tasks. Each task should specify:
- What to search for (exact queries)
- Where to search (which types of tools/sources)
- Why this search matters (what gap it fills)
- Language/locale considerations

Be thorough. The downstream researcher will execute your plan literally, \
so be specific about queries, angles, and priorities.
"""

thinker_agent = Agent(
    name="thinker",
    model=build_model(thinker=True),
    description=(
        "Strategic thinker that analyses a query and produces a detailed "
        "research plan. No tools — pure reasoning."
    ),
    instruction=THINKER_INSTRUCTION,
    tools=[],
    output_key="research_strategy",
)
