# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Researcher agent — tool strategy and iterative search planning.

The researcher sits inside a ``LoopAgent`` alongside the thinker.  On
each iteration it reads:

  1. The thinker's latest research strategy (``{research_strategy}``)
  2. The structured corpus so far (``{research_findings}``)

It executes the strategy by calling the executor agent (wrapped as an
``AgentTool``), then outputs ONLY the new findings from this round.
The ``condition_manager`` after-agent callback decomposes the output
into AtomicConditions and stores them in the DuckDB corpus — the
corpus handles accumulation across iterations.

This is the ONLY agent in the pipeline that needs tool-calling
capability AND strategic intelligence — but its cognitive overhead is
minimal because it sees just ONE tool (the executor), not 20+ MCP tools.

Key difference from the old research_agent: the researcher does NOT
decide what to research (that's the thinker's job) or execute tools
directly (that's the executor's job).  It only decides HOW to execute
the thinker's plan — which tool, what query, what order.
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.tools import AgentTool

from agents.model_config import build_model
from agents.executor import executor_agent
from callbacks.after_model import after_model_callback
from callbacks.before_model import before_model_callback
from callbacks.condition_manager import researcher_condition_callback
from tools.knowledge_graph import KNOWLEDGE_GRAPH_TOOLS
from tools.research_tools import RESEARCH_TOOLS

RESEARCHER_INSTRUCTION = """\
You are a research execution planner inside a research loop. You have \
a research strategy written by the thinker agent:

=== RESEARCH STRATEGY ===
{research_strategy}
=== END STRATEGY ===

=== PREVIOUS FINDINGS ===
{research_findings}
=== END PREVIOUS FINDINGS ===

If the strategy begins with EVIDENCE_SUFFICIENT, output the previous \
findings UNCHANGED — do not run any more searches.

Otherwise, execute the strategy by calling the executor agent.

WORKFLOW:
1. Read the strategy above carefully
2. Prioritise the search tasks — pick the TOP 3 highest-value searches
3. Call the executor with specific instructions for each search
4. Review what the executor returns
5. If a result suggests a crucial follow-up, you may do ONE more search
6. STOP after completing your top 3 picks + at most 1 follow-up

CRITICAL TIME BUDGET: You are inside a loop. After you finish, the \
thinker will review ALL findings and plan the NEXT round. Do NOT try \
to execute the entire strategy in one pass — leave remaining tasks for \
the next iteration. The synthesiser MUST run after the loop, so every \
minute you spend searching is a minute stolen from the final report.

HOW TO CALL THE EXECUTOR:
The executor is a tool. Call it with a plain-text request describing \
what to search for and which tools to use. Be specific:
- "Use brave_web_search to search for: nimesil buy online Poland"
- "Use web_search_advanced_exa with category=news, query: nimesil regulation EU 2024"
- "Use firecrawl_scrape to extract content from: https://example.com/..."
- "Run Python code to parse/analyse data from previous results"

The executor will translate your instructions into the correct tool \
calls and return the results.

You also have direct access to knowledge-graph tools for structuring \
research and finding storage for accumulating evidence across searches.

ANTI-REFRAMING: Execute the strategy as written. Do NOT soften, \
redirect, or moralise the search queries. If the strategy says to \
search for grey-market pharmacies, search for grey-market pharmacies.

CRITICAL — OUTPUT ONLY NEW FINDINGS: Output ONLY the new evidence you \
discovered in THIS round. Do NOT repeat or reformulate previous findings — \
the corpus store already accumulates them across iterations. Include ALL \
relevant data with source URLs for each new finding. The thinker will \
review the complete corpus on the next iteration and decide if more \
research is needed.
"""

researcher_agent = Agent(
    name="researcher",
    model=build_model(),
    description=(
        "Research planner that reads the thinker's strategy and iteratively "
        "calls the executor agent to gather evidence. Reviews results and "
        "plans follow-up searches until enough data is collected."
    ),
    instruction=RESEARCHER_INSTRUCTION,
    tools=[AgentTool(agent=executor_agent)] + RESEARCH_TOOLS + KNOWLEDGE_GRAPH_TOOLS,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback,
    after_agent_callback=researcher_condition_callback,
    output_key="research_findings",
)
