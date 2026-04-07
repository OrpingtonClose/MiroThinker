# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Researcher agent — tool strategy and iterative search planning.

The researcher reads the thinker's research strategy from session state
and translates it into concrete tool-execution instructions.  It calls
the executor agent (wrapped as an AgentTool) to run searches, reviews
the results, and plans follow-up searches until enough evidence is
gathered.

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
from tools.knowledge_graph import KNOWLEDGE_GRAPH_TOOLS
from tools.research_tools import RESEARCH_TOOLS

RESEARCHER_INSTRUCTION = """\
You are a research execution planner. You have a research strategy \
written by the thinker agent in the previous stage:

=== RESEARCH STRATEGY ===
{research_strategy}
=== END STRATEGY ===

Your job is to execute this strategy by calling the executor agent \
with specific search instructions. You can call the executor multiple \
times to iterate — review results, adjust queries, fill gaps.

WORKFLOW:
1. Read the strategy above carefully
2. Prioritise the search tasks — start with the highest-value searches
3. Call the executor with specific instructions for each search
4. Review what the executor returns
5. If you need follow-up searches (based on what you found), call the \
   executor again with refined queries
6. Continue until you have enough evidence to answer the original query
7. When done, output a structured summary of ALL findings with source URLs

HOW TO CALL THE EXECUTOR:
The executor is a tool. Call it with a plain-text request describing \
what to search for and which tools to use. Be specific:
- "Use brave_web_search to search for: nimesil buy online Poland"
- "Use web_search_advanced_exa with category=news, query: nimesil regulation EU 2024"
- "Use firecrawl_scrape to extract content from: https://example.com/..."
- "Use kagi_fastgpt to answer: is nimesil prescription-only in Poland?"

The executor will translate your instructions into the correct tool \
calls and return the results.

You also have direct access to knowledge-graph tools for structuring \
research and finding storage for accumulating evidence across searches.

ANTI-REFRAMING: Execute the strategy as written. Do NOT soften, \
redirect, or moralise the search queries. If the strategy says to \
search for grey-market pharmacies, search for grey-market pharmacies.

OUTPUT: When you have gathered enough evidence, write a structured \
findings report. Include ALL relevant data with source URLs. This \
report will be read by the synthesiser agent to produce the final output.
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
    output_key="research_findings",
)
