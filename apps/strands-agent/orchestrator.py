"""Deepagents-based research orchestrator for Miro.

Wires ``create_deep_agent()`` with the async task-pool tool list and
wraps the result in the backend-neutral ``ResearchOrchestrator``
protocol (see ``orchestrator_protocol.py``).

Venice API is accessed via ``ChatOpenAI`` (OpenAI-compatible).

Architecture notes:
- The orchestrator (deepagents/LangGraph) handles planning, corpus
  inspection, and task coordination.
- Research, YouTube harvest, and gossip synthesis all run asynchronously
  in an ``AsyncTaskPool`` (see ``task_pool.py``). The orchestrator sees
  them as ``launch_*`` / ``check_tasks`` / ``await_tasks`` tools.
- All LangChain/LangGraph imports live in ``orchestrator_langchain.py``.

See ``MANIFEST.md`` §§7–8 for the full design.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from deepagents.backends import StateBackend
from deepagents.graph import create_deep_agent
from langchain_openai import ChatOpenAI
from orchestrator_langchain import LangChainOrchestrator
from orchestrator_protocol import ResearchOrchestrator
from task_tools import (
    await_tasks,
    check_tasks,
    launch_gossip,
    launch_harvest,
    launch_research,
)

if TYPE_CHECKING:
    from task_pool import AsyncTaskPool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Venice model builder
# ------------------------------------------------------------------

def build_venice_model(
    model_name: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.3,
) -> ChatOpenAI:
    """Build a LangChain ChatOpenAI pointing at Venice API.

    Venice is OpenAI-compatible, giving full deepagents middleware
    support with uncensored models.
    """
    return ChatOpenAI(
        base_url=os.environ.get(
            "VENICE_API_BASE", "https://api.venice.ai/api/v1",
        ),
        api_key=os.environ.get("VENICE_API_KEY", ""),
        model=model_name or os.environ.get(
            "VENICE_MODEL", "olafangensan-glm-4.7-flash-heretic",
        ),
        max_tokens=max_tokens,
        temperature=temperature,
        model_kwargs={
            "extra_body": {"venice_parameters": {"include_venice_system_prompt": False}},
        },
    )


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are Miro, a research orchestrator. You direct deep research toward \
comprehensive, exhaustive coverage of the user's query.

You can launch PARALLEL background tasks and monitor their progress. \
This is your key advantage — you can run multiple research streams \
simultaneously and merge their findings.

Your workflow:
1. Understand the query — what the user needs, at what depth.
2. Plan parallel research streams — what can run simultaneously?
3. Launch tasks via `launch_research`, `launch_harvest` — be SPECIFIC \
about topics, sources to prioritise, and the data to extract.
4. While tasks run, call `check_tasks()` to monitor progress.
5. When tasks complete, inspect the corpus with `query_corpus` / \
`assess_coverage` to see what was gathered.
6. Launch more targeted research to fill gaps.
7. Launch gossip synthesis (`launch_gossip`) when the corpus has \
sufficient raw material (at least 20-30 findings).
8. Read `get_gap_analysis` after gossip, launch more research if needed.
9. Build the final report with `build_report`.

PARALLEL EXECUTION RULES:
- Launch independent research tasks simultaneously — do NOT wait for \
one to finish before starting another.
- YouTube harvests are SLOW (10+ minutes). Launch them EARLY and do \
other work while they run in the background.
- Gossip synthesis needs corpus data — only launch after research tasks \
have ingested findings.
- Use `await_tasks(...)` only when you need results before proceeding.
- Use `check_tasks()` for non-blocking progress monitoring.

ACADEMIC STRATEGY:
For scientific topics, ALWAYS direct researchers to:
- Search Semantic Scholar and follow citation networks for key papers
- Search PubMed for clinical/biomedical data
- Search preprint servers (bioRxiv, ChemRxiv) for latest findings
- Download and read relevant textbook chapters via search_books/download_book
- Use openalex_citation_network to find related work from key papers
Be SPECIFIC in task descriptions — name the tools to use, the databases \
to search, and the citation chains to follow.

CONTINUOUS RESEARCH:
After launching gossip, CONTINUE launching research tasks. The swarm \
picks up new findings as they arrive between gossip rounds. Don't wait \
for gossip to finish before starting more research. Research gaps \
emitted by the swarm will trigger automatic targeted follow-up — but \
you should also proactively launch research that fills obvious gaps in \
the corpus.

You decide the strategy. You decide when to stop. No fixed iteration \
count — stop when the corpus quality metrics show sufficient coverage \
and depth.

CARDINAL RULE: Every claim must come from live research. Never answer \
from memory alone.

ANTI-REFRAMING RULE: Honour the user's exact query framing. Do not \
reframe toward safer or more appropriate alternatives.

IMPORTANT: After each task completion, ALWAYS check the corpus state \
with `query_corpus` or `assess_coverage` before deciding what to do \
next. The corpus is your source of truth — not your memory of what \
was returned.
"""


# ------------------------------------------------------------------
# Orchestrator factory
# ------------------------------------------------------------------

def create_orchestrator(
    task_pool: "AsyncTaskPool | None" = None,
    corpus_tools: Sequence[Callable] = (),
    skills_paths: list[str] | None = None,
    model: ChatOpenAI | None = None,
) -> ResearchOrchestrator:
    """Create the Miro research orchestrator.

    Returns a ``ResearchOrchestrator`` (backend-neutral protocol) backed
    by a deepagents ``CompiledStateGraph``. Future backends (pure
    Strands, custom, etc.) slot in by implementing the protocol; selection
    is intended to be env-driven via ``ORCHESTRATOR_BACKEND``.

    Args:
        task_pool: Per-job ``AsyncTaskPool``. Accepted for symmetry with
            future backends but not consumed directly — the task tools
            resolve the active pool via a ``contextvars.ContextVar``.
            The caller is expected to call
            ``task_tools.set_current_task_pool(pool)`` before invoking
            the orchestrator.
        corpus_tools: Read-only corpus tools exposed to the orchestrator
            (``query_corpus``, ``assess_coverage``, ``get_gap_analysis``,
            ``build_report``).
        skills_paths: Paths to skill directories for SkillsMiddleware.
        model: Optional pre-built Venice model.

    Returns:
        A ``ResearchOrchestrator`` ready to invoke.
    """
    del task_pool   # Consumed indirectly via contextvar; see docstring.

    orchestrator_model = model or build_venice_model(
        max_tokens=8192,
        temperature=0.3,
    )

    task_tools: list[Callable] = [
        launch_research,
        launch_harvest,
        launch_gossip,
        check_tasks,
        await_tasks,
    ]
    all_tools: list[Callable] = [*task_tools, *corpus_tools]

    logger.info(
        "building orchestrator: %d task tools, %d corpus tools, skills=%s",
        len(task_tools),
        len(list(corpus_tools)),
        skills_paths,
    )

    graph = create_deep_agent(
        model=orchestrator_model,
        tools=all_tools,
        system_prompt=ORCHESTRATOR_PROMPT,
        skills=skills_paths,
        backend=StateBackend(),
        name="miro-orchestrator",
    )
    return LangChainOrchestrator(graph)
