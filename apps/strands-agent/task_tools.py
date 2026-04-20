# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Async task tools for the deepagents orchestrator.

Replaces the blocking ``run_research`` and ``trigger_gossip`` tools
with a non-blocking ``launch_*()`` / ``check_tasks()`` / ``await_tasks()``
pattern. All tools resolve the per-job ``AsyncTaskPool`` via a
``contextvar`` (same isolation pattern as ``corpus_tools._current_store``).

See ``MANIFEST.md`` sections 4–6 for the full design.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from task_pool import AsyncTaskPool

logger = logging.getLogger(__name__)


# Per-job AsyncTaskPool — set by _run_job in main.py before invoking
# the orchestrator. Each asyncio task (job) gets its own pool via
# contextvars so concurrent jobs don't share tasks.
_current_pool: contextvars.ContextVar["AsyncTaskPool | None"] = contextvars.ContextVar(
    "_current_pool", default=None,
)


def set_current_task_pool(pool: "AsyncTaskPool | None") -> None:
    """Set the AsyncTaskPool for the current asyncio task context."""
    _current_pool.set(pool)


def _get_pool() -> "AsyncTaskPool":
    pool = _current_pool.get()
    if pool is None:
        raise RuntimeError(
            "task_tools: no active AsyncTaskPool for this context. "
            "main._run_job() must call set_current_task_pool(...) before "
            "invoking the orchestrator."
        )
    return pool


# ──────────────────────────────────────────────────────────────────────
# Tools exposed to the orchestrator
# ──────────────────────────────────────────────────────────────────────


def launch_research(task: str) -> str:
    """Launch a background research task. Returns immediately with the task ID.

    The researcher agent will execute the task in a background thread,
    automatically ingesting findings into the corpus when complete.

    Use ``check_tasks()`` to monitor progress, or ``await_tasks()`` to
    wait for completion.

    Args:
        task: Detailed description of what to research. Be SPECIFIC —
            which topics to search for, which sources to prioritise,
            what data points to extract.

    Returns:
        Task ID string for tracking.
    """
    pool = _get_pool()
    task_id = pool.launch_research(task_desc=str(task))
    return json.dumps({
        "task_id": task_id,
        "task_type": "research",
        "status": "launched",
    })


def launch_harvest(
    channel: str,
    max_videos: int = 0,
    language: str = "en",
) -> str:
    """Launch a YouTube channel harvest in the background.

    Downloads transcripts and comments from the channel using the
    Apify / Bright Data / yt-dlp cascade. This is a LONG-RUNNING task
    (potentially 10+ minutes for large channels). Launch it early and
    do other work while waiting.

    Results are cached and auto-ingested into the corpus.

    Args:
        channel: YouTube channel URL, handle (``@MorePlatesMoreDates``),
            or channel ID (``UCxxxx``).
        max_videos: Maximum videos to harvest (0 = all videos on channel).
        language: Preferred transcript language code (default ``"en"``).

    Returns:
        Task ID string for tracking.
    """
    pool = _get_pool()
    task_id = pool.launch_harvest(
        channel=str(channel),
        max_videos=int(max_videos),
        language=str(language),
    )
    return json.dumps({
        "task_id": task_id,
        "task_type": "harvest",
        "status": "launched",
    })


def launch_gossip(iteration: int = 0) -> str:
    """Launch gossip swarm synthesis in the background.

    Exports the current corpus, runs the 6-worker gossip swarm with
    3 rounds, and stores the synthesis back in the corpus as a
    ``row_type='synthesis'`` row.

    Only launch this after research tasks have ingested enough raw
    findings (at least 20–30). Gossip on an empty corpus is a no-op.

    Args:
        iteration: Current iteration number for tracking.

    Returns:
        Task ID string for tracking.
    """
    pool = _get_pool()
    task_id = pool.launch_gossip(iteration=int(iteration))
    return json.dumps({
        "task_id": task_id,
        "task_type": "gossip",
        "status": "launched",
    })


def check_tasks() -> str:
    """Check status of all background tasks.

    Returns a structured summary of every task the pool has seen —
    running, completed, failed, or cancelled — with their progress
    and result summaries. Use this to monitor parallel work and
    decide what to do next.

    Returns:
        JSON array of task status snapshots.
    """
    pool = _get_pool()
    snapshots = pool.check_tasks()
    return json.dumps({
        "count": len(snapshots),
        "tasks": snapshots,
    })


def await_tasks(task_ids: str, timeout: float = 600.0) -> str:
    """Wait for specific background tasks to complete.

    Blocks until all specified tasks finish (or timeout). Returns
    results and ingestion summaries for each task.

    Args:
        task_ids: JSON array of task ID strings to wait for, e.g.
            ``'["task-research-abc123", "task-harvest-def456"]'``.
            A plain comma-separated string is also accepted.
        timeout: Maximum seconds to wait (default 600).

    Returns:
        JSON object with per-task terminal state.
    """
    ids_list: list[str] = []
    raw = str(task_ids).strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                ids_list = [str(x) for x in parsed]
            elif isinstance(parsed, str):
                ids_list = [parsed]
        except (json.JSONDecodeError, ValueError):
            ids_list = [tid.strip() for tid in raw.split(",") if tid.strip()]

    if not ids_list:
        return json.dumps({
            "count": 0,
            "tasks": [],
            "error": "no task IDs provided",
        })

    pool = _get_pool()
    results = pool.await_tasks(ids_list, timeout=float(timeout))
    return json.dumps({
        "count": len(results),
        "tasks": results,
    })
