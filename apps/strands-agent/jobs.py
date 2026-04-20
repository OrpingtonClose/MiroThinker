# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Async job state management for /query/multi streaming.

Provides an in-memory job registry that tracks background research jobs.
Each job has an asyncio.Queue for SSE event streaming, an asyncio.Event
for cancellation, and storage for the final result.

Jobs are auto-purged after a configurable TTL to prevent memory leaks
from abandoned or completed jobs.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# TTL for completed/failed/cancelled jobs before cleanup (seconds)
_JOB_TTL = int(60 * 60)  # 1 hour


class JobCancelledError(Exception):
    """Raised when a job is cancelled mid-execution."""


@dataclass
class JobState:
    """Tracks a single /query/multi background job."""

    job_id: str
    query: str
    status: str = "pending"  # pending | running | complete | failed | cancelled
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    # Cancellation signal — checked between phases
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    # SSE consumers read events from here
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Final result — set on completion
    result: dict[str, Any] | None = None

    # Progress tracking
    current_phase: str = "idle"  # idle | research | gossip
    current_iteration: int = 0
    total_iterations: int = 0
    tool_calls: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    def emit(self, event: dict[str, Any]) -> None:
        """Put a structured event onto the queue for SSE consumers."""
        event.setdefault("time", time.time())
        self.event_queue.put_nowait(event)

    def snapshot(self) -> dict[str, Any]:
        """Lightweight status snapshot for polling."""
        return {
            "job_id": self.job_id,
            "query": self.query,
            "status": self.status,
            "current_phase": self.current_phase,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "tool_calls": self.tool_calls,
            "elapsed_s": round(time.time() - self.created_at, 1),
            "error": self.error,
        }


class JobStore:
    """In-memory job registry.

    Thread-safe for reads (dict access is atomic in CPython).
    Writes are only done from the asyncio event loop thread.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, JobState] = {}

    def create(self, query: str, iterations: int) -> JobState:
        """Create and register a new job."""
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = JobState(
            job_id=job_id,
            query=query,
            total_iterations=iterations,
        )
        self._jobs[job_id] = job
        logger.info(
            "job_id=<%s>, iterations=<%d> | job created",
            job_id, iterations,
        )
        return job

    def get(self, job_id: str) -> JobState | None:
        """Retrieve a job by ID."""
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Signal cancellation for a running job.

        Returns True if the job was found and cancellation was signalled.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status not in ("pending", "running"):
            return False
        job.cancel_event.set()
        logger.info("job_id=<%s> | cancellation signalled", job_id)
        return True

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all jobs (snapshots)."""
        return [j.snapshot() for j in self._jobs.values()]

    def cleanup_expired(self) -> int:
        """Remove completed/failed/cancelled jobs older than TTL.

        Returns the number of jobs removed.
        """
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if j.status in ("complete", "failed", "cancelled")
            and j.finished_at > 0
            and (now - j.finished_at) > _JOB_TTL
        ]
        for jid in expired:
            del self._jobs[jid]
        if expired:
            logger.info("cleaned_up=<%d> | expired jobs removed", len(expired))
        return len(expired)


# Module-level singleton
job_store = JobStore()
