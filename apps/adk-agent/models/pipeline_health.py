# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Unified pipeline health model — systemic architectural guardrail.

Every pipeline phase reports structured health data to a cumulative
``PipelineHealth`` object in session state.  A single ``HealthGate``
validates phase contracts at each boundary and can abort, degrade, or
continue the pipeline.

Architecture::

    PipelineHealth (lives in state["_pipeline_health"])
    ├── phases: list[PhaseReport]   ← each phase appends one
    ├── verdict: OK / DEGRADED / FAILED
    ├── errors: list[str]           ← cumulative error messages
    └── warnings: list[str]         ← cumulative warnings

    Health Gate (called at each phase boundary):
    1. Algorithmic checks FIRST (fast, deterministic, cheap)
       → Hard pass/fail based on PhaseContract
    2. IF algorithmic checks pass but health is ambiguous:
       → Agentic quality check (LLM) that can only DOWNGRADE, never upgrade
    3. Update cumulative verdict
    4. Return routing decision: CONTINUE / ABORT

Key design principle: the agentic layer NEVER overrides a deterministic
failure.  If the numbers say CRITICAL, it's CRITICAL regardless of what
the LLM thinks.  The LLM can only downgrade from OK to DEGRADED.

This replaces 6 scattered guardrails with one unified health model:
- Circuit breaker → search executor phase report
- Query validator → search executor phase report
- Heartbeats → phase progress tracking
- Mandatory scoring gate → maestro phase contract
- Diagnostic tool → reads PipelineHealth directly
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Health severity level for a single check."""
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class Verdict(str, Enum):
    """Cumulative pipeline health verdict."""
    HEALTHY = "HEALTHY"      # all phases OK
    DEGRADED = "DEGRADED"    # some warnings but no critical failures
    FAILED = "FAILED"        # at least one critical failure


class RoutingDecision(str, Enum):
    """What the health gate tells the pipeline to do."""
    CONTINUE = "CONTINUE"    # proceed normally
    ABORT = "ABORT"          # stop the pipeline, report honestly


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HealthCheck:
    """A single health check result within a phase."""
    name: str
    severity: Severity
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseReport:
    """Structured health report from one pipeline phase."""
    name: str                                    # e.g. "search_executor_iter1"
    status: Severity = Severity.OK
    checks: list[HealthCheck] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    duration_s: float = 0.0

    def add_check(self, name: str, severity: Severity, message: str,
                  **evidence: Any) -> None:
        """Add a health check result to this phase report."""
        self.checks.append(HealthCheck(
            name=name, severity=severity, message=message, evidence=evidence,
        ))
        # Phase status is the worst severity of any check
        if severity.value == "CRITICAL":
            self.status = Severity.CRITICAL
        elif severity.value == "WARNING" and self.status != Severity.CRITICAL:
            self.status = Severity.WARNING

    def complete(self) -> None:
        """Mark this phase as complete and compute duration."""
        self.completed_at = time.time()
        self.duration_s = self.completed_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Serialise for session state storage."""
        return {
            "name": self.name,
            "status": self.status.value,
            "checks": [
                {
                    "name": c.name,
                    "severity": c.severity.value,
                    "message": c.message,
                    "evidence": c.evidence,
                }
                for c in self.checks
            ],
            "metrics": self.metrics,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class PhaseContract:
    """Declarative contract for what a phase promises to deliver.

    Each check_fn receives the PhaseReport and the session state,
    and returns a HealthCheck.  The contract is validated by the
    health gate after the phase completes.
    """
    phase_name: str
    checks: list  # list of callables: (PhaseReport, state) -> HealthCheck


# ---------------------------------------------------------------------------
# PipelineHealth — the cumulative health object
# ---------------------------------------------------------------------------

class PipelineHealth:
    """Cumulative pipeline health tracker.

    Lives in ``state["_pipeline_health"]`` as a serialised dict.
    Use ``PipelineHealth.from_state(state)`` to hydrate and
    ``self.save(state)`` to persist.
    """

    def __init__(self) -> None:
        self.verdict: Verdict = Verdict.HEALTHY
        self.phases: list[PhaseReport] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.started_at: float = time.time()
        self.aborted: bool = False
        self.abort_reason: str = ""

    # ── Phase lifecycle ──────────────────────────────────────────────

    def begin_phase(self, name: str) -> PhaseReport:
        """Start a new phase and return its report object."""
        report = PhaseReport(name=name)
        self.phases.append(report)
        return report

    def complete_phase(self, report: PhaseReport) -> None:
        """Mark a phase as complete and update cumulative verdict."""
        report.complete()
        self._update_verdict(report)

    # ── Verdict computation ──────────────────────────────────────────

    def _update_verdict(self, report: PhaseReport) -> None:
        """Update cumulative verdict based on a completed phase."""
        for check in report.checks:
            if check.severity == Severity.CRITICAL:
                self.errors.append(f"[{report.name}] {check.message}")
            elif check.severity == Severity.WARNING:
                self.warnings.append(f"[{report.name}] {check.message}")

        if report.status == Severity.CRITICAL:
            self.verdict = Verdict.FAILED
        elif report.status == Severity.WARNING and self.verdict == Verdict.HEALTHY:
            self.verdict = Verdict.DEGRADED

    # ── Health gate ──────────────────────────────────────────────────

    def evaluate_gate(self, report: PhaseReport) -> RoutingDecision:
        """Evaluate the health gate after a phase completes.

        Algorithmic-only: returns ABORT if cumulative verdict is FAILED,
        CONTINUE otherwise.  The agentic layer (if enabled) can only
        DOWNGRADE from CONTINUE to ABORT, never upgrade.
        """
        self.complete_phase(report)

        if self.verdict == Verdict.FAILED:
            logger.error(
                "HEALTH GATE: pipeline verdict FAILED after phase '%s' — "
                "critical errors: %s",
                report.name,
                "; ".join(self.errors[-3:]),  # last 3 errors
            )
            return RoutingDecision.ABORT

        if self.verdict == Verdict.DEGRADED:
            logger.warning(
                "HEALTH GATE: pipeline DEGRADED after phase '%s' — "
                "warnings: %s",
                report.name,
                "; ".join(self.warnings[-3:]),
            )

        return RoutingDecision.CONTINUE

    def abort(self, reason: str) -> None:
        """Mark the pipeline as aborted with a reason."""
        self.aborted = True
        self.abort_reason = reason
        self.verdict = Verdict.FAILED
        self.errors.append(f"ABORTED: {reason}")
        logger.error("PIPELINE ABORTED: %s", reason)

    # ── Agentic quality layer ────────────────────────────────────────

    def agentic_downgrade(self, phase_name: str, reason: str) -> None:
        """Allow the agentic layer to downgrade health (never upgrade).

        Can only move: HEALTHY → DEGRADED, or DEGRADED → FAILED.
        Cannot move: FAILED → anything, or DEGRADED → HEALTHY.
        """
        if self.verdict == Verdict.HEALTHY:
            self.verdict = Verdict.DEGRADED
            self.warnings.append(f"[agentic:{phase_name}] {reason}")
            logger.warning(
                "AGENTIC DOWNGRADE: %s → DEGRADED: %s",
                phase_name, reason,
            )
        elif self.verdict == Verdict.DEGRADED:
            self.verdict = Verdict.FAILED
            self.errors.append(f"[agentic:{phase_name}] {reason}")
            logger.error(
                "AGENTIC DOWNGRADE: %s → FAILED: %s",
                phase_name, reason,
            )

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full health state for session state storage."""
        # Include both historical phases (from previous hydrations)
        # and new phases added since last hydration.
        historical = getattr(self, '_raw_phases', [])
        current = [p.to_dict() for p in self.phases]
        return {
            "verdict": self.verdict.value,
            "phases": historical + current,
            "errors": self.errors,
            "warnings": self.warnings,
            "elapsed_s": round(time.time() - self.started_at, 2),
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }

    def save(self, state: dict) -> None:
        """Persist to session state."""
        state["_pipeline_health"] = self.to_dict()

    @classmethod
    def from_state(cls, state: dict) -> PipelineHealth:
        """Hydrate from session state, or create a fresh instance."""
        raw = state.get("_pipeline_health")
        health = cls()
        if raw and isinstance(raw, dict):
            health.verdict = Verdict(raw.get("verdict", "HEALTHY"))
            health.errors = raw.get("errors", [])
            health.warnings = raw.get("warnings", [])
            health.aborted = raw.get("aborted", False)
            health.abort_reason = raw.get("abort_reason", "")
            # Phases are already serialised — we don't rehydrate them
            # fully (they're historical), but we keep the raw list for
            # the diagnostic tool.
            health._raw_phases = raw.get("phases", [])
        return health

    # ── Convenience ──────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line health summary for logging."""
        n_critical = sum(1 for e in self.errors if not e.startswith("ABORTED"))
        n_warning = len(self.warnings)
        return (
            f"Pipeline health: {self.verdict.value} "
            f"({len(self.phases)} phases, "
            f"{n_critical} errors, {n_warning} warnings)"
        )


# ---------------------------------------------------------------------------
# Pre-built phase contracts (algorithmic health checks)
# ---------------------------------------------------------------------------

def check_search_executor(report: PhaseReport, state: dict) -> None:
    """Algorithmic health checks for the search executor phase.

    Checks:
    1. Did it ingest any findings?
    2. Were too many queries rejected as noise?
    3. Were any circuit breakers tripped?
    4. Did it time out?
    """
    metrics = report.metrics

    # Check 1: findings ingested
    ingested = metrics.get("total_ingested", 0)
    if ingested == 0:
        report.add_check(
            "findings_ingested", Severity.CRITICAL,
            "Search executor produced 0 findings — no data for pipeline to work with",
            ingested=0,
        )
    elif ingested < 3:
        report.add_check(
            "findings_ingested", Severity.WARNING,
            f"Search executor produced only {ingested} findings — thin corpus",
            ingested=ingested,
        )
    else:
        report.add_check(
            "findings_ingested", Severity.OK,
            f"Search executor ingested {ingested} findings",
            ingested=ingested,
        )

    # Check 2: noise rejection
    # Compare against total_raw_queries (pre-filtering count) so the
    # populations match.  Fallback: rejected + strategy_searches.
    rejected = metrics.get("queries_rejected_noise", 0)
    total_raw = metrics.get("total_raw_queries", 0)
    if total_raw == 0:
        # Fallback when total_raw_queries wasn't tracked
        total_raw = rejected + metrics.get("strategy_searches", 0)
    if rejected > 0 and metrics.get("strategy_searches", 0) == 0:
        report.add_check(
            "query_noise", Severity.CRITICAL,
            f"All {rejected} queries rejected as noise — thinker strategy "
            "has zero topical overlap with user question",
            rejected=rejected, total=total_raw,
        )
    elif total_raw > 0 and rejected > total_raw * 0.5:
        report.add_check(
            "query_noise", Severity.WARNING,
            f"{rejected}/{total_raw} queries rejected as noise",
            rejected=rejected, total=total_raw,
        )

    # Check 3: circuit breakers
    tripped = metrics.get("circuit_breakers_tripped", 0)
    if tripped >= 3:
        report.add_check(
            "circuit_breakers", Severity.CRITICAL,
            f"{tripped} API circuit breakers tripped — most search APIs are down",
            tripped=tripped,
        )
    elif tripped > 0:
        report.add_check(
            "circuit_breakers", Severity.WARNING,
            f"{tripped} API circuit breaker(s) tripped",
            tripped=tripped,
        )

    # Check 4: timeout
    if metrics.get("timed_out", False):
        report.add_check(
            "timeout", Severity.WARNING,
            "Search executor timed out — results may be incomplete",
        )


def check_maestro(report: PhaseReport, state: dict) -> None:
    """Algorithmic health checks for the maestro phase.

    Checks:
    1. Are findings scored (score_version > 0)?
    2. Did the maestro produce any output?
    """
    metrics = report.metrics

    # Check 1: unscored findings
    unscored = metrics.get("unscored_findings", 0)
    total = metrics.get("total_findings", 0)
    if total > 0 and unscored == total:
        report.add_check(
            "all_unscored", Severity.CRITICAL,
            f"All {total} findings are unscored (score_version=0) — "
            "Flock scoring never ran",
            unscored=unscored, total=total,
        )
    elif total > 0 and unscored > total * 0.5:
        report.add_check(
            "mostly_unscored", Severity.WARNING,
            f"{unscored}/{total} findings still unscored",
            unscored=unscored, total=total,
        )

    # Check 2: maestro output
    if not metrics.get("maestro_produced_output", True):
        report.add_check(
            "no_output", Severity.WARNING,
            "Maestro produced no output — may have had no work to do",
        )


def check_thinker(report: PhaseReport, state: dict) -> None:
    """Algorithmic health checks for the thinker phase.

    Checks:
    1. Did the thinker produce a strategy?
    2. Are extractable queries present?
    """
    metrics = report.metrics

    strategy_len = metrics.get("strategy_length", 0)
    if strategy_len == 0:
        report.add_check(
            "no_strategy", Severity.CRITICAL,
            "Thinker produced no research strategy",
        )
    elif strategy_len < 50:
        report.add_check(
            "short_strategy", Severity.WARNING,
            f"Thinker strategy is very short ({strategy_len} chars)",
            strategy_length=strategy_len,
        )

    queries = metrics.get("extractable_queries", 0)
    if strategy_len > 0 and queries == 0:
        report.add_check(
            "no_queries", Severity.WARNING,
            "Thinker strategy has no extractable search queries",
            strategy_length=strategy_len,
        )


def check_synthesiser(report: PhaseReport, state: dict) -> None:
    """Algorithmic health checks for the synthesiser phase.

    Checks:
    1. Did the synthesiser produce a non-empty report?
    2. Is the corpus it received non-trivial?
    """
    metrics = report.metrics

    report_len = metrics.get("report_length", 0)
    if report_len == 0:
        report.add_check(
            "no_report", Severity.CRITICAL,
            "Synthesiser produced no output — pipeline has no deliverable",
        )
    elif report_len < 200:
        report.add_check(
            "short_report", Severity.WARNING,
            f"Synthesiser report is very short ({report_len} chars)",
            report_length=report_len,
        )

    corpus_findings = metrics.get("corpus_findings", 0)
    if corpus_findings == 0:
        report.add_check(
            "empty_corpus", Severity.CRITICAL,
            "Synthesiser received an empty corpus — nothing to synthesise",
        )


def check_scout(report: PhaseReport, state: dict) -> None:
    """Algorithmic health checks for the scout phase.

    Checks:
    1. Did the scout produce sub-queries?
    2. Did it find any initial landscape?
    """
    metrics = report.metrics

    sub_queries = metrics.get("sub_queries", 0)
    if sub_queries == 0:
        report.add_check(
            "no_sub_queries", Severity.WARNING,
            "Scout produced no sub-queries — research may lack decomposition",
        )

    initial_findings = metrics.get("initial_findings", 0)
    if initial_findings == 0:
        report.add_check(
            "no_initial_findings", Severity.WARNING,
            "Scout found no initial landscape — topic may be very niche",
        )
