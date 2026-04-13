"""MiroThinker Pipeline Diagnostic Tool — Architectural Guardrail Edition.

Standalone diagnostic that analyses pipeline logs and DuckDB corpus state.
Unlike the original BAT.AI-pattern diagnostic, this version uses
**deterministic health checks first** — not LLM vibes — to establish
severity.  The LLM only runs AFTER deterministic checks have flagged
the issues.

Architecture:
    1. Deterministic health checks (SQL queries + log pattern matching)
       → each check returns a severity (CRITICAL / WARNING / OK)
    2. Aggregate severity → overall verdict (HEALTHY / DEGRADED / FAILED)
    3. LLM analysis ONLY if there are findings to explain
    4. Structured report with transparent failure evidence

The key insight: the original diagnostic reported "2 minor things" for a
run where academic search was 100% broken, 60% of findings were noise,
Flock scoring never ran, and the synthesiser never ran.  That happened
because it delegated all judgment to a permissive LLM.  This version
makes the *detection* deterministic and only uses the LLM for
*explanation* of detected issues.

Usage (standalone):
    python -m tools.pipeline_diagnostic \\
        --log /tmp/mirothinker_pipeline.log \\
        --db  /path/to/corpus.duckdb \\
        --question "Why did the pipeline stop after iteration 1?"

    # Health-check mode (no question):
    python -m tools.pipeline_diagnostic \\
        --log /tmp/mirothinker_pipeline.log \\
        --db  /path/to/corpus.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import duckdb
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity model
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Diagnostic check severity levels."""
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = {Severity.OK: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        return order[self] > order[other]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        order = {Severity.OK: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        return order[self] >= order[other]


class Verdict(str, Enum):
    """Overall pipeline health verdict."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass
class CheckResult:
    """Result of a single deterministic health check."""
    name: str
    severity: Severity
    message: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    verdict: Verdict
    checks: list[CheckResult]
    corpus_summary: dict[str, Any]
    log_summary: dict[str, Any]
    llm_analysis: str = ""
    elapsed_s: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == Severity.WARNING)


# ---------------------------------------------------------------------------
# Log parser
# ---------------------------------------------------------------------------

def parse_log_file(log_path: str) -> dict[str, Any]:
    """Parse a pipeline log file into structured summary.

    Extracts:
      - Error/warning counts and messages
      - Phase completion events
      - Search executor stats
      - Circuit breaker trips
      - Stall/timeout events
      - Heartbeat events
    """
    summary: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "phases_completed": [],
        "search_executor_stats": {},
        "circuit_breakers_tripped": [],
        "stall_events": [],
        "heartbeats": [],
        "scoring_events": [],
        "total_lines": 0,
        "error_count": 0,
        "warning_count": 0,
    }

    if not log_path or not Path(log_path).exists():
        return summary

    try:
        text = Path(log_path).read_text(errors="replace")
    except Exception:
        return summary

    lines = text.splitlines()
    summary["total_lines"] = len(lines)

    for line in lines:
        lower = line.lower()

        # Count errors and warnings
        if " error " in lower or "error:" in lower or "failed" in lower:
            summary["error_count"] += 1
            if len(summary["errors"]) < 50:
                summary["errors"].append(line.strip()[:300])

        if " warning " in lower or "warning:" in lower:
            summary["warning_count"] += 1
            if len(summary["warnings"]) < 30:
                summary["warnings"].append(line.strip()[:300])

        # Phase completion
        if "phase" in lower and "complete" in lower:
            summary["phases_completed"].append(line.strip()[:200])

        # Circuit breaker trips
        if "circuit breaker tripped" in lower:
            summary["circuit_breakers_tripped"].append(line.strip()[:200])

        # Stall/timeout events
        if "timed out" in lower or "stall" in lower or "timeout" in lower:
            summary["stall_events"].append(line.strip()[:200])

        # Heartbeats
        if "heartbeat" in lower:
            summary["heartbeats"].append(line.strip()[:200])

        # Scoring
        if "scoring" in lower or "scored" in lower:
            summary["scoring_events"].append(line.strip()[:200])

        # Search executor stats line
        if "search executor complete" in lower or "search executor stats" in lower:
            summary["search_executor_stats"]["raw"] = line.strip()[:300]

        # Query rejection
        if "query rejected" in lower or "queries rejected" in lower:
            if "query_rejections" not in summary:
                summary["query_rejections"] = []
            summary["query_rejections"].append(line.strip()[:200])

        # Academic search failures
        if ("semantic scholar" in lower or "arxiv" in lower or "scite" in lower) and (
            "failed" in lower or "error" in lower
        ):
            if "academic_failures" not in summary:
                summary["academic_failures"] = []
            summary["academic_failures"].append(line.strip()[:200])

        # Mandatory scoring gate
        if "mandatory scoring gate" in lower:
            if "mandatory_scoring" not in summary:
                summary["mandatory_scoring"] = []
            summary["mandatory_scoring"].append(line.strip()[:200])

    return summary


# ---------------------------------------------------------------------------
# Corpus analyser
# ---------------------------------------------------------------------------

def analyse_corpus(db_path: str) -> dict[str, Any]:
    """Analyse a DuckDB corpus file and return structured summary.

    All checks are deterministic SQL queries — no LLM involved.
    """
    summary: dict[str, Any] = {
        "exists": False,
        "total_rows": 0,
        "total_findings": 0,
        "total_thoughts": 0,
        "total_serendipity": 0,
        "quality_distribution": {"strong": 0, "moderate": 0, "weak": 0},
        "unscored_count": 0,
        "unscored_pct": 0.0,
        "processing_status": {},
        "source_types": {},
        "avg_composite_quality": 0.0,
        "contradiction_count": 0,
        "iterations_seen": 0,
        "consider_for_use_count": 0,
        "excluded_count": 0,
        "columns": [],
    }

    if not db_path or not Path(db_path).exists():
        return summary

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception as exc:
        logger.warning("Cannot open corpus DB %s: %s", db_path, exc)
        return summary

    summary["exists"] = True

    try:
        # Get columns
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'conditions'"
        ).fetchall()
        summary["columns"] = [c[0] for c in cols]

        # Total rows by type
        rows = conn.execute(
            "SELECT row_type, COUNT(*) FROM conditions GROUP BY row_type"
        ).fetchall()
        for row_type, count in rows:
            if row_type == "finding":
                summary["total_findings"] = count
            elif row_type == "thought":
                summary["total_thoughts"] = count
            elif row_type == "serendipity":
                summary["total_serendipity"] = count
            summary["total_rows"] += count

        # Quality distribution (findings only)
        if summary["total_findings"] > 0:
            qd = conn.execute(
                "SELECT "
                "  SUM(CASE WHEN CAST(composite_quality AS FLOAT) >= 0.6 "
                "    THEN 1 ELSE 0 END) as strong, "
                "  SUM(CASE WHEN CAST(composite_quality AS FLOAT) >= 0.3 "
                "    AND CAST(composite_quality AS FLOAT) < 0.6 "
                "    THEN 1 ELSE 0 END) as moderate, "
                "  SUM(CASE WHEN CAST(composite_quality AS FLOAT) < 0.3 "
                "    THEN 1 ELSE 0 END) as weak "
                "FROM conditions WHERE row_type = 'finding'"
            ).fetchone()
            if qd:
                summary["quality_distribution"] = {
                    "strong": qd[0] or 0,
                    "moderate": qd[1] or 0,
                    "weak": qd[2] or 0,
                }

        # Unscored findings (score_version = 0 or composite_quality < 0)
        if "score_version" in summary["columns"]:
            unscored = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE row_type = 'finding' AND score_version = 0"
            ).fetchone()[0]
        else:
            unscored = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE row_type = 'finding' "
                "AND CAST(composite_quality AS FLOAT) < 0"
            ).fetchone()[0]
        summary["unscored_count"] = unscored
        if summary["total_findings"] > 0:
            summary["unscored_pct"] = (
                unscored / summary["total_findings"] * 100
            )

        # Processing status distribution
        if "processing_status" in summary["columns"]:
            ps_rows = conn.execute(
                "SELECT processing_status, COUNT(*) "
                "FROM conditions WHERE row_type = 'finding' "
                "GROUP BY processing_status"
            ).fetchall()
            summary["processing_status"] = {
                row[0]: row[1] for row in ps_rows
            }

        # Source type distribution
        if "source_type" in summary["columns"]:
            st_rows = conn.execute(
                "SELECT source_type, COUNT(*) "
                "FROM conditions GROUP BY source_type"
            ).fetchall()
            summary["source_types"] = {
                row[0]: row[1] for row in st_rows
            }

        # Average composite quality
        avg = conn.execute(
            "SELECT AVG(CAST(composite_quality AS FLOAT)) "
            "FROM conditions WHERE row_type = 'finding' "
            "AND CAST(composite_quality AS FLOAT) >= 0"
        ).fetchone()[0]
        summary["avg_composite_quality"] = round(avg, 3) if avg else 0.0

        # Contradictions
        if "contradiction_flag" in summary["columns"]:
            summary["contradiction_count"] = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE contradiction_flag = TRUE"
            ).fetchone()[0]

        # Iterations
        if "iteration" in summary["columns"]:
            max_iter = conn.execute(
                "SELECT MAX(iteration) FROM conditions"
            ).fetchone()[0]
            summary["iterations_seen"] = (max_iter or 0) + 1

        # Consider for use
        if "consider_for_use" in summary["columns"]:
            summary["consider_for_use_count"] = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE consider_for_use = TRUE"
            ).fetchone()[0]
            summary["excluded_count"] = conn.execute(
                "SELECT COUNT(*) FROM conditions "
                "WHERE consider_for_use = FALSE"
            ).fetchone()[0]

    except Exception as exc:
        logger.warning("Error analysing corpus: %s", exc, exc_info=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return summary


# ---------------------------------------------------------------------------
# Deterministic health checks
# ---------------------------------------------------------------------------

def check_unscored_findings(corpus: dict[str, Any]) -> CheckResult:
    """CRITICAL if >50% of findings are unscored (score_version=0)."""
    unscored = corpus.get("unscored_count", 0)
    total = corpus.get("total_findings", 0)
    pct = corpus.get("unscored_pct", 0.0)

    if total == 0:
        return CheckResult(
            name="unscored_findings",
            severity=Severity.CRITICAL,
            message="No findings in corpus at all",
            evidence=["total_findings=0"],
        )

    if pct >= 50:
        return CheckResult(
            name="unscored_findings",
            severity=Severity.CRITICAL,
            message=(
                f"{unscored}/{total} findings ({pct:.0f}%) are unscored "
                f"— Flock scoring never ran or failed"
            ),
            evidence=[
                f"unscored_count={unscored}",
                f"total_findings={total}",
                f"unscored_pct={pct:.1f}%",
            ],
        )

    if pct >= 10:
        return CheckResult(
            name="unscored_findings",
            severity=Severity.WARNING,
            message=f"{unscored}/{total} findings ({pct:.0f}%) are unscored",
            evidence=[f"unscored_pct={pct:.1f}%"],
        )

    return CheckResult(
        name="unscored_findings",
        severity=Severity.OK,
        message=f"All findings scored ({total} total, {unscored} unscored)",
    )


def check_processing_status(corpus: dict[str, Any]) -> CheckResult:
    """CRITICAL if all findings are stuck at 'raw' processing status."""
    status_dist = corpus.get("processing_status", {})
    total = corpus.get("total_findings", 0)

    if not status_dist or total == 0:
        return CheckResult(
            name="processing_status",
            severity=Severity.OK,
            message="No processing status data available",
        )

    raw_count = status_dist.get("raw", 0)
    if raw_count == total and total > 0:
        return CheckResult(
            name="processing_status",
            severity=Severity.CRITICAL,
            message=(
                f"ALL {total} findings stuck at processing_status='raw' "
                f"— algorithm battery never ran"
            ),
            evidence=[
                f"processing_status distribution: {status_dist}",
            ],
        )

    raw_pct = raw_count / total * 100 if total > 0 else 0
    if raw_pct >= 50:
        return CheckResult(
            name="processing_status",
            severity=Severity.WARNING,
            message=f"{raw_count}/{total} ({raw_pct:.0f}%) still at 'raw'",
            evidence=[f"processing_status: {status_dist}"],
        )

    return CheckResult(
        name="processing_status",
        severity=Severity.OK,
        message=f"Processing status healthy: {status_dist}",
    )


def check_quality_distribution(corpus: dict[str, Any]) -> CheckResult:
    """CRITICAL if >60% of findings are weak quality (noise)."""
    qd = corpus.get("quality_distribution", {})
    strong = qd.get("strong", 0)
    moderate = qd.get("moderate", 0)
    weak = qd.get("weak", 0)
    total = strong + moderate + weak

    if total == 0:
        return CheckResult(
            name="quality_distribution",
            severity=Severity.WARNING,
            message="No quality distribution data",
        )

    weak_pct = weak / total * 100

    if weak_pct >= 60:
        return CheckResult(
            name="quality_distribution",
            severity=Severity.CRITICAL,
            message=(
                f"{weak}/{total} findings ({weak_pct:.0f}%) are weak quality "
                f"— likely noise from off-topic queries"
            ),
            evidence=[
                f"strong={strong}, moderate={moderate}, weak={weak}",
                f"weak_pct={weak_pct:.1f}%",
            ],
        )

    if weak_pct >= 40:
        return CheckResult(
            name="quality_distribution",
            severity=Severity.WARNING,
            message=f"{weak_pct:.0f}% weak findings (borderline noise)",
            evidence=[f"strong={strong}, moderate={moderate}, weak={weak}"],
        )

    return CheckResult(
        name="quality_distribution",
        severity=Severity.OK,
        message=(
            f"Quality healthy: {strong} strong, {moderate} moderate, "
            f"{weak} weak"
        ),
    )


def check_academic_search(log: dict[str, Any]) -> CheckResult:
    """CRITICAL if academic search was attempted but returned 0 results."""
    academic_failures = log.get("academic_failures", [])
    circuit_trips = log.get("circuit_breakers_tripped", [])

    # Check if Phase C ran and got zero results
    phase_c_zero = any(
        "zero academic results" in line.lower()
        for line in log.get("errors", [])
    )

    if phase_c_zero:
        return CheckResult(
            name="academic_search",
            severity=Severity.CRITICAL,
            message=(
                "Academic search (Phase C) returned ZERO results — "
                "all academic APIs failed"
            ),
            evidence=academic_failures[:5] + circuit_trips[:3],
        )

    if len(academic_failures) >= 3:
        return CheckResult(
            name="academic_search",
            severity=Severity.WARNING,
            message=f"{len(academic_failures)} academic search failures",
            evidence=academic_failures[:5],
        )

    if circuit_trips:
        return CheckResult(
            name="academic_search",
            severity=Severity.WARNING,
            message=f"Circuit breakers tripped: {len(circuit_trips)}",
            evidence=circuit_trips[:3],
        )

    return CheckResult(
        name="academic_search",
        severity=Severity.OK,
        message="Academic search healthy (no failures detected)",
    )


def check_pipeline_stall(log: dict[str, Any]) -> CheckResult:
    """CRITICAL if the pipeline stalled/timed out."""
    stall_events = log.get("stall_events", [])

    if any("timed out" in e.lower() for e in stall_events):
        return CheckResult(
            name="pipeline_stall",
            severity=Severity.CRITICAL,
            message="Pipeline timed out — search executor exceeded deadline",
            evidence=stall_events[:5],
        )

    if stall_events:
        return CheckResult(
            name="pipeline_stall",
            severity=Severity.WARNING,
            message=f"{len(stall_events)} stall-related events",
            evidence=stall_events[:5],
        )

    return CheckResult(
        name="pipeline_stall",
        severity=Severity.OK,
        message="No stall or timeout events detected",
    )


def check_synthesiser_ran(log: dict[str, Any], corpus: dict[str, Any]) -> CheckResult:
    """CRITICAL if the synthesiser never ran (pipeline died before it)."""
    phases = log.get("phases_completed", [])
    # Check if swarm synthesis or synthesiser appears in log
    synth_ran = any(
        "synth" in p.lower() or "swarm" in p.lower()
        for p in phases
    )

    # Also check log errors for evidence
    log_lines = log.get("errors", []) + log.get("warnings", [])
    synth_in_log = any(
        "synthesiser" in line.lower() or "swarm synthesis" in line.lower()
        for line in log_lines
    )

    iterations = corpus.get("iterations_seen", 0)

    if not synth_ran and not synth_in_log and iterations > 0:
        return CheckResult(
            name="synthesiser_ran",
            severity=Severity.CRITICAL,
            message=(
                "Synthesiser never ran — pipeline stalled before "
                "reaching the synthesis phase"
            ),
            evidence=[
                f"iterations_seen={iterations}",
                f"phases_completed: {[p[:60] for p in phases]}",
            ],
        )

    return CheckResult(
        name="synthesiser_ran",
        severity=Severity.OK,
        message="Synthesiser ran successfully",
    )


def check_serendipity(corpus: dict[str, Any]) -> CheckResult:
    """WARNING if serendipity/contrarian queries produced no results."""
    serendipity = corpus.get("total_serendipity", 0)
    total = corpus.get("total_findings", 0)

    if total > 10 and serendipity == 0:
        return CheckResult(
            name="serendipity",
            severity=Severity.WARNING,
            message=(
                "Zero serendipity findings despite having "
                f"{total} findings — contrarian search may not be working"
            ),
            evidence=[f"total_serendipity=0, total_findings={total}"],
        )

    return CheckResult(
        name="serendipity",
        severity=Severity.OK,
        message=f"Serendipity: {serendipity} rows",
    )


def check_query_noise(log: dict[str, Any]) -> CheckResult:
    """WARNING/CRITICAL if many queries were rejected as noise."""
    rejections = log.get("query_rejections", [])

    if len(rejections) >= 5:
        return CheckResult(
            name="query_noise",
            severity=Severity.CRITICAL,
            message=(
                f"{len(rejections)} queries rejected as off-topic noise "
                f"— thinker is producing generic queries"
            ),
            evidence=rejections[:5],
        )

    if rejections:
        return CheckResult(
            name="query_noise",
            severity=Severity.WARNING,
            message=f"{len(rejections)} queries rejected as noise",
            evidence=rejections[:3],
        )

    return CheckResult(
        name="query_noise",
        severity=Severity.OK,
        message="No query noise detected",
    )


def check_error_rate(log: dict[str, Any]) -> CheckResult:
    """CRITICAL if error rate is unusually high."""
    error_count = log.get("error_count", 0)
    total_lines = log.get("total_lines", 1)
    error_rate = error_count / max(total_lines, 1) * 100

    if error_count >= 20 or error_rate >= 5:
        return CheckResult(
            name="error_rate",
            severity=Severity.CRITICAL,
            message=(
                f"{error_count} errors in {total_lines} log lines "
                f"({error_rate:.1f}% error rate)"
            ),
            evidence=log.get("errors", [])[:10],
        )

    if error_count >= 5:
        return CheckResult(
            name="error_rate",
            severity=Severity.WARNING,
            message=f"{error_count} errors in log",
            evidence=log.get("errors", [])[:5],
        )

    return CheckResult(
        name="error_rate",
        severity=Severity.OK,
        message=f"{error_count} errors — within normal range",
    )


def check_mandatory_scoring(log: dict[str, Any]) -> CheckResult:
    """WARNING if the mandatory scoring gate had to fire."""
    mandatory = log.get("mandatory_scoring", [])

    if mandatory:
        return CheckResult(
            name="mandatory_scoring_gate",
            severity=Severity.WARNING,
            message=(
                "Mandatory scoring gate fired during cleanup — "
                "findings were left unscored by the normal pipeline path"
            ),
            evidence=mandatory[:3],
        )

    return CheckResult(
        name="mandatory_scoring_gate",
        severity=Severity.OK,
        message="Mandatory scoring gate did not need to fire",
    )


# ---------------------------------------------------------------------------
# Aggregate diagnosis
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_unscored_findings,
    check_processing_status,
    check_quality_distribution,
    check_academic_search,
    check_pipeline_stall,
    check_query_noise,
    check_error_rate,
    check_mandatory_scoring,
]

# These checks need both log and corpus
DUAL_CHECKS = [
    check_synthesiser_ran,
]

# These only need corpus
CORPUS_ONLY_CHECKS = [
    check_serendipity,
]


def run_deterministic_checks(
    log: dict[str, Any],
    corpus: dict[str, Any],
) -> list[CheckResult]:
    """Run all deterministic health checks."""
    results: list[CheckResult] = []

    for check_fn in ALL_CHECKS:
        # Some checks need log, some need corpus — try both
        try:
            sig = check_fn.__code__.co_varnames[:check_fn.__code__.co_argcount]
            if "log" in sig and "corpus" in sig:
                results.append(check_fn(log, corpus))
            elif "log" in sig:
                results.append(check_fn(log))
            elif "corpus" in sig:
                results.append(check_fn(corpus))
        except Exception as exc:
            logger.warning("Health check %s failed: %s", check_fn.__name__, exc)

    for check_fn in DUAL_CHECKS:
        try:
            results.append(check_fn(log, corpus))
        except Exception as exc:
            logger.warning("Health check %s failed: %s", check_fn.__name__, exc)

    for check_fn in CORPUS_ONLY_CHECKS:
        try:
            results.append(check_fn(corpus))
        except Exception as exc:
            logger.warning("Health check %s failed: %s", check_fn.__name__, exc)

    return results


def compute_verdict(checks: list[CheckResult]) -> Verdict:
    """Compute overall verdict from check results.

    - FAILED: any CRITICAL check
    - DEGRADED: any WARNING check
    - HEALTHY: all OK
    """
    if any(c.severity == Severity.CRITICAL for c in checks):
        return Verdict.FAILED
    if any(c.severity == Severity.WARNING for c in checks):
        return Verdict.DEGRADED
    return Verdict.HEALTHY


# ---------------------------------------------------------------------------
# LLM analysis (optional, runs AFTER deterministic checks)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Call the LLM for analysis — only used for explaining detected issues."""
    try:
        import httpx
    except ImportError:
        return "(httpx not available — skipping LLM analysis)"

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "(no OPENROUTER_API_KEY — skipping LLM analysis)"

    model = os.environ.get("DIAGNOSTIC_MODEL", "openai/gpt-4o-mini")
    base = os.environ.get(
        "OPENAI_API_BASE",
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

    try:
        resp = httpx.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.2,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return f"(LLM analysis failed: {exc})"


def generate_llm_analysis(
    checks: list[CheckResult],
    corpus: dict[str, Any],
    log: dict[str, Any],
    question: str = "",
) -> str:
    """Generate LLM explanation of detected issues.

    Only called when deterministic checks found CRITICAL or WARNING issues.
    The LLM doesn't decide severity — it explains what the deterministic
    checks already found.
    """
    critical = [c for c in checks if c.severity == Severity.CRITICAL]
    warnings = [c for c in checks if c.severity == Severity.WARNING]

    if not critical and not warnings:
        return "All deterministic health checks passed — no issues to explain."

    findings_section = ""
    for c in critical:
        findings_section += (
            f"\n[CRITICAL] {c.name}: {c.message}\n"
            f"  Evidence: {'; '.join(c.evidence[:3])}\n"
        )
    for c in warnings:
        findings_section += (
            f"\n[WARNING] {c.name}: {c.message}\n"
            f"  Evidence: {'; '.join(c.evidence[:3])}\n"
        )

    corpus_section = (
        f"Corpus: {corpus.get('total_findings', 0)} findings, "
        f"{corpus.get('total_thoughts', 0)} thoughts, "
        f"{corpus.get('unscored_count', 0)} unscored, "
        f"avg quality={corpus.get('avg_composite_quality', 0):.2f}, "
        f"contradictions={corpus.get('contradiction_count', 0)}, "
        f"iterations={corpus.get('iterations_seen', 0)}"
    )

    log_section = (
        f"Log: {log.get('total_lines', 0)} lines, "
        f"{log.get('error_count', 0)} errors, "
        f"{log.get('warning_count', 0)} warnings"
    )

    question_section = ""
    if question:
        question_section = f"\n\nUser question: {question}"

    prompt = textwrap.dedent(f"""\
        You are a pipeline diagnostic analyst. The deterministic health
        checks have already identified the issues below.  Your job is to
        EXPLAIN the root causes and suggest fixes.  Do NOT downgrade
        severity — the checks are authoritative.

        == DETECTED ISSUES ==
        {findings_section}

        == CORPUS STATE ==
        {corpus_section}

        == LOG SUMMARY ==
        {log_section}
        {question_section}

        Provide:
        1. Root cause analysis (what went wrong and why)
        2. Impact assessment (what was lost/degraded)
        3. Specific fix recommendations (code/config changes)

        Be direct and technical.  No hedging.
    """)

    return _call_llm(prompt)


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------

def format_report(report: DiagnosticReport) -> str:
    """Format the diagnostic report as readable text."""
    lines: list[str] = []

    # Header
    verdict_emoji = {
        Verdict.HEALTHY: "HEALTHY",
        Verdict.DEGRADED: "DEGRADED",
        Verdict.FAILED: "FAILED",
    }
    lines.append(f"=== PIPELINE DIAGNOSTIC: {verdict_emoji[report.verdict]} ===")
    lines.append(
        f"Critical: {report.critical_count}  "
        f"Warning: {report.warning_count}  "
        f"Elapsed: {report.elapsed_s:.1f}s"
    )
    lines.append("")

    # Check results
    lines.append("--- HEALTH CHECKS ---")
    for check in report.checks:
        prefix = {
            Severity.CRITICAL: "[CRITICAL]",
            Severity.WARNING: "[WARNING] ",
            Severity.OK: "[OK]      ",
        }[check.severity]
        lines.append(f"{prefix} {check.name}: {check.message}")
        for ev in check.evidence[:3]:
            lines.append(f"           {ev}")
    lines.append("")

    # Corpus summary
    lines.append("--- CORPUS SUMMARY ---")
    cs = report.corpus_summary
    lines.append(
        f"Findings: {cs.get('total_findings', 0)}  "
        f"Thoughts: {cs.get('total_thoughts', 0)}  "
        f"Serendipity: {cs.get('total_serendipity', 0)}"
    )
    lines.append(
        f"Unscored: {cs.get('unscored_count', 0)} "
        f"({cs.get('unscored_pct', 0):.0f}%)  "
        f"Avg quality: {cs.get('avg_composite_quality', 0):.3f}"
    )
    qd = cs.get("quality_distribution", {})
    lines.append(
        f"Quality: strong={qd.get('strong', 0)}  "
        f"moderate={qd.get('moderate', 0)}  "
        f"weak={qd.get('weak', 0)}"
    )
    lines.append(
        f"Contradictions: {cs.get('contradiction_count', 0)}  "
        f"Iterations: {cs.get('iterations_seen', 0)}"
    )
    lines.append("")

    # LLM analysis
    if report.llm_analysis:
        lines.append("--- ROOT CAUSE ANALYSIS ---")
        lines.append(report.llm_analysis)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_diagnostic(
    log_path: str = "",
    db_path: str = "",
    question: str = "",
    skip_llm: bool = False,
) -> DiagnosticReport:
    """Run the full diagnostic pipeline.

    1. Parse log file (deterministic)
    2. Analyse corpus (deterministic SQL)
    3. Run health checks (deterministic)
    4. Compute verdict (deterministic)
    5. LLM analysis (optional — only if issues found)

    Returns a structured DiagnosticReport.
    """
    start = time.monotonic()

    # Step 1 & 2: Parse inputs
    log_summary = parse_log_file(log_path)
    corpus_summary = analyse_corpus(db_path)

    # Step 3: Deterministic health checks
    checks = run_deterministic_checks(log_summary, corpus_summary)

    # Step 4: Compute verdict
    verdict = compute_verdict(checks)

    # Step 5: LLM analysis (only if issues found and not skipped)
    llm_analysis = ""
    if not skip_llm and verdict != Verdict.HEALTHY:
        llm_analysis = generate_llm_analysis(
            checks, corpus_summary, log_summary, question,
        )

    elapsed = time.monotonic() - start

    return DiagnosticReport(
        verdict=verdict,
        checks=checks,
        corpus_summary=corpus_summary,
        log_summary=log_summary,
        llm_analysis=llm_analysis,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for standalone diagnostic runs."""
    parser = argparse.ArgumentParser(
        description="MiroThinker Pipeline Diagnostic Tool",
    )
    parser.add_argument(
        "--log", default="",
        help="Path to pipeline log file",
    )
    parser.add_argument(
        "--db", default="",
        help="Path to DuckDB corpus file",
    )
    parser.add_argument(
        "--question", default="",
        help="Optional question about the pipeline run",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM analysis (deterministic checks only)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON instead of text",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    report = run_diagnostic(
        log_path=args.log,
        db_path=args.db,
        question=args.question,
        skip_llm=args.skip_llm,
    )

    if args.json_output:
        output = {
            "verdict": report.verdict.value,
            "critical_count": report.critical_count,
            "warning_count": report.warning_count,
            "checks": [
                {
                    "name": c.name,
                    "severity": c.severity.value,
                    "message": c.message,
                    "evidence": c.evidence,
                }
                for c in report.checks
            ],
            "corpus_summary": report.corpus_summary,
            "llm_analysis": report.llm_analysis,
            "elapsed_s": report.elapsed_s,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(report))

    # Exit with non-zero code if pipeline failed
    if report.verdict == Verdict.FAILED:
        sys.exit(2)
    elif report.verdict == Verdict.DEGRADED:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
